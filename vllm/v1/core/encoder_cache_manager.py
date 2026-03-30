# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from collections import OrderedDict  # 导入有序字典（用于 FIFO 驱逐顺序）
from collections.abc import Mapping  # 导入映射抽象基类
from typing import TYPE_CHECKING  # 导入类型检查标志

from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.v1.request import Request  # 导入请求类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from vllm.config import SchedulerConfig  # 导入调度器配置

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# [中文注释] 多模态编码器输出缓存管理器（用于 vision-language 等多模态模型）。
#   按 mm_hash 缓存编码器输出，支持跨请求共享相同多模态数据的 embedding。
#   内存管理采用引用计数 + LRU 驱逐策略：
#     cached: dict[mm_hash → set[request_id]] — 每个缓存项被哪些请求引用
#     freeable: OrderedDict[mm_hash → num_embeds] — 引用为 0 可回收的项（FIFO 驱逐）
#     freed: list[mm_hash] — 已驱逐的项（通知 worker 释放实际内存）
#   关键流程：
#     check_and_update_cache() — 检查缓存命中并更新引用
#     can_allocate() — 检查空间是否足够，不够则驱逐 freeable 中最老的项
#     allocate() — 分配缓存空间（逻辑上，不分配物理内存）
#     free() — 请求结束时释放引用，引用归零的项进入 freeable
class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data
    item (identified by their hash value) between different requests,
    and eviction takes place at allocation time when there's no free
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.

    NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
    instead of encoder tokens (i.e. all tokens that represent the multimodal data
    in the input sequence). This means all break/text tokens in-between multimodal
    embeddings are not considered with respect to the cache size and the number
    of free slots.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    encoder embeddings from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder embeddings.
        num_free_slots: Current available cache capacity in encoder embeddings.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder embeddings).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_encoder_embeds) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    def __init__(self, cache_size: int):
        """初始化编码器缓存管理器"""
        self.cache_size = cache_size  # 缓存总容量（以编码器 embedding 数计）
        self.num_free_slots = cache_size  # 当前可用容量
        self.num_freeable_slots = cache_size  # 当前可回收容量（包括 freeable 中的）

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}  # mm_hash → 引用该数据的请求 ID 集合

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()  # 引用为 0 可回收的项（FIFO 顺序）
        self.freed: list[str] = []  # 已驱逐的 mm_hash 列表

    def reset(self) -> None:
        """Reset the encoder cache to its initial state.

        This clears all cached encoder outputs and resets capacity tracking.
        Called when model weights are updated to invalidate stale embeddings.
        """
        self.cached.clear()  # 清空缓存映射
        self.freeable.clear()  # 清空可回收列表
        self.freed.clear()  # 清空已释放列表
        self.num_free_slots = self.cache_size  # 重置可用容量
        self.num_freeable_slots = self.cache_size  # 重置可回收容量

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier  # 获取多模态输入的哈希标识符
        # Not cached at all
        if mm_hash not in self.cached:  # 完全未缓存
            return False  # 返回未命中

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:  # 缓存存在但无引用
            num_encoder_embeds = self.freeable.pop(mm_hash)  # 从可回收列表移除
            self.num_freeable_slots -= num_encoder_embeds  # 减少可回收容量

        self.cached[mm_hash].add(request.request_id)  # 添加请求引用
        return True  # 返回命中

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True.
        Older entries are evicted first.

        Returns False only if the requested number of tokens exceeds both
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder embeddings allowed to be
                computed when this method is invoked.
            num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder
        output but only the state of EncoderCacheManager.
        """
        num_embeds = request.get_num_encoder_embeds(input_id)  # 获取该输入需要的 embedding 数

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:  # 超出计算预算
            return False  # 无法分配

        num_embeds += num_embeds_to_schedule  # 加上已调度待分配的 embedding 数

        # Enough free slots
        if num_embeds <= self.num_free_slots:  # 空闲容量足够
            return True  # 可以分配

        # Not enough reclaimable slots
        if num_embeds > self.num_freeable_slots:  # 可回收容量也不足
            return False  # 无法分配

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_embeds > self.num_free_slots:  # 逐个驱逐直到空间足够
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)  # 驱逐最老的项
            del self.cached[mm_hash]  # 从缓存中删除
            self.freed.append(mm_hash)  # 记录已驱逐的 hash
            self.num_free_slots += num_free_embeds  # 增加空闲容量
        return True  # 驱逐后空间足够

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier  # 获取多模态输入的哈希标识符
        request_id = request.request_id  # 获取请求 ID
        if mm_hash not in self.cached:  # 如果尚未在缓存中
            self.cached[mm_hash] = set()  # 创建空引用集合

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)  # 获取需要的 embedding 数

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_embeds  # 确保空闲容量足够
        assert self.num_freeable_slots >= num_encoder_embeds  # 确保可回收容量足够

        self.cached[mm_hash].add(request_id)  # 添加请求引用
        self.num_free_slots -= num_encoder_embeds  # 减少空闲容量
        self.num_freeable_slots -= num_encoder_embeds  # 减少可回收容量

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freeable`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """释放请求对指定编码器输入的引用，引用归零时加入可回收列表"""
        req_id = request.request_id  # 获取请求 ID
        mm_hash = request.mm_features[input_id].identifier  # 获取多模态哈希标识
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):  # mm_hash 不在缓存中或引用集为空
            return  # 直接返回
        self.cached[mm_hash].discard(req_id)  # 移除该请求的引用
        if not self.cached[mm_hash]:  # 如果引用集变为空
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)  # 获取 embedding 数
            self.freeable[mm_hash] = num_encoder_embeds  # 加入可回收列表
            self.num_freeable_slots += num_encoder_embeds  # 增加可回收容量

    def free(self, request: Request) -> None:
        """释放请求持有的所有编码器缓存引用"""
        input_ids = self.get_cached_input_ids(request)  # 获取该请求的所有已缓存输入 ID
        for input_id in input_ids:  # 遍历每个输入 ID
            self.free_encoder_input(request, input_id)  # 释放引用

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which
            encoder outputs can be removed from their caches. The internal
            list is cleared after this call.
        """
        freed = self.freed  # 保存当前已释放列表
        self.freed = []  # 重置为空列表
        return freed  # 返回已释放的 mm_hash 列表


# [中文注释] 计算多模态编码器的计算预算和缓存空间预算。
#   compute budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)
#   cache size = max(encoder_cache_size, max_tokens_per_mm_item)
#   确保至少能容纳一个最大的多模态输入项。
def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        mm_max_toks_per_item: The maximum number of tokens per item for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """

    if not mm_max_toks_per_item:  # 如果没有非文本模态
        logger.warning(  # 记录警告日志
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0  # 返回 0 预算

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())  # 获取最大的单项 token 数

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(  # 计算预算取配置值与最大项的较大者
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(  # 缓存空间取配置值与最大项的较大者
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size  # 返回计算预算和缓存大小


# NOTE (NickLucche): Temporary implementation for encoder-decoder models that only
# use the manager for scheduling purposes. Encoder-decoder models will eventually
# utilize the cache and this class will fold into EncoderCacheManager, as
# differences with MM models shrink.
# [中文注释] 编码器-解码器模型的简化缓存管理器（如 Whisper）。
#   不支持跨请求共享缓存（check_and_update_cache 总是返回 False）。
#   使用 allocated/to_free 双缓冲机制延迟释放：
#     allocated → 本步骤分配的 → 下一步变为 to_free → 再下一步实际释放
#   这样确保 model runner 在执行完当前步骤后才释放编码器输出。
class EncoderDecoderCacheManager(EncoderCacheManager):
    """编码器-解码器模型的简化缓存管理器（不支持跨请求共享）"""

    def __init__(self, cache_size: int):
        """初始化编码器-解码器缓存管理器"""
        self.cache_size = cache_size  # 缓存总容量
        self.num_free_slots = cache_size  # 当前空闲容量
        self.allocated: list[str] = []  # 本步骤分配的 mm_hash 列表
        self.to_free: list[str] = []  # 待释放的 mm_hash 列表（上一步分配的）

    def reset(self) -> None:
        """重置编码器缓存到初始状态"""
        self.num_free_slots = self.cache_size  # 重置空闲容量
        self.allocated.clear()  # 清空分配列表
        self.to_free.clear()  # 清空待释放列表

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """编码器-解码器模型不支持跨请求缓存共享，总是返回 False"""
        return False  # 总是返回未命中

    def can_allocate(
        self,
        request: Request,  # 请求对象
        input_id: int,  # 多模态输入索引
        encoder_compute_budget: int,  # 编码器计算预算
        num_embeds_to_schedule: int,  # 已调度的 embedding 数
    ) -> bool:
        """检查是否有足够的缓存空间和计算预算"""
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)  # 获取需要的 embedding 数
        # Not enough compute budget
        if num_encoder_embeds > encoder_compute_budget:  # 超出计算预算
            return False  # 无法分配

        num_encoder_embeds += num_embeds_to_schedule  # 加上已调度待分配的
        # Enough free slots
        return num_encoder_embeds <= self.num_free_slots  # 检查空闲容量是否足够

    def allocate(self, request: Request, input_id: int) -> None:
        """分配缓存空间并记录到 allocated 列表"""
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)  # 获取需要的 embedding 数
        self.num_free_slots -= num_encoder_embeds  # 减少空闲容量

        mm_hash = request.mm_features[input_id].identifier  # 获取多模态哈希标识
        self.allocated.append(mm_hash)  # 记录到本步骤分配列表

    def free(self, request: Request) -> None:
        """释放请求的所有编码器输入引用"""
        for input_id in range(len(request.mm_features)):  # 遍历所有多模态输入
            self.free_encoder_input(request, input_id)  # 释放每个输入的引用

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """返回请求的所有多模态输入 ID（编码器-解码器模型全部视为已缓存）"""
        return set(range(len(request.mm_features)))  # 返回全部输入 ID 的集合

    def get_freed_mm_hashes(self) -> list[str]:
        """获取已释放的 mm_hash 列表，使用双缓冲延迟一步释放"""
        # As encoder cache is not used for enc-dec models, we can free the entries here
        # The actual free happens in the runner, *before* the model is executed.
        # Therefore, `freeable` acts as a buffer to free the entries only after the
        # model is executed, mimicking the state transition of `EncoderCacheManager`.
        to_free = self.to_free  # 保存当前待释放列表
        self.to_free = self.allocated  # 本步骤分配的变为下步待释放
        self.allocated = []  # 重置本步骤分配列表
        return to_free  # 返回待释放列表

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """释放单个编码器输入的缓存空间"""
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)  # 获取 embedding 数
        self.num_free_slots += num_encoder_embeds  # 增加空闲容量
