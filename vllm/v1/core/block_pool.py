# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
from collections.abc import Iterable, Sequence  # 导入可迭代和序列抽象基类
from typing import Any  # 导入通用类型标注

from vllm.distributed.kv_events import (  # 从分布式 KV 事件模块导入
    MEDIUM_GPU,  # GPU 介质常量
    AllBlocksCleared,  # 所有 block 清除事件
    BlockRemoved,  # block 移除事件
    BlockStored,  # block 存储事件
    KVCacheEvent,  # KV 缓存事件基类
)
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector  # 导入 KV 缓存指标收集器
from vllm.v1.core.kv_cache_utils import (  # 从 KV 缓存工具模块导入
    BlockHash,  # block 哈希类型
    BlockHashList,  # block 哈希列表类型
    BlockHashListWithBlockSize,  # 带 block 大小的哈希列表
    BlockHashWithGroupId,  # 带 group ID 的 block 哈希
    ExternalBlockHash,  # 外部 block 哈希类型
    FreeKVCacheBlockQueue,  # 空闲 KV 缓存 block 队列
    KVCacheBlock,  # KV 缓存 block 类
    generate_block_hash_extra_keys,  # 生成 block 哈希额外键函数
    get_block_hash,  # 获取 block 哈希函数
    make_block_hash_with_group_id,  # 生成带 group ID 的 block 哈希
    maybe_convert_block_hash,  # 可能转换 block 哈希格式
)
from vllm.v1.request import Request  # 导入请求类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# [中文注释] 前缀缓存的 hash→block 映射表。
#   将 BlockHashWithGroupId 映射到 KVCacheBlock（单个）或 dict[block_id, KVCacheBlock]（多个）。
#   使用 union 类型（单 block 直接存储，多 block 用 dict）以减少 GC 开销。
#   核心操作：
#     get_one_block(key) — 查找任意一个匹配 hash 的 block
#     insert(key, block) — 插入 block（自动从单值升级为 dict）
#     pop(key, block_id) — 移除指定 block_id 的 block
class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        """初始化映射表，创建空的缓存字典"""
        self._cache: dict[  # 内部缓存字典，key 为带 group ID 的 block hash
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]  # value 为单个 block 或 block 字典
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """根据 block hash key 获取任意一个匹配的缓存 block"""
        blocks = self._cache.get(key)  # 从缓存中查找 key
        if blocks is not None:  # 如果找到了
            if isinstance(blocks, KVCacheBlock):  # 如果是单个 block
                return blocks  # 直接返回
            if isinstance(blocks, dict):  # 如果是多个 block 的字典
                return next(iter(blocks.values()))  # 返回第一个 block
            self._unexpected_blocks_type(blocks)  # 未知类型，抛出异常
        return None  # 未找到返回 None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """将 KVCacheBlock 插入缓存，自动处理单值到字典的升级"""
        blocks = self._cache.get(key)  # 查找当前 key 的值
        if blocks is None:  # key 不存在
            # When key is not found, attach a single block to the key
            self._cache[key] = block  # 直接存储单个 block
        elif isinstance(blocks, KVCacheBlock):  # 已有一个 block
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}  # 升级为字典存储
        elif isinstance(blocks, dict):  # 已经是字典
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block  # 直接添加到字典
        else:
            self._unexpected_blocks_type(blocks)  # 未知类型，抛出异常

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """从缓存中弹出指定 block_id 的 block，不存在则返回 None"""
        blocks = self._cache.pop(key, None)  # 弹出整个 key 的值
        if blocks is None:  # key 不存在
            # block_hash not found in the cache
            return None  # 返回 None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):  # 单个 block
            if blocks.block_id == block_id:  # block_id 匹配
                return blocks  # 返回该 block
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks  # 不匹配，放回原处
            return None  # 返回 None
        if isinstance(blocks, dict):  # 多个 block 的字典
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)  # 从字典中弹出指定 block
            if len(blocks) > 0:  # 字典中还有其他 block
                self._cache[key] = blocks  # 放回缓存
            return block  # 返回弹出的 block
        self._unexpected_blocks_type(blocks)  # 未知类型，抛出异常
        return None  # 兜底返回 None

    def __len__(self) -> int:
        """返回缓存中的 key 数量"""
        return len(self._cache)  # 返回缓存字典长度

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        """遇到未预期的 block 类型时抛出断言错误"""
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")  # 抛出异常


# [中文注释] KV Cache Block 池管理器。
#   管理所有 GPU 上的 KVCacheBlock，提供分配、释放和前缀缓存功能。
#   核心数据结构：
#     blocks: list[KVCacheBlock] — 所有 block 的数组（按 block_id 索引）
#     free_block_queue: FreeKVCacheBlockQueue — 空闲 block 的双向链表（LRU 驱逐顺序）
#     cached_block_hash_to_block: BlockHashToBlockMap — hash→block 的前缀缓存映射
#     null_block — block_id=0 的占位 block，用于 sliding window 等需要跳过的位置
#   分配流程：get_new_blocks() 从队列头部弹出 block，若 block 有缓存 hash 则先驱逐
#   释放流程：free_blocks() 将 block 按驱逐优先级（反向）追加到队列尾部
#   缓存流程：cache_full_blocks() 为满 block 计算 hash 并存入映射表
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,  # GPU block 总数
        enable_caching: bool,  # 是否启用前缀缓存
        hash_block_size: int,  # 用于计算 hash 的 block 大小
        enable_kv_cache_events: bool = False,  # 是否启用 KV 缓存事件
        metrics_collector: KVCacheMetricsCollector | None = None,  # 可选的指标收集器
    ):
        """初始化 BlockPool，创建所有 block 并构建空闲队列"""
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0  # 确保 block 数为正整数
        self.num_gpu_blocks = num_gpu_blocks  # 保存 GPU block 总数
        self.enable_caching = enable_caching  # 保存缓存启用标志
        self.hash_block_size = hash_block_size  # 保存 hash block 大小
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [  # 创建所有 KVCacheBlock 对象列表
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)  # 按索引创建 block
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)  # 用所有 block 初始化空闲队列

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()  # 初始化前缀缓存映射表

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()  # 从队列弹出第一个 block 作为 null block
        self.null_block.is_null = True  # 标记为 null block

        self.enable_kv_cache_events = enable_kv_cache_events  # 保存事件启用标志
        self.kv_event_queue: list[KVCacheEvent] = []  # 初始化事件队列

        self.metrics_collector = metrics_collector  # 保存指标收集器

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]  # block 哈希和 group ID 列表
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []  # 初始化结果列表
        for group_id in kv_cache_group_ids:  # 遍历每个 group ID
            block_hash_with_group_id = make_block_hash_with_group_id(  # 组合 hash 和 group ID
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(  # 从缓存映射中查找
                block_hash_with_group_id
            )
            if not block:  # 如果任一 group 未命中
                return None  # 返回 None
            cached_blocks.append(block)  # 将命中的 block 加入结果
        return cached_blocks  # 返回所有 group 的命中 block

    def cache_full_blocks(
        self,
        request: Request,  # 请求对象
        blocks: list[KVCacheBlock],  # 请求的所有 block
        num_cached_blocks: int,  # 已缓存的 block 数量
        num_full_blocks: int,  # 需要缓存的满 block 数量
        block_size: int,  # 每个 block 的 token 数
        kv_cache_group_id: int,  # KV 缓存 group ID
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:  # 如果已缓存数 >= 需缓存数
            return  # 无需处理，直接返回
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]  # 取出新的满 block
        assert len(request.block_hashes) >= num_full_blocks  # 确保请求的 hash 数量足够
        if block_size == self.hash_block_size:  # 常见情况：block 大小与 hash block 大小一致
            # Common case.
            block_hashes: BlockHashList = request.block_hashes  # 直接使用请求的 hash 列表
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0  # 确保 block 大小是 hash block 大小的整数倍
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(  # 创建粒度转换后的 hash 列表
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]  # 取出新 block 对应的 hash
        new_hashes: list[ExternalBlockHash] | None = (  # 如果启用事件则创建外部 hash 列表
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):  # 遍历新的满 block
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:  # 跳过 null block
                continue
            assert blk.block_hash is None  # 确保 block 尚未被缓存
            block_hash = new_block_hashes[i]  # 获取对应的 hash

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(  # 组合 hash 和 group ID
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id  # 设置 block 的 hash
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)  # 插入缓存映射
            if new_hashes is not None:  # 如果需要记录事件 hash
                new_hashes.append(maybe_convert_block_hash(block_hash))  # 添加到事件 hash 列表

        if self.enable_kv_cache_events:  # 如果启用了 KV 缓存事件
            if num_cached_blocks == 0:  # 如果是第一批缓存
                parent_block_hash: ExternalBlockHash | None = None  # 没有父 block hash
            else:
                parent_block_hash = maybe_convert_block_hash(  # 获取父 block 的 hash
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size  # 起始 token 索引
            end_token_idx = num_full_blocks * block_size  # 结束 token 索引

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []  # 额外键列表
            curr_mm_idx = 0  # 当前多模态索引
            for i in range(num_cached_blocks, num_full_blocks):  # 遍历新缓存的 block 范围
                if blocks[i].is_null:  # 跳过 null block
                    continue
                block_start = i * block_size  # 当前 block 的起始 token
                block_end = block_start + block_size  # 当前 block 的结束 token
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(  # 生成额外键
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)  # 添加到列表

            self.kv_event_queue.append(  # 将存储事件加入队列
                BlockStored(
                    block_hashes=new_hashes,  # 新存储的 hash 列表
                    parent_block_hash=parent_block_hash,  # 父 block hash
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],  # token ID 范围
                    block_size=block_size,  # block 大小
                    lora_id=request.lora_request.adapter_id  # LoRA 适配器 ID
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,  # 存储介质为 GPU
                    lora_name=request.lora_request.name  # LoRA 名称
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,  # 额外键列表
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():  # 检查空闲 block 是否足够
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")  # 不足则抛出异常

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)  # 从队列头部弹出 n 个 block

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:  # 如果启用了缓存
            for block in ret:  # 遍历分配的 block
                self._maybe_evict_cached_block(block)  # 如果 block 有缓存则驱逐
                assert block.ref_cnt == 0  # 确保引用计数为 0
                block.ref_cnt += 1  # 增加引用计数
                if self.metrics_collector:  # 如果有指标收集器
                    self.metrics_collector.on_block_allocated(block)  # 记录分配事件
        else:  # 未启用缓存
            for block in ret:  # 遍历分配的 block
                assert block.ref_cnt == 0  # 确保引用计数为 0
                block.ref_cnt += 1  # 增加引用计数
                if self.metrics_collector:  # 如果有指标收集器
                    self.metrics_collector.on_block_allocated(block)  # 记录分配事件
        return ret  # 返回分配的 block 列表

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:  # 如果有指标收集器
            self.metrics_collector.on_block_evicted(block)  # 记录驱逐事件

        block_hash = block.block_hash  # 获取 block 的 hash
        if block_hash is None:  # block 没有 hash
            # The block doesn't have hash, eviction is not needed
            return False  # 无需驱逐

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:  # 从缓存中弹出
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False  # block 不在缓存中，无需驱逐

        block.reset_hash()  # 重置 block 的 hash

        if self.enable_kv_cache_events:  # 如果启用了事件
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(  # 将移除事件加入队列
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],  # 转换并记录 hash
                    medium=MEDIUM_GPU,  # 存储介质为 GPU
                )
            )
        return True  # 驱逐成功

    # [中文注释] touch() — 当另一个请求命中相同前缀时，增加 block 的引用计数。
    #   若 block 的 ref_cnt 为 0（即在空闲队列中作为驱逐候选），则将其从队列中移除。
    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:  # 遍历要触摸的 block
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:  # ref_cnt 为 0 且不是 null block
                self.free_block_queue.remove(block)  # 从空闲队列中移除
            block.ref_cnt += 1  # 增加引用计数
            if self.metrics_collector:  # 如果有指标收集器
                self.metrics_collector.on_block_accessed(block)  # 记录访问事件

    # [中文注释] free_blocks() — 释放一组 block。block 按驱逐优先级排序传入（头部先被驱逐）。
    #   先递减 ref_cnt，然后将 ref_cnt 降为 0 且非 null 的 block 追加到空闲队列。
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)  # 将可迭代对象转为列表（允许多次遍历）
        for block in blocks_list:  # 遍历 block 递减引用计数
            block.ref_cnt -= 1  # 减少引用计数
        self.free_block_queue.append_n(  # 将 ref_cnt 为 0 的非 null block 追加到空闲队列
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    # [中文注释] evict_blocks() — 按 block ID 强制驱逐前缀缓存中的 block。
    #   用于 KV connector 通知 scheduler 某些 block 已被外部修改，需要使缓存失效。
    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:  # 遍历要驱逐的 block ID
            assert block_id < len(self.blocks), (  # 确保 block ID 有效
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]  # 根据 ID 获取 block
            self._maybe_evict_cached_block(block)  # 驱逐该 block

    # [中文注释] reset_prefix_cache() — 重置前缀缓存（清空 hash 映射和所有 block 的 hash）。
    #   用于 RLHF 权重更新后使缓存失效，或基准测试中重置缓存状态。
    #   仅在所有 block 都已释放时才能成功（除 null_block 外）。
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()  # 计算使用中的 block 数
        if num_used_blocks != 1:  # The null block is always marked as used  # 除 null block 外应无使用中的 block
            logger.warning(  # 记录警告日志
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,  # 减去 null block
            )
            return False  # 返回失败

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()  # 重新创建空的缓存映射

        # Remove all hashes from all blocks.
        for block in self.blocks:  # 遍历所有 block
            block.reset_hash()  # 重置每个 block 的 hash

        if self.metrics_collector:  # 如果有指标收集器
            self.metrics_collector.reset()  # 重置指标

        logger.info("Successfully reset prefix cache")  # 记录成功日志

        if self.enable_kv_cache_events:  # 如果启用了事件
            self.kv_event_queue.append(AllBlocksCleared())  # 添加全部清除事件

        return True  # 返回成功

    def get_num_free_blocks(self) -> int:
        """获取空闲 block 数量"""
        return self.free_block_queue.num_free_blocks  # 返回空闲队列中的 block 数

    def get_usage(self) -> float:
        """获取 KV 缓存使用率（0.0 到 1.0 之间）"""

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1  # 总 block 数减去 null block
        if not total_gpu_blocks:  # 如果没有可用 block
            return 0  # 返回 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)  # 计算使用率

    def take_events(self) -> list[KVCacheEvent]:
        """原子地取走所有事件并清空队列"""
        if not self.enable_kv_cache_events:  # 如果未启用事件
            return []  # 返回空列表
        events = self.kv_event_queue  # 保存当前事件列表
        self.kv_event_queue = []  # 创建新的空列表
        return events  # 返回旧的事件列表
