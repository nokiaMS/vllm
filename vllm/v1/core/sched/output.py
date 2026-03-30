# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass  # 导入数据类装饰器
from functools import cached_property  # 导入缓存属性装饰器
from typing import TYPE_CHECKING  # 导入类型检查标志

if TYPE_CHECKING:  # 仅在类型检查时导入以下模块（避免循环导入和运行时开销）
    import numpy as np  # NumPy 数组库
    import numpy.typing as npt  # NumPy 类型注解
    import torch  # PyTorch 张量库

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata  # EC连接器元数据
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata  # KV连接器元数据
    from vllm.lora.request import LoRARequest  # LoRA请求
    from vllm.multimodal.inputs import MultiModalFeatureSpec  # 多模态特征规格
    from vllm.pooling_params import PoolingParams  # 池化参数
    from vllm.sampling_params import SamplingParams  # 采样参数
    from vllm.v1.request import Request  # 请求对象
else:
    ECConnectorMetadata = object  # 运行时占位类型
    KVConnectorMetadata = object  # 运行时占位类型
    LoRARequest = object  # 运行时占位类型
    MultiModalFeatureSpec = object  # 运行时占位类型
    PoolingParams = object  # 运行时占位类型
    SamplingParams = object  # 运行时占位类型
    Request = object  # 运行时占位类型


# [中文注释] NewRequestData — 首次调度的新请求数据（发送给 worker 缓存）。
#   包含 prompt_token_ids、多模态特征、采样参数、block_ids 等完整信息。
#   后续步骤只发送增量 diff（通过 CachedRequestData），减少通信开销。
@dataclass
class NewRequestData:
    """新请求数据，首次调度时发送给 worker 进行缓存。"""

    req_id: str  # 请求唯一标识
    prompt_token_ids: list[int] | None  # 提示词的 token ID 列表
    mm_features: list[MultiModalFeatureSpec]  # 多模态特征列表
    sampling_params: SamplingParams | None  # 采样参数
    pooling_params: PoolingParams | None  # 池化参数（用于嵌入模型）
    block_ids: tuple[list[int], ...]  # 分配的 KV 缓存块 ID（按缓存组分组）
    num_computed_tokens: int  # 已计算的 token 数量（前缀缓存命中数）
    lora_request: LoRARequest | None  # LoRA 适配器请求
    prompt_embeds: "torch.Tensor | None" = None  # 预计算的提示词嵌入

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None  # 预填充 token ID（仅 v2 模型运行器使用）

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        """从 Request 对象构造 NewRequestData 实例。"""
        return cls(
            req_id=request.request_id,  # 请求ID
            prompt_token_ids=request.prompt_token_ids,  # 提示词token
            mm_features=request.mm_features,  # 多模态特征
            sampling_params=request.sampling_params,  # 采样参数
            pooling_params=request.pooling_params,  # 池化参数
            block_ids=block_ids,  # KV缓存块ID
            num_computed_tokens=request.num_computed_tokens,  # 已计算token数
            lora_request=request.lora_request,  # LoRA请求
            prompt_embeds=request.prompt_embeds,  # 提示词嵌入
            prefill_token_ids=prefill_token_ids,  # 预填充token ID
        )

    def __repr__(self) -> str:
        """返回完整的字符串表示（包含prompt数据）。"""
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None  # 获取嵌入张量的形状
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        """返回脱敏的字符串表示（隐藏prompt具体内容，只显示长度）。"""
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None  # 提示词token数量
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None  # 嵌入张量形状
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None  # 预填充token数量
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


# [中文注释] CachedRequestData — 已缓存请求的增量更新数据。
#   仅包含与上一步的差异：new_token_ids（PP 场景）、new_block_ids、num_computed_tokens 等。
#   resumed_req_ids 标记被抢占后恢复的请求（其 block_ids 是完整替换而非追加）。
#   all_token_ids 仅包含上一步未被调度的请求的完整 token 列表（用于 connector 同步）。
@dataclass
class CachedRequestData:
    """已缓存请求的增量更新数据，仅发送与上一步的差异。"""

    req_ids: list[str]  # 请求ID列表
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]  # 被抢占后恢复的请求ID集合（其block_ids为完整替换而非追加）
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]  # 新增的token ID（仅流水线并行时使用）
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]]  # 上一步未调度请求的完整token列表（用于connector同步）
    new_block_ids: list[tuple[list[int], ...] | None]  # 新分配的KV缓存块ID
    num_computed_tokens: list[int]  # 每个请求已计算的token数
    num_output_tokens: list[int]  # 每个请求已输出的token数

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        """返回脱敏的字符串表示（隐藏token具体内容，只显示长度）。"""
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]  # 各请求新token数量
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()  # 各请求全部token数量
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        """默认使用脱敏表示。"""
        return self.anon_repr()  # 默认使用脱敏版本的repr

    @property
    def num_reqs(self) -> int:
        """返回请求数量。"""
        return len(self.req_ids)  # 请求ID列表的长度即请求数

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))  # 构建请求ID到输出token数的映射

    def is_context_phase(self, req_id: str) -> bool:
        """判断请求是否处于上下文（预填充）阶段（输出token数为0）。"""
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)  # 查找输出token数
        return num_output_tokens is not None and num_output_tokens == 0  # 输出为0表示仍在预填充

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """创建空的 CachedRequestData 实例。"""
        return cls(
            req_ids=[],  # 空请求ID列表
            resumed_req_ids=set(),  # 空恢复请求集合
            new_token_ids=[],  # 空新token列表
            all_token_ids={},  # 空全token映射
            new_block_ids=[],  # 空新块ID列表
            num_computed_tokens=[],  # 空已计算token列表
            num_output_tokens=[],  # 空输出token列表
        )


# [中文注释] SchedulerOutput — 单步调度的完整输出，传递给 model runner。
#   核心字段：
#     scheduled_new_reqs — 新请求的完整数据
#     scheduled_cached_reqs — 已缓存请求的增量数据
#     num_scheduled_tokens — 每个请求本步调度的 token 数
#     scheduled_spec_decode_tokens — 投机解码的 draft token
#     scheduled_encoder_inputs — 需要编码器处理的多模态输入索引
#     num_common_prefix_blocks — 公共前缀 block 数（用于 cascade attention）
#     finished_req_ids — 上一步到本步之间结束的请求 ID（通知 worker 释放缓存）
#     kv_connector_metadata / ec_connector_metadata — KV/EC 传输元数据
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]  # 首次调度的新请求数据列表
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData  # 已缓存请求的增量更新数据

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]  # 每个请求本步调度的token数
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int  # 所有请求本步调度的token总数
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]  # 投机解码的draft token映射
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]  # 需要编码器处理的多模态输入索引
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]  # 公共前缀块数（用于级联注意力）

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]  # 上一步到当前步之间结束的请求ID集合
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]  # 需要从编码器缓存中释放的多模态哈希列表

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None  # 本步被抢占的请求ID集合（仅v2模型运行器）

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False  # 是否有使用结构化输出的请求

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False  # 是否有待处理的结构化输出token

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None  # 无效的投机token数（用于调整接受率）

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None  # KV缓存传输连接器元数据

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None  # EC缓存连接器元数据

    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    new_block_ids_to_zero: list[int] | None = None  # 新分配需要清零的块ID列表

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        """创建空的 SchedulerOutput 实例。"""
        return cls(
            scheduled_new_reqs=[],  # 无新请求
            scheduled_cached_reqs=CachedRequestData.make_empty(),  # 空的缓存请求数据
            num_scheduled_tokens={},  # 无调度token
            total_num_scheduled_tokens=0,  # 总调度token数为0
            scheduled_spec_decode_tokens={},  # 无投机解码token
            scheduled_encoder_inputs={},  # 无编码器输入
            num_common_prefix_blocks=[],  # 无公共前缀块
            finished_req_ids=set(),  # 无已完成请求
            free_encoder_mm_hashes=[],  # 无需释放的编码器缓存
        )


# [中文注释] GrammarOutput — 结构化输出的语法约束 bitmask。
#   structured_output_request_ids: 使用结构化输出的请求 ID 列表
#   grammar_bitmask: 对应的 int32 bitmask 数组（行顺序与 request_ids 一致）
@dataclass
class GrammarOutput:
    """结构化输出的语法约束 bitmask 数据。"""

    # ids of structured output requests.
    structured_output_request_ids: list[str]  # 使用结构化输出的请求ID列表
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"  # 语法约束bitmask数组（行顺序与request_ids一致）
