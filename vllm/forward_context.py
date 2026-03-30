# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time  # 导入时间模块，用于性能计时
from collections import defaultdict  # 导入默认字典，用于批量大小统计
from contextlib import contextmanager  # 导入上下文管理器装饰器
from dataclasses import dataclass, field  # 导入数据类装饰器和字段工厂
from typing import Any  # 导入通用类型提示

import torch  # 导入PyTorch框架

import vllm.envs as envs  # 导入vLLM环境变量配置
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig  # 导入CUDA图模式、并行配置和vLLM配置类
from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.platforms import current_platform  # 导入当前平台信息
from vllm.v1.attention.backend import AttentionMetadata  # 导入注意力元数据类
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp  # 导入数据并行批次协调工具
from vllm.v1.worker.ubatch_utils import UBatchSlices  # 导入微批次切片工具

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0  # 是否追踪批量大小的标志
last_logging_time: float = 0  # 上次日志记录的时间戳
forward_start_time: float = 0  # 前向传播开始的时间戳
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL  # 批量大小日志记录的时间间隔
batchsize_forward_time: defaultdict = defaultdict(list)  # 按批量大小分组存储前向传播耗时的字典


@dataclass(frozen=True)  # 定义不可变数据类
class BatchDescriptor:
    """
    批次描述符，用于CUDA图调度。应保持尽可能少的字段数量，
    以正确且唯一地描述用于CUDA图的填充批次。
    """

    num_tokens: int  # 批次中的token数量
    num_reqs: int | None = None  # 批次中的请求数量，对于PIECEWISE模式可为None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False  # 批次中所有请求是否具有相同的token数量
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False  # 该批次是否有活跃的LoRA适配器
    """
    Whether this batch has active LoRA adapters.
    """
    num_active_loras: int = 0  # 活跃的不同LoRA适配器数量
    """
    Number of distinct active LoRA adapters in this batch.
    When cudagraph_specialize_lora_count is enabled, separate CUDA graphs
    are captured for each num_active_loras value. This allows kernels
    (like fused_moe_lora) whose grid size depends on num_active_loras
    to be properly captured.
    """


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    """计算序列并行情况下每个rank的token数量。

    Args:
        num_tokens_across_dp_cpu: 各数据并行rank上的token数量张量。
        sequence_parallel_size: 序列并行的大小。

    Returns:
        各序列并行rank上分配的token数量列表。
    """
    sp_tokens = (  # 计算每个DP rank在SP分片后的token数（向上取整）
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)  # 将每个DP rank的SP token数复制SP次
    return sp_tokens.tolist()  # 转换为Python列表返回


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    max_num_tokens: int,
    chunk_idx: int,
) -> list[int]:
    """计算分块执行时每个rank的本地token数量。

    Args:
        num_tokens_across_dp_cpu: 各数据并行rank上的token数量张量。
        sequence_parallel_size: 序列并行的大小。
        max_num_tokens: 每个rank每个分块的最大token数。
        chunk_idx: 当前分块的索引。

    Returns:
        各rank在当前分块中需要处理的token数量列表。
    """
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)  # 计算SP分片后的token数
    sp_size = len(sp_tokens)  # 获取总的SP rank数量

    local_size = [-1] * sp_size  # 初始化每个rank的本地大小列表
    for i in range(sp_size):  # 遍历每个SP rank
        # Take into account sharding if MoE activation is sequence parallel.
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))  # 计算当前分块中该rank的token数
        if local_size[i] <= 0:  # 如果该rank已无剩余token需处理
            local_size[i] = 1  # ensure lockstep even if done  # 设为1以确保各rank同步执行
    return local_size  # 返回每个rank的本地token数量


@dataclass  # 定义数据并行元数据的数据类
class DPMetadata:
    """数据并行（Data Parallel）元数据类，存储跨DP rank的token数量信息。"""

    max_tokens_across_dp_cpu: torch.Tensor  # 所有DP rank中最大的token数量
    num_tokens_across_dp_cpu: torch.Tensor  # 各DP rank的token数量张量

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None  # 分块执行时每个rank的本地token数，仅由上下文管理器设置

    @staticmethod  # 静态工厂方法
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        """创建DPMetadata实例的工厂方法。

        Args:
            parallel_config: 并行配置对象。
            num_tokens: 当前rank的token数量。
            num_tokens_across_dp_cpu: 各DP rank的token数量张量。

        Returns:
            构造好的DPMetadata实例。
        """
        assert num_tokens_across_dp_cpu is not None  # 断言token数量张量不为空
        assert parallel_config.data_parallel_size > 1  # 断言数据并行大小大于1
        assert parallel_config.is_moe_model is not False  # 断言模型为MoE模型
        dp_rank = parallel_config.data_parallel_rank  # 获取当前数据并行rank
        batchsize = num_tokens  # 当前rank的批量大小即为token数

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (  # 断言当前rank的token数一致
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)  # 计算所有DP rank中的最大token数
        return DPMetadata(max_tokens_across_dp_cpu, num_tokens_across_dp_cpu)  # 返回构造的DPMetadata实例

    @contextmanager  # 上下文管理器装饰器
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ):
        """
        用于计算并临时设置分块前向执行期间每个rank的本地token数量的上下文管理器。

        这确保每个DP（数据并行）rank在同步执行中处理其指定部分的token，
        即使token数量不均匀或某些rank已完成其输入也是如此。

        对于分块执行，我们将每个rank上的总token分成多个分块
        （每块最多 `max_chunk_size_per_rank`），对于给定的 `chunk_idx`，
        此上下文管理器将 `self.local_sizes` 设置为该分块中每个rank需要处理的token数。

        `self.local_sizes` 仅在上下文内有效。

        Args:
            sequence_parallel_size: 当注意力层使用TP而MoE层使用EP时，
                                    我们在层之间使用SP以避免冗余操作。
                                    需要此值来计算分块大小。
            max_chunk_size_per_rank: 每个rank在此分块中允许处理的最大token数。
            chunk_idx: 要计算大小的分块索引。
        """
        self.local_sizes = _compute_chunked_local_num_tokens(  # 计算分块后的本地token数量
            self.num_tokens_across_dp_cpu,  # 各DP rank的token数量
            sequence_parallel_size,  # 序列并行大小
            max_chunk_size_per_rank,  # 每个rank的最大分块大小
            chunk_idx,  # 分块索引
        )
        try:  # 进入上下文
            yield self.local_sizes  # 返回本地大小供调用方使用
        finally:  # 退出上下文时清理
            self.local_sizes = None  # 重置本地大小为None

    @contextmanager  # 上下文管理器装饰器
    def sp_local_sizes(self, sequence_parallel_size: int):
        """
        设置 self.local_sizes 的上下文管理器。与 self.chunked_sizes 类似，
        但不进行分块处理。

        Args:
            sequence_parallel_size: 序列并行的大小。
        """
        self.local_sizes = _compute_sp_num_tokens(  # 计算SP分片后的token数量
            self.num_tokens_across_dp_cpu, sequence_parallel_size  # 使用各DP rank的token数和SP大小
        )
        try:  # 进入上下文
            yield self.local_sizes  # 返回本地大小供调用方使用
        finally:  # 退出上下文时清理
            self.local_sizes = None  # 重置本地大小为None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        """获取各DP rank的分块大小列表。

        Returns:
            各rank的本地token数量列表。
        """
        assert self.local_sizes is not None  # 断言本地大小已被设置
        return self.local_sizes  # 返回本地大小列表

    # Get the cumulative tokens across sequence parallel ranks.
    # In this case the input to the MoEs will be distributed w.r.t both
    # DP and TP rank.
    # When sp_size==1, this is just the cumulative num tokens across DP.
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        """计算跨序列并行rank的累积token数量。

        当sp_size==1时，等同于跨DP的累积token数。
        MoE的输入会根据DP和TP rank进行分布。

        Args:
            sp_size: 序列并行的大小。

        Returns:
            累积token数量的张量。
        """
        num_tokens_across_sp_cpu = (  # 计算每个DP rank在SP分片后的token数（向上取整）
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)  # 将每个DP rank的值复制SP次
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)  # 返回累积和张量


@dataclass  # 定义前向上下文的数据类
class ForwardContext:
    """
    前向传播上下文类，存储每次前向传播所需的元数据和配置信息。
    包括注意力元数据、CUDA图模式、数据并行信息、MoE层信息等。
    """

    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]  # 不参与编译的层字典，从vllm_config.compilation_config.static_forward_context复制
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]  # 注意力元数据，v1为字典，DBO为列表
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]  # KV缓存槽位映射
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass  # 虚拟引擎索引，每次前向传播动态设置
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None  # 数据并行元数据，每次前向传播动态设置
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE  # 运行时CUDA图模式，默认不使用
    batch_descriptor: BatchDescriptor | None = None  # 批次描述符，用于CUDA图调度

    ubatch_slices: UBatchSlices | None = None  # 微批次切片信息

    # If True, bypass the compiled model call, e.g. by using .forward() directly
    skip_compiled: bool = False  # 是否跳过编译模型调用，直接使用.forward()

    # For torch.compile cold start times, we need to avoid hard-coding
    # any strings into the graph. Right now, the vllm.moe_forward
    # and vllm.moe_forward_shared custom operators hard-code strings into
    # the graph.
    #
    # The workaround is to store a list of the strings that each of those
    # custom ops needs in the ForwardContext (all_moe_layers)
    # as well as a counter (moe_layer_index).
    # The ForwardContext object is alive for the duration of the forward pass.
    # When the custom op needs a layer string, get the next string
    # from all_moe_layers and increment the counter.
    #
    # This assumes that the custom operators will always be executed in
    # order and that torch.compile will not try to reorder these
    # operations with respect to each other.
    #
    # TODO(https://github.com/vllm-project/vllm/issues/31985):
    # There are longer-term solutions, like unwrapping the moe custom operator,
    # that aren't ready yet.
    # We could also treat the string as a "symbolic input" to the graph but
    # the PyTorch-side bits for that aren't ready yet either.
    #
    # If this value is None (like in some tests), then we end up baking the string
    # into the graph. Otherwise, the moe custom ops will pop a string from this list.
    all_moe_layers: list[str] | None = None  # 所有MoE层名称列表，用于避免在torch.compile图中硬编码字符串
    moe_layer_index: int = 0  # 当前MoE层索引计数器，随前向传播递增

    additional_kwargs: dict[str, Any] = field(default_factory=dict)  # 额外的关键字参数字典

    def __post_init__(self):
        """初始化后验证，确保CUDA图运行时模式有效。"""
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (  # 断言CUDA图运行时模式合法
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


_forward_context: ForwardContext | None = None  # 全局前向上下文变量，存储当前前向传播的上下文


def get_forward_context() -> ForwardContext:
    """获取当前的前向传播上下文。

    Returns:
        当前的ForwardContext实例。

    Raises:
        AssertionError: 如果前向上下文未设置则抛出异常。
    """
    assert _forward_context is not None, (  # 断言前向上下文已设置
        "Forward context is not set. "  # 错误消息：前向上下文未设置
        "Please use `set_forward_context` to set the forward context."  # 提示使用set_forward_context设置
    )
    return _forward_context  # 返回当前前向上下文


def is_forward_context_available() -> bool:
    """检查前向传播上下文是否可用。

    Returns:
        如果前向上下文已设置则返回True，否则返回False。
    """
    return _forward_context is not None  # 判断全局前向上下文是否不为None


def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
):
    """创建前向传播上下文对象。

    Args:
        attn_metadata: 注意力元数据。
        vllm_config: vLLM配置对象。
        virtual_engine: 虚拟引擎索引，默认为0。
        dp_metadata: 数据并行元数据，可选。
        cudagraph_runtime_mode: CUDA图运行时模式，默认不使用。
        batch_descriptor: 批次描述符，可选。
        ubatch_slices: 微批次切片，可选。
        slot_mapping: KV缓存槽位映射，可选。
        additional_kwargs: 额外关键字参数，可选。
        skip_compiled: 是否跳过编译模型调用。

    Returns:
        构造好的ForwardContext实例。
    """
    if vllm_config.compilation_config.fast_moe_cold_start:  # 如果启用了MoE快速冷启动
        all_moe_layers = vllm_config.compilation_config.static_all_moe_layers  # 从编译配置中获取所有MoE层名称
    else:  # 否则
        all_moe_layers = None  # 不使用MoE层名称列表

    return ForwardContext(  # 构造并返回ForwardContext实例
        no_compile_layers=vllm_config.compilation_config.static_forward_context,  # 设置不参与编译的层
        all_moe_layers=all_moe_layers,  # 设置MoE层名称列表
        virtual_engine=virtual_engine,  # 设置虚拟引擎索引
        attn_metadata=attn_metadata,  # 设置注意力元数据
        slot_mapping=slot_mapping or {},  # 设置槽位映射，默认为空字典
        dp_metadata=dp_metadata,  # 设置数据并行元数据
        cudagraph_runtime_mode=cudagraph_runtime_mode,  # 设置CUDA图运行时模式
        batch_descriptor=batch_descriptor,  # 设置批次描述符
        ubatch_slices=ubatch_slices,  # 设置微批次切片
        skip_compiled=skip_compiled,  # 设置是否跳过编译
        additional_kwargs=additional_kwargs or {},  # 设置额外参数，默认为空字典
    )


@contextmanager  # 上下文管理器装饰器
def override_forward_context(forward_context: ForwardContext | None):
    """覆盖当前前向传播上下文的上下文管理器。

    用于在特定的前向传播过程中临时替换前向上下文。

    Args:
        forward_context: 要设置的新前向上下文，可为None。
    """
    global _forward_context  # 声明使用全局前向上下文变量
    prev_context = _forward_context  # 保存当前的前向上下文
    _forward_context = forward_context  # 设置新的前向上下文
    try:  # 进入上下文
        yield  # 让出控制权给调用方
    finally:  # 退出上下文时恢复
        _forward_context = prev_context  # 恢复之前的前向上下文


@contextmanager  # 上下文管理器装饰器
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
):
    """设置前向传播上下文的上下文管理器。

    存储当前前向传播上下文（注意力元数据等），并在此注入每次模型前向传播的通用逻辑。

    Args:
        attn_metadata: 注意力元数据。
        vllm_config: vLLM配置对象。
        virtual_engine: 虚拟引擎索引，默认为0。
        num_tokens: 当前rank的token数量，可选。
        num_tokens_across_dp: 各DP rank的token数量张量，可选。
        cudagraph_runtime_mode: CUDA图运行时模式，默认不使用。
        batch_descriptor: 批次描述符，可选。
        ubatch_slices: 微批次切片，可选。
        slot_mapping: KV缓存槽位映射，可选。
        skip_compiled: 是否跳过编译模型调用。
    """
    global forward_start_time  # 声明使用全局前向传播开始时间变量
    need_to_track_batchsize = track_batchsize and attn_metadata is not None  # 判断是否需要追踪批量大小
    if need_to_track_batchsize:  # 如果需要追踪批量大小
        forward_start_time = time.perf_counter()  # 记录前向传播开始时间

    dp_metadata: DPMetadata | None = None  # 初始化数据并行元数据为None
    if (  # 如果满足以下条件
        vllm_config.parallel_config.data_parallel_size > 1  # 数据并行大小大于1
        and vllm_config.parallel_config.is_moe_model is not False  # 且为MoE模型
        and (attn_metadata is not None or num_tokens is not None)  # 且有注意力元数据或token数量
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if num_tokens_across_dp is None:  # 如果跨DP的token数量尚未初始化
            assert ubatch_slices is None  # 断言微批次切片为空
            assert num_tokens is not None  # 断言token数量不为空
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(  # 通过all-reduce协调各DP rank的批次
                num_tokens_unpadded=num_tokens,  # 未填充的token数量
                parallel_config=vllm_config.parallel_config,  # 并行配置
                allow_microbatching=False,  # 禁用微批次
            )
            assert num_tokens_across_dp is not None  # 断言协调后的token数量不为空
        dp_metadata = DPMetadata.make(  # 创建DPMetadata实例
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp  # 传入并行配置、token数和跨DP token数
        )

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:  # 如果使用CUDA图且token数量已知
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)  # 创建默认批次描述符

    additional_kwargs = current_platform.set_additional_forward_context(  # 调用平台特定方法设置额外前向上下文参数
        attn_metadata=attn_metadata,  # 注意力元数据
        vllm_config=vllm_config,  # vLLM配置
        virtual_engine=virtual_engine,  # 虚拟引擎索引
        dp_metadata=dp_metadata,  # 数据并行元数据
        num_tokens=num_tokens,  # token数量
        num_tokens_across_dp=num_tokens_across_dp,  # 跨DP的token数量
        cudagraph_runtime_mode=cudagraph_runtime_mode,  # CUDA图运行时模式
        batch_descriptor=batch_descriptor,  # 批次描述符
        ubatch_slices=ubatch_slices,  # 微批次切片
    )

    forward_context = create_forward_context(  # 创建前向上下文对象
        attn_metadata,  # 注意力元数据
        vllm_config,  # vLLM配置
        virtual_engine,  # 虚拟引擎索引
        dp_metadata,  # 数据并行元数据
        cudagraph_runtime_mode,  # CUDA图运行时模式
        batch_descriptor,  # 批次描述符
        ubatch_slices,  # 微批次切片
        slot_mapping,  # 槽位映射
        additional_kwargs,  # 额外参数
        skip_compiled,  # 是否跳过编译
    )

    try:  # 进入上下文
        with override_forward_context(forward_context):  # 使用override_forward_context设置前向上下文
            yield  # 让出控制权给调用方执行前向传播
    finally:  # 前向传播完成后执行清理和统计
        global last_logging_time, batchsize_logging_interval  # 声明使用全局日志时间和间隔变量
        if need_to_track_batchsize:  # 如果需要追踪批量大小
            batchsize = num_tokens  # 批量大小即为token数量
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            synchronize = current_platform.synchronize  # 获取平台同步方法
            if synchronize is not None:  # 如果平台支持同步操作
                synchronize()  # 执行GPU同步，确保前向传播完成
            now = time.perf_counter()  # 获取当前时间
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)  # 记录前向传播耗时（毫秒）
            if now - last_logging_time > batchsize_logging_interval:  # 如果距上次日志记录已超过间隔
                last_logging_time = now  # 更新上次日志记录时间
                forward_stats = []  # 初始化前向传播统计列表
                for bs, times in batchsize_forward_time.items():  # 遍历各批量大小的耗时记录
                    if len(times) <= 1:  # 如果记录数不超过1次
                        # can be cudagraph / profiling run
                        continue  # 跳过，可能是CUDA图捕获或性能分析运行
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()  # 计算耗时的中位数
                    medium = round(medium, 2)  # 四舍五入到两位小数
                    forward_stats.append((bs, len(times), medium))  # 添加统计信息（批量大小、次数、中位耗时）
                forward_stats.sort(key=lambda x: x[1], reverse=True)  # 按执行次数降序排序
                if forward_stats:  # 如果有统计数据
                    logger.info(  # 记录批量大小前向传播时间统计日志
                        (
                            "Batchsize forward time stats "  # 日志消息前缀
                            "(batchsize, count, median_time(ms)): %s"  # 日志格式：批量大小、次数、中位耗时
                        ),
                        forward_stats,  # 传入统计数据
                    )
