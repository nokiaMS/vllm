# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 导入枚举模块，用于定义各种枚举类型
import enum
# 导入时间模块，用于获取单调时钟时间戳
import time
# 从 collections.abc 导入 Mapping 抽象基类，用于类型注解中的映射类型
from collections.abc import Mapping
# 从 typing 导入 Any（任意类型）和 Literal（字面量类型），用于类型注解
from typing import Any, Literal

# 导入 msgspec 库，用于高效的结构体定义和序列化/反序列化
import msgspec
# 导入 numpy 库，用于数组操作（如路由专家信息）
import numpy as np
# 导入 PyTorch 库，用于张量操作（如嵌入输出和 prompt 嵌入）
import torch

# 从 vllm.lora.request 导入 LoRA 请求类，用于支持 LoRA 适配器推理
from vllm.lora.request import LoRARequest
# 从 vllm.multimodal.inputs 导入多模态特征规格类，描述图片/音频等多模态输入
from vllm.multimodal.inputs import MultiModalFeatureSpec
# 从 vllm.pooling_params 导入池化参数类，用于 embedding/分类等池化任务
from vllm.pooling_params import PoolingParams
# 从 vllm.sampling_params 导入采样参数类，包含 temperature、top_p 等生成配置
from vllm.sampling_params import SamplingParams
# 从 vllm.v1.metrics.stats 导入调度器统计信息类，包含队列长度、运行请求数等指标
from vllm.v1.metrics.stats import SchedulerStats
# 从 vllm.v1.outputs 导入日志概率的列表和张量表示类型
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
# 从 vllm.v1.serial_utils 导入 UtilityResult 类，用于包装 RPC 工具方法的返回值
from vllm.v1.serial_utils import UtilityResult

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
# 暂停生成模式的类型定义："abort" 立即中止、"wait" 等待完成、"keep" 冻结队列
PauseMode = Literal["abort", "wait", "keep"]  # 定义暂停模式的字面量联合类型

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
# 请求结束原因的字符串表示，作为外部 API 的一部分，与 FinishReason 枚举值一一对应
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")  # 结束原因字符串元组，索引与 FinishReason 枚举值对应

# [中文注释] Elastic EP 通知使用的特殊 call_id，用于区分普通 utility 调用和 EEP 伸缩通知
EEP_NOTIFICATION_CALL_ID = -1  # 弹性专家并行通知的特殊调用 ID，值为 -1 以区别于正常的非负 call_id


# [中文注释] Elastic Expert Parallelism 弹性伸缩过程中的通知类型枚举：
#   NEW_CORE_ENGINES_INIT_READY — 新引擎进程初始化完成
#   NEW_CORE_ENGINES_WEIGHTS_INIT_READY — 新引擎权重加载完成
#   RECONFIGURE_FINISHED — 所有引擎完成分布式重配置
#   SHUTDOWN_COMPLETE — 缩容场景中被移除的引擎已关闭
class EEPNotificationType(enum.Enum):  # 弹性专家并行通知类型枚举类定义
    """弹性专家并行（Elastic Expert Parallelism）通知类型枚举。

    用于在弹性伸缩过程中，各引擎之间传递生命周期事件通知。
    """

    NEW_CORE_ENGINES_INIT_READY = "NEW_CORE_ENGINES_INIT_READY"  # 新核心引擎初始化就绪
    NEW_CORE_ENGINES_WEIGHTS_INIT_READY = "NEW_CORE_ENGINES_WEIGHTS_INIT_READY"  # 新核心引擎权重初始化就绪
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"  # 分布式重配置完成
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"  # 引擎关闭完成（缩容场景）


# [中文注释] 请求结束原因枚举。使用 IntEnum 而非字符串，序列化更紧凑。
#   STOP=0 — 匹配到停止词；LENGTH=1 — 达到 max_tokens/max_model_len；
#   ABORT=2 — 客户端主动取消；ERROR=3 — 可重试的内部错误（如 KV 加载失败）；
#   REPETITION=4 — 检测到重复 token 模式（幻觉）
class FinishReason(enum.IntEnum):  # 请求结束原因整数枚举类定义
    """请求结束原因枚举类。

    使用整数枚举（IntEnum）而非字符串枚举，以实现更紧凑的序列化。
    每个枚举值对应一种请求结束的原因。

    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    STOP = 0  # 匹配到停止词或停止 token，正常结束
    LENGTH = 1  # 达到最大 token 数限制（max_tokens 或 max_model_len）
    ABORT = 2  # 被客户端主动取消/中止
    ERROR = 3  # 可重试的请求级内部错误（例如 KV 缓存加载失败），始终转为 500 错误
    REPETITION = 4  # 检测到重复 token 模式（幻觉检测），主动终止生成

    def __str__(self):  # 定义字符串转换方法，将枚举值转为可读字符串
        """将枚举值转换为对应的字符串表示，通过索引 FINISH_REASON_STRINGS 元组实现。"""
        return FINISH_REASON_STRINGS[self.value]  # 使用枚举整数值作为索引，返回对应的字符串


# [中文注释] 从 Client 发往 Engine Core 的推理请求。使用 msgspec.Struct 实现高效序列化。
#   array_like=True — 序列化为数组（而非字典），更紧凑
#   omit_defaults=True — 默认值字段不参与序列化，减少数据量
#   gc=False — 告诉 Python GC 不追踪此对象（纯数据结构，无循环引用风险），提升性能
#   关键字段：
#     request_id — 内部请求 ID（用于路由和追踪）
#     prompt_token_ids — 已 tokenize 的 prompt
#     mm_features — 多模态特征（图片/音频等）
#     sampling_params / pooling_params — 采样或池化参数（二选一）
#     data_parallel_rank — 显式指定目标 DP rank（可选）
#     current_wave — DP 场景的 wave 序号，处理请求发送与 wave 结束通知之间的竞争条件
#     client_index — 前端 scale-out 时标识请求来源的客户端编号
class EngineCoreRequest(  # 引擎核心推理请求类定义
    msgspec.Struct,  # 继承 msgspec.Struct 基类，支持高效序列化
    array_like=True,  # type: ignore[call-arg]  # 序列化为数组格式，比字典更紧凑
    omit_defaults=True,  # type: ignore[call-arg]  # 省略默认值字段，减少序列化体积
    gc=False,  # 禁用垃圾回收追踪，纯数据结构无循环引用风险
):  # type: ignore[call-arg]  # 禁用 GC 追踪，此结构体无循环引用
    """引擎核心推理请求结构体。

    封装从客户端发往引擎核心的所有推理请求信息，包括 prompt token、
    多模态特征、采样/池化参数、LoRA 配置等。
    使用 msgspec.Struct 实现高效的零拷贝序列化。
    """

    request_id: str  # 请求的唯一标识符（内部 ID，用于引擎内部路由和追踪）
    prompt_token_ids: list[int] | None  # 已分词的 prompt token ID 列表，None 表示使用 prompt_embeds
    mm_features: list[MultiModalFeatureSpec] | None  # 多模态特征规格列表（图片/音频/视频），None 表示纯文本
    sampling_params: SamplingParams | None  # 采样参数（生成任务使用），与 pooling_params 二选一
    pooling_params: PoolingParams | None  # 池化参数（embedding/分类任务使用），与 sampling_params 二选一
    arrival_time: float  # 请求到达时间戳，用于计算排队延迟和调度优先级
    lora_request: LoRARequest | None  # LoRA 适配器请求信息，None 表示不使用 LoRA
    cache_salt: str | None  # 缓存盐值，用于前缀缓存的隔离（不同盐值的请求不共享缓存）
    data_parallel_rank: int | None  # 显式指定的数据并行 rank，None 表示由调度器自动分配
    prompt_embeds: torch.Tensor | None = None  # 预计算的 prompt 嵌入张量，替代 prompt_token_ids 使用

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0  # 客户端索引，前端水平扩展时用于将响应路由回正确的客户端

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0  # 当前 wave 序号，DP 场景下解决请求发送与 wave 完成通知之间的竞争条件
    priority: int = 0  # 请求优先级，数值越大优先级越高，用于优先级调度器

    trace_headers: Mapping[str, str] | None = None  # 分布式追踪头信息，用于链路追踪（如 OpenTelemetry）
    resumable: bool = False  # 是否支持断点续传，True 表示请求可以在中断后恢复

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None  # 用户提供的外部请求 ID，用于输出和外部取消操作

    reasoning_ended: bool | None = None  # 推理/思考阶段是否已结束，用于支持思维链推理模型

    @property  # 声明为属性方法装饰器
    def params(self) -> SamplingParams | PoolingParams:  # 获取请求参数的属性方法
        """返回已处理的参数（采样参数或池化参数）。

        根据请求类型返回对应的参数对象：生成任务返回 SamplingParams，
        池化任务返回 PoolingParams。两者必有其一不为 None。
        """
        if self.sampling_params is not None:  # 如果采样参数不为空，说明是生成任务
            return self.sampling_params  # 返回采样参数
        assert self.pooling_params is not None  # 断言池化参数不为空（两者必有其一）
        return self.pooling_params  # 返回池化参数


# [中文注释] 请求在 Engine Core 内部的事件类型：QUEUED=入队、SCHEDULED=被调度、PREEMPTED=被抢占
class EngineCoreEventType(enum.IntEnum):  # 引擎核心事件类型整数枚举类定义
    """引擎核心请求事件类型枚举。

    表示请求在引擎核心内部经历的生命周期事件，
    用于指标统计和性能分析。
    """

    QUEUED = 1  # 请求进入等待队列
    SCHEDULED = 2  # 请求被调度器选中，开始执行
    PREEMPTED = 3  # 请求被抢占（因资源不足让出 GPU 资源）


# [中文注释] 带时间戳的请求事件记录，使用 monotonic 时钟（仅在单进程内可比较）。
#   用于前端计算排队时间、调度延迟等指标。
class EngineCoreEvent(msgspec.Struct):  # 引擎核心事件结构体类定义，继承 msgspec.Struct
    """带时间戳的引擎核心事件结构体。

    记录请求在引擎核心中经历的事件及其发生时间。
    时间戳使用单调时钟（monotonic clock），仅用于同一进程内的时间间隔计算，
    不应与其他进程的时间戳进行比较。

    A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamps and is used for by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """

    type: EngineCoreEventType  # 事件类型（入队/调度/抢占）
    timestamp: float  # 事件发生的单调时钟时间戳

    @classmethod  # 声明为类方法装饰器
    def new_event(  # 创建新事件的工厂类方法
        cls, event_type: EngineCoreEventType, timestamp: float | None = None  # 参数：事件类型和可选时间戳
    ) -> "EngineCoreEvent":  # 返回值类型为 EngineCoreEvent 实例
        """创建新的引擎核心事件实例的工厂方法。

        Args:
            event_type: 事件类型（QUEUED/SCHEDULED/PREEMPTED）
            timestamp: 可选的时间戳，为 None 时自动使用当前单调时钟时间

        Returns:
            新创建的 EngineCoreEvent 实例
        """
        timestamp = time.monotonic() if timestamp is None else timestamp  # 如果未提供时间戳，使用当前单调时钟时间
        return cls(event_type, timestamp)  # 调用构造函数创建事件实例并返回


# [中文注释] 单个请求的推理输出。从 Engine Core 发回 Client。
#   关键字段：
#     request_id — 对应的请求 ID
#     new_token_ids — 本次 step 新生成的 token ID 列表（流式输出）
#     new_logprobs — 新 token 的 log 概率（可选）
#     pooling_output — 池化模型（如 embedding）的输出 tensor（可选）
#     finish_reason — 结束原因（None 表示尚未结束）
#     num_cached_tokens — 前缀缓存命中的 token 数
#     routed_experts — MoE 模型路由的专家信息
class EngineCoreOutput(  # 引擎核心单个请求输出类定义
    msgspec.Struct,  # 继承 msgspec.Struct 基类，支持高效序列化
    array_like=True,  # type: ignore[call-arg]  # 序列化为数组格式
    omit_defaults=True,  # type: ignore[call-arg]  # 省略默认值字段
    gc=False,  # 禁用垃圾回收追踪，纯数据结构无循环引用风险
):  # type: ignore[call-arg]  # 禁用 GC 追踪
    """引擎核心单个请求的输出结构体。

    包含一次推理步骤（step）中单个请求的所有输出信息，
    包括新生成的 token、日志概率、池化输出、结束原因等。
    通过 ZMQ 从引擎核心发送回客户端前端。
    """

    request_id: str  # 对应请求的唯一标识符
    new_token_ids: list[int]  # 本次 step 新生成的 token ID 列表（增量/流式输出）

    new_logprobs: LogprobsLists | None = None  # 新生成 token 的日志概率（列表格式），None 表示未请求
    new_prompt_logprobs_tensors: LogprobsTensors | None = None  # prompt token 的日志概率（张量格式），用于 prompt logprobs 功能

    pooling_output: torch.Tensor | None = None  # 池化模型的输出张量（embedding/分类结果），生成任务为 None

    finish_reason: FinishReason | None = None  # 请求结束原因，None 表示请求尚未完成
    stop_reason: int | str | None = None  # 具体的停止原因（停止词内容或停止 token ID）
    events: list[EngineCoreEvent] | None = None  # 请求生命周期事件列表，用于指标计算
    kv_transfer_params: dict[str, Any] | None = None  # KV 缓存传输参数，用于分离式推理（disaggregated inference）

    trace_headers: Mapping[str, str] | None = None  # 分布式追踪头信息，透传给输出
    # The number of tokens with prefix cache hits (local + external).
    num_cached_tokens: int = 0  # 前缀缓存命中的 token 数量（本地缓存 + 外部缓存之和）
    # The number of tokens computed remotely (original count from connector).
    num_external_computed_tokens: int = 0  # 远程计算的 token 数量（来自外部 KV 连接器的原始计数）
    routed_experts: np.ndarray | None = None  # MoE 模型中被路由到的专家索引数组
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    num_nans_in_logits: int = 0  # logits 中 NaN 的数量，大于 0 表示输出已损坏

    @property  # 声明为属性方法装饰器
    def finished(self) -> bool:  # 判断请求是否已完成的属性方法
        """判断请求是否已完成。

        Returns:
            True 表示请求已结束（有结束原因），False 表示仍在生成中。
        """
        return self.finish_reason is not None  # 如果 finish_reason 不为 None，说明请求已完成


# [中文注释] RPC 工具方法（utility）调用的返回消息。
#   call_id — 与请求时的 call_id 对应，用于匹配 Future
#   failure_message — 非 None 表示调用失败（此时 result 为 None）
#   result — UtilityResult 包装的返回值（使用 pickle 序列化以支持任意 Python 对象）
class UtilityOutput(  # RPC 工具方法输出类定义
    msgspec.Struct,  # 继承 msgspec.Struct 基类，支持高效序列化
    array_like=True,  # type: ignore[call-arg]  # 序列化为数组格式
    gc=False,  # 禁用垃圾回收追踪，纯数据结构无循环引用风险
):  # type: ignore[call-arg]  # 禁用 GC 追踪
    """RPC 工具方法调用的输出结构体。

    封装引擎核心执行工具方法（如 profile、reset_cache、add_lora 等）后的返回结果。
    通过 call_id 与请求端的 Future 进行匹配。
    """

    call_id: int  # 调用 ID，与请求时的 call_id 一一对应，用于匹配异步 Future

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None  # 失败消息，非 None 表示调用失败（此时 result 应为 None）
    result: UtilityResult | None = None  # 调用结果，使用 UtilityResult 包装（支持 pickle 序列化任意 Python 对象）


# [中文注释] Engine Core 向 Client 发送的批量输出消息（一次 step 的所有结果）。
#   这是 ZMQ output_socket 上传输的顶层消息结构：
#     engine_index — 产生此输出的引擎编号（DP 场景区分来源）
#     outputs — 本次 step 所有请求的 EngineCoreOutput 列表
#     scheduler_stats — 调度器统计信息（队列长度、运行请求数等）
#     utility_output — RPC 工具方法返回值（与普通 outputs 互斥：要么是推理结果，要么是 RPC 结果）
#     finished_requests — 本次结束的请求 ID 集合（DPLBAsyncMPClient 用于清理 reqs_in_flight）
#     wave_complete — DP 场景，非 None 表示当前 wave 结束，引擎进入暂停
#     start_wave — DP 场景，收到旧 wave 的请求时触发下一 wave 启动
class EngineCoreOutputs(  # 引擎核心批量输出类定义
    msgspec.Struct,  # 继承 msgspec.Struct 基类，支持高效序列化
    array_like=True,  # type: ignore[call-arg]  # 序列化为数组格式
    omit_defaults=True,  # type: ignore[call-arg]  # 省略默认值字段
    gc=False,  # 禁用垃圾回收追踪，纯数据结构无循环引用风险
):  # type: ignore[call-arg]  # 禁用 GC 追踪
    """引擎核心批量输出结构体。

    封装一次推理步骤（step）中所有请求的输出结果，是 ZMQ 输出通道上传输的顶层消息。
    同时携带调度器统计信息和数据并行 wave 控制信号。
    """

    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0  # 产生此输出的引擎索引，DP 场景下用于区分不同引擎的输出

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []  # 本次 step 所有请求的输出列表，长度等于本次处理的请求数
    scheduler_stats: SchedulerStats | None = None  # 调度器统计信息（等待队列长度、运行请求数、KV 缓存使用率等）
    timestamp: float = 0.0  # 输出时间戳，默认为 0.0 表示将在 __post_init__ 中自动设置

    utility_output: UtilityOutput | None = None  # RPC 工具方法的返回结果，与 outputs 互斥
    finished_requests: set[str] | None = None  # 本次结束的请求 ID 集合，用于清理 reqs_in_flight 映射

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None  # DP 场景的 wave 完成信号，非 None 表示指定 wave 已完成，引擎已暂停
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None  # DP 场景的 wave 启动信号，非 None 表示需要在其他引擎中启动新 wave

    def __post_init__(self):  # 结构体初始化后的后处理方法
        """初始化后处理：如果时间戳未设置（为默认值 0.0），则自动记录当前单调时钟时间。"""
        if self.timestamp == 0.0:  # 检查时间戳是否为默认值
            self.timestamp = time.monotonic()  # 使用单调时钟设置时间戳，避免系统时钟回拨影响


# [中文注释] Client → Engine Core 的请求类型枚举。
#   值为单字节 bytes，可直接作为 ZMQ 帧发送，无需额外序列化。
#   ADD=\x00 — 提交新推理请求
#   ABORT=\x01 — 取消指定请求
#   START_DP_WAVE=\x02 — DP 场景启动新一轮 wave
#   UTILITY=\x03 — RPC 工具方法调用（profile、reset_cache、add_lora 等）
#   EXECUTOR_FAILED=\x04 — 内部哨兵，executor 异常时由 EngineCoreProc 产生
#   WAKEUP=\x05 — 内部哨兵，shutdown 时唤醒阻塞的 input_queue.get()
class EngineCoreRequestType(enum.Enum):  # 引擎核心请求类型枚举类定义
    """引擎核心请求类型枚举。

    定义从客户端发往引擎核心的所有请求类型。
    每个枚举值为单字节 bytes，可直接作为 ZMQ 消息帧发送，无需额外编码步骤。

    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"  # 提交新的推理请求
    ABORT = b"\x01"  # 取消/中止指定的请求
    START_DP_WAVE = b"\x02"  # 数据并行场景下启动新一轮 wave
    UTILITY = b"\x03"  # RPC 工具方法调用（如 profile、reset_prefix_cache、add_lora 等）
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"  # 内部哨兵值，executor 执行失败时由 EngineCoreProc 产生
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"  # 内部哨兵值，关闭时用于唤醒阻塞在 input_queue.get() 上的线程


# [中文注释] Elastic EP 弹性伸缩时发送给引擎的分布式重配置请求。
#   包含新的 DP 大小、rank、master 地址/端口、以及各种 stateless 通信组端口列表。
#   引擎收到后调用 reinitialize_distributed 重建分布式通信组。
class ReconfigureDistributedRequest(msgspec.Struct):  # 分布式重配置请求结构体类定义
    """分布式重配置请求结构体。

    在弹性专家并行（Elastic EP）伸缩过程中使用，包含重建分布式通信组所需的全部配置信息。
    引擎收到此请求后会调用 reinitialize_distributed 方法重建所有分布式通信组。
    """

    new_data_parallel_size: int  # 新的数据并行大小（引擎总数）
    new_data_parallel_rank: int  # 当前引擎的新数据并行 rank
    new_data_parallel_rank_local: int  # 当前引擎在本节点内的新局部 rank
    new_data_parallel_master_ip: str  # 新的数据并行 master 节点 IP 地址
    new_data_parallel_master_port: int  # 新的数据并行 master 节点端口号
    new_data_parallel_master_port_list: list[int]  # 所有 DP rank 的 master 端口列表
    new_stateless_world_group_port_list: list[list[int]]  # 无状态全局通信组的端口列表（二维：每个组一个端口列表）
    new_stateless_dp_group_port_list: list[list[int]]  # 无状态数据并行通信组的端口列表
    new_stateless_ep_group_port_list: list[list[int]]  # 无状态专家并行通信组的端口列表
    new_stateless_eplb_group_port_list: list[list[int]]  # 无状态专家并行负载均衡通信组的端口列表


# [中文注释] 重配置时的 rank 特殊标记：
#   KEEP_CURRENT_RANK=-1 — 保持当前 rank 不变（扩容/缩容时现有引擎使用）
#   SHUTDOWN_CURRENT_RANK=-2 — 标记此引擎需要关闭（缩容时多余的引擎使用）
class ReconfigureRankType(enum.IntEnum):  # 重配置 rank 类型整数枚举类定义
    """重配置 rank 类型枚举。

    在分布式重配置过程中使用的特殊 rank 标记值，
    用于指示引擎应保持当前 rank 还是执行关闭操作。

    Rank type for reconfiguring distributed request.
    """

    KEEP_CURRENT_RANK = -1  # 保持当前 rank 不变，用于扩容/缩容时已有引擎保持原有角色
    SHUTDOWN_CURRENT_RANK = -2  # 标记当前引擎需要关闭，用于缩容时移除多余的引擎
