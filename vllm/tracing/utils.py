# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from collections.abc import Mapping  # 从 collections.abc 导入 Mapping 抽象基类

from vllm.logger import init_logger  # 从 vLLM 日志模块导入日志初始化函数
from vllm.utils.func_utils import run_once  # 导入 run_once 装饰器，确保函数只执行一次

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

# Standard W3C headers used for context propagation
# 用于上下文传播的标准 W3C 追踪头
TRACE_HEADERS = ["traceparent", "tracestate"]  # W3C 分布式追踪标准的请求头列表


class SpanAttributes:
    """Span 标准属性常量类。

    这些属性主要基于 OpenTelemetry 语义约定定义，
    但作为常量定义在此处，以便任何后端或日志记录器都可以使用。
    """

    # Attribute names copied from OTel semantic conventions to avoid version conflicts
    # 从 OTel 语义约定复制的属性名，避免版本冲突
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"  # 生成式AI使用的补全 token 数
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"  # 生成式AI使用的提示 token 数
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"  # 请求的最大 token 数
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"  # 请求的 top_p 采样参数
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"  # 请求的温度参数
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"  # 响应使用的模型名称

    # Custom attributes added until they are standardized
    # 自定义属性，在标准化之前使用
    GEN_AI_REQUEST_ID = "gen_ai.request.id"  # 请求唯一标识符
    GEN_AI_REQUEST_N = "gen_ai.request.n"  # 请求生成的序列数量
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"  # 使用的序列数量
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"  # 请求在队列中的等待时间
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"  # 首个 token 的延迟时间（TTFT）
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"  # 端到端延迟时间
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"  # 调度器中的耗时

    # Latency breakdowns
    # 延迟细分指标
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"  # 模型前向传播耗时
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"  # 模型执行耗时
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"  # 模型预填充阶段耗时
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"  # 模型解码阶段耗时
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"  # 模型推理耗时


class LoadingSpanAttributes:
    """代码级别追踪的自定义属性类（文件路径、行号等）。"""

    CODE_NAMESPACE = "code.namespace"  # 代码命名空间（模块名）
    CODE_FUNCTION = "code.function"  # 函数名称
    CODE_FILEPATH = "code.filepath"  # 文件路径
    CODE_LINENO = "code.lineno"  # 代码行号


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    """检查提供的请求头字典是否包含追踪上下文头。

    Args:
        headers: 需要检查的请求头映射。

    Returns:
        如果包含任何追踪头则返回 True，否则返回 False。
    """
    return any(h in headers for h in TRACE_HEADERS)  # 检查是否存在任何追踪相关的请求头


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    """从完整的请求头字典中提取追踪相关的请求头。

    适用于日志记录或将上下文传递给非 OTel 客户端的场景。

    Args:
        headers: 完整的请求头映射。

    Returns:
        仅包含追踪相关请求头的字典。
    """
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}  # 过滤并返回追踪相关的请求头


@run_once  # 确保该函数只执行一次的装饰器
def log_tracing_disabled_warning() -> None:
    """记录追踪被禁用的警告日志。

    当接收到包含追踪上下文的请求但追踪功能未启用时调用。
    使用 @run_once 装饰器确保警告只记录一次，避免日志泛滥。
    """
    logger.warning("Received a request with trace context but tracing is disabled")  # 记录追踪未启用的警告信息
