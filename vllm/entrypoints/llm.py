# SPDX-License-Identifier: Apache-2.0
# SPDX 许可证标识符：Apache-2.0 协议
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX 文件版权声明：版权归 vLLM 项目贡献者所有

import itertools  # 导入迭代工具模块，用于高效迭代操作
from collections.abc import Callable, Iterable, Sequence  # 从抽象基类模块导入可调用对象、可迭代对象和序列类型
from pathlib import Path  # 导入路径模块，用于文件路径操作
from typing import TYPE_CHECKING, Any  # 导入类型检查标志和通用类型

import cloudpickle  # 导入 cloudpickle，用于序列化复杂 Python 对象（如类）
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from pydantic import ValidationError  # 从 pydantic 导入数据验证错误类
from tqdm.auto import tqdm  # 导入自动适配环境的进度条模块
from typing_extensions import TypeVar, overload  # 从类型扩展模块导入 TypeVar 和函数重载装饰器

from vllm.beam_search import (  # 从 vllm 束搜索模块导入相关类和函数
    BeamSearchInstance,  # 束搜索实例类
    BeamSearchOutput,  # 束搜索输出类
    BeamSearchSequence,  # 束搜索序列类
    create_sort_beams_key_function,  # 创建束排序键函数的工厂函数
)
from vllm.config import (  # 从 vllm 配置模块导入各种配置类
    AttentionConfig,  # 注意力机制配置类
    CompilationConfig,  # 编译配置类
    PoolerConfig,  # 池化层配置类
    ProfilerConfig,  # 性能分析器配置类
    StructuredOutputsConfig,  # 结构化输出配置类
    is_init_field,  # 判断字段是否为初始化字段的函数
)
from vllm.config.compilation import CompilationMode  # 从编译配置模块导入编译模式枚举
from vllm.config.model import (  # 从模型配置模块导入相关选项类型
    ConvertOption,  # 模型转换选项类型
    HfOverrides,  # HuggingFace 配置覆盖类型
    ModelDType,  # 模型数据类型选项
    RunnerOption,  # 运行器选项类型
    TokenizerMode,  # 分词器模式类型
)
from vllm.distributed.weight_transfer.base import (  # 从分布式权重传输模块导入请求类
    WeightTransferInitRequest,  # 权重传输初始化请求类
    WeightTransferUpdateRequest,  # 权重传输更新请求类
)
from vllm.engine.arg_utils import EngineArgs  # 从引擎参数工具模块导入引擎参数类
from vllm.entrypoints.chat_utils import (  # 从聊天工具模块导入相关类型和函数
    ChatCompletionMessageParam,  # 聊天补全消息参数类型
    ChatTemplateConfig,  # 聊天模板配置类
    ChatTemplateContentFormatOption,  # 聊天模板内容格式选项
    load_chat_template,  # 加载聊天模板的函数
)
from vllm.entrypoints.pooling.io_processor_factories import init_pooling_io_processors  # 导入池化 IO 处理器初始化函数
from vllm.entrypoints.pooling.score.utils import (  # 从评分工具模块导入相关类和函数
    ScoreData,  # 评分数据类
    ScoreMultiModalParam,  # 多模态评分参数类
    _cosine_similarity,  # 余弦相似度计算函数
    compress_token_type_ids,  # 压缩 token 类型 ID 的函数
    compute_maxsim_score,  # 计算最大相似度评分的函数
    get_score_prompt,  # 获取评分提示的函数
    score_data_to_prompts,  # 将评分数据转换为提示的函数
    validate_score_input,  # 验证评分输入的函数
)
from vllm.entrypoints.utils import log_non_default_args  # 导入记录非默认参数的工具函数
from vllm.inputs.data import (  # 从输入数据模块导入各种提示类型
    DataPrompt,  # 数据提示类型
    ProcessorInputs,  # 处理器输入类型
    PromptType,  # 提示类型（联合类型）
    SingletonPrompt,  # 单例提示类型
    TextPrompt,  # 文本提示类型
    TokensPrompt,  # Token ID 提示类型
)
from vllm.logger import init_logger  # 从日志模块导入日志初始化函数
from vllm.lora.request import LoRARequest  # 从 LoRA 模块导入 LoRA 请求类
from vllm.model_executor.layers.quantization import QuantizationMethods  # 导入量化方法枚举
from vllm.outputs import (  # 从输出模块导入各种请求输出类
    ClassificationRequestOutput,  # 分类请求输出类
    EmbeddingRequestOutput,  # 嵌入请求输出类
    PoolingRequestOutput,  # 池化请求输出类
    RequestOutput,  # 通用请求输出类
    ScoringRequestOutput,  # 评分请求输出类
)
from vllm.platforms import current_platform  # 导入当前平台信息对象
from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.renderers import ChatParams, merge_kwargs  # 从渲染器模块导入聊天参数和合并关键字参数函数
from vllm.renderers.inputs.preprocess import (  # 从渲染器输入预处理模块导入相关函数
    conversation_to_seq,  # 将对话转换为序列的函数
    parse_model_prompt,  # 解析模型提示的函数
    prompt_to_seq,  # 将提示转换为序列的函数
)
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams  # 导入采样参数相关类
from vllm.tasks import PoolingTask  # 导入池化任务类型
from vllm.tokenizers import TokenizerLike  # 导入分词器接口类型
from vllm.usage.usage_lib import UsageContext  # 导入使用上下文枚举
from vllm.utils.counter import Counter  # 导入计数器工具类
from vllm.utils.mistral import is_mistral_tokenizer  # 导入 Mistral 分词器检测函数
from vllm.utils.tqdm_utils import maybe_tqdm  # 导入可选进度条工具函数
from vllm.v1.engine import PauseMode  # 导入引擎暂停模式枚举
from vllm.v1.engine.llm_engine import LLMEngine  # 导入 LLM 引擎类（v1 版本）
from vllm.v1.sample.logits_processor import LogitsProcessor  # 导入 logits 处理器基类

if TYPE_CHECKING:  # 仅在类型检查时执行的导入，避免运行时循环导入
    from vllm.v1.metrics.reader import Metric  # 导入指标读取器中的 Metric 类型（仅用于类型注解）

logger = init_logger(__name__)  # 使用当前模块名初始化日志记录器

# 定义输出类型变量 _O，约束为 RequestOutput 或 PoolingRequestOutput 的子类
_O = TypeVar(
    "_O",
    bound=RequestOutput | PoolingRequestOutput,  # 类型上界：请求输出或池化请求输出
    default=RequestOutput | PoolingRequestOutput,  # 默认类型
)
_P = TypeVar("_P", bound=SamplingParams | PoolingParams | None)  # 定义参数类型变量 _P，约束为采样参数、池化参数或 None
_R = TypeVar("_R", default=Any)  # 定义返回类型变量 _R，默认为 Any 类型


# LLM 类：用于从给定提示词和采样参数生成文本的大语言模型主类
class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images
            or videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        allowed_media_domains: If set, only media URLs that belong to this
            domain can be used for multi-modal inputs.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `dtype` attribute of the Transformers model's config. However,
            if the `dtype` in the config is `float32`, we will use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        chat_template: The chat template to apply.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,
            this is set to None and vllm can automatically infer the kv cache
            size based on gpu_memory_utilization. However, users may want to
            manually specify the kv cache memory size. kv_cache_memory_bytes
            allows more fine-grain control of how much memory gets used when
            compared with using gpu_memory_utilization. Note that
            kv_cache_memory_bytes (when not-None) ignores
            gpu_memory_utilization
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        offload_group_size: Prefetch offloading: Group every N layers
            together. Offload last `offload_num_in_group` layers of each group.
            Default is 0 (disabled).
        offload_num_in_group: Prefetch offloading: Number of layers to
            offload per group. Default is 1.
        offload_prefetch_step: Prefetch offloading: Number of layers to
            prefetch ahead. Higher values hide more latency but use more GPU
            memory. Default is 1.
        offload_params: Prefetch offloading: Set of parameter name segments
            to selectively offload. Only parameters whose names contain one of
            these segments will be offloaded (e.g., {"gate_up_proj", "down_proj"}
            for MLP weights, or {"w13_weight", "w2_weight"} for MoE expert
            weights). If None or empty, all parameters are offloaded.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        enable_return_routed_experts: Whether to return routed experts.
        disable_custom_all_reduce: See
            [ParallelConfig][vllm.config.ParallelConfig].
        hf_token: The token to use as HTTP bearer authorization for remote files
            . If `True`, will use the token generated when running
            `hf auth login` (stored in `~/.cache/huggingface/token`).
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor. Overrides for the
            multi-modal processor obtained from `AutoProcessor.from_pretrained`.
            The available overrides depend on the model that is being run.
            For example, for Phi-3-Vision: `{"num_crops": 4}`.
        pooler_config: Initialize non-default pooling config for the pooling model,
            e.g., `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the mode of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        attention_config: Configuration for attention mechanisms. Can be a
            dictionary or an AttentionConfig instance. If a dictionary, it will
            be converted to an AttentionConfig. Allows specifying the attention
            backend and other attention-related settings.
        **kwargs: Arguments for [`EngineArgs`][vllm.EngineArgs].

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the [AsyncLLMEngine][vllm.AsyncLLMEngine] class instead.

    （中文翻译）

    用于根据给定的提示词和采样参数生成文本的大语言模型类。

    该类包含一个分词器、一个语言模型（可能跨多个 GPU 分布式部署），
    以及为中间状态（即 KV 缓存）分配的 GPU 内存空间。给定一批提示词和
    采样参数，该类使用智能批处理机制和高效内存管理从模型生成文本。

    参数：
        model: HuggingFace Transformers 模型的名称或路径。
        tokenizer: HuggingFace Transformers 分词器的名称或路径。
        tokenizer_mode: 分词器模式。"auto" 会优先使用快速分词器（如果可用），
            "slow" 则始终使用慢速分词器。
        skip_tokenizer_init: 若为 True，跳过分词器和反分词器的初始化。
            此时输入应提供有效的 prompt_token_ids，prompt 应为 None。
        trust_remote_code: 下载模型和分词器时是否信任远程代码
            （例如来自 HuggingFace 的代码）。
        allowed_local_media_path: 允许 API 请求从服务器文件系统指定的目录
            读取本地图像或视频。这存在安全风险，仅应在受信任的环境中启用。
        allowed_media_domains: 若设置，仅允许属于该域名的媒体 URL
            用于多模态输入。
        tensor_parallel_size: 张量并行分布式执行使用的 GPU 数量。
        dtype: 模型权重和激活值的数据类型。目前支持 `float32`、`float16`
            和 `bfloat16`。若为 `auto`，则使用 Transformers 模型配置中的
            `dtype` 属性。但如果配置中的 `dtype` 为 `float32`，
            则会改用 `float16`。
        quantization: 模型权重的量化方法。目前支持 "awq"、"gptq"
            和 "fp8"（实验性）。若为 None，先检查模型配置文件中的
            `quantization_config` 属性；若也为 None，则假定模型权重
            未量化，使用 `dtype` 来确定权重的数据类型。
        revision: 要使用的特定模型版本。可以是分支名、标签名或提交 ID。
        tokenizer_revision: 要使用的特定分词器版本。可以是分支名、
            标签名或提交 ID。
        chat_template: 要应用的聊天模板。
        seed: 用于初始化采样随机数生成器的种子值。
        gpu_memory_utilization: 为模型权重、激活值和 KV 缓存预留的
            GPU 内存比例（介于 0 和 1 之间）。较高的值会增大 KV 缓存，
            从而提升模型吞吐量。但如果值过高，可能会导致内存溢出（OOM）错误。
        kv_cache_memory_bytes: 每个 GPU 上 KV 缓存的字节大小。默认为 None，
            vllm 可根据 gpu_memory_utilization 自动推断 KV 缓存大小。
            用户也可手动指定 KV 缓存内存大小。与 gpu_memory_utilization 相比，
            kv_cache_memory_bytes 提供了更细粒度的内存控制。
            注意：当 kv_cache_memory_bytes 不为 None 时，将忽略
            gpu_memory_utilization。
        cpu_offload_gb: 用于卸载模型权重的 CPU 内存大小（GiB）。这会虚拟地
            扩大可用于存放模型权重的 GPU 内存空间，代价是每次前向传播都需要
            进行 CPU-GPU 数据传输。
        offload_group_size: 预取卸载：每 N 层为一组。卸载每组最后
            `offload_num_in_group` 层。默认为 0（禁用）。
        offload_num_in_group: 预取卸载：每组卸载的层数。默认为 1。
        offload_prefetch_step: 预取卸载：提前预取的层数。较高的值可隐藏
            更多延迟，但会占用更多 GPU 内存。默认为 1。
        offload_params: 预取卸载：选择性卸载的参数名片段集合。仅名称包含
            这些片段之一的参数会被卸载（例如 {"gate_up_proj", "down_proj"}
            用于 MLP 权重，{"w13_weight", "w2_weight"} 用于 MoE 专家权重）。
            若为 None 或空集，则卸载所有参数。
        enforce_eager: 是否强制使用 eager 执行模式。若为 True，将禁用
            CUDA graph，始终以 eager 模式执行模型。若为 False，
            将混合使用 CUDA graph 和 eager 执行。
        enable_return_routed_experts: 是否返回路由专家信息。
        disable_custom_all_reduce: 参见
            [ParallelConfig][vllm.config.ParallelConfig]。
        hf_token: 用于远程文件 HTTP Bearer 授权的令牌。若为 `True`，
            将使用运行 `hf auth login` 时生成的令牌
            （存储在 `~/.cache/huggingface/token` 中）。
        hf_overrides: 若为字典，包含转发给 HuggingFace 配置的参数。
            若为可调用对象，则调用它来更新 HuggingFace 配置。
        mm_processor_kwargs: 转发给模型多模态数据处理器（如图像处理器）的参数。
            覆盖从 `AutoProcessor.from_pretrained` 获取的多模态处理器的默认值。
            可用的覆盖项取决于所运行的模型。
            例如，对于 Phi-3-Vision：`{"num_crops": 4}`。
        pooler_config: 为池化模型初始化非默认的池化配置，
            例如 `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`。
        compilation_config: 可以是整数或字典。若为整数，用作编译优化模式。
            若为字典，可指定完整的编译配置。
        attention_config: 注意力机制配置。可以是字典或 AttentionConfig 实例。
            若为字典，将被转换为 AttentionConfig。允许指定注意力后端
            和其他注意力相关设置。
        **kwargs: 传递给 [`EngineArgs`][vllm.EngineArgs] 的参数。

    注意：
        该类用于离线推理。在线服务请使用
        [AsyncLLMEngine][vllm.AsyncLLMEngine] 类。
    """

    # __init__: LLM 构造函数，初始化所有配置并创建推理引擎
    def __init__(
        self,
        model: str,  # 模型名称或路径（必填位置参数）
        *,  # 以下均为关键字参数
        runner: RunnerOption = "auto",  # 运行器选项，默认自动选择
        convert: ConvertOption = "auto",  # 模型转换选项，默认自动
        tokenizer: str | None = None,  # 分词器名称或路径，默认使用模型自带
        tokenizer_mode: TokenizerMode | str = "auto",  # 分词器模式，默认自动
        skip_tokenizer_init: bool = False,  # 是否跳过分词器初始化，默认不跳过
        trust_remote_code: bool = False,  # 是否信任远程代码，默认不信任
        allowed_local_media_path: str = "",  # 允许读取的本地媒体路径，默认为空（不允许）
        allowed_media_domains: list[str] | None = None,  # 允许的媒体域名列表，默认不限制
        tensor_parallel_size: int = 1,  # 张量并行 GPU 数量，默认 1（单 GPU）
        dtype: ModelDType = "auto",  # 模型数据类型，默认自动检测
        quantization: QuantizationMethods | None = None,  # 量化方法，默认不量化
        revision: str | None = None,  # 模型版本（分支/标签/提交 ID），默认使用最新版
        tokenizer_revision: str | None = None,  # 分词器版本，默认使用最新版
        chat_template: Path | str | None = None,  # 聊天模板路径或字符串，默认为 None
        seed: int = 0,  # 随机数种子，默认为 0
        gpu_memory_utilization: float = 0.9,  # GPU 内存利用率，默认 0.9（90%）
        cpu_offload_gb: float = 0,  # CPU 卸载内存大小（GiB），默认 0（不卸载）
        offload_group_size: int = 0,  # 预取卸载分组大小，默认 0（禁用）
        offload_num_in_group: int = 1,  # 每组卸载层数，默认 1
        offload_prefetch_step: int = 1,  # 预取步长，默认 1
        offload_params: set[str] | None = None,  # 选择性卸载的参数名片段集合，默认为 None
        enforce_eager: bool = False,  # 是否强制 eager 模式，默认 False
        enable_return_routed_experts: bool = False,  # 是否返回路由专家，默认 False
        disable_custom_all_reduce: bool = False,  # 是否禁用自定义 all-reduce，默认 False
        hf_token: bool | str | None = None,  # HuggingFace 访问令牌，默认为 None
        hf_overrides: HfOverrides | None = None,  # HuggingFace 配置覆盖，默认为 None
        mm_processor_kwargs: dict[str, Any] | None = None,  # 多模态处理器参数，默认为 None
        pooler_config: PoolerConfig | None = None,  # 池化器配置，默认为 None
        structured_outputs_config: dict[str, Any]  # 结构化输出配置（字典形式）
        | StructuredOutputsConfig  # 或 StructuredOutputsConfig 实例
        | None = None,  # 默认为 None
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,  # 性能分析器配置，默认为 None
        attention_config: dict[str, Any] | AttentionConfig | None = None,  # 注意力配置，默认为 None
        kv_cache_memory_bytes: int | None = None,  # KV 缓存内存字节数，默认为 None（自动推断）
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,  # 编译配置，默认为 None
        logits_processors: list[str | type[LogitsProcessor]] | None = None,  # logits 处理器列表，默认为 None
        **kwargs: Any,  # 其他传递给 EngineArgs 的关键字参数
    ) -> None:
        """LLM constructor."""

        if "swap_space" in kwargs:  # 检查是否传入了已废弃的 swap_space 参数
            kwargs.pop("swap_space")  # 从 kwargs 中移除 swap_space 参数
            import warnings  # 导入警告模块（延迟导入以减少启动开销）

            warnings.warn(  # 发出弃用警告
                "The 'swap_space' parameter is deprecated and ignored. "
                "It will be removed in a future version.",
                DeprecationWarning,  # 使用弃用警告类型
                stacklevel=2,  # 警告显示在调用方的代码位置
            )

        if "disable_log_stats" not in kwargs:  # 若未指定 disable_log_stats 参数
            kwargs["disable_log_stats"] = True  # 默认禁用统计日志以减少输出噪音

        if "worker_cls" in kwargs:  # 若传入了自定义 worker 类
            worker_cls = kwargs["worker_cls"]  # 获取 worker 类
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            # 如果 worker_cls 不是限定字符串名称，使用 cloudpickle 序列化以避免 pickle 问题
            if isinstance(worker_cls, type):  # 若 worker_cls 是一个类对象（而非字符串）
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)  # 用 cloudpickle 序列化该类

        if "kv_transfer_config" in kwargs and isinstance(  # 若传入了 kv_transfer_config 且为字典类型
            kwargs["kv_transfer_config"], dict
        ):
            from vllm.config.kv_transfer import KVTransferConfig  # 延迟导入 KV 传输配置类

            raw_config_dict = kwargs["kv_transfer_config"]  # 获取原始配置字典
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)  # 将字典转换为 KVTransferConfig 实例
            except ValidationError as e:  # 捕获 pydantic 数据验证错误
                logger.error(  # 记录错误日志
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,  # 记录原始配置字典内容
                    e,  # 记录具体错误信息
                )
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                # 考虑重新抛出更具体的 vLLM 错误或 ValueError 以向用户提供更好的上下文
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e  # 抛出包含原因的 ValueError

        if hf_overrides is None:  # 若未提供 HuggingFace 配置覆盖
            hf_overrides = {}  # 初始化为空字典

        def _make_config(value: Any, cls: type[_R]) -> _R:
            """Convert dict/None/instance to a config instance."""
            # 辅助函数：将字典、None 或实例统一转换为指定配置类的实例
            if value is None:  # 若值为 None
                return cls()  # 返回配置类的默认实例
            if isinstance(value, dict):  # 若值为字典
                return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})  # type: ignore[arg-type]
                # 仅传入该配置类支持的初始化字段，过滤掉不支持的键
            return value  # 若已经是配置实例，直接返回

        if isinstance(compilation_config, int):  # 若编译配置为整数（表示编译模式）
            compilation_config_instance = CompilationConfig(  # 将整数转换为编译配置实例
                mode=CompilationMode(compilation_config)  # 将整数转换为 CompilationMode 枚举
            )
        else:  # 若编译配置为字典、CompilationConfig 实例或 None
            compilation_config_instance = _make_config(  # 使用辅助函数统一转换
                compilation_config, CompilationConfig
            )

        structured_outputs_instance = _make_config(  # 将结构化输出配置转换为 StructuredOutputsConfig 实例
            structured_outputs_config, StructuredOutputsConfig
        )
        profiler_config_instance = _make_config(profiler_config, ProfilerConfig)  # 将性能分析器配置转换为 ProfilerConfig 实例
        attention_config_instance = _make_config(attention_config, AttentionConfig)  # 将注意力配置转换为 AttentionConfig 实例

        # warn about single-process data parallel usage.
        # 警告单进程数据并行用法
        _dp_size = int(kwargs.get("data_parallel_size", 1))  # 获取数据并行大小，默认为 1
        _distributed_executor_backend = kwargs.get("distributed_executor_backend")  # 获取分布式执行后端类型
        if (
            _dp_size > 1  # 若数据并行大小大于 1
            and not _distributed_executor_backend == "external_launcher"  # 且未使用外部启动器
            and not current_platform.is_tpu()  # 且当前平台不是 TPU
        ):
            raise ValueError(  # 抛出错误：单进程不支持数据并行
                f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
                "process usage and may hang. Please use "
                "the explicit multi-process data-parallel example at "
                "'examples/offline_inference/data_parallel.py'."
            )

        engine_args = EngineArgs(  # 创建引擎参数对象，汇总所有配置
            model=model,  # 模型名称或路径
            runner=runner,  # 运行器选项
            convert=convert,  # 模型转换选项
            tokenizer=tokenizer,  # 分词器名称或路径
            tokenizer_mode=tokenizer_mode,  # 分词器模式
            skip_tokenizer_init=skip_tokenizer_init,  # 是否跳过分词器初始化
            trust_remote_code=trust_remote_code,  # 是否信任远程代码
            allowed_local_media_path=allowed_local_media_path,  # 允许的本地媒体路径
            allowed_media_domains=allowed_media_domains,  # 允许的媒体域名
            tensor_parallel_size=tensor_parallel_size,  # 张量并行大小
            dtype=dtype,  # 模型数据类型
            quantization=quantization,  # 量化方法
            revision=revision,  # 模型版本
            tokenizer_revision=tokenizer_revision,  # 分词器版本
            seed=seed,  # 随机种子
            gpu_memory_utilization=gpu_memory_utilization,  # GPU 内存利用率
            kv_cache_memory_bytes=kv_cache_memory_bytes,  # KV 缓存内存字节数
            cpu_offload_gb=cpu_offload_gb,  # CPU 卸载内存大小
            offload_group_size=offload_group_size,  # 预取卸载分组大小
            offload_num_in_group=offload_num_in_group,  # 每组卸载层数
            offload_prefetch_step=offload_prefetch_step,  # 预取步长
            offload_params=offload_params or set(),  # 选择性卸载参数名片段集合（None 时转为空集）
            enforce_eager=enforce_eager,  # 是否强制 eager 模式
            enable_return_routed_experts=enable_return_routed_experts,  # 是否返回路由专家
            disable_custom_all_reduce=disable_custom_all_reduce,  # 是否禁用自定义 all-reduce
            hf_token=hf_token,  # HuggingFace 访问令牌
            hf_overrides=hf_overrides,  # HuggingFace 配置覆盖
            mm_processor_kwargs=mm_processor_kwargs,  # 多模态处理器参数
            pooler_config=pooler_config,  # 池化器配置
            structured_outputs_config=structured_outputs_instance,  # 结构化输出配置实例
            profiler_config=profiler_config_instance,  # 性能分析器配置实例
            attention_config=attention_config_instance,  # 注意力配置实例
            compilation_config=compilation_config_instance,  # 编译配置实例
            logits_processors=logits_processors,  # logits 处理器列表
            **kwargs,  # 其他额外参数
        )

        log_non_default_args(engine_args)  # 记录所有非默认的引擎参数，便于调试

        # 从引擎参数创建 LLMEngine实例。
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS  # 指定使用上下文为 LLM 类
        )

        # 保存引擎类型，供后续判断使用
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()  # 初始化请求计数器，用于生成唯一请求 ID
        self.default_sampling_params: dict[str, Any] | None = None  # 初始化默认采样参数缓存为 None

        supported_tasks = self.llm_engine.get_supported_tasks()  # 获取当前模型支持的任务列表
        logger.info("Supported tasks: %s", supported_tasks)  # 记录支持的任务信息
        self.supported_tasks = supported_tasks  # 保存支持的任务列表

        self.model_config = self.llm_engine.model_config  # 从引擎获取模型配置并保存
        self.renderer = self.llm_engine.renderer  # 从引擎获取渲染器并保存
        self.chat_template = load_chat_template(chat_template)  # 加载聊天模板（从路径或字符串）
        self.io_processor = self.llm_engine.io_processor  # 从引擎获取 IO 处理器并保存
        self.input_processor = self.llm_engine.input_processor  # 从引擎获取输入处理器并保存
        self.chat_template_config = ChatTemplateConfig(chat_template=self.chat_template)  # 创建聊天模板配置对象
        self.pooling_io_processors = init_pooling_io_processors(  # 初始化池化 IO 处理器
            supported_tasks=supported_tasks,  # 传入支持的任务列表
            model_config=self.model_config,  # 传入模型配置
            renderer=self.renderer,  # 传入渲染器
            chat_template_config=self.chat_template_config,  # 传入聊天模板配置
        )
        # Cache for __repr__ to avoid repeated collective_rpc calls
        # 缓存 __repr__ 的结果，避免重复的集合 RPC 调用
        self._cached_repr: str | None = None  # 初始化 __repr__ 缓存为 None

    # get_tokenizer: 获取当前使用的分词器实例
    def get_tokenizer(self) -> TokenizerLike:
        return self.llm_engine.get_tokenizer()  # 委托给 LLM 引擎的 get_tokenizer 方法

    # get_world_size: 获取分布式环境的总进程数（世界大小）
    def get_world_size(self, include_dp: bool = True) -> int:
        """Get the world size from the parallel config.

        Args:
            include_dp: If True (default), returns the world size including
                data parallelism (TP * PP * DP). If False, returns the world
                size without data parallelism (TP * PP).

        Returns:
            The world size (tensor_parallel_size * pipeline_parallel_size),
            optionally multiplied by data_parallel_size if include_dp is True.
        """
        parallel_config = self.llm_engine.vllm_config.parallel_config  # 从 vllm 配置中获取并行配置
        if include_dp:  # 若需要包含数据并行维度
            return parallel_config.world_size_across_dp  # 返回包含 DP 的总世界大小
        return parallel_config.world_size  # 返回不含 DP 的世界大小（TP × PP）

    # reset_mm_cache: 重置多模态缓存（同时清除渲染器和引擎中的缓存）
    def reset_mm_cache(self) -> None:
        self.renderer.clear_mm_cache()  # 清除渲染器中的多模态缓存
        self.llm_engine.reset_mm_cache()  # 清除 LLM 引擎中的多模态缓存

    # get_default_sampling_params: 获取默认采样参数，优先从模型配置中读取
    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:  # 若尚未缓存默认采样参数
            self.default_sampling_params = self.model_config.get_diff_sampling_param()  # 从模型配置获取差异采样参数
        if self.default_sampling_params:  # 若存在非空的差异采样参数
            return SamplingParams.from_optional(**self.default_sampling_params)  # 使用差异参数创建 SamplingParams 实例
        return SamplingParams()  # 否则返回全局默认 SamplingParams 实例

    # generate: 为输入提示词生成补全文本（核心生成接口）
    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],  # 单个提示词或提示词序列
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,  # 采样参数，默认为 None（使用默认值）
        *,  # 以下均为关键字参数
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条，默认为 True
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,  # LoRA 请求，默认为 None
        priority: list[int] | None = None,  # 请求优先级列表，默认为 None
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数，默认为 None
    ) -> list[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.
                If provided, must be a list of integers matching the length
                of `prompts`, where each priority value corresponds to the prompt
                at the same index.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.
        """
        runner_type = self.model_config.runner_type  # 获取模型配置中的运行器类型
        if runner_type != "generate":  # 若运行器类型不是生成模式
            raise ValueError(  # 抛出错误：generate() 仅支持生成式模型
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:  # 若未提供采样参数
            sampling_params = self.get_default_sampling_params()  # 使用默认采样参数

        return self._run_completion(  # 调用内部补全运行方法
            prompts=prompts,  # 传入提示词
            params=sampling_params,  # 传入采样参数
            output_type=RequestOutput,  # 指定输出类型为 RequestOutput
            use_tqdm=use_tqdm,  # 传入进度条设置
            lora_request=lora_request,  # 传入 LoRA 请求
            tokenization_kwargs=tokenization_kwargs,  # 传入分词器参数
            priority=priority,  # 传入请求优先级
        )

    # enqueue: 将提示词加入生成队列但不等待完成（异步入队接口）
    def enqueue(
        self,
        prompts: PromptType | Sequence[PromptType],  # 单个提示词或提示词序列
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,  # 采样参数
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        priority: list[int] | None = None,  # 请求优先级
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
    ) -> list[str]:
        """Enqueue prompts for generation without waiting for completion.

        This method adds requests to the engine queue but does not start
        processing them. Use wait_for_completion() to process the queued
        requests and get results.

        Args:
            prompts: The prompts to the LLM. See generate() for details.
            sampling_params: The sampling parameters for text generation.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
            use_tqdm: If True, shows a tqdm progress bar while adding requests.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of request IDs for the enqueued requests.
        """
        runner_type = self.model_config.runner_type  # 获取运行器类型
        if runner_type != "generate":  # 若不是生成模式则报错
            raise ValueError("LLM.enqueue() is only supported for generative models.")

        if sampling_params is None:  # 若未提供采样参数
            sampling_params = self.get_default_sampling_params()  # 使用默认采样参数

        return self._add_completion_requests(  # 调用内部方法添加补全请求到队列
            prompts=prompts,  # 传入提示词
            params=sampling_params,  # 传入采样参数
            use_tqdm=use_tqdm,  # 传入进度条设置
            lora_request=lora_request,  # 传入 LoRA 请求
            priority=priority,  # 传入优先级
            tokenization_kwargs=tokenization_kwargs,  # 传入分词器参数
        )

    @overload  # 函数重载：无 output_type 参数时的类型签名
    def wait_for_completion(
        self,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[RequestOutput | PoolingRequestOutput]: ...

    @overload  # 函数重载：指定 output_type 参数时的类型签名
    def wait_for_completion(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]: ...

    # wait_for_completion: 等待所有已入队的请求完成并返回结果
    def wait_for_completion(
        self,
        output_type: type[Any] | tuple[type[Any], ...] | None = None,  # 期望的输出类型
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
    ) -> list[Any]:
        """Wait for all enqueued requests to complete and return results.

        This method processes all requests currently in the engine queue
        and returns their outputs. Use after enqueue() to get results.

        Args:
            output_type: The expected output type, defaults to RequestOutput.
            use_tqdm: If True, shows a tqdm progress bar.

        Returns:
            A list of output objects for all completed requests.
        """
        if output_type is None:  # 若未指定输出类型
            output_type = (RequestOutput, PoolingRequestOutput)  # 默认接受两种输出类型

        return self._run_engine(output_type, use_tqdm=use_tqdm)  # 运行引擎处理所有请求

    # _resolve_mm_lora: 解析多模态 LoRA 请求，为多模态提示自动匹配默认的模态 LoRA
    def _resolve_mm_lora(
        self,
        prompt: ProcessorInputs,  # 处理器输入
        lora_request: LoRARequest | None,  # 用户显式提供的 LoRA 请求
    ) -> LoRARequest | None:
        if prompt["type"] != "multimodal":  # 若提示不是多模态类型
            return lora_request  # 直接返回用户提供的 LoRA 请求

        lora_config = self.llm_engine.vllm_config.lora_config  # 获取 LoRA 配置
        default_mm_loras = None if lora_config is None else lora_config.default_mm_loras  # 获取默认多模态 LoRA 映射
        if not default_mm_loras:  # 若没有配置默认多模态 LoRA
            return lora_request  # 直接返回

        prompt_modalities = prompt["mm_placeholders"].keys()  # 获取提示中包含的模态类型
        intersection = set(prompt_modalities).intersection(default_mm_loras.keys())  # 找到提示模态与已注册 LoRA 模态的交集
        if not intersection:  # 若没有交集
            return lora_request  # 直接返回

        if len(intersection) > 1:  # 若有多个模态匹配到 LoRA
            # TODO: Would be nice to be able to have multiple loras per prompt
            # TODO: 未来可能支持每个提示多个 LoRA
            logger.warning(  # 警告：当前仅支持每个请求一个 LoRA
                "Multiple modality specific loras were registered and would be "
                "used by a single prompt consuming several modalities; "
                "currently we only support one lora per request; as such, "
                "lora(s) registered with modalities: %s will be skipped",
                intersection,
            )
            return lora_request  # 返回用户提供的 LoRA 请求

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        # 构建 LoRA 请求；默认多模态 LoRA 的 ID 为模态名按字母排序的索引 + 1
        modality_name = intersection.pop()  # 获取匹配的模态名称
        modality_lora_path = default_mm_loras[modality_name]  # 获取该模态对应的 LoRA 路径
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1  # 计算 LoRA ID

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        # 若存在冲突，发出警告，但始终使用用户显式提供的请求
        if lora_request:  # 若用户已提供 LoRA 请求
            if lora_request.lora_int_id != modality_lora_id:  # 若用户提供的 LoRA ID 与模态默认 ID 不同
                logger.warning(  # 警告：存在 LoRA ID 冲突，回退到用户提供的请求
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request  # 返回用户提供的 LoRA 请求

        return LoRARequest(  # 创建并返回基于模态的默认 LoRA 请求
            modality_name,  # 模态名称作为 LoRA 名称
            modality_lora_id,  # LoRA ID
            modality_lora_path,  # LoRA 权重路径
        )

    # collective_rpc: 在所有 worker 上执行 RPC 调用
    def collective_rpc(
        self,
        method: str | Callable[..., _R],  # 要执行的 worker 方法名或可调用对象
        timeout: float | None = None,  # 最大等待时间（秒），None 表示无限等待
        args: tuple = (),  # 传递给 worker 方法的位置参数
        kwargs: dict[str, Any] | None = None,  # 传递给 worker 方法的关键字参数
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """

        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)  # 委托给 LLM 引擎执行集合 RPC

    # apply_model: 在每个 worker 的模型上直接运行自定义函数
    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.

        !!! warning
            To reduce the overhead of data transfer, avoid returning large
            arrays or tensors from this method. If you must return them,
            make sure you move them to CPU first to avoid taking up additional
            VRAM!
        """
        return self.llm_engine.apply_model(func)  # 委托给 LLM 引擎在所有 worker 上应用函数

    # beam_search: 使用束搜索生成序列
    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],  # 提示词列表，每个可以是文本或 token ID 列表
        params: BeamSearchParams,  # 束搜索参数
        lora_request: list[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        use_tqdm: bool = False,  # 是否显示进度条
        concurrency_limit: int | None = None,  # 最大并发请求数，None 表示不限制
    ) -> list[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        # TODO: 束搜索如何与长度惩罚、频率惩罚和停止条件等协同工作？
        beam_width = params.beam_width  # 获取束宽度（同时保留的候选序列数量）
        max_tokens = params.max_tokens  # 获取最大生成 token 数
        temperature = params.temperature  # 获取温度参数
        ignore_eos = params.ignore_eos  # 是否忽略结束符
        length_penalty = params.length_penalty  # 获取长度惩罚系数

        tokenizer = self.renderer.get_tokenizer()  # 获取分词器
        eos_token_id = tokenizer.eos_token_id  # 获取结束符 token ID
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)  # 创建束排序键函数

        engine_prompts = self._preprocess_cmpl(prompts)  # 预处理提示词为引擎格式
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_prompts))  # 将 LoRA 请求展开为序列

        if use_tqdm and concurrency_limit is not None:  # 若同时使用进度条和并发限制
            logger.warning(  # 警告：使用并发限制时不支持进度条
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar."
            )
            use_tqdm = False  # 禁用进度条

        if concurrency_limit is None:  # 若未设置并发限制
            concurrency_limit = len(engine_prompts)  # 并发限制设为提示词总数（即不限制）

        # generate 2 * beam_width candidates at each step
        # 每步生成 2 * beam_width 个候选
        # following the huggingface transformers implementation
        # 遵循 HuggingFace Transformers 的实现
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        sampling_params = SamplingParams(  # 创建束搜索专用的采样参数
            logprobs=2 * beam_width,  # 返回 2 倍束宽度的 log 概率
            max_tokens=1,  # 每步仅生成 1 个 token
            temperature=temperature,  # 温度参数
            skip_clone=True,  # Internal beam search, safe to skip clone  # 内部束搜索，安全跳过参数克隆
        )
        instances: list[BeamSearchInstance] = []  # 初始化束搜索实例列表

        for lora_req, prompt in zip(lora_requests, engine_prompts):  # 遍历每个 LoRA 请求和对应的引擎提示
            if prompt["type"] == "embeds":  # 若提示类型是嵌入向量
                raise NotImplementedError(  # 抛出未实现错误：束搜索不支持嵌入提示
                    "Embedding prompt not supported for beam search"
                )

            instances.append(  # 创建束搜索实例并添加到列表
                BeamSearchInstance(
                    prompt,  # 提示词
                    lora_request=lora_req,  # LoRA 请求
                    logprobs=None,  # 初始无 log 概率
                ),
            )

        for prompt_start in range(0, len(instances), concurrency_limit):  # 按并发限制分批处理
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]  # 获取当前批次的实例

            token_iter = range(max_tokens)  # 创建 token 迭代范围
            if use_tqdm:  # 若启用进度条
                token_iter = tqdm(  # 用 tqdm 包装迭代器
                    token_iter, desc="Beam search", unit="token", unit_scale=False
                )
                logger.warning(  # 警告：进度条显示的是 token 步数上限，可能提前结束
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress."
                )
            for _ in token_iter:  # 逐 token 迭代
                all_beams: list[BeamSearchSequence] = list(  # 收集当前批次所有实例的所有束
                    sum((instance.beams for instance in instances_batch), [])
                )
                pos = [0] + list(  # 计算每个实例的束在 all_beams 中的起始位置
                    itertools.accumulate(
                        len(instance.beams) for instance in instances_batch
                    )
                )
                instance_start_and_end: list[tuple[int, int]] = list(  # 计算每个实例的束起止索引
                    zip(pos[:-1], pos[1:])
                )

                if len(all_beams) == 0:  # 若没有剩余的束（所有束已完成）
                    break  # 退出 token 迭代循环

                # only runs for one step
                # 仅运行一步（生成一个 token）
                # we don't need to use tqdm here
                # 此处不需要使用 tqdm
                output = self._render_and_run_requests(  # 渲染提示并运行引擎获取输出
                    prompts=(beam.get_prompt() for beam in all_beams),  # 为每个束获取提示
                    params=self._params_to_seq(sampling_params, len(all_beams)),  # 将采样参数展开为序列
                    output_type=RequestOutput,  # 指定输出类型
                    lora_requests=[beam.lora_request for beam in all_beams],  # 每个束的 LoRA 请求
                    use_tqdm=False,  # 此处不显示进度条
                )

                for (start, end), instance in zip(  # 遍历每个实例的束范围
                    instance_start_and_end, instances_batch
                ):
                    instance_new_beams = []  # 该实例的新束列表
                    for i in range(start, end):  # 遍历该实例的每个束
                        current_beam = all_beams[i]  # 获取当前束
                        result = output[i]  # 获取当前束的输出结果

                        if result.outputs[0].logprobs is not None:  # 若输出包含 log 概率
                            # if `result.outputs[0].logprobs` is None, it means
                            # the sequence is completed because of the
                            # max-model-len or abortion. we don't need to add
                            # it to the new beams.
                            # 若 logprobs 为 None，表示序列因达到最大模型长度或被中止而完成，
                            # 不需要将其添加到新束中
                            logprobs = result.outputs[0].logprobs[0]  # 获取第一步的 log 概率分布
                            for token_id, logprob_obj in logprobs.items():  # 遍历每个候选 token
                                new_beam = BeamSearchSequence(  # 创建新的束搜索序列
                                    current_beam.orig_prompt,  # 原始提示
                                    tokens=current_beam.tokens + [token_id],  # 在当前序列后追加新 token
                                    logprobs=current_beam.logprobs + [logprobs],  # 追加 log 概率
                                    lora_request=current_beam.lora_request,  # 保持 LoRA 请求
                                    cum_logprob=current_beam.cum_logprob  # 累积 log 概率
                                    + logprob_obj.logprob,  # 加上新 token 的 log 概率
                                )

                                if token_id == eos_token_id and not ignore_eos:  # 若新 token 是结束符且不忽略
                                    instance.completed.append(new_beam)  # 将该束标记为已完成
                                else:  # 否则
                                    instance_new_beams.append(new_beam)  # 添加到新束候选列表
                    sorted_beams = sorted(  # 按排序键降序排列新束
                        instance_new_beams, key=sort_beams_key, reverse=True
                    )
                    instance.beams = sorted_beams[:beam_width]  # 仅保留前 beam_width 个束

        outputs = []  # 初始化最终输出列表
        for instance in instances:  # 遍历每个束搜索实例
            instance.completed.extend(instance.beams)  # 将未完成的束也加入已完成列表
            sorted_completed = sorted(  # 按排序键降序排列所有已完成的束
                instance.completed, key=sort_beams_key, reverse=True
            )
            best_beams = sorted_completed[:beam_width]  # 选取最佳的 beam_width 个束

            for beam in best_beams:  # 遍历最佳束
                beam.text = tokenizer.decode(beam.tokens)  # 将 token ID 解码为文本

            outputs.append(BeamSearchOutput(sequences=best_beams))  # 创建束搜索输出并添加到列表

        return outputs  # 返回所有束搜索输出

    # _preprocess_cmpl: 将 LLM API 的提示词输入转换为引擎可接受的处理器输入格式
    def _preprocess_cmpl(
        self,
        prompts: Sequence[PromptType],  # 提示词序列
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
    ) -> Sequence[ProcessorInputs]:
        """
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `ProcessorInputs` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer  # 获取渲染器
        model_config = self.model_config  # 获取模型配置

        parsed_prompts = [  # 解析每个提示词为标准格式
            parse_model_prompt(model_config, prompt) for prompt in prompts
        ]
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(  # 获取默认补全分词参数并合并额外参数
            **(tokenization_kwargs or {})
        )

        return renderer.render_cmpl(parsed_prompts, tok_params)  # 渲染补全提示并返回处理器输入

    # _preprocess_cmpl_one: 预处理单个提示词（_preprocess_cmpl 的单元素便捷版本）
    def _preprocess_cmpl_one(
        self,
        prompt: PromptType,  # 单个提示词
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
    ) -> ProcessorInputs:
        (engine_prompt,) = self._preprocess_cmpl([prompt], tokenization_kwargs)  # 处理单元素列表并解包
        return engine_prompt  # 返回处理后的单个引擎提示

    # _preprocess_chat: 将对话列表转换为引擎可接受的处理器输入格式
    def _preprocess_chat(
        self,
        conversations: Sequence[list[ChatCompletionMessageParam]],  # 对话列表（每个对话是消息列表）
        chat_template: str | None = None,  # 聊天模板
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",  # 内容格式选项
        chat_template_kwargs: dict[str, Any] | None = None,  # 聊天模板额外参数
        add_generation_prompt: bool = True,  # 是否添加生成提示
        continue_final_message: bool = False,  # 是否继续最后一条消息
        tools: list[dict[str, Any]] | None = None,  # 工具定义列表
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        mm_processor_kwargs: dict[str, Any] | None = None,  # 多模态处理器参数
    ) -> Sequence[ProcessorInputs]:
        """
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `ProcessorInputs` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer  # 获取渲染器

        chat_params = ChatParams(  # 创建聊天参数对象
            chat_template=chat_template,  # 聊天模板
            chat_template_content_format=chat_template_content_format,  # 内容格式
            chat_template_kwargs=merge_kwargs(  # 合并聊天模板参数
                chat_template_kwargs,
                dict(
                    add_generation_prompt=add_generation_prompt,  # 是否添加生成提示
                    continue_final_message=continue_final_message,  # 是否继续最后消息
                    tools=tools,  # 工具定义
                    tokenize=is_mistral_tokenizer(renderer.tokenizer),  # 是否为 Mistral 分词器
                ),
            ),
        )
        tok_params = renderer.default_chat_tok_params.with_kwargs(  # 获取默认聊天分词参数并合并
            **(tokenization_kwargs or {})
        )

        _, engine_prompts = renderer.render_chat(  # 渲染聊天对话为引擎提示
            conversations,  # 对话列表
            chat_params,  # 聊天参数
            tok_params,  # 分词参数
            prompt_extras={"mm_processor_kwargs": mm_processor_kwargs},  # 多模态处理器额外参数
        )

        return engine_prompts  # 返回处理后的引擎提示列表

    # _preprocess_chat_one: 预处理单个对话（_preprocess_chat 的单元素便捷版本）
    def _preprocess_chat_one(
        self,
        conversation: list[ChatCompletionMessageParam],  # 单个对话
        chat_template: str | None = None,  # 聊天模板
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",  # 内容格式
        chat_template_kwargs: dict[str, Any] | None = None,  # 模板额外参数
        add_generation_prompt: bool = True,  # 是否添加生成提示
        continue_final_message: bool = False,  # 是否继续最后消息
        tools: list[dict[str, Any]] | None = None,  # 工具定义
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
        mm_processor_kwargs: dict[str, Any] | None = None,  # 多模态处理器参数
    ) -> ProcessorInputs:
        (engine_prompt,) = self._preprocess_chat(  # 处理单个对话并解包
            [conversation],
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return engine_prompt  # 返回处理后的单个引擎提示

    # chat: 为聊天对话生成回复（聊天接口）
    def chat(
        self,
        messages: list[ChatCompletionMessageParam]  # 单个对话或对话序列
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,  # 采样参数
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        chat_template: str | None = None,  # 聊天模板
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",  # 内容格式
        add_generation_prompt: bool = True,  # 是否添加生成提示
        continue_final_message: bool = False,  # 是否继续最后消息
        tools: list[dict[str, Any]] | None = None,  # 工具定义
        chat_template_kwargs: dict[str, Any] | None = None,  # 模板额外参数
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
        mm_processor_kwargs: dict[str, Any] | None = None,  # 多模态处理器参数
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A sequence of conversations or a single conversation.

                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                  Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                  similar to OpenAI schema.
                  Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """
        model_config = self.model_config  # 获取模型配置
        runner_type = model_config.runner_type  # 获取运行器类型
        if runner_type != "generate":  # 若不是生成模式
            raise ValueError(  # 抛出错误：chat() 仅支持生成式模型
                "LLM.chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:  # 若未提供采样参数
            sampling_params = self.get_default_sampling_params()  # 使用默认采样参数

        return self._run_chat(  # 调用内部聊天运行方法
            messages=messages,  # 传入消息
            params=sampling_params,  # 传入采样参数
            output_type=RequestOutput,  # 指定输出类型
            use_tqdm=use_tqdm,  # 进度条设置
            lora_request=lora_request,  # LoRA 请求
            chat_template=chat_template,  # 聊天模板
            chat_template_content_format=chat_template_content_format,  # 内容格式
            chat_template_kwargs=chat_template_kwargs,  # 模板参数
            add_generation_prompt=add_generation_prompt,  # 是否添加生成提示
            continue_final_message=continue_final_message,  # 是否继续最后消息
            tools=tools,  # 工具定义
            tokenization_kwargs=tokenization_kwargs,  # 分词器参数
            mm_processor_kwargs=mm_processor_kwargs,  # 多模态处理器参数
        )

    # encode: 对输入提示词的隐藏状态进行池化操作（编码接口）
    def encode(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,  # 提示词或数据提示
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,  # 池化参数
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: list[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        pooling_task: PoolingTask | None = None,  # 池化任务类型
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
    ) -> list[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_task: Override the pooling task to use.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """

        if pooling_task is None:  # 若未指定池化任务类型
            raise ValueError(  # 抛出错误：池化任务是必须的
                "pooling_task required for `LLM.encode`\n"
                "Please use one of the more specific methods or set the "
                "pooling_task when using `LLM.encode`:\n"
                "  - For embeddings, use `LLM.embed(...)` "
                'or `pooling_task="embed"`.\n'
                "  - For classification logits, use `LLM.classify(...)` "
                'or `pooling_task="classify"`.\n'
                "  - For similarity scores, use `LLM.score(...)`.\n"
                "  - For rewards, use `LLM.reward(...)` "
                'or `pooling_task="token_classify"`\n'
                "  - For token classification, "
                'use `pooling_task="token_classify"`\n'
                '  - For multi-vector retrieval, use `pooling_task="token_embed"`'
            )

        model_config = self.model_config  # 获取模型配置
        runner_type = model_config.runner_type  # 获取运行器类型
        if runner_type != "pooling":  # 若不是池化模式
            raise ValueError(  # 抛出错误：encode() 仅支持池化模型
                "LLM.encode() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        if isinstance(prompts, dict) and "data" in prompts:  # 若提示为字典类型且包含 "data" 键（插件数据格式）
            if self.io_processor is None:  # 若未安装 IO 处理器插件
                raise ValueError(  # 抛出错误：未安装 IOProcessor 插件
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details."
                )

            # Validate the request data is valid for the loaded plugin
            # 验证请求数据对已加载的插件是否有效
            prompt_data = prompts.get("data")  # 获取提示中的数据字段
            if prompt_data is None:  # 若数据字段为 None
                raise ValueError(  # 抛出错误：data 字段不能为 None
                    "The 'data' field of the prompt is expected to contain "
                    "the prompt data and it cannot be None. "
                    "Refer to the documentation of the IOProcessor "
                    "in use for more details."
                )
            validated_prompt = self.io_processor.parse_data(prompt_data)  # 使用 IO 处理器解析和验证数据

            # obtain the actual model prompts from the pre-processor
            # 从预处理器获取实际的模型提示
            prompts = self.io_processor.pre_process(prompt=validated_prompt)  # 预处理数据为模型提示
            prompts_seq = prompt_to_seq(prompts)  # 将提示转换为序列

            params_seq: Sequence[PoolingParams] = [  # 为每个提示合并池化参数
                self.io_processor.merge_pooling_params(param)
                for param in self._params_to_seq(
                    pooling_params,
                    len(prompts_seq),
                )
            ]
            for p in params_seq:  # 遍历参数序列
                if p.task is None:  # 若未设置任务类型
                    p.task = "plugin"  # 设置为插件任务

            outputs = self._run_completion(  # 运行补全请求
                prompts=prompts_seq,  # 提示词序列
                params=params_seq,  # 参数序列
                output_type=PoolingRequestOutput,  # 输出类型为池化请求输出
                use_tqdm=use_tqdm,  # 进度条设置
                lora_request=lora_request,  # LoRA 请求
                tokenization_kwargs=tokenization_kwargs,  # 分词器参数
            )

            # get the post-processed model outputs
            # 获取后处理的模型输出
            assert self.io_processor is not None  # 断言 IO 处理器不为 None
            processed_outputs = self.io_processor.post_process(outputs)  # 对输出进行后处理

            return [  # 返回包装为 PoolingRequestOutput 的后处理输出
                PoolingRequestOutput[Any](
                    request_id="",  # 请求 ID（插件模式下为空）
                    outputs=processed_outputs,  # 后处理的输出
                    num_cached_tokens=getattr(  # 缓存的 token 数量
                        processed_outputs, "num_cached_tokens", 0
                    ),
                    prompt_token_ids=[],  # 提示 token ID（插件模式下为空）
                    finished=True,  # 标记为已完成
                )
            ]
        else:  # 若为标准提示格式
            if pooling_params is None:  # 若未提供池化参数
                # Use default pooling params.
                # 使用默认池化参数
                pooling_params = PoolingParams()  # 创建默认池化参数实例

            prompts_seq = prompt_to_seq(prompts)  # 将提示转换为序列
            params_seq = self._params_to_seq(pooling_params, len(prompts_seq))  # 将参数展开为序列

            for param in params_seq:  # 遍历参数序列
                if param.task is None:  # 若未设置任务类型
                    param.task = pooling_task  # 设置为指定的池化任务
                elif param.task != pooling_task:  # 若已设置的任务类型与指定的不同
                    msg = (  # 构造错误消息
                        f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                    )
                    raise ValueError(msg)  # 抛出错误：不能覆盖已设置的任务类型

            if pooling_task in self.pooling_io_processors:  # 若存在该任务的专用 IO 处理器
                io_processor = self.pooling_io_processors[pooling_task]  # 获取对应的 IO 处理器
                processor_inputs = io_processor.pre_process_offline(  # 离线预处理提示词
                    prompts_seq, tokenization_kwargs
                )
                seq_lora_requests = self._lora_request_to_seq(  # 将 LoRA 请求展开为序列
                    lora_request, len(prompts_seq)
                )
                seq_priority = self._priority_to_seq(None, len(prompts))  # 将优先级展开为序列

                self._render_and_add_requests(  # 渲染并添加请求到引擎
                    prompts=processor_inputs,  # 处理后的提示
                    params=params_seq,  # 参数序列
                    lora_requests=seq_lora_requests,  # LoRA 请求序列
                    priorities=seq_priority,  # 优先级序列
                )

                outputs = self._run_engine(  # 运行引擎获取输出
                    use_tqdm=use_tqdm, output_type=PoolingRequestOutput
                )
                outputs = io_processor.post_process_offline(outputs)  # 对输出进行离线后处理
            else:  # 若没有专用 IO 处理器
                outputs = self._run_completion(  # 使用通用补全方法运行
                    prompts=prompts_seq,  # 提示词序列
                    params=params_seq,  # 参数序列
                    output_type=PoolingRequestOutput,  # 输出类型
                    use_tqdm=use_tqdm,  # 进度条设置
                    lora_request=lora_request,  # LoRA 请求
                    tokenization_kwargs=tokenization_kwargs,  # 分词器参数
                )
        return outputs  # 返回输出结果

    # embed: 为每个提示词生成嵌入向量
    def embed(
        self,
        prompts: PromptType | Sequence[PromptType],  # 提示词或提示词序列
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,  # 池化参数
        lora_request: list[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
    ) -> list[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "embed" not in self.supported_tasks:  # 若模型不支持嵌入任务
            raise ValueError(  # 抛出错误
                "Embedding API is not supported by this model. "
                "Try converting the model using `--convert embed`."
            )

        items = self.encode(  # 调用 encode 方法进行编码
            prompts,  # 提示词
            use_tqdm=use_tqdm,  # 进度条设置
            pooling_params=pooling_params,  # 池化参数
            lora_request=lora_request,  # LoRA 请求
            pooling_task="embed",  # 指定池化任务为嵌入
            tokenization_kwargs=tokenization_kwargs,  # 分词器参数
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]  # 将基础输出转换为嵌入输出

    # classify: 为每个提示词生成分类 logits
    def classify(
        self,
        prompts: PromptType | Sequence[PromptType],  # 提示词或提示词序列
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,  # 池化参数
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: list[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
    ) -> list[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "classify" not in self.supported_tasks:  # 若模型不支持分类任务
            raise ValueError(  # 抛出错误
                "Classification API is not supported by this model. "
                "Try converting the model using `--convert classify`."
            )

        items = self.encode(  # 调用 encode 方法进行编码
            prompts,  # 提示词
            use_tqdm=use_tqdm,  # 进度条设置
            pooling_params=pooling_params,  # 池化参数
            lora_request=lora_request,  # LoRA 请求
            pooling_task="classify",  # 指定池化任务为分类
            tokenization_kwargs=tokenization_kwargs,  # 分词器参数
        )

        return [ClassificationRequestOutput.from_base(item) for item in items]  # 将基础输出转换为分类输出

    # reward: 为每个提示词生成奖励分数（用于 RLHF 等场景）
    def reward(
        self,
        prompts: PromptType | Sequence[PromptType],  # 提示词（仅位置参数）
        /,
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,  # 池化参数
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: list[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
    ) -> list[PoolingRequestOutput]:
        """
        Generate rewards for each prompt.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """
        return self.encode(  # 调用 encode 方法，任务类型为 token_classify
            prompts,  # 提示词
            use_tqdm=use_tqdm,  # 进度条设置
            lora_request=lora_request,  # LoRA 请求
            pooling_params=pooling_params,  # 池化参数
            pooling_task="token_classify",  # 指定池化任务为 token 分类（奖励模型使用）
            tokenization_kwargs=tokenization_kwargs,  # 分词器参数
        )

    # _embedding_score: 基于嵌入向量的余弦相似度评分
    def _embedding_score(
        self,
        data_1: list[ScoreData],  # 第一组评分数据
        data_2: list[ScoreData],  # 第二组评分数据
        *,
        use_tqdm: bool | Callable[..., tqdm],  # 是否显示进度条
        pooling_params: PoolingParams | None,  # 池化参数
        lora_request: list[LoRARequest] | LoRARequest | None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any],  # 分词器参数
    ) -> list[ScoringRequestOutput]:
        tokenizer = self.get_tokenizer()  # 获取分词器

        input_texts: list[str] = []  # 初始化输入文本列表
        for text in data_1 + data_2:  # 遍历两组数据
            if not isinstance(text, str):  # 若非字符串类型
                raise NotImplementedError(  # 抛出未实现错误：嵌入评分暂不支持多模态输入
                    "Embedding scores currently do not support multimodal input."
                )
            input_texts.append(text)  # 添加到输入文本列表

        encoded_output = self.encode(  # 对所有输入文本进行编码
            input_texts,  # 输入文本
            use_tqdm=use_tqdm,  # 进度条设置
            lora_request=lora_request,  # LoRA 请求
            pooling_params=pooling_params,  # 池化参数
            pooling_task="embed",  # 嵌入任务
            tokenization_kwargs=tokenization_kwargs,  # 分词器参数
        )

        encoded_output_1 = encoded_output[0 : len(data_1)]  # 分割第一组的编码输出
        encoded_output_2 = encoded_output[len(data_1) :]  # 分割第二组的编码输出

        if len(encoded_output_1) == 1:  # 若第一组只有一个元素（1 对 N 的情况）
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)  # 复制以匹配第二组的长度

        scores = _cosine_similarity(  # 计算余弦相似度
            tokenizer=tokenizer,  # 分词器
            embed_1=encoded_output_1,  # 第一组嵌入
            embed_2=encoded_output_2,  # 第二组嵌入
        )

        return [ScoringRequestOutput.from_base(item) for item in scores]  # 将结果转换为评分输出

    # _late_interaction_score: 延迟交互评分（ColBERT MaxSim 方法）
    def _late_interaction_score(
        self,
        data_1: list[ScoreData],  # 第一组评分数据（查询）
        data_2: list[ScoreData],  # 第二组评分数据（文档）
        *,
        use_tqdm: bool | Callable[..., tqdm],  # 是否显示进度条
        pooling_params: PoolingParams | None,  # 池化参数
        lora_request: list[LoRARequest] | LoRARequest | None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any],  # 分词器参数
    ) -> list[ScoringRequestOutput]:
        """
        Late interaction scoring (ColBERT MaxSim).

        Encodes queries and documents into per-token embeddings, then computes
        MaxSim: sum over query tokens of max similarity to any document token.
        """
        from vllm.outputs import PoolingOutput  # 延迟导入池化输出类

        tokenizer = self.get_tokenizer()  # 获取分词器

        # Convert ScoreData to PromptType (handles both text and multimodal)
        # 将评分数据转换为提示类型（处理文本和多模态）
        model_config = self.model_config  # 获取模型配置
        prompts_1 = score_data_to_prompts(data_1, "query", model_config)  # 将第一组数据转换为查询提示
        prompts_2 = score_data_to_prompts(data_2, "document", model_config)  # 将第二组数据转换为文档提示

        encoded_output: list[PoolingRequestOutput] = self.encode(  # 对所有提示进行 token 级别编码
            prompts_1 + prompts_2,  # 合并两组提示
            use_tqdm=use_tqdm,  # 进度条设置
            lora_request=lora_request,  # LoRA 请求
            pooling_params=pooling_params,  # 池化参数
            pooling_task="token_embed",  # 指定为 token 级嵌入任务
            tokenization_kwargs=tokenization_kwargs,  # 分词器参数
        )

        encoded_output_1: list[PoolingRequestOutput] = encoded_output[: len(prompts_1)]  # 分割查询编码输出
        encoded_output_2: list[PoolingRequestOutput] = encoded_output[len(prompts_1) :]  # 分割文档编码输出

        if len(encoded_output_1) == 1:  # 若查询只有一个（1 对 N 的情况）
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)  # 复制以匹配文档数量

        # Compute MaxSim scores
        # 计算 MaxSim 评分
        scores: list[PoolingRequestOutput] = []  # 初始化评分结果列表
        padding: list[int] = []  # 初始化填充 token 列表
        if (pad_token_id := tokenizer.pad_token_id) is not None:  # 若存在填充 token ID
            padding = [pad_token_id]  # 设置填充列表

        for emb_1, emb_2 in zip(encoded_output_1, encoded_output_2):  # 遍历每对查询-文档编码
            # emb_1.outputs.data: [query_len, dim]
            # emb_1.outputs.data: [查询长度, 嵌入维度]
            # emb_2.outputs.data: [doc_len, dim]
            # emb_2.outputs.data: [文档长度, 嵌入维度]
            q_emb = emb_1.outputs.data  # 获取查询嵌入矩阵
            d_emb = emb_2.outputs.data  # 获取文档嵌入矩阵

            maxsim_score = compute_maxsim_score(q_emb, d_emb)  # 计算 MaxSim 评分

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids  # 拼接 token ID（查询 + 填充 + 文档）

            scores.append(  # 创建评分输出并添加到列表
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",  # 组合请求 ID
                    outputs=PoolingOutput(data=maxsim_score),  # MaxSim 评分数据
                    prompt_token_ids=tokens,  # 合并的 token ID
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,  # 缓存 token 总数
                    finished=True,  # 标记为已完成
                )
            )

        return [ScoringRequestOutput.from_base(item) for item in scores]  # 转换为评分输出并返回

    # _cross_encoding_score: 交叉编码器评分（将两段文本合并后送入模型）
    def _cross_encoding_score(
        self,
        data_1: list[ScoreData],  # 第一组评分数据
        data_2: list[ScoreData],  # 第二组评分数据
        *,
        use_tqdm: bool | Callable[..., tqdm],  # 是否显示进度条
        pooling_params: PoolingParams | None,  # 池化参数
        lora_request: list[LoRARequest] | LoRARequest | None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any],  # 分词器参数
        score_template: str | None,  # 评分模板
    ) -> list[ScoringRequestOutput]:
        model_config = self.model_config  # 获取模型配置
        tokenizer = self.get_tokenizer()  # 获取分词器

        if is_mistral_tokenizer(tokenizer):  # 若为 Mistral 分词器
            raise ValueError("Score API is not supported for Mistral tokenizer")  # 抛出错误：不支持 Mistral 分词器

        if len(data_1) == 1:  # 若第一组只有一个元素（1 对 N）
            data_1 = data_1 * len(data_2)  # 复制以匹配第二组长度

        if pooling_params is None:  # 若未提供池化参数
            pooling_params = PoolingParams(task="score")  # 创建默认评分池化参数
        elif pooling_params.task is None:  # 若参数中未设置任务
            pooling_params.task = "score"  # 设置任务为评分

        pooling_params_list = list[PoolingParams]()  # 初始化池化参数列表

        prompts = list[PromptType]()  # 初始化提示词列表

        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]  # 创建输入对列表

        for q, d in input_pairs:  # 遍历每对输入
            _, engine_prompt = get_score_prompt(  # 获取评分提示
                model_config=model_config,  # 模型配置
                data_1=q,  # 第一段数据
                data_2=d,  # 第二段数据
                tokenizer=tokenizer,  # 分词器
                tokenization_kwargs=tokenization_kwargs,  # 分词器参数
                score_template=score_template,  # 评分模板
            )

            if token_type_ids := engine_prompt.pop("token_type_ids", None):  # 若存在 token 类型 ID
                params = pooling_params.clone()  # 克隆池化参数
                compressed = compress_token_type_ids(token_type_ids)  # 压缩 token 类型 ID
                params.extra_kwargs = {"compressed_token_type_ids": compressed}  # 添加压缩后的 token 类型 ID
                pooling_params_list.append(params)  # 添加到参数列表
            else:  # 若不存在 token 类型 ID
                pooling_params_list.append(pooling_params)  # 使用默认池化参数

            prompts.append(engine_prompt)  # 添加引擎提示到列表

        outputs = self._run_completion(  # 运行补全请求
            prompts=prompts,  # 提示词列表
            params=pooling_params_list,  # 池化参数列表
            output_type=PoolingRequestOutput,  # 输出类型
            use_tqdm=use_tqdm,  # 进度条设置
            lora_request=lora_request,  # LoRA 请求
        )

        return [ScoringRequestOutput.from_base(item) for item in outputs]  # 转换为评分输出并返回

    # score: 为所有输入对生成相似度评分（统一评分接口）
    def score(
        self,
        data_1: SingletonPrompt  # 第一组数据（位置参数）
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        data_2: SingletonPrompt  # 第二组数据（位置参数）
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        /,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        pooling_params: PoolingParams | None = None,  # 池化参数
        lora_request: list[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
        chat_template: str | None = None,  # 聊天模板（仅交叉编码器支持）
    ) -> list[ScoringRequestOutput]:
        """Generate similarity scores for all pairs `<text,text_pair>` or
          `<multi-modal data, multi-modal data pair>`.

        The inputs can be `1 -> 1`, `1 -> N` or `N -> N`.
        In the `1 - N` case the `data_1` input will be replicated `N`
        times to pair with the `data_2` inputs.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your inputs into a single list and pass it to this method.

        Supports both text and multi-modal data (images, etc.) when used with
        appropriate multi-modal models. For multi-modal inputs, ensure the
        prompt structure matches the model's expected input format.

        Args:
            data_1: Can be a single prompt, a list of prompts or
                `ScoreMultiModalParam`, which can contain either text or
                multi-modal data. When a list, it must have the same length as
                the `data_2` list.
            data_2: The data to pair with the query to form the input to
                the LLM. Can be text or multi-modal data. See [PromptType]
                [vllm.inputs.PromptType] for more details about the format of
                each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The chat template to use for the scoring. If None, we
                use the model's default chat template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        """
        model_config = self.model_config  # 获取模型配置

        runner_type = model_config.runner_type  # 获取运行器类型
        if runner_type != "pooling":  # 若不是池化模式
            raise ValueError(  # 抛出错误：score() 仅支持池化模型
                "LLM.score() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        supported_tasks = self.supported_tasks  # 获取支持的任务列表
        score_type = self.model_config.score_type  # 获取评分类型
        is_late_interaction = score_type == "late-interaction"  # 判断是否为延迟交互模型（如 ColBERT）
        is_cross_encoder = score_type == "cross-encoder"  # 判断是否为交叉编码器模型

        # Late interaction models (e.g., ColBERT) use token_embed for scoring
        # 延迟交互模型（如 ColBERT）使用 token_embed 进行评分
        if not is_late_interaction and all(  # 若不是延迟交互且不支持嵌入或分类任务
            t not in supported_tasks for t in ("embed", "classify")
        ):
            raise ValueError(  # 抛出错误：模型不支持评分 API
                "Score API is not supported by this model. "
                "Try converting the model using "
                "`--convert embed` or `--convert classify`."
            )

        if is_cross_encoder and getattr(model_config.hf_config, "num_labels", 0) != 1:  # 若为交叉编码器且标签数不为 1
            raise ValueError("Score API is only enabled for num_labels == 1.")  # 抛出错误：评分 API 仅适用于单标签

        if not is_cross_encoder and chat_template is not None:  # 若非交叉编码器但提供了聊天模板
            raise ValueError(  # 抛出错误：聊天模板仅支持交叉编码器
                "chat_template is only supported for cross-encoder models."
            )

        is_multimodal_model = model_config.is_multimodal_model  # 判断是否为多模态模型
        architecture = model_config.architecture  # 获取模型架构

        score_data_1, score_data_2 = validate_score_input(  # 验证并标准化评分输入
            data_1,  # type: ignore[arg-type]  # 第一组数据
            data_2,  # type: ignore[arg-type]  # 第二组数据
            is_multimodal_model=is_multimodal_model,  # 是否为多模态模型
            architecture=architecture,  # 模型架构
        )

        renderer = self.renderer  # 获取渲染器
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(  # 获取默认分词参数并合并
            **(tokenization_kwargs or {})
        )
        encode_kwargs = tok_params.get_encode_kwargs()  # 获取编码关键字参数

        if is_cross_encoder:  # 若为交叉编码器模型
            return self._cross_encoding_score(  # 使用交叉编码评分方法
                score_data_1,  # 第一组数据
                score_data_2,  # 第二组数据
                use_tqdm=use_tqdm,  # 进度条设置
                pooling_params=pooling_params,  # 池化参数
                lora_request=lora_request,  # LoRA 请求
                tokenization_kwargs=encode_kwargs,  # 编码参数
                score_template=chat_template,  # 评分模板
            )
        elif is_late_interaction:  # 若为延迟交互模型
            return self._late_interaction_score(  # 使用延迟交互评分方法
                score_data_1,  # 第一组数据
                score_data_2,  # 第二组数据
                use_tqdm=use_tqdm,  # 进度条设置
                pooling_params=pooling_params,  # 池化参数
                lora_request=lora_request,  # LoRA 请求
                tokenization_kwargs=encode_kwargs,  # 编码参数
            )
        else:  # 否则使用嵌入评分方法
            return self._embedding_score(  # 使用嵌入评分方法（余弦相似度）
                score_data_1,  # 第一组数据
                score_data_2,  # 第二组数据
                use_tqdm=use_tqdm,  # 进度条设置
                pooling_params=pooling_params,  # 池化参数
                lora_request=lora_request,  # LoRA 请求
                tokenization_kwargs=encode_kwargs,  # 编码参数
            )

    # start_profile: 开始性能分析（可选自定义追踪文件前缀）
    def start_profile(self, profile_prefix: str | None = None) -> None:
        """Start profiling with optional custom trace prefix.

        Args:
            profile_prefix: Optional prefix for the trace file names. If provided,
                           trace files will be named as "<prefix>_dp<X>_pp<Y>_tp<Z>".
                           If not provided, default naming will be used.
        """
        self.llm_engine.start_profile(profile_prefix)  # 委托给 LLM 引擎开始性能分析

    # stop_profile: 停止性能分析
    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()  # 委托给 LLM 引擎停止性能分析

    # reset_prefix_cache: 重置前缀缓存
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 是否重置运行中的请求和连接器
    ) -> bool:
        return self.llm_engine.reset_prefix_cache(  # 委托给 LLM 引擎重置前缀缓存
            reset_running_requests, reset_connector
        )

    # sleep: 将引擎置于睡眠状态（释放 GPU 资源）
    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level.
                - Level 0: Pause scheduling but continue accepting requests.
                           Requests are queued but not processed.
                - Level 1: Offload model weights to CPU, discard KV cache.
                           The content of kv cache is forgotten. Good for
                           sleeping and waking up the engine to run the same
                           model again. Please make sure there's enough CPU
                           memory to store the model weights.
                - Level 2: Discard all GPU memory (weights + KV cache).
                           Good for sleeping and waking up the engine to run
                           a different model or update the model, where
                           previous model weights are not needed. It reduces
                           CPU memory pressure.
            mode: How to handle any existing requests, can be "abort", "wait",
                or "keep".
        """
        self.llm_engine.sleep(level=level, mode=mode)  # 委托给 LLM 引擎进入睡眠

    # wake_up: 从睡眠模式唤醒引擎
    def wake_up(self, tags: list[str] | None = None):
        """
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache", "scheduling")`. If None, all memory
                is reallocated. wake_up should be called with all tags
                (or None) before the engine is used again.
                Use tags=["scheduling"] to resume from level 0 sleep.
        """
        self.llm_engine.wake_up(tags)  # 委托给 LLM 引擎唤醒

    # get_metrics: 获取 Prometheus 聚合指标的快照
    def get_metrics(self) -> list["Metric"]:
        """Return a snapshot of aggregated metrics from Prometheus.

        Returns:
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        return self.llm_engine.get_metrics()  # 委托给 LLM 引擎获取指标

    # _params_to_seq: 将单个参数或参数序列统一转换为参数序列
    def _params_to_seq(
        self,
        params: _P | Sequence[_P],  # 单个参数或参数序列
        num_requests: int,  # 请求数量
    ) -> Sequence[_P]:
        if isinstance(params, Sequence):  # 若已是序列
            if len(params) != num_requests:  # 长度不匹配则报错
                raise ValueError(
                    f"The lengths of prompts ({params}) "
                    f"and params ({len(params)}) must be the same."
                )

            return params  # 直接返回序列

        return [params] * num_requests  # 将单个参数复制为序列

    # _lora_request_to_seq: 将单个 LoRA 请求或序列统一转换为 LoRA 请求序列
    def _lora_request_to_seq(
        self,
        lora_request: LoRARequest | None | Sequence[LoRARequest | None],  # LoRA 请求
        num_requests: int,  # 请求数量
    ) -> Sequence[LoRARequest | None]:
        if isinstance(lora_request, Sequence):  # 若已是序列
            if len(lora_request) != num_requests:  # 长度不匹配则报错
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

            return lora_request  # 直接返回序列

        return [lora_request] * num_requests  # 将单个请求复制为序列

    # _priority_to_seq: 将优先级列表或 None 统一转换为优先级序列
    def _priority_to_seq(
        self,
        priority: list[int] | None,  # 优先级列表
        num_requests: int,  # 请求数量
    ) -> Sequence[int]:
        if priority is not None:  # 若提供了优先级列表
            if len(priority) != num_requests:  # 长度不匹配则报错
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )

            return priority  # 直接返回优先级列表

        return [0] * num_requests  # 默认所有请求优先级为 0

    # _add_completion_requests: 预处理提示词并将补全请求添加到引擎队列
    def _add_completion_requests(
        self,
        prompts: PromptType | Sequence[PromptType],  # 提示词
        params: SamplingParams  # 采样参数或池化参数
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        priority: list[int] | None = None,  # 优先级
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
    ) -> list[str]:
        seq_prompts = prompt_to_seq(prompts)  # 将提示词转换为序列
        seq_params = self._params_to_seq(params, len(seq_prompts))  # 将参数展开为序列
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))  # 将 LoRA 请求展开为序列
        seq_priority = self._priority_to_seq(priority, len(prompts))  # 将优先级展开为序列

        return self._render_and_add_requests(  # 渲染并添加请求
            prompts=(  # 使用生成器逐个预处理提示词（惰性求值，边处理边提交）
                self._preprocess_cmpl_one(prompt, tokenization_kwargs)
                for prompt in maybe_tqdm(  # 可选显示进度条
                    seq_prompts,
                    use_tqdm=use_tqdm,
                    desc="Rendering prompts",
                )
            ),
            params=seq_params,  # 参数序列
            lora_requests=seq_lora_requests,  # LoRA 请求序列
            priorities=seq_priority,  # 优先级序列
        )

    # _run_completion: 添加补全请求并运行引擎获取结果
    def _run_completion(
        self,
        prompts: PromptType | Sequence[PromptType],  # 提示词
        params: SamplingParams  # 采样参数或池化参数
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],  # 期望的输出类型
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        priority: list[int] | None = None,  # 优先级
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
    ):
        self._add_completion_requests(  # 先添加所有补全请求
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        )
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)  # 然后运行引擎获取结果

    # _run_chat: 处理聊天对话并运行引擎获取结果
    def _run_chat(
        self,
        messages: list[ChatCompletionMessageParam]  # 消息列表或消息列表序列
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams  # 采样参数或池化参数
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],  # 期望的输出类型
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,  # LoRA 请求
        chat_template: str | None = None,  # 聊天模板
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",  # 内容格式
        add_generation_prompt: bool = True,  # 是否添加生成提示
        continue_final_message: bool = False,  # 是否继续最后消息
        tools: list[dict[str, Any]] | None = None,  # 工具定义
        chat_template_kwargs: dict[str, Any] | None = None,  # 模板参数
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器参数
        mm_processor_kwargs: dict[str, Any] | None = None,  # 多模态处理器参数
    ):
        seq_convs = conversation_to_seq(messages)  # 将消息转换为对话序列
        seq_params = self._params_to_seq(params, len(seq_convs))  # 将参数展开为序列
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_convs))  # 将 LoRA 请求展开为序列

        return self._render_and_run_requests(  # 渲染并运行请求
            prompts=(  # 使用生成器逐个预处理对话（惰性求值）
                self._preprocess_chat_one(
                    conversation,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format,
                    chat_template_kwargs=chat_template_kwargs,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenization_kwargs=tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for conversation in maybe_tqdm(  # 可选显示进度条
                    seq_convs,
                    use_tqdm=use_tqdm,
                    desc="Rendering conversations",
                )
            ),
            params=seq_params,  # 参数序列
            output_type=output_type,  # 输出类型
            lora_requests=seq_lora_requests,  # LoRA 请求序列
            use_tqdm=use_tqdm,  # 进度条设置
        )

    # _render_and_run_requests: 渲染提示词、添加请求并运行引擎
    def _render_and_run_requests(
        self,
        prompts: Iterable[ProcessorInputs],  # 处理器输入的可迭代对象
        params: Sequence[SamplingParams | PoolingParams],  # 参数序列
        output_type: type[_O],  # 期望的输出类型
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,  # LoRA 请求序列
        priorities: Sequence[int] | None = None,  # 优先级序列
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
    ):
        if isinstance(prompts, (list, tuple)):  # 若提示已被完全实例化（非生成器）
            logger.warning_once(  # 警告：一次性渲染所有提示效率较低
                "Rendering all prompts before adding them to the engine "
                "is less efficient than performing both on the same prompt "
                "before processing the next prompt. You should instead pass "
                "a generator that renders one prompt per iteration, as that allows "
                "engine execution to begin for the first prompt while processing "
                "the next prompt."
            )

        self._render_and_add_requests(  # 渲染并添加请求到引擎
            prompts=prompts,
            params=params,
            lora_requests=lora_requests,
            priorities=priorities,
        )

        return self._run_engine(output_type, use_tqdm=use_tqdm)  # 运行引擎并返回结果

    # _render_and_add_requests: 渲染提示词并逐个添加请求到引擎队列
    def _render_and_add_requests(
        self,
        prompts: Iterable[ProcessorInputs],  # 处理器输入的可迭代对象
        params: Sequence[SamplingParams | PoolingParams],  # 参数序列
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,  # LoRA 请求序列
        priorities: Sequence[int] | None = None,  # 优先级序列
    ) -> list[str]:
        added_request_ids: list[str] = []  # 已添加的请求 ID 列表

        try:  # 尝试添加请求
            for i, prompt in enumerate(prompts):  # 遍历每个提示
                request_id = self._add_request(  # 添加单个请求
                    prompt,  # 处理器输入
                    params[i],  # 对应的参数
                    lora_request=self._resolve_mm_lora(  # 解析多模态 LoRA
                        prompt,
                        None if lora_requests is None else lora_requests[i],
                    ),
                    priority=0 if priorities is None else priorities[i],  # 优先级
                )
                added_request_ids.append(request_id)  # 记录已添加的请求 ID
        except Exception as e:  # 若添加过程中出错
            if added_request_ids:  # 若已有部分请求被添加
                self.llm_engine.abort_request(added_request_ids, internal=True)  # 中止已添加的请求
            raise e  # 重新抛出异常

        return added_request_ids  # 返回所有已添加的请求 ID

    # _add_request: 向引擎添加单个请求
    def _add_request(
        self,
        prompt: ProcessorInputs,  # 处理器输入
        params: SamplingParams | PoolingParams,  # 采样参数或池化参数
        lora_request: LoRARequest | None = None,  # LoRA 请求
        priority: int = 0,  # 优先级
    ) -> str:
        if isinstance(params, SamplingParams):  # 若为采样参数
            # We only care about the final output
            # 我们只关心最终输出（不需要中间结果）
            params.output_kind = RequestOutputKind.FINAL_ONLY  # 设置为仅返回最终输出

        request_id = str(next(self.request_counter))  # 生成唯一的请求 ID

        return self.llm_engine.add_request(  # 向 LLM 引擎添加请求
            request_id,  # 请求 ID
            prompt,  # 处理器输入
            params,  # 参数
            lora_request=lora_request,  # LoRA 请求
            priority=priority,  # 优先级
        )

    # _run_engine: 运行引擎处理所有未完成的请求并收集输出
    def _run_engine(
        self,
        output_type: type[_O] | tuple[type[_O], ...],  # 期望的输出类型
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # 是否显示进度条
    ) -> list[_O]:
        # Initialize tqdm.
        # 初始化 tqdm 进度条
        if use_tqdm:  # 若启用进度条
            num_requests = self.llm_engine.get_num_unfinished_requests()  # 获取未完成请求数量
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm  # 确定 tqdm 函数（自定义或默认）
            pbar = tqdm_func(  # 创建进度条
                total=num_requests,  # 总数为未完成请求数
                desc="Processed prompts",  # 描述：已处理的提示词
                dynamic_ncols=True,  # 动态列宽
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),  # 显示预估速度
            )

        # Run the engine.
        # 运行引擎
        outputs: list[_O] = []  # 初始化输出列表
        total_in_toks = 0  # 总输入 token 数
        total_out_toks = 0  # 总输出 token 数
        while self.llm_engine.has_unfinished_requests():  # 当还有未完成的请求时
            step_outputs = self.llm_engine.step()  # 执行引擎一步
            for output in step_outputs:  # 遍历步骤输出
                assert isinstance(output, output_type)  # 断言输出类型正确
                if output.finished:  # 若请求已完成
                    outputs.append(output)  # type: ignore[arg-type]  # 添加到输出列表
                    if use_tqdm:  # 若启用进度条
                        if isinstance(output, RequestOutput):  # 若为生成请求输出
                            # Calculate tokens only for RequestOutput
                            # 仅对 RequestOutput 计算 token 数
                            n = len(output.outputs)  # 输出序列数量
                            assert output.prompt_token_ids is not None  # 断言提示 token ID 不为 None
                            total_in_toks += len(output.prompt_token_ids) * n  # 累加输入 token 数
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]  # 计算输入速度
                            total_out_toks += sum(  # 累加输出 token 数
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]  # 计算输出速度
                            pbar.postfix = (  # 更新进度条后缀显示速度
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)  # 更新进度条
                        else:  # 若为池化请求输出
                            pbar.update(1)  # 更新进度条（每次加 1）
                        if pbar.n == num_requests:  # 若所有请求已完成
                            pbar.refresh()  # 刷新进度条显示

        if use_tqdm:  # 若启用了进度条
            pbar.close()  # 关闭进度条
        # Sort the outputs by request ID.
        # 按请求 ID 排序输出
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        # 这是必要的，因为某些请求可能比其前面的请求更早完成
        return sorted(outputs, key=lambda x: int(x.request_id))  # 按请求 ID 升序排序并返回

    # init_weight_transfer_engine: 初始化权重传输引擎（用于 RL 训练场景）
    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict  # 权重传输初始化请求
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        init_info_dict = (  # 从请求中提取初始化信息字典
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(  # 在所有 worker 上调用初始化方法
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    # update_weights: 更新模型权重（用于 RL 训练场景）
    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """
        Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info
        """
        update_info_dict = (  # 从请求中提取更新信息字典
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(  # 在所有 worker 上调用权重更新方法
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    # __repr__: 返回模型的 transformers 风格层次化视图字符串
    def __repr__(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        # Cache the result to avoid repeated collective_rpc calls
        # 缓存结果以避免重复的集合 RPC 调用
        if self._cached_repr is None:  # 若尚未缓存
            results = self.llm_engine.collective_rpc("get_model_inspection")  # 从 worker 获取模型信息
            # In distributed settings, we get results from all workers
            # 在分布式设置中，我们从所有 worker 获取结果
            # Just return the first one (they should all be the same)
            # 只返回第一个（它们应该都相同）
            if results:  # 若有结果
                self._cached_repr = results[0]  # 缓存第一个结果
            else:  # 若无结果
                self._cached_repr = f"LLM(model={self.model_config.model!r})"  # 使用简单格式作为后备
        return self._cached_repr  # 返回缓存的字符串表示
