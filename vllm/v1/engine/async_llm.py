# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio  # 导入异步IO模块
import os  # 导入操作系统接口模块
import socket  # 导入套接字模块
import time  # 导入时间模块
import warnings  # 导入警告模块
from collections.abc import AsyncGenerator, Iterable, Mapping  # 导入异步生成器、可迭代和映射抽象基类
from copy import copy  # 导入浅拷贝函数
from typing import Any  # 导入Any类型注解

import torch  # 导入PyTorch框架

import vllm.envs as envs  # 导入vLLM环境变量配置模块
from vllm import TokensPrompt  # 导入Token提示类
from vllm.config import VllmConfig  # 导入vLLM全局配置类
from vllm.distributed.weight_transfer.base import (  # 从分布式权重传输基础模块导入
    WeightTransferInitRequest,  # 权重传输初始化请求类
    WeightTransferUpdateRequest,  # 权重传输更新请求类
)  # 结束权重传输导入
from vllm.engine.arg_utils import AsyncEngineArgs  # 导入异步引擎参数工具类
from vllm.engine.protocol import EngineClient, StreamingInput  # 导入引擎客户端协议和流式输入类
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep  # 导入弹性EP扩缩容中间件设置函数
from vllm.inputs import ProcessorInputs, PromptType  # 导入处理器输入和提示类型
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.lora.request import LoRARequest  # 导入LoRA请求类
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 导入多模态注册表和注册表类
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput  # 导入流结束标记、池化请求输出和请求输出类
from vllm.plugins.io_processors import get_io_processor  # 导入IO处理器获取函数
from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.renderers import renderer_from_config  # 导入从配置创建渲染器的函数
from vllm.renderers.inputs.preprocess import extract_prompt_components  # 导入提取提示组件的函数
from vllm.sampling_params import RequestOutputKind, SamplingParams  # 导入请求输出类型和采样参数类
from vllm.tasks import SupportedTask  # 导入支持的任务类型
from vllm.tokenizers import TokenizerLike  # 导入分词器类型接口
from vllm.tracing import init_tracer  # 导入追踪器初始化函数
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value  # 导入配置序列化注册函数
from vllm.usage.usage_lib import UsageContext  # 导入使用上下文类
from vllm.utils.async_utils import cancel_task_threadsafe  # 导入线程安全取消任务函数
from vllm.utils.collection_utils import as_list  # 导入转换为列表的工具函数
from vllm.v1.engine import EngineCoreRequest, PauseMode  # 导入引擎核心请求和暂停模式类
from vllm.v1.engine.core_client import EngineCoreClient  # 导入引擎核心客户端类
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError  # 导入引擎死亡错误和生成错误类
from vllm.v1.engine.input_processor import InputProcessor  # 导入输入处理器类
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector  # 导入输出处理器和请求输出收集器类
from vllm.v1.engine.parallel_sampling import ParentRequest  # 导入并行采样父请求类
from vllm.v1.executor import Executor  # 导入执行器类
from vllm.v1.metrics.loggers import (  # 从指标日志模块导入
    StatLoggerFactory,  # 统计日志工厂类
    StatLoggerManager,  # 统计日志管理器类
    load_stat_logger_plugin_factories,  # 加载统计日志插件工厂函数
)  # 结束指标日志导入
from vllm.v1.metrics.prometheus import shutdown_prometheus  # 导入关闭Prometheus指标的函数
from vllm.v1.metrics.stats import IterationStats  # 导入迭代统计类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class InputStreamError(Exception):  # 输入流错误类，继承自Exception
    """输入流生成器错误的包装类。

    用于传播用户输入生成器的错误，
    而不会将其包装在EngineGenerateError中。
    """

    def __init__(self, cause: Exception):  # 初始化方法，接收原始异常
        self.cause = cause  # 保存原始异常原因
        super().__init__(str(cause))  # 调用父类初始化方法


# [中文注释] AsyncLLM — vLLM 的异步推理引擎前端，由 API Server 直接使用。
#   架构：三个核心组件协作
#     InputProcessor  — 将用户 prompt 转为 EngineCoreRequest
#     EngineCoreClient — 通过 ZMQ 与后台 EngineCore 进程通信
#     OutputProcessor  — 将 EngineCoreOutput 转为用户可见的 RequestOutput
#   请求生命周期：
#     1. generate()/encode() 调用 add_request() 提交请求
#     2. output_handler 后台 Task 从 EngineCore 拉取结果，经 OutputProcessor 处理后放入 queue
#     3. generate() 从 queue 中 yield RequestOutput 返回给调用者
#     4. 客户端断开时自动 abort 请求
#   注意：output_handler 使用 weakref 避免循环引用，确保 AsyncLLM 能被正确 GC
class AsyncLLM(EngineClient):  # 异步LLM类，继承自EngineClient引擎客户端
    """vLLM引擎的异步包装类。"""

    def __init__(  # 初始化方法
        self,  # 实例自身引用
        vllm_config: VllmConfig,  # vLLM全局配置对象
        executor_class: type[Executor],  # 执行器类，例如MultiprocExecutor
        log_stats: bool,  # 是否记录统计信息
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用上下文，默认为引擎上下文
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表，默认为全局注册表
        use_cached_outputs: bool = False,  # 是否使用缓存输出，默认否
        log_requests: bool = True,  # 是否记录请求日志，默认是
        start_engine_loop: bool = True,  # 是否启动引擎循环，默认是
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义统计日志工厂列表
        aggregate_engine_logging: bool = False,  # 是否聚合引擎日志，默认否
        client_addresses: dict[str, str] | None = None,  # 客户端地址字典
        client_count: int = 1,  # 客户端数量，默认为1
        client_index: int = 0,  # 客户端索引，默认为0
    ) -> None:  # 返回None
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # 确保可以序列化自定义transformer配置
        maybe_register_config_serialize_by_value()  # 注册配置按值序列化

        self.vllm_config = vllm_config  # 保存vLLM全局配置
        self.model_config = vllm_config.model_config  # 保存模型配置
        self.observability_config = vllm_config.observability_config  # 保存可观测性配置

        tracing_endpoint = self.observability_config.otlp_traces_endpoint  # 获取追踪端点地址
        if tracing_endpoint is not None:  # 如果追踪端点已配置
            init_tracer("vllm.llm_engine", tracing_endpoint)  # 初始化追踪器

        self.log_requests = log_requests  # 保存是否记录请求的标志

        custom_stat_loggers = list(stat_loggers or [])  # 创建自定义统计日志列表
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())  # 加载并添加统计日志插件工厂

        has_custom_loggers = bool(custom_stat_loggers)  # 检查是否有自定义日志记录器
        self.log_stats = log_stats or has_custom_loggers  # 如果有自定义日志记录器也开启统计日志
        if not log_stats and has_custom_loggers:  # 如果原始设置不记录但有自定义日志器
            logger.info(  # 记录信息日志
                "AsyncLLM created with log_stats=False, "  # 日志消息：创建时log_stats为False
                "but custom stat loggers were found; "  # 但发现了自定义统计日志器
                "enabling logging without default stat loggers."  # 启用日志但不使用默认统计日志器
            )  # 结束日志信息

        self.renderer = renderer = renderer_from_config(self.vllm_config)  # 从配置创建渲染器
        self.io_processor = get_io_processor(  # 获取IO处理器
            self.vllm_config,  # 传入vLLM配置
            self.renderer,  # 传入渲染器
            self.model_config.io_processor_plugin,  # 传入IO处理器插件名
        )  # 结束IO处理器创建

        # 将TokPrompt转换为EngineCoreRequest
        self.input_processor = InputProcessor(self.vllm_config, renderer)  # 创建输入处理器

        # 将EngineCoreOutputs转换为RequestOutput
        self.output_processor = OutputProcessor(  # 创建输出处理器
            renderer.tokenizer,  # 传入分词器
            log_stats=self.log_stats,  # 传入是否记录统计
            stream_interval=self.vllm_config.scheduler_config.stream_interval,  # 传入流式输出间隔
            tracing_enabled=tracing_endpoint is not None,  # 传入追踪是否启用
        )  # 结束输出处理器创建

        # EngineCore（在后台进程中启动引擎）
        self.engine_core = EngineCoreClient.make_async_mp_client(  # 创建异步多进程引擎核心客户端
            vllm_config=vllm_config,  # 传入vLLM配置
            executor_class=executor_class,  # 传入执行器类
            log_stats=self.log_stats,  # 传入是否记录统计
            client_addresses=client_addresses,  # 传入客户端地址
            client_count=client_count,  # 传入客户端数量
            client_index=client_index,  # 传入客户端索引
        )  # 结束引擎核心客户端创建

        # 日志记录器
        self.logger_manager: StatLoggerManager | None = None  # 初始化统计日志管理器为None
        if self.log_stats:  # 如果启用了统计日志
            self.logger_manager = StatLoggerManager(  # 创建统计日志管理器
                vllm_config=vllm_config,  # 传入vLLM配置
                engine_idxs=self.engine_core.engine_ranks_managed,  # 传入管理的引擎排名索引
                custom_stat_loggers=custom_stat_loggers,  # 传入自定义统计日志器
                enable_default_loggers=log_stats,  # 传入是否启用默认日志器
                client_count=client_count,  # 传入客户端数量
                aggregate_engine_logging=aggregate_engine_logging,  # 传入是否聚合引擎日志
            )  # 结束统计日志管理器创建
            self.logger_manager.log_engine_initialized()  # 记录引擎已初始化的日志

        self._client_count = client_count  # 保存客户端数量

        self.output_handler: asyncio.Task | None = None  # 初始化输出处理任务为None
        try:  # 尝试执行
            # 如果在asyncio事件循环中，则立即启动输出处理器
            asyncio.get_running_loop()  # 获取正在运行的事件循环
            self._run_output_handler()  # 启动输出处理器
        except RuntimeError:  # 捕获运行时错误（没有事件循环时抛出）
            pass  # 忽略错误，稍后启动

        if (  # 如果满足以下条件
            vllm_config.profiler_config.profiler == "torch"  # 使用torch性能分析器
            and not vllm_config.profiler_config.ignore_frontend  # 且不忽略前端分析
        ):  # 条件判断结束括号
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir  # 获取分析器输出目录
            logger.info(  # 记录信息日志
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501  # 分析器已启用，CPU跟踪将收集到指定目录
                profiler_dir,  # 传入分析器目录路径
            )  # 结束日志记录
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"  # 构建worker名称：主机名_进程ID.async_llm
            self.profiler = torch.profiler.profile(  # 创建torch性能分析器实例
                activities=[  # 指定分析活动列表
                    torch.profiler.ProfilerActivity.CPU,  # 分析CPU活动
                ],  # 结束活动列表
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,  # 是否包含调用栈信息
                on_trace_ready=torch.profiler.tensorboard_trace_handler(  # 追踪就绪时的处理函数
                    profiler_dir,  # 传入输出目录
                    worker_name=worker_name,  # 传入worker名称
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,  # 是否使用gzip压缩
                ),  # 结束追踪处理函数
            )  # 结束分析器创建
        else:  # 否则（不启用torch分析器）
            self.profiler = None  # 将分析器设为None

    @classmethod  # 类方法装饰器
    def from_vllm_config(  # 从vLLM配置创建AsyncLLM实例的工厂方法
        cls,  # 类自身引用
        vllm_config: VllmConfig,  # vLLM全局配置对象
        start_engine_loop: bool = True,  # 是否启动引擎循环，默认是
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用上下文，默认引擎上下文
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义统计日志工厂列表
        enable_log_requests: bool = False,  # 是否启用请求日志，默认否
        aggregate_engine_logging: bool = False,  # 是否聚合引擎日志，默认否
        disable_log_stats: bool = False,  # 是否禁用统计日志，默认否
        client_addresses: dict[str, str] | None = None,  # 客户端地址字典
        client_count: int = 1,  # 客户端数量，默认为1
        client_index: int = 0,  # 客户端索引，默认为0
    ) -> "AsyncLLM":  # 返回AsyncLLM实例
        # Create the LLMEngine.
        return cls(  # 调用构造函数创建实例
            vllm_config=vllm_config,  # 传入vLLM配置
            executor_class=Executor.get_class(vllm_config),  # 根据配置获取执行器类
            start_engine_loop=start_engine_loop,  # 传入是否启动引擎循环
            stat_loggers=stat_loggers,  # 传入统计日志工厂列表
            log_requests=enable_log_requests,  # 传入是否记录请求
            log_stats=not disable_log_stats,  # 传入是否记录统计（取反）
            aggregate_engine_logging=aggregate_engine_logging,  # 传入是否聚合引擎日志
            usage_context=usage_context,  # 传入使用上下文
            client_addresses=client_addresses,  # 传入客户端地址
            client_count=client_count,  # 传入客户端数量
            client_index=client_index,  # 传入客户端索引
        )  # 结束构造函数调用

    @classmethod  # 类方法装饰器
    def from_engine_args(  # 从引擎参数创建AsyncLLM实例的工厂方法
        cls,  # 类自身引用
        engine_args: AsyncEngineArgs,  # 异步引擎参数对象
        start_engine_loop: bool = True,  # 是否启动引擎循环，默认是
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用上下文，默认引擎上下文
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义统计日志工厂列表
    ) -> "AsyncLLM":  # 返回AsyncLLM实例
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)  # 从引擎参数创建引擎配置
        executor_class = Executor.get_class(vllm_config)  # 根据配置获取执行器类

        # Create the AsyncLLM.
        return cls(  # 调用构造函数创建实例
            vllm_config=vllm_config,  # 传入vLLM配置
            executor_class=executor_class,  # 传入执行器类
            log_requests=engine_args.enable_log_requests,  # 传入是否记录请求
            log_stats=not engine_args.disable_log_stats,  # 传入是否记录统计（取反）
            start_engine_loop=start_engine_loop,  # 传入是否启动引擎循环
            usage_context=usage_context,  # 传入使用上下文
            stat_loggers=stat_loggers,  # 传入统计日志工厂列表
        )  # 结束构造函数调用

    def __del__(self):  # 析构方法
        self.shutdown()  # 调用关闭方法进行清理

    def shutdown(self, timeout: float | None = None) -> None:  # 关闭方法，清理后台进程和IPC
        """关闭引擎，清理后台进程和进程间通信。"""
        shutdown_prometheus()  # 关闭Prometheus指标收集

        if renderer := getattr(self, "renderer", None):  # 如果渲染器存在
            renderer.shutdown()  # 关闭渲染器

        if engine_core := getattr(self, "engine_core", None):  # 如果引擎核心存在
            engine_core.shutdown(timeout=timeout)  # 关闭引擎核心，传入超时时间

        handler = getattr(self, "output_handler", None)  # 获取输出处理任务
        if handler is not None:  # 如果输出处理任务存在
            cancel_task_threadsafe(handler)  # 以线程安全方式取消该任务

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 异步获取支持的任务类型
        """获取引擎支持的任务类型元组。"""
        if not hasattr(self, "_supported_tasks"):  # 如果尚未缓存支持的任务
            # 缓存结果
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()  # 从引擎核心异步获取支持的任务

        return self._supported_tasks  # 返回支持的任务元组

    async def add_request(  # 异步添加请求方法
        self,  # 实例自身引用
        request_id: str,  # 请求唯一标识符
        prompt: EngineCoreRequest  # 提示可以是引擎核心请求
        | PromptType  # 或提示类型
        | ProcessorInputs  # 或处理器输入
        | AsyncGenerator[StreamingInput, None],  # 或异步流式输入生成器
        params: SamplingParams | PoolingParams,  # 采样参数或池化参数
        arrival_time: float | None = None,  # 请求到达时间
        lora_request: LoRARequest | None = None,  # LoRA适配器请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息
        priority: int = 0,  # 请求优先级，默认为0
        data_parallel_rank: int | None = None,  # 数据并行排名
        prompt_text: str | None = None,  # 提示文本
        reasoning_ended: bool | None = None,  # 推理是否结束标志
    ) -> RequestOutputCollector:  # 返回请求输出收集器
        """向AsyncLLM添加新请求。"""

        if self.errored:  # 如果引擎已出错
            raise EngineDeadError()  # 抛出引擎死亡错误

        is_pooling = isinstance(params, PoolingParams)  # 检查是否为池化请求

        if (  # 如果满足以下所有条件
            self.vllm_config.cache_config.kv_sharing_fast_prefill  # 启用了KV共享快速预填充
            and not is_pooling  # 且不是池化请求
            and params.prompt_logprobs  # 且需要提示的对数概率
        ):  # 条件判断结束
            raise ValueError(  # 抛出值错误
                "--kv-sharing-fast-prefill produces incorrect logprobs for "  # KV共享快速预填充会产生不正确的提示token对数概率
                "prompt tokens, please disable it when the requests need "  # 当请求需要提示对数概率时请禁用
                "prompt logprobs"  # 提示对数概率
            )  # 结束错误消息

        if isinstance(prompt, AsyncGenerator):  # 如果提示是异步生成器（流式输入）
            if reasoning_ended is not None:  # 如果设置了推理结束标志
                raise NotImplementedError  # 流式输入暂不支持推理结束标志

            # 流式输入的情况
            return await self._add_streaming_input_request(  # 异步调用添加流式输入请求方法
                request_id,  # 传入请求ID
                prompt,  # 传入流式提示生成器
                params,  # 传入参数
                arrival_time,  # 传入到达时间
                lora_request,  # 传入LoRA请求
                tokenization_kwargs,  # 传入分词参数
                trace_headers,  # 传入追踪头
                priority,  # 传入优先级
                data_parallel_rank,  # 传入数据并行排名
            )  # 结束流式请求添加

        # 将输入转换为请求
        if isinstance(prompt, EngineCoreRequest):  # 如果提示已经是引擎核心请求
            logger.warning_once(  # 记录一次性警告
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "  # 向AsyncLLM传递EngineCoreRequest已弃用
                "is deprecated and will be removed in v0.18. You should instead pass "  # 将在v0.18中移除
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."  # 应该传入渲染器的输出
            )  # 结束警告

            request = prompt  # 直接使用提示作为请求
            if request_id != request.request_id:  # 如果传入的请求ID与请求对象中的不匹配
                logger.warning_once(  # 记录一次性警告
                    "AsyncLLM.add_request() was passed a request_id parameter that "  # 传入的request_id参数
                    "does not match the EngineCoreRequest.request_id attribute. The "  # 与请求对象的属性不匹配
                    "latter will be used, and the former will be ignored."  # 将使用后者，忽略前者
                )  # 结束警告
        else:  # 否则需要处理输入
            request = self.input_processor.process_inputs(  # 通过输入处理器处理输入
                request_id,  # 传入请求ID
                prompt,  # 传入提示
                params,  # 传入参数
                supported_tasks=await self.get_supported_tasks(),  # 异步获取支持的任务类型
                arrival_time=arrival_time,  # 传入到达时间
                lora_request=lora_request,  # 传入LoRA请求
                tokenization_kwargs=tokenization_kwargs,  # 传入分词参数
                trace_headers=trace_headers,  # 传入追踪头
                priority=priority,  # 传入优先级
                data_parallel_rank=data_parallel_rank,  # 传入数据并行排名
            )  # 结束输入处理
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)  # 提取提示文本组件

        if reasoning_ended is not None:  # 如果指定了推理结束标志
            request.reasoning_ended = reasoning_ended  # 设置请求的推理结束状态

        self.input_processor.assign_request_id(request)  # 为请求分配内部请求ID

        # 在第一次调用add_request()时启动output_handler
        # 这样可以在事件循环之前调用__init__，使我们能在OpenAI服务器中优雅处理启动失败
        self._run_output_handler()  # 启动输出处理器

        # 为请求创建新的输出收集器
        queue = RequestOutputCollector(params.output_kind, request.request_id)  # 创建请求输出收集器

        # 使用可能在process_inputs()中被更新的克隆参数
        params = request.params  # 获取请求中的参数

        if is_pooling or params.n == 1:  # 如果是池化请求或只需要1个采样结果
            await self._add_request(request, prompt_text, None, 0, queue)  # 异步添加单个请求
            return queue  # 返回输出收集器

        parent_params = params  # 保存父请求参数
        assert isinstance(parent_params, SamplingParams)  # 断言参数为采样参数类型

        # 展开子请求（当n>1时进行并行采样）
        parent_request = ParentRequest(request)  # 创建父请求对象
        for idx in range(parent_params.n):  # 遍历每个采样索引
            request_id, child_params = parent_request.get_child_info(idx)  # 获取子请求的ID和参数
            child_request = request if idx == parent_params.n - 1 else copy(request)  # 最后一个子请求复用原请求，其他浅拷贝
            child_request.request_id = request_id  # 设置子请求的ID
            child_request.sampling_params = child_params  # 设置子请求的采样参数
            await self._add_request(  # 异步添加子请求
                child_request, prompt_text, parent_request, idx, queue  # 传入子请求、提示文本、父请求、索引和队列
            )  # 结束子请求添加
        return queue  # 返回输出收集器

    async def _add_request(  # 内部异步添加请求方法
        self,  # 实例自身引用
        request: EngineCoreRequest,  # 引擎核心请求对象
        prompt: str | None,  # 提示文本或None
        parent_req: ParentRequest | None,  # 父请求（并行采样时使用）或None
        index: int,  # 子请求在并行采样中的索引
        queue: RequestOutputCollector,  # 请求输出收集器队列
    ):  # 方法签名结束
        """将请求添加到输出处理器和引擎核心。"""
        # 将请求添加到OutputProcessor（当前进程中）
        self.output_processor.add_request(request, prompt, parent_req, index, queue)  # 在输出处理器中注册请求

        # 将EngineCoreRequest添加到EngineCore（独立进程中）
        await self.engine_core.add_request_async(request)  # 异步发送请求到引擎核心

        if self.log_requests:  # 如果启用了请求日志
            logger.info("Added request %s.", request.request_id)  # 记录添加请求的日志

    async def _add_streaming_input_request(  # 异步添加流式输入请求的内部方法
        self,  # 实例自身引用
        request_id: str,  # 请求唯一标识符
        input_stream: AsyncGenerator[StreamingInput, None],  # 流式输入异步生成器
        sampling_params: SamplingParams | PoolingParams,  # 采样参数或池化参数
        arrival_time: float | None = None,  # 请求到达时间，默认为None
        lora_request: LoRARequest | None = None,  # LoRA请求，默认为None
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数字典，默认为None
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息，默认为None
        priority: int = 0,  # 请求优先级，默认为0
        data_parallel_rank: int | None = None,  # 数据并行排名，默认为None
    ) -> RequestOutputCollector:  # 返回请求输出收集器
        self._validate_streaming_input_sampling_params(sampling_params)  # 验证流式输入的采样参数

        inputs = dict(  # 创建公共输入参数字典
            supported_tasks=await self.get_supported_tasks(),  # 异步获取支持的任务类型
            arrival_time=arrival_time,  # 到达时间
            lora_request=lora_request,  # LoRA请求
            tokenization_kwargs=tokenization_kwargs,  # 分词参数
            trace_headers=trace_headers,  # 追踪头
            priority=priority,  # 优先级
            data_parallel_rank=data_parallel_rank,  # 数据并行排名
        )  # 结束输入参数字典

        if not sampling_params.skip_clone:  # 如果采样参数尚未被克隆
            sampling_params = sampling_params.clone()  # 克隆采样参数
            sampling_params.skip_clone = True  # 标记已克隆，避免重复克隆

        # Create request for validation, also used as the finished signal
        # once the input stream is closed.
        final_req = self.input_processor.process_inputs(  # 创建用于验证的最终请求，也作为输入流关闭的信号
            request_id=request_id,  # 传入请求ID
            prompt=TokensPrompt(prompt_token_ids=[0]),  # 使用占位符token提示
            params=sampling_params,  # 传入采样参数
            **inputs,  # type: ignore[arg-type]  # 展开公共输入参数
        )  # 结束最终请求创建
        self.input_processor.assign_request_id(final_req)  # 为最终请求分配内部ID
        internal_req_id = final_req.request_id  # 获取内部请求ID

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)  # 创建请求输出收集器

        async def handle_inputs():  # 定义异步输入处理内部函数
            cancelled = False  # 初始化取消标志为False
            try:  # 尝试处理输入流
                async for input_chunk in input_stream:  # 异步遍历输入流中的每个块
                    sp = input_chunk.sampling_params  # 获取输入块的采样参数
                    if sp:  # 如果输入块提供了采样参数
                        self._validate_streaming_input_sampling_params(sp)  # 验证流式输入的采样参数
                    else:  # 否则使用默认采样参数
                        sp = sampling_params  # 使用初始传入的采样参数
                    # TODO(nick): Avoid re-validating reused sampling parameters
                    req = self.input_processor.process_inputs(  # 处理输入块为引擎核心请求
                        request_id=internal_req_id,  # 使用内部请求ID
                        prompt=input_chunk.prompt,  # 传入输入块的提示
                        params=sp,  # 传入采样参数
                        resumable=True,  # 标记为可恢复请求
                        **inputs,  # type: ignore[arg-type]  # 展开公共输入参数
                    )  # 结束输入处理
                    req.external_req_id = request_id  # 设置外部请求ID
                    if req.prompt_embeds is not None:  # 如果请求包含提示嵌入
                        raise ValueError(  # 抛出值错误
                            "prompt_embeds not supported for streaming inputs"  # 流式输入不支持提示嵌入
                        )  # 结束异常抛出
                    prompt_text, _, _ = extract_prompt_components(  # 提取提示文本组件
                        self.model_config, input_chunk.prompt  # 传入模型配置和提示
                    )  # 结束提取
                    await self._add_request(req, prompt_text, None, 0, queue)  # 异步添加请求到引擎
            except (asyncio.CancelledError, GeneratorExit):  # 捕获取消或生成器退出异常
                cancelled = True  # 设置取消标志为True
            except Exception as error:  # 捕获其他异常
                # Wrap in InputStreamError so generate() can propagate it
                # without wrapping in EngineGenerateError.
                queue.put(InputStreamError(error))  # 将错误包装为InputStreamError放入队列
            finally:  # 最终清理阶段
                queue._input_stream_task = None  # 清除输入流任务引用
                if not cancelled:  # 如果未被取消
                    # 发送空的最终请求表示输入已完成
                    # 如果已取消（会话已中止）则不发送
                    await self._add_request(final_req, None, None, 0, queue)  # 异步发送最终请求

        # 确保输出处理器正在运行
        self._run_output_handler()  # 启动输出处理器

        queue._input_stream_task = asyncio.create_task(handle_inputs())  # 创建异步任务处理输入流
        return queue  # 返回输出收集器

    @staticmethod  # 静态方法装饰器
    def _validate_streaming_input_sampling_params(  # 验证流式输入的采样参数
        params: SamplingParams | PoolingParams,  # 采样参数或池化参数
    ):  # 方法签名结束
        """验证流式输入的采样参数是否合法。"""
        if (  # 如果满足以下任一条件
            not isinstance(params, SamplingParams)  # 不是采样参数类型
            or params.n > 1  # 或采样数量大于1
            or params.output_kind == RequestOutputKind.FINAL_ONLY  # 或输出类型为仅最终结果
            or params.stop  # 或设置了停止字符串
        ):  # 条件判断结束
            raise ValueError(  # 抛出值错误
                "Input streaming not currently supported "  # 输入流式处理当前不支持
                "for pooling models, n > 1, request_kind = FINAL_ONLY "  # 池化模型、n>1、仅最终结果
                "or with stop strings."  # 或带停止字符串的情况
            )  # 结束错误消息

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(  # 异步生成方法，API服务器调用的主入口
        self,  # 实例自身引用
        prompt: EngineCoreRequest  # 提示可以是引擎核心请求
        | PromptType  # 或提示类型
        | ProcessorInputs  # 或处理器输入
        | AsyncGenerator[StreamingInput, None],  # 或异步流式输入生成器
        sampling_params: SamplingParams,  # 采样参数
        request_id: str,  # 请求唯一标识符
        *,  # 以下为仅关键字参数
        prompt_text: str | None = None,  # 提示文本
        lora_request: LoRARequest | None = None,  # LoRA适配器请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息
        priority: int = 0,  # 请求优先级
        data_parallel_rank: int | None = None,  # 数据并行排名
        reasoning_ended: bool | None = None,  # 推理是否结束标志
    ) -> AsyncGenerator[RequestOutput, None]:  # 返回请求输出的异步生成器
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        q: RequestOutputCollector | None = None  # 初始化请求输出收集器为None
        try:  # 尝试处理生成请求
            q = await self.add_request(  # 异步添加请求
                request_id,  # 传入请求ID
                prompt,  # 传入提示
                sampling_params,  # 传入采样参数
                lora_request=lora_request,  # 传入LoRA请求
                tokenization_kwargs=tokenization_kwargs,  # 传入分词参数
                trace_headers=trace_headers,  # 传入追踪头
                priority=priority,  # 传入优先级
                data_parallel_rank=data_parallel_rank,  # 传入数据并行排名
                prompt_text=prompt_text,  # 传入提示文本
                reasoning_ended=reasoning_ended,  # 传入推理结束标志
            )  # 结束添加请求

            # output_handler任务将结果推入队列
            # 本任务从队列拉取并yield给调用者
            finished = False  # 初始化完成标志为False
            while not finished:  # 当请求未完成时循环
                # 注意：尽可能不使用await来排空队列（避免负载下的任务切换，提升性能）
                out = q.get_nowait() or await q.get()  # 优先无等待获取，否则异步等待获取

                # 注意：OutputProcessor和EngineCore都基于finished标志处理各自的请求清理
                assert isinstance(out, RequestOutput)  # 断言输出为RequestOutput类型
                finished = out.finished  # 更新完成标志
                if out is not STREAM_FINISHED:  # 如果不是流结束标记
                    yield out  # 向调用者yield输出结果

        # 如果客户端断开连接，generate()被取消或生成器被垃圾回收
        # 则中止请求
        except (asyncio.CancelledError, GeneratorExit):  # 捕获取消和生成器退出异常
            if q is not None:  # 如果队列已创建
                await self.abort(q.request_id, internal=True)  # 异步中止请求
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s aborted.", request_id)  # 记录请求中止日志
            raise  # 重新抛出异常

        # 引擎已死亡，不中止因为已经关闭
        except EngineDeadError:  # 捕获引擎死亡错误
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s failed (engine dead).", request_id)  # 记录引擎死亡日志
            raise  # 重新抛出异常

        # 请求验证错误
        except ValueError as e:  # 捕获值错误
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s failed (bad request): %s.", request_id, e)  # 记录错误请求日志
            raise  # 重新抛出异常

        # 输入流生成器的错误 - 直接传播
        except InputStreamError as e:  # 捕获输入流错误
            if q is not None:  # 如果队列已创建
                await self.abort(q.request_id, internal=True)  # 异步中止请求
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s failed (input error): %s.", request_id, e)  # 记录输入错误日志
            raise e.cause from e  # 抛出原始异常，保留异常链

        # generate()任务中的意外错误（可能可恢复）
        except Exception as e:  # 捕获所有其他异常
            if q is not None:  # 如果队列已创建
                await self.abort(q.request_id, internal=True)  # 异步中止请求
            if self.log_requests:  # 如果启用请求日志
                try:  # 尝试格式化错误信息
                    s = f"{e.__class__.__name__}: {e}"  # 格式化异常类名和消息
                except Exception as e2:  # 如果格式化过程也出错
                    s = (  # 构建备用错误消息
                        f"{e.__class__.__name__}: "  # 原始异常类名
                        "error during printing an exception of class"  # 打印异常时出错
                        + e2.__class__.__name__  # 新异常类名
                    )  # 结束备用错误消息
                logger.info("Request %s failed due to %s.", request_id, s)  # 记录请求失败日志
            raise EngineGenerateError() from e  # 抛出引擎生成错误，保留原始异常链
        finally:  # 最终清理阶段
            if q is not None:  # 如果队列已创建
                q.close()  # 关闭输出收集器

    # [中文注释] 启动后台 output_handler 协程。核心循环：
    #   1. await engine_core.get_output_async() — 从 EngineCore 拉取批量输出
    #   2. output_processor.process_outputs() — detokenize + logprobs + 构造 RequestOutput
    #      分块处理（VLLM_V1_OUTPUT_PROC_CHUNK_SIZE），避免长时间阻塞事件循环
    #   3. abort 因 stop string 提前终止的请求
    #   4. 记录统计日志
    #   所有局部变量从 self 提取后传入闭包，避免 asyncio.Task 持有 self 引用
    def _run_output_handler(self):  # 启动后台输出处理器方法
        """后台循环：从EngineCore拉取输出并推送到异步流。"""

        if self.output_handler is not None:  # 如果输出处理器已经在运行
            return  # 直接返回，避免重复启动

        # 确保任务没有循环引用回AsyncLLM对象，否则不会被正确垃圾回收和清理
        engine_core = self.engine_core  # 提取引擎核心引用到局部变量
        output_processor = self.output_processor  # 提取输出处理器引用到局部变量
        log_stats = self.log_stats  # 提取是否记录统计的标志
        # 使用可变列表存储logger_manager，以便在弹性EP扩缩容时可以更新
        # （见scale_elastic_ep），同时避免通过self创建循环引用
        self._logger_ref = [self.logger_manager]  # 创建日志管理器的可变引用列表
        logger_ref = self._logger_ref  # 提取日志引用到局部变量
        renderer = self.renderer  # 提取渲染器引用到局部变量
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE  # 获取输出处理的分块大小

        async def output_handler():  # 定义异步输出处理函数
            """后台协程：持续从引擎核心拉取输出并处理。"""
            try:  # 尝试执行主循环
                while True:  # 无限循环处理输出
                    # 1) 从EngineCore拉取EngineCoreOutputs
                    outputs = await engine_core.get_output_async()  # 异步获取引擎输出
                    num_outputs = len(outputs.outputs)  # 获取输出数量

                    iteration_stats = (  # 创建迭代统计对象
                        IterationStats() if (log_stats and num_outputs) else None  # 有统计需求且有输出时创建，否则为None
                    )  # 结束迭代统计创建

                    # 将输出分块处理，每块最多VLLM_V1_OUTPUT_PROC_CHUNK_SIZE个
                    # 避免长时间阻塞事件循环
                    engine_core_outputs = outputs.outputs  # 获取引擎核心输出列表
                    for start in range(0, num_outputs, chunk_size):  # 按分块大小遍历
                        end = start + chunk_size  # 计算当前分块的结束位置
                        outputs_slice = engine_core_outputs[start:end]  # 切片获取当前分块
                        # 2) 处理EngineCoreOutputs
                        processed_outputs = output_processor.process_outputs(  # 调用输出处理器处理输出
                            outputs_slice, outputs.timestamp, iteration_stats  # 传入输出切片、时间戳和统计对象
                        )  # 结束输出处理
                        # 注意：RequestOutput已被推送到各自的队列中
                        assert not processed_outputs.request_outputs  # 断言处理后的输出列表为空（已推送到队列）

                        # 在分块之间允许其他asyncio任务运行
                        if end < num_outputs:  # 如果还有更多分块要处理
                            await asyncio.sleep(0)  # 让出事件循环控制权

                        # 3) 中止因停止字符串而结束的请求
                        if processed_outputs.reqs_to_abort:  # 如果有需要中止的请求
                            await engine_core.abort_requests_async(  # 异步中止请求
                                processed_outputs.reqs_to_abort  # 传入需要中止的请求列表
                            )  # 结束中止请求

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)  # 更新调度器统计信息

                    # 4) 日志记录
                    # TODO(rob): 一旦Prometheus开销变得不可忽略，改为协程并在后台线程中启动
                    if logger_ref[0]:  # 如果日志管理器存在
                        logger_ref[0].record(  # 记录统计信息
                            engine_idx=outputs.engine_index,  # 传入引擎索引
                            scheduler_stats=outputs.scheduler_stats,  # 传入调度器统计
                            iteration_stats=iteration_stats,  # 传入迭代统计
                            mm_cache_stats=renderer.stat_mm_cache(),  # 传入多模态缓存统计
                        )  # 结束统计记录
            except Exception as e:  # 捕获所有异常
                logger.exception("AsyncLLM output_handler failed.")  # 记录异常日志
                output_processor.propagate_error(e)  # 将错误传播到所有等待中的请求

        self.output_handler = asyncio.create_task(output_handler())  # 创建异步任务并保存引用

    async def abort(  # 异步中止请求方法
        self, request_id: str | Iterable[str], internal: bool = False  # 接收请求ID（单个或多个）和内部标志
    ) -> None:  # 返回None
        """在OutputProcessor和EngineCore中中止请求。"""

        request_ids = (  # 将请求ID统一转换为元组或列表
            (request_id,) if isinstance(request_id, str) else as_list(request_id)  # 单个ID转元组，多个转列表
        )  # 结束请求ID转换
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)  # 在输出处理器中中止请求并获取所有关联ID
        await self.engine_core.abort_requests_async(all_request_ids)  # 异步在引擎核心中中止请求

        if self.log_requests:  # 如果启用请求日志
            logger.info("Aborted request(s) %s.", ",".join(request_ids))  # 记录中止请求的日志

    async def pause_generation(  # 异步暂停生成方法
        self,  # 实例自身引用
        *,  # 以下为仅关键字参数
        mode: PauseMode = "abort",  # 暂停模式，默认为中止
        wait_for_inflight_requests: bool | None = None,  # 已弃用：是否等待进行中的请求
        clear_cache: bool = True,  # 是否清除缓存，默认是
    ) -> None:  # 返回None
        """
        Pause generation to allow model weight updates.

        All mode handling (abort / wait / keep) and cache clearing is done
        in the engine. New generation/encoding requests will not be scheduled
        until resume is called.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED: use mode argument.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
        """
        if wait_for_inflight_requests:  # 如果设置了已弃用的等待参数
            warnings.warn(  # 发出弃用警告
                "The `wait_for_inflight_requests` parameter in "  # wait_for_inflight_requests参数
                "`AsyncLLM.pause_generation()` is deprecated. "  # 在pause_generation中已弃用
                "Please use `mode` argument instead.",  # 请改用mode参数
                DeprecationWarning,  # 弃用警告类型
                stacklevel=2,  # 警告栈级别为2
            )  # 结束弃用警告
            mode = "wait"  # 将模式设为等待
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)  # 异步暂停调度器
        # 短暂休眠以帮助确保进行中请求的最终输出在此方法返回前已返回
        # 这些输出在等待空闲完成事件之前就从引擎出来，但涉及额外的异步输出处理任务
        # 注意这不是正确性所必需的，只是从调用者角度来看更直观的事件顺序
        await asyncio.sleep(0.02)  # 异步休眠20毫秒

    async def resume_generation(self) -> None:  # 异步恢复生成方法
        """Resume generation after :meth:`pause_generation`."""
        await self.engine_core.resume_scheduler_async()  # 异步恢复调度器

    async def is_paused(self) -> bool:  # 异步检查引擎是否已暂停
        """Return whether the engine is currently paused."""
        return await self.engine_core.is_scheduler_paused_async()  # 异步查询调度器暂停状态

    async def encode(  # 异步编码方法，用于池化/嵌入任务
        self,  # 实例自身引用
        prompt: PromptType | ProcessorInputs,  # 提示类型或处理器输入
        pooling_params: PoolingParams,  # 池化参数
        request_id: str,  # 请求唯一标识符
        lora_request: LoRARequest | None = None,  # LoRA请求，默认为None
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息，默认为None
        priority: int = 0,  # 请求优先级，默认为0
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数字典，默认为None
        reasoning_ended: bool | None = None,  # 推理是否结束标志，默认为None
    ) -> AsyncGenerator[PoolingRequestOutput, None]:  # 返回池化请求输出的异步生成器
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        q: RequestOutputCollector | None = None  # 初始化请求输出收集器为None
        try:  # 尝试处理编码请求
            q = await self.add_request(  # 异步添加请求
                request_id,  # 传入请求ID
                prompt,  # 传入提示
                pooling_params,  # 传入池化参数
                lora_request=lora_request,  # 传入LoRA请求
                tokenization_kwargs=tokenization_kwargs,  # 传入分词参数
                trace_headers=trace_headers,  # 传入追踪头
                priority=priority,  # 传入优先级
                reasoning_ended=reasoning_ended,  # 传入推理结束标志
            )  # 结束添加请求

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False  # 初始化完成标志为False
            while not finished:  # 当请求未完成时循环
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()  # 优先无等待获取，否则异步等待获取
                assert isinstance(out, PoolingRequestOutput)  # 断言输出为池化请求输出类型
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished  # 更新完成标志
                yield out  # 向调用者yield池化输出结果

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:  # 捕获取消异常（客户端断开连接）
            if q is not None:  # 如果队列已创建
                await self.abort(q.request_id, internal=True)  # 异步中止请求
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s aborted.", request_id)  # 记录请求中止日志
            raise  # 重新抛出异常

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:  # 捕获引擎死亡错误
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s failed (engine dead).", request_id)  # 记录引擎死亡日志
            raise  # 重新抛出异常

        # Request validation error.
        except ValueError:  # 捕获请求验证错误
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s failed (bad request).", request_id)  # 记录错误请求日志
            raise  # 重新抛出异常

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:  # 捕获其他意外异常
            if q is not None:  # 如果队列已创建
                await self.abort(q.request_id, internal=True)  # 异步中止请求
            if self.log_requests:  # 如果启用请求日志
                logger.info("Request %s failed.", request_id)  # 记录请求失败日志
            raise EngineGenerateError() from e  # 抛出引擎生成错误，保留原始异常链
        finally:  # 最终清理阶段
            if q is not None:  # 如果队列已创建
                q.close()  # 关闭输出收集器

    @property  # 属性装饰器
    def tokenizer(self) -> TokenizerLike | None:  # 获取分词器属性
        return self.renderer.tokenizer  # 返回渲染器的分词器

    def get_tokenizer(self) -> TokenizerLike:  # 获取分词器方法
        return self.renderer.get_tokenizer()  # 从渲染器获取分词器

    async def is_tracing_enabled(self) -> bool:  # 异步检查追踪是否启用
        return self.observability_config.otlp_traces_endpoint is not None  # 返回追踪端点是否已配置

    async def do_log_stats(self) -> None:  # 异步记录统计日志
        if self.logger_manager:  # 如果日志管理器存在
            self.logger_manager.log()  # 执行日志记录

    async def check_health(self) -> None:  # 异步健康检查方法
        logger.debug("Called check_health.")  # 记录调试日志
        if self.errored:  # 如果引擎已出错
            raise self.dead_error  # 抛出死亡错误

    async def start_profile(self, profile_prefix: str | None = None) -> None:  # 异步启动性能分析
        coros = [self.engine_core.profile_async(True, profile_prefix)]  # 创建引擎核心分析协程列表
        if self.profiler is not None:  # 如果前端分析器存在
            coros.append(asyncio.to_thread(self.profiler.start))  # 在线程中启动前端分析器
        await asyncio.gather(*coros)  # 并发执行所有分析协程

    async def stop_profile(self) -> None:  # 异步停止性能分析
        """停止性能分析器。"""
        coros = [self.engine_core.profile_async(False)]  # 创建引擎核心停止分析协程列表
        if self.profiler is not None:  # 如果前端分析器存在
            coros.append(asyncio.to_thread(self.profiler.stop))  # 在线程中停止前端分析器
        await asyncio.gather(*coros)  # 并发执行所有停止分析协程

    async def reset_mm_cache(self) -> None:  # 异步重置多模态缓存
        """重置多模态缓存。"""
        self.renderer.clear_mm_cache()  # 清除渲染器的多模态缓存
        await self.engine_core.reset_mm_cache_async()  # 异步重置引擎核心的多模态缓存

    async def reset_prefix_cache(  # 异步重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 是否重置运行中请求和连接器
    ) -> bool:  # 返回是否成功
        """重置前缀缓存。"""
        return await self.engine_core.reset_prefix_cache_async(  # 异步重置引擎核心的前缀缓存
            reset_running_requests, reset_connector  # 传入重置参数
        )  # 返回重置结果

    async def reset_encoder_cache(self) -> None:  # 异步重置编码器缓存
        """重置编码器缓存。"""
        await self.engine_core.reset_encoder_cache_async()  # 异步重置引擎核心的编码器缓存

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:  # 异步进入休眠模式
        """将引擎置于休眠状态以节省资源。"""
        await self.engine_core.sleep_async(level, mode)  # 异步通知引擎核心进入休眠

        if self.logger_manager is not None:  # 如果日志管理器存在
            self.logger_manager.record_sleep_state(1, level)  # 记录休眠状态

    async def wake_up(self, tags: list[str] | None = None) -> None:  # 异步唤醒引擎
        """将引擎从休眠状态唤醒。"""
        await self.engine_core.wake_up_async(tags)  # 异步通知引擎核心唤醒

        if self.logger_manager is not None:  # 如果日志管理器存在
            self.logger_manager.record_sleep_state(0, 0)  # 记录唤醒状态

    async def is_sleeping(self) -> bool:  # 异步检查是否在休眠
        """检查引擎是否处于休眠状态。"""
        return await self.engine_core.is_sleeping_async()  # 异步查询引擎核心休眠状态

    async def add_lora(self, lora_request: LoRARequest) -> bool:  # 异步添加LoRA适配器
        """将新的LoRA适配器加载到引擎中供后续请求使用。"""
        return await self.engine_core.add_lora_async(lora_request)  # 异步添加LoRA到引擎核心

    async def remove_lora(self, lora_id: int) -> bool:  # 异步移除LoRA适配器
        """移除已加载的LoRA适配器。"""
        return await self.engine_core.remove_lora_async(lora_id)  # 异步从引擎核心移除LoRA

    async def list_loras(self) -> set[int]:  # 异步列出所有LoRA适配器
        """列出所有已注册的适配器。"""
        return await self.engine_core.list_loras_async()  # 异步获取引擎核心中的LoRA列表

    async def pin_lora(self, lora_id: int) -> bool:  # 异步固定LoRA适配器
        """防止适配器被驱逐。"""
        return await self.engine_core.pin_lora_async(lora_id)  # 异步在引擎核心中固定LoRA

    async def collective_rpc(  # 异步集合RPC调用方法
        self,  # 实例自身引用
        method: str,  # RPC方法名
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数元组
        kwargs: dict | None = None,  # 关键字参数字典
    ):  # 方法签名结束
        """
        对给定路径执行集合RPC调用。
        """
        return await self.engine_core.collective_rpc_async(  # 异步执行引擎核心的集合RPC
            method, timeout, args, kwargs  # 传入方法名、超时、参数
        )  # 返回RPC结果

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):  # 异步等待请求排空
        """等待所有请求被排空。"""
        start_time = time.time()  # 记录开始时间
        while time.time() - start_time < drain_timeout:  # 在超时时间内循环检查
            if not self.engine_core.dp_engines_running():  # 如果数据并行引擎已空闲
                logger.info("Engines are idle, requests have been drained")  # 记录引擎空闲日志
                return  # 返回，排空完成

            logger.info("Engines are still running, waiting for requests to drain...")  # 记录等待排空日志
            await asyncio.sleep(1)  # 异步等待1秒后再次检查

        raise TimeoutError(  # 超时则抛出超时错误
            f"Timeout reached after {drain_timeout} seconds "  # 超时时间信息
            "waiting for requests to drain."  # 等待请求排空超时
        )  # 结束错误消息

    async def scale_elastic_ep(  # 异步弹性EP扩缩容方法
        self, new_data_parallel_size: int, drain_timeout: int = 300  # 新的数据并行大小和排空超时时间
    ):  # 方法签名结束
        """
        通过添加或移除引擎核心来扩大或缩小数据并行大小。
        Args:
            new_data_parallel_size: 新的数据并行工作进程数量
            drain_timeout: 等待请求排空的最大时间（秒）
        """
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size  # 获取当前数据并行大小
        if old_data_parallel_size == new_data_parallel_size:  # 如果新旧大小相同
            logger.info(  # 记录信息日志
                "Data parallel size is already %s, skipping scale",  # 数据并行大小已经是目标值，跳过扩缩容
                new_data_parallel_size,  # 传入目标大小
            )  # 结束日志记录
            return  # 直接返回

        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:  # 如果设置了排空请求环境变量
            logger.info(  # 记录信息日志
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "  # 排空请求环境变量已设置
                "waiting for requests to drain before scaling"  # 在扩缩容前等待请求排空
            )  # 结束日志记录
            await self.wait_for_requests_to_drain(drain_timeout)  # 异步等待请求排空

        # 重新创建统计日志记录器
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:  # 如果是扩容且启用了统计
            # TODO(rob): 与Ray团队讨论后修复此问题
            # 这会重置所有prometheus指标，因为我们在初始化时会取消注册
            # 需要更好地理解这里的预期行为
            self.logger_manager = StatLoggerManager(  # 创建新的统计日志管理器
                vllm_config=self.vllm_config,  # 传入vLLM配置
                engine_idxs=list(range(new_data_parallel_size)),  # 传入新的引擎索引列表
                custom_stat_loggers=None,  # 不使用自定义日志器
            )  # 结束统计日志管理器创建
            # 更新可变引用，使output_handler能获取新的日志器
            # 而不会通过self创建循环引用
            if hasattr(self, "_logger_ref"):  # 如果日志引用存在
                self._logger_ref[0] = self.logger_manager  # 更新日志管理器引用
            self.logger_manager.log_engine_initialized()  # 记录引擎已初始化的日志

        set_scaling_elastic_ep(True)  # 设置弹性EP扩缩容标志为True
        try:  # 尝试执行扩缩容
            await self.engine_core.scale_elastic_ep(new_data_parallel_size)  # 异步执行引擎核心的弹性EP扩缩容
            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size  # 更新配置中的数据并行大小
        finally:  # 无论成功失败都执行
            set_scaling_elastic_ep(False)  # 重置弹性EP扩缩容标志为False

    @property  # 属性装饰器
    def is_running(self) -> bool:  # 检查引擎是否正在运行
        """检查引擎是否正在运行。"""
        # 在循环启动前为None
        return self.output_handler is None or not self.output_handler.done()  # 输出处理器为空或未完成则认为在运行

    @property  # 属性装饰器
    def is_stopped(self) -> bool:  # 检查引擎是否已停止
        """检查引擎是否已停止。"""
        return self.errored  # 返回是否出错

    @property  # 属性装饰器
    def errored(self) -> bool:  # 检查引擎是否出错
        """检查引擎是否处于错误状态。"""
        return self.engine_core.resources.engine_dead or not self.is_running  # 引擎已死亡或未在运行

    @property  # 属性装饰器
    def dead_error(self) -> BaseException:  # 获取引擎死亡错误实例
        """获取引擎死亡错误。"""
        return EngineDeadError()  # 返回新的引擎死亡错误实例

    async def init_weight_transfer_engine(  # 异步初始化权重传输引擎
        self, request: WeightTransferInitRequest  # 权重传输初始化请求
    ) -> None:  # 返回None
        """
        为强化学习训练初始化权重传输。

        Args:
            request: 包含后端特定信息的权重传输初始化请求
        """
        from vllm.distributed.weight_transfer.base import (  # 延迟导入权重传输基础模块
            WeightTransferInitRequest,  # 权重传输初始化请求类
        )  # 结束延迟导入

        if isinstance(request, WeightTransferInitRequest):  # 如果请求是权重传输初始化请求类型
            init_info_dict = request.init_info  # 获取初始化信息字典
        else:  # 否则类型不匹配
            raise TypeError(f"Expected WeightTransferInitRequest, got {type(request)}")  # 抛出类型错误

        await self.collective_rpc(  # 异步执行集合RPC调用
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}  # 调用权重传输引擎初始化方法
        )  # 结束RPC调用

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:  # 异步更新权重方法
        """
        用于强化学习训练的批量权重更新。

        Args:
            request: 包含后端特定更新信息的权重更新请求
        """

        if isinstance(request, WeightTransferUpdateRequest):  # 如果请求是权重传输更新请求类型
            update_info_dict = request.update_info  # 获取更新信息字典
        else:  # 否则类型不匹配
            raise TypeError(  # 抛出类型错误
                f"Expected WeightTransferUpdateRequest, got {type(request)}"  # 期望WeightTransferUpdateRequest类型
            )  # 结束错误消息

        await self.collective_rpc(  # 异步执行集合RPC调用
            "update_weights", kwargs={"update_info": update_info_dict}  # 调用权重更新方法
        )  # 结束RPC调用
