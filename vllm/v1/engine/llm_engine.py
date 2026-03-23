# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time  # 导入时间模块
from collections.abc import Callable, Mapping  # 导入可调用对象和映射类型
from copy import copy  # 导入浅拷贝函数
from typing import Any  # 导入Any类型注解

import torch.nn as nn  # 导入PyTorch神经网络模块
from typing_extensions import TypeVar  # 导入扩展的TypeVar类型变量

import vllm.envs as envs  # 导入vLLM环境变量配置
from vllm.config import ParallelConfig, VllmConfig  # 导入并行配置和vLLM配置类
from vllm.distributed import stateless_destroy_torch_distributed_process_group  # 导入销毁分布式进程组的函数
from vllm.distributed.parallel_state import get_dp_group  # 导入获取数据并行组的函数
from vllm.engine.arg_utils import EngineArgs  # 导入引擎参数工具类
from vllm.inputs import ProcessorInputs, PromptType  # 导入处理器输入和提示类型
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.lora.request import LoRARequest  # 导入LoRA请求类
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 导入多模态注册表
from vllm.outputs import PoolingRequestOutput, RequestOutput  # 导入池化请求输出和请求输出类
from vllm.plugins.io_processors import get_io_processor  # 导入获取IO处理器的函数
from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.renderers import renderer_from_config  # 导入根据配置创建渲染器的函数
from vllm.renderers.inputs.preprocess import extract_prompt_components  # 导入提取提示组件的函数
from vllm.sampling_params import SamplingParams  # 导入采样参数类
from vllm.tasks import SupportedTask  # 导入支持的任务类型
from vllm.tokenizers import TokenizerLike  # 导入分词器类型接口
from vllm.tracing import init_tracer  # 导入追踪初始化函数
from vllm.usage.usage_lib import UsageContext  # 导入使用上下文枚举
from vllm.v1.engine import EngineCoreRequest, PauseMode  # 导入引擎核心请求和暂停模式
from vllm.v1.engine.core_client import EngineCoreClient  # 导入引擎核心客户端
from vllm.v1.engine.input_processor import InputProcessor  # 导入输入处理器
from vllm.v1.engine.output_processor import OutputProcessor  # 导入输出处理器
from vllm.v1.engine.parallel_sampling import ParentRequest  # 导入并行采样父请求类
from vllm.v1.executor import Executor  # 导入执行器类
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager  # 导入统计日志工厂和管理器
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot  # 导入指标类和获取指标快照函数
from vllm.v1.metrics.stats import IterationStats  # 导入迭代统计类
from vllm.v1.utils import record_function_or_nullcontext  # 导入记录函数或空上下文管理器
from vllm.v1.worker.worker_base import WorkerBase  # 导入工作进程基类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

_R = TypeVar("_R", default=Any)  # 定义泛型类型变量_R，默认为Any


# [中文注释] LLMEngine — vLLM 的同步推理引擎前端，用于离线批量推理（LLM 类使用）。
#   与 AsyncLLM 共享 InputProcessor/OutputProcessor/EngineCoreClient 三件套，
#   但使用同步 API（SyncMPClient 或 InprocClient）。
#   核心方法 step() 实现同步的 拉取输出→处理→返回 循环。
#   支持 DP 模式：通过 dp_group 进行 has_unfinished_requests 的 all-reduce 同步，
#   当本地无请求但全局有请求时执行 dummy_batch 保持 DP 同步。
class LLMEngine:  # LLM引擎类定义
    """Legacy LLMEngine for backwards compatibility."""
    """旧版LLM引擎，用于向后兼容。"""

    def __init__(  # 构造函数定义
        self,  # 实例自身引用
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计信息
        aggregate_engine_logging: bool = False,  # 是否聚合引擎日志
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用上下文
        stat_loggers: list[StatLoggerFactory] | None = None,  # 统计日志工厂列表
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表
        use_cached_outputs: bool = False,  # 是否使用缓存输出
        multiprocess_mode: bool = False,  # 是否启用多进程模式
    ) -> None:  # 返回None
        self.vllm_config = vllm_config  # 保存vLLM配置
        self.model_config = vllm_config.model_config  # 保存模型配置
        self.observability_config = vllm_config.observability_config  # 保存可观测性配置

        tracing_endpoint = self.observability_config.otlp_traces_endpoint  # 获取追踪端点地址
        if tracing_endpoint is not None:  # 如果追踪端点已配置
            init_tracer("vllm.llm_engine", tracing_endpoint)  # 初始化追踪器

        self.log_stats = log_stats  # 保存是否记录统计信息的标志

        parallel_config = vllm_config.parallel_config  # 获取并行配置
        executor_backend = parallel_config.distributed_executor_backend  # 获取分布式执行器后端

        self.external_launcher_dp = (  # 判断是否使用外部启动器的数据并行
            parallel_config.data_parallel_size > 1  # 数据并行大小大于1
            and executor_backend == "external_launcher"  # 且执行器后端为外部启动器
        )  # 外部启动器数据并行标志赋值结束
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        if (  # 如果满足以下条件则初始化数据并行组
            not multiprocess_mode  # 非多进程模式
            and parallel_config.data_parallel_size > 1  # 数据并行大小大于1
            and not self.external_launcher_dp  # 且不使用外部启动器数据并行
        ):  # 条件判断结束
            self.dp_group = parallel_config.stateless_init_dp_group()  # 初始化无状态数据并行组
        else:  # 否则
            self.dp_group = None  # 数据并行组设为None
        self.should_execute_dummy_batch = False  # 初始化是否执行虚拟批次标志为False

        self.renderer = renderer = renderer_from_config(self.vllm_config)  # 根据配置创建渲染器
        self.io_processor = get_io_processor(  # 获取IO处理器
            self.vllm_config,  # 传入vLLM配置
            self.renderer,  # 传入渲染器
            self.model_config.io_processor_plugin,  # 传入IO处理器插件
        )  # IO处理器创建完成

        # Convert TokPrompt --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)  # 创建输入处理器，将提示转换为引擎核心请求

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(  # 创建输出处理器，将引擎核心输出转换为请求输出
            renderer.tokenizer,  # 传入分词器
            log_stats=self.log_stats,  # 传入统计日志标志
            stream_interval=self.vllm_config.scheduler_config.stream_interval,  # 传入流式输出间隔
            tracing_enabled=tracing_endpoint is not None,  # 传入追踪是否启用
        )  # 输出处理器创建完成

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(  # 创建引擎核心客户端
            multiprocess_mode=multiprocess_mode,  # 传入多进程模式标志
            asyncio_mode=False,  # 同步模式，不使用asyncio
            vllm_config=vllm_config,  # 传入vLLM配置
            executor_class=executor_class,  # 传入执行器类
            log_stats=self.log_stats,  # 传入统计日志标志
        )  # 引擎核心客户端创建完成

        self.logger_manager: StatLoggerManager | None = None  # 初始化统计日志管理器为None
        if self.log_stats:  # 如果启用了统计日志
            self.logger_manager = StatLoggerManager(  # 创建统计日志管理器
                vllm_config=vllm_config,  # 传入vLLM配置
                custom_stat_loggers=stat_loggers,  # 传入自定义统计日志器
                enable_default_loggers=log_stats,  # 传入是否启用默认日志器
                aggregate_engine_logging=aggregate_engine_logging,  # 传入是否聚合引擎日志
            )  # 统计日志管理器创建完成
            self.logger_manager.log_engine_initialized()  # 记录引擎已初始化的日志

        if not multiprocess_mode:  # 如果不是多进程模式
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # 为v0兼容性设置模型执行器引用  # type: ignore

        if self.external_launcher_dp:  # 如果使用外部启动器数据并行
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            self.dp_group = get_dp_group().cpu_group  # 复用现有的数据并行组用于数据通信

        # Don't keep the dummy data in memory
        self.reset_mm_cache()  # 重置多模态缓存，释放虚拟数据内存

    @classmethod  # 类方法装饰器
    def from_vllm_config(  # 从vLLM配置创建引擎的工厂方法
        cls,  # 类自身引用
        vllm_config: VllmConfig,  # vLLM配置对象
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用上下文
        stat_loggers: list[StatLoggerFactory] | None = None,  # 统计日志工厂列表
        disable_log_stats: bool = False,  # 是否禁用统计日志
    ) -> "LLMEngine":  # 返回LLMEngine实例
        """从vLLM配置创建LLM引擎实例。"""
        return cls(  # 调用构造函数创建实例
            vllm_config=vllm_config,  # 传入vLLM配置
            executor_class=Executor.get_class(vllm_config),  # 根据配置获取执行器类
            log_stats=(not disable_log_stats),  # 设置是否记录统计信息
            usage_context=usage_context,  # 传入使用上下文
            stat_loggers=stat_loggers,  # 传入统计日志器
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,  # 传入多进程模式设置
        )  # 创建实例完成

    @classmethod  # 类方法装饰器
    def from_engine_args(  # 从引擎参数创建引擎的工厂方法
        cls,  # 类自身引用
        engine_args: EngineArgs,  # 引擎参数对象
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用上下文
        stat_loggers: list[StatLoggerFactory] | None = None,  # 统计日志工厂列表
        enable_multiprocessing: bool = False,  # 是否启用多进程
    ) -> "LLMEngine":  # 返回LLMEngine实例
        """Creates an LLM engine from the engine arguments."""
        """从引擎参数创建LLM引擎实例。"""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)  # 根据使用上下文创建引擎配置
        executor_class = Executor.get_class(vllm_config)  # 根据配置获取执行器类

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:  # 如果环境变量启用了V1多进程
            logger.debug("Enabling multiprocessing for LLMEngine.")  # 记录调试日志
            enable_multiprocessing = True  # 设置启用多进程

        # Create the LLMEngine.
        return cls(  # 调用构造函数创建LLM引擎实例
            vllm_config=vllm_config,  # 传入vLLM配置
            executor_class=executor_class,  # 传入执行器类
            log_stats=not engine_args.disable_log_stats,  # 设置是否记录统计信息
            usage_context=usage_context,  # 传入使用上下文
            stat_loggers=stat_loggers,  # 传入统计日志器
            multiprocess_mode=enable_multiprocessing,  # 传入多进程模式设置
        )  # 创建实例完成

    def get_num_unfinished_requests(self) -> int:  # 获取未完成请求数量
        """获取未完成的请求数量。"""
        return self.output_processor.get_num_unfinished_requests()  # 从输出处理器获取未完成请求数

    def has_unfinished_requests(self) -> bool:  # 检查是否有未完成的请求
        """检查是否存在未完成的请求。"""
        has_unfinished = self.output_processor.has_unfinished_requests()  # 检查输出处理器中是否有未完成请求
        if self.dp_group is None:  # 如果没有数据并行组
            return has_unfinished or self.engine_core.dp_engines_running()  # 返回本地未完成或数据并行引擎运行状态
        return self.has_unfinished_requests_dp(has_unfinished)  # 使用数据并行方式检查未完成请求

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:  # 数据并行模式下检查未完成请求
        """数据并行模式下检查所有节点是否有未完成请求。"""
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(  # 聚合所有数据并行节点的未完成状态
            self.dp_group, has_unfinished  # 传入数据并行组和本地未完成状态
        )  # 聚合结果获取完成
        if not has_unfinished and aggregated_has_unfinished:  # 如果本地无未完成但全局有未完成
            self.should_execute_dummy_batch = True  # 标记需要执行虚拟批次以保持数据并行同步
        return aggregated_has_unfinished  # 返回聚合后的未完成状态

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型
        """获取引擎支持的任务类型元组。"""
        if not hasattr(self, "_supported_tasks"):  # 如果尚未缓存支持的任务
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()  # 从引擎核心获取并缓存支持的任务

        return self._supported_tasks  # 返回缓存的支持任务

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:  # 中止请求方法
        """Remove request_ids from EngineCore and Detokenizer."""
        """从引擎核心和解码器中移除指定的请求。"""

        request_ids = self.output_processor.abort_requests(request_ids, internal)  # 从输出处理器中中止请求并获取需要中止的ID
        self.engine_core.abort_requests(request_ids)  # 从引擎核心中中止请求

    def add_request(  # 添加请求方法
        self,  # 实例自身引用
        request_id: str,  # 请求唯一标识符
        prompt: EngineCoreRequest | PromptType | ProcessorInputs,  # 提示输入，支持多种类型
        params: SamplingParams | PoolingParams,  # 采样参数或池化参数
        arrival_time: float | None = None,  # 请求到达时间
        lora_request: LoRARequest | None = None,  # LoRA适配器请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息
        priority: int = 0,  # 请求优先级
        prompt_text: str | None = None,  # 提示文本
    ) -> str:  # 返回请求ID字符串
        """添加一个新的推理请求到引擎中。"""
        # Validate the request_id type.
        if not isinstance(request_id, str):  # 如果请求ID不是字符串类型
            raise TypeError(f"request_id must be a string, got {type(request_id)}")  # 抛出类型错误

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):  # 如果提示已经是引擎核心请求类型
            logger.warning_once(  # 记录一次性警告日志
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "  # 警告消息第一行
                "is deprecated and will be removed in v0.18. You should instead pass "  # 警告消息第二行
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."  # 警告消息第三行
            )  # 警告日志记录完成

            request = prompt  # 直接使用传入的引擎核心请求
            if request_id != request.request_id:  # 如果传入的请求ID与请求对象的ID不匹配
                logger.warning_once(  # 记录一次性警告日志
                    "LLMEngine.add_request() was passed a request_id parameter that "  # 警告消息第一行
                    "does not match the EngineCoreRequest.request_id attribute. The "  # 警告消息第二行
                    "latter will be used, and the former will be ignored."  # 警告消息第三行
                )  # 警告日志记录完成
        else:  # 否则需要处理原始输入
            request = self.input_processor.process_inputs(  # 通过输入处理器处理原始输入
                request_id,  # 传入请求ID
                prompt,  # 传入提示
                params,  # 传入参数
                supported_tasks=self.get_supported_tasks(),  # 传入支持的任务类型
                arrival_time=arrival_time,  # 传入到达时间
                lora_request=lora_request,  # 传入LoRA请求
                tokenization_kwargs=tokenization_kwargs,  # 传入分词器参数
                trace_headers=trace_headers,  # 传入追踪头
                priority=priority,  # 传入优先级
            )  # 输入处理完成
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)  # 提取提示文本组件

        self.input_processor.assign_request_id(request)  # 分配最终的请求ID

        req_id = request.request_id  # 获取分配后的请求ID

        # Use cloned params that may have been updated in process_inputs()
        params = request.params  # 使用可能在处理输入时更新过的克隆参数

        n = params.n if isinstance(params, SamplingParams) else 1  # 获取并行采样数n，池化参数默认为1

        if n == 1:  # 如果只需要一个采样结果
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)  # 创建新的请求状态并加入队列
            # Add the request to EngineCore.
            self.engine_core.add_request(request)  # 将请求添加到引擎核心
            return req_id  # 返回请求ID

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request)  # 创建父请求对象用于并行采样
        for idx in range(n):  # 遍历每个子请求索引
            request_id, child_params = parent_req.get_child_info(idx)  # 获取子请求的ID和参数
            child_request = request if idx == n - 1 else copy(request)  # 最后一个子请求复用原始请求，其余浅拷贝
            child_request.request_id = request_id  # 设置子请求的ID
            child_request.sampling_params = child_params  # 设置子请求的采样参数

            # Make a new RequestState and queue.
            self.output_processor.add_request(  # 为子请求创建请求状态并加入队列
                child_request, prompt_text, parent_req, idx  # 传入子请求、提示文本、父请求和索引
            )  # 添加子请求完成
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)  # 将子请求添加到引擎核心

        return req_id  # 返回父请求ID

    # [中文注释] 同步推理主循环的单步执行：
    #   1. 从 EngineCore 获取输出（同步阻塞）
    #   2. OutputProcessor 处理输出（detokenize + logprobs + 构造 RequestOutput）
    #   3. Abort 因 stop string 提前终止的请求
    #   4. 记录统计并返回 RequestOutput 列表
    #   DP 模式下若需要执行 dummy_batch 则跳过正常流程
    def step(self) -> list[RequestOutput | PoolingRequestOutput]:  # 执行一步同步推理循环
        """执行一步同步推理：获取输出、处理输出、中止已完成请求、记录统计。"""
        if self.should_execute_dummy_batch:  # 如果需要执行虚拟批次（数据并行同步用）
            self.should_execute_dummy_batch = False  # 重置虚拟批次标志
            self.engine_core.execute_dummy_batch()  # 执行虚拟批次
            return []  # 返回空列表

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):  # 记录获取输出的性能追踪
            outputs = self.engine_core.get_output()  # 从引擎核心同步获取输出

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):  # 记录处理输出的性能追踪
            iteration_stats = IterationStats() if self.log_stats else None  # 如果启用统计则创建迭代统计对象
            processed_outputs = self.output_processor.process_outputs(  # 处理引擎核心输出
                outputs.outputs,  # 传入输出数据
                engine_core_timestamp=outputs.timestamp,  # 传入引擎核心时间戳
                iteration_stats=iteration_stats,  # 传入迭代统计对象
            )  # 输出处理完成
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)  # 更新调度器统计信息

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):  # 记录中止请求的性能追踪
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)  # 中止因停止字符串而结束的请求

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):  # 记录统计的性能追踪
            if (  # 如果满足记录统计的条件
                self.logger_manager is not None  # 日志管理器已初始化
                and outputs.scheduler_stats is not None  # 调度器统计不为空
                and len(outputs.outputs) > 0  # 输出列表不为空
            ):  # 条件判断结束
                self.logger_manager.record(  # 记录统计数据
                    scheduler_stats=outputs.scheduler_stats,  # 传入调度器统计
                    iteration_stats=iteration_stats,  # 传入迭代统计
                    mm_cache_stats=self.renderer.stat_mm_cache(),  # 传入多模态缓存统计
                )  # 统计记录完成
                self.do_log_stats_with_interval()  # 按时间间隔输出统计日志

        return processed_outputs.request_outputs  # 返回处理后的请求输出列表

    def start_profile(self, profile_prefix: str | None = None):  # 开始性能分析
        """开始性能分析。"""
        self.engine_core.profile(True, profile_prefix)  # 通知引擎核心开始性能分析

    def stop_profile(self):  # 停止性能分析
        """停止性能分析。"""
        self.engine_core.profile(False)  # 通知引擎核心停止性能分析

    def reset_mm_cache(self):  # 重置多模态缓存
        """重置多模态缓存，清除渲染器和引擎核心的缓存。"""
        self.renderer.clear_mm_cache()  # 清除渲染器的多模态缓存
        self.engine_core.reset_mm_cache()  # 重置引擎核心的多模态缓存

    def reset_prefix_cache(  # 重置前缀缓存方法
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 是否重置运行中请求和连接器
    ) -> bool:  # 返回是否成功
        """重置前缀缓存。"""
        return self.engine_core.reset_prefix_cache(  # 调用引擎核心重置前缀缓存
            reset_running_requests, reset_connector  # 传入重置参数
        )  # 返回重置结果

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        """重置编码器缓存以使所有缓存的编码器输出失效。"""
        self.engine_core.reset_encoder_cache()  # 调用引擎核心重置编码器缓存

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):  # 使引擎进入睡眠状态
        """使引擎进入睡眠状态，释放GPU资源。"""
        self.engine_core.sleep(level, mode)  # 调用引擎核心进入睡眠

        if self.logger_manager is not None:  # 如果日志管理器已初始化
            self.logger_manager.record_sleep_state(1, level)  # 记录睡眠状态

    def wake_up(self, tags: list[str] | None = None):  # 唤醒引擎
        """唤醒引擎，恢复GPU资源。"""
        self.engine_core.wake_up(tags)  # 调用引擎核心唤醒

        if self.logger_manager is not None:  # 如果日志管理器已初始化
            self.logger_manager.record_sleep_state(0, 0)  # 记录唤醒状态

    def is_sleeping(self) -> bool:  # 检查引擎是否在睡眠状态
        """检查引擎是否处于睡眠状态。"""
        return self.engine_core.is_sleeping()  # 返回引擎核心的睡眠状态

    def get_metrics(self) -> list[Metric]:  # 获取指标列表
        """获取当前的性能指标快照。"""
        assert self.log_stats, "Stat logging disabled"  # 断言统计日志已启用
        return get_metrics_snapshot()  # 返回指标快照

    @property  # 属性装饰器
    def tokenizer(self) -> TokenizerLike | None:  # 分词器属性
        """获取分词器实例，可能为None。"""
        return self.renderer.tokenizer  # 从渲染器获取分词器

    def get_tokenizer(self) -> TokenizerLike:  # 获取分词器方法
        """获取分词器实例。"""
        return self.renderer.get_tokenizer()  # 从渲染器获取分词器

    def do_log_stats(self) -> None:  # 输出统计日志
        """Log stats if logging is enabled."""
        """如果启用了日志记录则输出统计信息。"""
        if self.logger_manager:  # 如果日志管理器存在
            self.logger_manager.log()  # 输出日志

    def do_log_stats_with_interval(self) -> None:  # 按时间间隔输出统计日志
        """Log stats when the time interval has passed."""
        """当时间间隔已过时输出统计日志。"""
        now = time.time()  # 获取当前时间
        if not hasattr(self, "_last_log_time"):  # 如果尚未记录上次日志时间
            self._last_log_time = now  # 初始化上次日志时间为当前时间
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:  # 如果距上次日志时间已超过配置间隔
            self.do_log_stats()  # 输出统计日志
            self._last_log_time = now  # 更新上次日志时间为当前时间

    def add_lora(self, lora_request: LoRARequest) -> bool:  # 添加LoRA适配器
        """Load a new LoRA adapter into the engine for future requests."""
        """加载新的LoRA适配器到引擎中供后续请求使用。"""
        return self.engine_core.add_lora(lora_request)  # 调用引擎核心添加LoRA

    def remove_lora(self, lora_id: int) -> bool:  # 移除LoRA适配器
        """Remove an already loaded LoRA adapter."""
        """移除已加载的LoRA适配器。"""
        return self.engine_core.remove_lora(lora_id)  # 调用引擎核心移除LoRA

    def list_loras(self) -> set[int]:  # 列出所有LoRA适配器
        """List all registered adapters."""
        """列出所有已注册的适配器。"""
        return self.engine_core.list_loras()  # 调用引擎核心列出LoRA

    def pin_lora(self, lora_id: int) -> bool:  # 固定LoRA适配器防止被淘汰
        """Prevent an adapter from being evicted."""
        """防止适配器被淘汰。"""
        return self.engine_core.pin_lora(lora_id)  # 调用引擎核心固定LoRA

    def collective_rpc(  # 集体RPC调用方法
        self,  # 实例自身引用
        method: str | Callable[[WorkerBase], _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回结果列表
        """对所有工作进程执行集体RPC调用。"""
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)  # 调用引擎核心执行集体RPC

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:  # 对模型应用函数
        """对模型应用指定的函数。"""
        return self.collective_rpc("apply_model", args=(func,))  # 通过集体RPC调用apply_model

    def __del__(self):  # 析构函数
        """析构函数，清理数据并行进程组资源。"""
        dp_group = getattr(self, "dp_group", None)  # 安全获取数据并行组属性
        if dp_group is not None and not self.external_launcher_dp:  # 如果数据并行组存在且非外部启动器
            stateless_destroy_torch_distributed_process_group(dp_group)  # 销毁分布式进程组
