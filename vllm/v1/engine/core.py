# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os  # 导入操作系统接口模块
import queue  # 导入队列模块
import signal  # 导入信号处理模块
import threading  # 导入线程模块
import time  # 导入时间模块
from collections import defaultdict, deque  # 从collections导入默认字典和双端队列
from collections.abc import Callable, Generator  # 导入可调用和生成器抽象基类
from concurrent.futures import Future  # 导入Future异步结果对象
from contextlib import ExitStack, contextmanager  # 导入上下文管理工具
from enum import IntEnum  # 导入整数枚举基类
from functools import partial  # 导入偏函数工具
from inspect import isclass, signature  # 导入类检查和签名检查工具
from logging import DEBUG  # 导入DEBUG日志级别常量
from typing import Any, TypeVar, cast  # 导入类型注解工具

import msgspec  # 导入高性能序列化库msgspec
import zmq  # 导入ZeroMQ消息队列库

import vllm.envs as envs  # 导入vLLM环境变量配置模块
from vllm.config import ParallelConfig, VllmConfig  # 导入并行配置和vLLM配置类
from vllm.distributed import stateless_destroy_torch_distributed_process_group  # 导入销毁分布式进程组函数
from vllm.envs import enable_envs_cache  # 导入启用环境变量缓存函数
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.logging_utils.dump_input import dump_engine_exception  # 导入引擎异常转储函数
from vllm.lora.request import LoRARequest  # 导入LoRA请求类
from vllm.multimodal import MULTIMODAL_REGISTRY  # 导入多模态注册表
from vllm.tasks import POOLING_TASKS, SupportedTask  # 导入池化任务和支持的任务类型
from vllm.tracing import instrument, maybe_init_worker_tracer  # 导入追踪装饰器和工作进程追踪初始化
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value  # 导入配置序列化注册函数
from vllm.utils.gc_utils import (  # 从GC工具模块导入
    freeze_gc_heap,  # 冻结GC堆函数
    maybe_attach_gc_debug_callback,  # 可选的GC调试回调附加函数
)
from vllm.utils.hashing import get_hash_fn_by_name  # 导入按名称获取哈希函数的工具
from vllm.utils.network_utils import make_zmq_socket  # 导入创建ZMQ套接字的工具函数
from vllm.utils.system_utils import decorate_logs, set_process_title  # 导入日志装饰和进程标题设置工具
from vllm.v1.core.kv_cache_utils import (  # 从KV缓存工具模块导入
    BlockHash,  # 块哈希类型
    generate_scheduler_kv_cache_config,  # 生成调度器KV缓存配置函数
    get_kv_cache_configs,  # 获取KV缓存配置函数
    get_request_block_hasher,  # 获取请求块哈希器函数
    init_none_hash,  # 初始化空哈希函数
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface  # 导入暂停状态和调度器接口
from vllm.v1.core.sched.output import SchedulerOutput  # 导入调度器输出类
from vllm.v1.engine import (  # 从引擎模块导入
    EEP_NOTIFICATION_CALL_ID,  # 弹性专家并行通知调用ID
    EEPNotificationType,  # 弹性专家并行通知类型
    EngineCoreOutput,  # 引擎核心输出类
    EngineCoreOutputs,  # 引擎核心输出集合类
    EngineCoreRequest,  # 引擎核心请求类
    EngineCoreRequestType,  # 引擎核心请求类型枚举
    FinishReason,  # 完成原因枚举
    PauseMode,  # 暂停模式枚举
    ReconfigureDistributedRequest,  # 重新配置分布式请求类
    ReconfigureRankType,  # 重新配置rank类型
    UtilityOutput,  # 工具输出类
    UtilityResult,  # 工具结果类
)
from vllm.v1.engine.utils import (  # 从引擎工具模块导入
    EngineHandshakeMetadata,  # 引擎握手元数据类
    EngineZmqAddresses,  # 引擎ZMQ地址类
    SignalCallback,  # 信号回调类
    get_device_indices,  # 获取设备索引函数
)
from vllm.v1.executor import Executor  # 导入执行器类
from vllm.v1.kv_cache_interface import KVCacheConfig  # 导入KV缓存配置接口
from vllm.v1.metrics.stats import SchedulerStats  # 导入调度器统计类
from vllm.v1.outputs import ModelRunnerOutput  # 导入模型运行器输出类
from vllm.v1.request import Request, RequestStatus  # 导入请求和请求状态类
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # 导入Msgpack编解码器
from vllm.v1.structured_output import StructuredOutputManager  # 导入结构化输出管理器
from vllm.v1.utils import compute_iteration_details  # 导入计算迭代详情函数
from vllm.version import __version__ as VLLM_VERSION  # 导入vLLM版本号

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

HANDSHAKE_TIMEOUT_MINS = 5  # 握手超时时间（分钟）

_R = TypeVar("_R")  # 集体RPC的返回类型泛型变量


class EngineCore:
    """vLLM引擎的内部循环核心类。"""

    def __init__(  # 初始化引擎核心
        self,
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计信息
        executor_fail_callback: Callable | None = None,  # 执行器失败回调函数
        include_finished_set: bool = False,  # 是否包含已完成集合
    ):
        # 插件需要在引擎/调度器层级也加载
        from vllm.plugins import load_general_plugins  # 导入通用插件加载函数

        load_general_plugins()  # 加载通用插件

        self.vllm_config = vllm_config  # 保存vLLM配置
        if not vllm_config.parallel_config.data_parallel_rank_local:  # 如果不是本地数据并行rank
            logger.info(  # 记录初始化信息
                "Initializing a V1 LLM engine (v%s) with config: %s",  # 日志格式字符串
                VLLM_VERSION,  # vLLM版本
                vllm_config,  # 配置信息
            )

        self.log_stats = log_stats  # 保存是否记录统计信息的标志

        # 设置模型
        self.model_executor = executor_class(vllm_config)  # 创建模型执行器实例
        if executor_fail_callback is not None:  # 如果提供了失败回调
            self.model_executor.register_failure_callback(executor_fail_callback)  # 注册执行器失败回调

        self.available_gpu_memory_for_kv_cache = -1  # 初始化可用GPU内存为-1（未计算）

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:  # 如果启用了弹性专家并行扩容启动
            self._eep_scale_up_before_kv_init()  # 在KV初始化前执行弹性扩容

        # 设置KV缓存并在性能分析后更新缓存配置
        kv_cache_config = self._initialize_kv_caches(vllm_config)  # 初始化KV缓存
        self.structured_output_manager = StructuredOutputManager(vllm_config)  # 创建结构化输出管理器

        # 设置调度器
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()  # 获取调度器类

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102  # 如果没有KV缓存组
            # 没有KV缓存的编码器模型不支持分块预填充
            # 但SSM模型呢？
            if vllm_config.scheduler_config.enable_chunked_prefill:  # 如果启用了分块预填充
                logger.warning("Disabling chunked prefill for model without KVCache")  # 警告禁用分块预填充
                vllm_config.scheduler_config.enable_chunked_prefill = False  # 禁用分块预填充

        scheduler_block_size = (  # 计算调度器块大小
            vllm_config.cache_config.block_size  # 基础块大小
            * vllm_config.parallel_config.decode_context_parallel_size  # 乘以解码上下文并行度
            * vllm_config.parallel_config.prefill_context_parallel_size  # 乘以预填充上下文并行度
        )

        self.scheduler: SchedulerInterface = Scheduler(  # 创建调度器实例
            vllm_config=vllm_config,  # vLLM配置
            kv_cache_config=kv_cache_config,  # KV缓存配置
            structured_output_manager=self.structured_output_manager,  # 结构化输出管理器
            include_finished_set=include_finished_set,  # 是否包含已完成集合
            log_stats=self.log_stats,  # 是否记录统计
            block_size=scheduler_block_size,  # 块大小
        )
        self.use_spec_decode = vllm_config.speculative_config is not None  # 是否使用推测解码
        if self.scheduler.connector is not None:  # type: ignore  # 如果调度器有连接器
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore  # 初始化KV输出聚合器

        mm_registry = MULTIMODAL_REGISTRY  # 获取多模态注册表
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(  # 从配置创建多模态接收器缓存
            vllm_config  # 传入vLLM配置
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()  # 获取KV连接器
        if kv_connector is not None:  # 如果KV连接器存在
            # 从工作进程收集并存储KV连接器传输握手元数据
            # （在KV缓存注册之后）
            xfer_handshake_metadata = (  # 获取KV连接器握手元数据
                self.model_executor.get_kv_connector_handshake_metadata()  # 从模型执行器获取
            )

            if xfer_handshake_metadata:  # 如果有握手元数据
                # xfer_handshake_metadata是来自工作进程的字典列表
                # 每个字典已经有{tp_rank: metadata}的结构
                # 将所有工作进程字典合并为单个字典
                content: dict[int, Any] = {}  # 初始化合并后的内容字典
                for worker_dict in xfer_handshake_metadata:  # 遍历每个工作进程的字典
                    if worker_dict is not None:  # 如果字典不为空
                        content.update(worker_dict)  # 合并字典内容
                kv_connector.set_xfer_handshake_metadata(content)  # 设置传输握手元数据

        # 设置流水线并行的批次队列
        # 已调度批次的批次队列。这使我们能够异步地调度和执行批次，
        # 流水线并行需要它来消除流水线气泡
        self.batch_queue_size = self.model_executor.max_concurrent_batches  # 获取最大并发批次数
        self.batch_queue: (  # 批次队列类型声明
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None  # 双端队列或None
        ) = None  # 初始化为None
        if self.batch_queue_size > 1:  # 如果最大并发批次数大于1
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)  # 记录批次队列已启用
            self.batch_queue = deque(maxlen=self.batch_queue_size)  # 创建有最大长度限制的双端队列

        self.is_ec_consumer = (  # 判断是否为EC消费者
            vllm_config.ec_transfer_config is None  # 如果没有EC传输配置
            or vllm_config.ec_transfer_config.is_ec_consumer  # 或者配置为EC消费者
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"  # 判断是否为池化模型

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None  # 请求块哈希器初始化为None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:  # 如果启用前缀缓存或有KV连接器
            caching_hash_fn = get_hash_fn_by_name(  # 按名称获取缓存哈希函数
                vllm_config.cache_config.prefix_caching_hash_algo  # 前缀缓存哈希算法名称
            )
            init_none_hash(caching_hash_fn)  # 初始化空哈希

            self.request_block_hasher = get_request_block_hasher(  # 获取请求块哈希器
                scheduler_block_size, caching_hash_fn  # 传入块大小和哈希函数
            )

        self.step_fn = (  # 选择步进函数
            self.step if self.batch_queue is None else self.step_with_batch_queue  # 无批次队列用step，否则用step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling  # 是否启用异步调度

        self.aborts_queue = queue.Queue[list[str]]()  # 创建中止请求队列

        self._idle_state_callbacks: list[Callable] = []  # 空闲状态回调列表

        # 将启动堆标记为静态，使其被GC忽略
        # 减少最老代收集的暂停时间
        freeze_gc_heap()  # 冻结GC堆
        # 如果启用，在静态变量冻结后附加GC调试器
        maybe_attach_gc_debug_callback()  # 可选地附加GC调试回调
        # 启用环境变量缓存（即假设此时之后不再有环境变量覆盖）
        enable_envs_cache()  # 启用环境变量缓存

    @instrument(span_name="Prepare model")  # 追踪装饰器，标记为"准备模型"阶段
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        """初始化KV缓存并返回KV缓存配置。"""
        start = time.time()  # 记录开始时间

        # 获取模型所需的所有KV缓存规格
        kv_cache_specs = self.model_executor.get_kv_cache_specs()  # 从模型执行器获取KV缓存规格

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)  # 检查是否有KV缓存
        if has_kv_cache:  # 如果有KV缓存
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:  # 如果启用了弹性扩容启动
                # 注意(yongji): 应该在_eep_scale_up_before_kv_init期间已经设置
                assert self.available_gpu_memory_for_kv_cache > 0  # 断言可用GPU内存大于0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(  # 为每个规格复制可用内存
                    kv_cache_specs  # KV缓存规格列表
                )
            else:  # 否则正常分析
                # 分析模型的峰值内存使用量，以确定可为KV缓存分配多少内存
                available_gpu_memory = self.model_executor.determine_available_memory()  # 确定可用内存
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]  # 保存第一个设备的可用内存
        else:  # 如果没有KV缓存
            # 无注意力机制的模型不需要KV缓存内存
            available_gpu_memory = [0] * len(kv_cache_specs)  # 所有可用内存设为0

        assert len(kv_cache_specs) == len(available_gpu_memory)  # 断言规格数与可用内存数一致

        # 在KV缓存配置之前跟踪max_model_len以检测自动调整变化
        max_model_len_before = vllm_config.model_config.max_model_len  # 记录调整前的最大模型长度

        kv_cache_configs = get_kv_cache_configs(  # 获取KV缓存配置
            vllm_config, kv_cache_specs, available_gpu_memory  # 传入配置、规格和可用内存
        )

        # 如果自动调整减小了max_model_len，将新值同步到工作进程
        # 这是必要的，因为工作进程在内存分析之前已生成，缓存了原始（较大的）max_model_len
        max_model_len_after = vllm_config.model_config.max_model_len  # 获取调整后的最大模型长度
        if max_model_len_after != max_model_len_before:  # 如果长度发生了变化
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))  # 通过集体RPC同步新值

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)  # 生成调度器KV缓存配置
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks  # 更新GPU块数
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups  # 获取KV缓存组
        if kv_cache_groups:  # 如果有缓存组
            vllm_config.cache_config.block_size = min(  # 设置块大小为所有组中最小的块大小
                g.kv_cache_spec.block_size for g in kv_cache_groups  # 遍历所有缓存组的块大小
            )

        vllm_config.validate_block_size()  # 验证块大小

        # 初始化KV缓存并预热执行
        self.model_executor.initialize_from_config(kv_cache_configs)  # 从配置初始化模型执行器

        elapsed = time.time() - start  # 计算耗时
        logger.info_once(  # 仅记录一次初始化耗时信息
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",  # 日志格式
            elapsed,  # 耗时秒数
            scope="local",  # 本地作用域
        )
        return scheduler_kv_cache_config  # 返回调度器KV缓存配置

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型
        """获取模型支持的任务类型。"""
        return self.model_executor.supported_tasks  # 返回执行器支持的任务

    def add_request(self, request: Request, request_wave: int = 0):  # 添加请求到调度器
        """将请求添加到调度器。

        `request_wave`: 在数据并行情况下，指示这个请求预期属于哪一波
        """
        # 验证request_id的类型
        if not isinstance(request.request_id, str):  # 如果request_id不是字符串
            raise TypeError(  # 抛出类型错误
                f"request_id must be a string, got {type(request.request_id)}"  # 错误信息
            )

        if pooling_params := request.pooling_params:  # 如果有池化参数
            supported_pooling_tasks = [  # 获取支持的池化任务列表
                task for task in self.get_supported_tasks() if task in POOLING_TASKS  # 过滤支持的池化任务
            ]

            if pooling_params.task not in supported_pooling_tasks:  # 如果任务不在支持列表中
                raise ValueError(  # 抛出值错误
                    f"Unsupported task: {pooling_params.task!r} "  # 不支持的任务
                    f"Supported tasks: {supported_pooling_tasks}"  # 支持的任务列表
                )

        if request.kv_transfer_params is not None and (  # 如果有KV传输参数
            not self.scheduler.get_kv_connector()  # 但没有KV连接器
        ):
            logger.warning(  # 记录警告
                "Got kv_transfer_params, but no KVConnector found. "  # 警告信息
                "Disabling KVTransfer for this request."  # 禁用此请求的KV传输
            )

        self.scheduler.add_request(request)  # 将请求添加到调度器

    def abort_requests(self, request_ids: list[str]):  # 中止请求
        """从调度器中中止请求。"""

        # TODO: 调度器实际上不需要知道具体的完成原因，
        # 待定是否传播该信息（即客户端中止 vs 停止条件满足）
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)  # 以中止状态完成请求

    @contextmanager  # 上下文管理器装饰器
    def log_error_detail(self, scheduler_output: SchedulerOutput):  # 记录错误详情
        """执行模型并在失败时记录详细信息。"""
        try:  # 尝试执行
            yield  # 让出控制权给上下文体
        except Exception as err:  # 捕获异常
            # 我们不想在这里捕获BaseException，因为我们只
            # 对execute_model本身的错误导致的异常感兴趣

            # 注意：此方法不会抛出异常
            dump_engine_exception(  # 转储引擎异常信息
                self.vllm_config, scheduler_output, self.scheduler.make_stats()  # 传入配置、调度输出和统计
            )
            raise err  # 重新抛出异常

    @contextmanager  # 上下文管理器装饰器
    def log_iteration_details(self, scheduler_output: SchedulerOutput):  # 记录迭代详情
        """记录每次迭代的详细信息（请求数、token数、耗时等）。"""
        if not self.vllm_config.observability_config.enable_logging_iteration_details:  # 如果未启用迭代详情日志
            yield  # 直接让出控制权
            return  # 返回
        self._iteration_index = getattr(self, "_iteration_index", 0)  # 获取或初始化迭代索引
        iteration_details = compute_iteration_details(scheduler_output)  # 计算迭代详情
        before = time.monotonic()  # 记录开始时间（单调时钟）
        yield  # 让出控制权给上下文体
        logger.info(  # 记录迭代信息
            "".join(  # 拼接日志字符串
                [
                    "Iteration(",  # 迭代前缀
                    str(self._iteration_index),  # 迭代索引
                    "): ",  # 分隔符
                    str(iteration_details.num_ctx_requests),  # 上下文请求数
                    " context requests, ",  # 上下文请求标签
                    str(iteration_details.num_ctx_tokens),  # 上下文token数
                    " context tokens, ",  # 上下文token标签
                    str(iteration_details.num_generation_requests),  # 生成请求数
                    " generation requests, ",  # 生成请求标签
                    str(iteration_details.num_generation_tokens),  # 生成token数
                    " generation tokens, iteration elapsed time: ",  # 迭代耗时标签
                    format((time.monotonic() - before) * 1000, ".2f"),  # 计算并格式化耗时（毫秒）
                    " ms",  # 毫秒单位
                ]
            )
        )
        self._iteration_index += 1  # 递增迭代索引

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:  # 执行一步调度和推理
        """调度、执行并生成输出。

        返回输出字典和一个表示模型是否执行了的标志的元组。
        """

        # 检查调度器中是否有剩余请求——未完成的或已完成但尚未从批次中移除的
        if not self.scheduler.has_requests():  # 如果调度器没有请求
            return {}, False  # 返回空字典和False
        scheduler_output = self.scheduler.schedule()  # 执行调度
        future = self.model_executor.execute_model(scheduler_output, non_block=True)  # 非阻塞执行模型
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)  # 获取语法位掩码
        with (  # 使用上下文管理器
            self.log_error_detail(scheduler_output),  # 错误详情日志
            self.log_iteration_details(scheduler_output),  # 迭代详情日志
        ):
            model_output = future.result()  # 等待模型执行结果
            if model_output is None:  # 如果模型输出为None
                model_output = self.model_executor.sample_tokens(grammar_output)  # 进行token采样

        # 在处理模型输出之前，先处理模型执行期间发生的中止请求
        self._process_aborts_queue()  # 处理中止队列
        engine_core_outputs = self.scheduler.update_from_output(  # 从输出更新调度器状态
            scheduler_output, model_output  # 传入调度输出和模型输出
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0  # 返回输出和是否有token被调度

    def post_step(self, model_executed: bool) -> None:  # 步进后处理
        """步进后处理，主要用于推测解码的草稿token更新。"""
        # 使用异步调度时无法提前获取草稿token ID，
        # 所以在工作进程中更新草稿token ID，此处不需要更新
        if not self.async_scheduling and self.use_spec_decode and model_executed:  # 非异步调度且使用推测解码且模型已执行
            # 获取草稿token ID
            draft_token_ids = self.model_executor.take_draft_token_ids()  # 从执行器获取草稿token
            if draft_token_ids is not None:  # 如果有草稿token
                self.scheduler.update_draft_token_ids(draft_token_ids)  # 更新调度器的草稿token

    def step_with_batch_queue(  # 带批次队列的步进方法
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:  # 返回输出字典或None，以及是否执行了模型
        """使用批次队列调度和执行批次。
        注意如果此步骤没有输出，返回None。

        执行流程如下：
        1. 如果批次队列未满，尝试调度新批次。
        如果调度了新批次，直接返回空的引擎核心输出。
        换言之，填满批次队列比获取模型输出优先级更高。
        2. 如果没有新调度的批次，意味着批次队列已满
        或没有其他请求可调度，我们阻塞直到作业队列中的第一个批次完成。
        3. 从输出更新调度器。
        """

        batch_queue = self.batch_queue  # 获取批次队列引用
        assert batch_queue is not None  # 断言批次队列不为None

        # 如果批次队列未满，尝试调度新批次，但调度器可能返回空批次
        # （如果所有请求都已调度）。注意这不是阻塞的。
        assert len(batch_queue) < self.batch_queue_size  # 断言队列未满

        model_executed = False  # 初始化模型执行标志为False
        deferred_scheduler_output = None  # 初始化延迟调度输出为None
        if self.scheduler.has_requests():  # 如果调度器有请求
            scheduler_output = self.scheduler.schedule()  # 执行调度
            with self.log_error_detail(scheduler_output):  # 使用错误日志上下文管理器
                exec_future = self.model_executor.execute_model(  # 非阻塞执行模型
                    scheduler_output, non_block=True  # 传入调度输出，非阻塞模式
                )
            if self.is_ec_consumer:  # 如果是EC消费者
                model_executed = scheduler_output.total_num_scheduled_tokens > 0  # 检查是否有token被调度

            if self.is_pooling_model or not model_executed:  # 如果是池化模型或模型未执行
                # 不需要采样（没有请求被调度）
                future = cast(Future[ModelRunnerOutput], exec_future)  # 转换Future类型
            else:  # 否则需要采样
                if not scheduler_output.pending_structured_output_tokens:  # 如果没有等待中的结构化输出token
                    # 我们不需要等待任何token，获取语法输出并立即采样
                    grammar_output = self.scheduler.get_grammar_bitmask(  # 获取语法位掩码
                        scheduler_output  # 传入调度输出
                    )
                    future = self.model_executor.sample_tokens(  # 非阻塞采样token
                        grammar_output, non_block=True  # 传入语法输出，非阻塞模式
                    )
                else:  # 如果有等待中的结构化输出token
                    # 我们需要延迟采样，直到处理完前一步的模型输出
                    deferred_scheduler_output = scheduler_output  # 保存延迟的调度输出

            if not deferred_scheduler_output:  # 如果没有延迟的调度输出
                # 将本步的future添加到队列
                batch_queue.appendleft((future, scheduler_output, exec_future))  # 添加到队列头部
                if (  # 如果满足以下所有条件
                    model_executed  # 模型已执行
                    and len(batch_queue) < self.batch_queue_size  # 且队列未满
                    and not batch_queue[-1][0].done()  # 且队尾的future未完成
                ):
                    # 除非队列已满或没有更多请求要调度，否则不阻塞等待下一个工作进程响应
                    return None, True  # 返回None和True表示已执行但无输出

        elif not batch_queue:  # 如果队列为空
            # 队列为空。不应到达这里，因为此方法只应在调度器有请求或队列非空时调用
            return None, False  # 返回None和False

        # 阻塞直到下一个结果可用
        future, scheduler_output, exec_model_fut = batch_queue.pop()  # 从队列尾部弹出
        with (  # 使用上下文管理器
            self.log_error_detail(scheduler_output),  # 错误详情日志
            self.log_iteration_details(scheduler_output),  # 迭代详情日志
        ):
            model_output = future.result()  # 阻塞等待模型输出结果
            if model_output is None:  # 如果模型输出为None
                # sample_tokens()返回None意味着原始execute_model()调用失败——抛出该异常
                exec_model_fut.result()  # 获取执行模型future的结果（会抛出异常）
                raise RuntimeError("unexpected error")  # 抛出运行时错误

        # 在处理模型输出之前，先处理模型执行期间发生的中止请求
        self._process_aborts_queue()  # 处理中止队列
        engine_core_outputs = self.scheduler.update_from_output(  # 从输出更新调度器状态
            scheduler_output, model_output  # 传入调度输出和模型输出
        )

        # 注意(nick): 我们可以在这里处理延迟任务，也可以保存到字段中
        # 并在step_with_batch_queue再次被调用时立即执行。后者略微有利于TTFT而非TPOT/吞吐量。
        if deferred_scheduler_output:  # 如果有延迟的调度输出
            # 如果我们正在使用推测解码和结构化输出，需要在计算延迟请求的语法位掩码之前
            # 从上一步获取草稿token ID
            if self.use_spec_decode:  # 如果使用推测解码
                draft_token_ids = self.model_executor.take_draft_token_ids()  # 获取草稿token ID
                assert draft_token_ids is not None  # 断言草稿token不为None
                # 更新调度输出中的草稿token ID以过滤无效的推测token，
                # 无效token将用-1填充并在语法位掩码计算中被跳过
                self.scheduler.update_draft_token_ids_in_output(  # 更新输出中的草稿token ID
                    draft_token_ids, deferred_scheduler_output  # 传入草稿token和延迟调度输出
                )
            # 现在我们有了计算延迟请求位掩码所需的token，获取位掩码并调用采样token
            grammar_output = self.scheduler.get_grammar_bitmask(  # 获取语法位掩码
                deferred_scheduler_output  # 传入延迟的调度输出
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)  # 非阻塞采样token
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))  # 将延迟任务添加到队列头部

        return engine_core_outputs, model_executed  # 返回引擎核心输出和模型执行标志

    def _process_aborts_queue(self):  # 处理中止请求队列
        """处理中止请求队列中的所有待处理中止请求。"""
        if not self.aborts_queue.empty():  # 如果中止队列非空
            request_ids = []  # 初始化请求ID列表
            while not self.aborts_queue.empty():  # 循环直到队列为空
                ids = self.aborts_queue.get_nowait()  # 非阻塞获取队列中的ID
                # 这里应该是列表，但也处理字符串以防万一
                request_ids.extend((ids,) if isinstance(ids, str) else ids)  # 将ID添加到列表中
            # 作为单个批次中止所有请求更高效
            self.abort_requests(request_ids)  # 批量中止请求

    def shutdown(self):  # 关闭引擎核心
        """关闭引擎核心，释放所有资源。"""
        self.structured_output_manager.clear_backend()  # 清除结构化输出后端
        if self.model_executor:  # 如果模型执行器存在
            self.model_executor.shutdown()  # 关闭模型执行器
        if self.scheduler:  # 如果调度器存在
            self.scheduler.shutdown()  # 关闭调度器

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):  # 性能分析开关
        """启动或停止性能分析。"""
        self.model_executor.profile(is_start, profile_prefix)  # 委托给模型执行器进行性能分析

    def reset_mm_cache(self):  # 重置多模态缓存
        """重置多模态缓存。"""
        # 注意：由于这主要用于调试，我们不尝试重新同步内部缓存（P0发送者，P1接收者）
        if self.scheduler.has_unfinished_requests():  # 如果有未完成的请求
            logger.warning(  # 记录警告
                "Resetting the multi-modal cache when requests are "  # 重置多模态缓存时有请求在处理中
                "in progress may lead to desynced internal caches."  # 可能导致内部缓存不同步
            )

        # 缓存存在于EngineCore或WorkerWrapperBase中
        if self.mm_receiver_cache is not None:  # 如果多模态接收器缓存存在
            self.mm_receiver_cache.clear_cache()  # 清除缓存

        self.model_executor.reset_mm_cache()  # 重置模型执行器的多模态缓存

    def reset_prefix_cache(  # 重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 是否重置运行中的请求和连接器
    ) -> bool:  # 返回是否成功
        """重置前缀缓存。"""
        return self.scheduler.reset_prefix_cache(  # 委托给调度器重置前缀缓存
            reset_running_requests, reset_connector  # 传入参数
        )

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        """重置编码器缓存以使所有缓存的编码器输出失效。

        当模型权重更新时应调用此方法，以确保
        使用旧权重计算的过时视觉嵌入不会被重用。
        同时清除调度器的缓存管理器和GPU模型运行器的缓存。
        """
        # 注意：由于这主要用于调试，我们不尝试重新同步内部缓存（P0发送者，P1接收者）
        if self.scheduler.has_unfinished_requests():  # 如果有未完成的请求
            logger.warning(  # 记录警告
                "Resetting the encoder cache when requests are "  # 重置编码器缓存时有请求在处理中
                "in progress may lead to desynced internal caches."  # 可能导致内部缓存不同步
            )

        # 重置调度器的编码器缓存管理器（逻辑状态）
        self.scheduler.reset_encoder_cache()  # 重置调度器编码器缓存
        # 重置GPU模型运行器的编码器缓存（物理存储）
        self.model_executor.reset_encoder_cache()  # 重置执行器编码器缓存

    def _reset_caches(self, reset_running_requests=True) -> None:  # 重置所有缓存
        """重置所有缓存（前缀缓存、多模态缓存、编码器缓存）。"""
        self.reset_prefix_cache(reset_running_requests=reset_running_requests)  # 重置前缀缓存
        self.reset_mm_cache()  # 重置多模态缓存
        self.reset_encoder_cache()  # 重置编码器缓存

    def pause_scheduler(  # 暂停调度器
        self, mode: PauseMode = "abort", clear_cache: bool = True  # 暂停模式和是否清除缓存
    ) -> Future | None:  # 返回Future或None
        """暂停生成；行为取决于模式。

        所有暂停模式都会将新的添加排队——"abort"和"keep"跳过step()；
        "wait"允许step()以便进行中的请求可以排空。

        - ``abort``: 设置PAUSED_NEW，中止所有请求，等待中止输出发送
          （在使用output_queue运行时），可选清除缓存，然后完成返回的Future。
        - ``wait``: 设置PAUSED_NEW（排队添加，继续步进）；排空后，
          可选清除缓存，然后完成返回的Future。
        - ``keep``: 设置PAUSED_ALL；返回一个在输出队列为空时完成的Future。
        """
        if mode not in ("keep", "abort", "wait"):  # 如果模式无效
            raise ValueError(f"Invalid pause mode: {mode}")  # 抛出值错误
        if mode == "wait":  # 如果是等待模式
            raise ValueError("'wait' mode can't be used in inproc-engine mode")  # 进程内引擎模式不支持wait

        if mode == "abort":  # 如果是中止模式
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)  # 中止所有请求

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW  # 根据模式选择暂停状态
        self.scheduler.set_pause_state(pause_state)  # 设置调度器暂停状态
        if clear_cache:  # 如果需要清除缓存
            self._reset_caches()  # 重置所有缓存

        return None  # 返回None

    def resume_scheduler(self) -> None:  # 恢复调度器
        """恢复调度器并刷新暂停期间排队的所有请求。"""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)  # 设置调度器为未暂停状态

    def is_scheduler_paused(self) -> bool:  # 检查调度器是否暂停
        """返回调度器是否处于任何暂停状态。"""
        return self.scheduler.pause_state != PauseState.UNPAUSED  # 比较暂停状态

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:  # 让引擎进入休眠
        """让引擎在指定级别进入休眠。

        参数:
            level: 休眠级别。
                - 级别0: 仅暂停调度。请求仍被接受但不处理。不改变GPU内存。
                - 级别1: 将模型权重卸载到CPU，丢弃KV缓存。
                - 级别2: 丢弃所有GPU内存。
            mode: 暂停模式——如何处理现有请求，参见pause_scheduler方法文档。
        """

        # 休眠前暂停调度器
        clear_prefix_cache = level >= 1  # 级别1及以上需要清除前缀缓存
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)  # 暂停调度器
        if level < 1:  # 如果级别小于1
            return pause_future  # 直接返回暂停Future

        # 级别1+: 委托给执行器进行GPU内存管理
        model_executor = self.model_executor  # 获取模型执行器引用
        if pause_future is None:  # 如果暂停Future为None（已立即完成）
            model_executor.sleep(level)  # 直接让执行器休眠
            return None  # 返回None

        future = Future[Any]()  # 创建新的Future对象

        def pause_complete(f: Future):  # 暂停完成回调函数
            try:  # 尝试执行
                f.result()  # 传播任何异常
                future.set_result(model_executor.sleep(level))  # 设置休眠结果
            except Exception as e:  # 捕获异常
                future.set_exception(e)  # 设置异常

        logger.info("Waiting for in-flight requests to complete before sleeping...")  # 记录等待信息
        pause_future.add_done_callback(pause_complete)  # 添加完成回调
        return future  # 返回Future

    def wake_up(self, tags: list[str] | None = None):  # 唤醒引擎
        """从休眠中唤醒引擎。

        参数:
            tags: 要唤醒的标签。使用["scheduling"]进行级别0唤醒。
        """
        if tags is not None and "scheduling" in tags:  # 如果标签中包含"scheduling"
            # 如果还有其他标签要处理，从标签中移除"scheduling"
            tags = [t for t in tags if t != "scheduling"]  # 过滤掉"scheduling"

        if tags is None or tags:  # 如果标签为None或非空
            self.model_executor.wake_up(tags)  # 唤醒模型执行器

        # 恢复调度（适用于所有级别）
        self.resume_scheduler()  # 恢复调度器

    def is_sleeping(self) -> bool:  # 检查是否在休眠
        """检查引擎是否在任何级别休眠。"""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping  # 检查调度器暂停或执行器休眠

    def execute_dummy_batch(self):  # 执行虚拟批次
        """执行一个虚拟批次用于预热。"""
        self.model_executor.execute_dummy_batch()  # 委托给模型执行器

    def add_lora(self, lora_request: LoRARequest) -> bool:  # 添加LoRA适配器
        """添加LoRA适配器。"""
        return self.model_executor.add_lora(lora_request)  # 委托给模型执行器

    def remove_lora(self, lora_id: int) -> bool:  # 移除LoRA适配器
        """移除LoRA适配器。"""
        return self.model_executor.remove_lora(lora_id)  # 委托给模型执行器

    def list_loras(self) -> set[int]:  # 列出LoRA适配器
        """列出所有已加载的LoRA适配器ID。"""
        return self.model_executor.list_loras()  # 委托给模型执行器

    def pin_lora(self, lora_id: int) -> bool:  # 固定LoRA适配器
        """固定LoRA适配器到内存。"""
        return self.model_executor.pin_lora(lora_id)  # 委托给模型执行器

    def save_sharded_state(  # 保存分片状态
        self,
        path: str,  # 保存路径
        pattern: str | None = None,  # 文件名模式
        max_size: int | None = None,  # 最大文件大小
    ) -> None:
        """保存模型的分片状态。"""
        self.model_executor.save_sharded_state(  # 委托给模型执行器保存
            path=path, pattern=pattern, max_size=max_size  # 传入保存参数
        )

    def collective_rpc(  # 集体RPC调用
        self,
        method: str | Callable[..., _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回结果列表
        """在所有工作进程上执行集体RPC调用。"""
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)  # 委托给模型执行器

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:  # 预处理添加请求
        """预处理请求。

        此函数可直接在输入处理线程中使用，允许请求初始化与模型前向传播并行运行。
        """
        # 关于线程安全的说明：没有竞态条件。
        # `mm_receiver_cache`在LLMEngine初始化结束时重置，
        # 之后只在输入处理线程中访问。
        if self.mm_receiver_cache is not None and request.mm_features:  # 如果有多模态缓存且请求有多模态特征
            request.mm_features = self.mm_receiver_cache.get_and_update_features(  # 获取并更新特征
                request.mm_features  # 传入多模态特征
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)  # 从引擎核心请求创建Request对象
        if req.use_structured_output:  # 如果使用结构化输出
            # 关于线程安全的说明：没有竞态条件。
            # `grammar_init`仅在输入处理线程中调用。对于
            # `structured_output_manager`，每个请求是独立的，
            # 语法编译是异步的。调度器总是在调度请求前检查语法编译状态。
            self.structured_output_manager.grammar_init(req)  # 初始化语法
        return req, request.current_wave  # 返回请求对象和当前波次

    def _eep_scale_up_before_kv_init(self):  # 弹性专家并行KV初始化前扩容
        """弹性专家并行在KV初始化前的扩容操作（需子类实现）。"""
        raise NotImplementedError  # 抛出未实现异常

    def _eep_send_engine_core_notification(  # 发送弹性专家并行引擎核心通知
        self,
        notification_type: EEPNotificationType,  # 通知类型
        vllm_config: VllmConfig | None = None,  # 可选的vLLM配置
    ):
        """发送弹性专家并行引擎核心通知（需子类实现）。"""
        raise NotImplementedError  # 抛出未实现异常


class EngineShutdownState(IntEnum):  # 引擎关闭状态枚举
    """引擎关闭状态枚举类。"""
    RUNNING = 0  # 运行中
    REQUESTED = 1  # 已请求关闭
    SHUTTING_DOWN = 2  # 正在关闭


class EngineCoreProc(EngineCore):  # 引擎核心进程类，继承自EngineCore
    """用于在后台进程中运行EngineCore的ZMQ包装器。"""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"  # 引擎核心死亡标记常量
    addresses: EngineZmqAddresses  # ZMQ地址类型声明

    @instrument(span_name="EngineCoreProc init")  # 追踪装饰器，标记为"引擎核心进程初始化"
    def __init__(  # 初始化引擎核心进程
        self,
        vllm_config: VllmConfig,  # vLLM配置对象
        local_client: bool,  # 是否为本地客户端
        handshake_address: str,  # 握手地址
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计信息
        client_handshake_address: str | None = None,  # 客户端握手地址（可选）
        *,
        engine_index: int = 0,  # 引擎索引，默认为0
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()  # 创建输入队列
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()  # 创建输出队列
        executor_fail_callback = lambda: self.input_queue.put_nowait(  # 创建执行器失败回调
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")  # 放入失败消息到输入队列
        )

        self.engine_index = engine_index  # 保存引擎索引
        identity = self.engine_index.to_bytes(length=2, byteorder="little")  # 将引擎索引转为2字节小端序
        self.engines_running = False  # 初始化引擎运行标志为False
        self.shutdown_state = EngineShutdownState.RUNNING  # 初始化关闭状态为运行中

        with self._perform_handshakes(  # 执行握手
            handshake_address,  # 握手地址
            identity,  # 身份标识
            local_client,  # 是否本地客户端
            vllm_config,  # vLLM配置
            client_handshake_address,  # 客户端握手地址
        ) as addresses:  # 获取地址对象
            # 设置数据并行环境
            self.has_coordinator = addresses.coordinator_output is not None  # 判断是否有协调器
            self.frontend_stats_publish_address = (  # 获取前端统计发布地址
                addresses.frontend_stats_publish_address  # 从地址对象获取
            )
            logger.debug(  # 记录调试信息
                "Has DP Coordinator: %s, stats publish address: %s",  # 日志格式
                self.has_coordinator,  # 是否有DP协调器
                self.frontend_stats_publish_address,  # 统计发布地址
            )
            internal_dp_balancing = (  # 判断是否使用内部数据并行负载均衡
                self.has_coordinator  # 有协调器
                and not vllm_config.parallel_config.data_parallel_external_lb  # 且未使用外部负载均衡
            )
            # 仅在"内部"和"混合"负载均衡模式下将请求队列统计发布到协调器
            self.publish_dp_lb_stats = internal_dp_balancing  # 是否发布DP负载均衡统计

            self.addresses = addresses  # 保存地址对象
            self.process_input_queue_block = True  # 处理输入队列时阻塞
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:  # 如果启用弹性扩容启动
                self._eep_send_engine_core_notification(  # 发送弹性专家并行通知
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,  # 通知类型：新核心引擎初始化就绪
                    vllm_config=vllm_config,  # 传入配置
                )
            self._init_data_parallel(vllm_config)  # 初始化数据并行

            super().__init__(  # 调用父类EngineCore的初始化
                vllm_config,  # vLLM配置
                executor_class,  # 执行器类
                log_stats,  # 是否记录统计
                executor_fail_callback,  # 失败回调
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

        raise SystemExit

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there are waiting requests
        # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow
        # background threads (like NIXL handshake) to make progress.
        # Without this, the tight polling loop can starve background threads.
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_shutdown(self) -> bool:
        """ 检查并处理shutdown。 """
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout

            logger.info("Shutdown initiated (timeout=%d)", shutdown_timeout)

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info("Aborting %d requests", num_requests)
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "Draining %d in-flight requests (timeout=%ds)",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info("Shutdown complete")
            return False

        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (method := getattr(self, method_name)) and method(
                *self._convert_msgspec_args(method, args)
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.info("Rejecting request %s (server shutting down)", request.request_id)
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning("Rejecting utility call %s (server shutting down)", method_name)
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if not self.has_work():
            if clear_cache:
                self._reset_caches()
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for req_id, client_index in aborted_reqs:
                by_client[client_index].add(req_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        super().resume_scheduler()
        if (
            self.has_coordinator
            and not self.engines_running
            and self.scheduler.has_unfinished_requests()
        ):
            # Wake up other DP engines.
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        """
        Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale up: new core engines to notify existing core engines
           that they are ready;
        2) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        3) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        if vllm_config is None:
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        else:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ):
        """
        Handle notification received from EngineCoreClient
        (forwarded from new core engines).
        """
        assert self.eep_scaling_state is not None
        if isinstance(notification_type, str):
            notification_type = EEPNotificationType(notification_type)
        self.eep_scaling_state.handle_notification(notification_type)

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        self.model_executor.collective_rpc("init_device")
        self.model_executor.collective_rpc("load_model")
        self._eep_send_engine_core_notification(
            EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
        )
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("receive_weights",)
        )
        self.available_gpu_memory_for_kv_cache = (
            ParallelConfig.sync_kv_cache_memory_size(self.dp_group, -1)
        )
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("prepare_new_worker",)
        )
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Initialize tracer for distributed tracing if configured.
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
