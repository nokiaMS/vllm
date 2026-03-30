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

    # 添加请求到调度器
    def add_request(self, request: Request, request_wave: int = 0):
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

            # IO的后台线程和队列。这些使我们能够将ZMQ套接字IO与GPU重叠，
            # 因为它们释放了GIL，并且可以将一些序列化/反序列化与模型前向传播重叠。
            # 线程处理 Socket <-> 队列，core_busy_loop使用队列。
            ready_event = threading.Event()  # 创建就绪事件
            input_thread = threading.Thread(  # 创建输入处理线程
                target=self.process_input_sockets,  # 目标函数为处理输入套接字
                args=(  # 线程参数
                    addresses.inputs,  # 输入地址列表
                    addresses.coordinator_input,  # 协调器输入地址
                    identity,  # 身份标识
                    ready_event,  # 就绪事件
                ),
                daemon=True,  # 设置为守护线程
            )
            input_thread.start()  # 启动输入线程

            self.output_thread = threading.Thread(  # 创建输出处理线程
                target=self.process_output_sockets,  # 目标函数为处理输出套接字
                args=(  # 线程参数
                    addresses.outputs,  # 输出地址列表
                    addresses.coordinator_output,  # 协调器输出地址
                    self.engine_index,  # 引擎索引
                ),
                daemon=True,  # 设置为守护线程
            )
            self.output_thread.start()  # 启动输出线程

            # 在收到DP协调器就绪消息之前，不要完成握手。
            while not ready_event.wait(timeout=10):  # 等待就绪事件，超时10秒
                if not input_thread.is_alive():  # 如果输入线程已死亡
                    raise RuntimeError("Input socket thread died during startup")  # 抛出运行时错误
                assert addresses.coordinator_input is not None  # 断言协调器输入地址不为None
                logger.info("Waiting for READY message from DP Coordinator...")  # 记录等待协调器就绪消息

    @contextmanager  # 上下文管理器装饰器
    def _perform_handshakes(  # 执行握手
        self,
        handshake_address: str,  # 握手地址
        identity: bytes,  # 身份标识
        local_client: bool,  # 是否本地客户端
        vllm_config: VllmConfig,  # vLLM配置
        client_handshake_address: str | None,  # 客户端握手地址（可选）
    ) -> Generator[EngineZmqAddresses, None, None]:  # 返回ZMQ地址生成器
        """执行启动握手。

        对于DP=1或离线模式，与共置的前端进程握手。

        对于DP>1且使用内部负载均衡，与可能在不同节点上的共享前端进程握手。

        对于DP>1且使用外部或混合负载均衡，执行两次握手：
            - 与rank 0前端进程握手，获取DP协调器ZMQ地址和DP进程组地址。
            - 与共置的前端进程握手，获取客户端输入/输出套接字地址。
        rank 0和共置引擎本身不需要第二次握手。

        这里的"前端"进程可以是包含引擎核心客户端的进程（在API服务器未扩展的
        情况下为API服务器进程），或者是运行serve.py中run_multi_api_server()
        函数的启动器进程。
        """
        input_ctx = zmq.Context()  # 创建ZMQ上下文
        is_local = local_client and client_handshake_address is None  # 判断是否为本地连接
        headless = not local_client  # 判断是否为无头模式
        handshake = self._perform_handshake(  # 执行握手
            input_ctx,  # ZMQ上下文
            handshake_address,  # 握手地址
            identity,  # 身份标识
            is_local,  # 是否本地
            headless,  # 是否无头
            vllm_config,  # vLLM配置
            vllm_config.parallel_config,  # 并行配置
        )
        if client_handshake_address is None:  # 如果没有客户端握手地址
            with handshake as addresses:  # 使用握手获取地址
                yield addresses  # 返回地址
        else:  # 否则需要额外的本地握手
            assert local_client  # 断言是本地客户端
            local_handshake = self._perform_handshake(  # 执行本地握手
                input_ctx, client_handshake_address, identity, True, False, vllm_config  # 本地握手参数
            )
            with handshake as addresses, local_handshake as client_addresses:  # 获取两次握手的地址
                addresses.inputs = client_addresses.inputs  # 使用客户端的输入地址
                addresses.outputs = client_addresses.outputs  # 使用客户端的输出地址
                yield addresses  # 返回合并后的地址

        # 更新可能在握手过程中改变的配置
        vllm_config.__post_init__()  # 重新执行配置后初始化

    @contextmanager  # 上下文管理器装饰器
    def _perform_handshake(  # 执行单次握手
        self,
        ctx: zmq.Context,  # ZMQ上下文
        handshake_address: str,  # 握手地址
        identity: bytes,  # 身份标识
        local_client: bool,  # 是否本地客户端
        headless: bool,  # 是否无头模式
        vllm_config: VllmConfig,  # vLLM配置
        parallel_config_to_update: ParallelConfig | None = None,  # 待更新的并行配置（可选）
    ) -> Generator[EngineZmqAddresses, None, None]:  # 返回ZMQ地址生成器
        with make_zmq_socket(  # 创建ZMQ套接字
            ctx,  # ZMQ上下文
            handshake_address,  # 握手地址
            zmq.DEALER,  # DEALER类型套接字
            identity=identity,  # 身份标识
            linger=5000,  # 关闭时等待5000毫秒
            bind=False,  # 连接模式而非绑定模式
        ) as handshake_socket:  # 获取握手套接字
            # 向前端注册引擎。
            addresses = self.startup_handshake(  # 执行启动握手
                handshake_socket, local_client, headless, parallel_config_to_update  # 传入握手参数
            )
            yield addresses  # 返回地址对象

            # 发送就绪消息。
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks  # 获取GPU块数
            # 在此处传回协调器统计更新地址，用于外部负载均衡场景中
            # 共置前端使用（协调器仅在rank 0运行）。
            dp_stats_address = self.frontend_stats_publish_address  # 获取统计发布地址

            # 包含配置哈希用于DP配置验证
            ready_msg = {  # 构建就绪消息字典
                "status": "READY",  # 状态：就绪
                "local": local_client,  # 是否本地客户端
                "headless": headless,  # 是否无头模式
                "num_gpu_blocks": num_gpu_blocks,  # GPU块数
                "dp_stats_address": dp_stats_address,  # DP统计地址
            }
            if vllm_config.parallel_config.data_parallel_size > 1:  # 如果数据并行度大于1
                ready_msg["parallel_config_hash"] = (  # 添加并行配置哈希
                    vllm_config.parallel_config.compute_hash()  # 计算配置哈希
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))  # 发送编码后的就绪消息

    @staticmethod  # 静态方法装饰器
    def startup_handshake(  # 启动握手
        handshake_socket: zmq.Socket,  # 握手套接字
        local_client: bool,  # 是否本地客户端
        headless: bool,  # 是否无头模式
        parallel_config: ParallelConfig | None = None,  # 待更新的并行配置（可选）
    ) -> EngineZmqAddresses:  # 返回ZMQ地址对象
        """执行启动握手，注册引擎并接收初始化配置。"""
        # 发送注册消息。
        handshake_socket.send(  # 发送编码后的注册消息
            msgspec.msgpack.encode(  # 使用msgpack编码
                {
                    "status": "HELLO",  # 状态：你好
                    "local": local_client,  # 是否本地客户端
                    "headless": headless,  # 是否无头模式
                }
            )
        )

        # 接收初始化消息。
        logger.debug("Waiting for init message from front-end.")  # 记录等待前端初始化消息
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):  # 轮询等待，超时则报错
            raise RuntimeError(  # 抛出运行时错误
                "Did not receive response from front-end "  # 未收到前端响应
                f"process within {HANDSHAKE_TIMEOUT_MINS} "  # 超时分钟数
                f"minutes"  # 分钟
            )
        init_bytes = handshake_socket.recv()  # 接收初始化消息字节
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(  # 解码初始化消息
            init_bytes, type=EngineHandshakeMetadata  # 解码为EngineHandshakeMetadata类型
        )
        logger.debug("Received init message: %s", init_message)  # 记录收到的初始化消息

        if parallel_config is not None:  # 如果有并行配置需要更新
            for key, value in init_message.parallel_config.items():  # 遍历配置项
                setattr(parallel_config, key, value)  # 设置并行配置属性

        return init_message.addresses  # 返回地址信息

    @staticmethod  # 静态方法装饰器
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):  # 运行引擎核心
        """在后台进程中启动EngineCore繁忙循环。"""

        # 确保在子进程生成后可以序列化transformer配置
        maybe_register_config_serialize_by_value()  # 注册按值序列化配置

        engine_core: EngineCoreProc | None = None  # 初始化引擎核心为None
        signal_callback: SignalCallback | None = None  # 初始化信号回调为None
        try:  # 尝试启动引擎核心
            vllm_config: VllmConfig = kwargs["vllm_config"]  # 获取vLLM配置
            parallel_config: ParallelConfig = vllm_config.parallel_config  # 获取并行配置
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0  # 判断是否使用数据并行
            if data_parallel:  # 如果使用数据并行
                parallel_config.data_parallel_rank_local = local_dp_rank  # 设置本地DP rank
                process_title = f"EngineCore_DP{dp_rank}"  # 设置进程标题包含DP rank
            else:  # 否则
                process_title = "EngineCore"  # 使用默认进程标题
            set_process_title(process_title)  # 设置进程标题
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)  # 可选初始化工作进程追踪器
            decorate_logs()  # 装饰日志

            if data_parallel and vllm_config.kv_transfer_config is not None:  # 如果使用DP且有KV传输配置
                # 修改engine_id并附加local_dp_rank，确保每个DP rank的kv_transfer_config唯一。
                vllm_config.kv_transfer_config.engine_id = (  # 更新引擎ID
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"  # 附加DP rank后缀
                )
                logger.debug(  # 记录调试信息
                    "Setting kv_transfer_config.engine_id to %s",  # 日志格式
                    vllm_config.kv_transfer_config.engine_id,  # 新的引擎ID
                )

            parallel_config.data_parallel_index = dp_rank  # 设置数据并行索引
            if data_parallel and vllm_config.model_config.is_moe:  # 如果使用DP且是MoE模型
                # 为此引擎进程设置数据并行rank。
                parallel_config.data_parallel_rank = dp_rank  # 设置DP rank
                engine_core = DPEngineCoreProc(*args, **kwargs)  # 创建DP引擎核心进程
            else:  # 否则
                # 非MoE的DP rank完全独立，所以视为DP=1。
                # 注意parallel_config.data_parallel_index仍然反映原始DP rank。
                parallel_config.data_parallel_size = 1  # 设置DP大小为1
                parallel_config.data_parallel_size_local = 1  # 设置本地DP大小为1
                parallel_config.data_parallel_rank = 0  # 设置DP rank为0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)  # 创建普通引擎核心进程

            assert engine_core is not None  # 断言引擎核心已创建

            def wakeup_engine():  # 唤醒引擎函数
                # 当请求关闭时，通过input_queue唤醒空闲引擎
                # 在信号处理器中不安全——我们可能在主线程持有不可重入的input_queue.mutex时中断它
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))  # 放入唤醒消息

            signal_callback = SignalCallback(wakeup_engine)  # 创建信号回调

            def signal_handler(signum, frame):  # 信号处理器
                engine_core.shutdown_state = EngineShutdownState.REQUESTED  # 设置关闭状态为已请求
                signal_callback.trigger()  # 触发信号回调

            signal.signal(signal.SIGTERM, signal_handler)  # 注册SIGTERM信号处理器
            signal.signal(signal.SIGINT, signal_handler)  # 注册SIGINT信号处理器

            engine_core.run_busy_loop()  # 运行繁忙循环

        except SystemExit:  # 捕获系统退出
            logger.debug("EngineCore exiting.")  # 记录引擎核心退出
            raise  # 重新抛出
        except Exception as e:  # 捕获其他异常
            if engine_core is None:  # 如果引擎核心未创建
                logger.exception("EngineCore failed to start.")  # 记录启动失败
            else:  # 否则
                logger.exception("EngineCore encountered a fatal error.")  # 记录致命错误
                engine_core._send_engine_dead()  # 发送引擎死亡消息
            raise e  # 重新抛出异常
        finally:  # 最终清理
            signal.signal(signal.SIGTERM, signal.SIG_DFL)  # 恢复SIGTERM默认处理
            signal.signal(signal.SIGINT, signal.SIG_DFL)  # 恢复SIGINT默认处理
            if signal_callback is not None:  # 如果信号回调存在
                signal_callback.stop()  # 停止信号回调
            if engine_core is not None:  # 如果引擎核心存在
                engine_core.shutdown()  # 关闭引擎核心

    def _init_data_parallel(self, vllm_config: VllmConfig):  # 初始化数据并行（基类空实现）
        """初始化数据并行（基类空实现，子类可覆盖）。"""
        pass  # 空操作

    def has_work(self) -> bool:  # 判断是否有工作需要执行
        """如果引擎需要执行步进则返回True。"""
        return (
            self.engines_running  # 引擎正在运行
            or self.scheduler.has_requests()  # 或调度器有请求
            or bool(self.batch_queue)  # 或批次队列非空
        )

    def is_running(self) -> bool:  # 判断是否在运行
        """如果未请求关闭则返回True。"""
        return self.shutdown_state == EngineShutdownState.RUNNING  # 检查关闭状态

    # 运行繁忙循环
    def run_busy_loop(self):
        """EngineCore的核心繁忙循环。"""
        while self._handle_shutdown():  # 循环直到关闭
            # 1) 轮询输入队列直到有工作要做。
            self._process_input_queue()  # 处理输入队列
            # 2) 执行引擎核心步进并返回输出。
            self._process_engine_step()  # 处理引擎步进

        raise SystemExit  # 抛出系统退出异常

    # 处理输入队列
    def _process_input_queue(self):
        """当需要执行引擎步进时退出。"""

        waited = False  # 是否等待过的标志
        while not self.has_work() and self.is_running():  # 当没有工作且仍在运行时
            # 通知等待引擎空闲的回调。
            self._notify_idle_state_callbacks()  # 通知空闲状态回调
            if self.input_queue.empty():  # 如果输入队列为空
                # 清空中止队列；所有中止也通过input_queue处理。
                with self.aborts_queue.mutex:  # 获取中止队列的锁
                    self.aborts_queue.queue.clear()  # 清空中止队列
                if logger.isEnabledFor(DEBUG):  # 如果启用了DEBUG日志
                    logger.debug("EngineCore waiting for work.")  # 记录等待工作
                    waited = True  # 标记已等待
            block = self.process_input_queue_block  # 获取是否阻塞标志
            try:  # 尝试获取请求
                req = self.input_queue.get(block=block)  # 从输入队列获取请求
                self._handle_client_request(*req)  # 处理客户端请求
            except queue.Empty:  # 队列为空异常
                break  # 退出循环
            if not block:  # 如果非阻塞模式
                break  # 退出循环

        if waited:  # 如果之前等待过
            logger.debug("EngineCore loop active.")  # 记录循环已激活

        # 处理更多客户端请求。
        while not self.input_queue.empty():  # 当输入队列非空时
            req = self.input_queue.get_nowait()  # 非阻塞获取请求
            self._handle_client_request(*req)  # 处理客户端请求

    def _process_engine_step(self) -> bool:  # 处理引擎步进
        """仅在有未完成的本地请求时调用。"""

        # 执行引擎核心步进。
        outputs, model_executed = self.step_fn()  # 调用步进函数
        # 将EngineCoreOutputs放入输出队列。
        for output in outputs.items() if outputs else ():  # 遍历输出
            self.output_queue.put_nowait(output)  # 放入输出队列
        # 步进后钩子。
        self.post_step(model_executed)  # 执行步进后处理

        # 如果没有模型执行但有等待中的请求（例如WAITING_FOR_REMOTE_KVS），
        # 短暂释放GIL以允许后台线程（如NIXL握手）推进。
        # 没有这个，紧密的轮询循环会饿死后台线程。
        if not model_executed and self.scheduler.has_unfinished_requests():  # 未执行但有未完成请求
            time.sleep(0.001)  # 休眠1毫秒

        return model_executed  # 返回模型是否执行了

    def _notify_idle_state_callbacks(self) -> None:  # 通知空闲状态回调
        """通知所有等待引擎空闲状态的回调。"""
        while self._idle_state_callbacks:  # 当有回调时
            callback = self._idle_state_callbacks.pop()  # 弹出回调
            callback(self)  # 执行回调

    def _handle_shutdown(self) -> bool:  # 处理关闭逻辑
        """检查并处理shutdown。"""
        # 检查是否请求了关闭并处理
        if self.shutdown_state == EngineShutdownState.RUNNING:  # 如果仍在运行
            return True  # 继续运行

        if self.shutdown_state == EngineShutdownState.REQUESTED:  # 如果已请求关闭
            shutdown_timeout = self.vllm_config.shutdown_timeout  # 获取关闭超时时间

            logger.info("Shutdown initiated (timeout=%d)", shutdown_timeout)  # 记录关闭已启动

            if shutdown_timeout == 0:  # 如果超时为0，立即关闭
                num_requests = self.scheduler.get_num_unfinished_requests()  # 获取未完成请求数
                if num_requests > 0:  # 如果有未完成请求
                    logger.info("Aborting %d requests", num_requests)  # 记录中止请求数
                aborted_reqs = self.scheduler.finish_requests(  # 中止所有请求
                    None, RequestStatus.FINISHED_ABORTED  # 以中止状态完成
                )
                self._send_abort_outputs(aborted_reqs)  # 发送中止输出
            else:  # 否则，等待请求排空
                num_requests = self.scheduler.get_num_unfinished_requests()  # 获取未完成请求数
                if num_requests > 0:  # 如果有未完成请求
                    logger.info(  # 记录排空信息
                        "Draining %d in-flight requests (timeout=%ds)",  # 排空进行中的请求
                        num_requests,  # 请求数量
                        shutdown_timeout,  # 超时时间
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN  # 设置状态为正在关闭

        # 当没有剩余工作时退出
        if not self.has_work():  # 如果没有工作
            logger.info("Shutdown complete")  # 记录关闭完成
            return False  # 返回False停止循环

        return True  # 继续运行

    # 处理客户端请求
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any  # 请求类型和请求数据
    ) -> None:
        """分发来自客户端的请求。"""

        if request_type == EngineCoreRequestType.WAKEUP:  # 如果是唤醒请求
            return  # 直接返回
        elif request_type == EngineCoreRequestType.ADD:  # 如果是添加请求
            req, request_wave = request  # 解包请求和波次
            if self._reject_add_in_shutdown(req):  # 如果在关闭期间拒绝添加
                return  # 直接返回
            self.add_request(req, request_wave)  # 添加请求
        elif request_type == EngineCoreRequestType.ABORT:  # 如果是中止请求
            self.abort_requests(request)  # 中止请求
        elif request_type == EngineCoreRequestType.UTILITY:  # 如果是工具方法请求
            client_idx, call_id, method_name, args = request  # 解包客户端索引、调用ID、方法名和参数
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):  # 如果在关闭期间拒绝
                return  # 直接返回
            output = UtilityOutput(call_id)  # 创建工具输出对象
            # 延迟查找工具方法，以便失败会被处理/返回。
            get_result = lambda: (method := getattr(self, method_name)) and method(  # 获取并调用方法
                *self._convert_msgspec_args(method, args)  # 转换msgspec参数
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(  # 将输出加入队列的函数
                (client_idx, EngineCoreOutputs(utility_output=out))  # 包装为EngineCoreOutputs
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)  # 调用工具方法
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:  # 如果是执行器失败
            raise RuntimeError("Executor failed.")  # 抛出运行时错误
        else:  # 其他未识别的请求类型
            logger.error(  # 记录错误
                "Unrecognized input request type encountered: %s", request_type  # 未识别的请求类型
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:  # 在关闭期间拒绝添加请求
        """在关闭期间拒绝添加请求，返回是否拒绝。"""
        if self.shutdown_state == EngineShutdownState.RUNNING:  # 如果仍在运行
            return False  # 不拒绝

        logger.info("Rejecting request %s (server shutting down)", request.request_id)  # 记录拒绝信息
        self._send_abort_outputs_to_client([request.request_id], request.client_index)  # 发送中止输出给客户端
        return True  # 已拒绝

    def _reject_utility_in_shutdown(  # 在关闭期间拒绝工具方法调用
        self, client_idx: int, call_id: int, method_name: str  # 客户端索引、调用ID、方法名
    ) -> bool:  # 返回是否拒绝
        """在关闭期间拒绝工具方法调用。"""
        if self.shutdown_state == EngineShutdownState.RUNNING:  # 如果仍在运行
            return False  # 不拒绝

        logger.warning("Rejecting utility call %s (server shutting down)", method_name)  # 记录警告
        output = UtilityOutput(call_id, failure_message="Server shutting down")  # 创建失败输出
        self.output_queue.put_nowait(  # 放入输出队列
            (client_idx, EngineCoreOutputs(utility_output=output))  # 包装输出
        )
        return True  # 已拒绝

    @staticmethod  # 静态方法装饰器
    def _invoke_utility_method(  # 调用工具方法
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable  # 方法名、获取结果函数、输出对象、入队函数
    ):
        """调用工具方法并处理结果或异常。"""
        try:  # 尝试执行
            result = get_result()  # 获取结果
            if isinstance(result, Future):  # 如果结果是Future
                # 将工具输出处理推迟到Future完成时。
                callback = lambda future: EngineCoreProc._invoke_utility_method(  # 创建回调
                    name, future.result, output, enqueue_output  # 递归调用自身
                )
                result.add_done_callback(callback)  # 添加完成回调
                return  # 返回等待Future完成
            output.result = UtilityResult(result)  # 设置结果
        except Exception as e:  # 捕获异常
            logger.exception("Invocation of %s method failed", name)  # 记录方法调用失败
            output.failure_message = f"Call to {name} method failed: {str(e)}"  # 设置失败消息
        enqueue_output(output)  # 将输出加入队列

    @staticmethod  # 静态方法装饰器
    def _convert_msgspec_args(method, args):  # 转换msgspec参数
        """如果提供的参数类型与目标方法参数类型不匹配，尝试转换为msgspec对象。"""
        if not args:  # 如果没有参数
            return args  # 直接返回
        arg_types = signature(method).parameters.values()  # 获取方法参数类型
        assert len(args) <= len(arg_types)  # 断言参数数量不超过方法参数数量
        return tuple(  # 返回转换后的参数元组
            msgspec.convert(v, type=p.annotation)  # 使用msgspec转换类型
            if isclass(p.annotation)  # 如果注解是类
            and issubclass(p.annotation, msgspec.Struct)  # 且是msgspec.Struct的子类
            and not isinstance(v, p.annotation)  # 且值不是该类型的实例
            else v  # 否则保持原值
            for v, p in zip(args, arg_types)  # 遍历参数和类型
        )

    def _send_engine_dead(self):  # 发送引擎死亡状态
        """向EngineCoreClient发送引擎死亡状态。"""

        # 将ENGINE_CORE_DEAD放入队列。
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)  # 放入死亡标记

        # 等待守护线程发送消息后再关闭。
        self.output_thread.join(timeout=5.0)  # 等待输出线程最多5秒
        if self.output_thread.is_alive():  # 如果输出线程仍存活
            logger.fatal(  # 记录致命错误
                "vLLM shutdown signal from EngineCore failed "  # 引擎核心关闭信号发送失败
                "to send. Please report this issue."  # 请报告此问题
            )

    # 处理输入套接字
    def process_input_sockets(
        self,
        input_addresses: list[str],  # 输入地址列表
        coord_input_address: str | None,  # 协调器输入地址（可选）
        identity: bytes,  # 身份标识
        ready_event: threading.Event,  # 就绪事件
    ):
        """输入套接字IO线程。"""

        # Msgpack序列化解码。
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)  # 创建添加请求解码器
        generic_decoder = MsgpackDecoder()  # 创建通用解码器

        with ExitStack() as stack, zmq.Context() as ctx:  # 使用退出栈和ZMQ上下文
            input_sockets = [  # 创建输入套接字列表
                stack.enter_context(  # 进入上下文管理
                    make_zmq_socket(  # 创建ZMQ套接字
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False  # DEALER模式，连接模式
                    )
                )
                for input_address in input_addresses  # 遍历每个输入地址
            ]
            if coord_input_address is None:  # 如果没有协调器输入地址
                coord_socket = None  # 协调器套接字为None
            else:  # 否则创建协调器套接字
                coord_socket = stack.enter_context(  # 进入上下文管理
                    make_zmq_socket(  # 创建ZMQ套接字
                        ctx,  # ZMQ上下文
                        coord_input_address,  # 协调器输入地址
                        zmq.XSUB,  # XSUB类型套接字
                        identity=identity,  # 身份标识
                        bind=False,  # 连接模式
                    )
                )
                # 向协调器发送订阅消息。
                coord_socket.send(b"\x01")  # 发送订阅消息

            # 向轮询器注册套接字。
            poller = zmq.Poller()  # 创建轮询器
            for input_socket in input_sockets:  # 遍历输入套接字
                # 向每个输入套接字发送初始消息——这在前端ROUTER套接字
                # 可以向我们发送输入消息之前是必需的。
                input_socket.send(b"")  # 发送空消息
                poller.register(input_socket, zmq.POLLIN)  # 注册为可读

            if coord_socket is not None:  # 如果有协调器套接字
                # 等待协调器的就绪消息。
                assert coord_socket.recv() == b"READY"  # 断言收到READY消息
                poller.register(coord_socket, zmq.POLLIN)  # 注册为可读

            ready_event.set()  # 设置就绪事件
            del ready_event  # 删除就绪事件引用
            while True:  # 主循环
                for input_socket, _ in poller.poll():  # 轮询所有套接字
                    # (请求类型, 请求数据)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)  # 接收多部分消息
                    # 注意(yongji): 忽略DP协调器发送的READY消息，
                    # 该消息用于通知新启动的引擎
                    if type_frame.buffer == b"READY":  # 如果是READY消息
                        assert input_socket == coord_socket  # 断言来自协调器套接字
                        continue  # 跳过
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))  # 解析请求类型

                    # 反序列化请求数据。
                    request: Any  # 请求变量声明
                    if request_type == EngineCoreRequestType.ADD:  # 如果是添加请求
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)  # 解码添加请求
                        try:  # 尝试预处理
                            request = self.preprocess_add_request(req)  # 预处理添加请求
                        except Exception:  # 捕获预处理异常
                            self._handle_request_preproc_error(req)  # 处理预处理错误
                            continue  # 跳过此请求
                    else:  # 其他请求类型
                        request = generic_decoder.decode(data_frames)  # 使用通用解码器解码

                        if request_type == EngineCoreRequestType.ABORT:  # 如果是中止请求
                            # 中止请求同时添加到两个队列，这允许我们急切地处理中止，
                            # 同时确保输入队列中的排序以避免泄漏请求。这是可以的，因为
                            # 调度器中的中止是幂等的。
                            self.aborts_queue.put_nowait(request)  # 将中止请求放入中止队列

                    # 推送到输入队列供核心繁忙循环处理。
                    self.input_queue.put_nowait((request_type, request))  # 放入输入队列

    # 处理输出套接字
    def process_output_sockets(
        self,
        output_paths: list[str],  # 输出路径列表
        coord_output_path: str | None,  # 协调器输出路径（可选）
        engine_index: int,  # 引擎索引
    ):
        """输出套接字IO线程。"""

        # Msgpack序列化编码。
        encoder = MsgpackEncoder()  # 创建编码器
        # 可复用的发送缓冲区。
        reuse_buffers: list[bytearray] = []  # 复用缓冲区列表
        # 保持对输出和缓冲区的引用，直到zmq使用完毕
        # （输出可能包含tensor/np数组，其底层缓冲区被提取用于零拷贝发送）。
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()  # 待处理的发送追踪器队列

        # 我们必须设置linger以确保ENGINE_CORE_DEAD
        # 消息在关闭套接字之前被发送。
        with ExitStack() as stack, zmq.Context() as ctx:  # 使用退出栈和ZMQ上下文
            sockets = [  # 创建输出套接字列表
                stack.enter_context(  # 进入上下文管理
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)  # 创建PUSH套接字
                )
                for output_path in output_paths  # 遍历输出路径
            ]
            coord_socket = (  # 创建协调器输出套接字
                stack.enter_context(  # 进入上下文管理
                    make_zmq_socket(  # 创建ZMQ套接字
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000  # PUSH模式，连接模式
                    )
                )
                if coord_output_path is not None  # 如果有协调器输出路径
                else None  # 否则为None
            )
            max_reuse_bufs = len(sockets) + 1  # 最大复用缓冲区数

            while True:  # 主循环
                output = self.output_queue.get()  # 从输出队列获取输出
                if output == EngineCoreProc.ENGINE_CORE_DEAD:  # 如果是引擎死亡标记
                    for socket in sockets:  # 向所有套接字发送死亡标记
                        socket.send(output)  # 发送死亡标记
                    break  # 退出循环
                assert not isinstance(output, bytes)  # 断言输出不是字节
                client_index, outputs = output  # 解包客户端索引和输出
                outputs.engine_index = engine_index  # 设置引擎索引

                if client_index == -1:  # 如果是协调器消息
                    # 不复用协调器消息的缓冲区，因为消息很小。
                    assert coord_socket is not None  # 断言协调器套接字存在
                    coord_socket.send_multipart(encoder.encode(outputs))  # 发送编码后的输出
                    continue  # 继续下一个

                # 回收zmq已使用完毕的缓冲区。
                while pending and pending[-1][0].done:  # 当有已完成的追踪器时
                    reuse_buffers.append(pending.pop()[2])  # 回收缓冲区

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()  # 获取或创建缓冲区
                buffers = encoder.encode_into(outputs, buffer)  # 编码输出到缓冲区
                tracker = sockets[client_index].send_multipart(  # 发送多部分消息
                    buffers, copy=False, track=True  # 零拷贝发送，启用追踪
                )
                if not tracker.done:  # 如果发送未完成
                    ref = outputs if len(buffers) > 1 else None  # 保持引用以防零拷贝
                    pending.appendleft((tracker, ref, buffer))  # 添加到待处理队列
                elif len(reuse_buffers) < max_reuse_bufs:  # 如果复用缓冲区未达上限
                    # 限制复用缓冲区的数量。
                    reuse_buffers.append(buffer)  # 回收缓冲区

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:  # 处理请求预处理错误
        """记录并返回请求作用域的错误响应，
        针对输入套接字处理线程中添加请求预处理时引发的异常。
        """
        logger.exception(  # 记录异常
            "Unexpected error pre-processing request %s", request.request_id  # 预处理请求时的意外错误
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)  # 发送错误输出给客户端

    def pause_scheduler(  # 暂停调度器（EngineCoreProc的覆盖版本）
        self, mode: PauseMode = "abort", clear_cache: bool = True  # 暂停模式和是否清除缓存
    ) -> Future | None:  # 返回Future或None
        """暂停生成；行为取决于模式。

        所有暂停模式都会排队新的添加——"abort"和"keep"跳过step()；
        "wait"允许step()以便进行中的请求可以排空。

        - ``abort``: 设置PAUSED_NEW，中止所有请求，等待中止输出发送，
          可选清除缓存，然后完成返回的Future。
        - ``wait``: 设置PAUSED_NEW（排队添加，继续步进）；排空后，
          可选清除缓存，然后完成返回的Future。
        - ``keep``: 设置PAUSED_ALL；返回一个在输出队列为空时完成的Future。
        """
        if mode not in ("keep", "abort", "wait"):  # 如果模式无效
            raise ValueError(f"Invalid pause mode: {mode}")  # 抛出值错误

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:  # 引擎空闲回调
            if clear_cache:  # 如果需要清除缓存
                engine._reset_caches()  # 重置缓存
            future.set_result(None)  # 设置Future结果

        if mode == "abort":  # 如果是中止模式
            aborted_reqs = self.scheduler.finish_requests(  # 中止所有请求
                None, RequestStatus.FINISHED_ABORTED  # 以中止状态完成
            )
            self._send_abort_outputs(aborted_reqs)  # 发送中止输出

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW  # 根据模式选择暂停状态
        self.scheduler.set_pause_state(pause_state)  # 设置调度器暂停状态
        if not self.has_work():  # 如果没有待处理的工作
            if clear_cache:  # 如果需要清除缓存
                self._reset_caches()  # 重置缓存
            return None  # 返回None

        future = Future[Any]()  # 创建Future对象
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))  # 添加空闲回调
        return future  # 返回Future

    def _send_finish_outputs_to_client(  # 向客户端发送完成输出
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason  # 请求ID列表、客户端索引、完成原因
    ) -> None:
        """向客户端发送带完成原因的输出。"""
        outputs = [  # 创建输出列表
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)  # 为每个请求ID创建输出
            for req_id in req_ids  # 遍历请求ID
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)  # 创建引擎核心输出集合
        self.output_queue.put_nowait((client_index, eco))  # 放入输出队列

    def _send_abort_outputs_to_client(  # 向客户端发送中止输出
        self, req_ids: list[str], client_index: int  # 请求ID列表和客户端索引
    ) -> None:
        """向客户端发送中止输出。"""
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)  # 以ABORT原因发送

    def _send_error_outputs_to_client(  # 向客户端发送错误输出
        self, req_ids: list[str], client_index: int  # 请求ID列表和客户端索引
    ) -> None:
        """向客户端发送错误输出。"""
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)  # 以ERROR原因发送

    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:  # 发送中止输出
        """向所有相关客户端发送中止输出。"""
        # TODO(nick) 这将被移到调度器内部
        if aborted_reqs:  # 如果有中止的请求
            # 将client_index映射到属于该客户端的request_ids列表。
            by_client = defaultdict[int, set[str]](set)  # 按客户端分组的字典
            for req_id, client_index in aborted_reqs:  # 遍历中止的请求
                by_client[client_index].add(req_id)  # 按客户端索引分组
            for client_index, req_ids in by_client.items():  # 遍历每个客户端
                self._send_abort_outputs_to_client(list(req_ids), client_index)  # 发送中止输出


class DPEngineCoreProc(EngineCoreProc):  # 数据并行引擎核心进程类
    """在数据并行上下文中，用于在后台进程中运行EngineCore的ZMQ包装器。"""

    def __init__(  # 初始化数据并行引擎核心进程
        self,
        vllm_config: VllmConfig,  # vLLM配置
        local_client: bool,  # 是否本地客户端
        handshake_address: str,  # 握手地址
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计
        client_handshake_address: str | None = None,  # 客户端握手地址（可选）
    ):
        assert vllm_config.model_config.is_moe, (  # 断言模型是MoE模型
            "DPEngineCoreProc should only be used for MoE models"  # 仅用于MoE模型
        )

        # 计数模型的前向传播次数，以便每N步与DP对等节点同步完成状态。
        self.step_counter = 0  # 步进计数器
        self.current_wave = 0  # 当前波次
        self.last_counts = (0, 0)  # 上次的请求计数

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState  # 导入弹性EP缩放状态

        self.eep_scaling_state: ElasticEPScalingState | None = None  # 弹性EP缩放状态

        # 初始化引擎。
        dp_rank = vllm_config.parallel_config.data_parallel_rank  # 获取DP rank
        super().__init__(  # 调用父类初始化
            vllm_config,  # vLLM配置
            local_client,  # 是否本地客户端
            handshake_address,  # 握手地址
            executor_class,  # 执行器类
            log_stats,  # 是否记录统计
            client_handshake_address,  # 客户端握手地址
            engine_index=dp_rank,  # 引擎索引为DP rank
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):  # 初始化数据并行
        """为数据并行配置GPU和无状态进程组。"""
        parallel_config = vllm_config.parallel_config  # 获取并行配置
        dp_rank = parallel_config.data_parallel_rank  # 获取DP rank
        dp_size = parallel_config.data_parallel_size  # 获取DP大小
        local_dp_rank = parallel_config.data_parallel_rank_local  # 获取本地DP rank

        assert dp_size > 1  # 断言DP大小大于1
        assert local_dp_rank is not None  # 断言本地DP rank不为None
        assert 0 <= local_dp_rank <= dp_rank < dp_size  # 断言rank范围有效

        self.dp_rank = dp_rank  # 保存DP rank
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)  # 初始化DP进程组
        self.dp_group, self.dp_store = dp_group, dp_store  # 保存DP进程组和存储

    def shutdown(self):  # 关闭数据并行引擎核心
        """关闭引擎核心并销毁DP进程组。"""
        super().shutdown()  # 调用父类关闭
        if dp_group := getattr(self, "dp_group", None):  # 如果有DP进程组
            stateless_destroy_torch_distributed_process_group(dp_group)  # 销毁DP进程组

    def add_request(self, request: Request, request_wave: int = 0):  # 添加请求（DP版本）
        """添加请求并处理波次同步。"""
        super().add_request(request, request_wave)  # 调用父类添加请求
        if self.has_coordinator and request_wave != self.current_wave:  # 如果有协调器且波次不同
            if request_wave > self.current_wave:  # 如果是更新的波次
                self.current_wave = request_wave  # 更新当前波次
            elif not self.engines_running:  # 如果引擎未运行
                # 收到已完成波次的请求，通知前端需要启动下一个波次。
                self.output_queue.put_nowait(  # 发送启动波次通知
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))  # 发送到协调器
                )

    def resume_scheduler(self):  # 恢复调度器（DP版本）
        """恢复调度器并唤醒其他DP引擎。"""
        super().resume_scheduler()  # 调用父类恢复调度器
        if (  # 如果满足以下条件
            self.has_coordinator  # 有协调器
            and not self.engines_running  # 且引擎未运行
            and self.scheduler.has_unfinished_requests()  # 且有未完成的请求
        ):
            # 唤醒其他DP引擎。
            self.output_queue.put_nowait(  # 发送启动波次通知
                (-1, EngineCoreOutputs(start_wave=self.current_wave))  # 发送到协调器
            )

    def _handle_client_request(  # 处理客户端请求（DP版本）
        self, request_type: EngineCoreRequestType, request: Any  # 请求类型和请求数据
    ) -> None:
        """分发来自客户端的请求，包含DP波次处理。"""
        if request_type == EngineCoreRequestType.START_DP_WAVE:  # 如果是启动DP波次请求
            new_wave, exclude_eng_index = request  # 解包新波次和排除的引擎索引
            if exclude_eng_index != self.engine_index and (  # 如果不是被排除的引擎
                new_wave >= self.current_wave  # 且新波次不小于当前波次
            ):
                self.current_wave = new_wave  # 更新当前波次
                if not self.engines_running:  # 如果引擎未运行
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)  # 记录启动空闲循环
                    self.engines_running = True  # 设置引擎运行标志
        else:  # 其他请求类型
            super()._handle_client_request(request_type, request)  # 调用父类处理

    def _maybe_publish_request_counts(self):  # 可能发布请求计数
        """如果请求计数发生变化，发布到协调器。"""
        if not self.publish_dp_lb_stats:  # 如果不发布DP负载均衡统计
            return  # 直接返回

        # 发布请求计数（如果有变化）。
        counts = self.scheduler.get_request_counts()  # 获取请求计数
        if counts != self.last_counts:  # 如果计数发生变化
            self.last_counts = counts  # 更新上次计数
            stats = SchedulerStats(  # 创建调度器统计
                *counts, step_counter=self.step_counter, current_wave=self.current_wave  # 传入计数、步进计数和波次
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))  # 发送统计到协调器

    def run_busy_loop(self):  # 运行繁忙循环（DP版本）
        """数据并行情况下EngineCore的核心繁忙循环。"""

        # 循环直到进程收到SIGINT或SIGTERM
        while self._handle_shutdown():  # 处理关闭请求
            # 1) 轮询输入队列直到有工作要做。
            self._process_input_queue()  # 处理输入队列

            if self.eep_scaling_state is not None:  # 如果有弹性EP缩放状态
                _ = self.eep_scaling_state.progress()  # 推进缩放进度
                if self.eep_scaling_state.is_complete():  # 如果缩放完成
                    self.process_input_queue_block = True  # 恢复输入队列阻塞
                    self.eep_scaling_state = None  # 清除缩放状态

            executed = self._process_engine_step()  # 执行引擎步进
            self._maybe_publish_request_counts()  # 可能发布请求计数

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()  # 检查是否有本地未完成请求
            if not executed:  # 如果模型未执行
                if not local_unfinished_reqs and not self.engines_running:  # 如果无未完成请求且引擎未运行
                    # 所有引擎空闲。
                    continue  # 继续下一轮循环

                # 我们处于运行状态，因此如果模型没有执行任何就绪请求，
                # 必须执行一个虚拟传播。
                self.execute_dummy_batch()  # 执行虚拟批次

            # 3) All-reduce操作以确定全局未完成请求。
            self.engines_running = self._has_global_unfinished_reqs(  # 检查全局未完成请求
                local_unfinished_reqs  # 传入本地状态
            )

            if not self.engines_running:  # 如果所有引擎都完成了
                if self.dp_rank == 0 or not self.has_coordinator:  # 如果是rank 0或没有协调器
                    # 通知客户端我们正在暂停循环。
                    logger.debug(  # 记录调试信息
                        "Wave %d finished, pausing engine loop.", self.current_wave  # 波次完成，暂停引擎循环
                    )
                    # 在协调器情况下，dp rank 0发送更新到协调器。
                    # 否则（离线SPMD情况），每个rank发送更新到其共置的前端进程。
                    client_index = -1 if self.has_coordinator else 0  # 确定客户端索引
                    self.output_queue.put_nowait(  # 发送波次完成通知
                        (
                            client_index,  # 客户端索引
                            EngineCoreOutputs(wave_complete=self.current_wave),  # 波次完成输出
                        )
                    )
                # 递增波次计数并重置步进计数器。
                self.current_wave += 1  # 递增波次
                self.step_counter = 0  # 重置步进计数器

        raise SystemExit  # 抛出系统退出异常

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:  # 检查全局未完成请求
        """通过all-reduce检查所有DP rank是否有未完成请求。"""
        # 优化——仅每32步执行一次完成同步all-reduce。
        self.step_counter += 1  # 递增步进计数器
        if self.step_counter % 32 != 0:  # 如果不是32的倍数
            return True  # 假设仍有未完成请求

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)  # 执行DP全局检查

    def reinitialize_distributed(  # 重新初始化分布式
        self, reconfig_request: ReconfigureDistributedRequest  # 重新配置请求
    ) -> None:
        """根据重新配置请求重新初始化分布式设置。"""
        from copy import deepcopy  # 导入深拷贝

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState  # 导入弹性EP缩放状态

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)  # 深拷贝并行配置
        old_dp_size = new_parallel_config.data_parallel_size  # 记录旧的DP大小
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size  # 设置新的DP大小
        if (  # 如果需要更新rank
            reconfig_request.new_data_parallel_rank  # 新的DP rank
            != ReconfigureRankType.KEEP_CURRENT_RANK  # 不是保持当前rank
        ):
            new_parallel_config.data_parallel_rank = (  # 设置新的DP rank
                reconfig_request.new_data_parallel_rank  # 新的DP rank值
            )
        new_parallel_config.data_parallel_master_ip = (  # 设置新的主节点IP
            reconfig_request.new_data_parallel_master_ip  # 新的主节点IP
        )
        new_parallel_config.data_parallel_master_port = (  # 设置新的主节点端口
            reconfig_request.new_data_parallel_master_port  # 新的主节点端口
        )
        new_parallel_config._data_parallel_master_port_list = (  # 设置新的主节点端口列表
            reconfig_request.new_data_parallel_master_port_list  # 新的端口列表
        )

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size  # 判断是否为缩容
        is_shutdown = (  # 判断是否关闭当前rank
            reconfig_request.new_data_parallel_rank  # 新的DP rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK  # 是否为关闭当前rank
        )

        self.eep_scaling_state = ElasticEPScalingState(  # 创建弹性EP缩放状态
            model_executor=self.model_executor,  # 模型执行器
            engine_core=self,  # 引擎核心
            vllm_config=self.vllm_config,  # vLLM配置
            new_parallel_config=new_parallel_config,  # 新的并行配置
            worker_type="removing" if is_shutdown else "existing",  # 工作类型：移除或现有
            scale_type="scale_down" if is_scale_down else "scale_up",  # 缩放类型：缩容或扩容
            reconfig_request=reconfig_request,  # 重新配置请求
        )
        self.process_input_queue_block = False  # 设置输入队列为非阻塞
        logger.info(  # 记录信息
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"  # 收到重新配置请求
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
