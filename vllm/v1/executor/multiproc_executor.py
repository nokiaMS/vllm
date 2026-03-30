# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing  # 导入多进程库
import os  # 导入操作系统接口库
import pickle  # 导入pickle序列化库
import queue  # 导入线程安全队列库
import signal  # 导入信号处理库
import threading  # 导入线程库
import time  # 导入时间库
import traceback  # 导入异常堆栈追踪库
import weakref  # 导入弱引用库
from collections import deque  # 导入双端队列
from collections.abc import Callable, Sequence  # 导入可调用和序列抽象类型
from concurrent.futures import Future, InvalidStateError  # 导入并发Future类和无效状态异常
from contextlib import suppress  # 导入异常抑制上下文管理器
from dataclasses import dataclass  # 导入数据类装饰器
from enum import Enum, auto  # 导入枚举基类和自动值
from functools import cached_property, partial  # 导入缓存属性和偏函数
from multiprocessing.connection import Connection  # 导入多进程连接类型
from multiprocessing.process import BaseProcess  # 导入多进程基类
from multiprocessing.synchronize import Lock as LockType  # 导入多进程锁类型
from threading import Thread  # 导入线程类
from typing import Any, cast  # 导入类型提示工具

import cloudpickle  # 导入cloudpickle序列化库（支持lambda等复杂对象）
import torch  # 导入PyTorch

import vllm.envs as envs  # 导入vLLM环境变量模块
from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel  # 导入分布式环境销毁函数
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue  # 导入共享内存消息队列和句柄
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator  # 导入KV输出聚合器
from vllm.distributed.parallel_state import (  # 从分布式并行状态模块导入
    get_dcp_group,  # 获取DCP并行组
    get_dp_group,  # 获取数据并行组
    get_ep_group,  # 获取专家并行组
    get_inner_dp_world_group,  # 获取DP内世界组
    get_pcp_group,  # 获取预填充上下文并行组
    get_pp_group,  # 获取流水线并行组
    get_tp_group,  # 获取张量并行组
    model_parallel_is_initialized,  # 检查模型并行是否已初始化
)
from vllm.envs import enable_envs_cache  # 导入环境变量缓存启用函数
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.platforms import current_platform  # 导入当前平台对象
from vllm.tracing import instrument, maybe_init_worker_tracer  # 导入追踪装饰器和Worker追踪初始化
from vllm.utils.network_utils import (  # 从网络工具模块导入
    get_distributed_init_method,  # 获取分布式初始化方法
    get_ip,  # 获取本机IP
    get_loopback_ip,  # 获取回环地址
    get_open_port,  # 获取可用端口
)
from vllm.utils.system_utils import (  # 从系统工具模块导入
    _maybe_force_spawn,  # 可能强制使用spawn多进程方式
    decorate_logs,  # 装饰日志输出
    get_mp_context,  # 获取多进程上下文
    set_process_title,  # 设置进程标题
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 导入语法输出和调度器输出
from vllm.v1.executor.abstract import Executor, FailureCallback  # 导入执行器基类和失败回调类型
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput  # 导入模型运行输出类型
from vllm.v1.worker.worker_base import WorkerWrapperBase  # 导入Worker包装基类

# ============================================================================
# 多进程执行器模块（MultiprocExecutor）
# 本文件实现了基于 Python multiprocessing 的分布式执行后端，是 vLLM 在单机多卡
# 场景下的默认高性能执行方案。
#
# 核心架构：
#   1. MultiprocExecutor（主进程端）：负责创建和管理 Worker 子进程，通过共享内存
#      消息队列（MessageQueue）广播调度指令并收集执行结果。
#   2. WorkerProc（子进程端）：每个 GPU 对应一个 WorkerProc 进程，内部运行
#      WorkerWrapperBase 完成实际的模型加载和推理计算。
#   3. 通信机制：使用基于共享内存的 MessageQueue 实现零拷贝的高效 RPC 通信，
#      支持广播（主进程→所有 Worker）和汇报（Worker→主进程）两个方向。
#   4. 支持流水线并行（PP）和张量并行（TP），通过 output_rank 优化只从
#      最后一个流水线阶段的首个 TP Worker 收集输出。
#   5. 异步调度：支持非阻塞 RPC 调用，通过 FutureWrapper 和队列实现
#      请求流水线化，避免主进程阻塞等待。
# ============================================================================

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# FutureWrapper 扩展了标准 Future，实现了按队列顺序排空（drain）的语义。
# 当调用 result() 时，会先依序完成队列中排在前面的所有 Future，
# 确保在流水线场景下结果按发送顺序被消费。
class FutureWrapper(Future):  # Future包装类，支持FIFO排空
    def __init__(  # 构造函数
        self,  # 实例自身
        futures_queue: deque[tuple["FutureWrapper", Callable]],  # 共享的Future队列引用
        aggregate: Callable = lambda x: x,  # 结果聚合函数，默认透传
    ):
        self.futures_queue = futures_queue  # 保存队列引用
        self.aggregate = aggregate  # 保存聚合函数
        super().__init__()  # 调用父类Future构造函数

    def result(self, timeout=None):  # 获取结果，先排空队列中前面的Future
        if timeout is not None:  # 不支持超时参数
            raise RuntimeError("timeout not implemented")  # 抛出运行时异常
        # Drain any futures ahead of us in the queue.
        while not self.done():  # 当前Future未完成时，处理队列中排在前面的
            future, get_response = self.futures_queue.pop()  # 从队尾弹出（FIFO顺序）
            future.wait_for_response(get_response)  # 等待该Future的响应
        return super().result()  # 返回当前Future的结果

    def wait_for_response(self, get_response: Callable):  # 等待响应并设置Future结果
        try:  # 尝试获取响应
            response = self.aggregate(get_response())  # 调用响应获取函数并聚合结果
            with suppress(InvalidStateError):  # 忽略Future已完成的异常
                self.set_result(response)  # 设置成功结果
        except Exception as e:  # 捕获异常
            with suppress(InvalidStateError):  # 忽略Future已完成的异常
                self.set_exception(e)  # 设置异常结果


# MultiprocExecutor：基于多进程的执行器实现。
# 设计要点：
#   - 主进程通过 rpc_broadcast_mq 向所有 Worker 广播 RPC 请求
#   - 每个 Worker 通过独立的 response_mq 返回执行结果
#   - 支持多节点部署（nnodes_within_dp > 1）时的跨节点消息队列
#   - 内置 Worker 健康监控线程，Worker 异常退出时自动触发关闭流程
#   - 使用 weakref.finalize 确保进程退出时自动清理资源
class MultiprocExecutor(Executor):  # 多进程执行器，管理Worker子进程
    supports_pp: bool = True  # 支持流水线并行

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):  # 构造函数
        self.monitor_workers = monitor_workers  # 是否监控Worker进程
        super().__init__(vllm_config)  # 调用父类构造函数

    # 初始化执行器：创建 Worker 进程、建立共享内存消息队列、启动健康监控。
    # 整体流程：验证并行配置 → 设置多进程环境 → 创建广播消息队列 →
    # 逐一启动 Worker 子进程 → 等待所有 Worker 就绪 → 建立响应消息队列。
    def _init_executor(self) -> None:  # 初始化执行器
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)  # 注册GC终结器，退出时自动清理
        self.is_failed = False  # 执行器失败标志
        self.failure_callback: FailureCallback | None = None  # 失败回调函数

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()  # 获取并行度配置
        assert self.world_size == tp_size * pp_size * pcp_size, (  # 断言world_size等于TP*PP*PCP
            f"world_size ({self.world_size}) must be equal to the "  # 错误消息
            f"tensor_parallel_size ({tp_size}) x pipeline"  # TP大小
            f"_parallel_size ({pp_size}) x prefill_context"  # PP大小
            f"_parallel_size ({pcp_size}). "  # PCP大小
        )

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()  # 设置多进程Worker环境变量

        # use the loopback address get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(  # 获取分布式初始化方法URL
            get_loopback_ip(), get_open_port()  # 使用回环地址和可用端口
        )
        self.rpc_broadcast_mq: MessageQueue | None = None  # RPC广播消息队列，初始为None
        scheduler_output_handle: Handle | None = None  # 调度器输出句柄，初始为None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:  # 如果是DP组的leader节点
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024  # 计算最大块大小（字节）
            mq_connect_ip = get_ip()  # 获取本机IP用于消息队列连接
            logger.info(  # 记录leader节点信息
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "  # 日志格式
                "master_addr=%s, mq_connect_ip=%s (local), "  # master地址
                "world_size=%d, local_world_size=%d",  # 世界大小
                self.parallel_config.node_rank,  # 节点rank
                self.parallel_config.node_rank_within_dp,  # DP内节点rank
                self.parallel_config.master_addr,  # master地址
                mq_connect_ip,  # 消息队列连接IP
                self.world_size,  # 全局世界大小
                self.local_world_size,  # 本地世界大小
            )
            self.rpc_broadcast_mq = MessageQueue(  # 创建RPC广播消息队列
                self.world_size,  # 全局世界大小
                self.local_world_size,  # 本地世界大小
                max_chunk_bytes=max_chunk_bytes,  # 最大块大小
                connect_ip=mq_connect_ip,  # 连接IP
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()  # 导出共享内存句柄
        # Create workers
        context = get_mp_context()  # 获取多进程上下文
        shared_worker_lock = context.Lock()  # 创建共享锁（Worker间同步用）
        unready_workers: list[UnreadyWorkerProcHandle] = []  # 未就绪Worker句柄列表
        success = False  # 初始化成功标志
        try:  # 尝试创建Worker进程
            global_start_rank = (  # 计算全局起始rank
                self.local_world_size * self.parallel_config.node_rank_within_dp  # 本地大小 × 节点rank
            )
            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = (  # fork模式下追踪继承的文件描述符
                [] if context.get_start_method() == "fork" else None  # fork模式才需要
            )

            for local_rank in range(self.local_world_size):  # 遍历本地所有rank，一个local rank代表一个gpu。
                global_rank = global_start_rank + local_rank  # 计算全局rank，全局rank是gpu在全局的唯一编号。
                is_driver_worker = self._is_driver_worker(global_rank)  # 判断是否为driver Worker
                unready_worker_handle = WorkerProc.make_worker_process(  # 创建Worker子进程
                    vllm_config=self.vllm_config,  # vLLM配置
                    local_rank=local_rank,  # 本地rank
                    rank=global_rank,  # 全局rank
                    distributed_init_method=distributed_init_method,  # 分布式初始化方法
                    input_shm_handle=scheduler_output_handle,  # 共享内存输入句柄
                    shared_worker_lock=shared_worker_lock,  # 共享锁
                    is_driver_worker=is_driver_worker,  # 是否为driver
                    inherited_fds=inherited_fds,  # 继承的文件描述符列表
                )
                unready_workers.append(unready_worker_handle)  # 添加到未就绪列表
                if inherited_fds is not None:  # 如果在追踪继承fd
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())  # 记录死亡管道fd
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())  # 记录就绪管道fd

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)  # 等待所有Worker就绪

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:  # 如果需要监控Worker
                self.start_worker_monitor()  # 启动Worker健康监控线程

            self.response_mqs = []  # 初始化响应消息队列列表
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:  # leader节点建立响应队列
                for rank in range(self.world_size):  # 遍历所有rank
                    if rank < self.local_world_size:  # 本地rank
                        local_message_queue = self.workers[rank].worker_response_mq  # 获取本地Worker响应队列
                        assert local_message_queue is not None  # 断言队列存在
                        self.response_mqs.append(local_message_queue)  # 添加到响应队列列表
                    else:  # 远程rank
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[  # 获取远程Worker响应队列
                            rank  # 远程rank索引
                        ]
                        assert remote_message_queue is not None  # 断言队列存在
                        self.response_mqs.append(remote_message_queue)  # 添加到响应队列列表

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:  # 如果广播队列存在
                self.rpc_broadcast_mq.wait_until_ready()  # 等待广播队列就绪
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:  # 遍历所有响应队列
                response_mq.wait_until_ready()  # 等待每个响应队列就绪

            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()  # 创建Future排空队列

            self._post_init_executor()  # 执行后初始化钩子

            success = True  # 标记初始化成功
        finally:  # 最终清理
            if not success:  # 如果初始化失败
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:  # 遍历未就绪Worker
                    if uw.death_writer is not None:  # 如果死亡管道存在
                        uw.death_writer.close()  # 关闭死亡管道，通知Worker退出
                        uw.death_writer = None  # 清除引用
                self._ensure_worker_termination([uw.proc for uw in unready_workers])  # 确保所有Worker进程终止

        self.output_rank = self._get_output_rank()  # 计算输出rank

    # 获取并行度配置：返回 (TP大小, PP大小, PCP大小)，同时设置 world_size 和 local_world_size
    def _get_parallel_sizes(self) -> tuple[int, int, int]:  # 获取并行度配置
        self.world_size = self.parallel_config.world_size  # 设置全局世界大小
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (  # 断言可被节点数整除
            f"global world_size ({self.parallel_config.world_size}) must be "  # 错误消息
            f"divisible by nnodes_within_dp "  # 节点数说明
            f"({self.parallel_config.nnodes_within_dp}). "  # 具体值
        )
        self.local_world_size = self.parallel_config.local_world_size  # 设置本地世界大小
        tp_size = self.parallel_config.tensor_parallel_size  # 张量并行大小
        pp_size = self.parallel_config.pipeline_parallel_size  # 流水线并行大小
        pcp_size = self.parallel_config.prefill_context_parallel_size  # 预填充上下文并行大小
        return tp_size, pp_size, pcp_size  # 返回三元组

    def _post_init_executor(self) -> None:  # 后初始化钩子（子类可覆盖）
        pass  # 默认空实现

    def _is_driver_worker(self, rank: int) -> bool:  # 判断是否为driver Worker（每个TP组的rank 0）
        return rank % self.parallel_config.tensor_parallel_size == 0  # TP rank为0即为driver

    # 启动 Worker 健康监控：在后台线程中监听所有 Worker 进程的存活状态，
    # 任何 Worker 异常退出时会触发执行器关闭并调用失败回调通知引擎。
    def start_worker_monitor(self, inline=False) -> None:  # 启动Worker健康监控
        workers = self.workers  # 获取Worker列表
        self_ref = weakref.ref(self)  # 创建弱引用避免循环引用

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():  # 监控Worker进程存活的内部函数
            sentinels = [h.proc.sentinel for h in workers]  # 收集所有Worker进程的哨兵描述符
            died = multiprocessing.connection.wait(sentinels)  # 等待任一进程退出
            _self = self_ref()  # 获取执行器强引用
            if not _self or getattr(_self, "shutting_down", False):  # 如果执行器已销毁或正在关闭
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")  # 记录调试日志
                return  # 直接返回
            _self.is_failed = True  # 标记执行器为失败状态
            proc_name = next(h.proc.name for h in workers if h.proc.sentinel == died[0])  # 找到死亡进程名称
            logger.error(  # 记录错误日志
                "Worker proc %s died unexpectedly, shutting down executor.", proc_name  # 进程名称
            )
            _self.shutdown()  # 关闭执行器
            callback = _self.failure_callback  # 获取失败回调
            if callback is not None:  # 如果有回调
                _self.failure_callback = None  # 清除回调（只调用一次）
                callback()  # 执行失败回调

        if not inline:  # 非内联模式，在后台线程中运行
            Thread(  # 创建守护线程
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"  # 线程目标和名称
            ).start()  # 启动线程
            return  # 返回

        monitor_workers()  # 内联模式，直接在当前线程运行

    def register_failure_callback(self, callback: FailureCallback):  # 注册失败回调
        if self.is_failed:  # 如果已经失败
            callback()  # 立即调用回调
        else:  # 否则
            self.failure_callback = callback  # 保存回调待后续使用

    # 重写 execute_model：通过 collective_rpc 分发模型执行请求，
    # 仅从 output_rank（最后PP阶段的首个TP Worker）收集输出以减少通信开销。
    def execute_model(  # type: ignore[override]  # 执行模型推理
        self, scheduler_output: SchedulerOutput, non_block: bool = False  # 调度器输出和是否非阻塞
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:  # 返回输出或Future
        return self.collective_rpc(  # 通过collective_rpc分发到所有Worker
            "execute_model",  # 调用Worker的execute_model方法
            args=(scheduler_output,),  # 传入调度器输出
            unique_reply_rank=self.output_rank,  # 只从output_rank收集结果
            non_block=non_block,  # 是否非阻塞
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,  # 超时时间
            kv_output_aggregator=self.kv_output_aggregator,  # KV输出聚合器
        )

    def sample_tokens(  # type: ignore[override]  # 采样token
        self, grammar_output: GrammarOutput | None, non_block: bool = False  # 语法输出和是否非阻塞
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:  # 返回输出或Future
        return self.collective_rpc(  # 通过collective_rpc分发
            "sample_tokens",  # 调用Worker的sample_tokens方法
            args=(grammar_output,),  # 传入语法输出
            unique_reply_rank=self.output_rank,  # 只从output_rank收集结果
            non_block=non_block,  # 是否非阻塞
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,  # 超时时间
            kv_output_aggregator=self.kv_output_aggregator,  # KV输出聚合器
        )

    def execute_dummy_batch(self) -> None:  # 执行虚拟批次
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)  # 广播到所有Worker

    def take_draft_token_ids(self) -> DraftTokenIds | None:  # 获取草稿token ID
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        return self.collective_rpc(  # 从output_rank获取结果
            "take_draft_token_ids", unique_reply_rank=self.output_rank  # 只从指定rank收集
        )

    # collective_rpc 的多进程实现：
    # 1. 通过 rpc_broadcast_mq 将方法名/参数广播给所有 Worker
    # 2. 从指定的 response_mq 中收集响应（unique_reply_rank 可指定只收单个 Worker 的结果）
    # 3. 支持 KV 输出聚合（kv_output_aggregator）以合并多 Worker 输出
    # 4. non_block=True 时返回 FutureWrapper，pending 的 Future 按 FIFO 排空
    def collective_rpc(  # type: ignore[override]  # 集合RPC调用实现
        self,  # 实例自身
        method: str | Callable,  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间（秒）
        args: tuple = (),  # 位置参数
        kwargs: dict | None = None,  # 关键字参数
        non_block: bool = False,  # 是否非阻塞
        unique_reply_rank: int | None = None,  # 只从指定rank收集结果
        kv_output_aggregator: KVOutputAggregator | None = None,  # KV输出聚合器
    ) -> Any:  # 返回结果
        """Returns single result if unique_reply_rank and/or kv_output_aggregator
        is provided, otherwise list."""
        assert self.rpc_broadcast_mq is not None, (  # 断言广播队列存在（仅leader节点可调用）
            "collective_rpc should not be called on follower node"  # 错误消息
        )
        if self.is_failed:  # 如果执行器已失败
            raise RuntimeError("Executor failed.")  # 抛出运行时异常

        deadline = None if timeout is None else time.monotonic() + timeout  # 计算截止时间
        kwargs = kwargs or {}  # 默认空字典

        if kv_output_aggregator is not None:  # 如果有KV输出聚合器
            output_rank = None  # 需要从所有rank收集结果
            aggregate: Callable[[Any], Any] = partial(  # 创建聚合函数
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0  # 指定输出rank
            )
        else:  # 无聚合器
            output_rank = unique_reply_rank  # 直接使用指定的输出rank
            aggregate = lambda x: x  # 透传结果

        if isinstance(method, str):  # 如果是方法名字符串
            send_method = method  # 直接发送
        else:  # 如果是可调用对象
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)  # 序列化为字节
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))  # 广播RPC请求

        response_mqs: Sequence[MessageQueue] = self.response_mqs  # 获取响应队列列表
        if output_rank is not None:  # 如果指定了输出rank
            response_mqs = (response_mqs[output_rank],)  # 只从该rank的队列收集

        def get_response():  # 获取响应的内部函数
            responses = []  # 收集响应列表
            for mq in response_mqs:  # 遍历需要收集的响应队列
                dequeue_timeout = (  # 计算剩余超时时间
                    None if deadline is None else (deadline - time.monotonic())  # 距截止时间的剩余
                )
                try:  # 尝试出队
                    status, result = mq.dequeue(timeout=dequeue_timeout)  # 从消息队列取出响应
                except TimeoutError as e:  # 超时异常
                    raise TimeoutError(f"RPC call to {method} timed out.") from e  # 抛出超时异常
                if status != WorkerProc.ResponseStatus.SUCCESS:  # 如果Worker返回失败
                    raise RuntimeError(  # 抛出运行时异常
                        f"Worker failed with error '{result}', please check the"  # 错误消息
                        " stack trace above for the root cause"  # 提示查看堆栈
                    )
                responses.append(result)  # 添加成功响应
            return responses[0] if output_rank is not None else responses  # 单rank返回单个，否则返回列表

        if non_block:  # 非阻塞模式
            future = FutureWrapper(self.futures_queue, aggregate=aggregate)  # 创建FutureWrapper
            self.futures_queue.appendleft((future, get_response))  # 加入Future队列
            return future  # 返回Future

        # First drain any pending futures in the queue.
        while self.futures_queue:  # 先排空队列中的pending Future
            future, get_fut_response = self.futures_queue.pop()  # 弹出最早的Future
            future.wait_for_response(get_fut_response)  # 等待并设置其结果

        return aggregate(get_response())  # 同步获取并聚合响应

    # 确保所有 Worker 进程被终止：先等待自行退出，再发 SIGTERM，最后发 SIGKILL。
    # 采用渐进式策略避免强制终止导致资源泄漏。
    @staticmethod  # 静态方法装饰器
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):  # 确保所有Worker进程终止
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):  # 等待进程终止的内部函数
            if not time:  # 如果time模块在解释器关闭时被置为None
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)  # 检查所有进程是否已退出
            start_time = time.time()  # 记录开始时间
            while time.time() - start_time < timeout:  # 在超时时间内循环检查
                if all(not proc.is_alive() for proc in procs):  # 所有进程已退出
                    return True  # 返回成功
                time.sleep(0.1)  # 短暂休眠100ms
            return False  # 超时返回失败

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]  # 获取仍存活的进程列表
        # Give processes time to clean themselves up properly first
        logger.debug("Worker Termination: allow workers to gracefully shutdown")  # 记录调试日志
        if wait_for_termination(active_procs(), 4):  # 等待4秒让进程自行退出
            return  # 全部退出则返回

        # Send SIGTERM if still running
        logger.debug("Worker Termination: workers still running sending SIGTERM")  # 记录发送SIGTERM
        for p in active_procs():  # 遍历仍存活的进程
            p.terminate()  # 发送SIGTERM信号
        if not wait_for_termination(active_procs(), 4):  # 再等4秒
            # Send SIGKILL if still running
            logger.debug(  # 记录发送SIGKILL
                "Worker Termination: resorting to SIGKILL to take down workers"  # 最终手段
            )
            for p in active_procs():  # 遍历仍存活的进程
                p.kill()  # 发送SIGKILL强制终止

    # 优雅关闭执行器：先关闭 death_writer 通知 Worker 退出，
    # 等待进程终止后再清理消息队列资源。
    def shutdown(self):  # 优雅关闭执行器
        """Properly shut down the executor and its workers"""
        if not getattr(self, "shutting_down", False):  # 如果尚未开始关闭
            logger.debug("Triggering shutdown of workers")  # 记录关闭开始
            self.shutting_down = True  # 标记正在关闭

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):  # 获取Worker列表
                for w in workers:  # 遍历所有Worker
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:  # 如果死亡管道存在
                        w.death_writer.close()  # 关闭管道，通知Worker退出
                        w.death_writer = None  # 清除引用
                self._ensure_worker_termination([w.proc for w in workers])  # 确保所有Worker终止

                for w in workers:  # 遍历所有Worker
                    # Shutdown response queues
                    if w.worker_response_mq is not None:  # 如果响应队列存在
                        w.worker_response_mq.shutdown()  # 关闭响应队列
                        w.worker_response_mq = None  # 清除引用

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):  # 获取广播队列
            rpc_broadcast_mq.shutdown()  # 关闭广播队列
            self.rpc_broadcast_mq = None  # 清除引用
        if response_mqs := getattr(self, "response_mqs", None):  # 获取响应队列列表
            for mq in response_mqs:  # 遍历所有响应队列
                mq.shutdown()  # 关闭队列
            self.response_mqs = []  # 清空列表

    def check_health(self) -> None:  # 健康检查
        self.collective_rpc("check_health", timeout=10)  # 广播健康检查RPC，10秒超时
        return  # 返回

    @cached_property  # 缓存属性装饰器
    def max_concurrent_batches(self) -> int:  # 最大并发批次数
        # PP requires PP-size concurrent batches to fill the pipeline.
        pp_size = self.parallel_config.pipeline_parallel_size  # 获取PP大小
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size  # 异步调度且无PP时为2，否则为PP大小

    # 计算输出 rank：返回最后一个 PP 阶段中 TP rank=0 的 Worker 的全局 rank。
    # 只有该 Worker 需要返回 ModelRunnerOutput，其他 Worker 的输出被忽略。
    def _get_output_rank(self) -> int:  # 计算输出rank
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return (  # 最后PP阶段的首个TP Worker
            self.world_size  # 全局世界大小
            - self.parallel_config.tensor_parallel_size  # 减去TP大小
            * self.parallel_config.prefill_context_parallel_size  # 乘以PCP大小
        )


# UnreadyWorkerProcHandle：Worker 进程在完成初始化之前的句柄，
# 包含进程对象、就绪管道和死亡通知管道。
@dataclass  # 数据类装饰器
class UnreadyWorkerProcHandle:  # Worker未就绪时的进程句柄
    """WorkerProcess handle before READY."""

    proc: BaseProcess  # Worker进程对象
    rank: int  # Worker的全局rank
    ready_pipe: Connection  # 就绪通知管道（子进程→父进程）
    death_writer: Connection | None = None  # 死亡检测管道（父进程端，关闭时通知子进程）


# WorkerProcHandle：Worker 进程就绪后的完整句柄，
# 包含进程对象、响应消息队列（本地和远程）等通信资源。
@dataclass  # 数据类装饰器
class WorkerProcHandle:  # Worker就绪后的完整进程句柄
    proc: BaseProcess  # Worker进程对象
    rank: int  # Worker的全局rank
    # The worker process writes to this MQ in single-node mode
    worker_response_mq: MessageQueue | None  # Worker响应消息队列（单节点模式）
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]  # 远程Worker响应消息队列列表（多节点模式）
    death_writer: Connection | None = None  # 死亡检测管道

    @classmethod  # 类方法装饰器
    def from_unready_handle(  # 从未就绪句柄创建完整句柄
        cls,  # 类自身
        unready_handle: UnreadyWorkerProcHandle,  # 未就绪句柄
        worker_response_mq: MessageQueue | None,  # 响应消息队列
        peer_worker_response_mqs: list[MessageQueue | None],  # 远程响应队列列表
    ) -> "WorkerProcHandle":  # 返回完整句柄
        return cls(  # 创建实例
            proc=unready_handle.proc,  # 复用进程对象
            rank=unready_handle.rank,  # 复用rank
            worker_response_mq=worker_response_mq,  # 设置响应队列
            peer_worker_response_mqs=peer_worker_response_mqs,  # 设置远程响应队列
            death_writer=unready_handle.death_writer,  # 复用死亡管道
        )


# WorkerProc：在独立子进程中运行的 Worker 封装类。
# 每个 WorkerProc 实例对应一个 GPU，负责：
#   1. 初始化分布式环境和模型加载
#   2. 建立与主进程的共享内存消息队列通信
#   3. 运行 busy loop 不断接收 RPC 请求并执行
#   4. 监控父进程存活状态（death pipe），父进程退出时自动清理
class WorkerProc:  # Worker进程封装类
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"  # 就绪信号字符串常量
    rpc_broadcast_mq: MessageQueue | None  # RPC广播消息队列（接收调度指令）
    worker_response_mq: MessageQueue | None  # 响应消息队列（发送执行结果）

    # 初始化消息队列：单节点模式下直接创建本地共享内存队列；
    # 多节点模式下通过分布式组创建跨节点消息队列。
    def _init_message_queues(  # 初始化消息队列
        self, input_shm_handle: Handle, vllm_config: VllmConfig  # 共享内存句柄和配置
    ) -> None:  # 无返回值
        if vllm_config.parallel_config.nnodes_within_dp == 1:  # 单节点模式
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(  # 从句柄创建广播接收队列
                input_shm_handle, self.worker.rank  # 传入句柄和rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)  # 创建1对1的本地响应队列
            self.peer_response_handles = []  # 单节点无远程句柄
        else:  # 多节点模式
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(  # 创建跨节点广播队列
                external_writer_handle=input_shm_handle,  # 外部写入句柄
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                blocking=False,  # 非阻塞模式，稍后握手
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            self.worker_response_mq, self.peer_response_handles = (  # 创建跨节点响应队列
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(  # 单读者多写者模式
                    reader_rank_in_group=0  # 读者为rank 0（driver）
                )
            )

    # WorkerProc 构造函数：初始化 Worker 实例、加载模型、建立消息队列。
    # 在子进程中被调用，完成设备初始化和模型权重加载。
    @instrument(span_name="Worker init")  # 追踪装饰器，记录Worker初始化耗时
    def __init__(  # WorkerProc构造函数
        self,  # 实例自身
        vllm_config: VllmConfig,  # vLLM配置对象
        local_rank: int,  # 本地rank
        rank: int,  # 全局rank
        distributed_init_method: str,  # 分布式初始化方法URL
        input_shm_handle: Handle,  # 共享内存输入句柄
        shared_worker_lock: LockType,  # 共享锁（Worker间同步）
        is_driver_worker: bool,  # 是否为driver Worker
    ):
        self.rank = rank  # 保存全局rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)  # 创建Worker包装实例
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [  # 构建所有Worker的初始化参数列表
            {} for _ in range(vllm_config.parallel_config.world_size)  # 每个rank一个空字典
        ]
        all_kwargs[local_rank] = {  # 只设置当前rank的参数
            "vllm_config": vllm_config,  # vLLM配置
            "local_rank": local_rank,  # 本地rank
            "rank": rank,  # 全局rank
            "distributed_init_method": distributed_init_method,  # 分布式初始化方法
            "is_driver_worker": is_driver_worker,  # 是否为driver
            "shared_worker_lock": shared_worker_lock,  # 共享锁
        }
        wrapper.init_worker(all_kwargs)  # 初始化Worker
        self.worker = wrapper  # 保存Worker包装实例

        scheduler_config = vllm_config.scheduler_config  # 获取调度器配置
        self.use_async_scheduling = scheduler_config.async_scheduling  # 是否使用异步调度
        if self.use_async_scheduling:  # 如果启用异步调度
            self.async_output_queue: queue.Queue = queue.Queue()  # 创建异步输出队列
            self.async_output_copy_thread = Thread(  # 创建异步输出处理线程
                target=self.async_output_busy_loop,  # 目标函数
                daemon=True,  # 守护线程
                name="WorkerAsyncOutputCopy",  # 线程名称
            )
            self.async_output_copy_thread.start()  # 启动异步输出线程

        self.setup_proc_title_and_log_prefix(  # 设置进程标题和日志前缀
            enable_ep=vllm_config.parallel_config.enable_expert_parallel  # 是否启用专家并行
        )

        # Load model
        is_eep_new_worker = envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH  # 是否为弹性EP扩容启动的新Worker
        if not is_eep_new_worker:  # 如果不是弹性扩容的新Worker
            self.worker.init_device()  # 初始化设备（GPU）
            # Update process title now that parallel groups are initialized
            self.setup_proc_title_and_log_prefix(  # 更新进程标题（并行组已初始化）
                enable_ep=vllm_config.parallel_config.enable_expert_parallel  # 是否启用专家并行
            )
            self.worker.load_model()  # 加载模型权重

        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)  # 根据注意力后端更新块大小

        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        self._init_message_queues(input_shm_handle, vllm_config)  # 初始化消息队列

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()  # 启用环境变量缓存

    # 创建 Worker 子进程的工厂方法：设置管道通信（就绪通知和死亡检测），
    # 启动守护进程运行 worker_main 入口。返回未就绪的进程句柄。
    @staticmethod  # 静态方法装饰器
    def make_worker_process(  # 创建Worker子进程的工厂方法
        vllm_config: VllmConfig,  # vLLM配置
        local_rank: int,  # 本地rank
        rank: int,  # 全局rank
        distributed_init_method: str,  # 分布式初始化方法
        input_shm_handle,  # Receive SchedulerOutput  # 共享内存输入句柄
        shared_worker_lock: LockType,  # 共享锁
        is_driver_worker: bool,  # 是否为driver
        inherited_fds: list[int] | None = None,  # fork模式下需要关闭的继承fd
    ) -> UnreadyWorkerProcHandle:  # 返回未就绪句柄
        context = get_mp_context()  # 获取多进程上下文
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)  # 创建单向就绪管道
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)  # 创建单向死亡检测管道
        if inherited_fds is not None:  # 如果在追踪继承fd
            inherited_fds = inherited_fds.copy()  # 拷贝列表避免修改原始
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))  # 添加当前管道fd
        process_kwargs = {  # 构建子进程参数字典
            "vllm_config": vllm_config,  # vLLM配置
            "local_rank": local_rank,  # 本地rank
            "rank": rank,  # 全局rank
            "distributed_init_method": distributed_init_method,  # 分布式初始化方法
            "input_shm_handle": input_shm_handle,  # 共享内存句柄
            "ready_pipe": ready_writer,  # 就绪管道写端（传给子进程）
            "death_pipe": death_reader,  # 死亡管道读端（传给子进程）
            "shared_worker_lock": shared_worker_lock,  # 共享锁
            "is_driver_worker": is_driver_worker,  # 是否为driver
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],  # 需要在子进程中关闭的fd
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(  # 创建子进程
            target=WorkerProc.worker_main,  # 目标入口函数
            kwargs=process_kwargs,  # 传入参数
            name=f"VllmWorker-{rank}",  # 进程名称
            daemon=True,  # 守护进程
        )

        proc.start()  # 启动子进程
        # Close child ends of pipes here in the parent
        ready_writer.close()  # 父进程关闭就绪管道写端
        death_reader.close()  # 父进程关闭死亡管道读端
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)  # 返回未就绪句柄

    @staticmethod  # 静态方法装饰器
    def wait_for_response_handle_ready(  # 等待响应句柄就绪并创建WorkerProcHandle
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle  # 句柄字典和未就绪句柄
    ) -> WorkerProcHandle:  # 返回完整句柄
        response_handle = handles["handle"]  # 获取响应队列句柄
        worker_response_mq: MessageQueue | None = None  # 初始化本地响应队列
        if len(response_handle.local_reader_ranks) > 0:  # 如果有本地读者
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)  # 从句柄创建消息队列
        peer_response_handles = handles["peer_response_handles"]  # 获取远程响应句柄列表
        peer_worker_response_mqs = [  # 创建远程响应队列列表
            MessageQueue.create_from_handle(handle, -1)  # 从句柄创建远程消息队列
            if handle.remote_subscribe_addr is not None  # 如果有远程订阅地址
            else None  # 否则为None
            for handle in peer_response_handles  # 遍历所有远程句柄
        ]
        return WorkerProcHandle.from_unready_handle(  # 从未就绪句柄创建完整句柄
            proc_handle,  # 未就绪句柄
            worker_response_mq,  # 本地响应队列
            peer_worker_response_mqs=peer_worker_response_mqs,  # 远程响应队列列表
        )

    # 等待所有 Worker 进程初始化完成：通过 ready_pipe 接收就绪信号，
    # 建立响应消息队列连接。任何 Worker 失败都会抛出异常。
    @staticmethod  # 静态方法装饰器
    def wait_for_ready(  # 等待所有Worker就绪
        unready_proc_handles: list[UnreadyWorkerProcHandle],  # 未就绪句柄列表
    ) -> list[WorkerProcHandle]:  # 返回完整句柄列表
        e = Exception(  # 预创建初始化失败异常
            "WorkerProc initialization failed due to an exception in a "  # 错误消息
            "background process. See stack trace for root cause."  # 提示查看堆栈
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}  # 管道→句柄映射
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(  # 按rank索引的就绪句柄列表
            unready_proc_handles  # 长度与未就绪列表相同
        )
        while pipes:  # 循环直到所有Worker就绪
            ready = multiprocessing.connection.wait(pipes.keys())  # 等待任一管道可读
            for pipe in ready:  # 遍历就绪的管道
                assert isinstance(pipe, Connection)  # 断言为Connection类型
                try:  # 尝试读取就绪信号
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)  # 从待处理映射中移除
                    response: dict[str, Any] = pipe.recv()  # 接收Worker发送的就绪消息
                    if response["status"] != "READY":  # 如果不是READY状态
                        raise e  # 抛出初始化失败异常

                    idx = unready_proc_handle.rank % len(ready_proc_handles)  # 计算在列表中的索引
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(  # 创建完整句柄
                        response, unready_proc_handle  # 传入响应和未就绪句柄
                    )
                except EOFError:  # 管道关闭（Worker进程异常退出）
                    e.__suppress_context__ = True  # 抑制异常链
                    raise e from None  # 抛出初始化失败异常

                finally:  # 最终清理
                    # Close connection.
                    pipe.close()  # 关闭就绪管道

        return cast(list[WorkerProcHandle], ready_proc_handles)  # 类型转换并返回

    def shutdown(self):  # Worker关闭清理
        if self.rpc_broadcast_mq is not None:  # 如果广播队列存在
            self.rpc_broadcast_mq.shutdown()  # 关闭广播队列
        if self.worker_response_mq is not None:  # 如果响应队列存在
            self.worker_response_mq.shutdown()  # 关闭响应队列
        self.worker.shutdown()  # 关闭Worker实例
        self.rpc_broadcast_mq = None  # 清除广播队列引用
        self.worker_response_mq = None  # 清除响应队列引用
        destroy_model_parallel()  # 销毁模型并行组
        destroy_distributed_environment()  # 销毁分布式环境

    # 监控死亡管道：在后台线程中检测父进程是否退出，
    # 父进程退出时关闭消息队列以触发 Worker 的优雅终止。
    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):  # 监控死亡管道
        if death_pipe is None:  # 如果无死亡管道
            return  # 直接返回

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):  # 死亡管道监控内部函数
            try:  # 尝试读取管道
                # This will block until parent process exits (pipe closes)
                death_pipe.recv()  # 阻塞等待，父进程退出时管道关闭
            except EOFError:  # 父进程退出，管道关闭
                logger.info_once("Parent process exited, terminating worker queues")  # 记录日志
                shutdown_requested.set()  # 设置关闭请求标志
                for mq in queues_to_shutdown:  # 遍历需要关闭的队列
                    if mq is not None:  # 如果队列存在
                        mq.shutdown()  # 关闭队列
            except Exception as e:  # 捕获其他异常
                logger.warning("Death monitoring error: %s", e)  # 记录警告日志

        # Pass queue references directly to avoid gc issues if passing self
        Thread(  # 创建守护线程
            target=death_pipe_monitor,  # 目标函数
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),  # 传入需要关闭的队列
            daemon=True,  # 守护线程
            name="DeathPipeMonitor",  # 线程名称
        ).start()  # 启动线程

    # Worker 进程的主入口：完成初始化后进入 busy loop 持续处理 RPC 请求。
    # 设置信号处理器以支持优雅终止（SIGTERM/SIGINT），
    # 异常退出时通过消息队列通知主进程。
    @staticmethod  # 静态方法装饰器
    def worker_main(*args, **kwargs):  # Worker进程主入口
        """Worker initialization and execution loops.
        This runs a background process"""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = threading.Event()  # 关闭请求事件标志

        def signal_handler(signum, frame):  # 信号处理器
            nonlocal shutdown_requested  # 引用外部变量
            if not shutdown_requested.is_set():  # 如果尚未请求关闭
                shutdown_requested.set()  # 设置关闭标志
                logger.debug(  # 记录调试日志
                    "WorkerProc handling signal %d, raising SystemExit", signum  # 信号编号
                )
                raise SystemExit()  # 抛出SystemExit以终止进程

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)  # 注册SIGTERM信号处理器
        signal.signal(signal.SIGINT, signal_handler)  # 注册SIGINT信号处理器

        worker = None  # Worker实例初始为None
        ready_writer = kwargs.pop("ready_pipe")  # 从参数中提取就绪管道写端
        death_pipe = kwargs.pop("death_pipe", None)  # 从参数中提取死亡管道读端

        # Close inherited pipes from parent (incl. other worker pipes)
        # Explicitly passing in existing pipes and closing them makes the pipe
        # behave when using fork. Otherwise, a hidden reference to the pipes
        # exist in the child process and prevents EOF closure.
        for fd in kwargs.pop("inherited_fds", []):  # 遍历需要关闭的继承fd
            try:  # 尝试关闭
                os.close(fd)  # 关闭文件描述符
            except Exception as e:  # 捕获异常
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)  # 记录警告

        try:  # 尝试初始化和运行Worker
            # Initialize tracer
            rank = kwargs.get("rank", 0)  # 获取rank参数
            maybe_init_worker_tracer(  # 初始化Worker追踪器
                instrumenting_module_name="vllm.worker",  # 模块名
                process_kind="worker",  # 进程类型
                process_name=f"Worker_{rank}",  # 进程名称
            )

            worker = WorkerProc(*args, **kwargs)  # 创建WorkerProc实例
            assert worker.worker_response_mq is not None  # 断言响应队列已初始化

            worker.monitor_death_pipe(death_pipe, shutdown_requested)  # 启动死亡管道监控

            # Send READY once we know everything is loaded
            ready_writer.send(  # 发送就绪信号给父进程
                {
                    "status": WorkerProc.READY_STR,  # 状态为READY
                    "handle": worker.worker_response_mq.export_handle(),  # 导出响应队列句柄
                    "peer_response_handles": worker.peer_response_handles,  # 远程响应句柄列表
                }
            )

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            if worker.rpc_broadcast_mq is not None:  # 如果广播队列存在
                worker.rpc_broadcast_mq.wait_until_ready()  # 等待广播队列就绪
            worker.worker_response_mq.wait_until_ready()  # 等待响应队列就绪
            ready_writer.close()  # 关闭就绪管道
            ready_writer = None  # 清除引用

            worker.worker_busy_loop()  # 进入主循环处理RPC请求

        except Exception:  # 捕获普通异常
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:  # 如果就绪管道未关闭（初始化阶段失败）
                logger.exception("WorkerProc failed to start.")  # 记录启动失败异常
            elif shutdown_requested.is_set():  # 如果已请求关闭
                logger.info("WorkerProc shutting down.")  # 记录正在关闭
            else:  # 运行时异常
                logger.exception("WorkerProc failed.")  # 记录运行失败异常

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested.set()  # 设置关闭标志

        except SystemExit as e:  # 捕获SystemExit（SIGTERM/SIGKILL触发）
            # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
            # the graceful shutdown process did not succeed
            logger.warning("WorkerProc was terminated")  # 记录Worker被终止
            # SystemExit must never be ignored
            raise e  # 重新抛出SystemExit

        finally:  # 最终清理
            if ready_writer is not None:  # 如果就绪管道仍打开
                ready_writer.close()  # 关闭就绪管道
            if death_pipe is not None:  # 如果死亡管道存在
                death_pipe.close()  # 关闭死亡管道
            # Clean up once worker exits busy loop
            if worker is not None:  # 如果Worker已创建
                worker.shutdown()  # 关闭Worker

    class ResponseStatus(Enum):  # 响应状态枚举，用于标记Worker返回结果的成功/失败
        SUCCESS = auto()  # 成功状态
        FAILURE = auto()  # 失败状态

    def enqueue_output(self, output: Any):  # 将Worker输出打包并放入响应消息队列
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):  # 如果是异步模型输出
            output = output.get_output()  # 提取实际输出结果

        if isinstance(output, Exception):  # 如果输出是异常
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))  # 标记为失败并将异常转为字符串
        else:  # 正常输出
            result = (WorkerProc.ResponseStatus.SUCCESS, output)  # 标记为成功
        if (response_mq := self.worker_response_mq) is not None:  # 如果响应队列存在
            response_mq.enqueue(result)  # 将结果放入共享内存响应队列

    def handle_output(self, output: Any):  # 处理Worker的输出结果
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:  # 如果启用了异步调度
            self.async_output_queue.put(output)  # 将输出放入异步队列，由专门线程处理
        else:  # 同步模式
            self.enqueue_output(output)  # 直接放入响应消息队列

    def async_output_busy_loop(self):  # 异步输出处理线程的入口函数
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:  # 无限循环等待输出
            output = self.async_output_queue.get()  # 阻塞等待异步输出队列中的数据
            self.enqueue_output(output)  # 将输出打包并放入响应消息队列

    # Worker 的主循环：不断从广播消息队列中取出 RPC 请求，
    # 解析方法名并在 Worker 实例上调用，将结果或异常通过响应队列返回。
    # output_rank 机制允许只有指定 rank 的 Worker 发送响应。
    def worker_busy_loop(self):  # Worker主循环：持续处理RPC请求
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None  # 确保广播消息队列已初始化
        while True:  # 无限循环处理请求
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(  # 从广播队列中取出RPC请求
                indefinite=True  # 无限期等待直到有消息到达
            )
            try:
                if isinstance(method, str):  # 如果方法名是字符串
                    func = getattr(self.worker, method)  # 通过反射获取Worker上的方法
                elif isinstance(method, bytes):  # 如果方法是序列化的字节（cloudpickle）
                    func = partial(cloudpickle.loads(method), self.worker)  # 反序列化并绑定Worker实例

                output = func(*args, **kwargs)  # 调用目标方法并获取输出
            except Exception as e:  # 捕获执行过程中的所有异常
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):  # 如果Python 3.11+支持异常注释
                    e.add_note(traceback.format_exc())  # 将完整堆栈信息附加到异常
                logger.exception("WorkerProc hit an exception.")  # 记录异常日志
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:  # 如果需要返回输出（output_rank匹配）
                    self.handle_output(e)  # 将异常作为输出处理
                continue  # 继续处理下一个请求

            if output_rank is None or self.rank == output_rank:  # 只有指定rank的Worker发送响应
                self.handle_output(output)  # 处理正常输出

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:  # 设置进程标题和日志前缀
        # Check if parallel groups are initialized first
        if not model_parallel_is_initialized():  # 如果并行组尚未初始化
            # Parallel groups not yet initialized, use default process name
            set_process_title(name="Worker")  # 使用默认进程名"Worker"
            decorate_logs("Worker")  # 设置默认日志前缀
            return  # 直接返回

        dp_size = get_dp_group().world_size  # 获取数据并行组大小
        dp_rank = get_dp_group().rank_in_group  # 获取当前进程在DP组中的排名
        pp_size = get_pp_group().world_size  # 获取流水线并行组大小
        pp_rank = get_pp_group().rank_in_group  # 获取当前进程在PP组中的排名
        pcp_size = get_pcp_group().world_size  # 获取PCP（流水线通信并行）组大小
        pcp_rank = get_pcp_group().rank_in_group  # 获取当前进程在PCP组中的排名
        tp_size = get_tp_group().world_size  # 获取张量并行组大小
        tp_rank = get_tp_group().rank_in_group  # 获取当前进程在TP组中的排名
        dcp_size = get_dcp_group().world_size  # 获取DCP（数据通信并行）组大小
        dcp_rank = get_dcp_group().rank_in_group  # 获取当前进程在DCP组中的排名
        process_name = "Worker"  # 初始进程名为"Worker"
        if dp_size > 1:  # 如果启用了数据并行
            process_name += f"_DP{dp_rank}"  # 追加DP排名后缀
        if pp_size > 1:  # 如果启用了流水线并行
            process_name += f"_PP{pp_rank}"  # 追加PP排名后缀
        if pcp_size > 1:  # 如果启用了PCP并行
            process_name += f"_PCP{pcp_rank}"  # 追加PCP排名后缀
        if tp_size > 1:  # 如果启用了张量并行
            process_name += f"_TP{tp_rank}"  # 追加TP排名后缀
        if dcp_size > 1:  # 如果启用了DCP并行
            process_name += f"_DCP{dcp_rank}"  # 追加DCP排名后缀
        if enable_ep:  # 如果启用了专家并行
            ep_rank = get_ep_group().rank_in_group  # 获取EP排名
            process_name += f"_EP{ep_rank}"  # 追加EP排名后缀
        set_process_title(name=process_name)  # 设置进程标题（如Worker_DP0_TP1）
        decorate_logs(process_name)  # 设置日志前缀以区分不同Worker


# 设置多进程 Worker 环境变量：强制使用 spawn 方式创建子进程，
# 并限制 OpenMP 线程数以避免多进程场景下的 CPU 资源争抢。
def set_multiprocessing_worker_envs():  # 设置多进程Worker环境变量
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()  # 强制使用spawn方式创建子进程（避免fork的安全问题）

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1  # 默认OpenMP线程数设为1
    if (  # 如果未手动设置OMP_NUM_THREADS且当前线程数超过默认值
        "OMP_NUM_THREADS" not in os.environ
        and (current_parallelism := torch.get_num_threads()) > default_omp_num_threads
    ):
        logger.warning(  # 记录警告：正在降低线程并行度以避免CPU争抢
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism,
            default_omp_num_threads,
        )
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)  # 设置环境变量限制OpenMP线程数
        torch.set_num_threads(default_omp_num_threads)  # 同步设置PyTorch的线程数
