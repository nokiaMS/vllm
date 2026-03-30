# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os  # 导入操作系统接口模块
from collections.abc import Callable  # 导入可调用对象的抽象基类
from concurrent.futures import Future, ThreadPoolExecutor  # 导入异步任务的 Future 和线程池执行器
from functools import cached_property  # 导入缓存属性装饰器
from multiprocessing import Lock  # 导入多进程锁
from typing import Any  # 导入通用类型注解

import torch  # 导入 PyTorch 框架
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块

import vllm.envs as envs  # 导入 vLLM 环境变量配置
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.platforms import current_platform  # 导入当前平台信息
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port  # 导入网络工具函数（分布式初始化方法、获取 IP 和空闲端口）
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 导入调度器输出相关类型
from vllm.v1.executor.abstract import Executor  # 导入执行器抽象基类
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput  # 导入模型运行输出相关类型
from vllm.v1.serial_utils import run_method  # 导入序列化工具中的方法调用函数
from vllm.v1.worker.worker_base import WorkerWrapperBase  # 导入 Worker 包装器基类

# ============================================================================
# 单进程执行器模块
# 本文件实现了两种轻量级执行器：
#   1. UniProcExecutor：在主进程内直接运行单个 Worker，无需进程间通信，
#      适用于单 GPU 推理场景，是最简单高效的执行方案。
#   2. ExecutorWithExternalLauncher：继承自 UniProcExecutor，设计用于
#      torchrun 等外部启动器的张量并行场景。每个引擎进程独立运行一个 Worker，
#      通过确定性调度保证多引擎输出一致，无需额外的状态同步。
#
# 设计特点：
#   - collective_rpc 直接在当前进程内调用 Worker 方法，无序列化/反序列化开销
#   - 支持异步输出线程以实现 async scheduling（max_concurrent_batches=2）
#   - single_value 参数控制返回单值还是列表，适配不同调用场景
# ============================================================================

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# UniProcExecutor：单进程执行器，Worker 与引擎运行在同一进程中。
# 适用于单设备推理，直接调用 Worker 方法无需 IPC 通信。
class UniProcExecutor(Executor):
    """单进程执行器类，继承自 Executor 抽象基类。
    Worker 直接在主进程中运行，无需跨进程通信，适用于单 GPU 推理场景。"""

    def _init_executor(self) -> None:
        """初始化执行器：创建 Worker 实例、设置分布式参数、加载模型。"""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)  # 创建 rank=0 的 Worker 包装器实例
        distributed_init_method, rank, local_rank = self._distributed_args()  # 获取分布式初始化参数
        kwargs = dict(  # 构建 Worker 初始化参数字典
            vllm_config=self.vllm_config,  # vLLM 配置对象
            local_rank=local_rank,  # 本地设备排名
            rank=rank,  # 全局排名
            distributed_init_method=distributed_init_method,  # 分布式初始化方法
            is_driver_worker=True,  # 标记为驱动 Worker
            shared_worker_lock=Lock(),  # 创建共享的多进程锁
        )

        self.async_output_thread: ThreadPoolExecutor | None = None  # 初始化异步输出线程为 None
        if self.max_concurrent_batches > 1:  # 如果支持多批次并发（异步调度）
            self.async_output_thread = ThreadPoolExecutor(  # 创建单线程的线程池用于异步输出处理
                max_workers=1, thread_name_prefix="WorkerAsyncOutput"  # 线程名前缀设为 WorkerAsyncOutput
            )

        is_eep_new_worker = envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH  # 检查是否为弹性专家并行扩容启动模式
        self.driver_worker.init_worker(all_kwargs=[kwargs])  # 初始化 Worker 实例
        if not is_eep_new_worker:  # 如果不是弹性扩容启动模式，执行常规初始化流程
            self.driver_worker.init_device()  # 初始化计算设备（GPU）
            self.driver_worker.load_model()  # 加载推理模型
            current_platform.update_block_size_for_backend(self.vllm_config)  # 根据后端更新内存块大小

    # 获取分布式初始化参数：单进程模式下 rank=0，使用本地 IP 和随机端口
    def _distributed_args(self) -> tuple[str, int, int]:
        """获取分布式初始化参数，返回 (分布式初始化方法, 全局排名, 本地排名) 三元组。"""
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())  # 根据本地 IP 和空闲端口生成分布式初始化方法
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")  # 将设备字符串按冒号分割（如 "cuda:0"）
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0  # 如果指定了设备索引则使用该索引，否则默认为 0
        return distributed_init_method, 0, local_rank  # 返回分布式初始化方法、rank=0 和本地排名

    @cached_property  # 使用缓存属性装饰器，计算一次后缓存结果
    def max_concurrent_batches(self) -> int:
        """获取最大并发批次数：异步调度时为 2，否则为 1。"""
        return 2 if self.scheduler_config.async_scheduling else 1  # 异步调度模式返回 2，同步模式返回 1

    # collective_rpc 的单进程实现：直接在当前进程中调用 Worker 方法。
    # non_block=True 时，若结果为 AsyncModelRunnerOutput 则提交到异步线程处理；
    # 否则立即包装为已完成的 Future。single_value 控制返回单值或列表。
    def collective_rpc(  # type: ignore[override]  # 覆盖父类方法，忽略类型检查警告
        self,
        method: str | Callable,  # 要调用的方法名或可调用对象
        timeout: float | None = None,  # 超时时间（单进程模式下未使用）
        args: tuple = (),  # 位置参数元组
        kwargs: dict | None = None,  # 关键字参数字典
        non_block: bool = False,  # 是否非阻塞执行
        single_value: bool = False,  # 是否返回单值而非列表
    ) -> Any:
        """集体远程过程调用的单进程实现，直接调用 Worker 方法。
        non_block=True 时返回 Future 对象，False 时直接返回结果。"""
        if kwargs is None:  # 如果未提供关键字参数
            kwargs = {}  # 初始化为空字典

        if not non_block:  # 如果是阻塞模式
            result = run_method(self.driver_worker, method, args, kwargs)  # 直接调用 Worker 方法并获取结果
            return result if single_value else [result]  # 根据 single_value 返回单值或列表

        try:  # 非阻塞模式的处理
            result = run_method(self.driver_worker, method, args, kwargs)  # 调用 Worker 方法获取结果
            if isinstance(result, AsyncModelRunnerOutput):  # 如果结果是异步模型输出
                if (async_thread := self.async_output_thread) is not None:  # 如果存在异步输出线程池
                    if single_value:  # 如果需要返回单值
                        return async_thread.submit(result.get_output)  # 提交获取输出的任务到线程池

                    def get_output_list() -> list[Any]:  # 定义获取输出列表的内部函数
                        return [result.get_output()]  # 将输出包装为列表返回

                    return async_thread.submit(get_output_list)  # 提交获取输出列表的任务到线程池
                result = result.get_output()  # 如果没有异步线程池，同步获取输出
            future = Future[Any]()  # 创建一个新的 Future 对象
            future.set_result(result if single_value else [result])  # 设置 Future 的结果值
        except Exception as e:  # 如果执行过程中发生异常
            future = Future[Any]()  # 创建一个新的 Future 对象
            future.set_exception(e)  # 将异常设置到 Future 中
        return future  # 返回 Future 对象

    def execute_model(  # type: ignore[override]  # 覆盖父类方法，忽略类型检查警告
        self, scheduler_output: SchedulerOutput, non_block: bool = False  # 接收调度器输出和非阻塞标志
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """执行模型推理：根据调度器输出运行模型前向传播，返回推理结果。"""
        output = self.collective_rpc(  # 通过集体 RPC 调用 Worker 的 execute_model 方法
            "execute_model",  # 调用的方法名
            args=(scheduler_output,),  # 传入调度器输出作为参数
            non_block=non_block,  # 是否非阻塞执行
            single_value=True,  # 返回单值
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():  # 非阻塞模式下，如果任务已完成
            # Raise the exception in-line if the task failed.
            output.result()  # 获取结果以便尽早抛出异常
        return output  # 返回推理输出结果

    def sample_tokens(  # type: ignore[override]  # 覆盖父类方法，忽略类型检查警告
        self, grammar_output: GrammarOutput | None, non_block: bool = False  # 接收语法输出和非阻塞标志
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """采样令牌：根据语法约束输出进行令牌采样，返回采样结果。"""
        return self.collective_rpc(  # 通过集体 RPC 调用 Worker 的 sample_tokens 方法
            "sample_tokens",  # 调用的方法名
            args=(grammar_output,),  # 传入语法输出作为参数
            non_block=non_block,  # 是否非阻塞执行
            single_value=True,  # 返回单值
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """获取草稿令牌 ID：用于投机解码（speculative decoding）中获取草稿模型生成的令牌。"""
        return self.collective_rpc("take_draft_token_ids", single_value=True)  # 调用 Worker 的 take_draft_token_ids 方法

    def check_health(self) -> None:
        """健康检查：单进程执行器只要在运行就是健康的，无需额外检查。"""
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return  # 直接返回，单进程模式下始终健康

    def shutdown(self) -> None:
        """关闭执行器：停止 Worker 并释放相关资源。"""
        if worker := self.driver_worker:  # 如果驱动 Worker 存在
            worker.shutdown()  # 调用 Worker 的关闭方法


# ExecutorWithExternalLauncher：为 torchrun 等外部启动器设计的执行器。
# 核心思想：每个进程独立运行一个 Worker，通过 env:// 方式初始化分布式通信，
# 依赖确定性调度保证所有引擎生成相同输出，从而避免显式的状态同步。
# determine_available_memory 通过 all_reduce 取最小值确保所有 rank 一致。
class ExecutorWithExternalLauncher(UniProcExecutor):
    """外部启动器执行器类，继承自 UniProcExecutor。
    专为 torchrun 等外部启动器设计，用于张量并行的离线推理。
    每个引擎独立运行一个 Worker，通过确定性调度保证所有引擎输出一致。"""

    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """初始化外部启动器执行器：验证多进程模式已禁用，然后调用父类初始化。"""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (  # 断言多进程模式未启用，确定性执行要求单进程
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()  # 调用父类的初始化方法

    # 从 torchrun 设置的环境变量中读取 rank 和分布式初始化方法
    def _distributed_args(self) -> tuple[str, int, int]:
        """从 torchrun 环境变量获取分布式参数，返回 (分布式初始化方法, 全局排名, 本地排名)。"""
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"  # 使用环境变量方式进行分布式初始化
        rank = int(os.environ["RANK"])  # 从环境变量获取全局排名
        local_rank = int(os.environ["LOCAL_RANK"])  # 从环境变量获取本地排名
        return distributed_init_method, rank, local_rank  # 返回分布式初始化方法和排名信息

    # 获取可用显存：通过 all_reduce MIN 操作取所有 rank 的最小值，
    # 确保 KV 缓存分配在各 rank 间保持一致。
    def determine_available_memory(self) -> list[int]:  # in bytes  # 返回可用内存（字节）
        """确定可用显存：通过 all_reduce MIN 操作在所有 rank 间取最小值，确保一致性。"""
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()  # 调用父类方法获取本地可用内存
        from vllm.distributed.parallel_state import get_world_group  # 导入获取全局通信组的函数

        cpu_group = get_world_group().cpu_group  # 获取 CPU 通信组
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)  # 将内存值转换为 CPU 上的张量
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)  # 在所有 rank 间执行 MIN 归约操作
        return [memory_tensor.item()]  # 将张量结果转换为 Python 整数并返回
