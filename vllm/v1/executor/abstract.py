# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time  # 导入时间模块，用于性能计时
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections.abc import Callable  # 导入可调用对象类型
from concurrent.futures import Future  # 导入 Future 类型，用于异步结果
from functools import cached_property  # 导入缓存属性装饰器
from typing import TYPE_CHECKING, Literal, TypeVar, overload  # 导入类型注解工具

from vllm.config import VllmConfig  # 导入 vLLM 总配置类
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator  # 导入 KV 输出聚合器
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # 导入 KV 连接器握手元数据类型
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.lora.request import LoRARequest  # 导入 LoRA 请求类
from vllm.tasks import SupportedTask  # 导入支持的任务类型枚举
from vllm.tracing import instrument  # 导入链路追踪装饰器
from vllm.utils.import_utils import resolve_obj_by_qualname  # 导入按全限定名解析对象的工具函数
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 导入语法输出和调度器输出类型
from vllm.v1.engine import ReconfigureDistributedRequest  # 导入分布式重配置请求类型
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec  # 导入 KV 缓存配置和规格接口
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput  # 导入草稿 token ID 和模型运行输出类型
from vllm.v1.worker.worker_base import WorkerBase  # 导入 Worker 基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase  # 导入 KV 连接器基类

# ============================================================================
# Executor 抽象基类模块
# 本文件定义了 vLLM v1 执行器层的核心抽象接口 Executor。
# 设计思路：
#   1. Executor 是模型执行的统一入口，屏蔽了底层单进程/多进程/Ray 等不同执行后端的差异。
#   2. 通过 get_class() 静态工厂方法，根据 VllmConfig 中的 distributed_executor_backend
#      配置项动态选择合适的执行器实现（uni/mp/ray/external_launcher）。
#   3. collective_rpc() 是核心通信原语，所有 Worker 上的方法调用都通过此接口完成，
#      支持同步和非阻塞两种模式。
#   4. 提供了模型执行（execute_model）、采样（sample_tokens）、KV 缓存管理、
#      LoRA 适配器管理、sleep/wake_up 等完整的生命周期管理接口。
# ============================================================================

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

_R = TypeVar("_R")  # 定义泛型类型变量，用于 collective_rpc 的返回类型

# 失败回调函数类型，当执行器进入不可恢复的失败状态时被调用
FailureCallback = Callable[[], None]  # 定义失败回调的类型别名


# Executor 抽象基类：定义了所有执行器实现必须遵循的统一接口。
# 子类包括 UniProcExecutor（单进程）、MultiprocExecutor（多进程）、
# RayDistributedExecutor（Ray 分布式）等。
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    # 静态工厂方法：根据配置选择并返回合适的 Executor 子类。
    # 支持 "ray"、"mp"（多进程）、"uni"（单进程）、"external_launcher"（外部启动器）
    # 以及通过全限定类名字符串或直接传入类对象来指定自定义执行器。
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:  # 根据配置返回对应的执行器类
        executor_class: type[Executor]  # 声明执行器类变量
        parallel_config = vllm_config.parallel_config  # 获取并行配置
        distributed_executor_backend = parallel_config.distributed_executor_backend  # 获取分布式执行后端配置
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):  # 如果后端配置是一个类类型
            if not issubclass(distributed_executor_backend, Executor):  # 检查是否为 Executor 的子类
                raise TypeError(  # 类型不匹配时抛出异常
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend  # 直接使用传入的类
        elif distributed_executor_backend == "ray":  # 使用 Ray 分布式后端
            from vllm.v1.executor.ray_executor import RayDistributedExecutor  # 延迟导入 Ray 执行器

            executor_class = RayDistributedExecutor  # 设置为 Ray 执行器类
        elif distributed_executor_backend == "mp":  # 使用多进程后端
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor  # 延迟导入多进程执行器

            executor_class = MultiprocExecutor  # 设置为多进程执行器类
        elif distributed_executor_backend == "uni":  # 使用单进程后端
            from vllm.v1.executor.uniproc_executor import UniProcExecutor  # 延迟导入单进程执行器

            executor_class = UniProcExecutor  # 设置为单进程执行器类
        elif distributed_executor_backend == "external_launcher":  # 使用外部启动器后端
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher  # 设置为外部启动器执行器类
        elif isinstance(distributed_executor_backend, str):  # 如果是字符串形式的全限定类名
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)  # 通过全限定名解析类
            if not issubclass(executor_class, Executor):  # 检查解析出的类是否为 Executor 子类
                raise TypeError(  # 类型不匹配时抛出异常
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:  # 未知的后端配置
            raise ValueError(  # 抛出值错误异常
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class  # 返回选定的执行器类

    # 构造函数：从 VllmConfig 中提取各子配置并保存为实例属性，
    # 然后调用子类实现的 _init_executor() 完成具体初始化。
    # 同时初始化 sleep 状态跟踪和 KV 输出聚合器。
    @instrument(span_name="Executor init")  # 添加链路追踪 span
    def __init__(
        self,
        vllm_config: VllmConfig,  # 接收 vLLM 总配置对象
    ) -> None:
        self.vllm_config = vllm_config  # 保存总配置
        self.model_config = vllm_config.model_config  # 保存模型配置
        self.cache_config = vllm_config.cache_config  # 保存缓存配置
        self.lora_config = vllm_config.lora_config  # 保存 LoRA 配置
        self.load_config = vllm_config.load_config  # 保存模型加载配置
        self.parallel_config = vllm_config.parallel_config  # 保存并行配置
        self.scheduler_config = vllm_config.scheduler_config  # 保存调度器配置
        self.device_config = vllm_config.device_config  # 保存设备配置
        self.speculative_config = vllm_config.speculative_config  # 保存推测解码配置
        self.observability_config = vllm_config.observability_config  # 保存可观测性配置
        self._init_executor()  # 调用子类实现的初始化方法
        self.is_sleeping = False  # 初始化 sleep 状态为未睡眠
        self.sleeping_tags: set[str] = set()  # 初始化睡眠标签集合为空
        self.kv_output_aggregator: KVOutputAggregator | None = None  # 初始化 KV 输出聚合器为空

    # 抽象方法：由子类实现，完成执行器的具体初始化（如创建 Worker 进程、建立通信通道等）
    @abstractmethod
    def _init_executor(self) -> None:  # 子类必须实现此方法
        raise NotImplementedError  # 基类调用时抛出未实现错误

    # 根据 KV 缓存配置初始化所有 Worker 的缓存，并启动模型编译/预热流程。
    # 编译时间取各 Worker 的最大值回传给主进程配置。
    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:  # 根据 KV 缓存配置初始化
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))  # 在所有 Worker 上初始化 KV 缓存
        compilation_times: list[float] = self.collective_rpc("compile_or_warm_up_model")  # 在所有 Worker 上编译或预热模型
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        if compilation_times:  # 如果有编译时间数据
            self.vllm_config.compilation_config.compilation_time = max(  # 取所有 Worker 编译时间的最大值
                compilation_times
            )

    # 注册失败回调：当执行器进入永久性失败状态时，通过此回调通知引擎层
    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027  # 注册失败回调函数
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass  # 默认空实现，子类可覆盖

    # 查询所有 Worker 的可用显存（字节），用于 KV 缓存容量规划
    def determine_available_memory(self) -> list[int]:  # in bytes  # 返回各 Worker 可用内存列表
        return self.collective_rpc("determine_available_memory")  # 调用所有 Worker 的内存查询方法

    # 获取所有 Worker 的 KV 缓存规格信息
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:  # 返回各 Worker 的 KV 缓存规格
        return self.collective_rpc("get_kv_cache_spec")  # 调用所有 Worker 的缓存规格查询方法

    # collective_rpc 是执行器的核心通信原语。
    # 它在所有 Worker 上并行执行指定方法，并收集返回值。
    # 支持按方法名字符串调用，也支持传入可序列化的 Callable。
    # non_block=True 时返回 Future，non_block=False 时阻塞等待结果列表。
    @overload  # 同步调用的类型重载声明
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间（秒），None 表示无限等待
        args: tuple = (),  # 位置参数
        kwargs: dict | None = None,  # 关键字参数
        non_block: Literal[False] = False,  # 同步阻塞模式
    ) -> list[_R]:  # 返回各 Worker 的结果列表
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
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        pass  # 类型重载占位，实际逻辑在下面的抽象方法中

    @overload  # 异步非阻塞调用的类型重载声明
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict | None = None,  # 关键字参数
        non_block: Literal[True] = True,  # 非阻塞模式
    ) -> Future[list[_R]]:  # 返回 Future 对象
        pass  # 类型重载占位

    @abstractmethod  # 抽象方法，子类必须实现
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False  # 实际的方法签名
    ):
        raise NotImplementedError  # 基类调用时抛出未实现错误

    # 获取 KV 连接器的握手元数据，用于分布式 KV 缓存传输的初始化协商
    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[int, KVConnectorHandshakeMetadata]]:  # 返回各 Worker 的握手元数据
        return self.collective_rpc("get_kv_connector_handshake_metadata")  # 调用所有 Worker 获取握手元数据

    # execute_model：将调度器输出发送给 Worker 执行模型前向推理。
    # 支持阻塞和非阻塞两种调用模式，非阻塞模式返回 Future 以支持流水线并行。
    @overload  # 同步调用的类型重载声明
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False  # 同步执行模型
    ) -> ModelRunnerOutput | None:  # 返回模型输出或 None
        pass  # 类型重载占位

    @overload  # 异步非阻塞调用的类型重载声明
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True  # 非阻塞执行模型
    ) -> Future[ModelRunnerOutput | None]:  # 返回 Future 对象
        pass  # 类型重载占位

    def execute_model(  # 执行模型的实际实现
        self, scheduler_output: SchedulerOutput, non_block: bool = False  # 接收调度器输出和阻塞模式参数
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:  # 返回模型输出、None 或 Future
        output = self.collective_rpc(  # type: ignore[call-overload]  # 通过 RPC 在所有 Worker 上执行模型
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]  # 返回第一个 Worker（rank 0）的输出

    # sample_tokens：执行 token 采样阶段，可选地应用语法约束（GrammarOutput）。
    # 在支持前向/采样分离的流水线中，此方法在 execute_model 之后调用。
    @overload  # 同步采样的类型重载声明
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False  # 同步采样 token
    ) -> ModelRunnerOutput:  # 返回模型运行输出
        pass  # 类型重载占位

    @overload  # 异步采样的类型重载声明
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True  # 非阻塞采样 token
    ) -> Future[ModelRunnerOutput]:  # 返回 Future 对象
        pass  # 类型重载占位

    def sample_tokens(  # 采样 token 的实际实现
        self, grammar_output: GrammarOutput | None, non_block: bool = False  # 接收语法输出和阻塞模式参数
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:  # 返回模型输出或 Future
        output = self.collective_rpc(  # type: ignore[call-overload]  # 通过 RPC 在所有 Worker 上执行采样
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]  # 返回第一个 Worker（rank 0）的采样结果

    # 执行空批次，用于流水线并行中的预热或填充空闲阶段
    def execute_dummy_batch(self) -> None:  # 执行空批次（无返回值）
        self.collective_rpc("execute_dummy_batch")  # 在所有 Worker 上执行空批次

    # 获取推测解码（speculative decoding）产生的草稿 token ID
    def take_draft_token_ids(self) -> DraftTokenIds | None:  # 获取草稿 token ID
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")  # 从所有 Worker 收集草稿 token
        return output[0]  # 返回第一个 Worker 的草稿 token

    @property  # 属性装饰器
    def max_concurrent_batches(self) -> int:  # 最大并发批次数，默认为 1
        return 1  # 返回默认值 1

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):  # 启动或停止性能分析
        self.collective_rpc("profile", args=(is_start, profile_prefix))  # 在所有 Worker 上执行性能分析

    # 保存模型的分片状态到磁盘，用于模型检查点持久化
    def save_sharded_state(  # 保存分片模型状态
        self,
        path: str,  # 保存路径
        pattern: str | None = None,  # 文件名匹配模式
        max_size: int | None = None,  # 单个分片的最大大小
    ) -> None:
        self.collective_rpc(  # 在所有 Worker 上执行分片保存
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod  # 抽象方法，子类必须实现健康检查
    def check_health(self) -> None:  # 检查执行器是否健康
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError  # 基类调用时抛出未实现错误

    def shutdown(self) -> None:  # 关闭执行器
        """Shutdown the executor."""
        self.collective_rpc("shutdown")  # 在所有 Worker 上执行关闭操作

    # 初始化 KV 输出聚合器，用于在分布式 KV 缓存传输场景下汇总多个 Worker 的输出
    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:  # 初始化 KV 输出聚合器
        """Init KVOutputAggregator"""
        self.kv_output_aggregator = KVOutputAggregator.from_connector(  # 通过连接器工厂方法创建聚合器
            connector, self.parallel_config.world_size
        )

    @cached_property  # Avoid unnecessary RPC calls  # 缓存属性，避免重复 RPC 调用
    def supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型元组
        output: list[tuple[SupportedTask, ...]]  # 声明输出类型
        output = self.collective_rpc("get_supported_tasks")  # 从所有 Worker 获取支持的任务
        return output[0]  # 返回第一个 Worker 的支持任务列表

    # LoRA 适配器管理：添加、移除、固定和列举 LoRA 适配器
    def add_lora(self, lora_request: LoRARequest) -> bool:  # 添加 LoRA 适配器
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."  # 断言 LoRA ID 必须大于 0
        return all(self.collective_rpc("add_lora", args=(lora_request,)))  # 在所有 Worker 上添加 LoRA 并返回是否全部成功

    def remove_lora(self, lora_id: int) -> bool:  # 移除 LoRA 适配器
        assert lora_id > 0, "lora_id must be greater than 0."  # 断言 LoRA ID 必须大于 0
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))  # 在所有 Worker 上移除 LoRA 并返回是否全部成功

    def pin_lora(self, lora_id: int) -> bool:  # 固定 LoRA 适配器（防止被换出）
        assert lora_id > 0, "lora_id must be greater than 0."  # 断言 LoRA ID 必须大于 0
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))  # 在所有 Worker 上固定 LoRA 并返回是否全部成功

    def list_loras(self) -> set[int]:  # 列举当前加载的所有 LoRA 适配器 ID
        sets: list[set[int]] = self.collective_rpc("list_loras")  # 从所有 Worker 获取 LoRA ID 集合
        for s in sets:  # 遍历每个 Worker 返回的集合
            assert s == sets[0], "All workers should have the same LORAs."  # 断言所有 Worker 的 LoRA 集合一致
        return sets[0]  # 返回第一个 Worker 的 LoRA 集合

    def reset_mm_cache(self) -> None:  # 重置多模态缓存
        """Reset the multi-modal cache in each worker."""
        self.collective_rpc("reset_mm_cache")  # 在所有 Worker 上重置多模态缓存

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        self.collective_rpc("reset_encoder_cache")  # 在所有 Worker 上重置编码器缓存

    # sleep/wake_up 机制：用于在空闲时释放 GPU 资源（权重和 KV 缓存），
    # 通过标签（tags）支持选择性唤醒部分资源。
    # 这一机制可用于多实例共享 GPU 等高级部署场景。
    def sleep(self, level: int = 1):  # 使执行器进入睡眠状态以释放 GPU 资源
        if self.is_sleeping:  # 如果已经处于睡眠状态
            logger.warning("Executor is already sleeping.")  # 记录警告日志
            return  # 直接返回
        time_before_sleep = time.perf_counter()  # 记录睡眠前的时间戳
        self.collective_rpc("sleep", kwargs=dict(level=level))  # 在所有 Worker 上执行睡眠操作
        time_after_sleep = time.perf_counter()  # 记录睡眠后的时间戳
        self.sleeping_tags = {"weights", "kv_cache"}  # 设置睡眠标签：权重和 KV 缓存
        self.is_sleeping = True  # 标记为睡眠状态
        logger.info(  # 记录睡眠耗时信息
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):  # 唤醒执行器，可选择性唤醒指定标签的资源
        if not self.is_sleeping:  # 如果未处于睡眠状态
            logger.warning("Executor is not sleeping.")  # 记录警告日志
            return  # 直接返回
        if tags:  # 如果指定了唤醒标签
            for tag in tags:  # 遍历每个标签
                if tag not in self.sleeping_tags:  # 如果标签不在睡眠标签集合中
                    logger.warning(  # 记录警告日志
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return  # 直接返回
        time_before_wakeup = time.perf_counter()  # 记录唤醒前的时间戳
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))  # 在所有 Worker 上执行唤醒操作
        time_after_wakeup = time.perf_counter()  # 记录唤醒后的时间戳
        logger.info(  # 记录唤醒耗时信息
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:  # 如果指定了唤醒标签
            for tag in tags:  # 遍历每个标签
                self.sleeping_tags.remove(tag)  # 从睡眠标签集合中移除已唤醒的标签
        else:  # 如果未指定标签，唤醒所有资源
            self.sleeping_tags.clear()  # 清空睡眠标签集合
        if not self.sleeping_tags:  # 如果所有资源都已唤醒
            self.is_sleeping = False  # 标记为非睡眠状态

    # 重新初始化分布式环境，用于动态扩缩容等运行时重配置场景
    def reinitialize_distributed(  # 重新初始化分布式环境
        self, reconfig_request: ReconfigureDistributedRequest  # 接收重配置请求
    ) -> None:
        raise NotImplementedError  # 默认未实现，子类可覆盖


from vllm.v1.executor.uniproc_executor import (  # noqa: E402  # 延迟导入外部启动器执行器（避免循环导入）
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402  # 延迟导入单进程执行器（避免循环导入）
    UniProcExecutor as _UniProcExecutor,
)

# For backwards compatibility.
UniProcExecutor = _UniProcExecutor  # 为向后兼容保留 UniProcExecutor 别名
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher  # 为向后兼容保留 ExecutorWithExternalLauncher 别名
