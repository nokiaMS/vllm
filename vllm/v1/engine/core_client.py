# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio  # 导入异步IO库
import contextlib  # 导入上下文管理工具库
import multiprocessing  # 导入多进程库
import queue  # 导入线程安全队列库
import sys  # 导入系统模块
import uuid  # 导入UUID生成库
import weakref  # 导入弱引用库
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections import defaultdict, deque  # 导入默认字典和双端队列
from collections.abc import Awaitable, Callable, Sequence  # 导入可等待、可调用、序列类型
from concurrent.futures import Future  # 导入并发Future类
from dataclasses import dataclass  # 导入数据类装饰器
from threading import Thread  # 导入线程类
from typing import Any, TypeAlias, TypeVar  # 导入类型提示工具

import msgspec.msgpack  # 导入msgspec的msgpack序列化库

#ZMQ相关。
import zmq  # 导入ZeroMQ消息队列库
import zmq.asyncio  # 导入ZeroMQ的异步IO支持

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.envs import VLLM_ENGINE_READY_TIMEOUT_S  # 导入引擎就绪超时时间（秒）
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.lora.request import LoRARequest  # 导入LoRA请求类
from vllm.tasks import SupportedTask  # 导入支持的任务类型
from vllm.tracing import instrument  # 导入追踪装饰器
from vllm.utils.async_utils import in_loop  # 导入判断是否在事件循环中的工具函数
from vllm.utils.network_utils import (  # 从网络工具模块导入
    close_sockets,  # 关闭ZMQ socket的工具函数
    get_open_zmq_inproc_path,  # 获取可用的ZMQ进程内通信路径
    make_zmq_socket,  # 创建ZMQ socket的工厂函数
)
from vllm.v1.engine import (  # 从v1引擎模块导入
    EEP_NOTIFICATION_CALL_ID,  # 弹性EP通知的特殊call_id常量
    EEPNotificationType,  # 弹性EP通知类型枚举
    EngineCoreOutputs,  # 引擎核心输出数据结构
    EngineCoreRequest,  # 引擎核心请求数据结构
    EngineCoreRequestType,  # 引擎核心请求类型枚举
    PauseMode,  # 暂停模式类型
    ReconfigureDistributedRequest,  # 分布式重配置请求
    ReconfigureRankType,  # 重配置rank类型枚举
    UtilityOutput,  # 工具方法输出数据结构
)
from vllm.v1.engine.coordinator import DPCoordinator  # 导入数据并行协调器
from vllm.v1.engine.core import EngineCore, EngineCoreProc  # 导入引擎核心类和引擎核心进程类
from vllm.v1.engine.exceptions import EngineDeadError  # 导入引擎死亡异常
from vllm.v1.engine.utils import (  # 从引擎工具模块导入
    CoreEngineActorManager,  # Ray Actor引擎管理器
    CoreEngineProcManager,  # 多进程引擎管理器
    get_engine_zmq_addresses,  # 获取引擎ZMQ地址
    launch_core_engines,  # 启动引擎核心进程
)
from vllm.v1.executor import Executor  # 导入执行器基类
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index  # 导入late interaction引擎索引获取函数
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr  # 导入msgpack编解码器和字节字符串类型

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

# [中文注释] 异步/同步 Future 的联合类型，统一 asyncio 和 threading 的 Future
AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]  # 定义异步/同步Future的联合类型别名

_R = TypeVar("_R")  # 泛型类型变量，用于collective_rpc的返回类型

# [中文注释] Engine 的 ZMQ 身份标识，实际为 2 字节小端整数（如 rank=0 → b'\x00\x00'）
EngineIdentity = bytes  # 引擎身份标识类型别名，实际为2字节小端整数


# [中文注释] EngineCore client的抽象基类。
# 三个子类分别对应不同的使用场景：
#   InprocClient:   同进程直接调用（无 IPC，用于 V0 风格的 LLMEngine）
#   SyncMPClient:   同步多进程客户端（ZMQ 通信，用于 LLM 同步推理）
#   AsyncMPClient:  异步多进程客户端（ZMQ + asyncio，用于 AsyncLLM 在线服务）
class EngineCoreClient(ABC):  # 引擎核心客户端抽象基类
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    # [中文注释] 工厂方法：根据运行模式（多进程/异步）创建对应的 Client 子类实例
    @staticmethod  # 静态方法装饰器
    def make_client(  # 工厂方法：创建对应的客户端子类实例
        multiprocess_mode: bool,  # 是否启用多进程模式
        asyncio_mode: bool,  # 是否启用异步模式
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计信息
    ) -> "EngineCoreClient":  # 返回EngineCoreClient实例
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:  # 异步模式但非多进程，暂不支持
            raise NotImplementedError(  # 抛出未实现异常
                "Running EngineCore in asyncio without multiprocessing "  # 错误消息第一行
                "is not currently supported."  # 错误消息第二行
            )  # 异常构造结束

        # 多进程+异步模式
        if multiprocess_mode and asyncio_mode:  # 多进程+异步模式
            return EngineCoreClient.make_async_mp_client(  # 创建异步多进程客户端
                vllm_config, executor_class, log_stats  # 传入配置参数
            )  # 返回异步多进程客户端

        # 多进程+同步模式
        if multiprocess_mode and not asyncio_mode:  # 多进程+同步模式
            return SyncMPClient(vllm_config, executor_class, log_stats)  # 创建同步多进程客户端

        # 默认创建进程内客户端
        return InprocClient(vllm_config, executor_class, log_stats)  # 默认创建进程内客户端

    # [中文注释] 创建异步多进程客户端。根据数据并行配置选择：
    #   DPAsyncMPClient:   外部负载均衡（每个 DP rank 一个 Client）
    #   DPLBAsyncMPClient: 内部负载均衡（Client 在多个 DP rank 间分发请求）
    #   AsyncMPClient:     单引擎模式
    @staticmethod  # 静态方法装饰器
    @instrument(span_name="Overall Loading")  # 追踪装饰器，记录加载耗时
    def make_async_mp_client(  # 创建异步多进程客户端的工厂方法
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计信息
        client_addresses: dict[str, str] | None = None,  # 可选的客户端地址字典
        client_count: int = 1,  # 客户端总数，默认为1
        client_index: int = 0,  # 当前客户端索引，默认为0
    ) -> "AsyncMPClient":  # 返回AsyncMPClient实例
        parallel_config = vllm_config.parallel_config  # 获取并行配置
        client_args = (  # 构造客户端参数元组
            vllm_config,  # vLLM配置
            executor_class,  # 执行器类
            log_stats,  # 统计日志开关
            client_addresses,  # 客户端地址
            client_count,  # 客户端数量
            client_index,  # 客户端索引
        )
        if parallel_config.data_parallel_size > 1:  # 数据并行大小大于1
            if parallel_config.data_parallel_external_lb:  # 使用外部负载均衡
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)  # 返回外部LB的DP异步客户端
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)  # 返回内部LB的DP异步客户端
        return AsyncMPClient(*client_args)  # 单引擎模式，返回普通异步客户端

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def shutdown(self, timeout: float | None = None) -> None: ...  # 关闭客户端，子类必须实现

    def get_output(self) -> EngineCoreOutputs:  # 获取引擎核心输出（同步）
        raise NotImplementedError  # 子类未实现时抛出异常

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型
        raise NotImplementedError  # 子类未实现时抛出异常

    def add_request(self, request: EngineCoreRequest) -> None:  # 添加推理请求（同步）
        raise NotImplementedError  # 子类未实现时抛出异常

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:  # 启动/停止性能分析
        raise NotImplementedError  # 子类未实现时抛出异常

    def reset_mm_cache(self) -> None:  # 重置多模态缓存
        raise NotImplementedError  # 子类未实现时抛出异常

    def reset_prefix_cache(  # 重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 可选是否重置运行中请求和连接器
    ) -> bool:  # 返回是否成功
        raise NotImplementedError  # 子类未实现时抛出异常

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        raise NotImplementedError  # 子类未实现时抛出异常

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:  # 使引擎进入休眠状态
        raise NotImplementedError  # 子类未实现时抛出异常

    def wake_up(self, tags: list[str] | None = None) -> None:  # 唤醒休眠的引擎
        raise NotImplementedError  # 子类未实现时抛出异常

    def is_sleeping(self) -> bool:  # 检查引擎是否处于休眠状态
        raise NotImplementedError  # 子类未实现时抛出异常

    def execute_dummy_batch(self) -> None:  # 执行虚拟批次（用于预热等场景）
        raise NotImplementedError  # 子类未实现时抛出异常

    async def execute_dummy_batch_async(self) -> None:  # 异步执行虚拟批次
        raise NotImplementedError  # 子类未实现时抛出异常

    def abort_requests(self, request_ids: list[str]) -> None:  # 中止指定请求（同步）
        raise NotImplementedError  # 子类未实现时抛出异常

    def add_lora(self, lora_request: LoRARequest) -> bool:  # 添加LoRA适配器
        raise NotImplementedError  # 子类未实现时抛出异常

    def remove_lora(self, lora_id: int) -> bool:  # 移除LoRA适配器
        raise NotImplementedError  # 子类未实现时抛出异常

    def list_loras(self) -> set[int]:  # 列出已加载的LoRA ID集合
        raise NotImplementedError  # 子类未实现时抛出异常

    def pin_lora(self, lora_id: int) -> bool:  # 固定LoRA适配器到GPU
        raise NotImplementedError  # 子类未实现时抛出异常

    def save_sharded_state(  # 保存分片状态
        self, path: str, pattern: str | None = None, max_size: int | None = None  # 保存路径、模式和最大大小
    ) -> None:  # 无返回值
        raise NotImplementedError  # 子类未实现时抛出异常

    def collective_rpc(  # 集合RPC调用
        self,  # 实例自身
        method: str | Callable[..., _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回所有worker的结果列表
        raise NotImplementedError  # 子类未实现时抛出异常

    def dp_engines_running(self) -> bool:  # 检查数据并行引擎是否正在运行
        """Returns True if data parallel engines are collectively in a
        running state."""
        raise NotImplementedError  # 子类未实现时抛出异常

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:  # 弹性EP伸缩（异步）
        raise NotImplementedError  # 子类未实现时抛出异常

    async def get_output_async(self) -> EngineCoreOutputs:  # 异步获取引擎核心输出
        raise NotImplementedError  # 子类未实现时抛出异常

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:  # 异步获取支持的任务类型
        raise NotImplementedError  # 子类未实现时抛出异常

    async def add_request_async(self, request: EngineCoreRequest) -> None:  # 异步添加推理请求
        raise NotImplementedError  # 子类未实现时抛出异常

    async def profile_async(  # 异步启动/停止性能分析
        self, is_start: bool = True, profile_prefix: str | None = None  # 是否开始、分析前缀
    ) -> None:  # 无返回值
        raise NotImplementedError  # 子类未实现时抛出异常

    async def reset_mm_cache_async(self) -> None:  # 异步重置多模态缓存
        raise NotImplementedError  # 子类未实现时抛出异常

    async def reset_prefix_cache_async(  # 异步重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 可选是否重置运行中请求和连接器
    ) -> bool:  # 返回是否成功
        raise NotImplementedError  # 子类未实现时抛出异常

    async def reset_encoder_cache_async(self) -> None:  # 异步重置编码器缓存
        raise NotImplementedError  # 子类未实现时抛出异常

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:  # 异步使引擎休眠
        raise NotImplementedError  # 子类未实现时抛出异常

    async def wake_up_async(self, tags: list[str] | None = None) -> None:  # 异步唤醒引擎
        raise NotImplementedError  # 子类未实现时抛出异常

    async def is_sleeping_async(self) -> bool:  # 异步检查引擎是否休眠
        raise NotImplementedError  # 子类未实现时抛出异常

    async def abort_requests_async(self, request_ids: list[str]) -> None:  # 异步中止请求
        raise NotImplementedError  # 子类未实现时抛出异常

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:  # 异步添加LoRA适配器
        raise NotImplementedError  # 子类未实现时抛出异常

    async def remove_lora_async(self, lora_id: int) -> bool:  # 异步移除LoRA适配器
        raise NotImplementedError  # 子类未实现时抛出异常

    async def list_loras_async(self) -> set[int]:  # 异步列出LoRA ID
        raise NotImplementedError  # 子类未实现时抛出异常

    async def pin_lora_async(self, lora_id: int) -> bool:  # 异步固定LoRA
        raise NotImplementedError  # 子类未实现时抛出异常

    async def save_sharded_state_async(  # 异步保存分片状态
        self, path: str, pattern: str | None = None, max_size: int | None = None  # 保存路径、模式和最大大小
    ) -> None:  # 无返回值
        raise NotImplementedError  # 子类未实现时抛出异常

    async def collective_rpc_async(  # 异步集合RPC调用
        self,  # 实例自身
        method: str | Callable[..., _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回所有worker的结果列表
        raise NotImplementedError  # 子类未实现时抛出异常


# [中文注释] 同进程客户端：EngineCore 在当前进程内运行，无 IPC，无 busy loop。
# 直接调用 EngineCore 方法推送请求和获取输出，用于 V0 风格的同步推理。
class InprocClient(EngineCoreClient):  # 同进程客户端，直接调用EngineCore
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        self.engine_core = EngineCore(*args, **kwargs)  # 在当前进程内创建EngineCore实例

    def get_output(self) -> EngineCoreOutputs:  # 获取引擎输出（同步步进）
        outputs, model_executed = self.engine_core.step_fn()  # 执行一步推理，返回输出和是否执行模型
        self.engine_core.post_step(model_executed=model_executed)  # 执行步后处理
        return outputs and outputs.get(0) or EngineCoreOutputs()  # 返回第一个输出或空输出

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型
        return self.engine_core.get_supported_tasks()  # 直接委托给engine_core

    def add_request(self, request: EngineCoreRequest) -> None:  # 添加推理请求
        req, request_wave = self.engine_core.preprocess_add_request(request)  # 预处理请求
        self.engine_core.add_request(req, request_wave)  # 将预处理后的请求添加到引擎

    def abort_requests(self, request_ids: list[str]) -> None:  # 中止指定请求
        if len(request_ids) > 0:  # 仅在有请求需要中止时执行
            self.engine_core.abort_requests(request_ids)  # 委托给engine_core中止请求

    def shutdown(self, timeout: float | None = None) -> None:  # 关闭引擎
        self.engine_core.shutdown()  # 委托给engine_core执行关闭

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:  # 性能分析
        self.engine_core.profile(is_start, profile_prefix)  # 委托给engine_core执行

    def reset_mm_cache(self) -> None:  # 重置多模态缓存
        self.engine_core.reset_mm_cache()  # 委托给engine_core执行

    def reset_prefix_cache(  # 重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 可选参数
    ) -> bool:  # 返回是否成功
        return self.engine_core.reset_prefix_cache(  # 委托给engine_core执行
            reset_running_requests, reset_connector  # 传递参数
        )

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        self.engine_core.reset_encoder_cache()  # 委托给engine_core执行

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:  # 使引擎休眠
        if mode == "wait":  # 进程内模式不支持wait暂停模式
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")  # 抛出值错误
        result = self.engine_core.sleep(level, mode)  # 执行休眠操作
        assert result is None  # 断言进程内模式下结果为None

    def wake_up(self, tags: list[str] | None = None) -> None:  # 唤醒引擎
        self.engine_core.wake_up(tags)  # 委托给engine_core执行

    def is_sleeping(self) -> bool:  # 检查是否休眠
        return self.engine_core.is_sleeping()  # 委托给engine_core检查

    def execute_dummy_batch(self) -> None:  # 执行虚拟批次
        self.engine_core.execute_dummy_batch()  # 委托给engine_core执行

    def add_lora(self, lora_request: LoRARequest) -> bool:  # 添加LoRA适配器
        return self.engine_core.add_lora(lora_request)  # 委托给engine_core执行

    def remove_lora(self, lora_id: int) -> bool:  # 移除LoRA适配器
        return self.engine_core.remove_lora(lora_id)  # 委托给engine_core执行

    def list_loras(self) -> set[int]:  # 列出LoRA ID
        return self.engine_core.list_loras()  # 委托给engine_core执行

    def pin_lora(self, lora_id: int) -> bool:  # 固定LoRA
        return self.engine_core.pin_lora(lora_id)  # 委托给engine_core执行

    def save_sharded_state(  # 保存分片状态
        self, path: str, pattern: str | None = None, max_size: int | None = None  # 路径、模式和大小参数
    ) -> None:  # 无返回值
        self.engine_core.save_sharded_state(path, pattern, max_size)  # 委托给engine_core执行

    def collective_rpc(  # 集合RPC调用
        self,  # 实例自身
        method: str | Callable[..., _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回结果列表
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)  # 委托给engine_core执行

    def dp_engines_running(self) -> bool:  # 检查DP引擎是否运行（进程内模式无DP）
        return False  # 进程内模式不支持数据并行，始终返回False


# [中文注释] 后台资源管理器，作为 weakref.finalize 的回调对象使用。
# 持有 ZMQ socket、引擎管理器、协调器等资源，避免与 Client 对象的循环引用。
# 当 Client 被垃圾回收时自动调用 __call__ 释放所有资源。
@dataclass  # 数据类装饰器，自动生成__init__等方法
class BackgroundResources:  # 后台资源管理器，GC时自动清理
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context  # ZMQ上下文对象
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None  # 引擎管理器（进程或Actor）
    coordinator: DPCoordinator | None = None  # 数据并行协调器
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None  # 输出socket（PULL，接收Core响应）
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None  # 输入socket（ROUTER，发送请求到Core）
    first_req_send_socket: zmq.asyncio.Socket | None = None  # 首请求发送socket（PAIR，通知Coordinator）
    first_req_rcv_socket: zmq.asyncio.Socket | None = None  # 首请求接收socket（PAIR，stats任务接收端）
    stats_update_socket: zmq.asyncio.Socket | None = None  # 统计更新socket（XSUB，订阅Coordinator广播）
    output_queue_task: asyncio.Task | None = None  # 异步输出队列任务
    stats_update_task: asyncio.Task | None = None  # 异步统计更新任务
    shutdown_path: str | None = None  # 同步关闭信号的ZMQ路径

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False  # 引擎是否已死亡的标志

    def __call__(self):  # 作为终结器被调用，清理所有后台资源
        """Clean up background resources."""

        self.engine_dead = True  # 标记引擎为已死亡
        if self.engine_manager is not None:  # 如果引擎管理器存在
            self.engine_manager.shutdown()  # 关闭引擎管理器
        if self.coordinator is not None:  # 如果协调器存在
            self.coordinator.shutdown()  # 关闭协调器

        if isinstance(self.output_socket, zmq.asyncio.Socket):  # 异步socket的清理路径
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None  # 获取事件循环

            sockets = (  # 收集所有需要关闭的socket
                self.output_socket,  # 输出socket
                self.input_socket,  # 输入socket
                self.first_req_send_socket,  # 首请求发送socket
                self.first_req_rcv_socket,  # 首请求接收socket
                self.stats_update_socket,  # 统计更新socket
            )

            tasks = (self.output_queue_task, self.stats_update_task)  # 收集所有异步任务

            def close_sockets_and_tasks():  # 关闭socket和取消任务的内部函数
                close_sockets(sockets)  # 关闭所有socket
                for task in tasks:  # 遍历所有任务
                    if task is not None and not task.done():  # 如果任务存在且未完成
                        with contextlib.suppress(Exception):  # 忽略取消时的异常
                            task.cancel()  # 取消任务

            if loop is not None:  # 如果事件循环存在
                if in_loop(loop):  # 如果当前在事件循环中
                    close_sockets_and_tasks()  # 直接关闭
                elif not loop.is_closed():  # 如果循环未关闭
                    loop.call_soon_threadsafe(close_sockets_and_tasks)  # 线程安全地调度关闭
            else:  # 事件循环已关闭
                # Loop has been closed, try to clean up directly.
                del tasks  # 删除任务引用
                del close_sockets_and_tasks  # 删除函数引用
                close_sockets(sockets)  # 直接关闭socket
                del self.output_queue_task  # 删除输出队列任务引用
                del self.stats_update_task  # 删除统计更新任务引用
        else:  # 同步socket的清理路径
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))  # 显式关闭同步socket

            if self.shutdown_path is not None:  # 如果存在关闭信号路径
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:  # 创建关闭信号发送socket
                    shutdown_sender.connect(self.shutdown_path)  # 连接到关闭信号路径
                    # Send shutdown signal.
                    shutdown_sender.send(b"")  # 发送空消息作为关闭信号

    # [中文注释] 检查收到的 ZMQ 帧是否为 ENGINE_CORE_DEAD 死亡信号
    def validate_alive(self, frames: Sequence[zmq.Frame]):  # 验证接收到的帧是否为死亡信号
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):  # 检查是否是引擎死亡信号
            self.engine_dead = True  # 标记引擎为已死亡
            raise EngineDeadError()  # 抛出引擎死亡异常


@dataclass  # 数据类装饰器
class ElasticScalingCache:  # 弹性伸缩缓存，跟踪伸缩过程中的状态
    """弹性伸缩缓存，用于跟踪弹性EP伸缩过程中的状态。"""
    existing_core_engines: list[EngineIdentity]  # 伸缩前已存在的引擎标识列表
    num_new_core_engines: int  # 新增引擎数量（正数为扩容，负数为缩容）
    pending_notifications: dict[EEPNotificationType, set[int]]  # 待处理的通知：{通知类型: {已通知的dp_rank集合}}


def allocate_stateless_group_ports(parallel_config, new_data_parallel_size: int):  # 为弹性EP分配无状态组端口
    """
    Allocate stateless group ports for elastic EP.
    """
    from vllm.utils.network_utils import get_open_ports_list  # 延迟导入获取可用端口列表的函数

    assert parallel_config.enable_elastic_ep, "Elastic EP must be enabled"  # 断言弹性EP已启用
    world_size = parallel_config.world_size  # 获取当前世界大小（单DP rank内的进程数）
    new_world_size_across_dp = world_size * new_data_parallel_size  # 计算跨DP的总世界大小
    num_world_groups = 1  # 世界组数量固定为1
    num_dp_groups = max(1, new_world_size_across_dp // new_data_parallel_size)  # 计算DP组数量
    num_ep_groups = max(  # 计算EP组数量
        1,  # 最少1个
        new_world_size_across_dp  # 总世界大小
        // (new_data_parallel_size * parallel_config.tensor_parallel_size),  # 除以DP*TP大小
    )
    num_eplb_groups = num_ep_groups  # EPLB组数量与EP组相同
    total_ports_needed = (  # 计算总共需要的端口数
        num_world_groups + num_dp_groups + num_ep_groups + num_eplb_groups  # 所有组数量之和
    ) * 3 + 5  # 每组3个端口，额外5个用于master端口
    all_ports = get_open_ports_list(total_ports_needed)  # 获取所有可用端口
    new_data_parallel_master_port_list = all_ports[-5:]  # 最后5个端口作为DP master端口
    all_ports = all_ports[:-5]  # 剩余端口用于组端口分配
    new_stateless_world_group_port_list = [  # 分配世界组端口列表
        all_ports[i : i + 3] for i in range(0, num_world_groups * 3, 3)  # 每组3个端口
    ]
    start_idx = num_world_groups * 3  # 计算DP组端口的起始索引
    new_stateless_dp_group_port_list = [  # 分配DP组端口列表
        all_ports[i : i + 3] for i in range(start_idx, start_idx + num_dp_groups * 3, 3)  # 每组3个端口
    ]
    start_idx += num_dp_groups * 3  # 更新起始索引到EP组
    new_stateless_ep_group_port_list = [  # 分配EP组端口列表
        all_ports[i : i + 3] for i in range(start_idx, start_idx + num_ep_groups * 3, 3)  # 每组3个端口
    ]
    start_idx += num_ep_groups * 3  # 更新起始索引到EPLB组
    new_stateless_eplb_group_port_list = [  # 分配EPLB组端口列表
        all_ports[i : i + 3]  # 每组3个端口
        for i in range(start_idx, start_idx + num_eplb_groups * 3, 3)  # 遍历EPLB组
    ]

    parallel_config._stateless_world_group_port_list = (  # 设置世界组端口列表到配置
        new_stateless_world_group_port_list  # 新分配的世界组端口
    )
    parallel_config._stateless_dp_group_port_list = new_stateless_dp_group_port_list  # 设置DP组端口列表
    parallel_config._stateless_ep_group_port_list = new_stateless_ep_group_port_list  # 设置EP组端口列表
    parallel_config._stateless_eplb_group_port_list = new_stateless_eplb_group_port_list  # 设置EPLB组端口列表
    parallel_config.data_parallel_master_port = new_data_parallel_master_port_list.pop()  # 弹出一个作为主master端口
    parallel_config._data_parallel_master_port_list = new_data_parallel_master_port_list  # 设置剩余master端口列表


# [中文注释] 多进程客户端的基类。EngineCore 运行在后台进程的 busy loop 中。
# 通信模式：
#   请求：Client → input_socket (ROUTER) → Core DEALER socket
#   响应：Core PUSH socket → output_socket (PULL) → Client
# 子类 SyncMPClient 用于同步场景，AsyncMPClient 用于异步场景。
class MPClient(EngineCoreClient):  # 多进程客户端基类，通过ZMQ与EngineCore通信
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(  # MPClient构造函数
        self,  # 实例自身
        asyncio_mode: bool,  # 是否使用异步模式
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, str] | None = None,  # 可选的客户端地址
    ):
        self.vllm_config = vllm_config  # 保存vLLM配置

        # Serialization setup.
        # [中文注释] 序列化：编码器用于将请求序列化为 msgpack，解码器用于反序列化响应
        self.encoder = MsgpackEncoder()  # 创建msgpack编码器
        self.decoder = MsgpackDecoder(EngineCoreOutputs)  # 创建msgpack解码器，目标类型为EngineCoreOutputs

        # ZMQ setup.
        # [中文注释] 创建 ZMQ 上下文，io_threads=2 用于并行处理 socket I/O
        sync_ctx = zmq.Context(io_threads=2)  # 创建同步ZMQ上下文，2个IO线程
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx  # 异步模式则包装为异步上下文

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)  # 创建后台资源管理器
        self._finalizer = weakref.finalize(self, self.resources)  # 注册GC终结器，释放时自动清理资源
        success = False  # 构造成功标志，用于异常时清理
        try:  # 尝试初始化，失败时通过finally清理
            # State used for data parallel.
            self.engines_running = False  # 引擎运行状态标志
            parallel_config = vllm_config.parallel_config  # 获取并行配置
            # Elastic EP can remove a rank and later add it back with the same
            # identity. The client input ROUTER needs handover to allow the new
            # engine to replace the dead connection.
            enable_input_socket_handover = parallel_config.enable_elastic_ep  # 是否启用socket连接移交

            self.stats_update_address: str | None = None  # 统计更新地址初始化为None
            if client_addresses:  # 如果提供了外部管理的地址
                # Engines are managed externally to this client.
                input_address = client_addresses["input_address"]  # 获取输入地址
                output_address = client_addresses["output_address"]  # 获取输出地址
                self.stats_update_address = client_addresses.get("stats_update_address")  # 获取统计更新地址
                self.input_socket = self.resources.input_socket = make_zmq_socket(  # 创建输入ROUTER socket
                    self.ctx,  # ZMQ上下文
                    input_address,  # 绑定地址
                    zmq.ROUTER,  # ROUTER类型socket
                    bind=True,  # 绑定模式
                    router_handover=enable_input_socket_handover,  # 连接移交设置
                )
                self.resources.output_socket = make_zmq_socket(  # 创建输出PULL socket
                    self.ctx, output_address, zmq.PULL  # 上下文、地址、PULL类型
                )
            else:  # 引擎由本客户端管理
                # Engines are managed by this client.
                addresses = get_engine_zmq_addresses(vllm_config)  # 获取引擎ZMQ地址
                self.input_socket = self.resources.input_socket = make_zmq_socket(  # 创建输入ROUTER socket
                    self.ctx,  # ZMQ上下文
                    addresses.inputs[0],  # 第一个输入地址
                    zmq.ROUTER,  # ROUTER类型
                    bind=True,  # 绑定模式
                    router_handover=enable_input_socket_handover,  # 连接移交设置
                )
                self.resources.output_socket = make_zmq_socket(  # 创建输出PULL socket
                    self.ctx, addresses.outputs[0], zmq.PULL  # 上下文、第一个输出地址、PULL类型
                )

                # 启动引擎核心进程（上下文管理器）
                with launch_core_engines(  # 启动引擎核心进程（上下文管理器）
                    vllm_config, executor_class, log_stats, addresses  # 传入配置和地址
                ) as (engine_manager, coordinator, addresses):  # 解构返回值
                    self.resources.coordinator = coordinator  # 保存协调器引用
                    self.resources.engine_manager = engine_manager  # 保存引擎管理器引用

                self.stats_update_address = addresses.frontend_stats_publish_address  # 保存统计更新地址
                if coordinator is not None:  # 如果协调器存在
                    assert self.stats_update_address == (  # 断言地址一致
                        coordinator.get_stats_publish_address()  # 从协调器获取统计发布地址
                    )

            dp_size = parallel_config.data_parallel_size  # 数据并行大小
            dp_rank = parallel_config.data_parallel_index  # 当前数据并行rank索引
            dp_local_size = parallel_config.data_parallel_size_local  # 本地数据并行大小
            offline_mode = parallel_config.data_parallel_rank_local is not None  # 是否为离线模式
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size  # 管理的rank数量
            self.engine_ranks_managed = (  # 本客户端管理的引擎rank列表
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))  # 离线模式只管一个rank
            )
            assert parallel_config.data_parallel_size_local <= len(  # 断言本地DP大小不超过管理的rank数
                self.engine_ranks_managed  # 管理的rank列表长度
            )

            # ZMQ identity of each engine that this client will talk to.
            # [中文注释] 将 rank 编号转为 2 字节小端整数，作为 ZMQ ROUTER socket 路由消息的身份标识
            self.core_engines: list[EngineIdentity] = [  # 构建引擎身份标识列表
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed  # 将rank转为2字节小端
            ]

            # Wait for ready messages from each engine on the input socket.
            # [中文注释] 握手阶段：等待所有 Engine Core 进程通过 DEALER socket 发送就绪消息
            identities = set(self.core_engines)  # 创建待确认的引擎身份集合
            sync_input_socket = zmq.Socket.shadow(self.input_socket)  # 创建同步影子socket用于阻塞轮询
            while identities:  # 循环直到所有引擎都发送就绪消息
                if not sync_input_socket.poll(  # 轮询等待消息
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # 超时时间转换为毫秒
                ):
                    raise TimeoutError(  # 超时则抛出异常
                        f"Timed out waiting for engine core processes to "  # 超时错误消息
                        f"start. This is often caused by slow weight loading "  # 常见原因说明
                        f"for large models. Waited "  # 等待时间说明
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "  # 配置来源
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "  # 如何增加超时
                        f"timeout, set the environment variable: "  # 环境变量说明
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"  # 环境变量名称
                    )
                identity, _ = sync_input_socket.recv_multipart()  # 接收就绪消息，获取引擎身份
                identities.remove(identity)  # 从待确认集合中移除已就绪的引擎

            self.core_engine: EngineIdentity = self.core_engines[0]  # 默认引擎设为第一个（用于单引擎场景）
            # [中文注释] utility_results: 存储 RPC 调用的 {call_id: Future} 映射，
            # 等待 Core 返回对应 call_id 的 UtilityOutput 时设置 Future 结果
            self.utility_results: dict[int, AnyFuture] = {}  # 初始化RPC调用结果映射

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            # [中文注释] pending_messages: 追踪已发送但 ZMQ 尚未完成传输的消息。
            # 必须持有 tensor 引用直到 ZMQ 传输完成，否则 tensor 内存可能被释放。
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()  # 初始化待传输消息队列

            # Start monitoring engine core processes for unexpected failures
            self.start_engine_core_monitor()  # 启动引擎核心进程监控线程

            success = True  # 标记构造成功
        finally:  # 无论是否异常都执行
            if not success:  # 如果构造失败
                self._finalizer()  # 调用终结器清理已分配的资源

    def shutdown(self, timeout: float | None = None) -> None:  # 关闭客户端并清理资源
        """Shutdown engine manager under timeout and clean up resources."""
        if self._finalizer.detach() is not None:  # 解除终结器并检查是否有效
            if self.resources.engine_manager is not None:  # 如果引擎管理器存在
                self.resources.engine_manager.shutdown(timeout=timeout)  # 带超时关闭引擎管理器
            self.resources()  # 调用资源清理函数

    def _format_exception(self, e: Exception) -> Exception:  # 格式化异常，引擎死亡时替换为EngineDeadError
        """If errored, use EngineDeadError so root cause is clear."""
        return (  # 返回格式化后的异常
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e  # 引擎死亡则返回EngineDeadError
        )

    def ensure_alive(self):  # 检查引擎是否存活
        if self.resources.engine_dead:  # 如果引擎已死亡
            raise EngineDeadError()  # 抛出引擎死亡异常

    # [中文注释] 记录待传输消息：tracker 追踪 ZMQ 发送状态，msg 持有 tensor 引用防止被 GC
    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):  # 添加待传输消息到队列
        if not tracker.done:  # 如果消息尚未传输完成
            self.pending_messages.appendleft((tracker, msg))  # 添加到队列头部（最新的在前）

    # [中文注释] 释放已完成传输的消息引用（从队尾弹出，FIFO 顺序）
    def free_pending_messages(self):  # 释放已完成传输的消息引用
        while self.pending_messages and self.pending_messages[-1][0].done:  # 从队尾检查已完成的消息
            self.pending_messages.pop()  # 弹出已完成的消息，释放tensor引用

    def dp_engines_running(self) -> bool:  # 检查DP引擎是否正在运行
        return self.engines_running  # 返回引擎运行状态

    def start_engine_core_monitor(self):  # 启动引擎核心进程监控线程
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager  # 获取引擎管理器
        if (  # 检查是否有进程需要监控
            engine_manager is None  # 管理器不存在
            or not hasattr(engine_manager, "processes")  # 没有processes属性
            or not engine_manager.processes  # 进程列表为空
        ):
            # No engine processes to monitor
            return  # 无进程可监控，直接返回

        engine_processes = engine_manager.processes  # 获取引擎进程列表
        self_ref = weakref.ref(self)  # 创建弱引用避免循环引用

        # Monitor engine core process liveness. If any die unexpectedly,
        # logs an error, shuts down the client and invokes the failure
        # callback to inform the engine.
        def monitor_engine_cores():  # 监控引擎核心进程的内部函数
            sentinels = [proc.sentinel for proc in engine_processes]  # 收集所有进程的哨兵描述符
            died = multiprocessing.connection.wait(sentinels)  # 等待任一进程退出
            _self = self_ref()  # 尝试获取客户端强引用
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:  # 如果客户端已销毁或引擎已死
                return  # 直接返回
            _self.resources.engine_dead = True  # 标记引擎为已死亡
            proc_name = next(  # 找到死亡进程的名称
                proc.name for proc in engine_processes if proc.sentinel == died[0]  # 匹配哨兵
            )
            logger.error(  # 记录错误日志
                "Engine core proc %s died unexpectedly, shutting down client.",  # 日志消息
                proc_name,  # 进程名称
            )
            _self.shutdown()  # 关闭客户端
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(  # 创建守护线程
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"  # 线程目标、守护模式、线程名
        ).start()  # 启动监控线程


# [中文注释] 处理 RPC 工具方法的返回值：根据 call_id 找到对应的 Future 并设置结果。
# 如果调用失败（failure_message 不为空），设置异常；否则设置返回值。
def _process_utility_output(  # 处理工具方法的返回结果
    output: UtilityOutput, utility_results: dict[int, AnyFuture]  # 输出对象和等待中的Future映射
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)  # 根据call_id取出对应的Future
    failure_message = output.failure_message  # 获取失败消息（如果有）
    try:  # 尝试设置Future结果
        if failure_message is not None:  # 如果调用失败
            future.set_exception(Exception(failure_message))  # 设置异常到Future
        else:  # 调用成功
            assert output.result is not None  # 断言结果不为空
            future.set_result(output.result.result)  # 设置成功结果到Future
    except asyncio.InvalidStateError:  # Future已被取消或已完成
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:  # 如果有失败消息
            logger.error(  # 记录错误日志
                "Cancelled call to utility method failed with error: %s",  # 日志消息
                failure_message,  # 失败消息内容
            )


# [中文注释] 同步多进程客户端，用于 LLMEngine 的同步推理场景。
# 启动后台线程接收 EngineCore 的输出，通过 queue.Queue 传递给主线程。
class SyncMPClient(MPClient):  # 同步多进程客户端
    """Synchronous client for multi-proc EngineCore."""

    @instrument(span_name="SyncMPClient init")  # 追踪装饰器，记录初始化耗时
    def __init__(  # SyncMPClient构造函数
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool  # 配置、执行器类、统计开关
    ):
        super().__init__(  # 调用父类MPClient构造函数
            asyncio_mode=False,  # 同步模式
            vllm_config=vllm_config,  # vLLM配置
            executor_class=executor_class,  # 执行器类
            log_stats=log_stats,  # 统计开关
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1  # 是否为数据并行模式
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()  # 创建线程安全的输出队列

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx  # 保存ZMQ上下文的局部引用（避免线程持有客户端引用）
        out_socket = self.resources.output_socket  # 保存输出socket的局部引用
        decoder = self.decoder  # 保存解码器的局部引用
        utility_results = self.utility_results  # 保存RPC结果映射的局部引用
        outputs_queue = self.outputs_queue  # 保存输出队列的局部引用

        shutdown_path = get_open_zmq_inproc_path()  # 获取进程内关闭信号路径
        resources = self.resources  # 保存资源管理器的局部引用
        resources.shutdown_path = shutdown_path  # 设置关闭信号路径

        # [中文注释] 后台线程函数：持续从 output_socket (PULL) 接收 Core 推送的响应，
        # 反序列化后区分 utility 响应和推理输出，分别路由到 Future 或 outputs_queue。
        def process_outputs_socket():  # 后台线程函数：处理输出socket消息
            assert isinstance(out_socket, zmq.Socket)  # 断言输出socket为同步类型
            shutdown_socket = ctx.socket(zmq.PAIR)  # 创建PAIR类型的关闭信号socket
            try:  # 尝试运行主循环
                shutdown_socket.bind(shutdown_path)  # 绑定关闭信号路径
                poller = zmq.Poller()  # 创建ZMQ轮询器
                poller.register(shutdown_socket, zmq.POLLIN)  # 注册关闭信号socket为可读
                poller.register(out_socket, zmq.POLLIN)  # 注册输出socket为可读
                while True:  # 主循环
                    socks = poller.poll()  # 轮询等待事件
                    if not socks:  # 无事件则继续
                        continue  # 跳过本轮
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:  # 收到关闭信号
                        # shutdown signal, exit thread.
                        break  # 退出循环

                    frames = out_socket.recv_multipart(copy=False)  # 零拷贝接收多帧消息
                    resources.validate_alive(frames)  # 验证是否为引擎死亡信号
                    outputs: EngineCoreOutputs = decoder.decode(frames)  # 反序列化为EngineCoreOutputs
                    if outputs.utility_output:  # 如果是utility RPC响应
                        _process_utility_output(outputs.utility_output, utility_results)  # 分发到对应的Future
                    else:  # 普通推理输出
                        outputs_queue.put_nowait(outputs)  # 放入输出队列供主线程消费
            except Exception as e:  # 捕获异常
                outputs_queue.put_nowait(e)  # 将异常转发到输出队列
            finally:  # 最终清理
                # Close sockets.
                shutdown_socket.close(linger=0)  # 关闭关闭信号socket
                out_socket.close(linger=0)  # 关闭输出socket

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(  # 创建输出处理线程
            target=process_outputs_socket,  # 目标函数
            name="EngineCoreOutputQueueThread",  # 线程名称
            daemon=True,  # 守护线程
        )
        self.output_queue_thread.start()  # 启动线程

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None  # 线程接管socket关闭责任，清除资源管理器引用

    def get_output(self) -> EngineCoreOutputs:  # 获取引擎输出（阻塞等待）
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()  # 从输出队列阻塞获取结果

        if isinstance(outputs, Exception):  # 如果是异常
            raise self._format_exception(outputs) from None  # 格式化并抛出异常
        if outputs.wave_complete is not None:  # 如果wave已完成
            self.engines_running = False  # 标记引擎为非运行状态
        return outputs  # 返回输出结果

    # [中文注释] 向 Engine Core 发送请求。构建 ZMQ 多帧消息：
    #   帧0: Engine Identity（ROUTER 路由标识）
    #   帧1: 请求类型（1字节，如 b"\x00" 表示 ADD）
    #   帧2: msgpack 序列化的请求主数据
    #   帧3+: tensor 辅助缓冲区（大 tensor 的零拷贝数据，可选）
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):  # 向EngineCore发送请求
        self.ensure_alive()  # 确保引擎存活
        self.free_pending_messages()  # 释放已完成传输的消息引用
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))  # 构建多帧消息

        if len(msg) <= 3:  # 无辅助缓冲区（无tensor数据）
            # No auxiliary buffers => no tensor backing buffers in request.
            self.input_socket.send_multipart(msg, copy=False)  # 零拷贝发送
            return  # 直接返回

        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)  # 带追踪的零拷贝发送
        self.add_pending_message(tracker, request)  # 记录待传输消息，持有tensor引用

    # [中文注释] 同步 RPC 调用：生成唯一 call_id，创建 Future，
    #   发送 UTILITY 类型请求（格式: (client_index=0, call_id, method, args)），
    #   然后阻塞等待后台线程通过 _process_utility_output 设置 Future 结果。
    def call_utility(self, method: str, *args) -> Any:  # 同步RPC调用
        call_id = uuid.uuid1().int >> 64  # 生成唯一的64位call_id
        future: Future[Any] = Future()  # 创建线程Future
        self.utility_results[call_id] = future  # 注册到等待映射
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))  # 发送UTILITY请求

        return future.result()  # 阻塞等待结果返回

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型
        return self.call_utility("get_supported_tasks")  # 委托给call_utility

    # [中文注释] 向 Engine Core 提交推理请求。若为数据并行模式，标记 engines_running=True
    def add_request(self, request: EngineCoreRequest) -> None:  # 添加推理请求
        if self.is_dp:  # 如果是数据并行模式
            self.engines_running = True  # 标记引擎为运行状态
        self._send_input(EngineCoreRequestType.ADD, request)  # 发送ADD类型请求

    def abort_requests(self, request_ids: list[str]) -> None:  # 中止指定请求
        if request_ids and not self.resources.engine_dead:  # 有请求且引擎存活
            self._send_input(EngineCoreRequestType.ABORT, request_ids)  # 发送ABORT类型请求

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:  # 性能分析
        self.call_utility("profile", is_start, profile_prefix)  # 委托给call_utility

    def reset_mm_cache(self) -> None:  # 重置多模态缓存
        self.call_utility("reset_mm_cache")  # 委托给call_utility

    def reset_prefix_cache(  # 重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 可选参数
    ) -> bool:  # 返回是否成功
        return self.call_utility(  # 委托给call_utility
            "reset_prefix_cache", reset_running_requests, reset_connector  # 传递参数
        )

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        self.call_utility("reset_encoder_cache")  # 委托给call_utility

    def add_lora(self, lora_request: LoRARequest) -> bool:  # 添加LoRA适配器
        return self.call_utility("add_lora", lora_request)  # 委托给call_utility

    def remove_lora(self, lora_id: int) -> bool:  # 移除LoRA适配器
        return self.call_utility("remove_lora", lora_id)  # 委托给call_utility

    def list_loras(self) -> set[int]:  # 列出LoRA ID集合
        return self.call_utility("list_loras")  # 委托给call_utility

    def pin_lora(self, lora_id: int) -> bool:  # 固定LoRA到GPU
        return self.call_utility("pin_lora", lora_id)  # 委托给call_utility

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:  # 使引擎休眠
        self.call_utility("sleep", level, mode)  # 委托给call_utility

    def wake_up(self, tags: list[str] | None = None) -> None:  # 唤醒引擎
        self.call_utility("wake_up", tags)  # 委托给call_utility

    def is_sleeping(self) -> bool:  # 检查是否休眠
        return self.call_utility("is_sleeping")  # 委托给call_utility

    def execute_dummy_batch(self) -> None:  # 执行虚拟批次
        self.call_utility("execute_dummy_batch")  # 委托给call_utility

    def collective_rpc(  # 集合RPC调用
        self,  # 实例自身
        method: str | Callable[..., _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回结果列表
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)  # 委托给call_utility

    def save_sharded_state(  # 保存分片状态
        self, path: str, pattern: str | None = None, max_size: int | None = None  # 路径和可选参数
    ) -> None:  # 无返回值
        self.call_utility("save_sharded_state", path, pattern, max_size)  # 委托给call_utility


# [中文注释] 异步多进程客户端。与 SyncMPClient 的区别：
#   1. 输出接收改为 asyncio Task（而非线程），通过 asyncio.Queue 传递结果
#   2. _send_input 返回 Awaitable，支持 async/await
#   3. call_utility_async 使用 asyncio.Future 替代 threading.Future
#   4. 支持 output_handler 回调（用于 DPLBAsyncMPClient 追踪已完成请求）
class AsyncMPClient(MPClient):  # 异步多进程客户端
    """Asyncio-compatible client for multi-proc EngineCore."""

    @instrument(span_name="AsyncMPClient init")  # 追踪装饰器，记录初始化耗时
    def __init__(  # AsyncMPClient构造函数
        self,  # 实例自身
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, str] | None = None,  # 可选的外部管理地址
        client_count: int = 1,  # 客户端总数
        client_index: int = 0,  # 当前客户端索引
    ):
        super().__init__(  # 调用父类MPClient构造函数
            asyncio_mode=True,  # 异步模式
            vllm_config=vllm_config,  # vLLM配置
            executor_class=executor_class,  # 执行器类
            log_stats=log_stats,  # 统计开关
            client_addresses=client_addresses,  # 客户端地址
        )

        self.client_count = client_count  # 保存客户端总数
        self.client_index = client_index  # 保存当前客户端索引
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()  # 创建异步输出队列
        try:  # 尝试在当前事件循环中启动输出队列任务
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()  # 获取当前事件循环
            self._ensure_output_queue_task()  # 启动输出队列任务
        except RuntimeError:  # 不在事件循环中
            pass  # 延迟启动，稍后懒初始化

    # [中文注释] 懒初始化异步输出队列任务。内部创建 process_outputs_socket 协程：
    #   - 循环 recv_multipart 从 output_socket 接收消息
    #   - utility_output → 分发给 _process_utility_output 或 EEP 通知回调
    #   - 普通 output → 可选调用 output_handler（如 DPLBAsyncMPClient.process_engine_outputs），
    #     然后放入 outputs_queue 供 get_output_async 消费
    #   - 使用 weakref 避免循环引用导致客户端无法被 GC
    def _ensure_output_queue_task(self):  # 懒初始化异步输出队列任务
        resources = self.resources  # 获取资源管理器
        if resources.output_queue_task is not None:  # 如果任务已存在
            return  # 直接返回

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder  # 保存解码器局部引用（避免循环引用）
        utility_results = self.utility_results  # 保存RPC结果映射局部引用
        outputs_queue = self.outputs_queue  # 保存输出队列局部引用
        output_handler: (  # 输出回调处理器（子类可定义process_engine_outputs方法）
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None  # 回调函数类型注解
        ) = getattr(self.__class__, "process_engine_outputs", None)  # 从子类获取输出处理方法
        _self_ref = weakref.ref(self) if output_handler else None  # 使用弱引用避免循环引用
        output_socket = resources.output_socket  # 获取输出socket
        assert output_socket is not None  # 断言输出socket已初始化

        notification_callback_handler: (  # EEP通知回调处理器（弹性EP场景）
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None  # 回调函数类型注解
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)  # 从子类获取EEP通知处理方法

        async def process_outputs_socket():  # 异步输出处理协程
            try:  # 尝试运行主循环
                while True:  # 持续接收输出
                    frames = await output_socket.recv_multipart(copy=False)  # 异步零拷贝接收多帧消息
                    resources.validate_alive(frames)  # 验证是否为引擎死亡信号
                    outputs: EngineCoreOutputs = decoder.decode(frames)  # 反序列化为EngineCoreOutputs
                    if outputs.utility_output:  # 如果是utility RPC响应
                        if (  # 检查是否为EEP通知
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID  # 匹配EEP通知call_id
                            and notification_callback_handler is not None  # 且存在通知回调
                        ):
                            assert _self_ref is not None  # 断言弱引用存在
                            _self = _self_ref()  # 获取客户端强引用
                            if not _self:  # 客户端已被GC
                                return  # 退出协程
                            if outputs.utility_output.result is None:  # 空结果则跳过
                                continue  # 继续下一轮
                            notification_data = outputs.utility_output.result.result  # 提取通知数据
                            assert isinstance(notification_data, Sequence)  # 断言为序列类型
                            assert len(notification_data) == 2  # 断言包含2个元素
                            asyncio.create_task(  # 创建异步任务处理EEP通知
                                notification_callback_handler(_self, notification_data)  # 调用通知回调
                            )
                        else:  # 普通utility响应
                            _process_utility_output(  # 分发到对应的Future
                                outputs.utility_output, utility_results  # 传入输出和结果映射
                            )
                        continue  # 跳过后续推理输出处理

                    if output_handler is not None:  # 如果定义了输出回调（如DPLBAsyncMPClient）
                        assert _self_ref is not None  # 断言弱引用存在
                        _self = _self_ref()  # 获取客户端强引用
                        if not _self:  # 客户端已被GC
                            # Client has been garbage collected, abort.
                            return  # 退出协程
                        await output_handler(_self, outputs)  # 调用输出回调

                    if outputs.outputs or outputs.scheduler_stats:  # 如果有推理输出或调度统计
                        outputs_queue.put_nowait(outputs)  # 放入输出队列
            except Exception as e:  # 捕获异常
                outputs_queue.put_nowait(e)  # 将异常转发到输出队列
            except asyncio.CancelledError:  # 捕获取消异常
                outputs_queue.put_nowait(EngineDeadError())  # 放入引擎死亡错误

        resources.output_queue_task = asyncio.create_task(  # 创建并注册异步任务
            process_outputs_socket(), name="EngineCoreOutputQueueTask"  # 任务名称
        )

    async def get_output_async(self) -> EngineCoreOutputs:  # 异步获取引擎输出
        self._ensure_output_queue_task()  # 确保输出队列任务已启动
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None  # 断言输出队列已初始化
        outputs = await self.outputs_queue.get()  # 异步等待输出
        if isinstance(outputs, Exception):  # 如果是异常
            raise self._format_exception(outputs) from None  # 格式化并抛出异常
        return outputs  # 返回输出结果

    # [中文注释] 异步版 _send_input：构建消息元组 (request_type, *encoded_data)，
    #   支持指定目标 engine（用于数据并行场景路由到特定 Core）
    def _send_input(  # 异步版发送请求
        self,  # 实例自身
        request_type: EngineCoreRequestType,  # 请求类型
        request: Any,  # 请求数据
        engine: EngineIdentity | None = None,  # 目标引擎（可选，默认为主引擎）
    ) -> Awaitable[Any]:  # 返回可等待对象
        if engine is None:  # 未指定目标引擎
            engine = self.core_engine  # 使用默认引擎

        message = (request_type.value, *self.encoder.encode(request))  # 构建消息元组
        return self._send_input_message(message, engine, request)  # 委托给底层发送方法

    # [中文注释] 异步版消息发送底层方法：
    #   - 若无辅助缓冲区（len(msg)<=3），直接 send_multipart（零拷贝）
    #   - 否则启用 track=True 获取 MessageTracker Future，完成后注册 add_pending 回调
    #     持有 objects 引用直到 ZMQ 确认发送完成
    def _send_input_message(  # 异步版消息发送底层方法
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any  # 消息元组、目标引擎、原始对象引用
    ) -> Awaitable[Any]:  # 返回可等待对象
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()  # 确保引擎存活
        self.free_pending_messages()  # 释放已完成传输的消息引用

        msg = (engine,) + message  # 拼接引擎身份标识和消息
        if not objects or len(msg) <= 3:  # 无辅助缓冲区
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)  # 零拷贝发送

        future: asyncio.Future[zmq.MessageTracker]  # 类型声明
        future = self.input_socket.send_multipart(msg, copy=False, track=True)  # 带追踪的零拷贝发送

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):  # 发送完成回调
            with contextlib.suppress(BaseException):  # 忽略异常
                self.add_pending_message(f.result(), objects)  # 注册待传输消息

        future.add_done_callback(add_pending)  # 注册回调
        return future  # 返回Future

    # [中文注释] 异步 RPC 调用入口，默认发送到 self.core_engine
    async def call_utility_async(self, method: str, *args) -> Any:  # 异步RPC调用入口
        return await self._call_utility_async(method, *args, engine=self.core_engine)  # 委托给内部实现，发送到默认引擎

    # [中文注释] 异步 RPC 调用实现：
    #   1. 生成 call_id（uuid1 截断为 64 位整数）
    #   2. 创建 asyncio.Future 并注册到 utility_results[call_id]
    #   3. 将 (client_index, call_id, method, args) 序列化后发送
    #   4. await future — 由 process_outputs_socket 中的 _process_utility_output 设置结果
    async def _call_utility_async(  # 异步RPC调用实现
        self, method: str, *args, engine: EngineIdentity  # 方法名、参数、目标引擎
    ) -> Any:  # 返回RPC结果
        call_id = uuid.uuid1().int >> 64  # 生成唯一的64位call_id
        future = asyncio.get_running_loop().create_future()  # 创建asyncio Future
        self.utility_results[call_id] = future  # 注册到等待映射
        message = (  # 构建UTILITY消息
            EngineCoreRequestType.UTILITY.value,  # 请求类型
            *self.encoder.encode((self.client_index, call_id, method, args)),  # 序列化请求数据
        )
        await self._send_input_message(message, engine, args)  # 发送消息到目标引擎
        self._ensure_output_queue_task()  # 确保输出队列任务已启动
        return await future  # 等待并返回结果

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:  # 异步获取支持的任务类型
        return await self.call_utility_async("get_supported_tasks")  # 委托给call_utility_async

    async def add_request_async(self, request: EngineCoreRequest) -> None:  # 异步添加推理请求
        request.client_index = self.client_index  # 设置客户端索引
        await self._send_input(EngineCoreRequestType.ADD, request)  # 发送ADD请求
        self._ensure_output_queue_task()  # 确保输出队列任务已启动

    async def abort_requests_async(self, request_ids: list[str]) -> None:  # 异步中止请求
        if request_ids and not self.resources.engine_dead:  # 有请求且引擎存活
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)  # 发送ABORT请求

    async def pause_scheduler_async(  # 异步暂停调度器
        self, mode: PauseMode = "abort", clear_cache: bool = True  # 暂停模式和是否清除缓存
    ) -> None:  # 无返回值
        await self.call_utility_async("pause_scheduler", mode, clear_cache)  # 委托给call_utility_async

    async def resume_scheduler_async(self) -> None:  # 异步恢复调度器
        await self.call_utility_async("resume_scheduler")  # 委托给call_utility_async

    async def is_scheduler_paused_async(self) -> bool:  # 异步检查调度器是否暂停
        return await self.call_utility_async("is_scheduler_paused")  # 委托给call_utility_async

    async def profile_async(  # 异步性能分析
        self, is_start: bool = True, profile_prefix: str | None = None  # 是否开始、前缀
    ) -> None:  # 无返回值
        await self.call_utility_async("profile", is_start, profile_prefix)  # 委托给call_utility_async

    async def reset_mm_cache_async(self) -> None:  # 异步重置多模态缓存
        await self.call_utility_async("reset_mm_cache")  # 委托给call_utility_async

    async def reset_prefix_cache_async(  # 异步重置前缀缓存
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 可选参数
    ) -> bool:  # 返回是否成功
        return await self.call_utility_async(  # 委托给call_utility_async
            "reset_prefix_cache", reset_running_requests, reset_connector  # 传递参数
        )

    async def reset_encoder_cache_async(self) -> None:  # 异步重置编码器缓存
        await self.call_utility_async("reset_encoder_cache")  # 委托给call_utility_async

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:  # 异步使引擎休眠
        await self.call_utility_async("sleep", level, mode)  # 委托给call_utility_async

    async def wake_up_async(self, tags: list[str] | None = None) -> None:  # 异步唤醒引擎
        await self.call_utility_async("wake_up", tags)  # 委托给call_utility_async

    async def is_sleeping_async(self) -> bool:  # 异步检查是否休眠
        return await self.call_utility_async("is_sleeping")  # 委托给call_utility_async

    async def execute_dummy_batch_async(self) -> None:  # 异步执行虚拟批次
        await self.call_utility_async("execute_dummy_batch")  # 委托给call_utility_async

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:  # 异步添加LoRA适配器
        return await self.call_utility_async("add_lora", lora_request)  # 委托给call_utility_async

    async def remove_lora_async(self, lora_id: int) -> bool:  # 异步移除LoRA适配器
        return await self.call_utility_async("remove_lora", lora_id)  # 委托给call_utility_async

    async def list_loras_async(self) -> set[int]:  # 异步列出LoRA ID
        return await self.call_utility_async("list_loras")  # 委托给call_utility_async

    async def pin_lora_async(self, lora_id: int) -> bool:  # 异步固定LoRA
        return await self.call_utility_async("pin_lora", lora_id)  # 委托给call_utility_async

    async def save_sharded_state_async(  # 异步保存分片状态
        self, path: str, pattern: str | None = None, max_size: int | None = None  # 路径和可选参数
    ) -> None:  # 无返回值
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)  # 委托给call_utility_async

    async def collective_rpc_async(  # 异步集合RPC调用
        self,  # 实例自身
        method: str | Callable[..., _R],  # 方法名或可调用对象
        timeout: float | None = None,  # 超时时间
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:  # 返回结果列表
        return await self.call_utility_async(  # 委托给call_utility_async
            "collective_rpc", method, timeout, args, kwargs  # 传递参数
        )


# [中文注释] 数据并行异步客户端。管理多个 Engine Core 进程，支持：
#   1. Wave 同步机制：通过 Coordinator 的 stats_update 通道接收全局负载信息
#   2. 引擎暂停/唤醒：当所有引擎空闲时通过 first_req_send_socket 通知 Coordinator 唤醒
#   3. lb_engines: 每个引擎的 [waiting, running] 计数，用于负载均衡决策
#   4. Elastic EP 弹性伸缩：支持动态增减 Engine Core 进程数
class DPAsyncMPClient(AsyncMPClient):  # 数据并行异步客户端
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    def __init__(  # DPAsyncMPClient构造函数
        self,  # 实例自身
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, str] | None = None,  # 可选的外部管理地址
        client_count: int = 1,  # 客户端总数
        client_index: int = 0,  # 当前客户端索引
    ):
        self.current_wave = 0  # 初始化当前wave编号为0

        super().__init__(  # 调用父类AsyncMPClient构造函数
            vllm_config,  # vLLM配置
            executor_class,  # 执行器类
            log_stats,  # 统计开关
            client_addresses,  # 客户端地址
            client_count,  # 客户端数量
            client_index,  # 客户端索引
        )

        # List of [waiting, running] pair per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]  # 初始化每个引擎的[等待数, 运行数]

        self.eep_scaling_cache: ElasticScalingCache | None = None  # 弹性伸缩缓存，初始为None

        self.first_req_sock_addr = get_open_zmq_inproc_path()  # 获取进程内通信地址用于首请求通知
        self.first_req_send_socket = self.resources.first_req_send_socket = (  # 创建首请求发送PAIR socket
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)  # PAIR类型，绑定模式
        )
        try:  # 尝试在当前事件循环中启动统计任务
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()  # 获取当前事件循环
            self._ensure_stats_update_task()  # 启动统计更新任务
        except RuntimeError:  # 不在事件循环中
            pass  # 延迟启动

    # [中文注释] 懒初始化统计更新任务。内部协程 run_engine_stats_update_task：
    #   - 通过 XSUB socket 订阅 Coordinator 的负载统计广播
    #   - 通过 PAIR socket 接收本地 add_request_async 发来的 FIRST_REQ 或 SCALE_ELASTIC_EP 信号
    #   - 收到 FIRST_REQ 时：向 Coordinator 发送 (target_eng_index, current_wave) 唤醒引擎
    #   - 收到统计更新时：解码 (counts, wave, running)，更新 lb_engines 负载计数
    #   - 收到 SCALE_ELASTIC_EP 时：更新 engine_ranks_managed 和 lb_engines 列表长度，
    #     并转发伸缩通知给 Coordinator
    def _ensure_stats_update_task(self):  # 懒初始化统计更新任务
        resources = self.resources  # 获取资源管理器
        if resources.stats_update_task is not None:  # 如果任务已存在
            return  # 直接返回

        assert self.stats_update_address is not None  # 断言统计更新地址已设置
        stats_addr: str = self.stats_update_address  # 保存统计更新地址的局部引用
        assert len(self.engine_ranks_managed) > 0  # 断言至少管理一个引擎

        async def run_engine_stats_update_task():  # 异步统计更新协程
            with (  # 使用上下文管理器创建socket
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,  # 创建XSUB订阅socket
                make_zmq_socket(  # 创建首请求接收PAIR socket
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0  # 连接模式
                ) as first_req_rcv_socket,  # 首请求接收socket
            ):
                assert isinstance(socket, zmq.asyncio.Socket)  # 断言为异步socket
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)  # 断言为异步socket
                self.resources.stats_update_socket = socket  # 保存到资源管理器
                self.resources.first_req_rcv_socket = first_req_rcv_socket  # 保存到资源管理器
                # Send subscription message.
                await socket.send(b"\x01")  # 发送XSUB订阅消息

                poller = zmq.asyncio.Poller()  # 创建异步轮询器
                poller.register(socket, zmq.POLLIN)  # 注册统计socket为可读
                poller.register(first_req_rcv_socket, zmq.POLLIN)  # 注册首请求socket为可读

                while True:  # 主循环
                    events = await poller.poll()  # 等待事件
                    if (  # 检查是否收到首请求或伸缩通知
                        not self.engines_running  # 引擎未运行
                        and len(events) == 2  # 两个socket都有事件
                        or (events[0][0] == first_req_rcv_socket)  # 首请求socket有事件
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()  # 非阻塞接收首请求消息

                        decoded = msgspec.msgpack.decode(buf)  # 反序列化消息
                        if (  # 检查是否为弹性EP伸缩通知
                            isinstance(decoded, (list, tuple))  # 是列表或元组
                            and len(decoded) == 2  # 包含2个元素
                            and decoded[0] == "SCALE_ELASTIC_EP"  # 类型为弹性EP伸缩
                        ):
                            # Extract new engine count from the decoded message
                            new_engine_count = decoded[1]  # 提取新引擎数量
                            # Update engine_ranks_managed and count_slice
                            parallel_config = self.vllm_config.parallel_config  # 获取并行配置
                            dp_size = parallel_config.data_parallel_size  # 当前DP大小
                            dp_rank = parallel_config.data_parallel_rank  # 当前DP rank
                            assert dp_rank == 0  # 断言为rank 0
                            assert dp_size == new_engine_count  # 断言DP大小与新引擎数一致
                            assert not (  # 断言非混合LB或外部LB模式
                                parallel_config.data_parallel_hybrid_lb  # 混合负载均衡
                                or parallel_config.data_parallel_external_lb  # 外部负载均衡
                            )
                            num_ranks = dp_size  # rank数量等于DP大小
                            self.engine_ranks_managed = list(  # 更新管理的引擎rank列表
                                range(dp_rank, dp_rank + num_ranks)  # 从dp_rank开始
                            )
                            if len(self.lb_engines) < new_engine_count:  # 需要扩容lb_engines
                                self.lb_engines = self.lb_engines + [  # 追加新引擎的计数
                                    [0, 0]  # 初始化为[0, 0]
                                    for _ in range(  # 循环新增数量次
                                        new_engine_count - len(self.lb_engines)  # 差额
                                    )
                                ]
                            else:  # 需要缩容lb_engines
                                self.lb_engines = self.lb_engines[:new_engine_count]  # 截断到新数量
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(  # 编码伸缩通知消息
                                ("SCALE_ELASTIC_EP", new_engine_count)  # 消息内容
                            )
                            await socket.send(scale_msg)  # 发送通知给Coordinator
                            continue  # 继续下一轮

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"  # 断言为首请求通知
                        target_eng_index = decoded[1]  # 获取目标引擎索引
                        self.engines_running = True  # 标记引擎为运行状态
                        msg = msgspec.msgpack.encode(  # 编码唤醒消息
                            (target_eng_index, self.current_wave)  # (目标引擎, 当前wave)
                        )
                        await socket.send(msg)  # 发送唤醒消息给Coordinator

                    buf = None  # 初始化缓冲区
                    while True:  # 排空所有统计事件（只关心最新的）
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)  # 非阻塞接收
                        if isinstance(future.exception(), zmq.Again):  # 无更多消息
                            break  # 退出排空循环
                        buf = future.result()  # 获取最新的统计数据
                    if buf is None:  # 没有新的统计数据
                        continue  # 跳过本轮

                    # Update local load-balancing state.
                    counts, wave, running = msgspec.msgpack.decode(buf)  # 解码统计数据：计数、wave、运行状态
                    self.current_wave = wave  # 更新当前wave
                    self.engines_running = running  # 更新引擎运行状态
                    if counts is not None:  # 如果有计数数据
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        ranks = self.engine_ranks_managed  # 获取管理的rank列表
                        count_slice = slice(ranks[0], ranks[-1] + 1)  # 创建切片范围
                        sliced_counts = counts[count_slice]  # 切片获取本客户端管理的引擎计数
                        self.lb_engines = sliced_counts  # 更新负载均衡引擎计数
                        logger.debug(  # 调试日志
                            "Received counts: %s (%s)", sliced_counts, count_slice  # 输出计数和切片信息
                        )

        resources.stats_update_task = asyncio.create_task(  # 创建并注册统计更新任务
            run_engine_stats_update_task()  # 运行统计更新协程
        )

    # [中文注释] DP 版 add_request：
    #   1. 设置请求的 current_wave 和 client_index
    #   2. 调用 get_core_engine_for_request 选择目标引擎（子类可覆盖实现负载均衡）
    #   3. 若引擎当前处于暂停状态，通过 first_req_send_socket 通知 Coordinator 唤醒
    async def add_request_async(self, request: EngineCoreRequest) -> None:  # DP版异步添加请求
        self._ensure_stats_update_task()  # 确保统计更新任务已启动

        request.current_wave = self.current_wave  # 设置请求的当前wave编号
        request.client_index = self.client_index  # 设置客户端索引

        chosen_engine = self.get_core_engine_for_request(request)  # 选择目标引擎
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)  # 发送ADD请求
        if not self.engines_running:  # 如果引擎当前暂停
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))  # 编码首请求通知
            await self.first_req_send_socket.send(req_msg)  # 通知Coordinator唤醒引擎

        await to_await  # 等待发送完成

        self._ensure_output_queue_task()  # 确保输出队列任务已启动

    def get_core_engine_for_request(self, request: EngineCoreRequest):  # 选择目标引擎（默认返回主引擎，子类可覆盖）
        return self.core_engine  # 返回默认引擎


# [中文注释] 带负载均衡的数据并行客户端。在 DPAsyncMPClient 基础上增加：
#   1. get_core_engine_for_request：根据 lb_engines 中的 [waiting, running] 计数
#      选择负载最低的引擎（score = waiting*4 + running），实现客户端侧负载均衡
#   2. reqs_in_flight: 记录每个 request_id → engine 映射，用于将 abort 路由到正确的引擎
#   3. process_engine_outputs: 输出回调，清理已完成请求的 reqs_in_flight 记录
#   4. call_utility_async: 向所有引擎广播 utility 调用，只返回第一个结果
#   5. abort_requests_async: 按引擎分组路由 abort 请求
class DPLBAsyncMPClient(DPAsyncMPClient):  # 带负载均衡的数据并行客户端
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    def __init__(  # DPLBAsyncMPClient构造函数
        self,  # 实例自身
        vllm_config: VllmConfig,  # vLLM配置对象
        executor_class: type[Executor],  # 执行器类类型
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, str] | None = None,  # 可选的外部管理地址
        client_count: int = 1,  # 客户端总数
        client_index: int = 0,  # 当前客户端索引
    ):
        self.client_count = client_count  # 保存客户端总数

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}  # 请求ID到引擎的映射，用于abort路由

        super().__init__(  # 调用父类DPAsyncMPClient构造函数
            vllm_config,  # vLLM配置
            executor_class,  # 执行器类
            log_stats,  # 统计开关
            client_addresses,  # 客户端地址
            client_count,  # 客户端数量
            client_index,  # 客户端索引
        )

        assert len(self.core_engines) > 1  # 断言至少有2个引擎（负载均衡需要多引擎）

        self.eng_start_index = (  # 计算引擎遍历起始索引，平衡空闲时的分配
            len(self.core_engines) * self.client_index  # 按客户端索引偏移
        ) // client_count  # 除以客户端总数

    # [中文注释] 负载均衡选择引擎的核心算法：
    #   1. 优先使用 request.data_parallel_rank（显式指定）
    #   2. 其次检查 late_interaction 绑定（pooling 模型场景）
    #   3. 否则遍历所有引擎，计算 score = waiting*4 + running，选最低分引擎
    #      从 eng_start_index 开始轮询以平衡空闲时的分配
    #   4. 本地递增 waiting 计数（+client_count），在下次 Coordinator 更新前保持精度
    #   5. 将 request_id → chosen_engine 记录到 reqs_in_flight，用于后续 abort 路由
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:  # 负载均衡选择目标引擎
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (  # 如果未显式指定DP rank
            eng_index := get_late_interaction_engine_index(  # 检查late interaction绑定
                request.pooling_params, len(self.core_engines)  # pooling参数和引擎数
            )
        ) is None:  # 也没有late interaction绑定
            current_counts = self.lb_engines  # 获取当前负载计数
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)  # 引擎总数
            min_score = sys.maxsize  # 初始化最小分数为最大值
            eng_index = 0  # 初始化最优引擎索引
            for i in range(num_engines):  # 遍历所有引擎
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines  # 从起始索引轮询
                waiting, running = current_counts[idx]  # 获取等待和运行计数
                score = waiting * 4 + running  # 计算负载分数（等待权重4:1运行权重）
                if score < min_score:  # 找到更低分数的引擎
                    min_score = score  # 更新最小分数
                    eng_index = idx  # 更新最优引擎索引
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count  # 本地递增等待计数

        chosen_engine = self.core_engines[eng_index]  # 获取选中引擎的身份标识
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine  # 记录请求到引擎的映射
        return chosen_engine  # 返回选中的引擎

    # [中文注释] 覆盖父类方法：向所有引擎广播 utility 调用（如 reset_prefix_cache），
    #   使用 asyncio.gather 并发发送，只返回第一个引擎的结果
    async def call_utility_async(self, method: str, *args) -> Any:  # 向所有引擎广播utility调用
        # Only the result from the first engine is returned.
        return (  # 返回第一个引擎的结果
            await asyncio.gather(  # 并发等待所有引擎
                *[  # 展开生成器为参数
                    self._call_utility_async(method, *args, engine=engine)  # 向每个引擎发送RPC
                    for engine in self.core_engines  # 遍历所有引擎
                ]
            )
        )[0]  # 取第一个引擎的结果

    # [中文注释] 输出回调：当 process_outputs_socket 收到推理结果时被调用，
    #   从 reqs_in_flight 中移除已完成的请求，释放路由记录
    @staticmethod  # 静态方法装饰器
    async def process_engine_outputs(  # 输出回调：清理已完成请求的路由记录
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs  # 客户端实例和输出
    ):
        if outputs.finished_requests and self.reqs_in_flight:  # 如果有已完成请求且存在飞行中请求
            for req_id in outputs.finished_requests:  # 遍历已完成的请求ID
                self.reqs_in_flight.pop(req_id, None)  # 从路由映射中移除

    # [中文注释] Elastic EP 通知回调：处理来自 Engine Core 的弹性伸缩事件通知。
    #   - RECONFIGURE_FINISHED: 所有引擎完成重配置，解除 _eep_wait_for_setup_switch_complete 的等待
    #   - SHUTDOWN_COMPLETE: 缩容场景，调用 engine_manager.scale_down_elastic_ep 清理已关闭的进程
    #   - NEW_CORE_ENGINES_WEIGHTS_INIT_READY: 新引擎权重初始化完成，通知现有引擎同步
    #   pending_notifications 按 dp_rank 去重计数，满足数量后触发下一阶段操作
    @staticmethod  # 静态方法装饰器
    async def eep_process_engine_core_notification(  # EEP通知回调：处理弹性伸缩事件
        self: "DPLBAsyncMPClient", notification_data: tuple[str, int]  # 客户端实例和通知数据(类型, dp_rank)
    ):
        cache = self.eep_scaling_cache  # 获取弹性伸缩缓存
        notification_type_str, dp_rank = notification_data  # 解构通知类型和dp_rank
        try:  # 尝试解析通知类型
            notification_type = EEPNotificationType(notification_type_str)  # 转换为枚举类型
        except ValueError as e:  # 未知通知类型
            raise ValueError(  # 抛出异常
                f"Unknown EEP notification type: {notification_type_str}"  # 错误消息
            ) from e  # 保留原始异常链

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:  # 如果是重配置完成通知
            from vllm.v1.engine import UtilityResult  # 延迟导入UtilityResult

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            dummy_output = UtilityOutput(  # 创建虚拟输出用于解除Future等待
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)  # 使用EEP通知的特殊call_id
            )
            _process_utility_output(dummy_output, self.utility_results)  # 设置Future结果，解除等待
            return  # 直接返回
        assert cache is not None  # 断言缓存存在
        if notification_type not in cache.pending_notifications:  # 如果该通知类型尚未记录
            cache.pending_notifications[notification_type] = set()  # 初始化为空集合
        if dp_rank in cache.pending_notifications[notification_type]:  # 检查是否重复通知
            raise ValueError(  # 抛出重复通知异常
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"  # 错误消息
            )
        cache.pending_notifications[notification_type].add(dp_rank)  # 记录已收到通知的dp_rank
        if len(cache.pending_notifications[notification_type]) >= abs(  # 检查是否所有引擎都已通知
            cache.num_new_core_engines  # 与需要通知的引擎数比较
        ):
            if notification_type == EEPNotificationType.SHUTDOWN_COMPLETE:  # 如果是关闭完成通知（缩容）
                assert isinstance(self.resources.engine_manager, CoreEngineActorManager)  # 断言为Actor管理器
                assert cache.num_new_core_engines < 0  # 断言为缩容（负数）
                old_dp_size = len(cache.existing_core_engines)  # 旧DP大小
                new_dp_size = old_dp_size + cache.num_new_core_engines  # 新DP大小
                self.resources.engine_manager.scale_down_elastic_ep(  # 执行缩容清理
                    old_dp_size, new_dp_size  # 传入新旧DP大小
                )
            else:  # 其他通知类型（如权重初始化完成）
                await asyncio.gather(  # 并发通知所有现有引擎
                    *[  # 展开任务列表
                        self._call_utility_async(  # 向每个引擎发送处理通知的RPC
                            "eep_handle_engine_core_notification",  # 方法名
                            notification_type,  # 通知类型
                            engine=engine,  # 目标引擎
                        )
                        for engine in cache.existing_core_engines  # 遍历现有引擎
                    ]
                )
            cache.pending_notifications[notification_type] = set()  # 清空该类型的待处理通知
            if notification_type in [  # 如果是终结性通知类型
                EEPNotificationType.SHUTDOWN_COMPLETE,  # 关闭完成
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,  # 权重初始化完成
            ]:
                self.eep_scaling_cache = None  # 清空弹性伸缩缓存

    # [中文注释] DP-LB 版 abort：根据 reqs_in_flight 查找每个请求所在的引擎，
    #   按引擎分组后分别发送 ABORT 消息。单请求走快速路径避免 dict 分组开销。
    async def abort_requests_async(self, request_ids: list[str]) -> None:  # DP-LB版异步中止请求
        if not request_ids or self.resources.engine_dead:  # 无请求或引擎已死
            return  # 直接返回

        if len(request_ids) == 1:  # 单请求快速路径
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):  # 查找请求所在引擎
                await self._abort_requests(request_ids, engine)  # 向对应引擎发送abort
            return  # 返回

        by_engine = defaultdict[EngineIdentity, list[str]](list)  # 按引擎分组
        for req_id in request_ids:  # 遍历所有请求ID
            if engine := self.reqs_in_flight.get(req_id):  # 查找请求所在引擎
                by_engine[engine].append(req_id)  # 分组到对应引擎
        for engine, req_ids in by_engine.items():  # 遍历各引擎组
            await self._abort_requests(req_ids, engine)  # 向每个引擎发送对应的abort

    async def _abort_requests(  # 向指定引擎发送abort请求
        self, request_ids: list[str], engine: EngineIdentity  # 请求ID列表和目标引擎
    ) -> None:  # 无返回值
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)  # 发送ABORT请求到指定引擎

    # [中文注释] Elastic Expert Parallelism 弹性伸缩入口：
    #   根据 new_data_parallel_size 与当前大小的比较，分发到 _scale_up 或 _scale_down
    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:  # 弹性EP伸缩入口
        """Scale elastic EP data parallel size"""
        cur_data_parallel_size = len(self.core_engines)  # 获取当前引擎数量

        assert new_data_parallel_size != cur_data_parallel_size, (  # 断言新旧大小不同
            f"new_data_parallel_size {new_data_parallel_size} must be "  # 错误消息
            f"different from cur_data_parallel_size {cur_data_parallel_size}"  # 错误消息
        )

        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (  # 断言使用Ray后端
            "Only ray DP backend supports scaling elastic EP"  # 仅Ray支持弹性EP
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size  # 判断是扩容还是缩容

        if scale_up:  # 扩容
            await self._scale_up_elastic_ep(  # 调用扩容方法
                cur_data_parallel_size, new_data_parallel_size  # 传入当前和目标大小
            )
        else:  # 缩容
            await self._scale_down_elastic_ep(  # 调用缩容方法
                cur_data_parallel_size, new_data_parallel_size  # 传入当前和目标大小
            )

    async def _eep_wait_for_setup_switch_complete(self) -> None:  # 等待所有引擎完成配置切换
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        future = asyncio.get_running_loop().create_future()  # 创建asyncio Future
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future  # 用EEP通知call_id注册Future
        self._ensure_output_queue_task()  # 确保输出队列任务已启动
        await future  # 等待Future被解除（由eep_process_engine_core_notification设置）

    # [中文注释] 弹性扩容流程（3 阶段）：
    #   阶段1: 向所有现有引擎发送 reinitialize_distributed 重配置请求（更新分布式参数）
    #   阶段2: 通过 engine_manager.scale_up_elastic_ep 创建新的 Engine Core 进程
    #   阶段3: 等待新引擎就绪（ZMQ 握手）+ 等待所有引擎完成 setup switch
    #   最后通知 Coordinator 更新全局 DP 大小
    async def _scale_up_elastic_ep(  # 弹性扩容实现
        self, cur_data_parallel_size: int, new_data_parallel_size: int  # 当前和目标DP大小
    ) -> None:  # 无返回值
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        cur_data_parallel_size = len(self.core_engines)  # 重新获取当前引擎数量

        self.eep_scaling_cache = ElasticScalingCache(  # 创建弹性伸缩缓存
            existing_core_engines=self.core_engines.copy(),  # 拷贝现有引擎列表
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,  # 计算新增引擎数
            pending_notifications=dict(),  # 初始化空的待处理通知
        )

        parallel_config = self.vllm_config.parallel_config  # 获取并行配置
        allocate_stateless_group_ports(parallel_config, new_data_parallel_size)  # 分配新的无状态组端口

        # Phase 1: Send reconfig messages to existing engines
        reconfig_futures = []  # 收集重配置任务的Future
        for engine in self.core_engines:  # 遍历现有引擎
            reconfig_request = ReconfigureDistributedRequest(  # 构建重配置请求
                new_data_parallel_size=new_data_parallel_size,  # 新DP大小
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,  # 保持当前rank
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,  # 保持当前本地rank
                new_data_parallel_master_ip=parallel_config.data_parallel_master_ip,  # master IP
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,  # master端口
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,  # master端口列表
                new_stateless_world_group_port_list=parallel_config._stateless_world_group_port_list,  # 世界组端口列表
                new_stateless_dp_group_port_list=parallel_config._stateless_dp_group_port_list,  # DP组端口列表
                new_stateless_ep_group_port_list=parallel_config._stateless_ep_group_port_list,  # EP组端口列表
                new_stateless_eplb_group_port_list=parallel_config._stateless_eplb_group_port_list,  # EPLB组端口列表
            )
            coro = self._call_utility_async(  # 创建异步RPC调用协程
                "reinitialize_distributed", reconfig_request, engine=engine  # 发送到指定引擎
            )
            reconfig_futures.append(asyncio.create_task(coro))  # 创建任务并收集

        # Phase 2: Create new engines
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)  # 断言为Actor管理器
        parallel_config.eplb_config.num_redundant_experts = 0  # 新引擎无冗余专家
        start_new_worker_future = asyncio.to_thread(  # 在线程中启动新引擎（避免阻塞事件循环）
            self.resources.engine_manager.scale_up_elastic_ep,  # 调用引擎管理器扩容方法
            self.vllm_config,  # vLLM配置
            new_data_parallel_size,  # 目标DP大小
        )
        wait_future = self._eep_wait_for_setup_switch_complete()  # 创建等待配置切换完成的Future

        # Phase 3: Wait for new engines to be created
        # and reconfig messages to be received
        await asyncio.gather(start_new_worker_future, *reconfig_futures)  # 并发等待新引擎启动和重配置完成
        logger.info("[Elastic EP] Successfully started new engines")  # 记录扩容成功日志

        # Create new CoreEngine objects for the new engines
        new_engine_identities = set()  # 新引擎身份标识集合
        for i in range(cur_data_parallel_size, new_data_parallel_size):  # 遍历新引擎rank
            new_engine = i.to_bytes(2, "little")  # 转换为2字节小端身份标识
            self.core_engines.append(new_engine)  # 添加到引擎列表
            # NOTE(yongji): we don't update lb_engines here,
            # we let run_engine_stats_update_task to update it.
            new_engine_identities.add(new_engine)  # 添加到待确认集合

        # Wait for ready messages from new engines on the input socket
        sync_input_socket = zmq.Socket.shadow(self.input_socket)  # 创建同步影子socket
        while new_engine_identities:  # 循环直到所有新引擎就绪
            if not sync_input_socket.poll(  # 轮询等待就绪消息
                timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms  # 超时时间
            ):
                raise TimeoutError(  # 超时异常
                    f"Timed out waiting for new engine core processes to "  # 错误消息
                    f"start. Waited "  # 等待时间说明
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "  # 配置来源
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "  # 如何增加超时
                    f"timeout, set the environment variable: "  # 环境变量说明
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"  # 环境变量名称
                )
            identity, _ = sync_input_socket.recv_multipart()  # 接收就绪消息
            new_engine_identities.discard(identity)  # 从待确认集合中移除

        # NOTE(yongji): Before we schedule any requests on the new workers,
        # we should wait for them to switch to the new setup.
        await wait_future  # 等待所有引擎完成配置切换
        # Update the parallel config
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size  # 更新并行配置的DP大小
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()  # 确保统计更新任务已启动
        scale_up_marker = msgspec.msgpack.encode(  # 编码扩容通知消息
            ("SCALE_ELASTIC_EP", new_data_parallel_size)  # 消息内容
        )
        await self.first_req_send_socket.send(scale_up_marker)  # 通知Coordinator更新DP大小

        logger.info(  # 记录扩容完成日志
            "[Elastic EP] Scale up completed, new data parallel size: %s",  # 日志消息
            new_data_parallel_size,  # 新DP大小
        )

    # [中文注释] 弹性缩容流程：
    #   1. 向所有引擎发送 reinitialize_distributed 请求，超出新 DP 大小的引擎标记为 SHUTDOWN
    #   2. 立即截断 core_engines 和 lb_engines 列表，停止向被移除的引擎发送新请求
    #   3. 通知 Coordinator 更新 DP 大小，等待 setup switch 完成
    #   被关闭的引擎会发送 SHUTDOWN_COMPLETE 通知，由 eep_process_engine_core_notification 处理清理
    async def _scale_down_elastic_ep(  # 弹性缩容实现
        self, cur_data_parallel_size: int, new_data_parallel_size: int  # 当前和目标DP大小
    ) -> None:  # 无返回值
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        cur_data_parallel_size = len(self.core_engines)  # 重新获取当前引擎数量

        self.eep_scaling_cache = ElasticScalingCache(  # 创建弹性伸缩缓存
            existing_core_engines=self.core_engines.copy(),  # 拷贝现有引擎列表
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,  # 负数表示缩容数量
            pending_notifications=dict(),  # 初始化空的待处理通知
        )

        parallel_config = self.vllm_config.parallel_config  # 获取并行配置
        allocate_stateless_group_ports(parallel_config, new_data_parallel_size)  # 分配新的无状态组端口

        reconfig_futures = []  # 收集重配置任务的Future
        for cur_dp_rank, engine in enumerate(self.core_engines):  # 遍历所有引擎（含即将关闭的）
            reconfig_request = ReconfigureDistributedRequest(  # 构建重配置请求
                new_data_parallel_size=new_data_parallel_size,  # 新DP大小
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,  # 默认保持当前rank
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,  # 保持当前本地rank
                new_data_parallel_master_ip=parallel_config.data_parallel_master_ip,  # master IP
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,  # master端口
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,  # master端口列表
                new_stateless_world_group_port_list=parallel_config._stateless_world_group_port_list,  # 世界组端口列表
                new_stateless_dp_group_port_list=parallel_config._stateless_dp_group_port_list,  # DP组端口列表
                new_stateless_ep_group_port_list=parallel_config._stateless_ep_group_port_list,  # EP组端口列表
                new_stateless_eplb_group_port_list=parallel_config._stateless_eplb_group_port_list,  # EPLB组端口列表
            )
            if cur_dp_rank >= new_data_parallel_size:  # 如果rank超出新DP大小，标记为关闭
                reconfig_request.new_data_parallel_rank = (  # 设置关闭标记
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK  # 关闭当前rank
                )
            coro = self._call_utility_async(  # 创建异步RPC调用协程
                "reinitialize_distributed", reconfig_request, engine=engine  # 发送到指定引擎
            )
            reconfig_futures.append(asyncio.create_task(coro))  # 创建任务并收集

        # NOTE(yongji): Immediately stop sending requests to the removing engines.
        self.core_engines = self.core_engines[:new_data_parallel_size]  # 截断引擎列表，停止向被移除引擎发送请求
        self.lb_engines = self.lb_engines[:new_data_parallel_size]  # 截断负载均衡计数列表
        wait_future = self._eep_wait_for_setup_switch_complete()  # 创建等待配置切换完成的Future

        await asyncio.gather(*reconfig_futures)  # 并发等待所有重配置完成

        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size  # 更新并行配置的DP大小
        self._ensure_stats_update_task()  # 确保统计更新任务已启动
        scale_down_marker = msgspec.msgpack.encode(  # 编码缩容通知消息
            ("SCALE_ELASTIC_EP", new_data_parallel_size)  # 消息内容
        )
        await self.first_req_send_socket.send(scale_down_marker)  # 通知Coordinator更新DP大小

        # NOTE(yongji): Unlike scaling up,
        # here we don't actually need to wait for the setup switch to complete.
        # We may want to remove it in the future.
        await wait_future  # 等待配置切换完成（未来可能移除）
        logger.info(  # 记录缩容完成日志
            "[Elastic EP] Scale down completed, new data parallel size: %s",  # 日志消息
            new_data_parallel_size,  # 新DP大小
        )
