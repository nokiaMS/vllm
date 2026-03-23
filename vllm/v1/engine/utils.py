# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 导入上下文管理器工具模块，用于创建上下文管理器
import contextlib  # 导入上下文管理器库
# 导入操作系统接口模块，用于环境变量操作
import os  # 导入操作系统接口
# 导入线程模块，用于信号回调中的线程管理
import threading  # 导入线程管理模块
# 导入弱引用模块，用于注册析构函数实现自动清理
import weakref  # 导入弱引用模块
# 从 collections.abc 导入 Callable（可调用对象类型）和 Iterator（迭代器类型）
from collections.abc import Callable, Iterator  # 导入可调用对象和迭代器类型
# 导入 dataclass 装饰器，用于简化数据类的定义
from dataclasses import dataclass  # 导入数据类装饰器
# 导入枚举基类 Enum 和自动枚举值生成器 auto
from enum import Enum, auto  # 导入枚举基类和自动值
# 从 multiprocessing 导入 Process（进程类）和 connection（进程间连接工具）
from multiprocessing import Process, connection  # 导入进程和连接模块
# 导入 BaseProcess 基类，作为进程列表的类型注解
from multiprocessing.process import BaseProcess  # 导入进程基类
# 导入 TYPE_CHECKING 标志，仅在类型检查时为 True，避免运行时循环导入
from typing import TYPE_CHECKING  # 导入类型检查标志
# 导入 unittest.mock.patch，用于临时修改环境变量
from unittest.mock import patch  # 导入 mock patch 工具

# 导入 msgspec 库，用于高效的消息序列化和反序列化（msgpack 格式）
import msgspec  # 导入高效序列化库
# 导入 ZeroMQ 消息队列库，用于引擎与客户端之间的进程间通信
import zmq  # 导入 ZeroMQ 通信库

# 从 vllm 导入环境变量配置模块
from vllm import envs  # 导入环境变量配置
# 从 vllm.config 导入缓存配置、并行配置和 vLLM 总配置类
from vllm.config import CacheConfig, ParallelConfig, VllmConfig  # 导入配置类
# 从 vllm.logger 导入日志初始化函数
from vllm.logger import init_logger  # 导入日志初始化函数
# 从 vllm.platforms 导入当前平台信息（CUDA/ROCm/XPU 等）
from vllm.platforms import current_platform  # 导入当前平台信息
# 从 vllm.ray.ray_env 导入获取需要复制到 Ray Actor 的环境变量列表的函数
from vllm.ray.ray_env import get_env_vars_to_copy  # 导入 Ray 环境变量复制函数
# 从 vllm.utils.network_utils 导入获取 ZMQ IPC 路径和 ZMQ socket 上下文管理器的工具函数
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx  # 导入网络工具函数
# 从 vllm.utils.system_utils 导入获取多进程上下文的函数（fork/spawn/forkserver）
from vllm.utils.system_utils import get_mp_context  # 导入多进程上下文获取函数
# 从 vllm.v1.engine.coordinator 导入数据并行协调器类
from vllm.v1.engine.coordinator import DPCoordinator  # 导入数据并行协调器
# 从 vllm.v1.executor 导入执行器基类
from vllm.v1.executor import Executor  # 导入执行器基类
# 从 vllm.v1.utils 导入获取引擎客户端 ZMQ 地址和关闭进程的工具函数
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown  # 导入地址获取和关闭函数

# 仅在类型检查时导入 Ray 的 PlacementGroup 类型，避免运行时依赖
if TYPE_CHECKING:  # 仅在类型检查阶段执行以下导入
    # 导入 Ray 放置组类型，用于类型注解
    from ray.util.placement_group import PlacementGroup  # 导入放置组类型注解

# 初始化当前模块的日志记录器
logger = init_logger(__name__)  # 创建当前模块的日志记录器实例

# 启动阶段轮询超时时间（毫秒），用于等待引擎进程就绪时的 ZMQ 轮询间隔
STARTUP_POLL_PERIOD_MS = 10000  # 启动轮询超时常量：10秒


# [中文注释] 引擎握手阶段的状态机：NEW → CONNECTED（收到 HELLO）→ READY（收到 READY）
class CoreEngineState(Enum):  # 核心引擎状态枚举类定义
    """核心引擎状态枚举类。

    定义了引擎在启动握手过程中的三个状态：
    - NEW: 新建状态，尚未与前端建立连接
    - CONNECTED: 已连接状态，已收到 HELLO 消息并回复了握手元数据
    - READY: 就绪状态，引擎已完成初始化并发送了 READY 消息
    """

    NEW = auto()  # 新建状态：引擎刚创建，等待 HELLO 消息
    CONNECTED = auto()  # 已连接状态：已完成 HELLO 握手，等待 READY 消息
    READY = auto()  # 就绪状态：引擎已完全初始化，可以处理请求


# [中文注释] 每个 DP rank 对应一个 CoreEngine 实例，用于启动阶段跟踪握手状态
class CoreEngine:  # 核心引擎实例类，跟踪每个 DP rank 的握手状态
    """核心引擎实例类，用于跟踪数据并行中每个引擎的握手状态。

    每个数据并行 rank 对应一个 CoreEngine 实例。在启动握手阶段，
    前端通过此对象跟踪每个引擎从 NEW → CONNECTED → READY 的状态转换。
    """
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):  # 初始化核心引擎实例
        """初始化核心引擎实例。

        Args:
            index: 引擎的数据并行 rank 索引
            local: 是否为本地引擎（与前端在同一节点）
        """
        self.local = local  # 标记此引擎是否在本地节点上运行
        self.identity = index.to_bytes(2, "little")  # 将索引转换为 2 字节小端序，作为 ZMQ 消息的身份标识

        self.state = CoreEngineState.NEW  # 初始状态设置为 NEW（新建）


# [中文注释] Engine 与 Client 之间的 ZMQ 地址配置。
#   inputs — 每个前端 Client 的请求输入地址（ROUTER socket 绑定）
#   outputs — 每个前端 Client 的响应输出地址（PUSH socket 连接）
#   coordinator_input/output — DP Coordinator 的输入/输出地址（可选）
#   frontend_stats_publish_address — 前端连接 Coordinator 订阅统计的地址（external DP LB 模式）
@dataclass  # 使用 dataclass 装饰器自动生成 __init__、__repr__ 等方法
class EngineZmqAddresses:  # 引擎 ZMQ 地址配置数据类
    """引擎 ZMQ 地址配置数据类。

    存储引擎与客户端之间通信所需的所有 ZMQ socket 地址，
    包括请求输入、响应输出以及数据并行协调器的地址。
    """

    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]  # 每个前端客户端的请求输入 ZMQ 地址列表（ROUTER socket 绑定地址）
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]  # 每个前端客户端的响应输出 ZMQ 地址列表（PUSH socket 连接地址）
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None  # 数据并行协调器的输入 socket 地址（引擎向协调器发送消息）
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None  # 数据并行协调器的输出 socket 地址（协调器向引擎发送消息）
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None  # 前端连接协调器订阅统计信息的地址（仅 external DP LB 模式使用）


@dataclass  # 使用 dataclass 装饰器定义握手元数据结构
class EngineHandshakeMetadata:  # 引擎握手元数据数据类
    """引擎握手元数据类。

    在启动握手阶段发送给每个引擎进程的元数据，
    包含前端 ZMQ 队列的地址信息以及并行配置参数。
    """
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses  # 引擎需要连接的 ZMQ 地址配置
    parallel_config: dict[str, int | str | list[int]]  # 并行配置参数字典（数据并行主节点 IP、端口、DP 大小等）


# [中文注释] 本地多进程模式的引擎进程管理器。
#   为每个本地 DP rank 启动一个 EngineCoreProc 子进程，管理其生命周期：
#   - 启动时设置 CUDA_VISIBLE_DEVICES（非 CUDA 平台或 Ray 模式）
#   - shutdown() 优雅终止所有子进程
#   - finished_procs() 检测已退出的进程（用于启动阶段错误检测）
class CoreEngineProcManager:  # 核心引擎进程管理器类定义
    """核心引擎进程管理器类。

    负责创建、监控和关闭 AsyncLLM 和 LLMEngine 所使用的后台引擎进程。
    在本地多进程模式下，为每个本地数据并行 rank 启动一个 EngineCoreProc 子进程，
    并通过 weakref.finalize 确保进程在对象被垃圾回收时自动清理。
    """
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(  # 构造函数定义
        self,  # 实例自身引用
        local_engine_count: int,  # 本地引擎进程数量
        start_index: int,  # 全局起始 DP rank 索引
        local_start_index: int,  # 本地起始 DP rank 索引
        vllm_config: VllmConfig,  # vLLM 总配置对象
        local_client: bool,  # 是否为本地客户端模式
        handshake_address: str,  # 握手通信的 ZMQ 地址
        executor_class: type[Executor],  # 执行器类（GPU/CPU 等）
        log_stats: bool,  # 是否记录统计信息
        client_handshake_address: str | None = None,  # 客户端专用握手地址（可选，用于 external DP LB 模式）
    ):  # 构造函数参数列表结束
        """初始化引擎进程管理器并启动所有本地引擎子进程。

        Args:
            local_engine_count: 需要启动的本地引擎进程数
            start_index: 全局数据并行 rank 起始索引
            local_start_index: 本地数据并行 rank 起始索引
            vllm_config: vLLM 配置对象
            local_client: 引擎是否与本地客户端通信
            handshake_address: 握手 ZMQ 地址
            executor_class: 执行器类
            log_stats: 是否启用统计日志
            client_handshake_address: 客户端握手地址（可选）
        """
        context = get_mp_context()  # 获取多进程上下文（fork/spawn/forkserver）
        common_kwargs = {  # 所有引擎进程共享的启动参数字典
            "vllm_config": vllm_config,  # vLLM 配置
            "local_client": local_client,  # 本地客户端标志
            "handshake_address": handshake_address,  # 握手地址
            "executor_class": executor_class,  # 执行器类
            "log_stats": log_stats,  # 统计日志开关
        }  # 结束公共参数字典定义

        if client_handshake_address:  # 如果指定了客户端握手地址
            common_kwargs["client_handshake_address"] = client_handshake_address  # 添加到公共参数中

        is_dp = vllm_config.parallel_config.data_parallel_size > 1  # 判断是否为数据并行模式（DP size > 1）

        from vllm.v1.engine.core import EngineCoreProc  # 延迟导入引擎核心进程类，避免循环导入

        self.processes: list[BaseProcess] = []  # 初始化子进程列表
        local_dp_ranks = []  # 本地 DP rank 列表，用于后续设置设备环境变量
        for index in range(local_engine_count):  # 遍历每个需要启动的本地引擎
            local_index = local_start_index + index  # 计算本地 DP rank 索引
            global_index = start_index + index  # 计算全局 DP rank 索引

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)  # 记录本地 DP rank
            self.processes.append(  # 创建并添加子进程到列表
                context.Process(  # 使用多进程上下文创建进程
                    target=EngineCoreProc.run_engine_core,  # 进程入口函数
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",  # 进程名称（DP 模式带 rank 编号）
                    kwargs=common_kwargs  # 公共参数
                    | {"dp_rank": global_index, "local_dp_rank": local_index},  # 合并 rank 参数
                )  # 结束 context.Process 构造
            )  # 结束 self.processes.append 调用

        self._finalizer = weakref.finalize(self, shutdown, self.processes)  # 注册弱引用析构函数，确保对象销毁时自动关闭子进程

        try:  # 尝试启动所有子进程
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):  # 遍历进程和对应的本地 rank
                # Adjust device control in DP for non-CUDA platforms
                # as well as external and ray launchers
                # For CUDA platforms, we use torch.accelerator.set_device_index()()
                if is_dp and (  # 在 DP 模式下，如果是非 CUDA 平台或使用 Ray
                    not current_platform.is_cuda_alike()  # 非 CUDA 类平台（如 XPU、ROCm）
                    or vllm_config.parallel_config.use_ray  # 或使用 Ray 分布式框架
                ):
                    with set_device_control_env_var(vllm_config, local_dp_rank):  # 临时设置设备控制环境变量
                        proc.start()  # 在设置了环境变量的上下文中启动子进程
                else:
                    proc.start()  # 直接启动子进程（CUDA 平台使用 torch.accelerator.set_device_index()）
        finally:  # 无论是否异常，都检查进程状态
            # Kill other procs if not all are running.
            if self.finished_procs():  # 如果有进程已经退出（启动失败）
                self.shutdown()  # 关闭所有已启动的进程

    def shutdown(self, timeout: float | None = None) -> None:  # 关闭所有引擎核心子进程方法
        """关闭所有引擎核心子进程。

        Args:
            timeout: 等待进程退出的超时时间（秒），None 表示使用默认超时
        """
        """Shutdown engine core processes with configurable timeout."""
        if self._finalizer.detach() is not None:  # 分离析构函数（防止重复清理），如果尚未被分离
            shutdown(self.processes, timeout=timeout)  # 调用 shutdown 工具函数终止所有子进程

    def join_first(self):  # 等待任意一个子进程退出方法
        """等待任意一个子进程退出。

        使用 multiprocessing.connection.wait 监听所有进程的 sentinel，
        当任意一个进程退出时立即返回。
        """
        """Wait for any process to exit."""
        connection.wait(proc.sentinel for proc in self.processes)  # 等待任意进程的 sentinel 变为可读（即进程退出）

    def sentinels(self) -> list:  # 获取所有子进程的 sentinel 列表方法
        """获取所有子进程的 sentinel 列表。

        Returns:
            所有子进程的 sentinel 对象列表，可用于 ZMQ Poller 或 connection.wait
        """
        return [proc.sentinel for proc in self.processes]  # 返回所有进程的 sentinel 句柄列表

    def finished_procs(self) -> dict[str, int]:  # 获取已退出进程及退出码方法
        """获取已退出的进程及其退出码。

        Returns:
            字典，键为进程名称，值为退出码。仅包含已退出的进程。
        """
        """Returns dict of proc name -> exit code for any finished procs."""
        return {  # 构建已退出进程的字典
            proc.name: proc.exitcode  # 进程名称 -> 退出码
            for proc in self.processes  # 遍历所有进程
            if proc.exitcode is not None  # 仅包含已退出的进程（exitcode 不为 None）
        }


class SignalCallback:  # 信号回调安全触发器类定义
    """信号回调安全触发器类。

    通过专用守护线程安全地从信号处理器上下文中触发回调函数。
    信号处理器中不能直接执行复杂操作，因此通过 Event 通知守护线程来执行回调。
    """
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    def __init__(self, callback: Callable[[], None]):  # 初始化信号回调触发器
        """初始化信号回调触发器。

        Args:
            callback: 收到信号时需要执行的回调函数
        """
        self._callback = callback  # 保存回调函数引用
        self._event = threading.Event()  # 创建线程事件对象，用于跨线程通知
        self._stopped = False  # 停止标志，防止在 stop() 后执行回调
        self._thread = threading.Thread(  # 创建守护线程
            target=self._run,  # 线程执行函数
            daemon=True,  # 设为守护线程，主线程退出时自动终止
            name="signal-callback",  # 线程名称
        )  # 守护线程对象构造完成
        self._thread.start()  # 启动守护线程，开始等待事件

    def _run(self):  # 守护线程执行函数
        """守护线程的执行函数，等待事件触发后执行回调。"""
        self._event.wait()  # 阻塞等待事件被 set()
        if not self._stopped:  # 如果不是被 stop() 触发的
            self._callback()  # 执行回调函数

    def trigger(self):  # 触发回调执行方法
        """触发回调执行，由信号处理器调用。"""
        self._event.set()  # 设置事件，唤醒守护线程执行回调

    def stop(self):  # 停止回调触发器方法
        """停止回调触发器，不再执行回调。"""
        self._stopped = True  # 设置停止标志
        self._event.set()  # 唤醒守护线程（但因 _stopped 为 True 不会执行回调）


@contextlib.contextmanager  # 将函数转换为上下文管理器
def set_device_control_env_var(  # 临时设置设备控制环境变量的上下文管理器函数
    vllm_config: VllmConfig, local_dp_rank: int  # 配置对象和本地 DP rank 参数
) -> Iterator[None]:  # 返回无值迭代器（上下文管理器）
    """临时设置设备控制环境变量的上下文管理器。

    在子进程启动前临时设置 CUDA_VISIBLE_DEVICES 或等效环境变量，
    使子进程只能看到分配给该 DP rank 的设备。

    Args:
        vllm_config: vLLM 配置对象
        local_dp_rank: 本地数据并行 rank 索引
    """
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = vllm_config.parallel_config.world_size  # 获取单个 DP rank 的世界大小（TP * PP）
    local_world_size = vllm_config.parallel_config.local_world_size  # 获取本地世界大小
    evar = current_platform.device_control_env_var  # 获取当前平台的设备控制环境变量名（如 CUDA_VISIBLE_DEVICES）

    value = get_device_indices(evar, local_dp_rank, world_size, local_world_size)  # 计算该 rank 应使用的设备索引字符串
    with patch.dict(os.environ, values=((evar, value),)):  # 临时将环境变量设置为计算出的值
        yield  # 在此上下文中执行子进程启动


def get_device_indices(  # 计算设备索引字符串的函数
    device_control_env_var: str,  # 设备控制环境变量名
    local_dp_rank: int,  # 本地数据并行 rank
    world_size: int,  # 单个 DP rank 的世界大小
    local_world_size: int | None = None,  # 本地世界大小（可选）
):  # 函数参数列表结束
    """计算指定数据并行 rank 对应的设备索引字符串。

    根据 local_dp_rank 和 world_size 计算该 rank 应使用的物理设备索引，
    返回逗号分隔的设备索引字符串。

    例如：如果 world_size=2，local_dp_rank=1，共有 4 个设备，
    则返回设备 2 和 3 的索引。

    Args:
        device_control_env_var: 设备控制环境变量名
        local_dp_rank: 本地数据并行 rank
        world_size: 每个 DP rank 需要的设备数
        local_world_size: 本地世界大小，默认等于 world_size

    Returns:
        逗号分隔的物理设备索引字符串
    """
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    if local_world_size is None:  # 如果未指定本地世界大小
        local_world_size = world_size  # 使用 world_size 作为默认值
    try:  # 尝试计算设备索引
        value = ",".join(  # 将设备索引列表用逗号连接成字符串
            str(current_platform.device_id_to_physical_device_id(i))  # 将逻辑设备 ID 转换为物理设备 ID
            for i in range(  # 遍历该 rank 的逻辑设备 ID 范围
                local_dp_rank * world_size,  # 起始逻辑设备 ID
                local_dp_rank * world_size + local_world_size,  # 结束逻辑设备 ID（不含）
            )
        )
    except IndexError as e:  # 如果索引超出范围（设备不够）
        raise Exception(  # 抛出带详细信息的异常
            f"Error setting {device_control_env_var}: "  # 说明哪个环境变量出错
            f"local range: [{local_dp_rank * world_size}, "  # 显示请求的设备范围
            f"{(local_dp_rank + 1) * world_size}) "  # 范围结束值
            "base value: "  # 当前环境变量值
            f'"{os.getenv(device_control_env_var)}"'  # 显示当前环境变量的值
        ) from e  # 保留原始异常链
    return value  # 返回设备索引字符串


# [中文注释] Ray 模式的引擎 Actor 管理器。与 CoreEngineProcManager 的区别：
#   - 使用 Ray Actor 替代 multiprocessing.Process，支持本地+远程节点的引擎管理
#   - 通过 PlacementGroup 控制引擎的 GPU 分配策略（strict/fill/span）
#   - 支持 Elastic EP 动态扩缩容：scale_up/scale_down_elastic_ep
#   - 管理 local_engine_actors 和 remote_engine_actors 两个列表
class CoreEngineActorManager:  # Ray 模式的核心引擎 Actor 管理器类定义
    """Ray 模式的核心引擎 Actor 管理器类。

    负责创建、初始化和关闭由 AsyncLLM 和 LLMEngine 使用的 Ray Actor 引擎实例。
    与 CoreEngineProcManager 不同，此类管理本地节点和远程节点上的引擎 Actor，
    支持通过 PlacementGroup 进行 GPU 资源分配，以及 Elastic EP 动态扩缩容。
    """
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(  # 构造函数定义
        self,  # 实例自身引用
        vllm_config: VllmConfig,  # vLLM 总配置对象
        addresses: EngineZmqAddresses,  # ZMQ 地址配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计信息
        placement_groups: list["PlacementGroup"] | None = None,  # 预创建的放置组列表（可选）
        local_dp_ranks: list[int] | None = None,  # 本地 DP rank 列表（与 placement_groups 配合使用）
    ):  # 构造函数参数列表结束
        """初始化 Ray 引擎 Actor 管理器并创建所有引擎 Actor。

        Args:
            vllm_config: vLLM 配置对象
            addresses: ZMQ 地址配置
            executor_class: 执行器类
            log_stats: 是否启用统计日志
            placement_groups: 预创建的放置组（可选，不提供则自动创建）
            local_dp_ranks: 本地 DP rank 列表
        """
        import copy  # 导入深拷贝模块，用于为每个 DP rank 创建独立的配置副本

        import ray  # 导入 Ray 分布式计算框架
        from ray.runtime_env import RuntimeEnv  # 导入 Ray 运行时环境配置类
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # 导入放置组调度策略

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor  # 导入引擎核心 Actor 类（MoE 和普通版本）

        dp_size = vllm_config.parallel_config.data_parallel_size  # 获取数据并行大小
        actor_class = (  # 根据模型类型选择 Actor 类
            DPMoEEngineCoreActor  # MoE 模型且 DP > 1 时使用带波次协调的 Actor
            if dp_size > 1 and vllm_config.model_config.is_moe  # 判断条件：DP > 1 且为 MoE 模型
            else EngineCoreActor  # 否则使用标准 Actor
        )

        self.local_engine_actors: list[ray.ActorHandle] = []  # 本地引擎 Actor 句柄列表
        self.remote_engine_actors: list[ray.ActorHandle] = []  # 远程引擎 Actor 句柄列表

        env_vars_list = get_env_vars_to_copy(destination=actor_class.__name__)  # 获取需要复制到 Actor 的环境变量名列表
        self.env_vars_dict = {  # 构建环境变量字典
            name: os.environ[name] for name in env_vars_list if name in os.environ  # 仅包含当前环境中存在的变量
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)  # 创建 Ray 运行时环境配置

        self.addresses = addresses  # 保存 ZMQ 地址配置
        self.executor_class = executor_class  # 保存执行器类引用
        self.log_stats = log_stats  # 保存统计日志开关
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local  # 获取本地引擎数量
        world_size = vllm_config.parallel_config.world_size  # 获取每个 DP rank 的世界大小

        if ray.is_initialized():  # 检查 Ray 是否已初始化
            logger.info("Ray is already initialized. Skipping Ray initialization.")  # 记录日志：跳过初始化
        else:  # Ray 未初始化
            ray.init()  # 初始化 Ray 运行时

        vllm_config.parallel_config.allocate_elastic_ep_ports()  # 为 Elastic EP 分配通信端口

        if placement_groups is not None:  # 如果提供了预创建的放置组
            assert local_dp_ranks is not None, (  # 断言同时提供了 local_dp_ranks
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (  # 断言两个列表长度一致
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")  # 记录日志：使用提供的放置组
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []  # 使用外部放置组时，自己不创建任何放置组
        else:  # 未提供放置组，需要自动创建
            placement_groups, local_dp_ranks = (  # 自动创建数据并行放置组
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            )
            self.created_placement_groups = placement_groups  # 记录自己创建的放置组（用于后续清理）
        assert len(placement_groups) == dp_size, (  # 断言放置组数量等于 DP 大小
            "Number of placement groups must match data parallel size"
        )

        self.placement_group_is_local = []  # 记录每个放置组是否在本地节点
        refs = []  # 用于收集初始化完成的 Future 引用
        for index, local_index, pg in zip(  # 遍历每个 DP rank 及其放置组
            range(dp_size), local_dp_ranks, placement_groups  # 全局索引、本地索引、放置组
        ):
            dp_vllm_config = copy.deepcopy(vllm_config)  # 为每个 DP rank 深拷贝独立的配置
            dp_vllm_config.parallel_config.placement_group = pg  # 设置该 rank 的放置组
            local_client = index < local_engine_count  # 判断该 rank 是否为本地引擎

            if dp_size > 1 and dp_vllm_config.kv_transfer_config is not None:  # 如果是 DP 模式且启用了 KV 传输
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                dp_vllm_config.kv_transfer_config.engine_id = (  # 修改 engine_id 使其对每个 DP rank 唯一
                    f"{dp_vllm_config.kv_transfer_config.engine_id}_dp{local_index}"  # 追加本地 rank 索引
                )

            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            if current_platform.is_xpu():  # 如果当前平台是 Intel XPU
                device_evar = current_platform.device_control_env_var  # 获取设备控制环境变量名
                device_indices = get_device_indices(  # 计算该 rank 的设备索引
                    device_evar, local_index, world_size  # 传入环境变量名、本地索引和世界大小
                )
                actor_env_vars = self.env_vars_dict.copy()  # 复制环境变量字典
                actor_env_vars[device_evar] = device_indices  # 添加设备索引到环境变量
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)  # 创建包含设备索引的运行时环境

            actor = (  # 创建 Ray Actor 实例
                ray.remote(actor_class)  # 将 Actor 类注册为 Ray 远程类
                .options(  # 设置 Actor 选项
                    scheduling_strategy=PlacementGroupSchedulingStrategy(  # 使用放置组调度策略
                        placement_group=pg,  # 指定放置组
                        placement_group_bundle_index=world_size,  # 使用 CPU bundle（索引为 world_size，即最后一个 bundle）
                    ),
                    runtime_env=runtime_env,  # 设置运行时环境（环境变量等）
                )
                .remote(  # 远程实例化 Actor
                    vllm_config=dp_vllm_config,  # 传入该 rank 的配置
                    executor_class=executor_class,  # 执行器类
                    log_stats=log_stats,  # 统计日志开关
                    local_client=local_client,  # 是否为本地客户端
                    addresses=addresses,  # ZMQ 地址配置
                    dp_rank=index,  # 全局 DP rank
                    local_dp_rank=local_index,  # 本地 DP rank
                )
            )
            if local_client:  # 如果是本地引擎
                self.local_engine_actors.append(actor)  # 添加到本地 Actor 列表
            else:  # 远程引擎
                self.remote_engine_actors.append(actor)  # 添加到远程 Actor 列表
            self.placement_group_is_local.append(local_client)  # 记录放置组的本地/远程标记
            refs.append(actor.wait_for_init.remote())  # 异步调用 Actor 的初始化等待方法

        ray.get(refs)  # 阻塞等待所有 Actor 完成初始化
        self.run_refs = []  # 初始化运行 Future 引用列表
        for actor in self.local_engine_actors + self.remote_engine_actors:  # 遍历所有 Actor
            self.run_refs.append(actor.run.remote())  # 异步启动每个 Actor 的运行循环

    @staticmethod  # 静态方法，不依赖实例状态
    def create_dp_placement_groups(
        vllm_config: VllmConfig,  # vLLM 配置对象
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """为数据并行创建 Ray 放置组。

        根据集群资源和打包策略（strict/fill/span），为每个 DP rank 创建
        包含所需 GPU 资源的放置组。

        Args:
            vllm_config: vLLM 配置对象

        Returns:
            (放置组列表, 本地 DP rank 列表) 的元组
        """
        """
        Create placement groups for data parallel.
        """

        import ray  # 导入 Ray 框架
        from ray._private.state import available_resources_per_node  # 导入获取每节点可用资源的函数

        logger.info("Creating placement groups for data parallel")  # 记录日志：开始创建放置组
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip  # 获取 DP 主节点 IP
        dp_size = vllm_config.parallel_config.data_parallel_size  # 获取数据并行大小
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local  # 获取本地 DP 大小

        available_resources = available_resources_per_node()  # 获取集群中每个节点的可用资源
        world_size = vllm_config.parallel_config.world_size  # 获取每个 DP rank 的世界大小
        placement_groups: list[PlacementGroup] = []  # 初始化放置组列表
        local_dp_ranks: list[int] = []  # 初始化本地 DP rank 列表

        dp_master_ip_key = f"node:{dp_master_ip}"  # 构造主节点的资源键名
        nodes = sorted(  # 对节点排序，主节点排在最前面
            available_resources.values(), key=lambda x: dp_master_ip_key not in x  # 主节点的 key 在资源中时排序值为 False（0），排在前面
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."  # 断言集群中有可用节点
        assert dp_master_ip_key in nodes[0], (  # 断言第一个节点是主节点
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )
        device_str = current_platform.ray_device_key  # 获取当前平台在 Ray 中的设备资源键名（如 "GPU"）
        n_node_devices: list[int] = [  # 计算每个节点的设备数量列表
            int(node_resources[device_str])  # 转换为整数
            for node_resources in nodes  # 遍历所有节点
            if device_str in node_resources  # 仅包含有设备资源的节点
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."  # 断言集群中有设备可用
        max_device_per_node = max(n_node_devices)  # 获取单节点最大设备数

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY  # 获取 DP 放置组打包策略环境变量
        _supported_pack_strategies = ("strict", "fill", "span")  # 支持的打包策略列表
        if pack_strategy not in _supported_pack_strategies:  # 验证策略是否合法
            raise ValueError(  # 抛出不支持的策略错误
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        all2all_backend = vllm_config.parallel_config.all2all_backend  # 获取 all-to-all 通信后端
        if pack_strategy == "fill" and (  # 如果使用 fill 策略
            all2all_backend == "deepep_high_throughput"  # 且 all2all 后端是 DeepEP 高吞吐
            or all2all_backend == "deepep_low_latency"  # 或 DeepEP 低延迟
        ):
            raise ValueError(  # 抛出不兼容错误：DeepEP 需要同节点 EP rank
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):  # strict 和 fill 策略
            placement_strategy = "STRICT_PACK"  # 使用 Ray 的 STRICT_PACK 策略（所有 bundle 在同一节点）
        else:  # span 策略
            placement_strategy = "PACK"  # 使用 Ray 的 PACK 策略（尽量打包但可跨节点）
            assert world_size > max_device_per_node, (  # 断言世界大小超过单节点设备数（否则不需要 span）
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # if we need multiple nodes per dp group, we require for now that
            # available nodes are homogeneous
            assert set(n_node_devices) == {max_device_per_node}, (  # 断言所有节点的设备数相同（同构集群要求）
                f"Nodes are not homogeneous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (  # 断言世界大小能被单节点设备数整除
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (  # 断言总设备数足够
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (  # span 模式下本地 DP 大小应为默认值 1
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        collected_bundles = []  # span 策略下跨节点收集的 bundle 列表
        for node_resources in nodes:  # 遍历每个节点的资源信息
            node_ip_keys = [  # 提取节点 IP 资源键
                key  # 资源键名
                for key in node_resources  # 遍历该节点的所有资源键
                if key != "node:__internal_head__" and key.startswith("node:")  # 排除内部 head 标记，仅保留 node:IP 格式的键
            ]
            assert len(node_ip_keys) == 1, (  # 断言每个节点恰好有一个 IP 键
                f"Zero or multiple node IP keys found in node resources: {node_ip_keys}"
            )
            node_ip_key = node_ip_keys[0]  # 获取节点 IP 键
            node_ip = node_ip_key.split(":")[1]  # 从 "node:IP" 格式中提取 IP 地址

            n_device_on_node = int(node_resources.get(device_str, 0))  # 获取该节点上的设备数量
            if pack_strategy == "span" and n_device_on_node != 0:  # span 策略且节点有设备
                # Strictly speaking,
                # dp_size_available = n_device_on_node / world_size
                # and is a fraction, but we use 1 for easier processing
                dp_size_available = 1  # span 策略下每个节点贡献部分设备，简化为 1 便于处理
            else:  # strict 或 fill 策略
                dp_size_available = n_device_on_node // world_size  # 计算该节点可容纳的完整 DP rank 数量

            if node_ip == dp_master_ip:  # 如果是主节点
                if dp_size_available < dp_size_local:  # 如果主节点资源不够分配本地 DP rank
                    raise ValueError(  # 抛出资源不足错误
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local  # 主节点分配本地 DP 大小数量的 rank
            elif pack_strategy == "strict":  # strict 策略下的非主节点
                if dp_size_available < dp_size_local:  # 如果该节点资源不够
                    logger.info(  # 记录日志：跳过该节点
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip,  # 节点 IP
                        dp_size_local,  # 需要的 DP rank 数
                        dp_size_available,  # 可容纳的 DP rank 数
                    )
                    continue  # 跳过该节点
                dp_size_to_allocate = dp_size_local  # strict 策略下每个节点分配 dp_size_local 个 rank
            else:  # fill 和 span 策略
                # for "pack_strategy" in "fill" and "span"
                # we always take everything that's available
                dp_size_to_allocate = dp_size_available  # 使用所有可用资源

            for i in range(dp_size_to_allocate):  # 遍历该节点上需要分配的 DP rank
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]  # 创建单设备 bundle（包含节点约束）
                if pack_strategy == "span":  # span 策略：跨节点收集 bundle
                    collected_bundles += device_bundle * n_device_on_node  # 将该节点的所有设备 bundle 加入收集列表
                    assert len(collected_bundles) <= world_size, (  # 断言收集的 bundle 数不超过世界大小
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # we only create a placement group if we collected enough devices
                    if len(collected_bundles) < world_size:  # 如果还没收集够一个完整 DP rank 的设备
                        continue  # 继续收集

                    bundles = collected_bundles + [{"CPU": 1.0}]  # 加上一个 CPU bundle 作为调度 bundle
                    collected_bundles = []  # 重置收集列表，开始下一个 DP rank
                else:  # strict 或 fill 策略
                    bundles = device_bundle * world_size + [{"CPU": 1.0}]  # 创建 world_size 个设备 bundle + 1 个 CPU bundle

                pg = ray.util.placement_group(  # 创建 Ray 放置组
                    name=f"dp_rank_{len(placement_groups)}",  # 放置组名称（包含 rank 编号）
                    strategy=placement_strategy,  # 放置策略（STRICT_PACK 或 PACK）
                    bundles=bundles,  # 资源 bundle 列表
                )
                placement_groups.append(pg)  # 添加到放置组列表
                local_dp_ranks.append(i)  # 记录本地 DP rank 索引
                if len(placement_groups) == dp_size:  # 如果已创建足够的放置组
                    break  # 退出内层循环

        if len(placement_groups) < dp_size:  # 如果创建的放置组数量不够
            raise ValueError(  # 抛出资源不足错误
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (  # 断言放置组数量正确
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (  # 断言本地 rank 列表长度正确
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks  # 返回放置组列表和本地 rank 列表

    @staticmethod  # 静态方法
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int  # 旧配置和新的 DP 大小
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """为 Elastic EP 扩容添加新的放置组。

        计算需要新增的放置组数量，扫描集群节点的可用资源，
        为新的 DP rank 创建放置组。

        Args:
            old_vllm_config: 当前的 vLLM 配置
            new_data_parallel_size: 扩容后的数据并行大小

        Returns:
            (新放置组列表, 新本地 DP rank 列表) 的元组
        """
        """
        Add placement groups for new data parallel size.
        """
        import ray  # 导入 Ray 框架
        from ray._private.state import (  # 从 Ray 内部状态模块导入
            available_resources_per_node,  # 获取每节点可用资源
            total_resources_per_node,  # 获取每节点总资源
        )
        from ray.util.state import list_nodes  # 导入列出集群节点的函数

        old_dp_size = old_vllm_config.parallel_config.data_parallel_size  # 获取旧的 DP 大小
        num_pg_to_create = new_data_parallel_size - old_dp_size  # 计算需要新创建的放置组数量

        if num_pg_to_create <= 0:  # 如果不需要创建新的放置组
            return [], []  # 返回空列表

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip  # 获取 DP 主节点 IP
        world_size = old_vllm_config.parallel_config.world_size  # 获取世界大小

        nodes = list_nodes()  # 列出集群中的所有节点
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)  # 排序：主节点优先
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"  # 断言第一个节点是主节点
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (  # 断言只有一个主节点
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()  # 获取每节点可用资源
        total_resources = total_resources_per_node()  # 获取每节点总资源

        placement_groups = []  # 新放置组列表
        local_dp_ranks = []  # 新本地 DP rank 列表
        num_pg_created = 0  # 已创建的放置组计数器

        device_str = current_platform.ray_device_key  # 获取设备资源键名
        for node in nodes:  # 遍历集群节点
            if num_pg_created >= num_pg_to_create:  # 如果已创建足够的放置组
                break  # 退出循环

            node_ip = node.node_ip  # 获取节点 IP
            node_id = node.node_id  # 获取节点 ID
            if device_str not in available_resources[node_id]:  # 如果该节点没有可用设备
                continue  # 跳过该节点
            available_gpus = int(available_resources[node_id][device_str])  # 获取可用 GPU 数量

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            total_gpus = int(total_resources[node_id][device_str])  # 获取该节点的 GPU 总数

            # Calculate used GPUs and used engines on this node
            used_gpus = max(0, total_gpus - available_gpus)  # 计算已使用的 GPU 数量
            used_engines_on_node = used_gpus // world_size  # 计算该节点上已运行的引擎数量

            # Calculate how many new engines this node can accommodate
            available_engine_count = available_gpus // world_size  # 计算该节点还能容纳的新引擎数量

            # Create placement groups for new engines on this node
            for i in range(available_engine_count):  # 遍历可新增的引擎
                if num_pg_created >= num_pg_to_create:  # 如果已创建足够的放置组
                    break  # 退出循环

                rank = old_dp_size + num_pg_created  # 计算新 DP rank 编号

                # Create bundles with node constraint for master node
                if node_ip == dp_master_ip:  # 如果是主节点
                    bundles = [  # 创建带节点约束的 bundle
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}  # GPU 资源 + 节点约束
                    ] * world_size + [{"CPU": 1.0}]  # 加上 CPU 调度 bundle
                else:  # 非主节点
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]  # 创建不带节点约束的 bundle

                pg = ray.util.placement_group(  # 创建放置组
                    name=f"dp_rank_{rank}",  # 放置组名称
                    strategy="STRICT_PACK",  # 使用严格打包策略
                    bundles=bundles,  # 资源 bundle
                )
                placement_groups.append(pg)  # 添加到列表

                # Local rank starts from the number of engines already used
                # on this node
                local_rank = used_engines_on_node + i  # 计算本地 rank（基于该节点已有引擎数）
                local_dp_ranks.append(local_rank)  # 记录本地 rank
                num_pg_created += 1  # 增加已创建计数

        return placement_groups, local_dp_ranks  # 返回新放置组和本地 rank 列表

    def scale_up_elastic_ep(
        self, cur_vllm_config: VllmConfig, new_data_parallel_size: int  # 当前配置和目标 DP 大小
    ) -> None:
        """Elastic EP 扩容：增加数据并行 rank 数量。

        创建新的放置组和 Ray Actor，将新引擎添加到管理列表中，
        并更新配置中的数据并行大小。

        Args:
            cur_vllm_config: 当前 vLLM 配置（会被就地修改）
            new_data_parallel_size: 扩容后的目标 DP 大小
        """
        import copy  # 导入深拷贝模块

        import ray  # 导入 Ray 框架
        from ray.runtime_env import RuntimeEnv  # 导入运行时环境配置
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # 导入放置组调度策略

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor  # 导入引擎 Actor 类

        actor_class = (  # 选择 Actor 类
            DPMoEEngineCoreActor  # MoE 模型使用带波次协调的 Actor
            if cur_vllm_config.model_config.is_moe  # 判断是否为 MoE 模型
            else EngineCoreActor  # 普通模型使用标准 Actor
        )

        cur_data_parallel_size = len(self.local_engine_actors) + len(  # 计算当前 DP 大小
            self.remote_engine_actors  # 本地 + 远程 Actor 总数
        )

        assert new_data_parallel_size > cur_data_parallel_size, (  # 断言新大小大于当前大小
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        placement_groups, local_dp_ranks = self.add_dp_placement_groups(  # 为新 rank 创建放置组
            cur_vllm_config, new_data_parallel_size  # 传入当前配置和目标大小
        )

        world_size = cur_vllm_config.parallel_config.world_size  # 获取世界大小
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip  # 获取主节点 IP
        new_local_engines = 0  # 新增本地引擎计数器

        runtime_env = RuntimeEnv(  # 创建带扩容标记的运行时环境
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}  # 添加扩容启动标记环境变量
        )
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):  # 遍历新放置组和本地 rank
            rank = cur_data_parallel_size + i  # 计算新 Actor 的全局 rank
            dp_vllm_config = copy.deepcopy(cur_vllm_config)  # 深拷贝配置
            dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size  # 更新 DP 大小
            dp_vllm_config.parallel_config.placement_group = pg  # 设置放置组

            # Check if this placement group is on the head node
            local_client = any(  # 检查放置组是否在主节点上
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs  # 检查 bundle 中是否有主节点约束
            )

            if local_client:  # 如果是本地引擎
                new_local_engines += 1  # 增加本地引擎计数
                # Update data_parallel_size_local
                dp_vllm_config.parallel_config.data_parallel_size_local = (  # 更新本地 DP 大小
                    cur_vllm_config.parallel_config.data_parallel_size_local  # 原始本地 DP 大小
                    + new_local_engines  # 加上新增的本地引擎数
                )

            actor = (  # 创建新的 Ray Actor
                ray.remote(actor_class)  # 注册为远程类
                .options(  # 设置选项
                    scheduling_strategy=PlacementGroupSchedulingStrategy(  # 放置组调度策略
                        placement_group=pg,  # 指定放置组
                        placement_group_bundle_index=world_size,  # CPU bundle 索引
                    ),
                    runtime_env=runtime_env,  # 运行时环境（包含扩容标记）
                )
                .remote(  # 远程实例化
                    vllm_config=dp_vllm_config,  # 配置
                    executor_class=self.executor_class,  # 执行器类
                    log_stats=self.log_stats,  # 统计日志开关
                    local_client=local_client,  # 是否本地
                    addresses=self.addresses,  # ZMQ 地址
                    dp_rank=rank,  # 全局 rank
                    local_dp_rank=local_rank,  # 本地 rank
                )
            )

            if local_client:  # 本地引擎
                self.local_engine_actors.append(actor)  # 添加到本地列表
            else:  # 远程引擎
                self.remote_engine_actors.append(actor)  # 添加到远程列表
            self.created_placement_groups.append(pg)  # 记录创建的放置组
            self.placement_group_is_local.append(local_client)  # 记录本地/远程标记

        ray.get(  # 阻塞等待所有新 Actor 完成初始化
            [
                actor.wait_for_init.remote()  # 调用初始化等待方法
                for actor in (  # 遍历新增的 Actor
                    self.local_engine_actors[-new_local_engines:]  # 新增的本地 Actor
                    if new_local_engines > 0  # 如果有新增本地 Actor
                    else []  # 否则为空列表
                )
                + self.remote_engine_actors[  # 加上新增的远程 Actor
                    -(len(placement_groups) - new_local_engines) :  # 取最后 N 个远程 Actor
                ]
            ]
        )

        actors = (  # 收集所有新增的 Actor
            self.local_engine_actors[-new_local_engines:]  # 新增本地 Actor
            if new_local_engines > 0  # 有新增本地 Actor
            else []  # 否则为空
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines) :]  # 加上新增远程 Actor

        for actor in actors:  # 遍历新增 Actor
            self.run_refs.append(actor.run.remote())  # 启动运行循环

        cur_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size  # 更新配置中的 DP 大小
        # Update old_vllm_config with new data_parallel_size_local if any new
        # local engines were added
        if new_local_engines > 0:  # 如果有新增本地引擎
            cur_vllm_config.parallel_config.data_parallel_size_local += (  # 更新本地 DP 大小
                new_local_engines  # 增加新本地引擎数
            )

    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int  # 当前和目标 DP 大小
    ) -> None:
        """Elastic EP 缩容：减少数据并行 rank 数量。

        从后往前移除多余的引擎 Actor 和对应的放置组。

        Args:
            cur_data_parallel_size: 当前数据并行大小
            new_data_parallel_size: 缩容后的目标 DP 大小
        """
        import ray  # 导入 Ray 框架

        assert cur_data_parallel_size > new_data_parallel_size, (  # 断言当前大小大于目标大小
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        for _ in range(cur_data_parallel_size - new_data_parallel_size):  # 遍历需要移除的 rank 数量
            pg = self.created_placement_groups.pop()  # 弹出最后一个放置组
            is_local = self.placement_group_is_local.pop()  # 弹出对应的本地/远程标记
            if is_local:  # 如果是本地引擎
                self.local_engine_actors.pop()  # 从本地列表中移除
            else:  # 远程引擎
                self.remote_engine_actors.pop()  # 从远程列表中移除
            ray.util.remove_placement_group(pg)  # 释放 Ray 放置组资源

    def get_run_refs(self):
        """获取所有引擎 Actor 的运行 Future 引用列表。

        Returns:
            Ray ObjectRef 列表，可用于 ray.get() 等待 Actor 运行完成
        """
        return self.run_refs  # 返回运行引用列表

    def shutdown(self, timeout: float | None = None) -> None:
        """关闭所有引擎 Actor 并释放放置组资源。

        Args:
            timeout: 未使用，保留用于接口兼容
        """
        import ray  # 导入 Ray 框架

        for actor in self.local_engine_actors + self.remote_engine_actors:  # 遍历所有 Actor
            ray.kill(actor)  # 强制终止 Actor
        for pg in self.created_placement_groups:  # 遍历所有创建的放置组
            ray.util.remove_placement_group(pg)  # 释放放置组资源


# [中文注释] 为 Engine-Client 通信分配 ZMQ 地址。
#   根据是否为本地引擎、offline 模式、elastic EP 等决定使用 IPC 还是 TCP 地址
def get_engine_zmq_addresses(
    vllm_config: VllmConfig,  # vLLM 配置对象
    num_api_servers: int = 1,  # API 服务器数量
) -> EngineZmqAddresses:
    """为引擎与客户端通信分配 ZMQ 地址。

    根据部署模式（离线/在线、本地/分布式、Elastic EP）决定使用
    IPC 还是 TCP 地址，并为每个 API 服务器创建独立的输入输出地址。

    Args:
        vllm_config: vLLM 配置对象
        num_api_servers: API 服务器数量

    Returns:
        包含所有所需 ZMQ 地址的 EngineZmqAddresses 对象
    """
    """Allocate ZMQ addresses for engine-client communication."""
    parallel_config = vllm_config.parallel_config  # 获取并行配置
    local_engine_count = parallel_config.data_parallel_size_local  # 本地引擎数量
    local_start_index = parallel_config.data_parallel_rank_local  # 本地起始 DP rank
    dp_size = parallel_config.data_parallel_size  # 数据并行大小
    host = parallel_config.data_parallel_master_ip  # 主节点 IP 地址
    local_engines_only = parallel_config.local_engines_only  # 是否仅使用本地引擎

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/offline_inference/data_parallel.py.
    offline_mode = local_start_index is not None  # 判断是否为离线模式（离线模式下每个 DP rank 有独立的 LLM 实例）

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = (  # 判断客户端是否仅与本地引擎通信
        offline_mode or local_engines_only or (local_engine_count == dp_size)  # 离线模式、仅本地引擎、或所有引擎都在本地
    )
    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:  # 如果启用了 Elastic EP
        client_local_only = False  # 强制使用 TCP 地址以支持跨节点通信

    return EngineZmqAddresses(  # 创建并返回 ZMQ 地址配置对象
        inputs=[  # 为每个 API 服务器生成输入地址
            get_engine_client_zmq_addr(client_local_only, host)  # 根据本地/远程模式生成 IPC 或 TCP 地址
            for _ in range(num_api_servers)  # 遍历 API 服务器数量
        ],
        outputs=[  # 为每个 API 服务器生成输出地址
            get_engine_client_zmq_addr(client_local_only, host)  # 根据本地/远程模式生成 IPC 或 TCP 地址
            for _ in range(num_api_servers)  # 遍历 API 服务器数量
        ],
    )


# [中文注释] 启动引擎进程和 DP Coordinator 的上下文管理器。完整启动流程：
#   1. 若需要 Coordinator（DP>1 且 rank=0），创建 DPCoordinator 子进程
#   2. Ray 模式 → 创建 CoreEngineActorManager
#      多进程模式 → 创建 CoreEngineProcManager 启动本地引擎子进程
#   3. yield 返回 (engine_manager, coordinator, addresses) 供调用者使用
#   4. 等待所有引擎完成握手（HELLO → READY 两阶段协议）
@contextlib.contextmanager  # 将函数转换为上下文管理器
def launch_core_engines(
    vllm_config: VllmConfig,  # vLLM 配置对象
    executor_class: type[Executor],  # 执行器类
    log_stats: bool,  # 是否记录统计信息
    addresses: EngineZmqAddresses,  # ZMQ 地址配置
    num_api_servers: int = 1,  # API 服务器数量
) -> Iterator[  # 返回迭代器类型（上下文管理器 yield 的值）
    tuple[  # 元组类型
        CoreEngineProcManager | CoreEngineActorManager | None,  # 引擎管理器（进程或 Actor 或无）
        DPCoordinator | None,  # 数据并行协调器（可选）
        EngineZmqAddresses,  # ZMQ 地址配置
    ]
]:
    """启动引擎核心进程和数据并行协调器的上下文管理器。

    完整启动流程：
    1. 如果需要，启动 DP Coordinator 进程
    2. Ray 模式下创建 CoreEngineActorManager；多进程模式下创建 CoreEngineProcManager
    3. yield 返回管理器和地址供调用者使用
    4. yield 返回后等待所有引擎完成 HELLO → READY 两阶段握手

    Args:
        vllm_config: vLLM 配置对象
        executor_class: 执行器类
        log_stats: 是否启用统计日志
        addresses: ZMQ 地址配置
        num_api_servers: API 服务器数量

    Yields:
        (引擎管理器, 协调器, ZMQ 地址) 三元组
    """
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config  # 获取并行配置
    dp_size = parallel_config.data_parallel_size  # 数据并行大小
    local_engine_count = parallel_config.data_parallel_size_local  # 本地引擎数量
    local_start_index = parallel_config.data_parallel_rank_local  # 本地起始 DP rank
    dp_rank = parallel_config.data_parallel_rank  # 当前 DP rank
    host = parallel_config.data_parallel_master_ip  # 主节点 IP
    local_engines_only = parallel_config.local_engines_only  # 是否仅本地引擎模式

    offline_mode = local_start_index is not None  # 判断是否为离线模式

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (  # 判断是否需要运行协调器
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0  # 需要协调器、非离线模式、且为 rank 0
    )

    if run_coordinator:  # 如果需要运行协调器
        coordinator = DPCoordinator(  # 创建数据并行协调器
            parallel_config,  # 传入并行配置
            enable_wave_coordination=vllm_config.model_config.is_moe,  # MoE 模型启用波次协调
        )

        addresses.coordinator_input, addresses.coordinator_output = (  # 获取协调器的输入输出地址
            coordinator.get_engine_socket_addresses()  # 从协调器获取 ZMQ socket 地址
        )
        addresses.frontend_stats_publish_address = (  # 获取前端统计发布地址
            coordinator.get_stats_publish_address()  # 从协调器获取统计发布地址
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)  # 记录协调器进程 PID
    else:  # 不需要协调器
        coordinator = None  # 设置为 None

    if parallel_config.data_parallel_backend == "ray":  # 如果使用 Ray 后端
        logger.info("Starting ray-based data parallel backend")  # 记录日志：启动 Ray DP 后端

        engine_actor_manager = CoreEngineActorManager(  # 创建 Ray Actor 管理器
            vllm_config=vllm_config,  # 配置
            addresses=addresses,  # ZMQ 地址
            executor_class=executor_class,  # 执行器类
            log_stats=log_stats,  # 统计日志开关
        )

        yield engine_actor_manager, coordinator, addresses  # 返回管理器、协调器和地址
        return  # Ray 模式下直接返回（Actor 在创建时已完成握手）

    if offline_mode:  # 离线模式
        assert local_engine_count == 1  # 断言离线模式下只有一个本地引擎
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]  # 创建单个本地引擎的握手对象
    elif dp_rank == 0:  # 在线模式且 rank 为 0
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [  # rank 0 需要与所有引擎握手
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)  # 为每个 DP rank 创建引擎对象
        ]
    else:  # 在线模式且 rank > 0
        # Rank > 0 handshakes with just the local cores it is managing.
        assert local_engines_only, (  # 断言 rank > 0 只在 local_engines_only 模式下运行
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [  # 仅与本地管理的引擎握手
            CoreEngine(index=i, local=True)  # 创建本地引擎对象
            for i in range(dp_rank, dp_rank + local_engine_count)  # 遍历该 rank 管理的引擎范围
        ]

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size  # 判断握手是否仅限本地

    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:  # 如果启用 Elastic EP
        handshake_local_only = False  # 强制使用远程握手以支持跨节点

    handshake_address = get_engine_client_zmq_addr(  # 获取握手 ZMQ 地址
        handshake_local_only, host, parallel_config.data_parallel_rpc_port  # 根据模式选择 IPC 或 TCP
    )

    if local_engines_only and dp_rank > 0:  # 如果是仅本地引擎模式且 rank > 0
        assert not handshake_local_only  # 断言不是仅本地握手（因为需要连接 rank 0）
        local_handshake_address = get_open_zmq_ipc_path()  # 获取本地 IPC 路径
        client_handshake_address = local_handshake_address  # 客户端使用本地 IPC 地址
    else:  # 其他情况
        local_handshake_address = handshake_address  # 使用标准握手地址
        client_handshake_address = None  # 不需要单独的客户端握手地址

    with zmq_socket_ctx(  # 创建 ZMQ ROUTER socket 上下文
        local_handshake_address, zmq.ROUTER, bind=True  # 绑定到握手地址，使用 ROUTER 类型
    ) as handshake_socket:  # 获取 socket 对象
        # Start local engines.
        if local_engine_count:  # 如果有本地引擎需要启动
            local_engine_manager = CoreEngineProcManager(  # 创建本地引擎进程管理器
                vllm_config=vllm_config,  # 配置
                executor_class=executor_class,  # 执行器类
                log_stats=log_stats,  # 统计日志开关
                handshake_address=handshake_address,  # 握手地址
                client_handshake_address=client_handshake_address,  # 客户端握手地址
                local_client=True,  # 本地客户端模式
                local_engine_count=local_engine_count,  # 本地引擎数量
                start_index=dp_rank,  # 全局起始 rank
                local_start_index=local_start_index or 0,  # 本地起始 rank
            )
        else:  # 没有本地引擎
            local_engine_manager = None  # 管理器设为 None

        yield local_engine_manager, coordinator, addresses  # 返回管理器、协调器和地址供调用者使用

        # Now wait for engines to start.
        wait_for_engine_startup(  # 等待所有引擎完成启动握手
            handshake_socket,  # 握手 socket
            addresses,  # ZMQ 地址配置
            engines_to_handshake,  # 需要握手的引擎列表
            parallel_config,  # 并行配置
            dp_size > 1 and vllm_config.model_config.is_moe,  # 是否为协调式 DP（MoE 模型且 DP > 1）
            vllm_config.cache_config,  # 缓存配置
            local_engine_manager,  # 本地引擎管理器
            coordinator.proc if coordinator else None,  # 协调器进程（用于监听异常退出）
        )


# [中文注释] 等待所有引擎进程完成启动握手的两阶段协议：
#   阶段1 (HELLO → CONNECTED): 引擎发送 HELLO，Coordinator 回复 EngineHandshakeMetadata（含地址配置）
#   阶段2 (READY → READY): 引擎完成初始化后发送 READY（含 num_gpu_blocks 等信息）
#   同时监听进程 sentinel，若引擎进程异常退出则立即报错
def wait_for_engine_startup(
    handshake_socket: zmq.Socket,  # 握手 ZMQ ROUTER socket
    addresses: EngineZmqAddresses,  # ZMQ 地址配置
    core_engines: list[CoreEngine],  # 需要握手的引擎列表
    parallel_config: ParallelConfig,  # 并行配置
    coordinated_dp: bool,  # 是否为协调式数据并行（MoE）
    cache_config: CacheConfig,  # 缓存配置（用于收集 GPU 块数）
    proc_manager: CoreEngineProcManager | None,  # 本地进程管理器（可选）
    coord_process: Process | None,  # 协调器进程（可选）
):
    """等待所有引擎进程完成启动握手。

    实现 HELLO → CONNECTED → READY 两阶段握手协议：
    1. 引擎发送 HELLO，前端回复 EngineHandshakeMetadata（含 ZMQ 地址和并行配置）
    2. 引擎完成初始化后发送 READY（含 num_gpu_blocks 等运行时信息）

    同时通过 ZMQ Poller 监听子进程 sentinel，若有进程异常退出则立即报错。

    Args:
        handshake_socket: 用于握手通信的 ZMQ ROUTER socket
        addresses: ZMQ 地址配置
        core_engines: 需要完成握手的引擎列表
        parallel_config: 并行配置
        coordinated_dp: 是否需要波次协调（MoE 模型）
        cache_config: 缓存配置，用于累加所有引擎的 GPU 块数
        proc_manager: 本地引擎进程管理器
        coord_process: 协调器进程
    """
    # Wait for engine core process(es) to send ready messages.
    local_count = parallel_config.data_parallel_size_local  # 本地引擎数量
    remote_count = len(core_engines) - local_count  # 远程引擎数量
    # [local, remote] counts
    conn_pending, start_pending = [local_count, remote_count], [0, 0]  # 等待连接和启动的引擎计数 [本地, 远程]
    poller = zmq.Poller()  # 创建 ZMQ 轮询器
    poller.register(handshake_socket, zmq.POLLIN)  # 注册握手 socket 到轮询器（监听可读事件）

    remote_should_be_headless = (  # 判断远程引擎是否应为无头模式
        not parallel_config.data_parallel_hybrid_lb  # 非混合负载均衡模式
        and not parallel_config.data_parallel_external_lb  # 且非外部负载均衡模式
    )

    if proc_manager is not None:  # 如果有本地进程管理器
        for sentinel in proc_manager.sentinels():  # 遍历所有子进程的 sentinel
            poller.register(sentinel, zmq.POLLIN)  # 注册到轮询器（进程退出时变为可读）
    if coord_process is not None:  # 如果有协调器进程
        poller.register(coord_process.sentinel, zmq.POLLIN)  # 注册协调器进程的 sentinel
    while any(conn_pending) or any(start_pending):  # 当还有引擎未完成连接或启动时循环
        events = poller.poll(STARTUP_POLL_PERIOD_MS)  # 轮询等待事件（超时 10 秒）
        if not events:  # 如果超时无事件
            if any(conn_pending):  # 如果有引擎等待连接
                logger.debug(  # 记录调试日志
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,  # 本地和远程等待数
                )
            if any(start_pending):  # 如果有引擎等待启动
                logger.debug(  # 记录调试日志
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,  # 本地和远程等待数
                )
            continue  # 继续轮询
        if len(events) > 1 or events[0][0] != handshake_socket:  # 如果有多个事件或事件不是来自握手 socket
            # One of the local core processes exited.
            finished = proc_manager.finished_procs() if proc_manager else {}  # 获取已退出的进程信息
            if coord_process is not None and coord_process.exitcode is not None:  # 如果协调器进程已退出
                finished[coord_process.name] = coord_process.exitcode  # 记录协调器的退出码
            raise RuntimeError(  # 抛出运行时错误：引擎初始化失败
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"  # 显示失败的进程及其退出码
            )

        # Receive HELLO and READY messages from the input socket.
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()  # 接收多部分消息：[引擎身份, 消息内容]
        eng_index = int.from_bytes(eng_identity, "little")  # 从身份字节解析引擎索引
        engine = next((e for e in core_engines if e.identity == eng_identity), None)  # 根据身份查找对应的引擎对象
        if engine is None:  # 如果找不到对应的引擎
            raise RuntimeError(  # 抛出错误：未知的引擎 rank
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        msg = msgspec.msgpack.decode(ready_msg_bytes)  # 反序列化 msgpack 消息
        status, local, headless = msg["status"], msg["local"], msg["headless"]  # 提取状态、本地标志和无头标志
        if local != engine.local:  # 如果本地标志不匹配
            raise RuntimeError(  # 抛出错误：本地/远程不一致
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # Remote engines must be headless iff we aren't in hybrid dp lb mode.
        if not local and headless != remote_should_be_headless:  # 如果远程引擎的无头状态不正确
            if headless:  # 如果引擎是无头的但不应该是
                raise RuntimeError(  # 抛出错误
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:  # 如果引擎不是无头的但应该是
                raise RuntimeError(  # 抛出错误
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        if status == "HELLO" and engine.state == CoreEngineState.NEW:  # 如果收到 HELLO 消息且引擎在 NEW 状态
            # Send init message with DP config info.
            init_message = msgspec.msgpack.encode(  # 编码握手元数据为 msgpack 格式
                EngineHandshakeMetadata(  # 创建握手元数据对象
                    addresses=addresses,  # ZMQ 地址配置
                    parallel_config={  # 并行配置字典
                        k: getattr(parallel_config, k)  # 从并行配置中提取指定属性
                        for k in (  # 需要传递的配置键列表
                            "data_parallel_master_ip",  # DP 主节点 IP
                            "data_parallel_master_port",  # DP 主节点端口
                            "_data_parallel_master_port_list",  # DP 主节点端口列表
                            "data_parallel_size",  # DP 大小
                        )
                    }
                    if coordinated_dp  # 仅在协调式 DP 模式下传递这些配置
                    else {},  # 非协调模式传空字典
                )
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)  # 发送握手响应给引擎（零拷贝）
            conn_pending[0 if local else 1] -= 1  # 减少等待连接的计数（本地或远程）
            start_pending[0 if local else 1] += 1  # 增加等待启动的计数
            engine.state = CoreEngineState.CONNECTED  # 更新引擎状态为 CONNECTED
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:  # 如果收到 READY 消息且引擎在 CONNECTED 状态
            # Setup KV cache config with initialization state from
            # engine core process. Sum values from all engines in DP case.
            num_gpu_blocks = cache_config.num_gpu_blocks or 0  # 获取当前 GPU 块数（或默认为 0）
            num_gpu_blocks += msg["num_gpu_blocks"]  # 累加该引擎报告的 GPU 块数
            cache_config.num_gpu_blocks = num_gpu_blocks  # 更新缓存配置中的 GPU 块数

            # In external DP LB mode, the coordinator address that the
            # front-end procs connect to is obtained from rank 0 via
            # one of the engine handshakes, and passed to the local
            # front-end process in the response from the other.
            if addresses.frontend_stats_publish_address is None:  # 如果前端统计发布地址尚未设置
                addresses.frontend_stats_publish_address = msg.get("dp_stats_address")  # 从引擎消息中获取

            # Validate config hash consistency across DP workers for MoE models.
            if coordinated_dp:  # 如果是协调式 DP（MoE 模型）
                worker_config_hash = msg.get("parallel_config_hash")  # 获取引擎报告的配置哈希
                expected_hash = parallel_config.compute_hash()  # 计算期望的配置哈希
                if worker_config_hash != expected_hash:  # 如果哈希不匹配
                    raise RuntimeError(  # 抛出配置不一致错误
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            start_pending[0 if local else 1] -= 1  # 减少等待启动的计数
            engine.state = CoreEngineState.READY  # 更新引擎状态为 READY
        else:  # 意外的状态转换
            raise RuntimeError(  # 抛出错误
                f"Unexpected {status} message for "  # 意外的消息类型
                f"{'local' if local else 'remote'} engine "  # 本地/远程标识
                f"{eng_index} in {engine.state} state."  # 当前状态
            )

        logger.debug(  # 记录调试日志
            "%s from %s core engine process %s.",  # 消息格式
            status,  # 消息状态（HELLO/READY）
            "local" if local else "remote",  # 本地/远程
            eng_index,  # 引擎索引
        )
