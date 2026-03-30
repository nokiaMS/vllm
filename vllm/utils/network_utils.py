# SPDX-License-Identifier: Apache-2.0  # 许可证标识：Apache-2.0 开源许可
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者
import contextlib  # 导入上下文管理器工具模块
import ipaddress  # 导入IP地址解析模块
import os  # 导入操作系统接口模块
import socket  # 导入套接字网络通信模块
import sys  # 导入系统相关参数和函数模块
import warnings  # 导入警告信息模块
from collections.abc import (  # 从集合抽象基类导入类型
    Iterator,  # 导入迭代器类型
    Sequence,  # 导入序列类型
)
from typing import Any  # 导入任意类型注解
from uuid import uuid4  # 导入UUID4随机唯一标识符生成函数

import psutil  # 导入系统进程和资源监控模块
import zmq  # 导入ZMQ消息队列库
import zmq.asyncio  # 导入ZMQ异步IO支持模块
from urllib3.util import parse_url  # 从urllib3工具模块导入URL解析函数

import vllm.envs as envs  # 导入vLLM环境变量配置模块
from vllm.logger import init_logger  # 从vLLM日志模块导入日志初始化函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# 批量关闭 ZMQ 套接字，linger=0 表示立即丢弃未发送的消息
# | 在python中表示或。Sequence表示有序、可遍历、可按序号访问的集合类型。
# 在python中，list,tuple都是Sequence。此处的Sequence是python的类型提示，不是运行时功能。
# 在实际使用此函数的时候，此处即能够传递list，也能够传递tuple，因为这两者都是Sequence.
def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    """批量关闭ZMQ套接字列表"""
    for sock in sockets:  # 遍历所有套接字
        if sock is not None:  # 如果套接字不为空
            sock.close(linger=0)  # 关闭套接字，linger=0表示不等待未发送的消息


# 获取本机 IP 地址：优先使用 VLLM_HOST_IP 环境变量，否则通过 UDP 连接探测本机出口 IP（先 IPv4 后 IPv6）
# -> str表示函数执行完毕之后会返回一个字符串。
def get_ip() -> str:
    """获取本机IP地址，优先从环境变量读取，否则自动探测"""
    host_ip = envs.VLLM_HOST_IP  # 从环境变量获取主机IP配置
    if "HOST_IP" in os.environ and "VLLM_HOST_IP" not in os.environ:  # 如果使用了已弃用的HOST_IP环境变量
        logger.warning(  # 输出警告日志
            "The environment variable HOST_IP is deprecated and ignored, as"  # 提示HOST_IP已弃用
            " it is often used by Docker and other software to"  # 因为Docker等软件也常用此变量
            " interact with the container's network stack. Please "  # 与容器网络栈交互
            "use VLLM_HOST_IP instead to set the IP address for vLLM processes"  # 请改用VLLM_HOST_IP
            " to communicate with each other."  # 用于vLLM进程间通信
        )
    if host_ip:  # 如果环境变量中设置了IP
        return host_ip  # 直接返回环境变量中的IP

    # IP is not set, try to get it from the network interface  # IP未设置，尝试从网络接口获取

    # try ipv4  # 尝试IPv4
    try:  # 尝试创建IPv4 UDP套接字
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # 创建IPv4 UDP套接字
            s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable  # 连接到Google DNS（无需实际可达，仅用于确定出口IP）
            return s.getsockname()[0]  # 返回本机出口IPv4地址
    except Exception:  # 如果IPv4获取失败
        pass  # 跳过，继续尝试IPv6

    # try ipv6  # 尝试IPv6
    try:  # 尝试创建IPv6 UDP套接字
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:  # 创建IPv6 UDP套接字
            # Google's public DNS server, see  # Google公共DNS服务器，参见
            # https://developers.google.com/speed/public-dns/docs/using#addresses  # Google公共DNS文档地址
            s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable  # 连接到Google IPv6 DNS（无需实际可达）
            return s.getsockname()[0]  # 返回本机出口IPv6地址
    except Exception:  # 如果IPv6也获取失败
        pass  # 跳过

    warnings.warn(  # 发出警告
        "Failed to get the IP address, using 0.0.0.0 by default."  # 获取IP失败，使用默认值0.0.0.0
        "The value can be set by the environment variable"  # 可以通过环境变量设置
        " VLLM_HOST_IP or HOST_IP.",  # VLLM_HOST_IP 或 HOST_IP
        stacklevel=2,  # 设置警告调用栈层级为2
    )
    return "0.0.0.0"  # 返回默认的全零IP地址


# 测试是否能绑定到指定的回环地址（用于检测系统支持 IPv4 还是 IPv6 回环）
def test_loopback_bind(address: str, family: int) -> bool:
    """测试是否能绑定到指定的回环地址"""
    try:  # 尝试绑定操作
        s = socket.socket(family, socket.SOCK_DGRAM)  # 创建指定地址族的UDP套接字
        s.bind((address, 0))  # Port 0 = auto assign  # 绑定到指定地址，端口0表示自动分配
        s.close()  # 关闭套接字
        return True  # 绑定成功，返回True
    except OSError:  # 如果绑定失败（操作系统错误）
        return False  # 返回False


# 获取回环地址：优先使用 VLLM_LOOPBACK_IP 环境变量，否则自动检测 127.0.0.1 或 ::1
def get_loopback_ip() -> str:
    """获取本机回环地址"""
    loopback_ip = envs.VLLM_LOOPBACK_IP  # 从环境变量获取回环IP配置
    if loopback_ip:  # 如果环境变量中设置了回环IP
        return loopback_ip  # 直接返回

    # VLLM_LOOPBACK_IP is not set, try to get it based on network interface  # 环境变量未设置，根据网络接口检测

    if test_loopback_bind("127.0.0.1", socket.AF_INET):  # 测试IPv4回环地址是否可用
        return "127.0.0.1"  # 返回IPv4回环地址
    elif test_loopback_bind("::1", socket.AF_INET6):  # 测试IPv6回环地址是否可用
        return "::1"  # 返回IPv6回环地址
    else:  # 两者都不可用
        raise RuntimeError(  # 抛出运行时错误
            "Neither 127.0.0.1 nor ::1 are bound to a local interface. "  # 127.0.0.1和::1都无法绑定到本地接口
            "Set the VLLM_LOOPBACK_IP environment variable explicitly."  # 请显式设置VLLM_LOOPBACK_IP环境变量
        )


# 验证字符串是否为合法的 IPv6 地址
def is_valid_ipv6_address(address: str) -> bool:
    """验证给定字符串是否为合法的IPv6地址"""
    try:  # 尝试解析IPv6地址
        ipaddress.IPv6Address(address)  # 解析IPv6地址
        return True  # 解析成功，是合法的IPv6地址
    except ValueError:  # 如果解析失败（值错误）
        return False  # 不是合法的IPv6地址


# 解析 "host:port" 字符串，支持 IPv6 方括号格式（如 "[::1]:8080"）
# 返回的tuple中包括两个部分：字符串，端口号。
def split_host_port(host_port: str) -> tuple[str, int]:
    """将 host:port 字符串拆分为主机和端口"""
    # ipv6  # IPv6格式处理
    if host_port.startswith("["):  # 如果以方括号开头，说明是IPv6格式
        host, port = host_port.rsplit("]", 1)  # 从右侧按"]"分割，得到主机和端口部分
        host = host[1:]  # 去掉开头的"["方括号
        port = port.split(":")[1]  # 去掉":"前缀，获取端口号字符串
        return host, int(port)  # 返回主机和整数端口号
    else:  # 非IPv6格式（IPv4或主机名）
        host, port = host_port.split(":")  # 按":"分割主机和端口
        return host, int(port)  # 返回主机和整数端口号


# 将主机和端口拼接为字符串，IPv6 地址自动加方括号
def join_host_port(host: str, port: int) -> str:
    """将主机和端口拼接为 host:port 字符串"""
    if is_valid_ipv6_address(host):  # 如果主机是IPv6地址
        return f"[{host}]:{port}"  # 用方括号包裹IPv6地址
    else:  # 如果是IPv4地址或主机名
        return f"{host}:{port}"  # 直接拼接


# 生成 PyTorch 分布式初始化所需的 TCP URI
def get_distributed_init_method(ip: str, port: int) -> str:
    """生成PyTorch分布式训练初始化方法的TCP URI"""
    return get_tcp_uri(ip, port)  # 调用get_tcp_uri生成TCP地址


# 构建 TCP URI 字符串，IPv6 地址自动加方括号
# f"tcp://{ip}:{port}"中，f表示format的意思。表示把变量ip,port直接塞进字符串中，最终返回的字符串格式为 tcp://12.12.12.12:1234 类似。
def get_tcp_uri(ip: str, port: int) -> str:
    """构建TCP协议的URI字符串"""
    if is_valid_ipv6_address(ip):  # 如果IP是IPv6地址
        return f"tcp://[{ip}]:{port}"  # 用方括号包裹IPv6地址构建URI
    else:  # 如果是IPv4地址
        return f"tcp://{ip}:{port}"  # 直接构建TCP URI


# 生成唯一的 ZMQ IPC 路径（基于 UUID），用于进程间通信
def get_open_zmq_ipc_path() -> str:
    """生成唯一的ZMQ IPC（进程间通信）路径"""
    base_rpc_path = envs.VLLM_RPC_BASE_PATH  # 获取RPC基础路径
    return f"ipc://{base_rpc_path}/{uuid4()}"  # 用UUID生成唯一的IPC路径


# 生成唯一的 ZMQ 进程内通信路径（基于 UUID）
def get_open_zmq_inproc_path() -> str:
    """生成唯一的ZMQ进程内通信路径"""
    return f"inproc://{uuid4()}"  # 用UUID生成唯一的进程内通信路径


# 获取一个可用端口号：在数据并行模式下自动避开主进程预留的端口范围
def get_open_port() -> int:
    """获取一个可用的网络端口，自动避开数据并行主进程预留端口"""
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:  # 如果设置了数据并行主进程端口环境变量
        dp_master_port = envs.VLLM_DP_MASTER_PORT  # 获取数据并行主进程端口
        reserved_port_range = range(dp_master_port, dp_master_port + 10)  # 预留主进程端口起始的10个端口
        while True:  # 循环直到找到不冲突的端口
            candidate_port = _get_open_port()  # 获取一个候选可用端口
            if candidate_port not in reserved_port_range:  # 如果候选端口不在预留范围内
                return candidate_port  # 返回该端口
    return _get_open_port()  # 无数据并行时直接获取可用端口


# 获取多个不重复的可用端口：当设置了 VLLM_PORT 时从该端口开始向上扫描
def get_open_ports_list(count: int = 5) -> list[int]:
    """获取指定数量的不重复可用端口列表"""
    """Get a list of unique open ports.

    When VLLM_PORT is set, scans upward from that port, advancing
    the start position after each find so every port is unique.
    """
    ports_set = set[int]()  # 创建端口集合用于去重
    if envs.VLLM_PORT is not None:  # 如果设置了VLLM_PORT环境变量
        next_port = envs.VLLM_PORT  # 从VLLM_PORT开始扫描
        for _ in range(count):  # 循环获取指定数量的端口
            port = _get_open_port(start_port=next_port, max_attempts=1000)  # 从next_port开始尝试获取可用端口
            ports_set.add(port)  # 将获取到的端口加入集合
            next_port = port + 1  # 下一次从当前端口+1开始扫描
        return list(ports_set)  # 返回端口列表
    else:  # 未设置VLLM_PORT
        while len(ports_set) < count:  # 循环直到收集足够数量的端口
            ports_set.add(get_open_port())  # 添加一个可用端口到集合

    return list(ports_set)  # 返回端口列表


# 底层端口获取：若指定起始端口则递增尝试，否则让系统自动分配（绑定端口 0）
def _get_open_port(
    start_port: int | None = None,  # 起始端口号，默认为None
    max_attempts: int | None = None,  # 最大尝试次数，默认为None（无限制）
) -> int:
    """底层端口获取函数，支持从指定端口递增扫描或系统自动分配"""
    start_port = start_port if start_port is not None else envs.VLLM_PORT  # 如果未指定起始端口则从环境变量获取
    port = start_port  # 将当前端口设为起始端口
    if port is not None:  # 如果指定了起始端口
        attempts = 0  # 初始化尝试计数器
        while True:  # 循环尝试绑定端口
            try:  # 尝试绑定
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建IPv4 TCP套接字
                    s.bind(("", port))  # 尝试绑定到当前端口
                    return port  # 绑定成功，返回该端口
            except OSError:  # 如果端口被占用
                port += 1  # Increment port number if already in use  # 端口号加1继续尝试
                logger.info("Port %d is already in use, trying port %d", port - 1, port)  # 记录端口占用信息
            attempts += 1  # 尝试次数加1
            if max_attempts is not None and attempts >= max_attempts:  # 如果超过最大尝试次数
                raise RuntimeError(  # 抛出运行时错误
                    f"Could not find open port after {max_attempts} "  # 在指定次数内未找到可用端口
                    f"attempts starting from port {start_port}"  # 从起始端口开始尝试
                )
    # try ipv4  # 尝试IPv4自动分配
    try:  # 尝试IPv4
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建IPv4 TCP套接字
            s.bind(("", 0))  # 绑定端口0，让系统自动分配可用端口
            return s.getsockname()[1]  # 返回系统分配的端口号
    except OSError:  # 如果IPv4自动分配失败
        # try ipv6  # 尝试IPv6自动分配
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:  # 创建IPv6 TCP套接字
            s.bind(("", 0))  # 绑定端口0，让系统自动分配可用端口
            return s.getsockname()[1]  # 返回系统分配的端口号


# 查找占用指定端口的进程（排除自身），用于优雅关闭时的冲突检测（macOS 不支持）
def find_process_using_port(port: int) -> psutil.Process | None:
    """查找占用指定端口的进程，排除当前进程自身"""
    # TODO: We can not check for running processes with network  # 待办：无法在macOS上检查占用网络端口的进程
    # port on macOS. Therefore, we can not have a full graceful shutdown  # 因此无法在macOS上完全优雅关闭
    # of vLLM. For now, let's not look for processes in this case.  # 目前在macOS上跳过进程查找
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/  # 参考链接：psutil在macOS上的权限问题
    if sys.platform.startswith("darwin"):  # 如果当前平台是macOS
        return None  # 直接返回None，跳过检测

    our_pid = os.getpid()  # 获取当前进程的PID
    for conn in psutil.net_connections():  # 遍历所有网络连接
        if conn.laddr.port == port and (conn.pid is not None and conn.pid != our_pid):  # 如果连接端口匹配且不是当前进程
            try:  # 尝试获取进程对象
                return psutil.Process(conn.pid)  # 返回占用该端口的进程对象
            except psutil.NoSuchProcess:  # 如果进程已不存在
                return None  # 返回None
    return None  # 没有找到占用该端口的进程，返回None


# 解析 ZMQ 路径为 (scheme, host, port) 三元组，并校验 TCP 必须有 host 和 port，schema即tcp，udp等。
def split_zmq_path(path: str) -> tuple[str, str, str]:
    """将ZMQ路径拆分为协议方案、主机和端口三部分"""
    """Split a zmq path into its parts."""
    parsed = parse_url(path)  # 使用urllib3解析URL
    if not parsed.scheme:  # 如果没有解析到协议方案
        raise ValueError(f"Invalid zmq path: {path}")  # 抛出值错误：无效的ZMQ路径

    scheme = parsed.scheme  # 获取协议方案（如tcp、ipc、inproc）
    host = parsed.hostname or ""  # 获取主机名，默认为空字符串
    port = str(parsed.port or "")  # 获取端口号字符串，默认为空字符串
    if host.startswith("[") and host.endswith("]"):  # 如果主机名被方括号包裹（IPv6格式）
        host = host[1:-1]  # Remove brackets for IPv6 address  # 去掉方括号提取IPv6地址

    if scheme == "tcp" and not all((host, port)):  # 如果是TCP协议但缺少主机或端口
        # The host and port fields are required for tcp  # TCP协议必须提供主机和端口
        raise ValueError(f"Invalid zmq path: {path}")  # 抛出值错误

    if scheme != "tcp" and port:  # 如果不是TCP协议但提供了端口
        # port only makes sense with tcp  # 端口仅对TCP协议有意义
        raise ValueError(f"Invalid zmq path: {path}")  # 抛出值错误

    return scheme, host, port  # 返回协议方案、主机、端口三元组


# 从各部分构建 ZMQ 路径字符串，自动处理 IPv6 方括号格式
def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """从协议方案、主机和端口构建ZMQ路径字符串"""
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:  # 如果未指定端口
        return f"{scheme}://{host}"  # 返回不带端口的ZMQ路径
    if is_valid_ipv6_address(host):  # 如果主机是IPv6地址
        return f"{scheme}://[{host}]:{port}"  # 用方括号包裹IPv6地址构建路径
    return f"{scheme}://{host}:{port}"  # 返回标准格式的ZMQ路径


# 创建并配置 ZMQ 套接字：根据系统内存自动调整缓冲区大小，处理 IPv6、身份标识、ROUTER 握手切换等选项
# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]  # ZMQ上下文对象（同步或异步）
    path: str,  # ZMQ路径字符串
    socket_type: Any,  # ZMQ套接字类型（如PUSH、PULL、ROUTER等）
    bind: bool | None = None,  # 是否绑定模式，默认根据套接字类型自动决定
    identity: bytes | None = None,  # 套接字身份标识，默认为None
    linger: int | None = None,  # 关闭时等待未发送消息的时间，默认为None
    router_handover: bool = False,  # 是否启用ROUTER握手切换，默认为False
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]  # 返回ZMQ套接字对象
    """创建并配置ZMQ套接字，自动处理缓冲区、IPv6和各种选项"""
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()  # 获取系统虚拟内存信息
    socket = ctx.socket(socket_type)  # 在上下文中创建指定类型的套接字

    # Calculate buffer size based on system memory  # 根据系统内存计算缓冲区大小
    total_mem = mem.total / 1024**3  # 计算总内存大小（GB）
    available_mem = mem.available / 1024**3  # 计算可用内存大小（GB）
    # For systems with substantial memory (>32GB total, >16GB available):  # 对于大内存系统（总内存>32GB，可用>16GB）：
    # - Set a large 0.5GB buffer to improve throughput  # 设置0.5GB大缓冲区以提高吞吐量
    # For systems with less memory:  # 对于小内存系统：
    # - Use system default (-1) to avoid excessive memory consumption  # 使用系统默认值(-1)避免内存过度消耗
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1  # 根据内存条件决定缓冲区大小

    if bind is None:  # 如果未指定绑定模式
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)  # PUSH、SUB、XSUB默认连接，其他默认绑定

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):  # 如果是接收类型的套接字
        socket.setsockopt(zmq.RCVHWM, 0)  # 设置接收高水位标记为0（无限制）
        socket.setsockopt(zmq.RCVBUF, buf_size)  # 设置接收缓冲区大小

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):  # 如果是发送类型的套接字
        socket.setsockopt(zmq.SNDHWM, 0)  # 设置发送高水位标记为0（无限制）
        socket.setsockopt(zmq.SNDBUF, buf_size)  # 设置发送缓冲区大小

    if socket_type == zmq.ROUTER and router_handover:  # 如果是ROUTER套接字且启用了握手切换
        # Let a new connection take over an identity left behind by a dead one.  # 允许新连接接管已断开连接遗留的身份标识
        socket.setsockopt(zmq.ROUTER_HANDOVER, 1)  # 启用ROUTER握手切换功能

    if identity is not None:  # 如果指定了身份标识
        socket.setsockopt(zmq.IDENTITY, identity)  # 设置套接字身份标识

    if linger is not None:  # 如果指定了linger值
        socket.setsockopt(zmq.LINGER, linger)  # 设置关闭时等待未发送消息的时间

    if socket_type == zmq.XPUB:  # 如果是XPUB类型套接字
        socket.setsockopt(zmq.XPUB_VERBOSE, True)  # 启用XPUB详细模式，接收所有订阅消息

    # Determine if the path is a TCP socket with an IPv6 address.  # 判断路径是否为带IPv6地址的TCP套接字
    # Enable IPv6 on the zmq socket if so.  # 如果是则启用IPv6支持
    scheme, host, _ = split_zmq_path(path)  # 解析ZMQ路径获取协议方案和主机
    if scheme == "tcp" and is_valid_ipv6_address(host):  # 如果是TCP协议且主机为IPv6地址
        socket.setsockopt(zmq.IPV6, 1)  # 在套接字上启用IPv6支持

    if bind:  # 如果是绑定模式
        socket.bind(path)  # 将套接字绑定到指定路径
    else:  # 如果是连接模式
        socket.connect(path)  # 将套接字连接到指定路径

    return socket  # 返回配置好的套接字对象


# ZMQ 套接字上下文管理器：自动创建上下文和套接字，退出时销毁上下文释放资源
@contextlib.contextmanager  # 装饰为上下文管理器
def zmq_socket_ctx(
    path: str,  # ZMQ路径字符串
    socket_type: Any,  # ZMQ套接字类型
    bind: bool | None = None,  # 是否绑定模式，默认自动决定
    linger: int = 0,  # 关闭时等待时间，默认为0
    identity: bytes | None = None,  # 套接字身份标识，默认为None
    router_handover: bool = False,  # 是否启用ROUTER握手切换，默认为False
) -> Iterator[zmq.Socket]:  # 返回ZMQ套接字的迭代器
    """ZMQ套接字上下文管理器，自动管理上下文的创建和销毁"""
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]  # 创建ZMQ上下文对象
    try:  # 尝试创建并使用套接字
        yield make_zmq_socket(  # 生成配置好的ZMQ套接字供with语句使用
            ctx,  # 传入ZMQ上下文
            path,  # 传入ZMQ路径
            socket_type,  # 传入套接字类型
            bind=bind,  # 传入绑定模式设置
            identity=identity,  # 传入身份标识
            router_handover=router_handover,  # 传入握手切换设置
        )
    except KeyboardInterrupt:  # 如果收到键盘中断信号（Ctrl+C）
        logger.debug("Got Keyboard Interrupt.")  # 记录调试日志：收到键盘中断

    finally:  # 无论是否异常，最终都执行
        ctx.destroy(linger=linger)  # 销毁ZMQ上下文，linger指定等待未发送消息的时间
