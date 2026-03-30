# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
import contextlib  # 导入上下文管理工具模块
import os  # 导入操作系统接口模块
import signal  # 导入信号处理模块
import subprocess  # 导入子进程管理模块
import time  # 导入时间模块
from types import TracebackType  # 导入异常追踪类型

import requests  # 导入HTTP请求库
from typing_extensions import Self  # 导入Self类型标注


class ServerProcess:
    """
    管理vLLM服务器进程的类，支持启动、停止、健康检查和缓存重置。
    """
    VLLM_RESET_CACHE_ENDPOINTS = [  # vLLM缓存重置端点列表
        "/reset_prefix_cache",  # 重置前缀缓存
        "/reset_mm_cache",  # 重置多模态缓存
        "/reset_encoder_cache",  # 重置编码器缓存
    ]

    def __init__(  # 初始化方法
        self,
        server_cmd: list[str],  # 服务器启动命令
        after_bench_cmd: list[str],  # 基准测试后执行的命令
        *,
        show_stdout: bool,  # 是否显示标准输出
    ) -> None:  # 无返回值
        """
        初始化服务器进程管理器。
        """
        super().__init__()  # 调用父类初始化

        self.server_cmd = server_cmd  # 保存服务器命令
        self.after_bench_cmd = after_bench_cmd  # 保存基准测试后命令
        self.show_stdout = show_stdout  # 保存是否显示输出的标志

    def __enter__(self) -> Self:  # 上下文管理器进入方法
        """
        进入上下文管理器时启动服务器。
        """
        self.start()  # 启动服务器
        return self  # 返回自身

    def __exit__(  # 上下文管理器退出方法
        self,
        exc_type: type[BaseException] | None,  # 异常类型
        exc_value: BaseException | None,  # 异常值
        exc_traceback: TracebackType | None,  # 异常追踪
    ) -> None:  # 无返回值
        """
        退出上下文管理器时停止服务器。
        """
        self.stop()  # 停止服务器

    def start(self):
        """
        启动服务器进程，创建新的进程组以便于清理。
        """
        # Create new process for clean termination
        self._server_process = subprocess.Popen(  # 创建子进程
            self.server_cmd,  # 服务器命令
            start_new_session=True,  # 创建新会话以便干净地终止
            stdout=None if self.show_stdout else subprocess.DEVNULL,  # 根据配置决定是否输出
            # Need `VLLM_SERVER_DEV_MODE=1` for `_reset_caches`
            env=os.environ | {"VLLM_SERVER_DEV_MODE": "1"},  # 设置开发模式环境变量
        )

    def stop(self):
        """
        停止服务器进程，通过杀死进程组确保所有相关进程都被终止。
        """
        server_process = self._server_process  # 获取服务器进程

        if server_process.poll() is None:  # 如果进程仍在运行
            # In case only some processes have been terminated
            with contextlib.suppress(ProcessLookupError):  # 忽略进程不存在错误
                # We need to kill both API Server and Engine processes
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)  # 杀死整个进程组

    def run_subcommand(self, cmd: list[str]):
        """
        运行子命令（如基准测试命令）。
        """
        return subprocess.run(  # 运行子进程
            cmd,  # 命令
            stdout=None if self.show_stdout else subprocess.DEVNULL,  # 控制输出
            check=True,  # 检查返回码
        )

    def after_bench(self) -> None:
        """
        基准测试完成后执行的操作：运行自定义命令或重置缓存。
        """
        if not self.after_bench_cmd:  # 如果没有自定义命令
            self.reset_caches()  # 重置缓存
            return  # 返回

        self.run_subcommand(self.after_bench_cmd)  # 运行自定义命令

    def _get_vllm_server_address(self) -> str:
        """
        从服务器命令中解析服务器地址（主机名和端口）。
        """
        server_cmd = self.server_cmd  # 获取服务器命令

        for host_key in ("--host",):  # 查找主机参数
            if host_key in server_cmd:  # 如果找到
                host = server_cmd[server_cmd.index(host_key) + 1]  # 获取主机名
                break  # 结束查找
        else:  # 如果未找到
            host = "localhost"  # 使用默认主机名

        for port_key in ("-p", "--port"):  # 查找端口参数
            if port_key in server_cmd:  # 如果找到
                port = int(server_cmd[server_cmd.index(port_key) + 1])  # 获取端口号
                break  # 结束查找
        else:  # 如果未找到
            port = 8000  # The default value in vllm serve  # 使用默认端口

        return f"http://{host}:{port}"  # 返回完整URL

    def is_server_ready(self) -> bool:
        """
        检查服务器是否已准备就绪（通过健康检查端点）。
        """
        server_address = self._get_vllm_server_address()  # 获取服务器地址
        try:  # 尝试请求
            response = requests.get(f"{server_address}/health")  # 发送健康检查请求
            return response.status_code == 200  # 返回是否成功
        except requests.RequestException:  # 捕获请求异常
            return False  # 返回未就绪

    def wait_until_ready(self, timeout: int) -> None:
        """
        等待服务器准备就绪，超时则抛出异常。
        """
        start_time = time.monotonic()  # 记录开始时间
        while not self.is_server_ready():  # 循环检查直到就绪
            # Check if server process has crashed
            if self._server_process.poll() is not None:  # 检查进程是否已崩溃
                returncode = self._server_process.returncode  # 获取返回码
                raise RuntimeError(  # 抛出运行时错误
                    f"Server process crashed with return code {returncode}"  # 服务器进程崩溃
                )
            if time.monotonic() - start_time > timeout:  # 如果超时
                raise TimeoutError(  # 抛出超时错误
                    f"Server failed to become ready within {timeout} seconds."  # 服务器未在指定时间内就绪
                )
            time.sleep(1)  # 等待1秒后重试

    def reset_caches(self) -> None:
        """
        重置服务器的各种缓存（前缀缓存、多模态缓存、编码器缓存）。
        """
        server_cmd = self.server_cmd  # 获取服务器命令

        # Use `.endswith()` to match `/bin/...`
        if server_cmd[0].endswith("vllm"):  # 如果是vLLM服务器
            server_address = self._get_vllm_server_address()  # 获取服务器地址
            print(f"Resetting caches at {server_address}")  # 打印重置信息

            for endpoint in self.VLLM_RESET_CACHE_ENDPOINTS:  # 遍历缓存重置端点
                res = requests.post(server_address + endpoint)  # 发送POST请求
                res.raise_for_status()  # 检查响应状态
        elif server_cmd[0].endswith("infinity_emb"):  # 如果是Infinity服务器
            if "--vector-disk-cache" in server_cmd:  # 如果启用了磁盘缓存
                raise NotImplementedError(  # 抛出未实现错误
                    "Infinity server uses caching but does not expose a method "  # Infinity服务器不提供缓存重置方法
                    "to reset the cache"  # 无法重置缓存
                )
        else:  # 其他服务器
            raise NotImplementedError(  # 抛出未实现错误
                f"No implementation of `reset_caches` for `{server_cmd[0]}` server. "  # 该服务器未实现缓存重置
                "Please specify a custom command via `--after-bench-cmd`."  # 请通过自定义命令指定
            )
