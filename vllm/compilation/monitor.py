# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib  # 导入上下文管理工具模块
import time  # 导入时间模块
from collections.abc import Generator  # 导入生成器类型

from vllm.config import CompilationMode, VllmConfig  # 导入编译模式和vLLM配置
from vllm.logger import init_logger  # 导入日志初始化函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

# Shared global so backends.py can read the start time for Dynamo timing.
torch_compile_start_time: float = 0.0  # 共享全局变量，记录torch.compile开始时间


@contextlib.contextmanager
def monitor_torch_compile(
    vllm_config: VllmConfig,
    message: str = "torch.compile took %.2f s in total",
) -> Generator[None, None, None]:
    """上下文管理器，用于监控torch.compile的编译时间并管理depyf调试。

    正常退出时：记录编译时间并退出depyf。
    异常退出时：清理depyf但不记录日志（编译失败）。
    """
    global torch_compile_start_time  # 声明使用全局变量
    torch_compile_start_time = time.perf_counter()  # 记录编译开始时间

    compilation_config = vllm_config.compilation_config  # 获取编译配置
    depyf_cm = None  # depyf上下文管理器初始化为None
    path = vllm_config.compile_debug_dump_path()  # 获取调试输出路径
    if compilation_config.mode == CompilationMode.VLLM_COMPILE and path:  # 如果是vLLM编译模式且路径有效
        import depyf  # 导入depyf调试模块

        path.mkdir(parents=True, exist_ok=True)  # 创建调试输出目录
        logger.debug("Dumping depyf output to %s", path)  # 记录调试输出路径
        depyf_cm = depyf.prepare_debug(path.as_posix())  # 准备depyf调试
        depyf_cm.__enter__()  # 进入depyf上下文

    try:  # 尝试执行编译
        yield  # 让出控制权给调用者
    except Exception:  # 捕获异常
        raise  # 重新抛出异常
    else:  # 正常完成时
        total_compile_time = time.perf_counter() - torch_compile_start_time  # 计算总编译时间
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:  # 如果是vLLM编译模式
            logger.info_once(message, total_compile_time, scope="local")  # 记录编译时间日志
    finally:  # 最终清理
        if depyf_cm is not None:  # 如果depyf上下文管理器存在
            try:  # 尝试退出depyf
                depyf_cm.__exit__(None, None, None)  # 退出depyf上下文
            except Exception:  # 捕获depyf清理异常
                logger.warning("Exception during depyf cleanup.", exc_info=True)  # 记录清理异常警告


@contextlib.contextmanager
def monitor_profiling_run() -> Generator[None, None, None]:
    """上下文管理器，用于监控初始性能分析运行的耗时。

    断言在性能分析运行期间不会发生后端编译
    （所有编译应在此之前完成）。
    """
    from vllm.compilation.counter import compilation_counter  # 导入编译计数器

    backend_compilations_before = compilation_counter.num_backend_compilations  # 记录运行前的后端编译次数
    start = time.perf_counter()  # 记录开始时间
    yield  # 让出控制权给调用者
    elapsed = time.perf_counter() - start  # 计算耗时
    assert (  # 断言没有发生后端编译
        compilation_counter.num_backend_compilations == backend_compilations_before
    ), (
        "backend compilation occurred during the initial profiling run; "
        "all compilation should be complete before the profiling run starts."
    )
    logger.info_once(  # 记录性能分析运行耗时
        "Initial profiling/warmup run took %.2f s",
        elapsed,
        scope="local",
    )


cudagraph_capturing_enabled: bool = True  # CUDA图捕获启用标志，默认启用


def validate_cudagraph_capturing_enabled() -> None:
    """验证CUDA图捕获是否在合法时间点被调用。

    在运行时用于监控CUDA图捕获是否合法，
    应在任何CUDA图捕获之前调用。
    如果发生非法的CUDA图捕获，则抛出错误。
    """
    global cudagraph_capturing_enabled  # 声明使用全局变量
    if not cudagraph_capturing_enabled:  # 如果CUDA图捕获被禁用
        raise RuntimeError(  # 抛出运行时错误
            "CUDA graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )


def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    """设置CUDA图捕获的启用状态。"""
    global cudagraph_capturing_enabled  # 声明使用全局变量
    cudagraph_capturing_enabled = enabled  # 更新启用状态
