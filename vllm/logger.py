# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM."""  # vLLM 的日志配置模块

import datetime  # 导入日期时间模块，用于生成时间戳
import json  # 导入 JSON 模块，用于解析日志配置文件
import logging  # 导入标准日志模块
import os  # 导入操作系统接口模块
import sys  # 导入系统模块，用于访问标准输入输出流
from collections.abc import Generator, Hashable  # 导入生成器和可哈希类型抽象基类
from contextlib import contextmanager  # 导入上下文管理器装饰器
from functools import lru_cache, partial  # 导入 LRU 缓存装饰器和偏函数工具
from logging import Logger  # 导入 Logger 类型
from logging.config import dictConfig  # 导入字典配置方式加载日志配置的函数
from os import path  # 导入路径操作工具
from types import MethodType  # 导入方法类型，用于动态绑定方法到实例
from typing import Any, Literal, cast  # 导入类型注解工具

import vllm.envs as envs  # 导入 vLLM 环境变量配置模块
from vllm.logging_utils import ColoredFormatter, NewLineFormatter  # 导入彩色格式化器和换行格式化器

_FORMAT = (  # 日志输出格式字符串
    f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "  # 包含前缀、日志级别和时间戳
    "[%(fileinfo)s:%(lineno)d] %(message)s"  # 包含文件信息、行号和日志消息
)
_DATE_FORMAT = "%m-%d %H:%M:%S"  # 日期格式：月-日 时:分:秒


def _use_color() -> bool:
    """判断是否启用彩色日志输出。

    根据环境变量和终端类型决定是否使用彩色格式化器。
    """
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":  # 如果设置了禁用颜色的环境变量
        return False  # 返回不使用颜色
    if envs.VLLM_LOGGING_COLOR == "1":  # 如果明确启用了颜色
        return True  # 返回使用颜色
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":  # stdout  # 如果日志输出到标准输出
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()  # 检查标准输出是否为终端
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":  # stderr  # 如果日志输出到标准错误
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()  # 检查标准错误是否为终端
    return False  # 默认不使用颜色


DEFAULT_LOGGING_CONFIG: dict[str, dict[str, Any] | Any] = {  # 默认日志配置字典
    "formatters": {  # 格式化器配置
        "vllm": {  # 普通格式化器（无颜色）
            "class": "vllm.logging_utils.NewLineFormatter",  # 使用换行格式化器类
            "datefmt": _DATE_FORMAT,  # 日期格式
            "format": _FORMAT,  # 日志格式
        },
        "vllm_color": {  # 彩色格式化器
            "class": "vllm.logging_utils.ColoredFormatter",  # 使用彩色格式化器类
            "datefmt": _DATE_FORMAT,  # 日期格式
            "format": _FORMAT,  # 日志格式
        },
    },
    "handlers": {  # 处理器配置
        "vllm": {  # vLLM 日志处理器
            "class": "logging.StreamHandler",  # 使用流处理器
            # Choose formatter based on color setting.
            "formatter": "vllm_color" if _use_color() else "vllm",  # 根据颜色设置选择格式化器
            "level": envs.VLLM_LOGGING_LEVEL,  # 日志级别由环境变量决定
            "stream": envs.VLLM_LOGGING_STREAM,  # 日志输出流由环境变量决定
        },
    },
    "loggers": {  # 日志器配置
        "vllm": {  # vLLM 命名空间的日志器
            "handlers": ["vllm"],  # 使用 vllm 处理器
            "level": envs.VLLM_LOGGING_LEVEL,  # 日志级别由环境变量决定
            "propagate": False,  # 不向父日志器传播
        },
    },
    "version": 1,  # 日志配置版本号
    "disable_existing_loggers": False,  # 不禁用已有的日志器
}


@lru_cache  # 使用 LRU 缓存确保相同消息只打印一次
def _print_debug_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """仅打印一次的 debug 级别日志。

    通过 LRU 缓存机制，相同参数的调用只会执行一次。
    """
    # Set the stacklevel to 3 to print the original caller's line info
    logger.debug(msg, *args, stacklevel=3)  # 设置栈层级为3以显示原始调用者的行信息


@lru_cache  # 使用 LRU 缓存确保相同消息只打印一次
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """仅打印一次的 info 级别日志。

    通过 LRU 缓存机制，相同参数的调用只会执行一次。
    """
    # Set the stacklevel to 3 to print the original caller's line info
    logger.info(msg, *args, stacklevel=3)  # 设置栈层级为3以显示原始调用者的行信息


@lru_cache  # 使用 LRU 缓存确保相同消息只打印一次
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """仅打印一次的 warning 级别日志。

    通过 LRU 缓存机制，相同参数的调用只会执行一次。
    """
    # Set the stacklevel to 3 to print the original caller's line info
    logger.warning(msg, *args, stacklevel=3)  # 设置栈层级为3以显示原始调用者的行信息


LogScope = Literal["process", "global", "local"]  # 日志作用域类型：进程级、全局级、本地级


def _should_log_with_scope(scope: LogScope) -> bool:
    """根据作用域判断是否应该记录日志。

    Args:
        scope: 日志作用域，可选值为 "process"（进程级）、"global"（全局级）、"local"（本地级）。

    Returns:
        bool: 是否应该记录日志。
    """
    if scope == "global":  # 如果是全局作用域
        from vllm.distributed.parallel_state import is_global_first_rank  # 延迟导入全局首位进程判断函数

        return is_global_first_rank()  # 仅在全局首位进程中记录日志
    if scope == "local":  # 如果是本地作用域
        from vllm.distributed.parallel_state import is_local_first_rank  # 延迟导入本地首位进程判断函数

        return is_local_first_rank()  # 仅在本地首位进程中记录日志
    # default "process" scope: always log
    return True  # 默认进程级作用域：始终记录日志


class _VllmLogger(Logger):
    """vLLM 自定义日志器类，提供"仅记录一次"系列方法的类型信息。

    注意:
        这个类仅用于提供类型信息。
        实际上我们是直接在 logging.Logger 实例上动态补丁方法，
        以避免与其他库（如 intel_extension_for_pytorch.utils._logger）产生冲突。
    """

    def debug_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None:
        """仅记录一次的 debug 级别日志方法。

        类似于 logging.Logger.debug，但相同消息的后续调用会被静默丢弃。

        Args:
            msg: 日志消息字符串。
            *args: 日志消息的格式化参数。
            scope: 日志作用域，默认为 "process"。
        """
        if not _should_log_with_scope(scope):  # 检查当前作用域是否允许记录日志
            return  # 不允许则直接返回
        _print_debug_once(self, msg, *args)  # 调用缓存函数确保仅打印一次

    def info_once(self, msg: str, *args: Hashable, scope: LogScope = "process") -> None:
        """仅记录一次的 info 级别日志方法。

        类似于 logging.Logger.info，但相同消息的后续调用会被静默丢弃。

        Args:
            msg: 日志消息字符串。
            *args: 日志消息的格式化参数。
            scope: 日志作用域，默认为 "process"。
        """
        if not _should_log_with_scope(scope):  # 检查当前作用域是否允许记录日志
            return  # 不允许则直接返回
        _print_info_once(self, msg, *args)  # 调用缓存函数确保仅打印一次

    def warning_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None:
        """仅记录一次的 warning 级别日志方法。

        类似于 logging.Logger.warning，但相同消息的后续调用会被静默丢弃。

        Args:
            msg: 日志消息字符串。
            *args: 日志消息的格式化参数。
            scope: 日志作用域，默认为 "process"。
        """
        if not _should_log_with_scope(scope):  # 检查当前作用域是否允许记录日志
            return  # 不允许则直接返回
        _print_warning_once(self, msg, *args)  # 调用缓存函数确保仅打印一次


# Pre-defined methods mapping to avoid repeated dictionary creation
_METHODS_TO_PATCH = {  # 预定义需要补丁的方法映射，避免重复创建字典
    "debug_once": _VllmLogger.debug_once,  # debug_once 方法映射
    "info_once": _VllmLogger.info_once,  # info_once 方法映射
    "warning_once": _VllmLogger.warning_once,  # warning_once 方法映射
}


def _configure_vllm_root_logger() -> None:
    """配置 vLLM 根日志器。

    根据环境变量加载默认配置或自定义 JSON 配置文件，
    并通过 dictConfig 应用日志配置。
    """
    logging_config: dict[str, dict[str, Any] | Any] = {}  # 初始化空的日志配置字典

    if not envs.VLLM_CONFIGURE_LOGGING and envs.VLLM_LOGGING_CONFIG_PATH:  # 如果禁用了日志配置但指定了配置文件路径
        raise RuntimeError(  # 抛出运行时错误
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "  # 错误信息：配置标志为 false
            "VLLM_LOGGING_CONFIG_PATH was given. VLLM_LOGGING_CONFIG_PATH "  # 但指定了配置文件路径
            "implies VLLM_CONFIGURE_LOGGING. Please enable "  # 提示用户启用日志配置
            "VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH."  # 或取消设置配置文件路径
        )

    if envs.VLLM_CONFIGURE_LOGGING:  # 如果启用了日志配置
        logging_config = DEFAULT_LOGGING_CONFIG  # 使用默认日志配置

        vllm_handler = logging_config["handlers"]["vllm"]  # 获取 vllm 处理器配置
        # Refresh these values in case env vars have changed.
        vllm_handler["level"] = envs.VLLM_LOGGING_LEVEL  # 刷新日志级别，以防环境变量已更改
        vllm_handler["stream"] = envs.VLLM_LOGGING_STREAM  # 刷新日志输出流
        vllm_handler["formatter"] = "vllm_color" if _use_color() else "vllm"  # 刷新格式化器选择

        vllm_loggers = logging_config["loggers"]["vllm"]  # 获取 vllm 日志器配置
        vllm_loggers["level"] = envs.VLLM_LOGGING_LEVEL  # 刷新日志器级别

    if envs.VLLM_LOGGING_CONFIG_PATH:  # 如果指定了自定义日志配置文件路径
        if not path.exists(envs.VLLM_LOGGING_CONFIG_PATH):  # 如果配置文件不存在
            raise RuntimeError(  # 抛出运行时错误
                "Could not load logging config. File does not exist: %s",  # 错误信息：文件不存在
                envs.VLLM_LOGGING_CONFIG_PATH,  # 附带文件路径
            )
        with open(envs.VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:  # 打开自定义配置文件
            custom_config = json.loads(file.read())  # 解析 JSON 配置内容

        if not isinstance(custom_config, dict):  # 如果解析结果不是字典
            raise ValueError(  # 抛出值错误
                "Invalid logging config. Expected dict, got %s.",  # 错误信息：期望字典类型
                type(custom_config).__name__,  # 附带实际类型名称
            )
        logging_config = custom_config  # 使用自定义配置覆盖默认配置

    for formatter in logging_config.get("formatters", {}).values():  # 遍历所有格式化器配置
        # This provides backwards compatibility after #10134.
        if formatter.get("class") == "vllm.logging.NewLineFormatter":  # 如果使用了旧的类路径（向后兼容 #10134）
            formatter["class"] = "vllm.logging_utils.NewLineFormatter"  # 替换为新的类路径

    if logging_config:  # 如果存在有效的日志配置
        dictConfig(logging_config)  # 应用字典格式的日志配置


def init_logger(name: str) -> _VllmLogger:
    """初始化并返回一个 vLLM 日志器实例。

    此函数的主要目的是确保在获取日志器时，
    vLLM 的根日志器已经完成配置。同时会为日志器实例
    动态添加 debug_once、info_once、warning_once 方法。

    Args:
        name: 日志器名称，通常使用 __name__。

    Returns:
        _VllmLogger: 带有额外"仅记录一次"方法的日志器实例。
    """

    logger = logging.getLogger(name)  # 通过标准日志模块获取日志器实例

    for method_name, method in _METHODS_TO_PATCH.items():  # 遍历需要补丁的方法
        setattr(logger, method_name, MethodType(method, logger))  # 将方法动态绑定到日志器实例

    return cast(_VllmLogger, logger)  # 将日志器强制转换为 _VllmLogger 类型并返回


@contextmanager  # 上下文管理器装饰器
def suppress_logging(level: int = logging.INFO) -> Generator[None, Any, None]:
    """临时抑制指定级别及以下的日志输出的上下文管理器。

    Args:
        level: 要抑制的日志级别，默认为 INFO。

    Yields:
        None: 在上下文块中不产出任何值。
    """
    current_level = logging.root.manager.disable  # 保存当前的日志禁用级别
    logging.disable(level)  # 禁用指定级别及以下的日志
    yield  # 暂停执行，将控制权交给 with 代码块
    logging.disable(current_level)  # 恢复之前的日志禁用级别


def current_formatter_type(logger: Logger) -> Literal["color", "newline", None]:
    """获取日志器当前使用的格式化器类型。

    沿日志器层级向上查找，返回第一个匹配的 vLLM 处理器的格式化器类型。

    Args:
        logger: 要检查的日志器实例。

    Returns:
        "color" 表示使用彩色格式化器，"newline" 表示使用换行格式化器，None 表示未找到。
    """
    lgr: Logger | None = logger  # 初始化当前查找的日志器
    while lgr is not None:  # 沿日志器层级向上遍历
        if lgr.handlers and len(lgr.handlers) == 1 and lgr.handlers[0].name == "vllm":  # 如果找到了 vllm 处理器
            formatter = lgr.handlers[0].formatter  # 获取该处理器的格式化器
            if isinstance(formatter, ColoredFormatter):  # 如果是彩色格式化器
                return "color"  # 返回 "color"
            if isinstance(formatter, NewLineFormatter):  # 如果是换行格式化器
                return "newline"  # 返回 "newline"
        lgr = lgr.parent  # 移动到父日志器继续查找
    return None  # 未找到匹配的格式化器，返回 None


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_vllm_root_logger()  # 模块导入时初始化根日志器，由 Python GIL 保证线程安全

# Transformers uses httpx to access the Hugging Face Hub. httpx is quite verbose,
# so we set its logging level to WARNING when vLLM's logging level is INFO.
if envs.VLLM_LOGGING_LEVEL == "INFO":  # 当 vLLM 日志级别为 INFO 时
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 将 httpx 日志级别设为 WARNING 以减少冗余输出

logger = init_logger(__name__)  # 初始化本模块的日志器实例


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    """函数调用追踪回调，用于记录每个函数的调用和返回事件。

    此函数作为 sys.settrace 的回调，记录 vLLM 代码中每个函数的调用和返回信息。

    Args:
        log_path: 追踪日志文件的路径。
        root_dir: 要追踪的代码根目录。
        frame: 当前栈帧对象。
        event: 事件类型，"call" 表示函数调用，"return" 表示函数返回。
        arg: 事件相关参数（未使用）。

    Returns:
        partial: 返回绑定了 log_path 和 root_dir 的偏函数，作为下一次追踪回调。
    """
    if event in ["call", "return"]:  # 如果事件是函数调用或返回
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename  # 提取当前帧的文件名
        lineno = frame.f_lineno  # 提取当前帧的行号
        func_name = frame.f_code.co_name  # 提取当前帧的函数名
        if not filename.startswith(root_dir):  # 如果文件不在 vLLM 根目录下
            # only log the functions in the vllm root_dir
            return  # 跳过非 vLLM 代码的函数
        # Log every function call or return
        try:  # 尝试记录函数调用或返回信息
            last_frame = frame.f_back  # 获取上一个栈帧（调用者）
            if last_frame is not None:  # 如果存在调用者栈帧
                last_filename = last_frame.f_code.co_filename  # 获取调用者的文件名
                last_lineno = last_frame.f_lineno  # 获取调用者的行号
                last_func_name = last_frame.f_code.co_name  # 获取调用者的函数名
            else:  # 如果不存在调用者栈帧
                # initial frame
                last_filename = ""  # 初始帧，文件名为空
                last_lineno = 0  # 初始帧，行号为0
                last_func_name = ""  # 初始帧，函数名为空
            with open(log_path, "a") as f:  # 以追加模式打开日志文件
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # 生成当前时间戳（精确到微秒）
                if event == "call":  # 如果是函数调用事件
                    f.write(  # 写入调用信息
                        f"{ts} Call to"  # 时间戳和调用标识
                        f" {func_name} in {filename}:{lineno}"  # 被调用函数的位置信息
                        f" from {last_func_name} in {last_filename}:"  # 调用者的位置信息
                        f"{last_lineno}\n"  # 调用者的行号
                    )
                else:  # 如果是函数返回事件
                    f.write(  # 写入返回信息
                        f"{ts} Return from"  # 时间戳和返回标识
                        f" {func_name} in {filename}:{lineno}"  # 返回函数的位置信息
                        f" to {last_func_name} in {last_filename}:"  # 返回目标的位置信息
                        f"{last_lineno}\n"  # 返回目标的行号
                    )
        except NameError:  # 捕获 NameError 异常
            # modules are deleted during shutdown
            pass  # 关闭期间模块可能已被删除，忽略此错误
    return partial(_trace_calls, log_path, root_dir)  # 返回绑定参数的偏函数作为下一次追踪回调


def enable_trace_function_call(log_file_path: str, root_dir: str | None = None):
    """启用函数调用追踪功能。

    启用对 root_dir 目录下所有函数调用的追踪记录，
    适用于调试程序挂起或崩溃的场景。

    注意：此调用是线程级别的，只有调用此函数的线程会启用追踪，
    其他线程不受影响。

    Args:
        log_file_path: 追踪日志文件的保存路径。
        root_dir: 要追踪的代码根目录。如果为 None，则默认使用 vLLM 根目录。
    """
    logger.warning(  # 输出警告日志
        "VLLM_TRACE_FUNCTION is enabled. It will record every"  # 提示追踪功能已启用
        " function executed by Python. This will slow down the code. It "  # 警告会降低代码执行速度
        "is suggested to be used for debugging hang or crashes only."  # 建议仅在调试挂起或崩溃时使用
    )
    logger.info("Trace frame log is saved to %s", log_file_path)  # 输出追踪日志文件保存路径
    if root_dir is None:  # 如果未指定根目录
        # by default, this is the vllm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))  # 默认使用 vLLM 项目根目录
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))  # 设置系统级别的函数追踪回调
