# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
from collections.abc import Callable
from functools import wraps
from typing import Any


# 基于 cProfile 的上下文管理器，用于对代码块进行性能分析；
# 支持将结果保存到文件或直接打印到标准输出
@contextlib.contextmanager
def cprofile_context(save_file: str | None = None):
    """Run a cprofile

    Args:
        save_file: path to save the profile result. "1" or
            None will result in printing to stdout.
    """
    import cProfile

    prof = cProfile.Profile()
    prof.enable()

    try:
        yield
    finally:
        prof.disable()
        if save_file and save_file != "1":
            prof.dump_stats(save_file)
        else:
            prof.print_stats(sort="cumtime")


# 装饰器版本的 cProfile 性能分析工具，可通过 enabled 参数控制是否启用，
# 方便在开发和生产环境间切换
def cprofile(save_file: str | None = None, enabled: bool = True):
    """Decorator to profile a Python method using cProfile.

    Args:
        save_file: Path to save the profile result.
            If "1", None, or "", results will be printed to stdout.
        enabled: Set to false to turn this into a no-op
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            if not enabled:
                # If profiling is disabled, just call the function directly.
                return func(*args, **kwargs)

            with cprofile_context(save_file):
                return func(*args, **kwargs)

        return wrapper

    return decorator
