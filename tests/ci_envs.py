# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CI 测试环境变量模块：定义仅在部分测试中生效的 CI 环境变量，支持懒加载求值]
"""
These envs only work for a small part of the tests, fix what you need!
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.envs import maybe_convert_bool

if TYPE_CHECKING:
    VLLM_CI_NO_SKIP: bool = False
    VLLM_CI_DTYPE: str | None = None
    VLLM_CI_HEAD_DTYPE: str | None = None
    VLLM_CI_HF_DTYPE: str | None = None

environment_variables: dict[str, Callable[[], Any]] = {
    # A model family has many models with the same architecture.
    # By default, a model family tests only one model.
    # Through this flag, all models can be tested.
    "VLLM_CI_NO_SKIP": lambda: bool(int(os.getenv("VLLM_CI_NO_SKIP", "0"))),
    # Allow changing the dtype used by vllm in tests
    "VLLM_CI_DTYPE": lambda: os.getenv("VLLM_CI_DTYPE", None),
    # Allow changing the head dtype used by vllm in tests
    "VLLM_CI_HEAD_DTYPE": lambda: os.getenv("VLLM_CI_HEAD_DTYPE", None),
    # Allow changing the head dtype used by transformers in tests
    "VLLM_CI_HF_DTYPE": lambda: os.getenv("VLLM_CI_HF_DTYPE", None),
    # Allow control over whether tests use enforce_eager
    "VLLM_CI_ENFORCE_EAGER": lambda: maybe_convert_bool(
        os.getenv("VLLM_CI_ENFORCE_EAGER", None)
    ),
}


# [懒加载属性访问：当访问模块属性时，动态求值对应的环境变量]
def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# [列出所有可用的环境变量名称]
def __dir__():
    return list(environment_variables.keys())


# [检查指定的环境变量是否在 os.environ 中被显式设置]
def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
