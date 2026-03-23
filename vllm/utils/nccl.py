# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NCCL/RCCL 通信库工具模块
# 提供查找 NCCL 共享库文件和头文件路径的功能，支持 CUDA 和 ROCm 两种后端

from __future__ import annotations

import importlib.util
import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


# 查找 NCCL/RCCL 共享库路径
# 优先使用环境变量 VLLM_NCCL_SO_PATH，否则根据 CUDA/ROCm 后端选择默认库名
def find_nccl_library() -> str:
    """Return NCCL/RCCL shared library name to load.

    Uses `VLLM_NCCL_SO_PATH` if set; otherwise chooses by torch backend.
    """
    so_file = envs.VLLM_NCCL_SO_PATH
    if so_file:
        logger.info(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s", so_file
        )
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.debug_once("Found nccl from library %s", so_file)
    return so_file


# 查找包含 nccl.h 头文件的目录路径列表
# 依次检查环境变量 VLLM_NCCL_INCLUDE_PATH 和 nvidia-nccl Python 包中的 include 目录，
# 返回去重后的路径列表，找不到则返回 None
def find_nccl_include_paths() -> list[str] | None:
    """Return possible include paths containing `nccl.h`.

    Considers `VLLM_NCCL_INCLUDE_PATH` and the `nvidia-nccl-cuXX` package.
    """
    paths: list[str] = []
    inc = envs.VLLM_NCCL_INCLUDE_PATH
    if inc and os.path.isdir(inc):
        paths.append(inc)

    try:
        spec = importlib.util.find_spec("nvidia.nccl")
        if spec and (locs := getattr(spec, "submodule_search_locations", None)):
            for loc in locs:
                inc_dir = os.path.join(loc, "include")
                if os.path.exists(os.path.join(inc_dir, "nccl.h")):
                    paths.append(inc_dir)
    except Exception as e:
        logger.debug("Failed to find nccl include path from nvidia.nccl package: %s", e)

    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out or None
