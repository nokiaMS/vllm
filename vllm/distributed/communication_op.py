# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any  # 导入Any类型注解

import torch  # 导入PyTorch框架
import torch.distributed  # 导入PyTorch分布式通信模块

from .parallel_state import get_tp_group  # 从并行状态模块导入获取张量并行组的函数


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.
    在模型并行组中对输入张量执行全归约操作。"""
    return get_tp_group().all_reduce(input_)  # 调用张量并行组的all_reduce方法


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group.
    在模型并行组中对输入张量执行全收集操作。"""
    return get_tp_group().all_gather(input_, dim)  # 调用张量并行组的all_gather方法


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group.
    在模型并行组中对输入张量执行归约散布操作。"""
    return get_tp_group().reduce_scatter(input_, dim)  # 调用张量并行组的reduce_scatter方法


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group.
    在模型并行组中对输入张量执行收集操作。"""
    return get_tp_group().gather(input_, dst, dim)  # 调用张量并行组的gather方法


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    """广播张量字典到模型并行组中的所有进程。

    Args:
        tensor_dict: 要广播的张量字典
        src: 源进程的排名
    """
    if not torch.distributed.is_initialized():  # 如果分布式环境未初始化
        return tensor_dict  # 直接返回原始字典
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)  # 通过张量并行组广播字典
