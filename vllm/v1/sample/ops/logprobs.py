# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""logprobs相关的实用工具函数，包括logits处理。"""

import torch  # 导入PyTorch张量库

from vllm.platforms import current_platform  # 导入当前平台信息


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)  # 使用torch.compile编译优化，支持动态形状
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    批量计算每行中大于对应值的元素数量。

    使用torch.compile生成优化的内核。如果不使用编译，
    会创建输入张量的额外副本导致内存问题。

    Args:
        x (torch.Tensor): 形状为(batch_size, n_elements)的二维张量。
        values (torch.Tensor): 形状为(batch_size, 1)的二维张量。

    Returns:
        torch.Tensor: 形状为(batch_size,)的一维张量，包含每行的计数结果。
    """
    return (x >= values).sum(-1)  # 统计每行中大于等于对应阈值的元素个数
