# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

import torch  # 导入PyTorch库

from vllm.triton_utils import triton  # 从triton_utils导入triton模块（可能是真实triton或占位符）


def set_triton_allocator(device: torch.device):  # 设置Triton的内存分配器，指定目标设备
    """设置Triton的自定义内存分配器。

    该函数为Triton配置一个自定义的内存分配函数，
    使其在指定的设备上分配内存。

    Args:
        device: 用于内存分配的目标PyTorch设备。
    """
    def alloc_fn(size: int, alignment: int, stream: int | None):  # 定义分配函数，接收大小、对齐方式和CUDA流
        """自定义内存分配回调函数。

        Args:
            size: 要分配的内存大小（字节）。
            alignment: 内存对齐要求。
            stream: 可选的CUDA流标识。

        Returns:
            在指定设备上分配的int8类型张量。
        """
        return torch.empty(size, device=device, dtype=torch.int8)  # 在指定设备上分配int8类型的空张量

    triton.set_allocator(alloc_fn)  # 将自定义分配函数注册到Triton
