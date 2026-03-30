# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者

import torch  # 导入 PyTorch 库，用于张量操作

from vllm.platforms import current_platform  # 导入当前平台检测工具，用于判断运行在 CUDA/XPU 等平台


def merge_attn_states(  # 定义注意力状态合并函数
    output: torch.Tensor,  # 输出张量，合并结果将写入此张量
    prefix_output: torch.Tensor,  # 前缀部分的注意力输出张量
    prefix_lse: torch.Tensor,  # 前缀部分的 log-sum-exp 值张量
    suffix_output: torch.Tensor,  # 后缀部分的注意力输出张量
    suffix_lse: torch.Tensor,  # 后缀部分的 log-sum-exp 值张量
    output_lse: torch.Tensor | None = None,  # 可选的输出 log-sum-exp 张量，默认为 None
) -> None:  # 无返回值，结果直接写入 output 张量
    """合并前缀和后缀的注意力状态。

    将分块计算的前缀和后缀注意力输出通过 log-sum-exp 加权合并，
    根据运行平台自动选择 CUDA 自定义内核或 Triton 内核实现。
    """
    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # does not support FP8 dtype, fallback to use Triton kernel.
    # 注意：当前自定义 CUDA 合并内核不支持 FP8 数据类型，需回退到 Triton 内核
    def supported_dtypes(o: torch.Tensor) -> bool:  # 定义支持的数据类型检查函数
        """检查张量的数据类型是否被 CUDA 自定义内核支持。"""
        return o.dtype in [torch.float32, torch.half, torch.bfloat16]  # 仅支持 float32、float16 和 bfloat16 三种类型

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size (float32 -> 4, half/bfloat16 -> 8).
    # 注意：CUDA 内核每次内存操作加载/存储 128 位（16 字节），
    # 因此 head_size 必须是 pack_size 的整数倍（float32 需整除 4，half/bfloat16 需整除 8）
    def supported_headdim(o: torch.Tensor) -> bool:  # 定义支持的头维度检查函数
        """检查注意力头维度是否满足 CUDA 内核的对齐要求。"""
        headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]  # 获取头维度大小，张量形状为 [token数, 头数, 头维度]
        if o.dtype == torch.float32:  # 如果数据类型是 float32
            return headdim % 4 == 0  # float32 每个元素 4 字节，128位/4字节=4，头维度需能被 4 整除
        return headdim % 8 == 0  # half/bfloat16 每个元素 2 字节，128位/2字节=8，头维度需能被 8 整除

    if (  # 判断是否可以使用 CUDA 自定义内核
        current_platform.is_cuda()  # 检查当前平台是否为 CUDA
        and supported_dtypes(output)  # 检查数据类型是否受支持
        and supported_headdim(output)  # 检查头维度是否满足对齐要求
    ):
        from vllm._custom_ops import merge_attn_states  # 动态导入 CUDA 自定义合并操作

        return merge_attn_states(  # 调用 CUDA 自定义内核执行合并
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
    else:  # 不满足 CUDA 条件时使用 Triton 实现
        from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states  # 动态导入 Triton 合并操作

        return merge_attn_states(  # 调用 Triton 内核执行合并
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
