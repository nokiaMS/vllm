# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch  # 导入PyTorch库

from vllm.triton_utils import tl, triton  # 从vllm工具包导入Triton语言模块和Triton编译器模块


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    output: torch.Tensor,  # 输出张量，用于存储合并后的注意力结果
    prefix_output: torch.Tensor,  # 前缀部分的注意力输出
    prefix_lse: torch.Tensor,  # 前缀部分的log-sum-exp值
    suffix_output: torch.Tensor,  # 后缀部分的注意力输出
    suffix_lse: torch.Tensor,  # 后缀部分的log-sum-exp值
    output_lse: torch.Tensor | None = None,  # 可选的输出log-sum-exp张量
) -> None:
    """合并前缀和后缀两部分的注意力状态。

    实现论文 https://www.arxiv.org/pdf/2501.01005 第2.2节的算法，
    用于在split-KV场景下将部分注意力计算结果合并为完整结果。
    """
    num_tokens = output.shape[0]  # 获取token数量（批次中的序列token总数）
    num_query_heads = output.shape[1]  # 获取查询头的数量
    head_size = output.shape[2]  # 获取每个注意力头的维度大小
    padded_head_size = triton.next_power_of_2(head_size)  # 将头维度大小向上取整到最近的2的幂次（Triton优化需要）
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the attention
    # backend.
    prefix_head_stride = prefix_output.stride(1)  # 获取前缀输出在注意力头维度上的步幅
    output_head_stride = output.stride(1)  # 获取输出张量在注意力头维度上的步幅
    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](  # 启动Triton内核，网格大小为(token数, 查询头数)
        output,  # 传入输出张量
        output_lse,  # 传入输出的log-sum-exp张量
        prefix_output,  # 传入前缀注意力输出
        prefix_lse,  # 传入前缀的log-sum-exp值
        suffix_output,  # 传入后缀注意力输出
        suffix_lse,  # 传入后缀的log-sum-exp值
        prefix_head_stride,  # 传入前缀头步幅
        output_head_stride,  # 传入输出头步幅
        head_size,  # 传入头维度大小
        padded_head_size,  # 传入填充后的头维度大小
        output_lse is not None,  # 传入是否需要计算输出lse的标志
    )


@triton.jit  # Triton即时编译装饰器，将函数编译为GPU内核
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] # 输出张量
    output_lse,  # [NUM_HEADS, NUM_TOKENS] # 输出的log-sum-exp张量
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] # 前缀注意力输出张量
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS] # 前缀的log-sum-exp张量
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] # 后缀注意力输出张量
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS] # 后缀的log-sum-exp张量
    prefix_head_stride,  # 前缀输出在头维度上的步幅
    output_head_stride,  # 输出张量在头维度上的步幅
    HEAD_SIZE: tl.constexpr,  # 编译时常量：注意力头的维度大小
    PADDED_HEAD_SIZE: tl.constexpr,  # 编译时常量：填充后的头维度大小（2的幂次）
    OUTPUT_LSE: tl.constexpr,  # 编译时常量：是否需要输出log-sum-exp值
):
    """Triton GPU内核：合并前缀和后缀的注意力状态。

    每个内核实例处理一个token的一个注意力头，通过log-sum-exp技巧
    对前缀和后缀的注意力输出进行数值稳定的加权合并。
    """
    token_idx = tl.program_id(0)  # 获取当前线程块在第0维（token维度）的索引
    num_tokens = tl.num_programs(0)  # 获取第0维（token维度）的线程块总数
    head_idx = tl.program_id(1)  # 获取当前线程块在第1维（注意力头维度）的索引
    num_heads = tl.num_programs(1)  # 获取第1维（注意力头维度）的线程块总数

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)  # 加载当前token和头对应的前缀log-sum-exp值
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)  # 加载当前token和头对应的后缀log-sum-exp值

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse  # 处理FA2的inf边界情况，统一转换为-inf
    s_lse = float("-inf") if s_lse == float("inf") else s_lse  # 处理FA2的inf边界情况，统一转换为-inf

    max_lse = tl.maximum(p_lse, s_lse)  # 取前缀和后缀lse的最大值，用于数值稳定性
    p_lse = p_lse - max_lse  # 前缀lse减去最大值，防止exp溢出
    s_lse = s_lse - max_lse  # 后缀lse减去最大值，防止exp溢出
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)  # 计算前缀的归一化指数值（sum-exp）
    s_se = tl.exp(s_lse)  # 计算后缀的归一化指数值（sum-exp）
    out_se = p_se + s_se  # 计算总的sum-exp值，用于后续归一化

    if OUTPUT_LSE:  # 如果需要输出合并后的log-sum-exp值
        out_lse = tl.log(out_se) + max_lse  # 计算合并后的lse值：log(sum_exp) + max_lse 还原真实尺度
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)  # 将合并后的lse值写入输出张量

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)  # 生成从0到PADDED_HEAD_SIZE的索引序列，用于向量化加载
    head_mask = head_arange < HEAD_SIZE  # 创建掩码，过滤掉填充部分的无效索引
    p_out = tl.load(  # 加载前缀注意力输出的一个头向量
        prefix_output
        + token_idx * num_heads * prefix_head_stride  # 定位到当前token的起始位置
        + head_idx * prefix_head_stride  # 定位到当前注意力头的起始位置
        + head_arange,  # 加载整个头维度的数据
        mask=head_mask,  # 使用掩码避免越界访问
    )
    s_out = tl.load(  # 加载后缀注意力输出的一个头向量
        suffix_output
        + token_idx * num_heads * prefix_head_stride  # 定位到当前token的起始位置
        + head_idx * prefix_head_stride  # 定位到当前注意力头的起始位置
        + head_arange,  # 加载整个头维度的数据
        mask=head_mask,  # 使用掩码避免越界访问
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se  # 计算前缀的缩放因子（归一化权重）
    s_scale = s_se / out_se  # 计算后缀的缩放因子（归一化权重）
    out = p_out * p_scale + s_out * s_scale  # 按权重加权合并前缀和后缀的注意力输出
    tl.store(  # 将合并后的结果写入输出张量
        output
        + token_idx * num_heads * output_head_stride  # 定位到当前token的起始位置
        + head_idx * output_head_stride  # 定位到当前注意力头的起始位置
        + head_arange,  # 写入整个头维度的数据
        out,  # 要存储的合并结果
        mask=head_mask,  # 使用掩码避免越界写入
    )
