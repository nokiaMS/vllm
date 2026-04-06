# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM 自定义算子模块。

本模块定义了 vLLM 中使用的自定义操作（custom ops），包括分页注意力（Paged Attention）等
核心推理算子的 Python 封装。这些算子通过 torch.ops._C 调用底层 C++/CUDA 实现，
用于高效的大语言模型推理。
"""

from typing import TYPE_CHECKING, Literal  # 导入类型检查标志和字面量类型

import torch  # 导入 PyTorch 深度学习框架

import vllm.envs as envs  # 导入 vLLM 环境变量配置模块
from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.platforms import current_platform  # 导入当前硬件平台信息
from vllm.scalar_type import ScalarType  # 导入标量类型定义
from vllm.utils.flashinfer import (
    flashinfer_quant_nvfp4_8x4_sf_layout,  # 导入 FlashInfer 的 NV FP4 量化缩放因子布局工具
)
from vllm.utils.math_utils import cdiv  # 导入向上取整除法工具函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

current_platform.import_kernels()  # 根据当前平台导入对应的计算内核

# 根据是否处于类型检查模式，选择不同的 register_fake 实现
if TYPE_CHECKING:
    # 类型检查时使用桩函数，避免导入实际实现
    def register_fake(fn):
        return lambda name: fn  # 返回一个接受名称参数并返回原函数的装饰器
else:
    try:
        from torch.library import register_fake  # 尝试导入新版本的 register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake  # 回退到旧版本的 impl_abstract


# 分页注意力算子
def paged_attention_v1(
    out: torch.Tensor,  # 输出张量，存储注意力计算结果
    query: torch.Tensor,  # 查询张量（Q）
    key_cache: torch.Tensor,  # 键缓存张量（K cache）
    value_cache: torch.Tensor,  # 值缓存张量（V cache）
    num_kv_heads: int,  # KV 头的数量（用于分组查询注意力 GQA）
    scale: float,  # 注意力缩放因子，通常为 1/sqrt(head_dim)
    block_tables: torch.Tensor,  # 块表，映射逻辑块到物理块的索引
    seq_lens: torch.Tensor,  # 每个序列的实际长度
    block_size: int,  # 每个分页块的大小（token 数量）
    max_seq_len: int,  # 批次中最大的序列长度
    alibi_slopes: torch.Tensor | None,  # ALiBi 位置编码的斜率，为 None 则不使用
    kv_cache_dtype: str,  # KV 缓存的数据类型（如 "auto", "fp8" 等）
    k_scale: torch.Tensor,  # 键缓存的量化缩放因子
    v_scale: torch.Tensor,  # 值缓存的量化缩放因子
    tp_rank: int = 0,  # 张量并行的当前 rank
    blocksparse_local_blocks: int = 0,  # 块稀疏注意力的本地块数量
    blocksparse_vert_stride: int = 0,  # 块稀疏注意力的垂直步幅
    blocksparse_block_size: int = 64,  # 块稀疏注意力的块大小
    blocksparse_head_sliding_step: int = 0,  # 块稀疏注意力的头部滑动步长
) -> None:
    """分页注意力 V1 算子。

    该函数是分页注意力机制的第一个版本的 Python 封装。
    适用于序列长度较短的场景，直接在一次内核调用中完成注意力计算。
    通过分页机制管理 KV 缓存，减少显存碎片化。
    """
    # 调用底层 C++/CUDA 实现的分页注意力 V1 内核
    torch.ops._C.paged_attention_v1(
        out,  # 输出张量
        query,  # 查询张量
        key_cache,  # 键缓存
        value_cache,  # 值缓存
        num_kv_heads,  # KV 头数量
        scale,  # 缩放因子
        block_tables,  # 块表
        seq_lens,  # 序列长度
        block_size,  # 块大小
        max_seq_len,  # 最大序列长度
        alibi_slopes,  # ALiBi 斜率
        kv_cache_dtype,  # KV 缓存数据类型
        k_scale,  # 键缩放因子
        v_scale,  # 值缩放因子
        tp_rank,  # 张量并行 rank
        blocksparse_local_blocks,  # 块稀疏本地块数
        blocksparse_vert_stride,  # 块稀疏垂直步幅
        blocksparse_block_size,  # 块稀疏块大小
        blocksparse_head_sliding_step,  # 块稀疏头部滑动步长
    )


def paged_attention_v2(
    out: torch.Tensor,  # 输出张量，存储最终注意力计算结果
    exp_sum: torch.Tensor,  # 指数求和张量，用于多轮归约中的数值稳定性
    max_logits: torch.Tensor,  # 最大 logits 张量，用于多轮归约中的数值稳定性
    tmp_out: torch.Tensor,  # 临时输出张量，存储每轮归约的中间结果
    query: torch.Tensor,  # 查询张量（Q）
    key_cache: torch.Tensor,  # 键缓存张量（K cache）
    value_cache: torch.Tensor,  # 值缓存张量（V cache）
    num_kv_heads: int,  # KV 头的数量（用于分组查询注意力 GQA）
    scale: float,  # 注意力缩放因子，通常为 1/sqrt(head_dim)
    block_tables: torch.Tensor,  # 块表，映射逻辑块到物理块的索引
    seq_lens: torch.Tensor,  # 每个序列的实际长度
    block_size: int,  # 每个分页块的大小（token 数量）
    max_seq_len: int,  # 批次中最大的序列长度
    alibi_slopes: torch.Tensor | None,  # ALiBi 位置编码的斜率，为 None 则不使用
    kv_cache_dtype: str,  # KV 缓存的数据类型（如 "auto", "fp8" 等）
    k_scale: torch.Tensor,  # 键缓存的量化缩放因子
    v_scale: torch.Tensor,  # 值缓存的量化缩放因子
    tp_rank: int = 0,  # 张量并行的当前 rank
    blocksparse_local_blocks: int = 0,  # 块稀疏注意力的本地块数量
    blocksparse_vert_stride: int = 0,  # 块稀疏注意力的垂直步幅
    blocksparse_block_size: int = 64,  # 块稀疏注意力的块大小
    blocksparse_head_sliding_step: int = 0,  # 块稀疏注意力的头部滑动步长
) -> None:
    """分页注意力 V2 算子。

    该函数是分页注意力机制的第二个版本的 Python 封装。
    相比 V1，V2 采用两阶段归约策略，适用于序列长度较长的场景。
    第一阶段将序列分成多个分区并行计算，第二阶段对分区结果进行归约合并。
    额外的 exp_sum、max_logits 和 tmp_out 参数用于支持多分区归约的数值稳定计算。
    """
    # 调用底层 C++/CUDA 实现的分页注意力 V2 内核
    torch.ops._C.paged_attention_v2(
        out,  # 输出张量
        exp_sum,  # 指数求和（用于归约）
        max_logits,  # 最大 logits（用于归约）
        tmp_out,  # 临时输出（用于归约）
        query,  # 查询张量
        key_cache,  # 键缓存
        value_cache,  # 值缓存
        num_kv_heads,  # KV 头数量
        scale,  # 缩放因子
        block_tables,  # 块表
        seq_lens,  # 序列长度
        block_size,  # 块大小
        max_seq_len,  # 最大序列长度
        alibi_slopes,  # ALiBi 斜率
        kv_cache_dtype,  # KV 缓存数据类型
        k_scale,  # 键缩放因子
        v_scale,  # 值缩放因子
        tp_rank,  # 张量并行 rank
        blocksparse_local_blocks,  # 块稀疏本地块数
        blocksparse_vert_stride,  # 块稀疏垂直步幅
        blocksparse_block_size,  # 块稀疏块大小
        blocksparse_head_sliding_step,  # 块稀疏头部滑动步长
    )


# ROCm 平台专用的分页注意力实现
def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    mfma_type: str = "fp8" if envs.VLLM_ROCM_FP8_MFMA_PAGE_ATTN else "f16",
) -> None:
    """ROCm 平台的分页注意力计算。
    针对 AMD GPU（ROCm）优化的分页注意力内核，支持 FP8 MFMA 指令。
    与 CUDA 版本类似，但使用 ROCm 专用的底层算子实现。
    支持可选的 FP8 输出缩放和 MFMA 类型配置。
    """
    # 调用 ROCm 专用的 C++ 分页注意力内核
    torch.ops._rocm_C.paged_attention(
        out,  # 输出张量
        exp_sum,  # 指数求和（用于多分区归约）
        max_logits,  # 最大 logits（用于数值稳定性）
        tmp_out,  # 临时输出缓冲区
        query,  # 查询张量
        key_cache,  # 键缓存
        value_cache,  # 值缓存
        num_kv_heads,  # KV 头数量
        scale,  # 注意力缩放因子
        block_tables,  # 块表（映射逻辑块到物理块）
        seq_lens,  # 每个序列的长度
        query_start_loc,  # 查询起始位置（可选）
        block_size,  # 每个块的大小
        max_seq_len,  # 最大序列长度
        alibi_slopes,  # ALiBi 位置编码斜率（可选）
        kv_cache_dtype,  # KV 缓存的数据类型
        k_scale,  # 键的量化缩放因子
        v_scale,  # 值的量化缩放因子
        fp8_out_scale,  # FP8 输出缩放因子（可选）
        mfma_type,  # MFMA 矩阵运算类型（fp8 或 f16）
    )


# MLA（Multi-head Latent Attention）CPU 解码函数
def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """MLA 解码阶段的 KV 缓存注意力计算（CPU 版本）。
    MLA 是一种多头潜在注意力机制，通过低秩压缩减少 KV 缓存的内存占用。
    此函数在 CPU 上执行解码阶段的注意力计算，从分页 KV 缓存中读取数据。
    """
    # 调用底层 C++ 实现的 MLA 解码内核
    torch.ops._C.mla_decode_kvcache(
        out,  # 输出张量
        query,  # 查询张量
        kv_cache,  # KV 缓存（潜在表示）
        scale,  # 注意力缩放因子
        block_tables,  # 块表（逻辑块到物理块的映射）
        seq_lens,  # 每个序列的长度
    )


# 合并注意力状态的操作
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    """合并前缀和后缀的注意力计算结果。
    在前缀缓存（prefix caching）场景中，前缀部分和后缀部分的注意力分别计算，
    然后通过 log-sum-exp (LSE) 值进行数值稳定的加权合并，得到最终的注意力输出。
    这是实现分块注意力计算和前缀共享的关键操作。
    """
    # 调用底层 C++ 内核合并前缀和后缀的注意力状态
    torch.ops._C.merge_attn_states(
        output,  # 合并后的最终输出
        output_lse,  # 合并后的 LSE 值（可选）
        prefix_output,  # 前缀部分的注意力输出
        prefix_lse,  # 前缀部分的 log-sum-exp 值
        suffix_output,  # 后缀部分的注意力输出
        suffix_lse,  # 后缀部分的 log-sum-exp 值
    )


# 将垂直和斜线稀疏索引转换为块级稀疏表示
def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,  # [BATCH, ] 查询序列长度
    kv_seqlens: torch.Tensor,  # [BATCH, ] 键值序列长度
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V] 垂直稀疏索引
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S] 斜线稀疏索引
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将垂直-斜线稀疏注意力模式的索引转换为块级表示。
    用于稀疏注意力机制（如 MInference），将高层稀疏模式描述
    （垂直列索引 + 斜线对角线索引）转换为可被块稀疏内核高效执行的
    块计数、块偏移、列计数和列索引格式。
    """
    batch_size = slash_indexes.size(0)  # 批次大小
    num_heads = slash_indexes.size(1)  # 注意力头数量
    nnz_slash = slash_indexes.size(2)  # 斜线方向非零元素数量
    nnz_vertical = vertical_indexes.size(2)  # 垂直方向非零元素数量
    # 计算块行数（向上取整）
    num_rows = (context_size + block_size_M - 1) // block_size_M

    # 初始化块计数张量：每个块行中包含的非零块数量
    block_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    # 初始化块偏移张量：每个块行中各非零块的列偏移
    block_offset = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    # 初始化列计数张量：每个块行中垂直稀疏列的数量
    column_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    # 初始化列索引张量：每个块行中垂直稀疏列的具体索引
    column_index = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    # 调用底层 C++ 内核进行索引转换
    torch.ops._C.convert_vertical_slash_indexes(
        block_count,  # 输出：块计数
        block_offset,  # 输出：块偏移
        column_count,  # 输出：列计数
        column_index,  # 输出：列索引
        q_seqlens,  # 查询序列长度
        kv_seqlens,  # 键值序列长度
        vertical_indexes,  # 垂直稀疏索引
        slash_indexes,  # 斜线稀疏索引
        context_size,  # 上下文窗口大小
        block_size_M,  # M 维度块大小（行方向）
        block_size_N,  # N 维度块大小（列方向）
        causal,  # 是否使用因果掩码
    )
    # 返回转换后的块级稀疏表示
    return block_count, block_offset, column_count, column_index


# 合并头版本的垂直-斜线稀疏索引转换
def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,  # [BATCH, ] 查询序列长度
    kv_seqlens: torch.Tensor,  # [BATCH, ] 键值序列长度
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V] 垂直稀疏索引
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S] 斜线稀疏索引
    # [N_HEADS] : different head use different number of indices
    vertical_indices_count: torch.Tensor,  # 每个头的垂直索引数量
    slash_indices_count: torch.Tensor,  # 每个头的斜线索引数量
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """合并头版本的垂直-斜线稀疏索引转换。
    与 convert_vertical_slash_indexes 类似，但支持不同注意力头使用不同数量的稀疏索引。
    通过 vertical_indices_count 和 slash_indices_count 指定每个头的实际索引数量，
    允许更灵活的稀疏模式配置。使用 torch.empty 而非 torch.zeros 以提升性能。
    """
    batch_size = slash_indexes.size(0)  # 批次大小
    num_heads = slash_indexes.size(1)  # 注意力头数量
    nnz_slash = slash_indexes.size(2)  # 斜线方向非零元素数量
    nnz_vertical = vertical_indexes.size(2)  # 垂直方向非零元素数量
    # 计算块行数（向上取整）
    num_rows = (context_size + block_size_M - 1) // block_size_M

    # 使用 empty（未初始化）分配块计数张量以提升性能
    block_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    # 分配块偏移张量
    block_offset = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    # 分配列计数张量
    column_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    # 分配列索引张量
    column_index = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    # 调用合并头版本的底层 C++ 内核
    torch.ops._C.convert_vertical_slash_indexes_mergehead(
        block_count,  # 输出：块计数
        block_offset,  # 输出：块偏移
        column_count,  # 输出：列计数
        column_index,  # 输出：列索引
        q_seqlens,  # 查询序列长度
        kv_seqlens,  # 键值序列长度
        vertical_indexes,  # 垂直稀疏索引
        slash_indexes,  # 斜线稀疏索引
        vertical_indices_count,  # 每个头的垂直索引计数
        slash_indices_count,  # 每个头的斜线索引计数
        context_size,  # 上下文窗口大小
        block_size_M,  # M 维度块大小
        block_size_N,  # N 维度块大小
        causal,  # 是否使用因果掩码
    )
    # 返回转换后的块级稀疏表示
    return block_count, block_offset, column_count, column_index


# 位置编码操作
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """旋转位置编码（RoPE）的原地应用。
    将旋转位置编码应用到查询和键张量上，实现相对位置信息的注入。
    RoPE 通过旋转向量空间中的嵌入来编码位置信息，
    使得内积自然地包含相对位置信息。
    支持 GPT-NeoX 和 GPT-J 两种不同的旋转维度排列方式。
    """
    # 调用底层 C++/CUDA 内核应用旋转位置编码
    torch.ops._C.rotary_embedding(
        positions,  # 每个 token 的位置索引
        query,  # 查询张量（原地修改）
        key,  # 键张量（原地修改，可选）
        head_size,  # 每个注意力头的维度大小
        cos_sin_cache,  # 预计算的 cos/sin 缓存表
        is_neox,  # 是否使用 NeoX 风格的维度交错方式
    )


# 层归一化操作
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """RMS 归一化（Root Mean Square Layer Normalization）。
    对输入张量进行 RMS 归一化：out = (input / RMS(input)) * weight，
    其中 RMS(input) = sqrt(mean(input^2) + epsilon)。
    相比 LayerNorm，RMS Norm 省去了均值中心化步骤，计算更高效。
    """
    # 调用底层 C++/CUDA 内核执行 RMS 归一化
    torch.ops._C.rms_norm(
        out,  # 归一化后的输出
        input,  # 输入张量
        weight,  # 归一化权重（可学习参数）
        epsilon,  # 防止除零的小常数
    )


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """融合的残差加法与 RMS 归一化操作。
    将残差连接和 RMS 归一化融合为一个内核操作，减少显存访问次数：
    residual = residual + input（原地更新残差）
    input = RMSNorm(residual) * weight（原地写入归一化结果）
    融合操作避免了中间结果的显存读写，显著提升推理性能。
    """
    # 调用融合的残差加法 + RMS 归一化内核
    torch.ops._C.fused_add_rms_norm(
        input,  # 输入张量（原地写入归一化结果）
        residual,  # 残差张量（原地累加）
        weight,  # 归一化权重
        epsilon,  # 防止除零的小常数
    )


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
) -> None:
    """融合的 QK 归一化与旋转位置编码操作。
    将查询/键的 RMS 归一化和旋转位置编码（RoPE）融合为单个内核，
    减少显存带宽消耗。操作流程：
    1. 对 Q 和 K 分别进行 RMS 归一化
    2. 对归一化后的 Q 和 K 应用旋转位置编码
    所有操作在 QKV 张量上原地执行。
    """
    # 调用融合的 QK 归一化 + RoPE 内核
    torch.ops._C.fused_qk_norm_rope(
        qkv,  # QKV 合并张量（原地修改 Q 和 K 部分）
        num_heads_q,  # 查询头数量
        num_heads_k,  # 键头数量
        num_heads_v,  # 值头数量
        head_dim,  # 每个头的维度
        eps,  # 归一化的 epsilon 参数
        q_weight,  # Q 的归一化权重
        k_weight,  # K 的归一化权重
        cos_sin_cache,  # 预计算的 cos/sin 缓存
        is_neox,  # 是否使用 NeoX 风格的旋转维度排列
        position_ids,  # 位置索引
    )


# 重复惩罚操作：PyTorch 纯算子实现（CPU 和非连续张量的回退方案）
def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """使用 PyTorch 原生算子应用重复惩罚（回退实现）。
    对已出现在提示或输出中的 token 施加惩罚，降低其再次被生成的概率。
    正 logits 除以惩罚值（降低概率），负 logits 乘以惩罚值（使其更负，也降低概率）。
    """
    # 将惩罚值从 [num_seqs] 扩展为 [num_seqs, vocab_size] 以便广播
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    # 若 token 出现在提示或输出中则应用惩罚，否则使用 1.0（不惩罚）
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    # 正 logits 除以惩罚值，负 logits 乘以惩罚值，确保惩罚总是降低概率
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    # 原地应用缩放因子到 logits
    logits *= scaling


# 重复惩罚操作：CUDA 优化实现
def apply_repetition_penalties_cuda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """使用 CUDA 内核应用重复惩罚（高性能实现）。
    功能与 PyTorch 版本相同，但通过自定义 CUDA 内核实现，
    将多个操作融合为单次内核调用，减少显存带宽消耗。
    """
    # 调用底层 C++/CUDA 内核原地应用重复惩罚
    torch.ops._C.apply_repetition_penalties_(
        logits,  # logits 张量（原地修改）
        prompt_mask,  # 提示中出现的 token 掩码
        output_mask,  # 输出中出现的 token 掩码
        repetition_penalties,  # 每个序列的惩罚系数
    )


# 重复惩罚的统一入口函数，自动选择最优实现
def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """原地应用重复惩罚到 logits 上。
    根据张量的设备和内存布局自动选择 CUDA 或 PyTorch 实现。

    Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    # 如果 logits 在 CUDA 上且内存连续，使用高效的 CUDA 内核
    if logits.is_cuda and logits.is_contiguous():
        apply_repetition_penalties_cuda(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        # 否则回退到 PyTorch 纯算子实现（适用于 CPU 或非连续张量）
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


# 融合量化层归一化操作：逐 token 动态量化
# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合 RMS 归一化与逐 token 动态量化。
    将 RMS 归一化和逐 token 量化融合为一个内核调用，
    输出量化后的张量和对应的缩放因子。
    可选地同时执行残差加法（若提供 residual 参数）。
    """
    # 创建与输入同形状的输出张量，数据类型为目标量化类型
    output = torch.empty(input.shape, dtype=quant_dtype, device=input.device)
    # 创建缩放因子张量，每个 token（行）一个缩放值
    scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )

    # 调用底层 C++/CUDA 融合内核执行归一化 + 动态量化
    torch.ops._C.rms_norm_dynamic_per_token_quant(
        output,  # 量化后的输出
        input,  # 输入张量
        weight,  # 归一化权重
        scales,  # 逐 token 缩放因子（输出）
        epsilon,  # 归一化 epsilon
        scale_ub,  # 缩放因子上界（可选，用于 FP8 限制范围）
        residual,  # 残差张量（可选，若提供则融合残差加法）
    )
    # 返回量化输出和缩放因子
    return output, scales


# 融合量化层归一化操作：逐块（block）量化
# fused quant layer norm ops blocked
def rms_norm_per_block_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    group_size: list[int],
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
    tma_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合 RMS 归一化与逐块量化。
    将 RMS 归一化和分块量化融合为一个内核调用。
    每个块（block）共享一个缩放因子，块大小由 group_size 指定。
    支持转置缩放因子布局和 TMA 对齐，适配不同硬件优化需求。
    """
    # 确保 group_size 是二维的 [行方向块大小, 列方向块大小]
    assert len(group_size) == 2
    # 创建与输入同形状的量化输出张量
    output = torch.empty(input.shape, dtype=quant_dtype, device=input.device)
    if is_scale_transposed:
        # 缩放因子需要转置布局的情况
        if tma_alignment == 0:
            # 无 TMA 对齐：创建转置的缩放因子张量
            scales = torch.empty(
                (input.shape[-1] // group_size[1], input.numel() // input.shape[-1]),
                device=input.device,
                dtype=torch.float32,
            ).transpose(0, 1)  # 转置以匹配内核期望的内存布局
        else:
            # 有 TMA 对齐：需要自定义步长以满足 TMA 硬件要求
            m = input.shape[-2]  # 行数
            sf_k = input.shape[-1] // group_size[1]  # 列方向块数
            # 将 m 向上对齐到 tma_alignment 的整数倍
            tma_aligned_m = (m + tma_alignment - 1) // tma_alignment * tma_alignment
            shape = input.shape[:-2] + (m, sf_k)  # 缩放因子的逻辑形状
            # 根据张量维度设置自定义步长，实现转置 + TMA 对齐
            stride = (
                (1, tma_aligned_m)
                if input.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            # 使用自定义步长创建缩放因子张量
            scales = torch.empty_strided(
                shape, stride, device=input.device, dtype=torch.float32
            )
    else:
        # 非转置布局：标准行优先缩放因子
        scales = torch.empty(
            (input.numel() // input.shape[-1], input.shape[-1] // group_size[1]),
            device=input.device,
            dtype=torch.float32,
        )

    # 验证 TMA 对齐参数合法性
    assert tma_alignment in [0, 4], "Expected TMA alignment 0 or 4, but got " + str(
        tma_alignment
    )

    # 调用底层 C++/CUDA 融合内核执行归一化 + 逐块量化
    torch.ops._C.rms_norm_per_block_quant(
        output,  # 量化后的输出
        input,  # 输入张量
        weight,  # 归一化权重
        scales,  # 逐块缩放因子（输出）
        epsilon,  # 归一化 epsilon
        scale_ub,  # 缩放因子上界（可选）
        residual,  # 残差张量（可选）
        group_size[1],  # 列方向块大小
        is_scale_transposed,  # 是否转置缩放因子
    )
    # 返回量化输出和缩放因子
    return output, scales


# 量化操作
# quantization ops
# AWQ（Activation-aware Weight Quantization）量化相关操作
# awq
def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    split_k_iters: int,
    thx: int,
    thy: int,
) -> torch.Tensor:
    """AWQ 反量化：将 AWQ 格式的量化权重还原为浮点权重。
    支持 Triton 和 CUDA 两种后端实现，通过环境变量选择。
    """
    # 若启用 Triton AWQ 后端，使用 Triton 实现
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton,
        )

        return awq_dequantize_triton(qweight, scales, zeros)
    # 否则使用 CUDA 内核实现反量化
    return torch.ops._C.awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy)


# 注册 AWQ 反量化的 FakeTensor 实现（用于 torch.compile 等场景的形状推断）
if hasattr(torch.ops._C, "awq_dequantize"):

    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        split_k_iters: torch.SymInt,
        thx: int,
        thy: int,
    ) -> torch.Tensor:
        """AWQ 反量化的 FakeTensor 实现，仅计算输出形状。
        每个量化权重包含 8 个压缩值，因此输出列数 = 量化列数 * 8。
        """
        in_c = qweight.size(0)  # 输入通道数（行数）
        qout_c = qweight.size(1)  # 量化后的输出通道数
        out_c = qout_c * 8  # 反量化后的实际输出通道数（每个元素解压为 8 个值）
        # 返回反量化后的空张量，形状为 [输入通道, 输出通道]
        return torch.empty((in_c, out_c), dtype=scales.dtype, device=scales.device)


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    """AWQ 量化矩阵乘法：直接在量化权重上执行 GEMM，无需先反量化。
    在内核内部完成反量化和矩阵乘法的融合，避免中间结果的显存开销。
    支持 Triton 和 CUDA 两种后端。
    """
    # 若启用 Triton AWQ 后端，使用 Triton 实现
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton

        return awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)
    # 否则使用 CUDA 内核实现量化 GEMM
    return torch.ops._C.awq_gemm(input, qweight, scales, qzeros, split_k_iters)


# 注册 AWQ GEMM 的 FakeTensor 实现
if hasattr(torch.ops._C, "awq_gemm"):

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        split_k_iters: torch.SymInt,
    ) -> torch.Tensor:
        """AWQ GEMM 的 FakeTensor 实现，仅计算输出形状。
        使用 split-k 策略：先创建 [split_k, M, N] 的中间结果，
        再沿第 0 维求和得到最终 [M, N] 输出。
        """
        num_in_feats = input.size(0)  # 输入特征数（batch 维度）
        # 创建 split-k 中间结果并求和，模拟实际内核的归约行为
        return torch.empty(
            (split_k_iters, num_in_feats, qweight.size(1) * 8),  # 每个量化列解压为 8 列
            dtype=input.dtype,
            device=input.device,
        ).sum(0)  # 沿 split-k 维度求和得到最终输出


# GPTQ（GPT 训练后量化）相关操作
# gptq
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    use_v2_format: bool,
    bit: int,
) -> torch.Tensor:
    """GPTQ 量化矩阵乘法：在 GPTQ 格式量化权重上执行 GEMM。
    支持 ExLlama 内核优化和 V2 格式，可处理不同位宽的量化。
    """
    # 调用 CUDA 内核执行 GPTQ 量化矩阵乘法
    return torch.ops._C.gptq_gemm(
        a,  # 输入激活张量
        b_q_weight,  # GPTQ 量化权重
        b_gptq_qzeros,  # 量化零点
        b_gptq_scales,  # 量化缩放因子
        b_g_idx,  # 分组索引（指定每行属于哪个量化组）
        use_exllama,  # 是否使用 ExLlama 优化内核
        use_v2_format,  # 是否使用 V2 数据格式
        bit,  # 量化位宽（如 4、8）
    )


# 注册 GPTQ GEMM 的 FakeTensor 实现
if hasattr(torch.ops._C, "gptq_gemm"):

    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_gptq_qzeros: torch.Tensor,
        b_gptq_scales: torch.Tensor,
        b_g_idx: torch.Tensor,
        use_exllama: bool,
        use_v2_format: bool,
        bit: int,
    ) -> torch.Tensor:
        """GPTQ GEMM 的 FakeTensor 实现，仅推断输出形状。
        输出形状为 [输入行数, 量化权重列数]。
        """
        # 输出形状：[M, N]，其中 M 来自输入 a，N 来自量化权重 b
        return torch.empty(
            (a.size(0), b_q_weight.size(1)), dtype=a.dtype, device=a.device
        )


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    """GPTQ 权重重排：对量化权重进行内存布局重排以优化访问模式。
    按照指定的排列顺序 q_perm 原地重新排列量化权重，
    使其内存布局更适合高效的矩阵乘法内核。
    """
    # 调用 CUDA 内核原地重排量化权重
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


# AllSpark W8A16 量化 GEMM 的 FakeTensor 注册（8 位权重、16 位激活）
if hasattr(torch.ops._C, "allspark_w8a16_gemm"):

    @register_fake("_C::allspark_w8a16_gemm")
    def _allspark_w8a16_gemm_fake(
        a: torch.Tensor,
        b_qweight: torch.Tensor,
        b_scales: torch.Tensor,
        b_qzeros: torch.Tensor | None,
        n: torch.SymInt,
        group_size: torch.SymInt,
        sm_count: torch.SymInt,
        sm_version: torch.SymInt,
        CUBLAS_M_THRESHOLD: torch.SymInt,
        has_zp: bool,
        n32k16_reorder: bool,
    ) -> torch.Tensor:
        """AllSpark W8A16 GEMM 的 FakeTensor 实现。
        输出形状为 [M, N]，数据类型与输入激活相同。
        """
        m = a.size(0)  # 输入行数（batch 维度）
        # 返回形状为 [M, N] 的空张量
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


# GGML 格式量化操作的 FakeTensor 注册
# GGML 是 llama.cpp 使用的量化格式，支持多种量化类型（Q4_0、Q8_0 等）
if hasattr(torch.ops._C, "ggml_dequantize"):

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """GGML 反量化的 FakeTensor 实现。
        将 GGML 量化权重还原为 float16 格式，输出形状为 [m, n]。
        """
        # 返回反量化后的 float16 张量
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        """GGML 矩阵-向量乘法的 FakeTensor 实现（A8 量化格式）。
        适用于单 token 推理（decode 阶段），输出形状为 [batch, row]。
        """
        # 返回矩阵-向量乘法结果
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        """GGML 矩阵-矩阵乘法的 FakeTensor 实现（A8 量化格式）。
        适用于多 token 推理（prefill 阶段），输出形状为 [batch, row]。
        """
        batch = X.size(0)  # batch 大小
        return torch.empty((batch, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        """GGML MoE（混合专家）矩阵乘法的 FakeTensor 实现。
        输出形状为 [tokens * top_k, row]，每个 token 对应 top_k 个专家的输出。
        """
        tokens = X.size(0)  # token 数量
        # 每个 token 产生 top_k 个专家输出
        return torch.empty((tokens * top_k, row), dtype=torch.float16, device=W.device)


# GGML MoE 向量化版本的 FakeTensor 注册
if hasattr(torch.ops._C, "ggml_moe_a8_vec"):

    @register_fake("_C::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        """GGML MoE 向量化矩阵乘法的 FakeTensor 实现。
        与 ggml_moe_a8 类似，但使用向量化内核优化单 token 场景。
        """
        tokens = X.size(0)  # token 数量
        # 每个 token 产生 top_k 个专家输出
        return torch.empty((tokens * top_k, row), dtype=X.dtype, device=W.device)


# CUTLASS（CUDA Templates for Linear Algebra Subroutines）相关操作
# cutlass
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool:
    """检查当前 CUDA 设备是否支持 CUTLASS FP4 缩放矩阵乘法。
    FP4 量化需要较新的 GPU 架构支持。
    """
    return torch.ops._C.cutlass_scaled_mm_supports_fp4(cuda_device_capability)


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """CUTLASS FP4 缩放矩阵乘法。
    对 FP4 量化的矩阵 a 和 b 执行带分块缩放因子的矩阵乘法，
    结果乘以全局缩放因子 alpha 后输出为指定的数据类型。
    """
    # 确保输入是二维矩阵
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]  # 输出矩阵的行数和列数
    # 创建输出张量
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    # 调用 CUTLASS 内核执行 FP4 缩放矩阵乘法
    torch.ops._C.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    """检查当前 CUDA 设备是否支持 CUTLASS FP8 缩放矩阵乘法。"""
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    """检查当前 CUDA 设备是否支持 CUTLASS 分块 FP8 缩放矩阵乘法。
    分块 FP8 为 DeepSeek V3 等模型使用的量化方案。
    """
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUTLASS 缩放矩阵乘法：融合缩放和矩阵乘法为一个高效内核。
    实现 output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype) 的融合版本。
    缩放因子通过 numpy 风格广播应用，并扩展支持分块（group）广播规则，
    以适配 DeepSeek V3 等模型的分块量化方案。

    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    # 输出类型仅支持 bfloat16 或 float16
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    # 偏置若存在，必须与输出列数匹配且类型一致
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # 将输入展平为 2D 矩阵，记录原始形状用于恢复
    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])  # 目标输出形状
    a = a.view(-1, a.shape[-1])  # 展平前面所有维度

    # 检查 b 的维度是否为 16 的倍数（CUTLASS 内核的对齐要求）
    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    if current_platform.is_rocm() or not cutlass_compatible_b:
        # ROCm 平台或尺寸不兼容时，回退到 Triton 实现
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        # 使用 CUTLASS 内核执行缩放矩阵乘法
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    # 将输出恢复为原始形状
    return out.view(*target_shape)


def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUTLASS 缩放矩阵乘法（带非对称零点校正）。
    在缩放矩阵乘法基础上增加非对称量化零点（AZP）校正，
    用于处理非对称量化（如 INT8 非对称量化）的矩阵乘法。

    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    # b 的维度必须是 16 的倍数（CUTLASS 对齐要求）
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    # 输出类型仅支持 bfloat16 或 float16
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    # 偏置若存在，必须与输出列数匹配
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # 将输入展平为 2D 矩阵
    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])  # 记录目标输出形状
    a = a.view(-1, a.shape[-1])  # 展平前面所有维度
    # 逐 token 模式下，azp 的元素数必须等于行数
    assert azp is None or azp.numel() == a.shape[0]

    # 创建输出张量并调用 CUTLASS 内核
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj, azp, bias)
    # 将输出恢复为原始形状
    return out.view(*target_shape)


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    """检查当前 CUDA 设备是否支持 CUTLASS 稀疏缩放矩阵乘法。"""
    return torch.ops._C.cutlass_sparse_scaled_mm_supported(cuda_device_capability)


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    """检查当前 CUDA 设备是否支持 CUTLASS 分组 GEMM。
    仅 SM90 到 SM109 架构（Hopper 系列）支持分组 GEMM。
    """
    # 仅 SM90-SM109 架构支持
    if cuda_device_capability < 90 or cuda_device_capability >= 110:
        return False
    try:
        return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)
    except AttributeError:
        # 在非 CUDA 平台上该算子不可用，返回 False
        # Return False on non-CUDA platforms where it is not available
        return False


def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CUTLASS 稀疏矩阵压缩：将稠密张量压缩为非零元素和元数据两部分。
    压缩后的表示与 CUTLASS 稀疏内核兼容，用于 2:4 结构化稀疏矩阵乘法。
    输入张量的数据类型必须为 int8、float8_e4m3fn、bfloat16 或 float16。
    返回 (a_nzs, a_meta) 元组，其中 a_nzs 形状为 (m, k//2)，a_meta 形状为 (m, k//2//4)。

    Compresses a sparse matrix for use with Cutlass sparse operations.

    This function takes a dense tensor and compresses it into two components:
    non-zero elements and metadata. The compressed representation is compatible
    with Cutlass sparse kernels.

    Args:
        a (torch.Tensor):
            The input tensor to be compressed. Must have one of the following data types:
            - `torch.int8`
            - `torch.float8_e4m3fn`
            - `torch.bfloat16`
            - `torch.float16`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
            - `a_nzs` (torch.Tensor): A tensor containing non-zero elements of `a`.
            - `a_meta` (torch.Tensor): A tensor containing metadata for the sparse representation.

    Raises:
        ValueError: If the compression operation fails.

    Notes:
        - The `a_meta` tensor has a data type of `torch.uint8`.
        - Each metadata element encodes the sparsity of 4 non-zero elements (i.e., `elemsPerMetaElem = 4`).
        - The shape of `a_nzs` is `(m, k // 2)`, where `m` and `k` are the dimensions of the input tensor.
        - The shape of `a_meta` is `(m, k // 2 // elemsPerMetaElem)`.
    """
    # 检查输入数据类型是否在支持的类型列表中
    assert a.dtype in [torch.int8, torch.float8_e4m3fn, torch.bfloat16, torch.float16]
    # 确保输入张量在内存中是连续的
    assert a.is_contiguous()

    # 元数据类型为 torch.uint8，每个元数据元素编码 4 个非零元素的稀疏模式
    # a_meta.dtype: torch.uint8 so elemsPerMetaElem = 8b / 2b_per_nz = 4
    elemsPerMetaElem = 4
    # 列数必须是 2*elemsPerMetaElem=8 的倍数（结构化稀疏的对齐要求）
    assert a.shape[1] % (2 * elemsPerMetaElem) == 0

    # 调用 CUTLASS 内核执行稀疏压缩
    return torch.ops._C.cutlass_sparse_compress(a)


def cutlass_scaled_sparse_mm(
    a: torch.Tensor,
    bt_nzs: torch.Tensor,
    bt_meta: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUTLASS 缩放稀疏矩阵乘法：使用 2:4 结构化稀疏进行高效矩阵乘法。
    对稠密矩阵 a 与压缩后的稀疏矩阵 b 执行带缩放因子的矩阵乘法。
    使用流程：先将 b 裁剪为 2:4 稀疏 -> 转置并压缩 -> 调用本函数计算。

    Performs a scaled sparse matrix multiplication using Cutlass.

    Steps:
    1. Create a dense matrix `a` of shape (m, k) on the CUDA device:
    `a = torch.randn((m, k), device='cuda')`.

    2. Create a dense matrix `b` of shape (k, n) on the CUDA device:
    `b = torch.randn((k, n), device='cuda')`.

    3. Prune matrix `b` to 2:4 sparsity along the specified dimension:
    `b = prune_to_2_4(b, dim=0)`.

    4. Compress the transposed sparse matrix `b.t()`:
    `bt_nzs, bt_meta = cutlass_sparse_compress(b.t())`.

    5. Perform sparse matrix multiplication using the compressed matrix,
    applying scaling factors for `a` and `b`, and the output data type:
    `out = cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype)`.

    Returns:
    - The result of the scaled sparse matrix multiplication.
    """
    # 压缩后的非零元素张量维度必须是 16 的倍数（CUTLASS 对齐要求）
    assert bt_nzs.shape[0] % 16 == 0 and bt_nzs.shape[1] % 16 == 0
    # 输出类型仅支持 bfloat16 或 float16
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    # 偏置若存在，其元素数必须与输出列数匹配且类型一致
    assert bias is None or bias.shape[0] == bt_nzs.shape[0] and bias.dtype == out_dtype

    m = a.shape[0]  # 输出行数（token 数）
    n = bt_nzs.shape[0]  # 输出列数（压缩矩阵的行数即原矩阵的列数）
    # 创建输出张量
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    # 调用 CUTLASS 稀疏矩阵乘法内核
    torch.ops._C.cutlass_scaled_sparse_mm(
        out, a, bt_nzs, bt_meta, scale_a, scale_b, bias
    )

    return out


def get_cutlass_moe_mm_data(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    blockscale_offsets: torch.Tensor | None = None,
):
    """准备 CUTLASS 分组矩阵乘法所需的数据，用于基于 CUTLASS 的融合 MoE。
    根据 topk_ids（token-专家映射）计算专家偏移量、问题规模、输入/输出置换等。
    这些数据用于后续的 cutlass_moe_mm 分组矩阵乘法调用。

    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    - blockscale_offsets: Optional argument passed for fp4 moe. Indices that
                          mark at which block scale index each expert begins
                          its computation. The number of block scale rows
                          computed with expert E is blockscale_offsets[E + 1] -
                          blockscale_offsets[E]
    """
    # 调用底层 C++ 算子计算 MoE 分组矩阵乘法所需的排列和偏移数据
    return torch.ops._C.get_cutlass_moe_mm_data(
        topk_ids,  # token 到专家的映射索引
        expert_offsets,  # 每个专家的起始偏移量（输出）
        problem_sizes1,  # 第一组矩阵乘法的 MxNxK 规模（输出）
        problem_sizes2,  # 第二组矩阵乘法的 MxNxK 规模（输出）
        input_permutation,  # 输入置换排列（输出）
        output_permutation,  # 输出置换排列（输出）
        num_experts,  # 专家总数
        n,  # 矩阵乘法的 N 维度
        k,  # 矩阵乘法的 K 维度
        blockscale_offsets,  # FP4 MoE 的分块缩放偏移量（可选）
    )


def get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    expert_first_token_offset: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    n: int,
    k: int,
    swap_ab: bool,
):
    """根据专家首 token 偏移量计算每个专家的 (M, N, K) 问题规模。
    当已知每个专家处理的 token 数量时，直接从偏移量推导矩阵乘法维度，
    避免重新计算排列信息。swap_ab 控制是否交换 A/B 矩阵的角色。

    Compute per-expert (M, N, K) problem sizes from expert_first_token_offset"""
    # 调用底层 C++ 算子从专家偏移量推导问题规模
    return torch.ops._C.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
        expert_first_token_offset,  # 每个专家的首 token 偏移量
        problem_sizes1,  # 第一组矩阵乘法的 MxNxK 规模（输出）
        problem_sizes2,  # 第二组矩阵乘法的 MxNxK 规模（输出）
        n,  # 矩阵乘法的 N 维度
        k,  # 矩阵乘法的 K 维度
        swap_ab,  # 是否交换 A/B 矩阵
    )


def shuffle_rows(input_tensor: torch.Tensor, dst2src_map: torch.Tensor):
    """按行重排输入张量：根据 dst2src_map 映射对输入进行行置换和扩展。
    在 MoE 中用于在执行分组矩阵乘法之前对输入 token 进行重新排列，
    使得同一专家处理的 token 在内存中连续排列。

    Shuffle and expand the input tensor according to the dst2src_map and store the result in output_tensor.
    This is used in MoE to permute the input tensor before performing grouped matrix multiplications.
    """
    # 置换后的 token 数量（可能大于原始数量，因为每个 token 可能被多个专家选中）
    num_tokens_permuted = dst2src_map.shape[0]
    # 创建输出张量，行数为置换后的 token 数，列数与输入一致
    output_tensor = torch.empty(
        (num_tokens_permuted, input_tensor.shape[1]),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    # 调用 MoE 专用的行重排内核
    torch.ops._moe_C.shuffle_rows(input_tensor, dst2src_map, output_tensor)
    return output_tensor


def get_cutlass_batched_moe_mm_data(
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    num_local_experts: int,
    padded_m: int,
    n: int,
    k: int,
):
    """准备 CUTLASS 批量 MoE 分组矩阵乘法所需的数据。
    根据每个专家的 token 计数计算专家偏移量和问题规模，
    适用于批量（batched）模式的 MoE，其中每个专家的 token 数量已预填充到 padded_m。

    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in expert_num_tokens (token count per expert) and
    non_zero_expert_idxs (consecutive indices of experts with non-zero token
    counts) and uses them to compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation.
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    """
    # 调用底层 C++ 算子计算批量 MoE 的偏移量和问题规模
    return torch.ops._C.get_cutlass_batched_moe_mm_data(
        expert_offsets,  # 每个专家的起始偏移量（输出）
        problem_sizes1,  # 第一组矩阵乘法的 MxNxK 规模（输出）
        problem_sizes2,  # 第二组矩阵乘法的 MxNxK 规模（输出）
        expert_num_tokens,  # 每个专家分配的 token 数量
        num_local_experts,  # 本地专家数量
        padded_m,  # 填充后的 M 维度（每个专家的最大 token 数）
        n,  # 矩阵乘法的 N 维度
        k,  # 矩阵乘法的 K 维度
    )


def cutlass_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
):
    """CUTLASS MoE 分组矩阵乘法：执行 FP8 量化的 OUT = A * B 分组矩阵乘法。
    每个专家独立执行矩阵乘法，专家偏移量和问题规模由 get_cutlass_moe_mm_data 预先计算。
    支持逐 token 缩放（per_act_token）和逐输出通道缩放（per_out_ch）两种量化模式。

    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    # 调用 CUTLASS 分组矩阵乘法内核
    return torch.ops._C.cutlass_moe_mm(
        out_tensors,  # 输出张量
        a_tensors,  # 输入激活张量（FP8 量化）
        b_tensors,  # 专家权重张量（FP8 量化）
        a_scales,  # 输入激活的缩放因子
        b_scales,  # 权重的缩放因子
        expert_offsets,  # 每个专家的起始偏移量
        problem_sizes,  # 每个专家的 MxNxK 问题规模
        a_strides,  # A 矩阵的步长
        b_strides,  # B 矩阵的步长
        c_strides,  # 输出矩阵 C 的步长
        per_act_token,  # 是否逐 token 缩放激活
        per_out_ch,  # 是否逐输出通道缩放权重
    )


def cutlass_fp4_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    alphas: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    sf_offsets: torch.Tensor,
):
    """CUTLASS FP4 MoE 分块缩放分组矩阵乘法。
    对 NVFP4 量化的输入和专家权重执行分组 GEMM，用于 NVFP4 量化融合 MoE 的前向计算。
    缩放因子使用 FP8-E4M3 精度的分块缩放，通过 sf_offsets 管理每个专家的缩放因子位置。

    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    # 调用 CUTLASS FP4 分组矩阵乘法内核
    return torch.ops._C.cutlass_fp4_group_mm(
        out_tensors,  # 输出张量
        a_tensors,  # NVFP4 量化的输入激活
        b_tensors,  # NVFP4 量化的专家权重
        a_scales,  # 输入的 FP8-E4M3 分块缩放因子
        b_scales,  # 权重的 FP8-E4M3 分块缩放因子
        alphas,  # 全局缩放因子
        problem_sizes,  # 每个专家的 MxNxK 问题规模
        expert_offsets,  # 每个专家的 token 起始偏移量
        sf_offsets,  # 每个专家的缩放因子起始偏移量
    )


def mxfp8_experts_quant(
    input_tensor: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    """MXFP8 专家量化：对 MoE 中每个专家的输入进行 MXFP8 量化。
    根据专家偏移量和问题规模，对各专家对应的输入片段分别执行量化，
    输出量化后的张量和对应的分块缩放因子。
    """
    # 调用底层 C++ 算子执行按专家分组的 MXFP8 量化
    torch.ops._C.mxfp8_experts_quant(
        input_tensor,  # 输入张量（待量化）
        problem_sizes,  # 每个专家的问题规模
        expert_offsets,  # 每个专家的 token 起始偏移量
        blockscale_offsets,  # 每个专家的分块缩放因子偏移量
        quant_output,  # 量化后的输出张量（原地写入）
        scale_factor,  # 缩放因子张量（原地写入）
    )


def cutlass_mxfp8_grouped_mm(
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_tensors: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> None:
    """CUTLASS MXFP8 分组矩阵乘法：对 MXFP8 量化的矩阵执行分组 GEMM。
    使用微缩放 FP8（MXFP8）格式进行高效的分组矩阵乘法，
    每个专家根据其偏移量和分块缩放因子独立计算。
    """
    # 调用 CUTLASS MXFP8 分组矩阵乘法内核
    torch.ops._C.cutlass_mxfp8_grouped_mm(
        a_tensors,  # MXFP8 量化的输入激活
        b_tensors,  # MXFP8 量化的专家权重
        a_scales,  # 输入的分块缩放因子
        b_scales,  # 权重的分块缩放因子
        out_tensors,  # 输出张量（原地写入）
        problem_sizes,  # 每个专家的 MxNxK 问题规模
        expert_offsets,  # 每个专家的 token 起始偏移量
        blockscale_offsets,  # 每个专家的分块缩放因子偏移量
    )


# 注册 mxfp8_experts_quant 的 FakeTensor 实现（用于 torch.compile 图追踪）
if hasattr(torch.ops._C, "mxfp8_experts_quant"):

    @register_fake("_C::mxfp8_experts_quant")
    def _mxfp8_experts_quant_fake(
        input_tensor: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        quant_output: torch.Tensor,
        scale_factor: torch.Tensor,
    ) -> None:
        """MXFP8 专家量化的 FakeTensor 实现。
        该函数原地修改 quant_output 和 scale_factor，因此返回 None。
        """
        return None


# 注册 cutlass_mxfp8_grouped_mm 的 FakeTensor 实现（用于 torch.compile 图追踪）
if hasattr(torch.ops._C, "cutlass_mxfp8_grouped_mm"):

    @register_fake("_C::cutlass_mxfp8_grouped_mm")
    def _cutlass_mxfp8_grouped_mm_fake(
        a_tensors: torch.Tensor,
        b_tensors: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        out_tensors: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
    ) -> None:
        """CUTLASS MXFP8 分组矩阵乘法的 FakeTensor 实现。
        该函数原地修改 out_tensors，因此返回 None。
        """
        return None


# gptq_marlin
# GPTQ Marlin 量化权重重新打包
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    """将 GPTQ 量化权重重新打包为 Marlin 内核所需的格式。

    参数:
        b_q_weight: GPTQ 量化后的权重张量
        perm: 列置换索引张量
        size_k: 权重矩阵的 K 维度大小
        size_n: 权重矩阵的 N 维度大小
        num_bits: 量化位数
        is_a_8bit: 激活值是否为 8 位量化

    返回:
        重新打包后的权重张量，适用于 Marlin 内核
    """
    # 调用底层 C++ 扩展执行 GPTQ Marlin 重打包操作
    return torch.ops._C.gptq_marlin_repack(
        b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit
    )


# 检查底层 C++ 扩展是否支持 gptq_marlin_repack 操作
if hasattr(torch.ops._C, "gptq_marlin_repack"):

    @register_fake("_C::gptq_marlin_repack")
    def _gptq_marlin_repack_fake(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: torch.SymInt,
        size_n: torch.SymInt,
        num_bits: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        """GPTQ Marlin 重打包的 FakeTensor 实现，用于图编译时推断输出形状。"""
        # 计算每个 32 位整数中打包的量化值数量
        pack_factor = 32 // num_bits
        # Marlin 内核的 tile 大小为 16
        marlin_tile_size = 16
        # 返回与重打包后形状一致的空张量
        return torch.empty(
            (size_k // marlin_tile_size, size_n * marlin_tile_size // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


# awq_marlin
# AWQ Marlin 量化权重重新打包
def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    """将 AWQ 量化权重重新打包为 Marlin 内核所需的格式。

    参数:
        b_q_weight: AWQ 量化后的权重张量
        size_k: 权重矩阵的 K 维度大小
        size_n: 权重矩阵的 N 维度大小
        num_bits: 量化位数
        is_a_8bit: 激活值是否为 8 位量化

    返回:
        重新打包后的权重张量，适用于 Marlin 内核
    """
    # 调用底层 C++ 扩展执行 AWQ Marlin 重打包操作
    return torch.ops._C.awq_marlin_repack(
        b_q_weight, size_k, size_n, num_bits, is_a_8bit
    )


# 检查底层 C++ 扩展是否支持 awq_marlin_repack 操作
if hasattr(torch.ops._C, "awq_marlin_repack"):

    @register_fake("_C::awq_marlin_repack")
    def _awq_marlin_repack_fake(
        b_q_weight: torch.Tensor,
        size_k: torch.SymInt,
        size_n: torch.SymInt,
        num_bits: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        """AWQ Marlin 重打包的 FakeTensor 实现，用于图编译时推断输出形状。"""
        # 计算每个 32 位整数中打包的量化值数量
        pack_factor = 32 // num_bits
        # Marlin 内核的 tile 大小为 16
        marlin_tile_size = 16
        # 返回与重打包后形状一致的空张量
        return torch.empty(
            (size_k // marlin_tile_size, size_n * marlin_tile_size // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    """将 GPTQ 量化的混合专家（MoE）权重重新打包为 Marlin 内核格式。

    对每个专家的权重分别调用 gptq_marlin_repack 进行重打包。

    参数:
        b_q_weight: 所有专家的 GPTQ 量化权重张量，第一维为专家数
        perm: 每个专家对应的列置换索引
        size_k: 权重矩阵的 K 维度大小
        size_n: 权重矩阵的 N 维度大小
        num_bits: 量化位数
        is_a_8bit: 激活值是否为 8 位量化

    返回:
        所有专家重新打包后的权重张量
    """
    # 获取专家数量
    num_experts = b_q_weight.shape[0]
    # K 维度必须是 16 的倍数（Marlin tile 大小要求）
    assert size_k % 16 == 0
    # 创建输出张量，形状为 (专家数, size_k//16, size_n*(num_bits//2))
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    # 逐个专家进行重打包
    for e in range(num_experts):
        output[e] = torch.ops._C.gptq_marlin_repack(
            b_q_weight[e], perm[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    """将 AWQ 量化的混合专家（MoE）权重重新打包为 Marlin 内核格式。

    对每个专家的权重分别调用 awq_marlin_repack 进行重打包。

    参数:
        b_q_weight: 所有专家的 AWQ 量化权重张量，第一维为专家数
        perm: 每个专家对应的列置换索引（AWQ 中未使用）
        size_k: 权重矩阵的 K 维度大小
        size_n: 权重矩阵的 N 维度大小
        num_bits: 量化位数
        is_a_8bit: 激活值是否为 8 位量化

    返回:
        所有专家重新打包后的权重张量
    """
    # 获取专家数量
    num_experts = b_q_weight.shape[0]
    # K 维度必须是 16 的倍数（Marlin tile 大小要求）
    assert size_k % 16 == 0
    # 创建输出张量，形状为 (专家数, size_k//16, size_n*(num_bits//2))
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    # 逐个专家进行重打包
    for e in range(num_experts):
        output[e] = torch.ops._C.awq_marlin_repack(
            b_q_weight[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def marlin_int4_fp8_preprocess(
    qweight: torch.Tensor,
    qzeros_or_none: torch.Tensor | None = None,
    inplace: bool = False,
):
    """对 INT4/FP8 量化权重进行 Marlin 内核所需的预处理。

    参数:
        qweight: 量化后的权重张量
        qzeros_or_none: 量化零点张量，可选
        inplace: 是否原地修改

    返回:
        预处理后的权重张量
    """
    # 调用底层 C++ 扩展进行 INT4/FP8 预处理
    return torch.ops._C.marlin_int4_fp8_preprocess(qweight, qzeros_or_none, inplace)


def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """执行 Marlin 量化矩阵乘法（GEMM）。

    使用 Marlin 内核对量化权重执行高效的矩阵乘法运算。

    参数:
        a: 输入激活张量
        c: 可选的输出缓冲张量
        b_q_weight: 量化后的权重张量（已经过 Marlin 重打包）
        b_bias: 可选的偏置张量
        b_scales: 权重量化缩放因子
        a_scales: 可选的激活量化缩放因子
        global_scale: 可选的全局缩放因子
        b_zeros: 可选的量化零点
        g_idx: 可选的分组索引（用于分组量化）
        perm: 可选的列置换索引
        workspace: Marlin 内核工作空间张量
        b_q_type: 量化权重的标量类型
        size_m: 矩阵 M 维度大小
        size_n: 矩阵 N 维度大小
        size_k: 矩阵 K 维度大小
        is_k_full: K 维度是否完整（非分组量化时为 True）
        use_atomic_add: 是否使用原子加法
        use_fp32_reduce: 是否使用 FP32 进行归约
        is_zp_float: 零点是否为浮点类型

    返回:
        矩阵乘法结果张量
    """
    # 调用底层 C++ 扩展执行 Marlin GEMM 运算
    return torch.ops._C.marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,  # 传递标量类型 ID
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


# 检查底层 C++ 扩展是否支持 marlin_gemm 操作
if hasattr(torch.ops._C, "marlin_gemm"):

    @register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(
        a: torch.Tensor,
        c: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        b_q_type_id: int,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
    ) -> torch.Tensor:
        """Marlin GEMM 的 FakeTensor 实现，用于图编译时推断输出形状和数据类型。"""
        # 确定输出数据类型：如果输入不是 half 或 bfloat16，则使用缩放因子的类型
        dtype = a.dtype
        if dtype not in [torch.half, torch.bfloat16]:
            dtype = b_scales.dtype
        # 返回形状为 (M, N) 的空张量
        return torch.empty((size_m, size_n), device=a.device, dtype=dtype)


# machete
# Machete 量化矩阵乘法内核
def machete_supported_schedules(
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
    group_zeros_type: torch.dtype | None = None,
    channel_scales_type: torch.dtype | None = None,
    token_scales_type: torch.dtype | None = None,
    out_type: torch.dtype | None = None,
) -> list[str]:
    """查询 Machete 内核支持的调度方案列表。

    根据指定的数据类型组合，返回可用的内核调度方案名称。

    参数:
        a_type: 激活张量的数据类型
        b_type: 量化权重的标量类型
        group_scales_type: 分组缩放因子的数据类型
        group_zeros_type: 分组零点的数据类型
        channel_scales_type: 通道缩放因子的数据类型
        token_scales_type: 令牌缩放因子的数据类型
        out_type: 输出张量的数据类型

    返回:
        支持的调度方案名称列表
    """
    # 调用底层 C++ 扩展查询支持的调度方案
    return torch.ops._C.machete_supported_schedules(
        a_type,
        b_type.id,  # 传递标量类型 ID
        group_scales_type,
        group_zeros_type,
        channel_scales_type,
        token_scales_type,
        out_type,
    )


def machete_mm(
    a: torch.Tensor,
    # b_q Should be the tensor returned by machete_prepack_B
    # b_q 应该是由 machete_prepack_B 返回的预打包张量
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: torch.dtype | None = None,
    b_group_scales: torch.Tensor | None = None,
    b_group_zeros: torch.Tensor | None = None,
    b_group_size: int | None = None,
    b_channel_scales: torch.Tensor | None = None,
    a_token_scales: torch.Tensor | None = None,
    schedule: str | None = None,
) -> torch.Tensor:
    """使用 Machete 内核执行量化矩阵乘法。

    参数:
        a: 输入激活张量
        b_q: 预打包的量化权重张量（由 machete_prepack_B 生成）
        b_type: 量化权重的标量类型
        out_type: 可选的输出数据类型
        b_group_scales: 可选的分组缩放因子
        b_group_zeros: 可选的分组零点
        b_group_size: 可选的分组大小
        b_channel_scales: 可选的通道缩放因子
        a_token_scales: 可选的令牌缩放因子
        schedule: 可选的调度方案名称

    返回:
        矩阵乘法结果张量
    """
    # 调用底层 C++ 扩展执行 Machete 矩阵乘法
    return torch.ops._C.machete_mm(
        a,
        b_q,
        b_type.id,  # 传递标量类型 ID
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )


# 检查底层 C++ 扩展是否支持 machete_mm 操作
if hasattr(torch.ops._C, "machete_mm"):

    @register_fake("_C::machete_mm")
    def machete_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        # b_q 应该是由 machete_prepack_B 返回的预打包张量
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: torch.dtype | None = None,
        b_group_scales: torch.Tensor | None = None,
        b_group_zeros: torch.Tensor | None = None,
        b_group_size: int | None = None,
        b_channel_scales: torch.Tensor | None = None,
        a_token_scales: torch.Tensor | None = None,
        schedule: str | None = None,
    ) -> torch.Tensor:
        """Machete 矩阵乘法的 FakeTensor 实现，用于图编译时推断输出形状。"""
        # 获取输出矩阵的 M 维度（来自激活张量）
        m = a.size(0)
        # 获取输出矩阵的 N 维度（来自权重张量）
        n = b_q.size(1)
        # 返回形状为 (M, N) 的空张量
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


def machete_prepack_B(
    b_q_weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
) -> torch.Tensor:
    """将量化权重预打包为 Machete 内核所需的布局格式。

    参数:
        b_q_weight: 量化后的权重张量
        a_type: 激活张量的数据类型
        b_type: 量化权重的标量类型
        group_scales_type: 分组缩放因子的数据类型

    返回:
        预打包后的权重张量
    """
    # 调用底层 C++ 扩展执行权重预打包
    return torch.ops._C.machete_prepack_B(
        b_q_weight, a_type, b_type.id, group_scales_type
    )


# 检查底层 C++ 扩展是否支持 machete_prepack_B 操作
if hasattr(torch.ops._C, "machete_prepack_B"):

    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(
        b_q_weight: torch.Tensor,
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: torch.dtype | None,
    ) -> torch.Tensor:
        """Machete 权重预打包的 FakeTensor 实现，返回与输入形状相同的空张量。"""
        # 返回与输入权重形状相同、连续内存格式的空张量
        return torch.empty_like(b_q_weight, memory_format=torch.contiguous_format)


# CUTLASS W4A8
# CUTLASS W4A8 量化矩阵乘法（4位权重、8位激活）
def cutlass_w4a8_mm(
    a: torch.Tensor,
    # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
    # b_q 应该是由 cutlass_encode_and_reorder_int4b 返回的编码重排后的张量
    b_q: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    b_channel_scales: torch.Tensor,
    a_token_scales: torch.Tensor,
    out_type: torch.dtype | None = None,
    maybe_schedule: str | None = None,
) -> torch.Tensor:
    """执行 CUTLASS W4A8 量化矩阵乘法。

    使用 4 位量化权重和 8 位量化激活进行高效矩阵乘法。

    参数:
        a: 输入激活张量
        b_q: 编码并重排后的 INT4 权重张量
        b_group_scales: 分组缩放因子
        b_group_size: 每个缩放因子对应的元素分组大小
        b_channel_scales: 通道缩放因子
        a_token_scales: 令牌级缩放因子
        out_type: 可选的输出数据类型
        maybe_schedule: 可选的内核调度方案

    返回:
        矩阵乘法结果张量
    """
    # 调用底层 C++ 扩展执行 W4A8 矩阵乘法
    return torch.ops._C.cutlass_w4a8_mm(
        a,
        b_q,
        b_group_scales,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        out_type,
        maybe_schedule,
    )


# 检查底层 C++ 扩展是否支持 cutlass_w4a8_mm 操作
if hasattr(torch.ops._C, "cutlass_w4a8_mm"):

    @register_fake("_C::cutlass_w4a8_mm")
    def cutlass_w4a8_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
        # b_q 应该是由 cutlass_encode_and_reorder_int4b 返回的编码重排后的张量
        b_q: torch.Tensor,
        b_group_scales: torch.Tensor,
        b_group_size: int,
        b_channel_scales: torch.Tensor,
        a_token_scales: torch.Tensor,
        out_type: torch.dtype | None = None,
        maybe_schedule: str | None = None,
    ) -> torch.Tensor:
        """CUTLASS W4A8 矩阵乘法的 FakeTensor 实现，用于图编译时推断输出形状。"""
        # 获取输出矩阵的 M 维度
        m = a.size(0)
        # 获取输出矩阵的 N 维度
        n = b_q.size(1)
        # 确定输出数据类型，默认使用 bfloat16
        out_dtype = out_type if out_type is not None else torch.bfloat16
        # 返回形状为 (M, N) 的空张量
        return torch.empty((m, n), device=a.device, dtype=out_dtype)


def cutlass_pack_scale_fp8(scales: torch.Tensor) -> torch.Tensor:
    """将 FP8 缩放因子打包为 CUTLASS 内核所需的格式。

    参数:
        scales: FP8 缩放因子张量

    返回:
        打包后的缩放因子张量
    """
    # 调用底层 C++ 扩展执行 FP8 缩放因子打包
    return torch.ops._C.cutlass_pack_scale_fp8(scales)


# 检查底层 C++ 扩展是否支持 cutlass_pack_scale_fp8 操作
if hasattr(torch.ops._C, "cutlass_pack_scale_fp8"):

    @register_fake("_C::cutlass_pack_scale_fp8")
    def cutlass_pack_scale_fp8_fake(scales: torch.Tensor) -> torch.Tensor:
        """FP8 缩放因子打包的 FakeTensor 实现。"""
        # 返回与输入形状相同、连续内存格式的空张量
        return torch.empty_like(scales, memory_format=torch.contiguous_format)


def cutlass_encode_and_reorder_int4b(b: torch.Tensor) -> torch.Tensor:
    """对 INT4 权重进行编码和重排，转换为 CUTLASS 内核所需的布局。

    参数:
        b: INT4 权重张量

    返回:
        编码并重排后的权重张量
    """
    # 调用底层 C++ 扩展执行 INT4 编码和重排
    return torch.ops._C.cutlass_encode_and_reorder_int4b(b)


# 检查底层 C++ 扩展是否支持 cutlass_encode_and_reorder_int4b 操作
if hasattr(torch.ops._C, "cutlass_encode_and_reorder_int4b"):

    @register_fake("_C::cutlass_encode_and_reorder_int4b")
    def cutlass_encode_and_reorder_int4b_fake(b: torch.Tensor) -> torch.Tensor:
        """INT4 编码重排的 FakeTensor 实现。"""
        # 返回与输入形状相同、连续内存格式的空张量
        return torch.empty_like(b, memory_format=torch.contiguous_format)


def cutlass_w4a8_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    group_scale_strides: torch.Tensor,
    maybe_schedule: str | None = None,
):
    """执行基于 CUTLASS 的 W4A8 量化方案的融合 MoE 分组矩阵乘法。
    使用分组量化（INT4 -> FP8）并在尾声中应用逐通道和逐令牌缩放。

    Executes the CUTLASS-based fused-MoE grouped matrix multiplication for the
    W4A8 quantization scheme. Uses group-wise quantization (INT4 -> FP8)
    and both per-channel + per-token scaling in the epilogue.

    Args:
        out_tensors:
            Output buffer for all experts (updated in-place).
            所有专家的输出缓冲区（原地更新）。
        a_tensors:
            FP8 (E4M3FN) activations for all experts.
            所有专家的 FP8 (E4M3FN) 激活张量。
        b_tensors:
            INT4-packed weight matrix for all experts, packed to INT32
            所有专家的 INT4 打包权重矩阵，打包为 INT32。
        a_scales:
            Per-token FP8 activation scales, applied in the epilogue.
            逐令牌 FP8 激活缩放因子，在尾声中应用。
        b_scales:
            Per-channel FP8 weight scales for each expert, applied in the epilogue.
            每个专家的逐通道 FP8 权重缩放因子，在尾声中应用。
        b_group_scales:
            FP8 scale values for group-wise INT4 weight blocks.
            分组 INT4 权重块的 FP8 缩放值。
        b_group_size:
            Number of elements grouped under each entry of b_group_scales.
            每个 b_group_scales 条目对应的元素分组大小。
        expert_offsets:
            Cumulative token offsets
            累积令牌偏移量。
        problem_sizes:
            Per-expert (M, N, K) GEMM sizes used by the grouped GEMM launcher.
            每个专家的 (M, N, K) GEMM 尺寸，供分组 GEMM 启动器使用。
        a/b/c/group_scale_strides:
            Strides describing the memory layout of the input tensors.
            描述输入张量内存布局的步幅。
        maybe_schedule:
            Optional override to choose a specific kernel or epilogue schedule.
            可选的内核或尾声调度方案覆盖。

    Returns:
        out_tensors updated in-place with the dequantized INT4xFP8 grouped GEMM result.
        原地更新的 out_tensors，包含反量化的 INT4xFP8 分组 GEMM 结果。
    """
    # 调用底层 C++ 扩展执行 W4A8 MoE 分组矩阵乘法
    return torch.ops._C.cutlass_w4a8_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        b_group_scales,
        b_group_size,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        group_scale_strides,
        maybe_schedule,
    )


def cutlass_encode_and_reorder_int4b_grouped(
    b_tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对分组的 INT4 权重进行编码和重排，适用于 MoE 场景。

    参数:
        b_tensors: 分组的 INT4 权重张量

    返回:
        编码并重排后的权重张量元组
    """
    # 调用底层 C++ 扩展执行分组 INT4 编码和重排
    return torch.ops._C.cutlass_encode_and_reorder_int4b_grouped(b_tensors)


# 检查底层 C++ 扩展是否支持 cutlass_encode_and_reorder_int4b_grouped 操作
if hasattr(torch.ops._C, "cutlass_encode_and_reorder_int4b_grouped"):

    @register_fake("_C::cutlass_encode_and_reorder_int4b_grouped")
    def cutlass_encode_and_reorder_int4b_grouped_fake(b: torch.Tensor) -> torch.Tensor:
        """分组 INT4 编码重排的 FakeTensor 实现。"""
        # 返回与输入形状相同、连续内存格式的空张量
        return torch.empty_like(b, memory_format=torch.contiguous_format)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """按照给定的置换索引重新排列张量的列。

    参数:
        a: 输入张量
        perm: 列置换索引张量

    返回:
        列重新排列后的张量
    """
    # 调用底层 C++ 扩展执行列置换操作
    return torch.ops._C.permute_cols(a, perm)


# 检查底层 C++ 扩展是否支持 permute_cols 操作
if hasattr(torch.ops._C, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """列置换的 FakeTensor 实现，返回与输入形状相同的空张量。"""
        return torch.empty_like(a)


# fp4
# FP4 量化操作
def scaled_fp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """将输入张量量化为 FP4 格式，并返回量化后的张量和缩放因子。

    该函数对输入张量的最后一维进行量化。每 16 个连续元素共享一个动态计算的
    缩放因子。该缩放因子使用 input_global_scale 进行量化，并以交织布局存储。

    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        use_8x4_sf_layout: Whether to use the 8x4 or 128x4 layout for the scaling

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    # 确保不在 ROCm 平台上运行（仅支持 CUDA）
    assert not current_platform.is_rocm()
    # 输入张量至少为 1 维
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    # 将输入重塑为 2D 张量，保留最后一维
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    # 获取 M（行数）和 N（列数）维度
    m, n = input.shape
    # 每 16 个元素一组共享一个缩放因子
    block_size = 16
    device = input.device

    # 最后一维必须是 16 的倍数
    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    # 输入数据类型必须是 fp16 或 bf16
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )

    # 当使用 TensorRT-LLM 后端且行数不超过 32 时，使用 8x4 缩放因子布局
    use_8x4_sf_layout = True if "trtllm" in backend and m <= 32 else False  # noqa: SIM210

    if use_8x4_sf_layout:
        # 使用 flashinfer 的 8x4 缩放因子布局进行 NVFP4 量化
        output, output_scale = flashinfer_quant_nvfp4_8x4_sf_layout(
            input, input_global_scale
        )
    else:
        # Two fp4 values will be packed into an uint8.
        # 两个 FP4 值打包到一个 uint8 中
        output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
        if is_sf_swizzled_layout:
            # We use the rounded values to store the swizzled values. Due to the
            # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
            # So, we first pad the scales to multiples of 128 and 4. Then, the scales
            # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
            # 使用向上取整的值来存储交织值。由于 Tensor Core 的要求，
            # 缩放因子的最小 tile 为 128x4。因此先将缩放因子填充到 128 和 4 的倍数，
            # 然后将每 4 个 float8_e4m3fn 缩放因子打包为一个 int32。
            round_up = lambda x, y: (x + y - 1) // y * y
            # 将 M 维度向上取整到 128 的倍数
            rounded_m = round_up(m, 128)
            # 计算缩放因子的 N 维度
            scale_n = n // block_size
            # 将缩放因子 N 维度向上取整到 4 的倍数
            rounded_n = round_up(scale_n, 4)
            # 创建交织布局的缩放因子张量，每 4 个打包为一个 int32
            output_scale = torch.empty(
                (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
            )
        else:
            # 非交织布局：每 16 个元素一个缩放因子，存储为 uint8
            output_scale = torch.empty((m, n // 16), device=device, dtype=torch.uint8)

        # 调用底层 C++ 扩展执行 FP4 缩放量化
        torch.ops._C.scaled_fp4_quant(
            output, input, output_scale, input_global_scale, is_sf_swizzled_layout
        )

    # 将缩放因子张量视图转换为 float8_e4m3fn 类型
    output_scale = output_scale.view(torch.float8_e4m3fn)
    # 返回量化后的输出张量和缩放因子
    return output, output_scale


# 将输入张量量化为NVFP4格式，用于打包的MoE输入
def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将输入张量量化为NVFP4格式，返回量化后的张量和缩放因子，用于打包的MoE输入。

    参数:
        input_tensor: 要量化为NVFP4的输入张量
        input_global_scale: 整个张量的全局缩放因子标量
        expert_offsets: 专家偏移量张量
        blockscale_offsets: 块缩放偏移量张量
        topk: 选择的top-k专家数量
    输出:
        output: NVFP4格式的量化张量
        output_scales: FP8-E4M3格式的块缩放张量
    """
    """
    Quantize input tensor to NVFP4 and return quantized tensor and scale, for
    packed MoE Inputs.
    Args:
        input_tensor: The input tensor to be quantized to NVFP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        expert_offsets: The expert offsets tensor
        blockscale_offsets: The blockscale offsets tensor
    Outputs:
        output: The quantized tensor in NVFP4
        output_scales: The blockscale tensor in FP8-E4M3
    """
    # 确保不在ROCm平台上运行
    assert not current_platform.is_rocm()
    # 确保输入张量是二维的
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    # 控制NVFP4 MoE专家量化支持的每个专家的最大token数，防止内核内存溢出
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    # 获取输入张量的形状：m_numtopk为token数乘以topk，k为隐藏维度
    m_numtopk, k = input_tensor.shape

    # 确保token数不超过每个专家的最大token数乘以topk
    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value."
    )
    # 计算缩放因子的k维度（每16个元素一个缩放因子）
    scales_k = k // 16
    # 将缩放因子的k维度填充为4的倍数
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    # 输出为uint8类型，包含打包的fp4值（每个字节存储两个fp4值）
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    # 创建输出缩放因子张量
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    # 调用底层C++算子执行FP4专家量化
    torch.ops._C.scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    # 将缩放因子重新解释为FP8-E4M3格式
    output_scales = output_scales.view(torch.float8_e4m3fn)
    # 返回量化后的张量和缩放因子
    return output, output_scales


# 融合SiLU激活+乘法+NVFP4量化，用于MoE中间激活值
def silu_and_mul_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合SiLU激活函数+乘法+NVFP4量化操作，用于MoE中间激活值的高效处理。

    参数:
        input_tensor: 输入张量，布局为 gate || up [m_topk, k*2]
        input_global_scale: 每个专家的缩放因子 [n_experts]
        expert_offsets: 专家偏移量张量 [n_experts+1]
        blockscale_offsets: 块缩放偏移量张量 [n_experts+1]
        topk: 选择的top-k专家数量
    输出:
        output: NVFP4格式的量化张量 [m_topk, k/2]
        output_scales: FP8-E4M3格式的块缩放张量
    """
    """
    Fused SiLU+Mul+NVFP4 quantization for MoE intermediate activations.

    Args:
        input_tensor: The input tensor with gate || up layout [m_topk, k*2]
        input_global_scale: A per-expert scaling factor [n_experts]
        expert_offsets: The expert offsets tensor [n_experts+1]
        blockscale_offsets: The blockscale offsets tensor [n_experts+1]
        topk: Number of top-k experts selected
    Outputs:
        output: The quantized tensor in NVFP4 [m_topk, k/2]
        output_scales: The blockscale tensor in FP8-E4M3
    """
    # 确保不在ROCm平台上运行
    assert not current_platform.is_rocm()
    # 确保输入张量是二维的
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    # 控制NVFP4 MoE专家量化支持的每个专家的最大token数，防止内核内存溢出
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    # 获取输入形状：m_numtopk为token数乘以topk，k_times_2为隐藏维度的两倍（gate和up拼接）
    m_numtopk, k_times_2 = input_tensor.shape
    # 确保输入宽度为偶数（gate || up布局）
    assert k_times_2 % 2 == 0, "input width must be even (gate || up layout)"
    # 计算实际的隐藏维度k
    k = k_times_2 // 2

    # 确保token数不超过每个专家的最大token数乘以topk
    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value."
    )
    # 计算缩放因子的k维度（每16个元素一个缩放因子）
    scales_k = k // 16
    # 将缩放因子的k维度填充为4的倍数
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    # 输出为uint8类型，包含打包的fp4值（每个字节存储两个fp4值）
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    # 创建输出缩放因子张量
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    # 调用底层C++算子执行融合的SiLU+乘法+FP4专家量化
    torch.ops._C.silu_and_mul_scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    # 将缩放因子重新解释为FP8-E4M3格式
    output_scales = output_scales.view(torch.float8_e4m3fn)
    # 返回量化后的张量和缩放因子
    return output, output_scales


# fp8
# FP8量化相关函数
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将输入张量量化为FP8格式并返回量化后的张量和缩放因子。

    支持静态量化和动态量化：提供scale参数时使用静态缩放，省略时动态计算缩放因子。
    还支持对输出张量进行可选填充，以便下游内核从填充中获益。

    参数:
        input: 要量化为FP8的输入张量（必须是二维: [M, N]）
        scale: 可选的FP8量化缩放因子，支持:
            - 0维或[1]: 逐张量缩放
            - 1维: 需要显式group_shape来区分逐通道和逐token（逐通道用(-1,1)，逐token用(1,-1)）
            - 2维 [M/group_m, N/group_n]: 分组缩放
        scale_ub: 动态逐token情况下缩放因子的可选上界
        num_token_padding: 如果指定，将输出的第一个维度填充到至少该值
        use_per_token_if_dynamic: 动态量化时是使用逐张量还是逐token方式
        group_shape: 可选的(group_m, group_n)元组，指定静态量化的分组形状

    返回:
        tuple[torch.Tensor, torch.Tensor]: FP8格式的输出张量和缩放因子
    """
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8 (must be 2D: [M, N])
        scale: Optional scaling factor for the FP8 quantization. Supports:
            - 0D or [1]: per-tensor scaling
            - 1D: requires explicit group_shape to disambiguate per-channel
              vs per-token (use (-1, 1) for per-channel, (1, -1) for per-token)
            - 2D [M/group_m, N/group_n]: group scaling (e.g. [M, N/128] for
              DeepSeek-style (1,128) groups, or [M/128, N/128] for (128,128))
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
        group_shape: Optional tuple (group_m, group_n) specifying the group
            shape for static quantization. Use -1 for "full extent" (e.g.,
            (-1, -1) for per-tensor, (-1, 1) for per-channel, etc.)
            Required for 1D scales; optional for 2D scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    # 假设batch_dim和num_tokens已经被展平
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    # 在ROCm MI300上，输出fp8数据类型为torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    # 如果指定了token填充数，则调整输出形状的第一个维度
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    # 如果未提供输出张量，则创建一个空的输出张量
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        # 如果提供了输出张量，则不支持填充
        assert num_token_padding is None, "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        # 动态量化分支：未提供缩放因子
        if use_per_token_if_dynamic:
            # 逐token动态量化：为每个token分配独立的缩放因子
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            # 逐张量动态量化：整个张量共享一个缩放因子
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # 静态量化分支：使用提供的缩放因子
        torch.ops._C.static_scaled_fp8_quant(output, input, scale, group_shape)

    # 返回量化后的输出张量和缩放因子
    return output, scale


# gptq allspark
# GPTQ AllSpark权重重排和矩阵乘法相关函数
def allspark_repack_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    has_zp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将量化权重、缩放因子和零点（如果是非对称量化）重排为n32k16格式，
    用于Ampere架构的W8A16融合矩阵乘法内核。

    参数:
        qweight: uint8权重张量，原始k x n格式
        scale: fp16/bf16权重缩放因子张量，1 x n格式
        zero_point: fp16/bf16权重零点张量，1 x n格式。非对称量化时必须提供
        has_zp: 对称量化时为False，非对称量化时为True

    返回:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            重排后的权重、缩放因子和可选的零点
    """
    """
    Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format
    for Ampere W8A16 Fused Gemm kernel

    Args:
        qweight: uint8 weight tensor, original k x n format.
        scale: fp16/bf16 weight scale tensor, 1 x n format.
        zero_point: fp16/bf16 weight zero_point tensor, 1 x n format.
            Must be provided for asymmetric quantization.
        has_zp: if use symmetric quantization, has_zp = False.
            if use asymmetric quantization, has_zp = True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] :
            rearranged weight, scale, and optionally zero_point.
    """
    # 获取权重张量的K维度（输入通道数）
    K = qweight.shape[0]
    # 获取权重张量的N维度（输出通道数）
    N = qweight.shape[1]
    # 将N维度对齐到32的倍数
    N_32align = (N + 32 - 1) // 32 * 32

    # 创建重排后的权重张量（转置为N x K格式）
    qweight_reorder = torch.empty(
        (N_32align, K), device=qweight.device, dtype=qweight.dtype
    )
    # 创建重排后的缩放因子张量
    scale_reorder = torch.empty((1, N_32align), device=scale.device, dtype=scale.dtype)
    # 初始化零点重排张量为None
    zero_point_reorder = None
    if has_zp:
        # 非对称量化时必须提供零点
        assert zero_point is not None, (
            "zero_point must be provided for asymmetric quantization."
        )
        # 创建重排后的零点张量
        zero_point_reorder = torch.empty(
            (1, N_32align), device=zero_point.device, dtype=zero_point.dtype
        )

    # 调用底层C++算子将权重重排为n32k16格式
    torch.ops._C.rearrange_kn_weight_as_n32k16_order(
        qweight,
        scale,
        zero_point,
        has_zp,
        qweight_reorder,
        scale_reorder,
        zero_point_reorder,
        K,
        N,
        N_32align,
    )

    # 返回重排后的权重、缩放因子和零点
    return qweight_reorder, scale_reorder, zero_point_reorder


# AllSpark W8A16矩阵乘法：8位权重与16位激活值的融合GEMM运算
def allspark_w8a16_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    n: int,
    group_size: int,
    sm_count: int,
    sm_version: int,
    CUBLAS_M_THRESHOLD: int,
    has_zp: bool,
    n32k16_reorder: bool,
) -> torch.Tensor:
    """
    AllSpark W8A16矩阵乘法运算。

    参数:
        a: 激活值张量（16位浮点）
        b_qweight: 量化后的权重张量（8位整数）
        b_scales: 权重的缩放因子
        b_qzeros: 权重的零点（非对称量化时使用）
        n: 输出维度大小
        group_size: 量化分组大小
        sm_count: GPU流处理器数量
        sm_version: GPU SM版本号
        CUBLAS_M_THRESHOLD: cuBLAS M维度阈值，用于选择内核策略
        has_zp: 是否使用非对称量化（包含零点）
        n32k16_reorder: 权重是否已经重排为n32k16格式

    返回:
        torch.Tensor: 矩阵乘法结果
    """
    # 调用底层C++算子执行W8A16矩阵乘法
    return torch.ops._C.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        b_qzeros,
        n,
        group_size,
        sm_count,
        sm_version,
        CUBLAS_M_THRESHOLD,
        has_zp,
        n32k16_reorder,
    )


# int8
# INT8量化相关函数
def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    将输入张量量化为int8，返回量化后的张量、缩放因子和可选的零点。

    参数:
        input: 要量化为int8的输入张量
        scale: 可选的int8量化缩放因子。未提供时使用动态逐token量化
        azp: 可选的int8量化零点。提供scale时非对称量化必须提供此参数
        symmetric: 是否使用对称量化（仅使用缩放因子，忽略零点）

    返回:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            int8输出张量、缩放因子和可选的零点
    """
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] : Output int8 tensor, scales, and optionally azp.
    """
    # 创建与输入形状相同的int8输出张量
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        # 静态逐张量量化分支
        # 确保对称量化时不提供零点，非对称量化时必须提供零点
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        # 调用静态int8量化算子
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    # 动态逐token量化分支
    # 为每个token分配独立的缩放因子
    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )
    # 非对称量化时创建零点张量，对称量化时为None
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    # 调用动态int8量化算子（输入需要连续内存布局）
    torch.ops._C.dynamic_scaled_int8_quant(
        output, input.contiguous(), input_scales, input_azp
    )
    # 返回量化后的输出、缩放因子和零点
    return output, input_scales, input_azp


# gguf
# GGUF/GGML格式相关函数（用于GGML量化模型的反量化和矩阵运算）

# GGML反量化：将GGML量化格式的权重反量化为指定数据类型
def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    """
    将GGML量化格式的权重张量反量化为浮点张量。

    参数:
        W: GGML量化格式的权重张量
        quant_type: GGML量化类型编号
        m: 输出矩阵的行数
        n: 输出矩阵的列数
        dtype: 输出数据类型

    返回:
        torch.Tensor: 反量化后的浮点权重张量
    """
    # 调用底层C++算子执行GGML反量化
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


# GGML矩阵向量乘法（A8量化版本）
def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """
    GGML格式的矩阵向量乘法（A8量化），适用于单token推理。

    参数:
        W: GGML量化格式的权重张量
        X: 输入向量张量
        quant_type: GGML量化类型编号
        row: 权重矩阵的行数

    返回:
        torch.Tensor: 矩阵向量乘法结果
    """
    # 调用底层C++算子执行GGML矩阵向量乘法
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


# GGML矩阵乘法（A8量化版本）
def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """
    GGML格式的矩阵乘法（A8量化），适用于批量token推理。

    参数:
        W: GGML量化格式的权重张量
        X: 输入矩阵张量
        quant_type: GGML量化类型编号
        row: 权重矩阵的行数

    返回:
        torch.Tensor: 矩阵乘法结果
    """
    # 调用底层C++算子执行GGML矩阵乘法
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


# GGML MoE矩阵乘法（A8量化版本）
def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """
    GGML格式的MoE（混合专家）矩阵乘法（A8量化版本）。

    参数:
        X: 输入张量
        W: GGML量化格式的专家权重张量
        sorted_token_ids: 排序后的token ID
        expert_ids: 专家ID
        num_tokens_post_padded: 填充后的token数量
        quant_type: GGML量化类型编号
        row: 权重矩阵的行数
        top_k: 选择的top-k专家数量
        tokens: token数量

    返回:
        torch.Tensor: MoE矩阵乘法结果
    """
    # 调用底层C++算子执行GGML MoE矩阵乘法
    return torch.ops._C.ggml_moe_a8(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


# GGML MoE向量化矩阵乘法（A8量化版本，适用于单token场景）
def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: torch.SymInt,
    tokens: torch.SymInt,
) -> torch.Tensor:
    """
    GGML格式的MoE向量化矩阵乘法（A8量化），适用于单token推理场景。

    参数:
        X: 输入张量
        W: GGML量化格式的专家权重张量
        topk_ids: top-k专家的ID
        top_k: 选择的top-k专家数量
        quant_type: GGML量化类型编号
        row: 权重矩阵的行数（支持符号整数）
        tokens: token数量（支持符号整数）

    返回:
        torch.Tensor: MoE向量化矩阵乘法结果
    """
    # 调用底层C++算子执行GGML MoE向量化矩阵乘法
    return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)


# 获取GGML MoE的块大小
def ggml_moe_get_block_size(quant_type: int) -> int:
    """
    获取指定GGML量化类型的MoE块大小。

    参数:
        quant_type: GGML量化类型编号

    返回:
        int: 对应量化类型的块大小
    """
    # 调用底层C++算子获取块大小
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


# mamba
# Mamba选择性状态空间模型（SSM）相关函数

# Mamba选择性扫描前向传播
def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    pad_slot_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    cu_chunk_seqlen: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
):
    """
    Mamba模型的选择性扫描前向传播。实现状态空间模型的核心递推计算。

    参数:
        u: 输入张量
        delta: 时间步长增量张量
        A: 状态转移矩阵
        B: 输入投影矩阵
        C: 输出投影矩阵
        D_: 可选的跳跃连接参数
        z_: 可选的门控张量
        delta_bias_: 可选的delta偏置
        delta_softplus: 是否对delta应用softplus激活
        query_start_loc: 查询起始位置（用于变长序列）
        cache_indices: 缓存索引（用于KV缓存管理）
        has_initial_state: 是否有初始状态的标志
        ssm_states: SSM状态张量（输入输出）
        pad_slot_id: 填充槽位ID
        block_size: 处理块大小，默认1024
        block_idx_first_scheduled_token: 每个块中第一个被调度token的索引
        block_idx_last_scheduled_token: 每个块中最后一个被调度token的索引
        initial_state_idx: 初始状态索引
        cu_chunk_seqlen: 累积分块序列长度
        last_chunk_indices: 最后一个分块的索引
    """
    # 调用底层C++算子执行选择性扫描前向传播
    torch.ops._C.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D_,
        z_,
        delta_bias_,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
        block_size,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        cu_chunk_seqlen,
        last_chunk_indices,
    )


# ROCm skinny gemms
# ROCm平台瘦矩阵（skinny）GEMM运算相关函数

# ROCm LLM矩阵乘法内核1
def LLMM1(a: torch.Tensor, b: torch.Tensor, rows_per_block: int) -> torch.Tensor:
    """
    ROCm平台的LLM矩阵乘法内核，针对瘦矩阵优化。

    参数:
        a: 输入矩阵A
        b: 输入矩阵B
        rows_per_block: 每个块处理的行数

    返回:
        torch.Tensor: 矩阵乘法结果
    """
    # 调用ROCm底层C++算子执行矩阵乘法
    return torch.ops._rocm_C.LLMM1(a, b, rows_per_block)


# ROCm SplitK矩阵乘法（权重向量乘法，K维度分割）
def wvSplitK(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    ROCm平台的SplitK矩阵乘法，沿K维度分割以提高并行度。

    参数:
        a: 输入矩阵A
        b: 输入矩阵B
        cu_count: 计算单元数量
        bias: 可选的偏置张量

    返回:
        torch.Tensor: 矩阵乘法结果（可能加上偏置）
    """
    # 调用ROCm底层C++算子执行SplitK矩阵乘法
    return torch.ops._rocm_C.wvSplitK(a, b, bias, cu_count)


# ROCm SplitK矩阵乘法（行列重排版本）
def wvSplitKrc(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    ROCm平台的SplitK矩阵乘法（行列重排版本），优化内存访问模式。

    参数:
        a: 输入矩阵A
        b: 输入矩阵B
        cu_count: 计算单元数量
        bias: 可选的偏置张量

    返回:
        torch.Tensor: 矩阵乘法结果（可能加上偏置）
    """
    # 调用ROCm底层C++算子执行行列重排版SplitK矩阵乘法
    return torch.ops._rocm_C.wvSplitKrc(a, b, bias, cu_count)


# ROCm SplitK量化矩阵乘法（支持量化输入和缩放因子）
def wvSplitKQ(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    cu_count: int,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    ROCm平台的量化SplitK矩阵乘法，支持量化输入的缩放。

    参数:
        a: 输入矩阵A（可能已量化）
        b: 输入矩阵B（可能已量化）
        out_dtype: 输出数据类型
        scale_a: 矩阵A的缩放因子
        scale_b: 矩阵B的缩放因子
        cu_count: 计算单元数量
        bias: 可选的偏置张量

    返回:
        torch.Tensor: 反量化后的矩阵乘法结果
    """
    # 创建输出张量，形状为 [b的行数, a的行数]
    out = torch.empty((b.shape[0], a.shape[0]), dtype=out_dtype, device=b.device)
    # 调用ROCm底层C++算子执行量化SplitK矩阵乘法
    torch.ops._rocm_C.wvSplitKQ(a, b, bias, out, scale_a, scale_b, cu_count)
    # 返回矩阵乘法结果
    return out


# moe
# 混合专家模型（MoE）相关函数

# MoE求和：将多个专家的输出加权求和
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    """
    MoE专家输出求和操作。

    参数:
        input: 输入张量（多个专家的加权输出）
        output: 输出张量（求和结果，原地写入）
    """
    # 调用MoE底层C++算子执行求和
    torch.ops._moe_C.moe_sum(input, output)


# MoE块大小对齐：将token按专家分组并对齐到指定块大小
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """
    将token按专家分组并对齐到指定块大小，用于MoE内核的高效执行。

    参数:
        topk_ids: 每个token选择的top-k专家ID
        num_experts: 专家总数
        block_size: 对齐的块大小
        sorted_token_ids: 输出，按专家排序后的token ID
        experts_ids: 输出，对应的专家ID
        num_tokens_post_pad: 输出，填充后的token数量
        expert_map: 可选的专家映射表
    """
    # 调用MoE底层C++算子执行块大小对齐
    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
    )


# 批量MoE块大小对齐：支持批量处理的MoE块大小对齐
def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """
    批量MoE块大小对齐，支持多批次的token分组和对齐。

    参数:
        max_tokens_per_batch: 每个批次的最大token数
        block_size: 对齐的块大小
        expert_num_tokens: 每个专家的token数量
        sorted_ids: 输出，排序后的token ID
        expert_ids: 输出，对应的专家ID
        num_tokens_post_pad: 输出，填充后的token数量
    """
    # 调用MoE底层C++算子执行批量块大小对齐
    torch.ops._moe_C.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )


# MoE LoRA块大小对齐：支持LoRA适配器的MoE块大小对齐
def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """
    支持LoRA适配器的MoE块大小对齐。在标准MoE对齐基础上额外处理LoRA适配器的映射。

    参数:
        topk_ids: 每个token选择的top-k专家ID
        token_lora_mapping: token到LoRA适配器的映射
        num_experts: 专家总数
        block_size: 对齐的块大小
        max_loras: 最大LoRA适配器数量
        max_num_tokens_padded: 填充后的最大token数量
        max_num_m_blocks: 最大M维度块数
        sorted_token_ids: 输出，排序后的token ID
        experts_ids: 输出，对应的专家ID
        num_tokens_post_pad: 输出，填充后的token数量
        adapter_enabled: 输出，适配器是否启用的标志
        lora_ids: 输出，LoRA适配器ID
        expert_map: 可选的专家映射表
    """
    # 调用MoE底层C++算子执行带LoRA的块大小对齐
    torch.ops._moe_C.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        expert_map,
    )


# MoE WNA16 GEMM：MoE中使用的权重N位激活16位矩阵乘法
def moe_wna16_gemm(
    input: torch.Tensor,
    output: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    bit: int,
) -> torch.Tensor:
    """
    MoE中使用的WNA16（权重N位，激活16位）矩阵乘法内核。

    参数:
        input: 输入激活张量
        output: 输出张量（原地写入）
        b_qweight: 量化后的权重张量
        b_scales: 权重缩放因子
        b_qzeros: 可选的权重零点
        topk_weights: 可选的top-k专家权重
        sorted_token_ids: 排序后的token ID
        experts_ids: 专家ID
        num_tokens_post_pad: 填充后的token数量
        top_k: 选择的top-k专家数量
        BLOCK_SIZE_M: M维度的块大小
        BLOCK_SIZE_N: N维度的块大小
        BLOCK_SIZE_K: K维度的块大小
        bit: 权重量化位数

    返回:
        torch.Tensor: 矩阵乘法结果
    """
    # 仅支持CUDA平台
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only available on CUDA platforms"
        )
    # 调用MoE底层C++算子执行WNA16矩阵乘法
    torch.ops._moe_C.moe_wna16_gemm(
        input,
        output,
        b_qweight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        top_k,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        bit,
    )


# 路由器BF16到FP32矩阵乘法：用于MoE路由器的高精度GEMM
def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    通过cuBLAS执行bf16 x bf16 -> fp32的矩阵乘法，用于MoE路由器计算。

    参数:
        input: BF16格式的输入张量
        weight: BF16格式的权重张量，形状为(N, K)

    返回:
        torch.Tensor: FP32格式的矩阵乘法结果
    """
    """bf16 x bf16 -> fp32 GEMM via cuBLAS. weight shape: (N, K)."""
    # 调用MoE底层C++算子执行BF16到FP32的矩阵乘法
    return torch.ops._moe_C.router_gemm_bf16_fp32(input, weight)


# 注册router_gemm_bf16_fp32的fake实现，用于torch.compile等场景
if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "router_gemm_bf16_fp32"):

    @register_fake("_moe_C::router_gemm_bf16_fp32")
    def router_gemm_bf16_fp32_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # 返回与实际输出形状相同的空张量（用于编译时形状推断）
        return torch.empty(
            input.shape[0], weight.shape[0], dtype=torch.float32, device=input.device
        )


# DeepSeek V3路由器矩阵乘法
def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    DeepSeek V3模型专用的路由器矩阵乘法。

    参数:
        hidden_states: 隐藏状态输入张量
        router_weight: 路由器权重张量
        output_dtype: 输出数据类型

    返回:
        torch.Tensor: 路由器矩阵乘法结果
    """
    # 创建输出张量，形状为 [token数, 专家数]
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=output_dtype,
    )
    # 调用MoE底层C++算子执行DeepSeek V3路由器矩阵乘法
    torch.ops._moe_C.dsv3_router_gemm(output, hidden_states, router_weight)
    # 返回路由器输出
    return output


# Top-K Softmax路由：对门控输出执行softmax后选择top-k专家
def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    """
    对门控输出执行softmax激活后选择top-k专家，计算路由权重。

    参数:
        topk_weights: 输出，top-k专家的路由权重
        topk_ids: 输出，top-k专家的ID
        token_expert_indices: 输出，token到专家的索引映射
        gating_output: 门控网络的原始输出（logits）
        renormalize: 是否对top-k权重重新归一化
        e_score_correction_bias: 可选的专家分数校正偏置
    """
    # 调用MoE底层C++算子执行top-k softmax路由
    torch.ops._moe_C.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )


# Top-K Sigmoid路由：对门控输出执行sigmoid后选择top-k专家
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    """
    对门控输出执行sigmoid激活后选择top-k专家，计算路由权重。

    参数:
        topk_weights: 输出，top-k专家的路由权重
        topk_ids: 输出，top-k专家的ID
        token_expert_indices: 输出，token到专家的索引映射
        gating_output: 门控网络的原始输出（logits）
        renormalize: 是否对top-k权重重新归一化
        e_score_correction_bias: 可选的专家分数校正偏置
    """
    # 调用MoE底层C++算子执行top-k sigmoid路由
    torch.ops._moe_C.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )


# 分组Top-K路由：对专家进行分组后执行top-k选择
def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    """
    执行MoE的分组top-k路由。先在专家组间选择top-k组，再在选中的组内选择top-k专家。

    参数:
        scores: 原始输入（scoring_func=1时为logits，scoring_func=0时为分数）
        num_expert_group: 专家组数量
        topk_group: 要选择的组数量
        topk: 每个token要选择的专家数量
        renormalize: 是否对输出权重重新归一化
        routed_scaling_factor: 路由权重的缩放因子
        bias: 偏置张量（专家分数校正偏置），在内核中融合计算
        scoring_func: 评分函数类型，0=无激活，1=sigmoid
    """
    """
    Perform grouped top-k routing for mixture of experts.

    Args:
        scores: Raw inputs (logits if scoring_func=1, scores if scoring_func=0)
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select
        topk: Number of experts to select per token
        renormalize: Whether to renormalize the output weights
        routed_scaling_factor: Scaling factor for routing weights
        bias: Bias tensor (e_score_correction_bias). Always fused in kernel.
        scoring_func: 0=none (no activation), 1=sigmoid
    """
    # 仅支持CUDA平台
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The fused grouped_topk kernel is only available on CUDA platforms"
        )
    # 调用MoE底层C++算子执行分组top-k路由
    return torch.ops._moe_C.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


# MoE WNA16 Marlin GEMM：MoE中使用Marlin内核的WNA16量化矩阵乘法
def moe_wna16_marlin_gemm(
    input: torch.Tensor,
    output: torch.Tensor | None,
    b_qweight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_qzeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
) -> torch.Tensor:
    """
    MoE中使用Marlin内核执行WNA16（权重N位，激活16位）量化矩阵乘法。

    参数:
        input: 输入激活张量
        output: 可选的输出张量
        b_qweight: Marlin格式的量化权重
        b_bias: 可选的偏置张量
        b_scales: 权重缩放因子
        a_scales: 可选的激活缩放因子
        global_scale: 可选的全局缩放因子
        b_qzeros: 可选的权重零点
        g_idx: 可选的分组索引
        perm: 可选的排列索引
        workspace: Marlin内核的工作空间张量
        sorted_token_ids: 排序后的token ID
        expert_ids: 专家ID
        num_tokens_past_padded: 填充后的token数量
        topk_weights: top-k专家权重
        moe_block_size: MoE的块大小
        top_k: 选择的top-k专家数量
        mul_topk_weights: 是否乘以top-k权重
        b_q_type: 量化类型标量
        size_m: M维度大小
        size_n: N维度大小
        size_k: K维度大小
        is_k_full: K维度是否完整（非分组量化）
        use_atomic_add: 是否使用原子加法
        use_fp32_reduce: 是否使用FP32精度进行归约
        is_zp_float: 零点是否为浮点类型
        thread_k: 每个线程处理的K维度大小，-1为自动
        thread_n: 每个线程处理的N维度大小，-1为自动
        blocks_per_sm: 每个SM的块数，-1为自动

    返回:
        torch.Tensor: Marlin量化矩阵乘法结果
    """
    # 调用MoE底层C++算子执行Marlin WNA16矩阵乘法
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        input,
        output,
        b_qweight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_qzeros,
        g_idx,
        perm,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        thread_k,
        thread_n,
        blocks_per_sm,
    )


# 注册Marlin MoE GEMM相关算子的fake实现，用于torch.compile等场景
if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):

    # Marlin GEMM MoE的fake实现：返回与实际输出形状相同的空张量
    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(
        a: torch.Tensor,
        b_q_weights: torch.Tensor,
        sorted_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        b_scales: torch.Tensor,
        b_zero_points: torch.Tensor,
        g_idx: torch.Tensor,
        perm: torch.Tensor,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
        is_k_full: bool,
        num_experts: int,
        topk: int,
        moe_block_size: int,
        replicate_input: bool,
        apply_weights: bool,
    ) -> torch.Tensor:
        """
        Marlin GEMM MoE的fake实现，用于编译时形状推断。

        返回:
            形状为(size_m, topk, size_n)的空张量
        """
        # 返回与实际输出形状相同的空张量
        return torch.empty((size_m, topk, size_n), dtype=a.dtype, device=a.device)

    # MoE WNA16 Marlin GEMM的fake实现
    @register_fake("_moe_C::moe_wna16_marlin_gemm")
    def moe_wna16_marlin_gemm_fake(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
    ):
        """
        MoE WNA16 Marlin GEMM的fake实现，用于编译时形状推断。

        返回:
            形状为(size_m * top_k, size_n)的空张量
        """
        # 返回与实际输出形状相同的空张量
        return torch.empty(
            (size_m * top_k, size_n), dtype=input.dtype, device=input.device
        )


# KV缓存重塑和缓存：将key/value张量重塑并写入分页KV缓存
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """
    将key和value张量重塑并写入分页KV缓存。

    参数:
        key: key张量
        value: value张量
        key_cache: key缓存张量（分页格式）
        value_cache: value缓存张量（分页格式）
        slot_mapping: 槽位映射，指定每个token写入缓存的位置
        kv_cache_dtype: KV缓存的数据类型字符串
        k_scale: key的缩放因子（用于量化缓存）
        v_scale: value的缩放因子（用于量化缓存）
    """
    # 调用缓存操作C++算子执行重塑和缓存写入
    torch.ops._C_cache_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


# Flash Attention格式的KV缓存重塑和写入
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """
    将key和value张量重塑并写入Flash Attention格式的分页KV缓存。

    参数:
        key: key张量
        value: value张量
        key_cache: Flash格式的key缓存张量
        value_cache: Flash格式的value缓存张量
        slot_mapping: 槽位映射
        kv_cache_dtype: KV缓存的数据类型字符串
        k_scale: key的缩放因子
        v_scale: value的缩放因子
    """
    # 调用缓存操作C++算子执行Flash格式的重塑和缓存写入
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


# MLA（多头潜在注意力）的拼接和缓存操作
def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """
    将MLA的压缩KV和位置编码key拼接后写入KV缓存。

    参数:
        kv_c: 压缩的KV张量（MLA的低秩压缩表示）
        k_pe: 位置编码部分的key张量
        kv_cache: KV缓存张量
        slot_mapping: 槽位映射
        kv_cache_dtype: KV缓存的数据类型字符串
        scale: 缩放因子（用于量化缓存）
    """
    # 调用缓存操作C++算子执行MLA拼接和缓存写入
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


# MLA拼接缓存与RoPE融合操作：将RoPE位置编码与MLA缓存写入融合为一步
def concat_and_cache_mla_rope_fused(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
) -> None:
    """
    融合RoPE旋转位置编码和MLA缓存写入操作，减少内存读写次数。

    参数:
        positions: token的位置ID
        q_pe: query的位置编码部分（将被原地应用RoPE）
        k_pe: key的位置编码部分（将被原地应用RoPE）
        kv_c: 压缩的KV张量
        cos_sin_cache: RoPE的cos/sin缓存表
        is_neox: 是否使用NeoX风格的RoPE（交错vs拆分）
        slot_mapping: 槽位映射
        kv_cache: KV缓存张量
        kv_cache_dtype: KV缓存的数据类型字符串
        kv_cache_scale: KV缓存缩放因子
    """
    # 调用缓存操作C++算子执行融合的RoPE和MLA缓存写入
    torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused(
        positions,
        q_pe,
        k_pe,
        kv_c,
        cos_sin_cache,
        is_neox,
        slot_mapping,
        kv_cache,
        kv_cache_dtype,
        kv_cache_scale,
    )


# 块交换操作：在CPU和GPU之间交换KV缓存块
def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_size_in_bytes: int,
    block_mapping: torch.Tensor,
) -> None:
    """
    在源张量和目标张量之间复制指定的块。用于KV缓存的CPU-GPU交换。

    Copy specific blocks from one tensor to another.

    This method assumes each of the two input tensors is composed of
    consecutive contiguous blocks, of size block_size_in_bytes.
    i.e. the memory layout for each tensor is:
    [block0] [block1] ... [block N]

    block_mapping determines the subset of blocks to copy of the source tensor,
    and their matching destination block number on the destination tensor.
    block_mapping is expected to be a tensor of shape (num_blocks_to_copy, 2)
    where each block_mapping[i] represents a single copy operation, copying
    block #block_mapping[i][0] from the source tensor
    to block #block_mapping[i][1] on the destination tensor.
    block_mapping should have dtype int64.

    The source and the destination tensors can be either on cpu or gpu,
    but not both on cpu.
    the block mapping tensor must on cpu.
    """
    # 调用缓存操作C++算子执行块交换
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_size_in_bytes, block_mapping)


# FP8转换：将张量转换为FP8格式
def convert_fp8(
    output: torch.Tensor, input: torch.Tensor, scale: float = 1.0, kv_dtype: str = "fp8"
) -> None:
    """
    将输入张量转换为FP8格式并写入输出张量。

    参数:
        output: 输出张量（FP8格式）
        input: 输入张量
        scale: 缩放因子，默认1.0
        kv_dtype: KV缓存数据类型字符串，默认"fp8"
    """
    # 调用缓存操作C++算子执行FP8转换
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


# 收集并可能反量化KV缓存：从分页缓存中收集数据并可能执行反量化
def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """
    从分页KV缓存中收集数据到连续内存，如果缓存是量化格式则同时执行反量化。

    参数:
        src_cache: 源KV缓存张量（分页格式）
        dst: 目标张量（连续格式）
        block_table: 块表，记录每个序列使用的物理块
        cu_seq_lens: 累积序列长度
        token_to_seq: token到序列的映射
        num_tokens: token总数
        kv_cache_dtype: KV缓存的数据类型字符串
        scale: 反量化缩放因子
        seq_starts: 可选的序列起始偏移
    """
    # 调用缓存操作C++算子执行收集和可能的反量化
    torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )


# 上下文并行（CP）收集缓存：在上下文并行场景下收集KV缓存
def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """
    在上下文并行（Context Parallelism）场景下从分页KV缓存收集数据。

    参数:
        src_cache: 源KV缓存张量
        dst: 目标张量
        block_table: 块表
        cu_seq_lens: 累积序列长度
        batch_size: 批次大小
        seq_starts: 可选的序列起始偏移
    """
    # 调用缓存操作C++算子执行上下文并行缓存收集
    torch.ops._C_cache_ops.cp_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts
    )


# 上下文并行收集并上转换FP8 KV缓存：将FP8缓存收集并转换为BF16
def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
) -> None:
    """
    收集FP8格式的KV缓存并上转换为BF16格式的工作空间。

    Gather and upconvert FP8 KV cache to BF16 workspace.

    Args:
        src_cache: FP8 KV cache [num_blocks, block_size, 656]
        dst: BF16 output workspace [total_tokens, 576]
        block_table: Block indices [num_reqs, max_blocks]
        seq_lens: Sequence lengths [num_reqs]
        workspace_starts: Workspace start offsets [num_reqs]
        batch_size: Number of requests
    """
    # 调用缓存操作C++算子执行FP8到BF16的收集和上转换
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache, dst, block_table, seq_lens, workspace_starts, batch_size
    )


# MLA查询拼接：将MLA的nope和rope部分拼接为完整查询
def concat_mla_q(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_out: torch.Tensor,
) -> None:
    """
    将MLA/DSA注意力的query nope分量和rope分量拼接为完整查询。

    Concatenate query nope and rope for MLA/DSA attention.

    Args:
        ql_nope: Query nope component [num_tokens, num_heads, nope_dim]
        q_pe: Query rope component [num_tokens, num_heads, rope_dim]
        q_out: Output tensor [num_tokens, num_heads, nope_dim + rope_dim]
    """
    # 调用缓存操作C++算子执行MLA查询拼接
    torch.ops._C_cache_ops.concat_mla_q(ql_nope, q_pe, q_out)


# 索引器key量化并写入缓存：对key进行分块量化后写入KV缓存
def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None:
    """
    对key张量进行分块量化后写入KV缓存。

    参数:
        k: key张量
        kv_cache: KV缓存张量
        slot_mapping: 槽位映射
        quant_block_size: 量化块大小
        kv_cache_dtype: KV缓存的数据类型字符串
    """
    # 调用缓存操作C++算子执行key量化和缓存写入
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
    )


# 上下文并行收集索引器key量化缓存
def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """
    在上下文并行场景下收集索引器格式的量化key缓存。

    参数:
        kv_cache: 量化的KV缓存张量
        dst_k: 目标key张量
        dst_scale: 目标缩放因子张量
        block_table: 块表
        cu_seq_lens: 累积序列长度
    """
    # 调用缓存操作C++算子执行上下文并行的量化key缓存收集
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )


# 获取CUDA设备属性
def get_device_attribute(attribute: int, device: int) -> int:
    """
    获取指定CUDA设备的属性值。

    参数:
        attribute: CUDA设备属性枚举值
        device: 设备ID

    返回:
        int: 设备属性值
    """
    # 调用CUDA工具C++算子查询设备属性
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


# 获取每个块的最大共享内存
def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    """
    获取指定CUDA设备每个块的最大共享内存大小。

    参数:
        device: 设备ID

    返回:
        int: 每个块的最大共享内存大小（字节）
    """
    # ruff: noqa: E501
    # 调用CUDA工具C++算子查询最大共享内存
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device
    )


# custom ar - 自定义AllReduce通信相关函数
# 用于多GPU间高效的自定义AllReduce操作

# 初始化自定义AllReduce
def init_custom_ar(
    ipc_tensors: list[torch.Tensor],
    rank_data: torch.Tensor,
    rank: int,
    fully_connected: bool,
) -> int:
    """
    初始化自定义AllReduce通信。

    参数:
        ipc_tensors: IPC（进程间通信）张量列表
        rank_data: 本rank的数据张量
        rank: 当前进程的rank编号
        fully_connected: 是否为全连接拓扑

    返回:
        int: AllReduce句柄
    """
    # 调用自定义AllReduce C++算子进行初始化
    return torch.ops._C_custom_ar.init_custom_ar(
        ipc_tensors, rank_data, rank, fully_connected
    )


# 执行自定义AllReduce操作
def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    """
    执行自定义AllReduce操作，将所有rank的数据归约到每个rank。

    参数:
        fa: AllReduce句柄
        inp: 输入张量
        out: 输出张量
        reg_buffer: 注册缓冲区指针
        reg_buffer_sz_bytes: 注册缓冲区大小（字节）
    """
    # 调用自定义AllReduce C++算子执行归约操作
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


# 释放自定义AllReduce资源
def dispose(fa: int) -> None:
    """
    释放自定义AllReduce句柄及其相关资源。

    参数:
        fa: AllReduce句柄
    """
    # 调用自定义AllReduce C++算子释放资源
    torch.ops._C_custom_ar.dispose(fa)


# 获取元数据大小
def meta_size() -> int:
    """
    获取自定义AllReduce元数据的大小。

    返回:
        int: 元数据大小（字节）
    """
    # 调用自定义AllReduce C++算子获取元数据大小
    return torch.ops._C_custom_ar.meta_size()


# 注册通信缓冲区
def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    """
    为自定义AllReduce注册IPC通信缓冲区。

    参数:
        fa: AllReduce句柄
        ipc_tensors: IPC张量句柄列表
    """
    # 调用自定义AllReduce C++算子注册缓冲区
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


# 获取图缓冲区的IPC元数据
def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    """
    获取CUDA图缓冲区的IPC元数据，用于图捕获场景。

    参数:
        fa: AllReduce句柄

    返回:
        tuple[list[int], list[int]]: (句柄列表, 偏移量列表)
    """
    # 调用自定义AllReduce C++算子获取图缓冲区IPC元数据
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


# 注册CUDA图缓冲区
def register_graph_buffers(
    fa: int, handles: list[list[int]], offsets: list[list[int]]
) -> None:
    """
    注册CUDA图捕获使用的缓冲区。

    参数:
        fa: AllReduce句柄
        handles: 所有rank的句柄列表
        offsets: 所有rank的偏移量列表
    """
    # 调用自定义AllReduce C++算子注册图缓冲区
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


# 分配共享缓冲区并获取句柄
def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    """
    分配共享内存缓冲区并返回其句柄。

    参数:
        size: 缓冲区大小（字节）

    返回:
        tuple[int, torch.Tensor]: (缓冲区指针, IPC句柄张量)
    """
    # 调用自定义AllReduce C++算子分配共享缓冲区
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


# 打开共享内存句柄
def open_mem_handle(mem_handle: torch.Tensor):
    """
    打开另一个进程的共享内存句柄，获取本进程可访问的指针。

    参数:
        mem_handle: IPC内存句柄张量

    返回:
        共享内存指针
    """
    # 调用自定义AllReduce C++算子打开内存句柄
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


# 释放共享缓冲区
def free_shared_buffer(ptr: int) -> None:
    """
    释放之前分配的共享内存缓冲区。

    参数:
        ptr: 缓冲区指针
    """
    # 调用自定义AllReduce C++算子释放共享缓冲区
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


# quick all reduce - 快速AllReduce通信相关函数
# 支持量化的快速AllReduce，适用于较小的数据量

# 初始化快速AllReduce
def init_custom_qr(rank: int, world_size: int, qr_max_size: int | None = None) -> int:
    """
    初始化快速AllReduce（Quick Reduce）通信。

    参数:
        rank: 当前进程的rank编号
        world_size: 总进程数
        qr_max_size: 可选的最大数据大小限制

    返回:
        int: 快速AllReduce句柄
    """
    # 调用自定义AllReduce C++算子初始化快速归约
    return torch.ops._C_custom_ar.init_custom_qr(rank, world_size, qr_max_size)


# 销毁快速AllReduce
def qr_destroy(fa: int) -> None:
    """
    销毁快速AllReduce句柄及其相关资源。

    参数:
        fa: 快速AllReduce句柄
    """
    # 调用自定义AllReduce C++算子销毁快速归约
    torch.ops._C_custom_ar.qr_destroy(fa)


# 执行快速AllReduce操作
def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool = False,
) -> None:
    """
    执行快速AllReduce操作，支持量化通信以减少带宽。

    参数:
        fa: 快速AllReduce句柄
        inp: 输入张量
        out: 输出张量
        quant_level: 量化级别（控制通信精度）
        cast_bf2half: 是否将BF16转换为FP16
    """
    # 调用自定义AllReduce C++算子执行快速归约
    torch.ops._C_custom_ar.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)


# 获取快速AllReduce的IPC句柄
def qr_get_handle(fa: int) -> torch.Tensor:
    """
    获取快速AllReduce的IPC通信句柄。

    参数:
        fa: 快速AllReduce句柄

    返回:
        torch.Tensor: IPC句柄张量
    """
    # 调用自定义AllReduce C++算子获取句柄
    return torch.ops._C_custom_ar.qr_get_handle(fa)


# 打开其他rank的快速AllReduce句柄
def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    """
    打开其他rank的快速AllReduce IPC句柄，建立通信连接。

    参数:
        fa: 快速AllReduce句柄
        handles: 其他rank的IPC句柄列表
    """
    # 调用自定义AllReduce C++算子打开远程句柄
    return torch.ops._C_custom_ar.qr_open_handles(fa, handles)


# 获取快速AllReduce的最大数据大小
def qr_max_size() -> int:
    """
    获取快速AllReduce支持的最大数据大小。

    返回:
        int: 最大数据大小（字节）
    """
    # 调用自定义AllReduce C++算子获取最大大小
    return torch.ops._C_custom_ar.qr_max_size()


# Flash MLA元数据获取：为Flash MLA注意力计算准备调度元数据
def get_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    获取Flash MLA注意力计算所需的调度元数据，包括分块调度信息和分割数。

    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    # 调用底层C++算子获取Flash MLA调度元数据
    return torch.ops._C.get_flash_mla_metadata(
        cache_seqlens, num_heads_per_head_k, num_heads_k
    )


# Flash MLA KV缓存注意力计算
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用Flash MLA算法从分页KV缓存中执行注意力计算。

    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    # 如果未指定softmax缩放因子，使用默认的 1/sqrt(head_dim)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # 调用底层C++算子执行Flash MLA前向KV缓存注意力
    out, softmax_lse = torch.ops._C.flash_mla_fwd_kvcache(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    # 返回注意力输出和softmax的log-sum-exp值
    return out, softmax_lse


# SM100 CUTLASS MLA解码：使用CUTLASS库在SM100架构上执行MLA解码注意力
def sm100_cutlass_mla_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
    num_kv_splits: int,
) -> torch.Tensor:
    """
    使用SM100（Blackwell）架构的CUTLASS内核执行MLA解码注意力。

    参数:
        out: 输出张量
        lse: log-sum-exp值张量
        q_nope: query的nope（非位置编码）分量
        q_pe: query的位置编码分量
        kv_c_and_k_pe_cache: 压缩KV和key位置编码的联合缓存
        seq_lens: 序列长度
        page_table: 页表，记录每个序列使用的物理页
        workspace: CUTLASS内核工作空间
        scale: softmax缩放因子
        num_kv_splits: KV分割数（用于分割KV以提高并行度）

    返回:
        torch.Tensor: 注意力输出
    """
    # 调用底层C++算子执行SM100 CUTLASS MLA解码
    torch.ops._C.sm100_cutlass_mla_decode(
        out,
        lse,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        scale,
        num_kv_splits,
    )
    # 返回注意力输出
    return out


# 获取SM100 CUTLASS MLA工作空间大小
def sm100_cutlass_mla_get_workspace_size(
    max_seq_len: int, num_batches: int, sm_count: int, num_kv_splits: int
) -> int:
    """
    计算SM100 CUTLASS MLA解码所需的工作空间大小。

    参数:
        max_seq_len: 最大序列长度
        num_batches: 批次数
        sm_count: SM（流式多处理器）数量
        num_kv_splits: KV分割数

    返回:
        int: 所需工作空间大小（字节）
    """
    # 调用底层C++算子计算工作空间大小
    return torch.ops._C.sm100_cutlass_mla_get_workspace_size(
        max_seq_len, num_batches, sm_count, num_kv_splits
    )


# DeepSeek V3融合A矩阵乘法：用于DeepSeek V2/V3的QKV A投影
def dsv3_fused_a_gemm(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
) -> None:
    """
    DeepSeek V3融合A矩阵乘法，针对小批量场景优化（SM 9.0+, bf16, 1-16个token）。

    DeepSeek V3 fused A GEMM (SM 9.0+, bf16 only, 1-16 tokens).

    Computes output = mat_a @ mat_b.T where:
      mat_a: [num_tokens, 7168] row-major bf16 (hidden states)
      mat_b: [7168, 2112] column-major bf16 (weight transposed)
      output: [num_tokens, 2112] row-major bf16

    Optimized for the DeepSeek V2/V3 QKV A-projection at small batch sizes.
    Requires SM 9.0+ (Hopper).
    """
    # 调用底层C++算子执行DeepSeek V3融合A矩阵乘法
    torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b)


# 注册CPU权重打包线性层的fake实现
if hasattr(torch.ops._C, "weight_packed_linear"):

    @register_fake("_C::weight_packed_linear")
    def weight_packed_linear_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        bias: torch.Tensor | None,
        is_vnni: bool,
    ) -> torch.Tensor:
        """
        CPU权重打包线性层的fake实现，用于编译时形状推断。

        返回:
            形状为(mat1行数, mat2行数)的空张量
        """
        # 返回与实际输出形状相同的空张量
        return torch.empty(
            (mat1.size(0), mat2.size(0)), dtype=mat1.dtype, device=mat2.device
        )


# 注册CPU融合MoE专家计算的fake实现
if hasattr(torch.ops._C, "fused_experts_cpu"):

    @register_fake("_C::fused_experts_cpu")
    def fused_experts_cpu_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool,
        use_int8_w8a8: bool,
        use_fp8_w8a16: bool,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        block_size: list[int] | None,
        a1_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        is_vnni: bool,
    ) -> torch.Tensor:
        """
        CPU融合MoE专家计算的fake实现，用于编译时形状推断。

        返回:
            与hidden_states形状相同的空张量
        """
        # 返回与输入形状相同的空张量
        return torch.empty_like(hidden_states)


# 注册CPU INT8缩放矩阵乘法加量化的fake实现
if hasattr(torch.ops._C, "int8_scaled_mm_with_quant"):

    @register_fake("_C::int8_scaled_mm_with_quant")
    def int8_scaled_mm_with_quant_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        scales2: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        is_vnni: bool,
    ) -> torch.Tensor:
        """
        CPU INT8缩放矩阵乘法加量化的fake实现，用于编译时形状推断。

        返回:
            形状为(M, N)的空张量
        """
        # 获取输出维度
        M = mat1.size(0)
        N = mat2.size(0)
        # 返回与实际输出形状相同的空张量
        return torch.empty((M, N), dtype=out_dtype)


# CPU DNNL（oneDNN）GEMM处理器类：管理oneDNN矩阵乘法的资源
class CPUDNNLGEMMHandler:
    """
    CPU DNNL（oneDNN）GEMM处理器，封装oneDNN矩阵乘法内核的生命周期管理。

    该类持有oneDNN底层矩阵乘法处理器的句柄，在析构时自动释放资源。
    """
    def __init__(self) -> None:
        """初始化DNNL GEMM处理器，设置默认值。"""
        # oneDNN处理器句柄张量，存储C++层的指针
        self.handler_tensor: torch.Tensor | None = None
        # 权重矩阵的N维度（输出维度）
        self.n = -1
        # 权重矩阵的K维度（缩减维度）
        self.k = -1

    def __del__(self):
        """析构函数，释放oneDNN底层矩阵乘法处理器资源。"""
        if self.handler_tensor is not None:
            # 调用C++算子释放DNNL矩阵乘法处理器
            torch.ops._C.release_dnnl_matmul_handler(self.handler_tensor.item())


# 检查是否支持oneDNN矩阵乘法
_supports_onednn = bool(hasattr(torch.ops._C, "create_onednn_mm_handler"))


# 检查oneDNN ACL（Arm Compute Library）是否支持
def is_onednn_acl_supported():
    """
    检查oneDNN的Arm Compute Library后端是否可用。

    返回:
        bool: 是否支持oneDNN ACL
    """
    # 调用底层C++算子检查ACL支持
    return torch.ops._C.is_onednn_acl_supported()


# 创建oneDNN矩阵乘法处理器
def create_onednn_mm(
    weight: torch.Tensor,  # [K, N]
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    """
    创建oneDNN矩阵乘法处理器，预编译内核以加速后续计算。

    参数:
        weight: 权重张量，形状为[K, N]
        primitive_cache_size: 原语缓存大小，默认128

    返回:
        CPUDNNLGEMMHandler: oneDNN GEMM处理器
    """
    # 创建处理器实例
    handler = CPUDNNLGEMMHandler()
    # 记录权重矩阵的维度
    handler.k, handler.n = weight.size()
    # 将处理器指针存储在张量中，防止被编译器内联优化
    handler.handler_tensor = torch.tensor(
        torch.ops._C.create_onednn_mm_handler(weight, primitive_cache_size),
        dtype=torch.int64,
    )
    return handler


# 执行oneDNN矩阵乘法
def onednn_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    使用oneDNN处理器执行矩阵乘法 output = x @ weight + bias。

    参数:
        dnnl_handler: oneDNN GEMM处理器
        x: 输入张量
        bias: 可选的偏置张量

    返回:
        torch.Tensor: 矩阵乘法结果
    """
    # 创建输出张量，最后一维为权重的N维度
    output = torch.empty((*x.shape[0:-1], dnnl_handler.n), dtype=x.dtype)
    # 调用底层C++算子执行oneDNN矩阵乘法，输入需要reshape为2D
    torch.ops._C.onednn_mm(
        output, x.reshape(-1, dnnl_handler.k), bias, dnnl_handler.handler_tensor
    )

    return output


# 创建oneDNN缩放矩阵乘法处理器：支持量化的矩阵乘法
def create_onednn_scaled_mm(
    weight: torch.Tensor,  # [K, N]
    weight_scales: torch.Tensor,
    output_type: torch.dtype,
    dynamic_quant: bool,
    use_azp: bool,
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    """
    创建支持缩放（量化）的oneDNN矩阵乘法处理器。

    参数:
        weight: 量化权重张量，形状为[K, N]
        weight_scales: 权重缩放因子
        output_type: 输出数据类型
        dynamic_quant: 是否使用动态量化
        use_azp: 是否使用非对称零点（Asymmetric Zero Point）
        primitive_cache_size: 原语缓存大小，默认128

    返回:
        CPUDNNLGEMMHandler: oneDNN缩放GEMM处理器
    """
    # 创建处理器实例
    handler = CPUDNNLGEMMHandler()
    # 记录权重矩阵的维度
    handler.k, handler.n = weight.size()
    # 将处理器指针存储在张量中，防止被编译器内联优化
    handler.handler_tensor = torch.tensor(
        torch.ops._C.create_onednn_scaled_mm_handler(
            weight,
            weight_scales,
            output_type,
            dynamic_quant,
            use_azp,
            primitive_cache_size,
        ),
        dtype=torch.int64,
    )
    return handler


# oneDNN INT8缩放量化：将输入张量量化为INT8格式
def onednn_scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
):
    """
    将输入张量量化为INT8格式，返回量化后的张量、缩放因子和可选的零点。

    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] : Output int8 tensor, scales, and optionally azp.
    """
    # 创建与输入形状相同的INT8输出张量
    output = torch.empty_like(input, dtype=torch.int8)
    # 计算token数量（将多维输入展平为2D）
    token_num = input.numel() // input.shape[-1]
    # 将输入reshape为(token_num, hidden_dim)的2D张量
    input = input.view((token_num, input.shape[-1]))
    if scale is not None:
        # 静态逐张量量化模式
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        # 调用C++算子执行静态INT8量化
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # 动态逐token量化模式
    # 为每个token创建缩放因子张量
    input_scales = torch.empty((token_num, 1), device=input.device, dtype=torch.float32)
    # 非对称量化时创建零点张量，对称量化时为None
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    # 调用C++算子执行动态INT8量化
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales, input_azp)
    return output, input_scales, input_azp


# oneDNN缩放矩阵乘法：使用量化权重执行矩阵乘法
def onednn_scaled_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor | None,
    input_zp: torch.Tensor | None,
    input_zp_adj: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    使用oneDNN处理器执行缩放（量化）矩阵乘法。

    参数:
        dnnl_handler: oneDNN缩放GEMM处理器
        x: 输入张量（可能已量化为INT8）
        output: 输出张量
        input_scale: 可选的输入缩放因子
        input_zp: 可选的输入零点
        input_zp_adj: 可选的输入零点调整值
        bias: 可选的偏置张量

    返回:
        torch.Tensor: 缩放矩阵乘法结果
    """
    # 调用底层C++算子执行oneDNN缩放矩阵乘法
    torch.ops._C.onednn_scaled_mm(
        output,
        x,
        input_scale,
        input_zp,
        input_zp_adj,
        bias,
        dnnl_handler.handler_tensor,
    )

    return output


# CPU注意力调度元数据获取：为CPU注意力内核准备调度信息
def cpu_attn_get_scheduler_metadata(
    num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    dtype: torch.dtype,
    query_start_loc: torch.Tensor,
    causal: bool,
    sliding_window_size: int,
    isa: str,
    enable_kv_split: bool,
) -> torch.Tensor:
    """
    获取CPU注意力内核的调度元数据，包含线程分配和工作分区信息。

    参数:
        num_reqs: 请求数量
        num_heads: 注意力头数
        num_kv_heads: KV注意力头数
        head_dim: 每个头的维度
        seq_lens: 序列长度张量
        dtype: 数据类型
        query_start_loc: query的起始位置
        causal: 是否使用因果注意力掩码
        sliding_window_size: 滑动窗口大小
        isa: 指令集架构字符串（如"avx512"）
        enable_kv_split: 是否启用KV分割

    返回:
        torch.Tensor: 调度元数据张量
    """
    # 调用底层C++算子计算CPU注意力调度元数据
    scheduler_metadata = torch.ops._C.get_scheduler_metadata(
        num_reqs,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_lens,
        dtype,
        query_start_loc,
        causal,
        sliding_window_size,
        isa,
        enable_kv_split,
    )
    # 返回调度元数据
    return scheduler_metadata


# CPU注意力的重塑和缓存：在CPU上将key/value写入分页缓存
def cpu_attn_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    isa: str,
) -> None:
    """
    在CPU上将key和value张量重塑并写入分页KV缓存。

    参数:
        key: key张量
        value: value张量
        key_cache: key缓存张量
        value_cache: value缓存张量
        slot_mapping: 槽位映射
        isa: 指令集架构字符串
    """
    # 调用底层C++算子执行CPU注意力的重塑和缓存写入
    torch.ops._C.cpu_attn_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        isa,
    )


# CPU注意力计算：使用KV缓存在CPU上执行注意力运算
def cpu_attention_with_kv_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor | None,
) -> None:
    """
    在CPU上使用分页KV缓存执行注意力计算。

    参数:
        query: query张量
        key_cache: key缓存张量
        value_cache: value缓存张量
        output: 输出张量
        query_start_loc: query的起始位置
        seq_lens: 序列长度
        scale: softmax缩放因子
        causal: 是否使用因果注意力掩码
        alibi_slopes: 可选的ALiBi位置编码斜率
        sliding_window: 滑动窗口大小，(左窗口, 右窗口)
        block_table: 块表
        softcap: softmax上限值（0表示不使用）
        scheduler_metadata: 调度元数据
        s_aux: 可选的辅助缩放张量
    """
    # 调用底层C++算子执行CPU注意力计算
    torch.ops._C.cpu_attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        causal,
        alibi_slopes,
        sliding_window[0],  # 左滑动窗口大小
        sliding_window[1],  # 右滑动窗口大小
        block_table,
        softcap,
        scheduler_metadata,
        s_aux,
    )


# CPU WNA16矩阵乘法：在CPU上执行权重N位激活16位的矩阵乘法
def cpu_gemm_wna16(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    bias: torch.Tensor | None,
    pack_factor: int,
    isa_hint: str,
) -> torch.Tensor:
    """
    在CPU上执行WNA16（权重N位，激活16位）量化矩阵乘法。

    参数:
        input: 输入激活张量
        q_weight: 量化权重张量
        scales: 缩放因子
        zeros: 可选的零点
        g_idx: 可选的分组索引
        bias: 可选的偏置
        pack_factor: 权重打包因子（每个元素包含的量化值数）
        isa_hint: 指令集提示字符串

    返回:
        torch.Tensor: 矩阵乘法结果
    """
    # 创建输出张量，列数由缩放因子的列数决定
    output = torch.empty((input.size(0), scales.size(1)), dtype=input.dtype)
    # 调用底层C++算子执行CPU WNA16矩阵乘法
    torch.ops._C.cpu_gemm_wna16(
        input,
        q_weight,
        output,
        scales,
        zeros,
        g_idx,
        bias,
        pack_factor,
        isa_hint,
    )
    # 返回矩阵乘法结果
    return output


# CPU MoE权重预打包：将MoE权重重排为CPU内核友好的格式
def cpu_prepack_moe_weight(
    weight: torch.Tensor,
    isa: str,
) -> torch.Tensor:
    """
    将MoE专家权重预打包为CPU内核优化的内存布局。

    参数:
        weight: 原始权重张量
        isa: 指令集架构字符串

    返回:
        torch.Tensor: 预打包后的权重张量
    """
    # 创建与输入形状相同的输出张量
    output = torch.empty_like(weight)
    # 调用底层C++算子执行MoE权重预打包
    torch.ops._C.prepack_moe_weight(weight, output, isa)
    return output


# CPU融合MoE：在CPU上执行融合的MoE专家计算
def cpu_fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act: str,
    isa: str,
    skip_weighted: bool = False,
) -> torch.Tensor:
    """
    在CPU上执行融合的MoE（混合专家模型）计算，包括门控、激活和专家计算。

    参数:
        input: 输入张量
        w13: gate和up投影的融合权重
        w2: down投影权重
        w13_bias: 可选的gate/up偏置
        w2_bias: 可选的down偏置
        topk_weights: top-k专家权重
        topk_ids: top-k专家ID
        act: 激活函数类型字符串（如"silu"）
        isa: 指令集架构字符串
        skip_weighted: 是否跳过加权求和

    返回:
        torch.Tensor: MoE计算结果
    """
    # 创建与输入形状相同的输出张量
    output = torch.empty_like(input)
    # 调用底层C++算子执行CPU融合MoE计算
    torch.ops._C.cpu_fused_moe(
        output,
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        skip_weighted,
        act,
        isa,
    )
    # 返回MoE计算结果
    return output


# 注册MXF4 BF16矩阵乘法（TN布局）的fake实现
if hasattr(torch.ops._qutlass_C, "matmul_mxf4_bf16_tn"):

    @register_fake("_qutlass_C::matmul_mxf4_bf16_tn")
    def _fake_matmul_mxf4_bf16_tn(
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
    ):
        """
        MXF4 BF16矩阵乘法的fake实现，用于编译时形状推断。

        返回:
            形状为(*a.shape[:-1], b.shape[0])的BF16空张量
        """
        # 返回与实际输出形状相同的空张量
        return a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.bfloat16)


# MXF4 BF16矩阵乘法（TN布局）：使用MX微缩放浮点4位格式
def matmul_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    执行MXF4格式的BF16矩阵乘法（TN布局），使用CUTLASS内核。

    参数:
        a: 输入矩阵A（MXF4格式）
        b: 输入矩阵B（MXF4格式）
        a_sf: 矩阵A的缩放因子
        b_sf: 矩阵B的缩放因子
        alpha: 全局缩放因子

    返回:
        torch.Tensor: BF16格式的矩阵乘法结果
    """
    # 调用qutlass C++算子执行MXF4 BF16矩阵乘法
    return torch.ops._qutlass_C.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


# 注册Ada MXF4 BF16矩阵乘法（TN布局）的fake实现
if hasattr(torch.ops._qutlass_C, "matmul_ada_mxf4_bf16_tn"):

    @register_fake("_qutlass_C::matmul_ada_mxf4_bf16_tn")
    def _fake_matmul_ada_mxf4_bf16_tn(
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
    ):
        """
        Ada MXF4 BF16矩阵乘法的fake实现，用于编译时形状推断。

        返回:
            形状为(*a.shape[:-1], b.shape[0])的BF16空张量
        """
        # 返回与实际输出形状相同的空张量
        return a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.bfloat16)


# Ada MXF4 BF16矩阵乘法（TN布局）：Ada架构的MX微缩放浮点4位格式
def matmul_ada_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    执行Ada架构的MXF4格式BF16矩阵乘法（TN布局）。

    参数:
        a: 输入矩阵A（MXF4格式）
        b: 输入矩阵B（MXF4格式）
        a_sf: 矩阵A的缩放因子
        b_sf: 矩阵B的缩放因子
        alpha: 全局缩放因子

    返回:
        torch.Tensor: BF16格式的矩阵乘法结果
    """
    # 调用qutlass C++算子执行Ada MXF4 BF16矩阵乘法
    return torch.ops._qutlass_C.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


# 注册MX Quest量化的fake实现
if hasattr(torch.ops._qutlass_C, "fusedQuantizeMxQuest"):

    @register_fake("_qutlass_C::fusedQuantizeMxQuest")
    def _fake_fused_quantize_mx_quest(
        a: torch.Tensor, b: torch.Tensor, xh_e2m1: torch.Tensor, xh_e8m0: torch.Tensor
    ):
        """
        MX Quest量化的fake实现，用于编译时形状推断。

        返回:
            (e2m1量化数据, e8m0缩放因子) 的元组
        """
        # 直接返回输出张量（形状已由调用者确定）
        return xh_e2m1, xh_e8m0


# 注册MX AbsMax量化的fake实现
if hasattr(torch.ops._qutlass_C, "fusedQuantizeMxAbsMax"):

    @register_fake("_qutlass_C::fusedQuantizeMxAbsMax")
    def _fake_fused_quantize_mx_absmax(
        a: torch.Tensor, b: torch.Tensor, xh_e2m1: torch.Tensor, xh_e8m0: torch.Tensor
    ):
        """
        MX AbsMax量化的fake实现，用于编译时形状推断。

        返回:
            (e2m1量化数据, e8m0缩放因子) 的元组
        """
        # 直接返回输出张量
        return xh_e2m1, xh_e8m0


# 融合MX量化：将张量量化为MX（微缩放）浮点格式
def fusedQuantizeMx(
    a: torch.Tensor, b: torch.Tensor, *, method: Literal["quest", "abs_max"] = "quest"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将张量融合量化为MX（微缩放）浮点格式，支持Quest和AbsMax两种方法。

    参数:
        a: 输入张量，最后一维必须能被32整除
        b: 辅助张量（与a在同一设备上）
        method: 量化方法，"quest"使用Quest算法，"abs_max"使用绝对值最大值算法

    返回:
        tuple[torch.Tensor, torch.Tensor]: (e2m1量化数据, e8m0缩放因子)
    """
    # 检查输入维度有效性
    if a.dim() == 0:
        raise ValueError("`a` must have at least 1 dimension.")
    # 检查最后一维是否能被32整除（MX格式的块大小要求）
    if a.size(-1) % 32 != 0:
        raise ValueError(f"last dim of `a` must be divisible by 32, got {a.size(-1)}.")
    # 检查两个张量是否在同一设备上
    if b.device != a.device:
        raise ValueError("`a` and `b` must be on the same device.")

    # 创建e2m1格式的量化数据张量（2个值打包为1个uint8）
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )

    # 计算缩放因子张量的维度（按128x4的块对齐）
    rows, cols = a.numel() // a.size(-1), a.size(-1) // 32
    n_row_blocks = cdiv(rows, 128)  # 向上取整到128的倍数
    n_col_blocks = cdiv(cols, 4)    # 向上取整到4的倍数
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # 创建e8m0格式的缩放因子张量
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=a.device
    )

    # 检查qutlass扩展是否已加载
    if not hasattr(torch.ops, "_qutlass_C"):
        raise RuntimeError(
            "The `_qutlass_C` extension is not loaded. "
            "Make sure your custom op library is imported before calling fusedQuantizeMx."
        )

    # 根据方法选择对应的量化算子
    if method == "quest":
        # 使用Quest量化方法
        return torch.ops._qutlass_C.fusedQuantizeMxQuest(a, b, xh_e2m1, xh_e8m0)
    elif method == "abs_max":
        # 使用AbsMax量化方法
        return torch.ops._qutlass_C.fusedQuantizeMxAbsMax(a, b, xh_e2m1, xh_e8m0)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


# 注册NV（NVIDIA）融合量化的fake实现
if hasattr(torch.ops._qutlass_C, "fusedQuantizeNv"):

    @register_fake("_qutlass_C::fusedQuantizeNv")
    def _fake_fused_quantize_nv(
        a: torch.Tensor,
        b: torch.Tensor,
        xh_e2m1: torch.Tensor,
        xh_e4m3: torch.Tensor,
        global_scale: torch.Tensor,
    ):
        """
        NV融合量化的fake实现，用于编译时形状推断。

        返回:
            (e2m1量化数据, e4m3缩放因子) 的元组
        """
        # 直接返回输出张量
        return xh_e2m1, xh_e4m3


# NV融合量化：NVIDIA专用的MX浮点量化
def fusedQuantizeNv(
    a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    执行NVIDIA专用的MX浮点融合量化，使用全局缩放因子。

    参数:
        a: 输入张量
        b: 辅助张量
        global_scale: 全局缩放因子

    返回:
        tuple[torch.Tensor, torch.Tensor]: (e2m1量化数据, e4m3缩放因子)
    """
    # 创建e2m1格式的量化数据张量（2个值打包为1个uint8）
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )

    # 计算缩放因子张量的维度（按128x4的块对齐，NV格式每组16个元素）
    rows, cols = a.numel() // a.size(-1), a.size(-1) // 16
    n_row_blocks = cdiv(rows, 128)  # 向上取整到128的倍数
    n_col_blocks = cdiv(cols, 4)    # 向上取整到4的倍数
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    # 创建e4m3格式的缩放因子张量
    xh_e4m3 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=a.device
    )

    # 调用qutlass C++算子执行NV融合量化
    return torch.ops._qutlass_C.fusedQuantizeNv(a, b, xh_e2m1, xh_e4m3, global_scale)


# Hadacore变换：使用Hadamard变换进行高效矩阵运算
def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    使用Hadacore内核执行Hadamard变换，利用Sylvester Hadamard矩阵的递归性质，无需变换权重数据。

    Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)
    kernels. Note that these kernels exploit the recursive properties of
    Sylvester Hadamards, and therefore do not require transform weight data

    Note that sylvester hadamard transforms are also symmetric, which means that
    this function is also applies the (transpose <=> inverse) transform.

    :param x: value to be transformed inplace
    :param inplace: modify value in place
    :return: value after transformation
    """
    # 调用底层C++算子执行Hadacore变换
    return torch.ops._C.hadacore_transform(x, inplace)


# 注册Hadacore变换的fake实现
if hasattr(torch.ops._C, "hadacore_transform"):

    @register_fake("_C::hadacore_transform")
    def _hadacore_transform_fake(x: torch.Tensor, inplace: bool) -> torch.Tensor:
        """
        Hadacore变换的fake实现，用于编译时形状推断。

        返回:
            原地操作时返回输入张量本身，否则返回形状相同的空张量
        """
        # 非原地操作返回新张量，原地操作返回输入本身
        return torch.empty_like(x) if not inplace else x
