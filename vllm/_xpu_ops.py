# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU（Intel GPU）自定义算子封装模块。
提供XPU平台的Flash Attention、FP8/INT4 GEMM、KV缓存量化等算子的实现。
"""

from typing import TYPE_CHECKING  # 导入类型检查标志

import torch  # 导入PyTorch框架
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func  # 导入XPU平台的变长Flash Attention函数

from vllm.logger import init_logger  # 导入vLLM日志初始化函数
from vllm.platforms import current_platform  # 导入当前平台信息

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

if TYPE_CHECKING:  # 仅在类型检查时执行（用于IDE静态分析）

    def register_fake(fn):  # 定义虚拟的注册函数用于类型检查
        return lambda name: fn  # 返回一个简单的lambda装饰器
else:  # 运行时执行
    try:  # 尝试从torch.library导入register_fake
        from torch.library import register_fake  # 导入fake张量注册函数
    except ImportError:  # 如果导入失败（旧版PyTorch）
        from torch.library import impl_abstract as register_fake  # 使用旧版API作为替代

if hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):  # 检查XPU是否注册了FP8 GEMM算子

    @register_fake("_xpu_C::fp8_gemm_w8a16")  # 注册FP8 GEMM算子的fake实现（用于torch.compile）
    def _fp8_gemm_w8a16_fake(
        input: torch.Tensor,  # 输入张量
        q_weight: torch.Tensor,  # 量化权重张量
        weight_scale: torch.Tensor,  # 权重缩放因子
        bias: torch.Tensor | None = None,  # 可选的偏置张量
    ) -> torch.Tensor:
        """FP8权重（W8A16）GEMM算子的fake张量实现。
        用于torch.compile追踪时推断输出形状。
        """
        input_2d = input.view(-1, input.shape[-1])  # 将输入展平为二维张量
        M = input_2d.size(0)  # 获取批次维度大小
        N = q_weight.size(1)  # 获取输出维度大小
        return torch.empty((M, N), dtype=input.dtype, device=input.device)  # 返回正确形状的空张量


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):  # 检查XPU是否注册了INT4 GEMM算子

    @register_fake("_xpu_C::int4_gemm_w4a16")  # 注册INT4 GEMM算子的fake实现
    def _int4_gemm_w4a16_fake(
        input: torch.Tensor,  # 输入张量
        q_weight: torch.Tensor,  # INT4量化权重张量
        bias: torch.Tensor | None,  # 可选的偏置张量
        weight_scale: torch.Tensor,  # 权重缩放因子
        qzeros: torch.Tensor,  # 量化零点
        group_size: int,  # 量化分组大小
        group_idx: torch.Tensor | None = None,  # 可选的分组索引
    ) -> torch.Tensor:
        """INT4权重（W4A16）GEMM算子的fake张量实现。
        用于torch.compile追踪时推断输出形状。
        """
        input_2d = input.view(-1, input.shape[-1])  # 将输入展平为二维张量
        M = input_2d.size(0)  # 获取批次维度大小
        N = q_weight.size(1)  # 获取输出维度大小
        return torch.empty((M, N), dtype=input.dtype, device=input.device)  # 返回正确形状的空张量


class xpu_ops:
    """XPU平台算子集合类。
    提供Flash Attention、调度器元数据、KV缓存量化等XPU专用算子的静态方法实现。
    """

    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens_q: torch.Tensor,  # 查询序列的累积长度
        max_seqlen_q: int,  # 查询的最大序列长度
        max_seqlen_k: int,  # 键的最大序列长度
        softmax_scale: float | None = None,  # softmax缩放因子
        causal: bool = False,  # 是否使用因果注意力掩码
        out: torch.Tensor | None = None,  # 可选的输出张量
        block_table: torch.Tensor | None = None,  # 分页注意力的块表
        alibi_slopes: torch.Tensor | None = None,  # ALiBi位置编码斜率
        window_size: list[int] | None = None,  # 滑动窗口大小
        softcap: float | None = 0.0,  # softmax上限值
        seqused_k: torch.Tensor | None = None,  # 键序列的实际使用长度
        cu_seqlens_k: torch.Tensor | None = None,  # 键序列的累积长度
        # passed in qwen vl  # 在Qwen VL中传入的参数
        dropout_p: float = 0.0,  # dropout概率
        # The following parameters are not used in xpu kernel currently,
        # we keep API compatible to CUDA's.
        # 以下参数目前在XPU内核中未使用，保持与CUDA API兼容
        scheduler_metadata=None,  # 调度器元数据（XPU未使用）
        fa_version: int = 2,  # Flash Attention版本（XPU未使用）
        q_descale=None,  # 查询反量化缩放（XPU未使用）
        k_descale=None,  # 键反量化缩放（XPU未使用）
        v_descale=None,  # 值反量化缩放（XPU未使用）
        num_splits=0,  # 分割数量（XPU未使用）
        return_softmax_lse: bool | None = False,  # 是否返回softmax的log-sum-exp
        s_aux: torch.Tensor | None = None,  # 辅助张量
    ):
        """XPU平台的变长序列Flash Attention实现。
        支持分页KV缓存和因果注意力掩码。

        Args:
            q: 查询张量。
            k: 键张量。
            v: 值张量。
            cu_seqlens_q: 查询序列的累积长度。
            max_seqlen_q: 查询的最大序列长度。
            max_seqlen_k: 键的最大序列长度。
            softmax_scale: softmax缩放因子。
            causal: 是否启用因果掩码。
            out: 预分配的输出张量。
            block_table: 分页注意力的块索引表。
            seqused_k: 键序列的实际使用长度。
            cu_seqlens_k: 键序列的累积长度。
        """
        assert cu_seqlens_k is not None or seqused_k is not None, (  # 断言：必须提供cu_seqlens_k或seqused_k之一
            "cu_seqlens_k or seqused_k must be provided"
        )
        assert cu_seqlens_k is None or seqused_k is None, (  # 断言：不能同时提供cu_seqlens_k和seqused_k
            "cu_seqlens_k and seqused_k cannot be provided at the same time"
        )
        assert block_table is None or seqused_k is not None, (  # 断言：启用块表时需要seqused_k
            "when enable block_table, seqused_k is needed"
        )
        assert block_table is not None or cu_seqlens_k is not None, (  # 断言：禁用块表时需要cu_seqlens_k
            "when block_table is disabled, cu_seqlens_k is needed"
        )
        if out is None:  # 如果未提供输出张量
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)  # 创建与查询同形状的空张量
        real_window_size: tuple[int, int]  # 声明实际窗口大小的类型
        if window_size is None:  # 如果未指定窗口大小
            real_window_size = (-1, -1)  # 使用-1表示无限上下文窗口
        else:  # 如果指定了窗口大小
            assert len(window_size) == 2  # 断言窗口大小为二元组
            real_window_size = (window_size[0], window_size[1])  # noqa: F841  # 转换为元组

        # In encode attention, k and v maybe not contiguous and current
        # kernel can't handle it
        # 在编码注意力中，k和v可能不连续，当前内核无法处理非连续张量
        if block_table is None:  # 如果没有使用分页注意力
            k = k.contiguous()  # 确保键张量内存连续
            v = v.contiguous()  # 确保值张量内存连续
        return flash_attn_varlen_func(  # 调用XPU Flash Attention内核
            out=out,  # 输出张量
            q=q.contiguous(),  # 确保查询张量连续
            k=k,  # 键张量
            v=v,  # 值张量
            cu_seqlens_q=cu_seqlens_q,  # 查询累积序列长度
            cu_seqlens_k=cu_seqlens_k,  # 键累积序列长度
            seqused_k=seqused_k,  # 键实际使用长度
            max_seqlen_q=max_seqlen_q,  # 查询最大序列长度
            max_seqlen_k=max_seqlen_k,  # 键最大序列长度
            softmax_scale=softmax_scale,  # softmax缩放因子
            causal=causal,  # 因果掩码标志
            block_table=block_table,  # 分页块表
            s_aux=s_aux,  # 辅助张量
            window_size=real_window_size,  # 滑动窗口大小
            # alibi_slopes = alibi_slopes,  # ALiBi斜率（暂未启用）
            # softcap=softcap,  # softmax上限（暂未启用）
            return_softmax_lse=return_softmax_lse,  # 是否返回log-sum-exp
        )

    @staticmethod
    def get_scheduler_metadata(
        batch_size,  # 批次大小
        max_seqlen_q,  # 查询最大序列长度
        max_seqlen_k,  # 键最大序列长度
        num_heads_q,  # 查询头数量
        num_heads_kv,  # KV头数量
        headdim,  # 头维度
        cache_seqlens: torch.Tensor,  # 缓存序列长度
        qkv_dtype=torch.bfloat16,  # QKV数据类型
        headdim_v=None,  # 值的头维度（可选）
        cu_seqlens_q: torch.Tensor | None = None,  # 查询累积序列长度
        cu_seqlens_k_new: torch.Tensor | None = None,  # 新键累积序列长度
        cache_leftpad: torch.Tensor | None = None,  # 缓存左填充
        page_size: int | None = None,  # 页大小
        max_seqlen_k_new=0,  # 新键的最大序列长度
        causal=False,  # 是否因果
        window_size=(-1, -1),  # -1 means infinite context window  # 滑动窗口大小，-1表示无限上下文
        has_softcap=False,  # 是否有softmax上限
        num_splits=0,  # Can be tuned for speed  # 分割数量，可调优性能
        pack_gqa=None,  # Can be tuned for speed  # GQA打包选项，可调优性能
        sm_margin=0,  # Can be tuned if some SMs are used for communication  # SM边距，部分SM用于通信时可调
    ) -> None:
        """获取调度器元数据。
        XPU平台尚未实现此功能，返回None并发出警告。
        """
        logger.warning_once(  # 仅警告一次
            "get_scheduler_metadata is not implemented for xpu_ops, returning None."  # 提示此功能未实现
        )
        return None  # 返回None

    @staticmethod
    def indexer_k_quant_and_cache(
        k: torch.Tensor,  # 键张量
        kv_cache: torch.Tensor,  # KV缓存张量
        slot_mapping: torch.Tensor,  # 槽位映射
        quant_block_size: int,  # 量化块大小
        scale_fmt: str | None,  # 缩放格式
    ) -> None:
        """对键进行量化并写入KV缓存。
        将键张量进行FP8分组量化，然后将量化后的数据和缩放因子存入KV缓存。

        Args:
            k: 键张量。
            kv_cache: KV缓存张量。
            slot_mapping: 将token映射到缓存槽位的索引。
            quant_block_size: 量化分组大小。
            scale_fmt: 缩放因子格式（如"ue8m0"）。
        """
        head_dim = k.shape[-1]  # 获取头维度大小
        k = k.view(-1, head_dim)  # [total_tokens, head_dim]  # 将键张量重塑为二维

        def group_quant_torch(
            x: torch.Tensor,  # 输入张量
            group_size: int,  # 分组大小
            eps: float = 1e-10,  # 防止除零的epsilon
            dtype: torch.dtype | None = None,  # 目标量化数据类型
            column_major_scales: bool = False,  # 是否使用列主序缩放
            out_q: torch.Tensor | None = None,  # 预分配的量化输出
            use_ue8m0: bool | None = None,  # 是否使用UE8M0格式
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """基于PyTorch的分组量化实现。
            将输入张量按组进行FP8量化，返回量化后的张量和缩放因子。

            Args:
                x: 输入张量。
                group_size: 每组的元素数量。
                eps: 防止除零的小数值。
                dtype: 目标FP8数据类型。
                column_major_scales: 缩放因子是否使用列主序。
                out_q: 预分配的量化输出张量。
                use_ue8m0: 是否使用UE8M0缩放格式。

            Returns:
                (量化张量, 缩放因子)元组。
            """
            if use_ue8m0 is None:  # 如果未指定UE8M0选项
                # Default fallback - could import is_deep_gemm_e8m0_used if needed
                # 默认回退 - 需要时可导入is_deep_gemm_e8m0_used
                use_ue8m0 = False  # 默认不使用UE8M0

            if dtype is None:  # 如果未指定量化数据类型
                dtype = current_platform.fp8_dtype()  # 使用当前平台的FP8数据类型

            # Validate inputs  # 验证输入
            assert x.shape[-1] % group_size == 0, (  # 断言最后一维能被分组大小整除
                f"Last dimension {x.shape[-1]} must be divisible by "
                f"group_size {group_size}"
            )
            assert x.stride(-1) == 1, "Input tensor groups must be contiguous"  # 断言输入张量的最后一维是连续的

            # Prepare output tensor  # 准备输出张量
            if out_q is None:  # 如果未提供预分配输出
                x_q = torch.empty_like(x, dtype=dtype)  # 创建与输入同形状的空量化张量
            else:  # 如果提供了预分配输出
                assert out_q.shape == x.shape  # 断言形状匹配
                x_q = out_q  # 使用预分配的输出张量

            # Reshape input for group processing
            # Original shape: (..., last_dim)
            # Target shape: (..., num_groups, group_size)
            # 重塑输入以进行分组处理
            # 原始形状: (..., last_dim) -> 目标形状: (..., num_groups, group_size)
            original_shape = x.shape  # 保存原始形状
            num_groups = original_shape[-1] // group_size  # 计算分组数量

            # Reshape to separate groups  # 重塑以分离各组
            group_shape = original_shape[:-1] + (num_groups, group_size)  # 构建分组形状
            x_grouped = x.view(group_shape)  # 将输入重塑为分组形式

            # Compute per-group absolute maximum values
            # Shape: (..., num_groups)
            # 计算每组的绝对最大值，形状: (..., num_groups)
            abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)  # 计算每组绝对值的最大值
            abs_max = torch.maximum(  # 确保最大值不小于epsilon
                abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype)
            )

            # Compute scales  # 计算缩放因子
            FP8_MAX = torch.finfo(dtype).max  # 获取FP8数据类型的最大值
            FP8_MIN = torch.finfo(dtype).min  # 获取FP8数据类型的最小值
            scale_raw = abs_max / FP8_MAX  # 计算原始缩放因子

            if use_ue8m0:  # 如果使用UE8M0格式
                # For UE8M0 format, scales must be powers of 2
                # UE8M0格式要求缩放因子必须是2的幂
                scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))  # 向上取整到最近的2的幂
            else:  # 不使用UE8M0格式
                scales = scale_raw  # 直接使用原始缩放因子

            # Expand scales for broadcasting with grouped data
            # Shape: (..., num_groups, 1)
            # 扩展缩放因子以便与分组数据广播，形状: (..., num_groups, 1)
            scales_expanded = scales.unsqueeze(-1)  # 在最后添加一个维度

            # Quantize the grouped data  # 量化分组数据
            x_scaled = x_grouped / scales_expanded  # 将数据除以缩放因子
            x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)  # 将值裁剪到FP8范围内
            x_quantized = x_clamped.to(dtype)  # 转换为FP8数据类型

            # Reshape back to original shape  # 重塑回原始形状
            x_q.copy_(x_quantized.view(original_shape))  # 将量化结果复制到输出张量

            # Prepare scales tensor in requested format  # 按请求的格式准备缩放因子张量
            if column_major_scales:  # 如果使用列主序缩放
                # Column-major: (num_groups,) + batch_dims
                # 列主序: (num_groups,) + 批次维度
                # Transpose the scales to put group dimension first
                # 转置缩放因子，将分组维度放在最前面
                scales_shape = (num_groups,) + original_shape[:-1]  # 构建列主序形状
                x_s = scales.permute(-1, *range(len(original_shape) - 1))  # 转置维度
                x_s = x_s.contiguous().view(scales_shape)  # 确保连续并重塑
            else:  # 行主序
                # Row-major: batch_dims + (num_groups,)
                # 行主序: 批次维度 + (num_groups,)
                x_s = scales.contiguous()  # 确保缩放因子连续

            # Ensure scales are float32  # 确保缩放因子为float32类型
            return x_q, x_s.float()  # 返回量化张量和float32缩放因子

        k_fp8, k_scale = group_quant_torch(  # 对键进行分组量化
            k,  # 输入键张量
            group_size=quant_block_size,  # 量化分组大小
            column_major_scales=False,  # 使用行主序缩放
            use_ue8m0=(scale_fmt == "ue8m0"),  # 根据格式决定是否使用UE8M0
        )

        k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)  # 将FP8键数据视为uint8字节
        scale_bytes = k_scale.view(torch.uint8).view(-1, 4)  # 将缩放因子视为uint8字节
        k = torch.cat(  # 拼接量化数据和缩放因子
            [k_fp8_bytes, scale_bytes], dim=-1
        )  # [total_tokens, head_dim + 4]  # 结果形状：[总token数, 头维度+4]

        slot_mapping = slot_mapping.flatten()  # 将槽位映射展平为一维
        # kv_cache: [num_block, block_size, head_dim + 4]
        # kv_cache形状: [块数, 块大小, 头维度+4]
        kv_cache.view(-1, kv_cache.shape[-1]).index_copy_(0, slot_mapping, k)  # 将量化后的键按槽位写入KV缓存

    @staticmethod
    def cp_gather_indexer_k_quant_cache(
        kv_cache: torch.Tensor,  # 量化KV缓存张量
        dst_k: torch.Tensor,  # 目标键输出张量
        dst_scale: torch.Tensor,  # 目标缩放因子输出张量
        block_table: torch.Tensor,  # 块索引表
        cu_seq_lens: torch.Tensor,  # 累积序列长度
    ) -> None:
        """从量化KV缓存中收集键值和缩放因子。
        根据块表和序列长度信息，从分页KV缓存中提取量化的键数据和对应的缩放因子。

        Args:
            kv_cache: [num_blocks, block_size, cache_stride] - 量化KV缓存。
                    每个块的布局: [k_values, scale_values]
                    - k_values: [block_size * head_dim]
                    - scale_values: [block_size * head_dim * 4 / quant_block_size]
            dst_k: [num_tokens, head_dim] - 键值的输出张量。
            dst_scale: [num_tokens, head_dim / quant_block_size * 4] - 缩放因子的输出张量。
            block_table: [batch_size, num_blocks] - 块索引表。
            cu_seq_lens: [batch_size + 1] - 累积序列长度。
        """
        batch_size = block_table.size(0)  # 获取批次大小
        num_tokens = dst_k.size(0)  # 获取总token数
        head_dim = dst_k.size(1)  # 获取头维度大小
        cache_block_size = kv_cache.size(1)  # 获取缓存块大小
        quant_block_size = head_dim * 4 // dst_scale.size(1)  # 计算量化块大小

        # For each token, find which batch it belongs to using searchsorted
        # 使用searchsorted为每个token找到它所属的批次
        token_indices = torch.arange(num_tokens, device=dst_k.device) + 1  # 创建token索引（从1开始）
        # cu_seq_lens is [batch_size + 1], we need to find which interval each
        # token belongs to
        # cu_seq_lens形状为[batch_size + 1]，需要找到每个token属于哪个区间
        batch_indices = torch.searchsorted(cu_seq_lens, token_indices) - 1  # 通过二分查找确定批次索引
        batch_indices = torch.clamp(batch_indices, 0, batch_size - 1)  # 将批次索引裁剪到有效范围

        # Calculate the in-batch sequence index for each token
        # 计算每个token在其批次内的序列索引
        inbatch_seq_indices = token_indices - cu_seq_lens[batch_indices]  # 批次内的相对位置

        # Find which block each token belongs to
        # 找到每个token所属的块
        block_indices_in_table = inbatch_seq_indices // cache_block_size  # 计算块表中的块索引
        physical_block_indices = block_table[batch_indices, block_indices_in_table]  # 获取物理块索引

        # Calculate the offset within each block
        # 计算每个token在块内的偏移量
        inblock_offsets = (inbatch_seq_indices - 1) % cache_block_size  # 块内偏移

        # Calculate strides  # 计算步长
        block_stride = kv_cache.stride(0)  # stride for each block  # 每个块的步长

        # Flatten kv_cache for easier indexing
        # 展平kv_cache以便于索引
        kv_cache_flat = kv_cache.view(-1)  # 展平为一维张量

        # Calculate source offset for K values for all tokens (vectorized)
        # 向量化计算所有token的K值源偏移量
        src_block_offsets = physical_block_indices * block_stride  # 块的起始偏移
        src_k_offsets = src_block_offsets + inblock_offsets * head_dim  # K值的偏移

        # Gather K values using advanced indexing
        # Create indices for all elements we need to gather
        # 使用高级索引收集K值，创建所有需要收集元素的索引
        k_indices = src_k_offsets.unsqueeze(1) + torch.arange(  # 构建K值的完整索引
            head_dim, device=dst_k.device
        )
        dst_k[:] = kv_cache_flat[k_indices]  # 从展平的缓存中收集K值

        # Calculate source offset for scale values (vectorized)
        # Scales are stored after all K values for each block
        # 向量化计算缩放因子的源偏移量（缩放因子存储在每个块的K值之后）
        scale_size = head_dim * 4 // quant_block_size  # 计算缩放因子的大小
        src_scale_offsets = src_block_offsets + head_dim + inblock_offsets * scale_size  # 缩放因子的偏移

        # Gather scale values  # 收集缩放因子
        scale_indices = src_scale_offsets.unsqueeze(1) + torch.arange(  # 构建缩放因子的完整索引
            scale_size, device=dst_scale.device
        )
        dst_scale[:] = kv_cache_flat[scale_indices]  # 从展平的缓存中收集缩放因子

    @staticmethod
    def top_k_per_row_prefill(
        logits: torch.Tensor,  # 注意力分数/logits张量
        cu_seqlen_ks: torch.Tensor,  # 键序列的累积起始位置
        cu_seqlen_ke: torch.Tensor,  # 键序列的累积结束位置
        raw_topk_indices: torch.Tensor,  # 输出的top-k索引张量
        num_rows: int,  # 行数
        stride0: int,  # 第一维步长
        strdide1: int,  # 第二维步长
        topk_tokens: int,  # 要选取的top-k数量
    ) -> torch.Tensor:
        """预填充阶段的逐行top-k选择。
        从每行的logits中选择top-k个token，并调整索引使其相对于各自序列的起始位置。

        Args:
            logits: 注意力分数张量。
            cu_seqlen_ks: 键序列累积起始位置。
            cu_seqlen_ke: 键序列累积结束位置。
            raw_topk_indices: 用于存储结果的输出张量。
            num_rows: 行数。
            stride0: 第一维步长。
            strdide1: 第二维步长。
            topk_tokens: 要选取的top-k数量。
        """
        real_topk = min(topk_tokens, logits.shape[-1])  # 取top-k和logits维度的较小值
        topk_indices = logits.topk(real_topk, dim=-1)[1].to(torch.int32)  # 获取top-k的索引并转为int32
        topk_indices -= cu_seqlen_ks[:, None]  # 将全局索引转换为相对于各序列起始位置的局部索引
        mask_lo = topk_indices >= 0  # 创建下界掩码（索引不小于0）
        mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0  # 创建上界掩码（索引不超过序列长度）
        mask = torch.full_like(  # 创建全False的掩码张量
            topk_indices, False, dtype=torch.bool, device=topk_indices.device
        )
        mask = mask_lo & mask_hi  # 合并上下界掩码
        topk_indices.masked_fill_(~mask, -1)  # 将超出范围的索引设为-1
        raw_topk_indices[: topk_indices.shape[0], : topk_indices.shape[1]] = (  # 将结果写入输出张量
            topk_indices
        )

    @staticmethod
    def top_k_per_row_decode(
        logits: torch.Tensor,  # 注意力分数/logits张量
        next_n: int,  # 下一步要生成的token数
        seq_lens: torch.Tensor,  # 序列长度
        raw_topk_indices: torch.Tensor,  # 输出的top-k索引张量
        num_rows: int,  # 行数
        stride0: int,  # 第一维步长
        stride1: int,  # 第二维步长
        topk_tokens: int,  # 要选取的top-k数量
    ) -> torch.Tensor:
        """解码阶段的逐行top-k选择。
        在解码阶段从每行的logits中选择top-k个token，考虑序列长度和多步生成偏移。

        Args:
            logits: 注意力分数张量。
            next_n: 每步生成的token数。
            seq_lens: 各序列的长度。
            raw_topk_indices: 用于存储结果的输出张量。
            num_rows: 行数。
            stride0: 第一维步长。
            stride1: 第二维步长。
            topk_tokens: 要选取的top-k数量。
        """
        device = logits.device  # 获取设备信息
        batch_size = seq_lens.size(0)  # 获取批次大小
        # padded query len  # 填充后的查询长度
        padded_num_tokens = batch_size * next_n  # 计算填充后的总token数
        positions = (  # 创建位置索引矩阵
            torch.arange(logits.shape[-1], device=device)  # 生成列索引
            .unsqueeze(0)  # 添加行维度
            .expand(batch_size * next_n, -1)  # 扩展到所有行
        )
        row_indices = torch.arange(padded_num_tokens, device=device) // next_n  # 计算每行对应的批次索引
        next_n_offset = torch.arange(padded_num_tokens, device=device) % next_n  # 计算多步生成的偏移量
        index_end_pos = (seq_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)  # 计算每行的结束位置索引
        # index_end_pos: [B * N, 1]  # 结束位置形状: [批次*步数, 1]
        mask = positions <= index_end_pos  # 创建位置掩码
        # mask: [B * N, L]  # 掩码形状: [批次*步数, 序列长度]
        logits = logits.masked_fill(~mask, float("-inf"))  # 将超出范围的位置填充为负无穷
        topk_indices = logits.topk(topk_tokens, dim=-1)[1].to(torch.int32)  # [B * N, K]  # 获取top-k索引
        # ensure we don't set indices for the top k
        # that is out of range(masked already)
        # this will happen if context length is shorter than K
        # 确保top-k索引不超出范围（已被掩码处理），当上下文长度小于K时会出现这种情况
        topk_indices[topk_indices > index_end_pos] = -1  # 将超出范围的索引设为-1
        raw_topk_indices[: topk_indices.shape[0], : topk_indices.shape[1]] = (  # 将结果写入输出张量
            topk_indices
        )
