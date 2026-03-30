# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
# Copyright (c) 2023, Tri Dao.  # 原始版权：Tri Dao（Flash Attention作者）
# ruff: noqa: E501  # 禁用ruff的行长度检查规则


import torch  # 导入PyTorch库

# isort: off  # 关闭isort排序（以下导入顺序有依赖要求）
# We need to import the CUDA kernels after importing torch
# Use relative import to support build-from-source installation in vLLM

try:  # 尝试导入FA2的CUDA内核扩展
    from . import _vllm_fa2_C  # type: ignore[attr-defined]  # noqa: F401  # 导入FA2 C++扩展模块

    FA2_UNAVAILABLE_REASON = None  # FA2可用，无不可用原因
    FA2_AVAILABLE = True  # 标记FA2为可用
except ImportError as e:  # 捕获导入失败异常
    FA2_UNAVAILABLE_REASON = str(e)  # 记录FA2不可用的原因
    FA2_AVAILABLE = False  # 标记FA2为不可用

try:  # 尝试导入FA3的CUDA内核扩展
    from . import _vllm_fa3_C  # type: ignore[attr-defined]  # noqa: F401  # 导入FA3 C++扩展模块

    FA3_UNAVAILABLE_REASON = None  # FA3可用，无不可用原因
    FA3_AVAILABLE = True  # 标记FA3为可用
except ImportError as e:  # 捕获导入失败异常
    FA3_UNAVAILABLE_REASON = str(e)  # 记录FA3不可用的原因
    FA3_AVAILABLE = False  # 标记FA3为不可用


try:  # 尝试检测FA4（CUTE接口）是否可用
    import os  # 导入操作系统模块

    _cute_interface_path = os.path.join(  # 构建CUTE接口文件的路径
        os.path.dirname(__file__), "cute", "interface.py"  # 在当前文件所在目录下查找cute/interface.py
    )
    if not os.path.exists(_cute_interface_path):  # 如果CUTE接口文件不存在
        raise ImportError("vllm.vllm_flash_attn.cute.interface not found")  # 抛出导入错误

    FA4_UNAVAILABLE_REASON = None  # FA4可用，无不可用原因
    FA4_AVAILABLE = True  # 标记FA4为可用
except (ImportError, ModuleNotFoundError) as e:  # 捕获导入相关异常
    FA4_UNAVAILABLE_REASON = str(e)  # 记录FA4不可用的原因
    FA4_AVAILABLE = False  # 标记FA4为不可用

# isort: on  # 重新开启isort排序

DEFAULT_FA_VERSION = 2  # 默认使用的Flash Attention版本为2


def _is_fa2_supported() -> tuple[bool, str | None]:  # 检查FA2是否在当前硬件上受支持
    """检查Flash Attention 2是否在当前硬件上受支持。

    Returns:
        一个元组，第一个元素为布尔值表示是否支持，
        第二个元素为不支持时的原因字符串（支持时为None）。
    """
    if not FA2_AVAILABLE:  # 如果FA2的C扩展不可用
        return False, f"FA2 is unavailable due to: {FA2_UNAVAILABLE_REASON}"  # 返回不可用原因
    from vllm.platforms import current_platform  # 延迟导入当前平台信息

    if not current_platform.has_device_capability(80):  # 检查设备计算能力是否>=8.0（Ampere及以上）
        return False, "FA2 is only supported on devices with compute capability >= 8"  # 返回硬件不支持信息
    return True, None  # FA2受支持，返回True


def _is_fa3_supported() -> tuple[bool, str | None]:  # 检查FA3是否在当前硬件上受支持
    """检查Flash Attention 3是否在当前硬件上受支持。

    Returns:
        一个元组，第一个元素为布尔值表示是否支持，
        第二个元素为不支持时的原因字符串（支持时为None）。
    """
    if not FA3_AVAILABLE:  # 如果FA3的C扩展不可用
        return False, f"FA3 is unavailable due to: {FA3_UNAVAILABLE_REASON}"  # 返回不可用原因
    from vllm.platforms import current_platform  # 延迟导入当前平台信息

    if not current_platform.is_device_capability_family(90):  # 检查设备计算能力是否为9.x系列（Hopper）
        return False, "FA3 is only supported on devices with compute capability 9.x"  # 返回硬件不支持信息
    return True, None  # FA3受支持，返回True


def _is_fa4_supported() -> tuple[bool, str | None]:  # 检查FA4是否在当前硬件上受支持
    """检查Flash Attention 4是否在当前硬件上受支持。

    Returns:
        一个元组，第一个元素为布尔值表示是否支持，
        第二个元素为不支持时的原因字符串（支持时为None）。
    """
    if not FA4_AVAILABLE:  # 如果FA4接口不可用
        return False, f"FA4 is unavailable due to: {FA4_UNAVAILABLE_REASON}"  # 返回不可用原因
    from vllm.platforms import current_platform  # 延迟导入当前平台信息

    if not (  # 检查设备计算能力是否为9.x、10.x或11.x系列
        current_platform.is_device_capability_family(90)  # Hopper架构（SM90）
        or current_platform.is_device_capability_family(100)  # Blackwell架构（SM100）
        or current_platform.is_device_capability_family(110)  # 未来架构（SM110）
    ):
        return (  # 返回硬件不支持信息
            False,
            "FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x",  # 仅支持SM9x/10x/11x
        )
    return True, None  # FA4受支持，返回True


def is_fa_version_supported(fa_version: int) -> bool:  # 检查指定的FA版本是否受支持
    """检查指定的Flash Attention版本是否在当前硬件上受支持。

    Args:
        fa_version: Flash Attention版本号（2、3或4）。

    Returns:
        如果指定版本受支持则返回True，否则返回False。

    Raises:
        ValueError: 如果传入不支持的版本号。
    """
    if fa_version == 2:  # 检查FA2
        return _is_fa2_supported()[0]  # 返回FA2支持状态的布尔值
    elif fa_version == 3:  # 检查FA3
        return _is_fa3_supported()[0]  # 返回FA3支持状态的布尔值
    elif fa_version == 4:  # 检查FA4
        return _is_fa4_supported()[0]  # 返回FA4支持状态的布尔值
    else:  # 未知版本
        raise ValueError(f"Unsupported FA version: {fa_version}")  # 抛出值错误异常


def fa_version_unsupported_reason(fa_version: int) -> str | None:  # 获取指定FA版本不受支持的原因
    """获取指定Flash Attention版本不受支持的原因。

    Args:
        fa_version: Flash Attention版本号（2、3或4）。

    Returns:
        如果版本不受支持，返回原因字符串；如果受支持，返回None。

    Raises:
        ValueError: 如果传入不支持的版本号。
    """
    if fa_version == 2:  # 查询FA2
        return _is_fa2_supported()[1]  # 返回FA2不支持的原因（或None）
    elif fa_version == 3:  # 查询FA3
        return _is_fa3_supported()[1]  # 返回FA3不支持的原因（或None）
    elif fa_version == 4:  # 查询FA4
        return _is_fa4_supported()[1]  # 返回FA4不支持的原因（或None）
    else:  # 未知版本
        raise ValueError(f"Unsupported FA version: {fa_version}")  # 抛出值错误异常


#
#  For vLLM we only care about `flash_attn_varlen_func` and
#   `flash_attn_with_kvcache` so we only maintain wrappers for these two.
#  对于vLLM，我们只关心`flash_attn_varlen_func`和`flash_attn_with_kvcache`，
#  因此只维护这两个函数的包装器。
#


def maybe_contiguous(x):  # 确保张量在内存中连续存储
    """确保张量在内存中是连续的。

    如果张量存在且最后一个维度的步长不为1（非连续），
    则返回连续化后的张量副本；否则直接返回原张量。

    Args:
        x: 输入张量，可以为None。

    Returns:
        连续化后的张量，或原始张量（如果已经连续或为None）。
    """
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x  # 非连续时调用contiguous()使其连续


# NOTE only used in FA3  # 注意：此函数仅在FA3中使用
def get_scheduler_metadata(  # 获取FA3调度器元数据
    batch_size,  # 批次大小
    max_seqlen_q,  # 最大查询序列长度
    max_seqlen_k,  # 最大键序列长度
    num_heads_q,  # 查询头的数量
    num_heads_kv,  # 键值头的数量
    headdim,  # 每个头的维度
    cache_seqlens: torch.Tensor,  # 缓存中每个序列的长度
    qkv_dtype=torch.bfloat16,  # QKV张量的数据类型，默认bf16
    headdim_v=None,  # 值头的维度，默认与headdim相同
    cu_seqlens_q: torch.Tensor | None = None,  # 查询序列的累积长度
    cu_seqlens_k_new: torch.Tensor | None = None,  # 新键序列的累积长度
    cache_leftpad: torch.Tensor | None = None,  # 缓存左填充量
    page_size: int | None = None,  # 分页KV缓存的页大小
    max_seqlen_k_new=0,  # 新键序列的最大长度
    causal=False,  # 是否使用因果注意力掩码
    window_size=(-1, -1),  # -1 means infinite context window  # 滑动窗口大小，-1表示无限上下文窗口
    has_softcap=False,  # 是否启用softcap注意力
    num_splits=0,  # Can be tuned for speed  # 分片数量，可调优以提升速度
    pack_gqa=None,  # Can be tuned for speed  # GQA打包选项，可调优以提升速度
    sm_margin=0,  # Can be tuned if some SMs are used for communication  # SM余量，当部分SM用于通信时可调优
):
    """获取Flash Attention 3的调度器元数据。

    该函数调用FA3的C++扩展来生成调度器元数据，
    用于优化注意力计算的执行调度。

    Args:
        batch_size: 批次大小。
        max_seqlen_q: 批次中最大查询序列长度。
        max_seqlen_k: 批次中最大键序列长度。
        num_heads_q: 查询注意力头数量。
        num_heads_kv: 键值注意力头数量。
        headdim: 每个注意力头的维度。
        cache_seqlens: 每个序列在KV缓存中的长度张量。
        qkv_dtype: QKV张量的数据类型。
        headdim_v: 值头的维度，默认等于headdim。
        cu_seqlens_q: 查询序列的累积长度张量。
        cu_seqlens_k_new: 新键序列的累积长度张量。
        cache_leftpad: 缓存左填充张量。
        page_size: 分页KV缓存的页大小。
        max_seqlen_k_new: 新键序列的最大长度。
        causal: 是否应用因果注意力掩码。
        window_size: 滑动窗口大小元组(left, right)。
        has_softcap: 是否使用softcap。
        num_splits: 计算分片数量。
        pack_gqa: GQA打包配置。
        sm_margin: SM余量设置。

    Returns:
        调度器元数据对象，供FA3前向传播使用。
    """
    cache_seqlens = maybe_contiguous(cache_seqlens)  # 确保缓存序列长度张量是连续的
    if headdim_v is None:  # 如果未指定值头维度
        headdim_v = headdim  # 使用与键头相同的维度
    scheduler_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(  # 调用FA3 C++扩展获取调度元数据
        batch_size,  # 批次大小
        max_seqlen_q,  # 最大查询序列长度
        max_seqlen_k,  # 最大键序列长度
        num_heads_q,  # 查询头数量
        num_heads_kv,  # 键值头数量
        headdim,  # 键头维度
        headdim_v,  # 值头维度
        qkv_dtype,  # QKV数据类型
        cache_seqlens,  # 缓存序列长度
        cu_seqlens_q,  # 查询累积序列长度
        None,  # cu_seqlens_k  # 键累积序列长度（未使用，传None）
        cu_seqlens_k_new,  # 新键累积序列长度
        None,  # seqused_q  # 已使用的查询序列长度（未使用，传None）
        cache_leftpad,  # 缓存左填充
        page_size,  # 页大小
        max_seqlen_k_new,  # 新键最大序列长度
        causal,  # 因果掩码标志
        window_size[0],  # 左窗口大小
        window_size[1],  # 右窗口大小
        has_softcap,  # softcap标志
        num_splits,  # 分片数量
        pack_gqa,  # GQA打包配置
        sm_margin,  # SM余量
    )

    return scheduler_metadata  # 返回调度器元数据


def flash_attn_varlen_func(  # 变长序列的Flash Attention函数
    q,  # 查询张量 (total_q, nheads, headdim)
    k,  # 键张量 (total_k, nheads_k, headdim)
    v,  # 值张量 (total_k, nheads_k, headdim)
    max_seqlen_q,  # 最大查询序列长度
    cu_seqlens_q,  # 查询序列的累积长度
    max_seqlen_k,  # 最大键序列长度
    cu_seqlens_k=None,  # only used for non-paged prefill  # 键序列的累积长度（仅用于非分页预填充）
    seqused_k=None,  # 每个序列实际使用的键长度
    q_v=None,  # 可选的查询-值张量（FA3特有）
    dropout_p=0.0,  # Dropout概率，评估时应设为0.0
    softmax_scale=None,  # softmax缩放因子，默认为1/sqrt(headdim)
    causal=False,  # 是否应用因果注意力掩码
    window_size: list[int] | None = None,  # 滑动窗口大小 [left, right]
    softcap=0.0,  # 0.0 means deactivated  # softcap值，0.0表示未激活
    alibi_slopes=None,  # ALiBi位置编码的斜率
    deterministic=False,  # 是否使用确定性实现
    return_attn_probs=False,  # 是否返回注意力概率（仅用于测试）
    block_table=None,  # 分页KV缓存的块表
    return_softmax_lse=False,  # 是否返回softmax的logsumexp
    out=None,  # 可选的预分配输出张量
    # FA3 Only  # 以下参数仅FA3使用
    scheduler_metadata=None,  # FA3调度器元数据
    q_descale=None,  # 查询的反量化缩放因子
    k_descale=None,  # 键的反量化缩放因子
    v_descale=None,  # 值的反量化缩放因子
    num_splits: int = 0,  # 计算分片数量
    # Version selector  # 版本选择器
    fa_version: int = DEFAULT_FA_VERSION,  # 使用的FA版本，默认为2
    s_aux=None,  # 辅助注意力分数输出（FA3特有）
    cp_world_size=1,  # 上下文并行的世界大小
    cp_rank=0,  # 上下文并行的当前rank
    cp_tot_seqused_k=None,  # 上下文并行中总的已使用键序列长度
):
    """变长序列的Flash Attention前向计算函数。

    在评估阶段dropout_p应设为0.0。
    支持多查询注意力（MQA）和分组查询注意力（GQA），
    通过传入头数较少的K、V来实现。注意Q的头数必须能被KV的头数整除。
    例如，如果Q有6个头，K和V有2个头，则Q的第0、1、2头关注K、V的第0头，
    Q的第3、4、5头关注K、V的第1头。

    如果causal=True，因果掩码会对齐到注意力矩阵的右下角。
    例如，如果seqlen_q=2且seqlen_k=5，因果掩码（1=保留，0=遮蔽）为：
        1 1 1 1 0
        1 1 1 1 1
    如果seqlen_q=5且seqlen_k=2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为零，则该行的输出也为零。

    如果window_size!=(-1,-1)，实现滑动窗口局部注意力。
    位置i的查询只关注位于
    [i+seqlen_k-seqlen_q-window_size[0], i+seqlen_k-seqlen_q+window_size[1]]
    范围内的键（含端点）。

    Args:
        q: (total_q, nheads, headdim)，total_q为批次中查询token总数。
        k: (total_k, nheads_k, headdim)，total_k为批次中键token总数。
        v: (total_k, nheads_k, headdim)，total_k为批次中值token总数。
        cu_seqlens_q: (batch_size+1,)，int32类型，批次中序列的累积查询长度。
        cu_seqlens_k: (batch_size+1,)，int32类型，批次中序列的累积键长度。
        max_seqlen_q: 批次中最大查询序列长度。
        max_seqlen_k: 批次中最大键序列长度。
        dropout_p: Dropout概率。
        softmax_scale: QK^T的缩放因子，默认为1/sqrt(headdim)。
        causal: 是否应用因果注意力掩码。
        window_size: (left, right)滑动窗口大小。(-1,-1)表示不使用。
        softcap: 大于0时激活softcapping注意力。
        alibi_slopes: (nheads,)或(batch_size, nheads)，fp32类型的ALiBi斜率。
        deterministic: 是否使用确定性反向传播实现。
        return_attn_probs: 是否返回注意力概率（仅测试用）。

    Returns:
        out: (total, nheads, headdim)。
        softmax_lse [可选]: (nheads, total_q_seqlen)，QK^T*scale矩阵每行的logsumexp。
    """
    assert cu_seqlens_k is not None or seqused_k is not None, (  # 断言：cu_seqlens_k或seqused_k至少提供一个
        "cu_seqlens_k or seqused_k must be provided"  # 错误信息：必须提供cu_seqlens_k或seqused_k
    )
    assert cu_seqlens_k is None or seqused_k is None, (  # 断言：cu_seqlens_k和seqused_k不能同时提供
        "cu_seqlens_k and seqused_k cannot be provided at the same time"  # 错误信息：两者不能同时提供
    )
    assert block_table is None or seqused_k is not None, (  # 断言：使用分页KV缓存时必须提供seqused_k
        "seqused_k must be provided if block_table is provided"  # 错误信息：提供block_table时需要seqused_k
    )

    if softmax_scale is None:  # 如果未指定softmax缩放因子
        softmax_scale = q.shape[-1] ** (-0.5)  # 使用默认值 1/sqrt(headdim)
    # custom op does not support non-tuple input  # 自定义算子不支持非元组输入
    real_window_size: tuple[int, int]  # 声明真实窗口大小的类型
    if window_size is None:  # 如果未指定窗口大小
        real_window_size = (-1, -1)  # 使用默认值(-1,-1)表示无限窗口
    else:  # 如果指定了窗口大小
        assert len(window_size) == 2  # 断言窗口大小必须包含2个元素
        real_window_size = (window_size[0], window_size[1])  # 转换为元组
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]  # 确保q、k、v张量在内存中连续

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)  # 创建一个与cu_seqlens_q形状相同的占位张量

    if fa_version == 2:  # 使用Flash Attention 2
        if (  # 检查是否传入了FA2不支持的参数
            scheduler_metadata is not None  # FA2不支持调度器元数据
            and q_descale is not None  # FA2不支持查询反量化
            and k_descale is not None  # FA2不支持键反量化
            and v_descale is not None  # FA2不支持值反量化
        ):
            raise NotImplementedError(  # 抛出未实现错误
                "FA2 does not support scheduler_metadata, q_descale, "  # FA2不支持这些特性
                "k_descale, v_descale"
            )
        if s_aux is not None:  # 如果传入了辅助注意力分数
            raise NotImplementedError("FA2 does not support s_aux")  # FA2不支持s_aux
        if num_splits > 1:  # 如果分片数量大于1
            raise NotImplementedError("FA2 does not support num_splits > 1")  # FA2不支持多分片
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(  # 调用FA2的变长前向计算内核
            q,  # 查询张量
            k,  # 键张量
            v,  # 值张量
            out,  # 预分配的输出张量（可为None）
            cu_seqlens_q,  # 查询累积序列长度
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,  # 键累积序列长度（使用seqused_k时传占位张量）
            seqused_k,  # 每个序列实际使用的键长度
            None,  # 占位参数
            block_table,  # 分页KV缓存的块表
            alibi_slopes,  # ALiBi位置编码斜率
            max_seqlen_q,  # 最大查询序列长度
            max_seqlen_k,  # 最大键序列长度
            dropout_p,  # Dropout概率
            softmax_scale,  # softmax缩放因子
            False,  # zero_tensors标志
            causal,  # 因果掩码标志
            real_window_size[0],  # 左窗口大小
            real_window_size[1],  # 右窗口大小
            softcap,  # softcap值
            return_softmax_lse and dropout_p > 0,  # 是否返回softmax概率（仅当有dropout时）
            num_splits,  # 分片数量
            None,  # 生成器（用于dropout的随机种子）
        )
    elif fa_version == 3:  # 使用Flash Attention 3
        assert alibi_slopes is None, "Alibi is not supported in FA3"  # 断言：FA3不支持ALiBi
        out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(  # 调用FA3的前向计算内核
            q,  # 查询张量
            k,  # 键张量
            v,  # 值张量
            None,  # K新增部分（未使用）
            None,  # k_new, v_new  # V新增部分（未使用）
            q_v,  # 查询-值张量
            out,  # 预分配的输出张量
            cu_seqlens_q,  # 查询累积序列长度
            cu_seqlens_k,  # cu_seqlens_k  # 键累积序列长度
            None,  # cu_seqlens_k_new  # 新键累积序列长度（未使用）
            None,  # 占位参数
            seqused_k,  # seqused_q, seqused_k  # 已使用的查询/键序列长度
            max_seqlen_q,  # 最大查询序列长度
            max_seqlen_k,  # 最大键序列长度
            block_table,  # 分页KV缓存的块表
            None,  # kv_batch_idx  # KV批次索引（未使用）
            None,  # leftpad_k  # 键左填充（未使用）
            None,  # 占位参数
            None,  # 占位参数
            None,  # rotary_cos, rotary_sin, seqlens_rotary  # 旋转位置编码参数（未使用）
            q_descale,  # 查询反量化缩放因子
            k_descale,  # 键反量化缩放因子
            v_descale,  # 值反量化缩放因子
            softmax_scale,  # softmax缩放因子
            causal,  # 因果掩码标志
            real_window_size[0],  # 左窗口大小
            real_window_size[1],  # 右窗口大小
            softcap,  # softcap值
            True,  # rotary_interleaved  # 旋转编码交错模式
            scheduler_metadata,  # 调度器元数据
            num_splits,  # 分片数量
            None,  # pack_gqa  # GQA打包配置（未使用）
            0,  # sm_margin  # SM余量
            s_aux,  # s_aux  # 辅助注意力分数
            cp_world_size,  # 上下文并行世界大小
            cp_rank,  # 上下文并行rank
            cp_tot_seqused_k,  # 上下文并行总已使用键长度
        )
    elif fa_version == 4:  # 使用Flash Attention 4
        assert alibi_slopes is None, "Alibi is not supported in FA4"  # 断言：FA4不支持ALiBi
        # FA4 on SM90 doesn't support paged KV; SM100+ does  # FA4在SM90上不支持分页KV；SM100+支持
        from vllm.platforms import current_platform  # 延迟导入当前平台信息

        if block_table is not None and current_platform.is_device_capability_family(90):  # 如果使用分页KV且在SM90上
            raise NotImplementedError(  # 抛出未实现错误
                "FA4 with paged KV is not supported on SM90 (Hopper). "  # FA4分页KV在Hopper上不支持
                "Use FA3 or upgrade to Blackwell (SM100+)."  # 建议使用FA3或升级到Blackwell
            )
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd  # 导入FA4的CUTE前向计算接口

        out, softmax_lse = _flash_attn_fwd(  # 调用FA4前向计算
            q,  # 查询张量
            k,  # 键张量
            v,  # 值张量
            cu_seqlens_q=cu_seqlens_q,  # 查询累积序列长度
            cu_seqlens_k=cu_seqlens_k,  # 键累积序列长度
            seqused_k=seqused_k,  # 每个序列实际使用的键长度
            max_seqlen_q=max_seqlen_q,  # 最大查询序列长度
            max_seqlen_k=max_seqlen_k,  # 最大键序列长度
            page_table=block_table,  # 分页KV缓存的页表
            softmax_scale=softmax_scale,  # softmax缩放因子
            causal=causal,  # 因果掩码标志
            softcap=softcap,  # softcap值
            window_size_left=real_window_size[0] if real_window_size[0] >= 0 else None,  # 左窗口大小（负值转为None）
            window_size_right=real_window_size[1] if real_window_size[1] >= 0 else None,  # 右窗口大小（负值转为None）
            num_splits=num_splits,  # 分片数量
            return_lse=return_softmax_lse,  # 是否返回logsumexp
            out=out,  # 预分配的输出张量
        )
    else:  # 未知的FA版本
        raise ValueError(f"Unsupported FA version: {fa_version}")  # 抛出值错误异常
    return (out, softmax_lse) if return_softmax_lse else out  # 根据需要返回输出和logsumexp，或仅返回输出


def sparse_attn_func(  # 稀疏注意力函数（非变长版本）
    q,  # 查询张量 (batch_size, seqlen, nheads, headdim)
    k,  # 键张量 (batch_size, seqlen, nheads_k, headdim)
    v,  # 值张量 (batch_size, seqlen, nheads_k, headdim)
    block_count,  # 斜线稀疏模式的块计数
    block_offset,  # 斜线稀疏模式的块偏移
    column_count,  # 垂直稀疏模式的列计数
    column_index,  # 垂直稀疏模式的列索引
    dropout_p=0.0,  # Dropout概率
    softmax_scale=None,  # softmax缩放因子
    causal=False,  # 是否使用因果掩码
    softcap=0.0,  # 0.0 means deactivated  # softcap值，0.0表示未激活
    alibi_slopes=None,  # ALiBi位置编码斜率
    deterministic=False,  # 是否使用确定性实现
    return_attn_probs=False,  # 是否返回注意力概率
    *,  # 以下为仅限关键字参数
    return_softmax_lse=False,  # 是否返回softmax的logsumexp
    out=None,  # 预分配的输出张量
):
    """计算带有垂直和斜线稀疏模式的注意力。

    大部分参数与flash_attn_func接口相同，额外增加4个参数：
    block_count和block_offset用于斜线稀疏模式，
    column_count和column_index用于垂直稀疏模式。
    详细信息请参阅论文 https://arxiv.org/abs/2407.02490 附录C.4.2。

    Args:
        q: (batch_size, seqlen, nheads, headdim) 查询张量。
        k: (batch_size, seqlen, nheads_k, headdim) 键张量。
        v: (batch_size, seqlen, nheads_k, headdim) 值张量。
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M)) 每行的稀疏块数量。
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S) 稀疏块偏移。
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M)) 每行的垂直列数量。
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V) 垂直列索引。
        dropout_p: Dropout概率。
        softmax_scale: QK^T的缩放因子，默认为1/sqrt(headdim)。
        causal: 是否应用因果注意力掩码。
        alibi_slopes: (nheads,)或(batch_size, nheads)，fp32类型的ALiBi斜率。
        deterministic: 是否使用确定性反向传播实现。
        return_attn_probs: 是否返回注意力概率（仅测试用）。

    Returns:
        out: (batch_size, seqlen, nheads, headdim)。
        softmax_lse [可选]: (batch_size, nheads, seqlen)，QK^T*scale矩阵每行的logsumexp。
    """
    if softmax_scale is None:  # 如果未指定softmax缩放因子
        softmax_scale = q.shape[-1] ** (-0.5)  # 使用默认值 1/sqrt(headdim)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]  # 确保q、k、v张量在内存中连续
    out, softmax_lse = torch.ops._vllm_fa2_C.fwd_sparse(  # 调用FA2的稀疏注意力前向内核
        q,  # 查询张量
        k,  # 键张量
        v,  # 值张量
        block_count,  # 斜线稀疏块计数
        block_offset,  # 斜线稀疏块偏移
        column_count,  # 垂直稀疏列计数
        column_index,  # 垂直稀疏列索引
        out,  # 预分配的输出张量
        alibi_slopes,  # ALiBi斜率
        dropout_p,  # Dropout概率
        softmax_scale,  # softmax缩放因子
        causal,  # 因果掩码标志
        softcap,  # softcap值
        return_attn_probs and dropout_p > 0,  # 是否返回注意力概率（仅当有dropout时）
        None,  # 生成器（用于dropout的随机种子）
    )
    return (out, softmax_lse) if return_softmax_lse else out  # 根据需要返回输出和logsumexp，或仅返回输出


def sparse_attn_varlen_func(  # 变长序列的稀疏注意力函数
    q,  # 查询张量 (total_q, nheads, headdim)
    k,  # 键张量 (total_k, nheads_k, headdim)
    v,  # 值张量 (total_k, nheads_k, headdim)
    block_count,  # 斜线稀疏模式的块计数
    block_offset,  # 斜线稀疏模式的块偏移
    column_count,  # 垂直稀疏模式的列计数
    column_index,  # 垂直稀疏模式的列索引
    cu_seqlens_q,  # 查询序列的累积长度
    cu_seqlens_k,  # 键序列的累积长度
    max_seqlen_q,  # 最大查询序列长度
    max_seqlen_k,  # 最大键序列长度
    dropout_p=0.0,  # Dropout概率
    softmax_scale=None,  # softmax缩放因子
    causal=False,  # 是否使用因果掩码
    softcap=0.0,  # 0.0 means deactivated  # softcap值，0.0表示未激活
    alibi_slopes=None,  # ALiBi位置编码斜率
    deterministic=False,  # 是否使用确定性实现
    return_attn_probs=False,  # 是否返回注意力概率
    *,  # 以下为仅限关键字参数
    return_softmax_lse=False,  # 是否返回softmax的logsumexp
    out=None,  # 预分配的输出张量
):
    """计算带有垂直和斜线稀疏模式的变长序列注意力。

    大部分参数与flash_attn_varlen_func接口相同，额外增加4个参数：
    block_count和block_offset用于斜线稀疏模式，
    column_count和column_index用于垂直稀疏模式。
    详细信息请参阅论文 https://arxiv.org/abs/2407.02490 附录C.4.2。

    Args:
        q: (total_q, nheads, headdim)，total_q为批次中查询token总数。
        k: (total_k, nheads_k, headdim)，total_k为批次中键token总数。
        v: (total_k, nheads_k, headdim)，total_k为批次中值token总数。
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M)) 每行的稀疏块数量。
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S) 稀疏块偏移。
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M)) 每行的垂直列数量。
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V) 垂直列索引。
        cu_seqlens_q: (batch_size+1,)，int32类型，批次中序列的累积查询长度。
        cu_seqlens_k: (batch_size+1,)，int32类型，批次中序列的累积键长度。
        max_seqlen_q: 批次中最大查询序列长度。
        max_seqlen_k: 批次中最大键序列长度。
        dropout_p: Dropout概率。
        softmax_scale: QK^T的缩放因子，默认为1/sqrt(headdim)。
        causal: 是否应用因果注意力掩码。
        softcap: 大于0时激活softcapping注意力。
        alibi_slopes: (nheads,)或(batch_size, nheads)，fp32类型的ALiBi斜率。
        deterministic: 是否使用确定性反向传播实现。
        return_attn_probs: 是否返回注意力概率（仅测试用）。

    Returns:
        out: (total, nheads, headdim)。
        softmax_lse [可选]: (nheads, total_q_seqlen)，QK^T*scale矩阵每行的logsumexp。
    """
    if softmax_scale is None:  # 如果未指定softmax缩放因子
        softmax_scale = q.shape[-1] ** (-0.5)  # 使用默认值 1/sqrt(headdim)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]  # 确保q、k、v张量在内存中连续
    out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd_sparse(  # 调用FA2的变长稀疏注意力前向内核
        q,  # 查询张量
        k,  # 键张量
        v,  # 值张量
        block_count,  # 斜线稀疏块计数
        block_offset,  # 斜线稀疏块偏移
        column_count,  # 垂直稀疏列计数
        column_index,  # 垂直稀疏列索引
        out,  # 预分配的输出张量
        cu_seqlens_q,  # 查询累积序列长度
        cu_seqlens_k,  # 键累积序列长度
        None,  # 占位参数
        alibi_slopes,  # ALiBi斜率
        max_seqlen_q,  # 最大查询序列长度
        max_seqlen_k,  # 最大键序列长度
        dropout_p,  # Dropout概率
        softmax_scale,  # softmax缩放因子
        False,  # zero_tensors标志
        causal,  # 因果掩码标志
        softcap,  # softcap值
        return_attn_probs and dropout_p > 0,  # 是否返回注意力概率（仅当有dropout时）
        None,  # 生成器（用于dropout的随机种子）
    )
    return (out, softmax_lse) if return_softmax_lse else out  # 根据需要返回输出和logsumexp，或仅返回输出
