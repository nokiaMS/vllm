# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch  # 导入 PyTorch 库

from vllm.logger import init_logger  # 从 vllm 日志模块导入日志初始化函数
from vllm.platforms import current_platform  # 从 vllm 平台模块导入当前平台信息

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

if current_platform.is_cuda():  # 如果当前平台是 CUDA（GPU）
    try:
        import vllm._flashmla_C  # noqa: F401  # 尝试导入 FlashMLA 的 C 扩展模块

        _flashmla_C_AVAILABLE = True  # 标记 FlashMLA C 扩展可用
    except ImportError:  # 如果导入失败
        _flashmla_C_AVAILABLE = False  # 标记 FlashMLA C 扩展不可用
else:  # 如果不是 CUDA 平台
    _flashmla_C_AVAILABLE = False  # 标记 FlashMLA C 扩展不可用

if current_platform.is_cuda():  # 如果当前平台是 CUDA（GPU）
    try:
        import vllm._flashmla_extension_C  # noqa: F401  # 尝试导入 FlashMLA 扩展 C 模块

        _flashmla_extension_C_AVAILABLE = True  # 标记 FlashMLA 扩展 C 模块可用
    except ImportError:  # 如果导入失败
        _flashmla_extension_C_AVAILABLE = False  # 标记 FlashMLA 扩展 C 模块不可用
else:  # 如果不是 CUDA 平台
    _flashmla_extension_C_AVAILABLE = False  # 标记 FlashMLA 扩展 C 模块不可用


def _is_flashmla_available() -> tuple[bool, str | None]:
    """检查 FlashMLA 是否可用，返回 (是否可用, 不可用原因) 的元组"""
    if not _flashmla_C_AVAILABLE:  # 如果 FlashMLA C 扩展不可用
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:  # 如果 FlashMLA 扩展 C 模块不可用
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None  # 两个模块都可用，返回 True


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    检查是否支持 FlashMLA 稠密（Dense）模式。
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()  # 检查 FlashMLA 基础可用性
    if not is_available:  # 如果基础模块不可用
        return False, maybe_reason  # 返回不可用及原因
    if not current_platform.is_device_capability_family(90):  # 如果设备不属于 Hopper 架构（SM90）
        return False, "FlashMLA Dense is only supported on Hopper devices."  # 返回不支持的原因
    return True, None  # 支持，返回 True


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    检查是否支持 FlashMLA 稀疏（Sparse）模式。
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()  # 检查 FlashMLA 基础可用性
    if not is_available:  # 如果基础模块不可用
        return False, maybe_reason  # 返回不可用及原因
    if not (
        current_platform.is_device_capability_family(90)  # 检查是否为 Hopper 架构（SM90）
        or current_platform.is_device_capability_family(100)  # 或 Blackwell 架构（SM100）
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell devices.",
        )
    return True, None  # 支持，返回 True


def _raise_flashmla_unavailable(*_args, **_kwargs):
    """当 FlashMLA 不可用时，抛出运行时错误的占位函数"""
    _, reason = _is_flashmla_available()  # 获取不可用的原因
    raise RuntimeError(reason or "FlashMLA is not available")  # 抛出运行时异常


if _is_flashmla_available()[0]:  # 如果 FlashMLA 可用
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401  # 从第三方模块导入 FlashMLA 相关接口
        FlashMLASchedMeta,  # FlashMLA 调度元数据类
        flash_attn_varlen_func,  # 变长注意力计算函数
        flash_attn_varlen_kvpacked_func,  # 变长 KV 打包注意力计算函数
        flash_attn_varlen_qkvpacked_func,  # 变长 QKV 打包注意力计算函数
        flash_mla_sparse_fwd,  # FlashMLA 稀疏前向计算函数
        flash_mla_with_kvcache,  # 带 KV 缓存的 FlashMLA 计算函数
        get_mla_metadata,  # 获取 MLA 元数据函数
    )
else:  # 如果 FlashMLA 不可用

    class FlashMLASchedMeta:  # type: ignore[no-redef]  # FlashMLA 调度元数据的占位类（不可用时使用）
        """FlashMLA 不可用时的空占位类"""
        pass

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]  # 变长注意力函数指向不可用异常
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]  # 变长 KV 打包注意力函数指向不可用异常
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]  # 变长 QKV 打包注意力函数指向不可用异常
    flash_mla_sparse_fwd = _raise_flashmla_unavailable  # type: ignore[assignment]  # FlashMLA 稀疏前向函数指向不可用异常
    flash_mla_with_kvcache = _raise_flashmla_unavailable  # type: ignore[assignment]  # 带 KV 缓存的 FlashMLA 函数指向不可用异常
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]  # 获取 MLA 元数据函数指向不可用异常


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """获取 MLA 稠密 FP8 模式的解码元数据，返回 (tile调度元数据, 分片数) 的元组"""
    if not _is_flashmla_available()[0]:  # 如果 FlashMLA 不可用
        _raise_flashmla_unavailable()  # 抛出不可用异常
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(  # 调用 C 扩展获取稠密 FP8 解码元数据
        cache_seqlens,  # 缓存序列长度张量
        num_q_tokens_per_head_k,  # 每个 K 头对应的 Q token 数量
        num_heads_k,  # K 头的数量
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用 FP8 精度执行带 KV 缓存的 FlashMLA 前向计算，返回 (输出张量, softmax_lse) 的元组"""
    if not _is_flashmla_available()[0]:  # 如果 FlashMLA 不可用
        _raise_flashmla_unavailable()  # 抛出不可用异常
    if softmax_scale is None:  # 如果未指定 softmax 缩放因子
        softmax_scale = q.shape[-1] ** (-0.5)  # 使用查询向量最后一维的倒数平方根作为默认缩放因子
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(  # 调用 C 扩展执行 FP8 前向计算
        q,  # 查询张量
        k_cache,  # 键缓存张量
        head_dim_v,  # 值头维度
        cache_seqlens,  # 缓存序列长度
        block_table,  # 块表（页表，用于分页 KV 缓存）
        softmax_scale,  # softmax 缩放因子
        causal,  # 是否使用因果遮罩
        tile_scheduler_metadata,  # tile 调度元数据
        num_splits,  # 分片数量
        descale_q,  # 查询的反量化缩放因子
        descale_k,  # 键的反量化缩放因子
    )
    return out, softmax_lse  # 返回输出张量和 softmax 的 log-sum-exp 值


#
# TODO: Add fake functions
#
# @register_fake("_flashmla_C::get_mla_metadata")
# def _get_mla_metadata_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
# @register_fake("_flashmla_C::fwd_kvcache_mla")
# def _fwd_kvcache_mla_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
