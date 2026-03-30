# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache  # 导入缓存装饰器，用于缓存函数结果
from typing import NamedTuple, cast, get_args  # 导入类型提示工具

import torch  # 导入PyTorch库

from vllm.config.cache import CacheDType  # 导入缓存数据类型定义
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.utils.import_utils import resolve_obj_by_qualname  # 导入通过限定名称解析对象的工具函数
from vllm.v1.attention.backend import AttentionBackend, AttentionType  # 导入注意力后端基类和注意力类型
from vllm.v1.attention.backends.registry import (  # 从注意力后端注册表导入Mamba相关映射
    MAMBA_TYPE_TO_BACKEND_MAP,  # Mamba类型到后端的映射字典
    MambaAttentionBackendEnum,  # Mamba注意力后端枚举类
)

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# 注意力选择器配置类，用于封装选择注意力后端所需的所有配置参数
class AttentionSelectorConfig(NamedTuple):
    """注意力选择器的配置，使用NamedTuple实现不可变的配置对象，便于缓存和哈希"""
    head_size: int  # 注意力头的维度大小
    dtype: torch.dtype  # 数据类型（如float16、bfloat16等）
    kv_cache_dtype: CacheDType | None  # KV缓存的数据类型，可为None表示使用默认类型
    block_size: int | None  # 分块大小，可为None表示未指定
    use_mla: bool = False  # 是否使用多头潜在注意力（Multi-head Latent Attention）
    has_sink: bool = False  # 是否使用注意力汇聚（Attention Sink）机制
    use_sparse: bool = False  # 是否使用稀疏注意力
    use_mm_prefix: bool = False  # 是否使用多模态前缀
    use_per_head_quant_scales: bool = False  # 是否使用逐头量化缩放因子
    attn_type: str = AttentionType.DECODER  # 注意力类型，默认为解码器注意力

    def __repr__(self):
        """返回配置对象的字符串表示，用于调试和日志输出"""
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "  # 输出头维度大小
            f"dtype={self.dtype}, "  # 输出数据类型
            f"kv_cache_dtype={self.kv_cache_dtype}, "  # 输出KV缓存数据类型
            f"block_size={self.block_size}, "  # 输出分块大小
            f"use_mla={self.use_mla}, "  # 输出是否使用MLA
            f"has_sink={self.has_sink}, "  # 输出是否有注意力汇聚
            f"use_sparse={self.use_sparse}, "  # 输出是否使用稀疏注意力
            f"use_mm_prefix={self.use_mm_prefix}, "  # 输出是否使用多模态前缀
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "  # 输出是否使用逐头量化
            f"attn_type={self.attn_type})"  # 输出注意力类型
        )


def get_attn_backend(
    head_size: int,  # 注意力头的维度大小
    dtype: torch.dtype,  # 数据类型
    kv_cache_dtype: str | None,  # KV缓存数据类型字符串
    use_mla: bool = False,  # 是否使用多头潜在注意力
    has_sink: bool = False,  # 是否使用注意力汇聚
    use_sparse: bool = False,  # 是否使用稀疏注意力
    use_mm_prefix: bool = False,  # 是否使用多模态前缀
    use_per_head_quant_scales: bool = False,  # 是否使用逐头量化缩放因子
    attn_type: str | None = None,  # 注意力类型，可选
    num_heads: int | None = None,  # 注意力头数量，可选
) -> type[AttentionBackend]:
    """选择使用哪个注意力后端并延迟导入它。根据当前平台和配置选择最合适的注意力实现。"""

    if kv_cache_dtype is not None:  # 如果指定了KV缓存数据类型
        valid_cache_dtypes = get_args(CacheDType)  # 获取所有合法的缓存数据类型
        assert kv_cache_dtype in valid_cache_dtypes, (  # 断言指定的类型是合法的
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config  # 延迟导入当前vLLM配置获取函数

    vllm_config = get_current_vllm_config()  # 获取当前的vLLM全局配置

    cache_config = vllm_config.cache_config  # 获取缓存配置
    if cache_config is not None and cache_config.user_specified_block_size:  # 如果用户显式指定了分块大小
        block_size = cache_config.block_size  # 使用用户指定的分块大小
    else:
        block_size = None  # 否则设为None，由后端自行决定

    attn_selector_config = AttentionSelectorConfig(  # 创建注意力选择器配置对象
        head_size=head_size,  # 设置头维度大小
        dtype=dtype,  # 设置数据类型
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),  # 类型转换KV缓存数据类型
        block_size=block_size,  # 设置分块大小
        use_mla=use_mla,  # 设置是否使用MLA
        has_sink=has_sink,  # 设置是否有注意力汇聚
        use_sparse=use_sparse,  # 设置是否使用稀疏注意力
        use_mm_prefix=use_mm_prefix,  # 设置是否使用多模态前缀
        use_per_head_quant_scales=use_per_head_quant_scales,  # 设置是否使用逐头量化
        attn_type=attn_type or AttentionType.DECODER,  # 设置注意力类型，默认为解码器
    )

    return _cached_get_attn_backend(  # 调用缓存版本的后端获取函数
        backend=vllm_config.attention_config.backend,  # 传入配置中指定的后端名称
        attn_selector_config=attn_selector_config,  # 传入选择器配置
        num_heads=num_heads,  # 传入注意力头数量
    )


@cache  # 使用缓存装饰器，避免重复计算相同配置的后端选择
def _cached_get_attn_backend(
    backend,  # 后端名称或标识
    attn_selector_config: AttentionSelectorConfig,  # 注意力选择器配置
    num_heads: int | None = None,  # 注意力头数量，可选
) -> type[AttentionBackend]:
    """缓存版本的注意力后端获取函数，根据平台和配置解析并返回对应的注意力后端类"""
    from vllm.platforms import current_platform  # 延迟导入当前平台信息

    attention_cls = current_platform.get_attn_backend_cls(  # 从当前平台获取注意力后端类名
        backend,  # 传入后端标识
        attn_selector_config=attn_selector_config,  # 传入选择器配置
        num_heads=num_heads,  # 传入注意力头数量
    )
    if not attention_cls:  # 如果未找到合适的注意力后端类
        raise ValueError(  # 抛出值错误异常
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)  # 通过限定名称解析并导入后端类对象

    # 如果选定的后端要求特定的KV缓存布局，则进行调整
    required_layout = backend.get_required_kv_cache_layout()  # 获取后端要求的KV缓存布局
    if required_layout is not None:  # 如果后端有特定的布局要求
        from vllm.v1.attention.backends.utils import set_kv_cache_layout  # 延迟导入KV缓存布局设置函数

        set_kv_cache_layout(required_layout)  # 设置KV缓存布局为后端要求的布局
        logger.info(  # 记录日志信息
            "Using %s KV cache layout for %s backend.",  # 日志格式字符串
            required_layout,  # KV缓存布局名称
            backend.get_name(),  # 后端名称
        )

    return backend  # 返回解析后的注意力后端类


def get_mamba_attn_backend(
    mamba_type: str,  # Mamba模型类型字符串
) -> type[AttentionBackend]:
    """选择使用哪个Mamba注意力后端并延迟导入它。用于Mamba架构（状态空间模型）的注意力后端选择。"""
    return _cached_get_mamba_attn_backend(mamba_type)  # 调用缓存版本的Mamba后端获取函数


@cache  # 使用缓存装饰器，避免重复解析相同类型的Mamba后端
def _cached_get_mamba_attn_backend(
    mamba_type: str,  # Mamba模型类型字符串
) -> type[AttentionBackend]:
    """缓存版本的Mamba注意力后端获取函数，根据Mamba类型查找并返回对应的后端类"""
    assert mamba_type and isinstance(mamba_type, str)  # 断言mamba_type非空且为字符串类型

    selected_backend = None  # 初始化选定的后端为None
    try:
        backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]  # 从映射字典中查找对应的后端名称
        selected_backend = MambaAttentionBackendEnum[backend_name]  # 从枚举中获取对应的后端枚举值
    except KeyError as e:  # 捕获键不存在的异常
        raise ValueError(  # 抛出值错误异常，提示无效的Mamba后端类型
            f"Invalid mamba attention backend type: '{backend_name}'. Valid "
            f"backends are: {list(MambaAttentionBackendEnum.__members__.keys())}"
        ) from e  # 保留原始异常链

    mamba_attn_backend = selected_backend.get_class()  # 从枚举值获取实际的后端类
    return mamba_attn_backend  # 返回Mamba注意力后端类
