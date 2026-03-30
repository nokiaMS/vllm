# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者
from dataclasses import dataclass  # 导入数据类装饰器，用于定义注意力元数据结构

from vllm.v1.attention.backend import AttentionBackend  # 导入注意力后端抽象基类
from vllm.v1.attention.backends.mamba_attn import (  # 从 Mamba 注意力模块导入基类
    BaseMambaAttentionMetadata,  # Mamba 注意力元数据基类
    BaseMambaAttentionMetadataBuilder,  # Mamba 注意力元数据构建器基类
)


# 短卷积注意力后端类，用于处理短卷积类型的注意力计算
# 继承自 AttentionBackend 抽象基类，复用 Mamba 注意力的元数据结构
class ShortConvAttentionBackend(AttentionBackend):
    """短卷积注意力后端，复用 Mamba 注意力机制的元数据结构和构建逻辑。"""

    @staticmethod  # 静态方法装饰器
    def get_name() -> str:  # 获取后端名称的静态方法
        """返回该注意力后端的名称标识。"""
        return "SHORT_CONV_ATTN"  # 返回短卷积注意力的名称字符串

    @staticmethod  # 静态方法装饰器
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:  # 获取元数据构建器类的静态方法
        """返回对应的元数据构建器类。"""
        return ShortConvAttentionMetadataBuilder  # 返回短卷积注意力元数据构建器类


@dataclass  # 数据类装饰器，自动生成 __init__、__repr__ 等方法
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    """短卷积注意力元数据类，继承自 Mamba 注意力元数据基类，无需额外字段。"""
    pass  # 直接继承基类的所有字段，无需额外定义


# 短卷积注意力元数据构建器类，继承自 Mamba 注意力元数据构建器基类
# 复用基类的 build 方法来构建元数据
class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    """短卷积注意力元数据构建器，复用 Mamba 基类的构建逻辑。"""
    metadata_cls = ShortConvAttentionMetadata  # 指定构建的元数据类型为短卷积注意力元数据类
