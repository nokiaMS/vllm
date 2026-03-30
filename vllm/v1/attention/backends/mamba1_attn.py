# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace  # 导入数据类装饰器和替换函数
from typing import Any  # 导入Any类型注解

from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata  # 导入注意力后端基类和通用注意力元数据
from vllm.v1.attention.backends.mamba_attn import (  # 从Mamba注意力模块导入基类
    BaseMambaAttentionMetadata,  # Mamba注意力元数据基类
    BaseMambaAttentionMetadataBuilder,  # Mamba注意力元数据构建器基类
)


# Mamba1注意力后端类，继承自AttentionBackend，用于提供Mamba1架构的注意力计算后端
class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:  # 获取后端名称的静态方法
        return "MAMBA1_ATTN"  # 返回Mamba1注意力后端的名称标识

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:  # 获取元数据构建器类的静态方法
        return Mamba1AttentionMetadataBuilder  # 返回Mamba1注意力元数据构建器类


@dataclass  # 数据类装饰器
# Mamba1注意力元数据类，继承自BaseMambaAttentionMetadata，用于存储Mamba1注意力计算所需的元数据
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    pass  # 直接继承基类，不添加额外字段


# Mamba1注意力元数据构建器类，继承自BaseMambaAttentionMetadataBuilder，负责构建Mamba1注意力计算所需的元数据
class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]  # 使用Mamba1AttentionMetadata作为泛型参数
):
    metadata_cls = Mamba1AttentionMetadata  # 指定元数据类为Mamba1AttentionMetadata

    # 构建注意力元数据的方法，根据前缀长度和通用注意力元数据生成Mamba1专用的注意力元数据
    def build(
        self,
        common_prefix_len: int,  # 公共前缀长度
        common_attn_metadata: CommonAttentionMetadata,  # 通用注意力元数据
        fast_build: bool = False,  # 是否使用快速构建模式
        **kwargs: Any,  # 额外的关键字参数
    ) -> Mamba1AttentionMetadata:  # 返回Mamba1注意力元数据
        common = self._compute_common_metadata(common_attn_metadata)  # 计算通用元数据

        if (
            common.num_prefills > 0  # 如果存在预填充请求
            and self.vllm_config.cache_config.mamba_cache_mode == "all"  # 且Mamba缓存模式为"all"
        ):
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (  # 解包分块元数据张量
                self._build_chunk_metadata_tensors(  # 构建分块元数据张量
                    self.kv_cache_spec.block_size,  # 传入KV缓存的块大小
                    common,  # 传入通用元数据
                    common_attn_metadata,  # 传入通用注意力元数据
                )
            )
            return replace(  # 使用replace函数创建新的元数据对象，替换分块相关字段
                common,  # 基于通用元数据
                cu_chunk_seqlen_p=cu_chunk_seqlen_p,  # 设置累积分块序列长度（预填充）
                last_chunk_indices_p=last_chunk_indices_p,  # 设置最后分块索引（预填充）
            )

        return common  # 无预填充或非"all"缓存模式时直接返回通用元数据
