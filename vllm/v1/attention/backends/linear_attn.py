# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass  # 导入数据类装饰器

import torch  # 导入PyTorch库

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.v1.attention.backend import (  # 从注意力后端模块导入基类和相关类型
    AttentionBackend,  # 注意力后端基类
    AttentionCGSupport,  # 注意力CUDA图支持枚举
    AttentionMetadataBuilder,  # 注意力元数据构建器基类
    CommonAttentionMetadata,  # 通用注意力元数据类
)
from vllm.v1.attention.backends.utils import (  # 从注意力后端工具模块导入辅助函数
    mamba_get_block_table_tensor,  # 获取Mamba块表张量的函数
    split_decodes_and_prefills,  # 分离解码和预填充请求的函数
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec  # 导入注意力规格和Mamba规格接口


class LinearAttentionBackend(AttentionBackend):
    """线性注意力后端类，继承自AttentionBackend，提供线性注意力机制的后端实现。"""

    @staticmethod
    def get_name() -> str:  # 获取后端名称的静态方法
        """返回线性注意力后端的名称字符串。"""
        return "LINEAR_ATTN"  # 返回后端名称"LINEAR_ATTN"

    @staticmethod
    def get_builder_cls() -> type["LinearAttentionMetadataBuilder"]:  # 获取元数据构建器类的静态方法
        """返回线性注意力元数据构建器的类型。"""
        return LinearAttentionMetadataBuilder  # 返回线性注意力元数据构建器类


@dataclass
class LinearAttentionMetadata:
    """线性注意力元数据类，使用数据类装饰器定义，存储线性注意力计算所需的元信息。"""

    num_prefills: int  # 预填充请求的数量
    num_prefill_tokens: int  # 预填充的token总数
    num_decodes: int  # 解码请求的数量
    num_decode_tokens: int  # 解码的token总数
    query_start_loc: torch.Tensor  # 查询起始位置张量，记录每个序列的查询起始索引
    seq_lens: torch.Tensor  # 序列长度张量，记录每个序列的长度

    state_indices_tensor: torch.Tensor  # 状态索引张量，形状为[batch,]，用于索引缓存状态


class LinearAttentionMetadataBuilder(AttentionMetadataBuilder[LinearAttentionMetadata]):
    """线性注意力元数据构建器类，负责构建线性注意力所需的元数据，继承自泛型元数据构建器。"""

    reorder_batch_threshold: int = 1  # 批次重排序阈值，设为1表示单token即触发重排序

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE  # CUDA图支持模式，设为统一单token解码模式

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,  # KV缓存规格参数
        layer_names: list[str],  # 层名称列表
        vllm_config: VllmConfig,  # vLLM配置对象
        device: torch.device,  # 计算设备（如CPU或GPU）
    ):
        """初始化线性注意力元数据构建器，验证缓存规格必须为MambaSpec类型。"""
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)  # 调用父类初始化方法
        assert isinstance(kv_cache_spec, MambaSpec)  # 断言KV缓存规格必须是MambaSpec实例

    def build(
        self,
        common_prefix_len: int,  # 公共前缀长度
        common_attn_metadata: CommonAttentionMetadata,  # 通用注意力元数据
        fast_build: bool = False,  # 是否启用快速构建模式
    ) -> LinearAttentionMetadata:
        """构建线性注意力元数据，从通用元数据中提取并组装线性注意力所需的各项信息。"""
        query_start_loc = common_attn_metadata.query_start_loc  # 获取查询起始位置张量
        seq_lens = common_attn_metadata.seq_lens  # 获取序列长度张量

        state_indices_tensor = mamba_get_block_table_tensor(  # 通过Mamba块表获取状态索引张量
            common_attn_metadata.block_table_tensor,  # 传入块表张量
            common_attn_metadata.seq_lens,  # 传入序列长度
            self.kv_cache_spec,  # 传入KV缓存规格
            self.vllm_config.cache_config.mamba_cache_mode,  # 传入Mamba缓存模式配置
        )[:, 0]  # 取第一列，将二维张量降为一维

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (  # 分离解码和预填充的统计信息
            split_decodes_and_prefills(  # 调用分离函数
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold  # 传入通用元数据和解码阈值
            )
        )

        attn_metadata = LinearAttentionMetadata(  # 创建线性注意力元数据实例
            num_prefills=num_prefills,  # 设置预填充请求数量
            num_prefill_tokens=num_prefill_tokens,  # 设置预填充token数量
            num_decodes=num_decodes,  # 设置解码请求数量
            num_decode_tokens=num_decode_tokens,  # 设置解码token数量
            query_start_loc=query_start_loc,  # 设置查询起始位置
            seq_lens=seq_lens,  # 设置序列长度
            state_indices_tensor=state_indices_tensor,  # 设置状态索引张量
        )
        return attn_metadata  # 返回构建好的线性注意力元数据
