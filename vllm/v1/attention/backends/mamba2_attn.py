# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools  # 导入迭代工具模块，用于累加求和
from dataclasses import dataclass, replace  # 导入数据类装饰器和替换函数
from typing import Any  # 导入Any类型注解

import torch  # 导入PyTorch框架

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.v1.attention.backend import (  # 导入注意力后端基类和通用注意力元数据
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (  # 导入Mamba注意力的基础元数据类和构建器基类
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec  # 导入注意力规格接口


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构建Mamba2 SSD内核所需的按块对齐的变长元数据。

    给定每个序列的累积token起始位置 `query_start_loc`（形状为[B+1]）
    和物理 `chunk_size`，返回三个同设备上的张量：
      - cu_chunk_seqlens:   (nchunks+1,) int32   逻辑块长度的排他前缀和
        （每个逻辑块不会跨越序列或物理块的边界）。
      - last_chunk_indices: (B,)         int32   每个序列最后一个逻辑块的索引
        （空序列为-1）。
      - seq_idx_chunks:     (nchunks,)   int32   每个逻辑块按顺序对应的序列索引。

    此函数有意保持轻量级并在CPU端执行；它镜像了V1 Mamba2元数据构建器产生的元数据，
    并导出以便测试（和其他调用者）可以避免重复此逻辑。
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"  # 断言输入必须是一维张量
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"  # 断言第一个元素必须为0
    device = query_start_loc.device  # 获取输入张量所在的设备

    qsl64 = query_start_loc.to(torch.int64)  # 将输入转换为int64类型以避免溢出
    starts = qsl64[:-1].tolist()  # 获取每个序列的起始位置列表
    ends = qsl64[1:].tolist()  # 获取每个序列的结束位置列表
    total = int(qsl64[-1].item())  # 获取所有序列的token总数

    chunk_lens: list[int] = []  # 存储每个逻辑块的长度
    seq_idx_chunks: list[int] = []  # 存储每个逻辑块对应的序列索引
    last_chunk_indices: list[int] = [-1] * len(starts)  # 初始化每个序列最后一个块的索引为-1

    for b, (s, e) in enumerate(zip(starts, ends)):  # 遍历每个序列的起止位置
        if e <= s:  # 如果序列为空则跳过
            # empty sequence
            continue
        pos = s  # 初始化当前位置为序列起始位置
        while pos < e:  # 当当前位置未到达序列末尾时循环
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)  # 计算当前物理块内剩余的空间
            take = min(room, e - pos)  # 取物理块剩余空间和序列剩余长度的较小值
            chunk_lens.append(int(take))  # 将该逻辑块的长度添加到列表
            seq_idx_chunks.append(b)  # 记录该逻辑块属于哪个序列
            last_chunk_indices[b] = len(chunk_lens) - 1  # 更新该序列最后一个块的索引
            pos += take  # 移动当前位置

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:  # 如果存在逻辑块
        cu_chunk_seqlens = torch.tensor(  # 构建逻辑块长度的累积和张量
            [0] + list(itertools.accumulate(chunk_lens)),  # 以0开头的前缀和列表
            device=device,  # 使用与输入相同的设备
            dtype=torch.int32,  # 使用int32数据类型
        )
        # Final boundary must equal total tokens
        assert int(cu_chunk_seqlens[-1].item()) == total  # 断言最终边界值等于token总数
    else:  # 如果没有逻辑块（所有序列为空）
        cu_chunk_seqlens = torch.tensor([0], device=device, dtype=torch.int32)  # 返回仅含0的张量

    last_chunk_indices_t = (  # 构建最后块索引张量
        torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)  # 从列表创建张量
        if len(starts) > 0  # 如果存在序列
        else torch.empty((0,), device=device, dtype=torch.int32)  # 否则返回空张量
    )
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, device=device, dtype=torch.int32)  # 构建序列索引张量
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t  # 返回三个元数据张量


# Mamba2注意力后端类，继承自AttentionBackend，负责提供Mamba2注意力机制的后端实现
class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        """获取该注意力后端的名称标识符"""
        return "MAMBA2_ATTN"  # 返回Mamba2注意力后端的名称字符串

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        """获取该后端对应的元数据构建器类"""
        return Mamba2AttentionMetadataBuilder  # 返回Mamba2注意力元数据构建器类


@dataclass
# Mamba2注意力元数据类，继承自BaseMambaAttentionMetadata，存储Mamba2注意力计算所需的元数据
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False  # 是否需要准备初始状态，默认为False
    chunk_size: int = 0  # 分块大小，默认为0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None  # 预填充阶段的序列索引张量，仅预填充时使用


# Mamba2注意力元数据构建器类，继承自BaseMambaAttentionMetadataBuilder，负责构建Mamba2注意力所需的元数据
class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata  # 指定该构建器生成的元数据类型

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化Mamba2注意力元数据构建器，设置分块大小等参数"""
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)  # 调用父类初始化方法
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()  # 从模型配置中获取Mamba分块大小
        assert chunk_size is not None, (  # 断言分块大小不能为空
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size  # 保存分块大小到实例属性

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        """构建Mamba2注意力元数据，包括分块信息和序列索引等"""
        common = self._compute_common_metadata(  # 计算通用元数据
            common_attn_metadata, num_accepted_tokens=kwargs.get("num_accepted_tokens")  # 传入已接受的token数量
        )

        seq_idx_p = None  # 初始化预填充序列索引为空
        cu_chunk_seqlen_p = None  # 初始化预填充累积块序列长度为空
        last_chunk_indices_p = None  # 初始化预填充最后块索引为空
        prep_initial_states = False  # 初始化是否准备初始状态为False

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:  # 如果存在预填充请求
            prep_initial_states = (  # 判断是否需要准备初始状态
                torch.any(common.has_initial_states_p).item()  # 检查是否有任何序列具有初始状态
                if common.has_initial_states_p is not None  # 如果初始状态标志不为空
                else False  # 否则默认不需要准备初始状态
            )

            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (  # 构建分块元数据张量
                self._build_chunk_metadata_tensors(  # 调用父类方法构建分块元数据
                    self.chunk_size,  # 传入分块大小
                    common,  # 传入通用元数据
                    common_attn_metadata,  # 传入通用注意力元数据
                )
            )

        return replace(  # 使用replace创建新的元数据对象，保留common的字段并更新以下字段
            common,
            prep_initial_states=prep_initial_states,  # 设置是否准备初始状态
            chunk_size=self.chunk_size,  # 设置分块大小
            seq_idx_p=seq_idx_p,  # 设置预填充序列索引
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,  # 设置预填充累积块序列长度
            last_chunk_indices_p=last_chunk_indices_p,  # 设置预填充最后块索引
        )
