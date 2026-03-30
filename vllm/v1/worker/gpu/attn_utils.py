# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence  # 导入序列抽象基类
from typing import Any, cast  # 导入类型提示工具

import numpy as np  # 导入 NumPy 数值计算库
import torch  # 导入 PyTorch 深度学习框架

from vllm.config import VllmConfig, get_layers_from_vllm_config  # 导入 vLLM 配置类和层获取函数
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase  # 导入注意力层基类
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata  # 导入注意力后端和通用注意力元数据
from vllm.v1.kv_cache_interface import (  # 导入 KV 缓存接口相关类
    AttentionSpec,  # 注意力规格
    KVCacheConfig,  # KV 缓存配置
    KVCacheSpec,  # KV 缓存规格
    UniformTypeKVCacheSpecs,  # 统一类型 KV 缓存规格
)
from vllm.v1.worker.utils import AttentionGroup, bind_kv_cache  # 导入注意力分组和 KV 缓存绑定工具


# 遍历模型中所有注意力层，收集每一层的 KV 缓存规格（如块大小、头数等）。
# 跳过不需要 KV 缓存的层（如纯编码器注意力）。
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}  # 初始化 KV 缓存规格字典
    layer_type = cast(type[Any], AttentionLayerBase)  # 将注意力层基类转换为 Any 类型以兼容类型系统
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)  # 从配置中获取所有注意力层
    for layer_name, attn_module in attn_layers.items():  # 遍历每个注意力层
        # Skip modules that don't need KV cache (eg encoder-only attention)
        if spec := attn_module.get_kv_cache_spec(vllm_config):  # 获取该层的 KV 缓存规格，跳过不需要的层
            kv_cache_spec[layer_name] = spec  # 将规格存入字典
    return kv_cache_spec  # 返回所有层的 KV 缓存规格


# 初始化注意力后端：将注意力层按后端类型和 KV 缓存规格分组（AttentionGroup），
# 为每个分组创建元数据构建器，并在各构建器之间共享工作空间缓冲区以节省显存。
def init_attn_backend(
    kv_cache_config: KVCacheConfig, vllm_config: VllmConfig, device: torch.device
):
    attn_backends: dict[str, type[AttentionBackend]] = {}  # 初始化注意力后端字典，层名到后端类型的映射
    attn_groups: list[list[AttentionGroup]] = []  # 初始化注意力分组列表
    attn_backend_workspace: torch.Tensor | None = None  # 初始化共享的工作空间缓冲区为 None
    for kv_cache_group_id, kv_cache_group_spec in enumerate(  # 遍历每个 KV 缓存组
        kv_cache_config.kv_cache_groups
    ):
        layer_names = kv_cache_group_spec.layer_names  # 获取该组包含的层名列表

        layer_type = cast(type[Any], AttentionLayerBase)  # 将注意力层基类转换为 Any 类型
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)  # 根据层名获取对应的注意力层

        group_map: dict[tuple[tuple[str, str], KVCacheSpec], AttentionGroup] = {}  # 用于按后端+规格去重的映射
        group_order: list[tuple[tuple[str, str], KVCacheSpec]] = []  # 记录分组的插入顺序

        for layer_name in layer_names:  # 遍历该 KV 缓存组中的每一层
            attn_backend = attn_layers[layer_name].get_attn_backend()  # 获取该层使用的注意力后端
            attn_backends[layer_name] = attn_backend  # 记录层名到后端的映射

            layer_kv_cache_spec: KVCacheSpec = kv_cache_group_spec.kv_cache_spec  # 获取该组的 KV 缓存规格
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):  # 如果是统一类型规格集合
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]  # 取出该层特定的规格

            key = (attn_backend.full_cls_name(), layer_kv_cache_spec)  # 构建分组键：(后端类名, KV缓存规格)
            if key not in group_map:  # 如果该组合尚未出现过
                group_map[key] = AttentionGroup(  # 创建新的注意力分组
                    attn_backend,
                    [layer_name],
                    layer_kv_cache_spec,
                    kv_cache_group_id,
                )
                group_order.append(key)  # 记录插入顺序
            else:  # 如果该组合已存在
                group_map[key].layer_names.append(layer_name)  # 将层名追加到现有分组

        groups = [group_map[key] for key in group_order]  # 按插入顺序构建分组列表
        for group in groups:  # 遍历每个分组
            group.create_metadata_builders(  # 为分组创建注意力元数据构建器
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=None,
                num_metadata_builders=1,
            )
            builder = group.get_metadata_builder(0)  # 获取第一个（也是唯一的）元数据构建器
            if attn_backend_workspace is None:  # 如果还没有工作空间缓冲区
                if hasattr(builder, "_get_workspace_buffer"):  # 如果构建器支持获取工作空间缓冲区
                    attn_backend_workspace = builder._get_workspace_buffer()  # 获取工作空间缓冲区供后续共享
            else:  # 如果已有工作空间缓冲区
                if hasattr(builder, "set_workspace_buffer"):  # 如果构建器支持设置工作空间缓冲区
                    builder.set_workspace_buffer(attn_backend_workspace)  # 将共享的工作空间缓冲区设置给当前构建器
        attn_groups.append(groups)  # 将该 KV 缓存组的分组列表添加到总列表
    return attn_backends, attn_groups  # 返回注意力后端字典和注意力分组列表


# 在 GPU 上分配 KV 缓存的原始张量（int8 格式）。
# 同一张量可被多个层共享（通过 shared_by 机制），以减少显存占用。
def _allocate_kv_cache(kv_cache_config: KVCacheConfig, device: torch.device):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}  # 初始化原始 KV 缓存张量字典
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:  # 遍历每个 KV 缓存张量配置
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)  # 在 GPU 上分配 int8 格式的零初始化张量
        for layer_name in kv_cache_tensor.shared_by:  # 遍历共享该张量的所有层
            kv_cache_raw_tensors[layer_name] = tensor  # 让多个层指向同一个张量

    layer_names = set()  # 初始化层名集合用于校验
    for group in kv_cache_config.kv_cache_groups:  # 遍历所有 KV 缓存组
        for layer_name in group.layer_names:  # 遍历组内的层名
            layer_names.add(layer_name)  # 收集所有层名
    assert layer_names == set(kv_cache_raw_tensors.keys()), (  # 断言所有层都已正确初始化
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors  # 返回原始 KV 缓存张量字典


# 将原始 int8 KV 缓存张量按照各注意力后端所需的形状和步幅顺序进行重塑（reshape）。
# 通过 view + permute 将扁平张量转换为 [num_blocks, block_size, num_heads, head_size] 等形状。
def _reshape_kv_cache(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    attn_backends: dict[str, AttentionBackend],
) -> dict[str, torch.Tensor]:
    kv_caches: dict[str, torch.Tensor] = {}  # 初始化重塑后的 KV 缓存字典
    for kv_cache_group_spec in kv_cache_config.kv_cache_groups:  # 遍历每个 KV 缓存组规格
        kv_cache_spec = kv_cache_group_spec.kv_cache_spec  # 获取该组的 KV 缓存规格
        assert isinstance(kv_cache_spec, AttentionSpec)  # 断言规格类型为 AttentionSpec
        for layer_name in kv_cache_group_spec.layer_names:  # 遍历该组中的每一层
            raw_tensor = kv_cache_raw_tensors[layer_name]  # 获取原始张量
            assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0  # 断言张量大小能被页大小整除
            num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes  # 计算块数量

            attn_backend = attn_backends[layer_name]  # 获取该层的注意力后端
            kv_cache_shape = attn_backend.get_kv_cache_shape(  # 获取后端要求的 KV 缓存形状
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            # FIXME(woosuk): Add kv_cache_stride_order to all attention backends.
            try:  # 尝试获取后端的步幅顺序
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()  # 获取 KV 缓存的步幅排列顺序
                assert len(kv_cache_stride_order) == len(kv_cache_shape)  # 断言步幅顺序维度与形状一致
            except (AttributeError, NotImplementedError):  # 如果后端未实现此方法
                kv_cache_stride_order = tuple(range(len(kv_cache_shape)))  # 使用默认的自然顺序

            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)  # 按步幅顺序重排形状
            inv_order = [  # 计算逆排列顺序，用于 permute 恢复到目标布局
                kv_cache_stride_order.index(i)
                for i in range(len(kv_cache_stride_order))
            ]

            dtype = kv_cache_spec.dtype  # 获取 KV 缓存的数据类型
            raw_tensor = raw_tensor.view(dtype)  # 将 int8 张量重新解释为目标数据类型
            raw_tensor = raw_tensor.view(kv_cache_shape)  # 将扁平张量重塑为目标形状
            kv_caches[layer_name] = raw_tensor.permute(*inv_order)  # 按逆序排列维度并存入字典
    return kv_caches  # 返回重塑后的 KV 缓存字典


# KV 缓存初始化入口：分配原始张量、按后端需求重塑、并绑定到前向传播上下文中。
def init_kv_cache(
    runner_kv_caches: list[torch.Tensor],  # 运行器的 KV 缓存列表
    forward_context: dict[str, Any],  # 前向传播上下文字典
    kv_cache_config: KVCacheConfig,  # KV 缓存配置
    attn_backends: dict[str, AttentionBackend],  # 注意力后端字典
    device: torch.device,  # 目标设备
) -> dict[str, torch.Tensor]:
    kv_cache_raw_tensors = _allocate_kv_cache(kv_cache_config, device)  # 分配原始 KV 缓存张量
    kv_caches = _reshape_kv_cache(kv_cache_config, kv_cache_raw_tensors, attn_backends)  # 按后端需求重塑张量
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)  # 将 KV 缓存绑定到前向传播上下文
    return kv_caches  # 返回初始化后的 KV 缓存字典


# 将按 KV 缓存组索引的 slot_mappings 展开为按层名称索引的字典，
# 使得每一层都能直接查找到自己的 slot 映射。
def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}  # 初始化按层名索引的 slot 映射字典
    kv_cache_groups = kv_cache_config.kv_cache_groups  # 获取 KV 缓存组列表
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):  # 遍历每个组的 slot 映射和组配置
        for layer_name in kv_cache_group.layer_names:  # 遍历组内的每个层名
            slot_mappings_by_layer[layer_name] = slot_mapping  # 将 slot 映射关联到层名
    return slot_mappings_by_layer  # 返回按层名索引的 slot 映射字典


# 构建注意力元数据：为每个 KV 缓存组和注意力分组生成 CommonAttentionMetadata，
# 其中包含 query 位置、序列长度、block table、slot mapping 等信息，
# 供注意力后端在前向传播中使用。支持编码器-解码器模型和分布式上下文并行（DCP）。
def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],  # 注意力分组列表
    num_reqs: int,  # 当前批次的请求数量
    num_tokens: int,  # 当前批次的 token 总数
    query_start_loc_gpu: torch.Tensor,  # GPU 上的 query 起始位置张量
    query_start_loc_cpu: torch.Tensor,  # CPU 上的 query 起始位置张量
    max_query_len: int,  # 最大 query 长度
    seq_lens: torch.Tensor,  # 序列长度张量
    max_seq_len: int,  # 最大序列长度
    block_tables: Sequence[torch.Tensor],  # 块表序列
    slot_mappings: torch.Tensor,  # 槽位映射张量
    kv_cache_config: KVCacheConfig,  # KV 缓存配置
    dcp_local_seq_lens: torch.Tensor | None = None,  # DCP 本地序列长度（可选）
    encoder_seq_lens: dict[int, tuple[torch.Tensor, np.ndarray]] | None = None,  # 编码器序列长度（可选）
) -> dict[str, Any]:
    seq_lens = seq_lens[:num_reqs]  # 截取实际请求数量的序列长度
    if dcp_local_seq_lens is not None:  # 如果存在 DCP 本地序列长度
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]  # 截取实际请求数量的 DCP 本地序列长度

    attn_metadata: dict[str, Any] = {}  # 初始化注意力元数据字典
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)  # 获取 KV 缓存组数量
    for i in range(num_kv_cache_groups):  # 遍历每个 KV 缓存组
        block_table = block_tables[i]  # 获取该组的块表
        slot_mapping = slot_mappings[i]  # 获取该组的槽位映射

        common_attn_metadata = CommonAttentionMetadata(  # 构建通用注意力元数据
            query_start_loc=query_start_loc_gpu,  # GPU 上的 query 起始位置
            query_start_loc_cpu=query_start_loc_cpu,  # CPU 上的 query 起始位置
            seq_lens=seq_lens,  # 序列长度
            max_seq_len=max_seq_len,  # 最大序列长度
            num_reqs=num_reqs,  # 请求数量
            num_actual_tokens=num_tokens,  # 实际 token 数量
            max_query_len=max_query_len,  # 最大 query 长度
            block_table_tensor=block_table,  # 块表张量
            slot_mapping=slot_mapping,  # 槽位映射
            causal=True,  # 使用因果注意力掩码
            dcp_local_seq_lens=dcp_local_seq_lens,  # DCP 本地序列长度
        )
        if encoder_seq_lens and i in encoder_seq_lens:  # 如果存在编码器序列长度且当前组有对应信息
            encoder_seq_lens_gpu, encoder_seq_lens_cpu = encoder_seq_lens[i]  # 解包 GPU 和 CPU 端的编码器序列长度
            common_attn_metadata.encoder_seq_lens = encoder_seq_lens_gpu  # 设置 GPU 端编码器序列长度
            common_attn_metadata.encoder_seq_lens_cpu = encoder_seq_lens_cpu  # 设置 CPU 端编码器序列长度

        for attn_group in attn_groups[i]:  # 遍历该 KV 缓存组中的每个注意力分组
            attn_metadata_builder = attn_group.get_metadata_builder(0)  # 获取该分组的元数据构建器
            metadata = attn_metadata_builder.build(  # 构建注意力元数据
                common_prefix_len=0, common_attn_metadata=common_attn_metadata
            )
            for layer_name in attn_group.layer_names:  # 遍历该分组包含的所有层
                attn_metadata[layer_name] = metadata  # 将元数据关联到层名
    return attn_metadata  # 返回注意力元数据字典
