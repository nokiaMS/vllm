# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass  # 导入数据类装饰器
from typing import ClassVar  # 导入类变量类型提示

import torch  # 导入PyTorch库

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.config.cache import CacheDType  # 导入缓存数据类型定义
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.model_executor.layers.attention.mla_attention import (  # 导入MLA注意力相关基类和工具
    MLACommonBackend,  # MLA通用后端基类
    MLACommonDecodeMetadata,  # MLA通用解码元数据基类
    MLACommonImpl,  # MLA通用实现基类
    MLACommonMetadata,  # MLA通用元数据基类
    MLACommonMetadataBuilder,  # MLA通用元数据构建器基类
    QueryLenSupport,  # 查询长度支持枚举
)
from vllm.model_executor.layers.batch_invariant import (  # 导入批次不变性相关工具
    vllm_is_batch_invariant,  # 检查当前是否处于批次不变模式
)
from vllm.platforms.interface import DeviceCapability  # 导入设备能力接口
from vllm.utils.platform_utils import num_compute_units  # 导入计算单元数量获取函数
from vllm.v1.attention.backend import (  # 导入注意力后端相关类
    AttentionCGSupport,  # 注意力CUDA Graph支持枚举
    AttentionLayer,  # 注意力层接口
    AttentionType,  # 注意力类型枚举
    MultipleOf,  # 倍数约束类
)
from vllm.v1.attention.backends.utils import (  # 导入注意力后端工具函数
    reshape_attn_output_for_spec_decode,  # 重塑注意力输出以适配推测解码
    reshape_query_for_spec_decode,  # 重塑查询以适配推测解码
)
from vllm.v1.attention.ops.flashmla import (  # 导入FlashMLA底层操作
    FlashMLASchedMeta,  # FlashMLA调度元数据类
    flash_mla_with_kvcache,  # FlashMLA带KV缓存的计算函数
    flash_mla_with_kvcache_fp8,  # FlashMLA带KV缓存的FP8计算函数
    get_mla_metadata,  # 获取MLA元数据函数
    get_mla_metadata_dense_fp8,  # 获取MLA稠密FP8元数据函数
    is_flashmla_dense_supported,  # 检查FlashMLA稠密模式是否支持
)
from vllm.v1.kv_cache_interface import AttentionSpec  # 导入注意力规格接口

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class FlashMLABackend(MLACommonBackend):
    """FlashMLA注意力后端，基于FlashMLA内核实现高效的MLA（Multi-head Latent Attention）计算。
    支持float16和bfloat16数据类型，以及多种KV缓存格式（包括FP8）。
    仅支持SM90（Hopper）和SM100（Blackwell）架构的GPU。"""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]  # 支持的数据类型列表
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [  # 支持的KV缓存数据类型列表
        "auto",  # 自动选择
        "bfloat16",  # BF16格式
        "fp8",  # FP8格式
        "fp8_e4m3",  # FP8 E4M3格式
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取支持的内核块大小列表，FlashMLA仅支持块大小为64"""
        return [64]  # 返回支持的块大小

    @staticmethod
    def get_name() -> str:
        """获取后端名称"""
        return "FLASHMLA"  # 返回后端标识名

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        """获取元数据构建器类"""
        return FlashMLAMetadataBuilder  # 返回FlashMLA元数据构建器类

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        """获取实现类"""
        return FlashMLAImpl  # 返回FlashMLA实现类

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持指定的GPU计算能力，仅支持SM90和SM100架构"""
        return capability.major in [9, 10]  # 支持Hopper(9)和Blackwell(10)架构

    @classmethod
    def supports_combination(
        cls,
        head_size: int,  # 注意力头大小
        dtype: torch.dtype,  # 数据类型
        kv_cache_dtype: CacheDType | None,  # KV缓存数据类型
        block_size: int | None,  # 块大小
        use_mla: bool,  # 是否使用MLA
        has_sink: bool,  # 是否有注意力汇聚
        use_sparse: bool,  # 是否使用稀疏注意力
        device_capability: DeviceCapability,  # 设备计算能力
    ) -> str | None:
        """检查给定的参数组合是否被支持，返回None表示支持，返回字符串表示不支持的原因"""
        if use_sparse:  # 如果使用稀疏注意力
            from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported  # 导入稀疏支持检查函数

            return is_flashmla_sparse_supported()[1]  # 返回稀疏模式支持状态
        else:  # 如果使用稠密注意力
            from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported  # 导入稠密支持检查函数

            return is_flashmla_dense_supported()[1]  # 返回稠密模式支持状态


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    """FlashMLA解码阶段的元数据，继承自MLA通用解码元数据，
    额外包含FlashMLA调度元数据用于内核调度。"""

    scheduler_metadata: FlashMLASchedMeta  # FlashMLA调度元数据


@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    """FlashMLA的完整元数据类，参数化解码元数据为FlashMLADecodeMetadata。"""
    pass  # 继承基类所有字段，无需额外字段


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    """FlashMLA元数据构建器，负责构建FlashMLA所需的注意力元数据。
    支持CUDA Graph的统一批处理模式，要求统一的查询长度。"""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH  # CUDA Graph支持模式：统一批处理
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM  # 查询长度支持：统一长度
    reorder_batch_threshold: int = 128  # 小于此阈值的预填充请求将通过解码路径处理
    # ^ TODO(matt): tune this  # 待调优此阈值

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,  # KV缓存规格
        layer_names: list[str],  # 层名称列表
        vllm_config: VllmConfig,  # vLLM配置
        device: torch.device,  # 设备
    ):
        """初始化FlashMLA元数据构建器"""
        super().__init__(  # 调用父类构造函数
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata  # 传入FlashMLA元数据类
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(  # 获取查询头数量
            vllm_config.parallel_config  # 根据并行配置获取
        )

        self.cg_buf_tile_scheduler_metadata = None  # CUDA Graph的瓦片调度元数据缓冲区
        self.cg_buf_num_splits = None  # CUDA Graph的分片数量缓冲区
        self.is_fp8_kvcache = vllm_config.cache_config.cache_dtype.startswith("fp8")  # 判断是否使用FP8 KV缓存

        num_sms = num_compute_units(self.device.index)  # 获取GPU的SM（流多处理器）数量

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():  # 如果启用了完整CUDA Graph
            self.cg_buf_tile_scheduler_metadata = torch.zeros(  # 创建瓦片调度元数据缓冲区
                # Upper bound on size (<= #SMs, TileSchedulerMetaDataSize)  # 大小上界
                # TileSchedulerMetaDataSize = 8  # 每个SM的调度元数据大小为8个int32
                (num_sms, 8),  # 形状: (SM数量, 8)
                device=self.device,  # 放在GPU上
                dtype=torch.int32,  # 使用int32类型
            )
            self.cg_buf_num_splits = torch.empty(  # 创建分片数量缓冲区
                (vllm_config.scheduler_config.max_num_seqs + 1),  # 形状: 最大序列数+1
                device=self.device,  # 放在GPU上
                dtype=torch.int32,  # 使用int32类型
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,  # 块表张量
        seq_lens_device: torch.Tensor,  # 序列长度（设备端）
        max_seq_len: int,  # 最大序列长度
        query_start_loc_cpu: torch.Tensor,  # 查询起始位置（CPU端）
        query_start_loc_device: torch.Tensor,  # 查询起始位置（设备端）
        num_decode_tokens: int,  # 解码token数量
        dcp_tot_seq_lens_device: torch.Tensor | None,  # 数据并行总序列长度（设备端）
    ) -> FlashMLADecodeMetadata:
        """构建解码阶段的元数据，包含FlashMLA调度信息"""
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]  # 计算每个请求的查询长度
        # we use the max but all should be the same due to uniform length requirement  # 使用最大值但由于统一长度要求所有值应相同
        max_query_len = query_lens_cpu.max().item()  # 获取最大查询长度
        num_q_tokens_per_head_k = max_query_len * self.num_q_heads // 1  # 计算每个KV头对应的Q token数
        scheduler_metadata, _ = get_mla_metadata(  # 获取MLA调度元数据
            seq_lens_device,  # 序列长度
            num_q_tokens_per_head_k,  # 每个KV头的Q token数
            1,  # MQA for the decode path  # 解码路径使用MQA（多查询注意力），KV头数为1
            is_fp8_kvcache=self.is_fp8_kvcache,  # 是否使用FP8 KV缓存
        )
        if self.is_fp8_kvcache:  # 如果使用FP8 KV缓存
            tile_scheduler_metadata, num_splits = get_mla_metadata_dense_fp8(  # 获取FP8稠密模式的调度元数据
                seq_lens_device,  # 序列长度
                num_q_tokens_per_head_k,  # 每个KV头的Q token数
                1,  # MQA for the decode path  # 解码路径使用MQA
            )
            scheduler_metadata.tile_scheduler_metadata = tile_scheduler_metadata  # 设置瓦片调度元数据
            scheduler_metadata.num_splits = num_splits  # 设置分片数量

        return FlashMLADecodeMetadata(  # 返回FlashMLA解码元数据
            block_table=block_table_tensor,  # 块表
            seq_lens=seq_lens_device,  # 序列长度
            scheduler_metadata=scheduler_metadata,  # 调度元数据
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,  # 数据并行总序列长度
        )


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    """FlashMLA注意力实现类，实现基于FlashMLA内核的MQA（多查询注意力）前向计算。
    支持FP8和非FP8两种KV缓存模式，以及推测解码。"""

    can_return_lse_for_decode: bool = True  # 解码阶段可以返回log-sum-exp值

    def __init__(
        self,
        num_heads: int,  # 注意力头数量
        head_size: int,  # 每个注意力头的维度
        scale: float,  # 注意力缩放因子
        num_kv_heads: int,  # KV头数量
        alibi_slopes: list[float] | None,  # ALiBi位置编码的斜率
        sliding_window: int | None,  # 滑动窗口大小
        kv_cache_dtype: str,  # KV缓存数据类型
        logits_soft_cap: float | None,  # logits软上限
        attn_type: str,  # 注意力类型
        kv_sharing_target_layer_name: str | None,  # KV共享目标层名称
        # MLA Specific Arguments  # MLA特定参数
        **mla_args,  # 其他MLA参数
    ) -> None:
        """初始化FlashMLA实现，验证支持性并检查不支持的特性"""
        super().__init__(  # 调用父类构造函数
            num_heads,  # 注意力头数
            head_size,  # 头维度
            scale,  # 缩放因子
            num_kv_heads,  # KV头数
            alibi_slopes,  # ALiBi斜率
            sliding_window,  # 滑动窗口
            kv_cache_dtype,  # KV缓存类型
            logits_soft_cap,  # 软上限
            attn_type,  # 注意力类型
            kv_sharing_target_layer_name,  # KV共享层名
            **mla_args,  # 其他MLA参数
        )

        is_supported, reason = is_flashmla_dense_supported()  # 检查FlashMLA稠密模式是否支持
        assert is_supported, reason  # 断言支持，否则报错

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]  # 不支持的特性列表
        if any(unsupported_features):  # 如果任何不支持的特性被启用
            raise NotImplementedError(  # 抛出未实现错误
                "FlashMLAImpl does not support one of the following: "  # FlashMLA不支持以下特性之一
                "alibi_slopes, sliding_window, logits_soft_cap"  # ALiBi斜率、滑动窗口、logits软上限
            )

        if attn_type != AttentionType.DECODER:  # 如果注意力类型不是解码器
            raise NotImplementedError(  # 抛出未实现错误
                "Encoder self-attention and "  # 编码器自注意力和
                "encoder/decoder cross-attention "  # 编码器/解码器交叉注意力
                "are not implemented for "  # 未在FlashMLA中实现
                "FlashMLAImpl"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],  # 查询张量或查询元组(nope部分, pe部分)
        kv_c_and_k_pe_cache: torch.Tensor,  # KV压缩缓存和K位置编码缓存
        attn_metadata: FlashMLAMetadata,  # FlashMLA元数据
        layer: AttentionLayer,  # 注意力层
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MQA（多查询注意力）前向计算，执行FlashMLA解码内核。
        支持FP8和非FP8两种KV缓存模式，以及推测解码的查询重塑。
        返回注意力输出和log-sum-exp值。"""
        # TODO: (zyongye) decode function for mla here  # 待实现：MLA的解码函数
        assert kv_c_and_k_pe_cache.numel() > 0  # 断言KV缓存非空
        assert attn_metadata.decode is not None  # 断言解码元数据存在

        if type(q) is tuple:  # 如果q是元组形式
            q = torch.cat(q, dim=-1)  # 将nope和pe部分拼接

        # mypy assertion: q is now always a tensor  # mypy类型断言：q现在一定是张量
        assert isinstance(q, torch.Tensor)  # 断言q是张量类型

        num_decodes = attn_metadata.num_decodes  # 获取解码请求数量
        q = reshape_query_for_spec_decode(q, num_decodes)  # 为推测解码重塑查询

        scheduler_metadata = attn_metadata.decode.scheduler_metadata  # 获取调度元数据
        if vllm_is_batch_invariant() and not self.kv_cache_dtype.startswith("fp8"):  # 如果处于批次不变模式且非FP8缓存
            device = q.device  # 获取设备
            dtype = torch.int32  # 使用int32数据类型

            B = q.shape[0]  # 批大小
            # block_table shape: [batch_size, max_num_blocks_per_seq]  # 块表形状
            # The number of blocks per sequence is in the second dimension  # 每个序列的块数在第二维
            topk = attn_metadata.decode.block_table.shape[-1]  # 获取每个序列的最大块数
            B_TOPK = 64  # 每个瓦片处理的块数
            assert topk % B_TOPK == 0, f"topk ({topk}) must be divisible by {B_TOPK}"  # 断言块数能被64整除
            end_block_idx = topk // B_TOPK  # 计算结束块索引

            # Single partition => num_sm_parts = 1  # 单分区 => SM分片数 = 1
            # TileSchedulerMetaDataSize = 8, layout:  # 瓦片调度元数据大小为8，布局如下
            # [begin_idx, begin_block_idx, end_idx, end_block_idx,  # [起始索引, 起始块索引, 结束索引, 结束块索引,
            #  begin_n_split_idx, _, _, _]  #  起始N分片索引, 保留, 保留, 保留]
            tile_scheduler_metadata = torch.zeros((1, 8), dtype=dtype, device=device)  # 创建瓦片调度元数据
            tile_scheduler_metadata[0, 0] = 0  # begin_idx  # 起始索引
            tile_scheduler_metadata[0, 1] = 0  # sched_begin_block_idx  # 调度起始块索引
            tile_scheduler_metadata[0, 2] = B - 1  # end_idx  # 结束索引
            tile_scheduler_metadata[0, 3] = end_block_idx  # 结束块索引
            tile_scheduler_metadata[0, 4] = 0  # begin_n_split_idx  # 起始N分片索引
            # fields [5..7] stay 0  # 字段[5..7]保持为0

            # Non-split path ignores num_splits, but the API requires it:  # 非分片路径忽略num_splits，但API需要它
            # zeros of length B+1  # 长度为B+1的零张量
            num_splits = torch.zeros((B + 1,), dtype=dtype, device=device)  # 创建分片数量张量
            scheduler_metadata.tile_scheduler_metadata = tile_scheduler_metadata  # 设置瓦片调度元数据
            scheduler_metadata.num_splits = num_splits  # 设置分片数量

        if self.kv_cache_dtype.startswith("fp8"):  # 如果使用FP8 KV缓存
            o, lse = flash_mla_with_kvcache_fp8(  # 调用FP8版FlashMLA内核
                q=q,  # 查询张量
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1  # 添加维度为1的头维度
                block_table=attn_metadata.decode.block_table,  # 块表
                cache_seqlens=attn_metadata.decode.seq_lens,  # 缓存序列长度
                head_dim_v=self.kv_lora_rank,  # V的头维度（等于KV LoRA秩）
                tile_scheduler_metadata=scheduler_metadata.tile_scheduler_metadata,  # 瓦片调度元数据
                num_splits=scheduler_metadata.num_splits,  # 分片数量
                softmax_scale=self.scale,  # softmax缩放因子
                causal=True,  # 因果注意力
                descale_q=layer._q_scale.reshape(1),  # Q的反量化缩放因子
                descale_k=layer._k_scale.reshape(1),  # K的反量化缩放因子
            )
        else:  # 如果使用非FP8 KV缓存
            o, lse = flash_mla_with_kvcache(  # 调用标准FlashMLA内核
                q=q,  # 查询张量
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1  # 添加维度为1的头维度
                block_table=attn_metadata.decode.block_table,  # 块表
                cache_seqlens=attn_metadata.decode.seq_lens,  # 缓存序列长度
                head_dim_v=self.kv_lora_rank,  # V的头维度
                tile_scheduler_metadata=scheduler_metadata,  # 调度元数据
                softmax_scale=self.scale,  # softmax缩放因子
                causal=True,  # 因果注意力
                is_fp8_kvcache=False,  # 不使用FP8 KV缓存
            )

        o = reshape_attn_output_for_spec_decode(o)  # 为推测解码重塑注意力输出

        return o, lse  # 返回注意力输出和log-sum-exp值
