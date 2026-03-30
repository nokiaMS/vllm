# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from dataclasses import dataclass  # 导入数据类装饰器
from functools import partial  # 导入偏函数工具
from typing import ClassVar  # 导入类变量类型注解

import numpy as np  # 导入numpy数值计算库
import torch  # 导入PyTorch深度学习框架
from flashinfer import (  # 从flashinfer库导入注意力包装器
    BatchDecodeWithPagedKVCacheWrapper,  # 批量解码分页KV缓存包装器
    BatchPrefillWithPagedKVCacheWrapper,  # 批量预填充分页KV缓存包装器
    BatchPrefillWithRaggedKVCacheWrapper,  # 批量预填充不规则KV缓存包装器
    MultiLevelCascadeAttentionWrapper,  # 多级级联注意力包装器
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache  # 导入快速解码计划和TRTLLM批量解码函数
from flashinfer.prefill import trtllm_batch_context_with_kv_cache  # 导入TRTLLM批量上下文预填充函数
from flashinfer.utils import FP4Tensor  # 导入FP4张量工具类
from typing_extensions import override  # 导入方法重写装饰器

from vllm import envs  # 导入vLLM环境变量配置
from vllm.config import (  # 从vLLM配置模块导入
    CUDAGraphMode,  # CUDA图模式枚举
    VllmConfig,  # vLLM配置类
    get_current_vllm_config_or_none,  # 获取当前vLLM配置或None
)
from vllm.config.cache import CacheDType  # 导入缓存数据类型
from vllm.distributed.parallel_state import get_dcp_group  # 导入获取DCP分布式组函数
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.model_executor.layers.batch_invariant import (  # 导入批次不变性相关
    vllm_is_batch_invariant,  # 判断是否启用批次不变性模式
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # 导入量化工具
    QuantKey,  # 量化键类型
    kFp8StaticTensorSym,  # FP8静态张量对称量化常量
    kNvfp4Dynamic,  # NVFP4动态量化常量
)
from vllm.platforms import current_platform  # 导入当前平台信息
from vllm.platforms.interface import DeviceCapability  # 导入设备计算能力接口
from vllm.triton_utils import tl, triton  # 导入Triton内核编程工具
from vllm.utils.flashinfer import (  # 导入FlashInfer辅助工具函数
    can_use_trtllm_attention,  # 判断是否可以使用TRTLLM注意力
    use_trtllm_attention,  # 判断是否使用TRTLLM注意力
)
from vllm.utils.math_utils import cdiv  # 导入向上取整除法函数
from vllm.utils.platform_utils import is_pin_memory_available  # 导入判断锁页内存是否可用
from vllm.utils.torch_utils import is_strictly_contiguous  # 导入判断张量是否严格连续
from vllm.v1.attention.backend import (  # 从注意力后端模块导入
    AttentionBackend,  # 注意力后端基类
    AttentionCGSupport,  # 注意力CUDA图支持级别
    AttentionImpl,  # 注意力实现基类
    AttentionMetadataBuilder,  # 注意力元数据构建器基类
    AttentionType,  # 注意力类型枚举
    CommonAttentionMetadata,  # 通用注意力元数据
    MultipleOf,  # 倍数约束类型
)
from vllm.v1.attention.backends.utils import (  # 从注意力后端工具模块导入
    KVCacheLayoutType,  # KV缓存布局类型
    get_dcp_local_seq_lens,  # 获取DCP本地序列长度
    get_kv_cache_layout,  # 获取KV缓存布局
    get_per_layer_parameters,  # 获取每层参数
    infer_global_hyperparameters,  # 推断全局超参数
    split_decodes_and_prefills,  # 分离解码和预填充请求
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs  # 导入上下文并行LSE聚合归约函数
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce  # 导入DCP全对全LSE归约函数
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # 导入合并注意力状态函数
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs  # 导入注意力规格和统一类型KV缓存规格
from vllm.v1.utils import CpuGpuBuffer  # 导入CPU-GPU双缓冲区工具类

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024  # 批次不变性模式下的工作空间缓冲区大小（2GB）

FP8_DTYPE = current_platform.fp8_dtype()  # 获取当前平台的FP8数据类型
FP4_DTYPE = torch.uint8  # FP4数据类型使用uint8存储

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

trtllm_gen_workspace_buffer = None  # TRTLLM生成工作空间缓冲区，初始为空


def _get_trtllm_gen_workspace_buffer():
    """获取或创建TRTLLM生成阶段的工作空间缓冲区。"""
    global trtllm_gen_workspace_buffer  # 声明使用全局变量
    if trtllm_gen_workspace_buffer is None:  # 如果缓冲区尚未创建
        trtllm_gen_workspace_buffer = torch.zeros(  # 创建零初始化的GPU缓冲区
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"  # 使用环境变量指定的大小，uint8类型，分配在CUDA设备上
        )
    return trtllm_gen_workspace_buffer  # 返回工作空间缓冲区


@triton.jit  # Triton即时编译装饰器
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,  # KV缓存指针
    block_tables_prefill_ptr,  # 预填充块表指针
    block_table_stride,  # 块表步长
    mock_kv_cache_ptr,  # 模拟KV缓存指针（用于存储反量化结果）
    k_scale_ptr,  # K缓存的缩放因子指针
    v_scale_ptr,  # V缓存的缩放因子指针
    K_CACHE_STRIDE: tl.constexpr,  # K缓存步长（编译时常量）
    KV_CACHE_STRIDE: tl.constexpr,  # KV缓存步长（编译时常量）
):
    """Triton内核：将FP8格式的KV缓存反量化为高精度格式，用于TRTLLM预填充注意力。"""
    batch_idx = tl.program_id(0).to(tl.int64)  # 获取批次索引（第0维度的程序ID）
    mock_block_table_idx = tl.program_id(1).to(tl.int64)  # 获取模拟块表索引（第1维度的程序ID）
    orig_page_num = tl.load(  # 从块表中加载原始页号
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx  # 计算块表中的偏移地址
    ).to(tl.int64)  # 转换为int64类型
    if orig_page_num <= 0:  # 如果页号无效（<=0），直接返回
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty  # 获取反量化目标数据类型

    # Dequantize K
    k_scale_val = tl.load(k_scale_ptr)  # 加载K缓存的缩放因子
    offset = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)  # 计算K缓存在原始KV缓存中的偏移
    fp8_vals = tl.load(kv_cache_ptr + offset)  # 从KV缓存中加载FP8格式的K值
    dequantized_vals = fp8_vals.to(tl.float32) * k_scale_val  # 将FP8值转换为float32并乘以缩放因子完成反量化
    mock_cache_offset = (  # 计算反量化后数据在模拟缓存中的存储偏移
        batch_idx * block_table_stride + mock_block_table_idx + 1  # 计算模拟块表中的页索引
    ) * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)  # 乘以KV步长并加上K缓存内偏移
    dequantized_vals = dequantized_vals.to(dequant_dtype)  # 转换为目标数据类型
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)  # 将反量化后的K值存储到模拟缓存

    # Dequantize V
    v_scale_val = tl.load(v_scale_ptr)  # 加载V缓存的缩放因子
    offset = (  # 计算V缓存在原始KV缓存中的偏移
        orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)  # V在K之后，需要加上K的步长
    )
    fp8_vals = tl.load(kv_cache_ptr + offset)  # 从KV缓存中加载FP8格式的V值
    dequantized_vals = fp8_vals.to(tl.float32) * v_scale_val  # 将FP8值转换为float32并乘以缩放因子完成反量化
    mock_cache_offset = (  # 计算反量化后V数据在模拟缓存中的存储偏移
        (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE  # 计算页级偏移
        + K_CACHE_STRIDE  # V在K之后，加上K的步长
        + tl.arange(0, K_CACHE_STRIDE)  # 加上V缓存内偏移
    )
    dequantized_vals = dequantized_vals.to(dequant_dtype)  # 转换为目标数据类型
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)  # 将反量化后的V值存储到模拟缓存


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,  # 原始KV缓存张量
    block_tables_prefill: torch.Tensor,  # 预填充请求的块表
    k_scale: torch.Tensor,  # K缓存缩放因子
    v_scale: torch.Tensor,  # V缓存缩放因子
    dequant_dtype: torch.dtype,  # 反量化目标数据类型
) -> tuple[torch.Tensor, torch.Tensor]:
    """将FP8的KV缓存反量化为指定精度，返回模拟KV缓存和对应的模拟块表。用于TRTLLM预填充不支持BF16查询+FP8 KV缓存的情况。"""
    batch_size, num_of_page_per_token = block_tables_prefill.shape  # 获取批次大小和每个token的页数
    s = kv_cache.shape  # 获取KV缓存的形状
    assert s[1] == 2  # 断言第二维为2（K和V两个缓存）
    assert dequant_dtype in (torch.bfloat16, torch.float16)  # 断言反量化类型为bfloat16或float16
    k_cache_stride = s[2] * s[3] * s[4]  # 计算单个K缓存页的元素数量
    kv_cache_stride = k_cache_stride * s[1]  # 计算单个KV缓存页的元素数量（K+V）
    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])  # 计算模拟KV缓存的形状（多一个页用于填充）
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)  # 创建模拟KV缓存（仅包含预填充需要的页）
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(  # 创建顺序索引的模拟块表
        start=1,  # 从1开始（0号页保留）
        end=batch_size * num_of_page_per_token + 1,  # 到总页数+1
        dtype=torch.int32,  # 使用int32类型
        device=block_tables_prefill.device,  # 分配在与块表相同的设备上
    ).reshape(batch_size, num_of_page_per_token)  # 重塑为与原始块表相同的形状
    grid = (batch_size, num_of_page_per_token)  # 定义Triton内核的网格大小
    _trtllm_prefill_attn_kvfp8_dequant[grid](  # 启动Triton反量化内核
        kv_cache,  # 原始KV缓存
        block_tables_prefill,  # 预填充块表
        num_of_page_per_token,  # 每个token的页数（作为步长）
        mock_kv_cache,  # 模拟KV缓存（输出）
        k_scale,  # K缩放因子
        v_scale,  # V缩放因子
        k_cache_stride,  # K缓存步长
        kv_cache_stride,  # KV缓存步长
    )
    return mock_kv_cache, mock_block_table  # 返回模拟KV缓存和模拟块表


class BatchDCPPrefillWrapper:
    """批量DCP（解码上下文并行）预填充包装器。将预填充拆分为上下文注意力和新token注意力两部分，并通过DCP组进行通信合并。"""
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,  # 工作空间缓冲区
        dcp_a2a: bool = False,  # 是否使用全对全通信模式
    ):
        """初始化DCP预填充包装器，设置通信方式和内部注意力包装器。"""
        if dcp_a2a:  # 如果使用全对全通信模式
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)  # 设置DCP合并函数为全对全LSE归约
        else:  # 否则使用聚合-归约通信模式
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)  # 设置DCP合并函数为聚合输出归约
        self._context = BatchPrefillWithPagedKVCacheWrapper(  # 创建上下文注意力包装器（处理已有KV缓存）
            workspace_buffer, get_kv_cache_layout()  # 传入工作空间和KV缓存布局
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(  # 创建新token注意力包装器（处理新生成的KV）
            workspace_buffer, get_kv_cache_layout()  # 传入工作空间和KV缓存布局
        )

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,  # 查询/输出索引指针（CPU张量）
        paged_kv_indptr_cpu: torch.Tensor,  # 分页KV索引指针（CPU张量）
        paged_kv_indices: torch.Tensor,  # 分页KV索引
        paged_kv_last_page_len_cpu: torch.Tensor,  # 最后一页长度（CPU张量）
        page_size: int,  # 页大小
        num_qo_heads: int,  # 查询/输出头数
        dcp_world_size: int,  # DCP并行组大小
        num_kv_heads: int,  # KV头数
        head_dim: int,  # 头维度
        sm_scale: float,  # softmax缩放因子
        window_left: int,  # 滑动窗口左边界
        logits_soft_cap: float | None,  # logits软上限
        q_data_type: torch.dtype,  # 查询数据类型
        kv_cache_dtype: torch.dtype,  # KV缓存数据类型
        prefill_fixed_split_size: int,  # 预填充固定分割大小
        disable_split_kv: bool,  # 是否禁用KV分割
    ):
        """规划DCP预填充操作，分别为上下文注意力和新token注意力配置参数。"""
        """Plan the prefill operation with given parameters."""
        self._context.plan(  # 规划上下文注意力操作
            qo_indptr=qo_indptr_cpu,  # 查询/输出索引指针
            paged_kv_indptr=paged_kv_indptr_cpu,  # 分页KV索引指针
            paged_kv_indices=paged_kv_indices,  # 分页KV索引
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,  # 最后一页长度
            num_qo_heads=num_qo_heads * dcp_world_size,  # 查询头数乘以DCP并行度（跨DCP组聚合）
            num_kv_heads=num_kv_heads,  # KV头数
            head_dim_qk=head_dim,  # QK头维度
            page_size=page_size,  # 页大小
            causal=False,  # This is context run  # 上下文运行不需要因果掩码
            sm_scale=sm_scale,  # softmax缩放因子
            window_left=window_left,  # 滑动窗口左边界
            logits_soft_cap=logits_soft_cap,  # logits软上限
            q_data_type=q_data_type,  # 查询数据类型
            kv_data_type=kv_cache_dtype,  # KV数据类型
            fixed_split_size=prefill_fixed_split_size,  # 固定分割大小
            disable_split_kv=disable_split_kv,  # 是否禁用KV分割
        )
        self._new_tokens.plan(  # 规划新token注意力操作
            qo_indptr=qo_indptr_cpu,  # 查询/输出索引指针
            kv_indptr=qo_indptr_cpu,  # KV索引指针（与查询相同，因为是新token的自注意力）
            num_qo_heads=num_qo_heads,  # 查询头数（不需要乘DCP并行度）
            num_kv_heads=num_kv_heads,  # KV头数
            head_dim_qk=head_dim,  # QK头维度
            head_dim_vo=head_dim,  # VO头维度
            causal=True,  # This is newtokens run  # 新token运行需要因果掩码
            sm_scale=sm_scale,  # softmax缩放因子
            window_left=window_left,  # 滑动窗口左边界
            logits_soft_cap=logits_soft_cap,  # logits软上限
            q_data_type=q_data_type,  # 查询数据类型
        )

    def run(
        self,
        layer: torch.nn.Module,  # 注意力层模块
        prefill_query: torch.Tensor,  # 预填充查询张量
        kv_cache_permute: torch.Tensor,  # 重排后的KV缓存张量
        key: torch.Tensor,  # 键张量
        value: torch.Tensor,  # 值张量
        out: torch.Tensor,  # 输出张量
    ):
        """执行DCP预填充注意力计算，将上下文注意力和新token注意力的结果合并。"""
        prefill_query_across_dcp = get_dcp_group().all_gather(  # 跨DCP组全收集预填充查询
            prefill_query.contiguous(), dim=1  # 确保连续并在头维度上聚合
        )
        output_context_tmp, lse_context_tmp = self._context.run(  # 执行上下文注意力计算
            prefill_query_across_dcp,  # 跨DCP聚合后的查询
            kv_cache_permute,  # 重排后的KV缓存
            k_scale=layer._k_scale_float,  # K缩放因子
            v_scale=layer._v_scale_float,  # V缩放因子
            return_lse=True,  # 返回log-sum-exp值
        )
        output_context, lse_context = self._dcp_combine(  # 通过DCP通信合并上下文注意力结果
            output_context_tmp,  # 临时上下文输出
            lse_context_tmp,  # 临时上下文LSE
            get_dcp_group(),  # DCP通信组
            return_lse=True,  # 返回合并后的LSE
        )
        lse_context = lse_context.transpose(0, 1).contiguous()  # 转置LSE维度并确保连续

        output_query, lse_query = self._new_tokens.run(  # 执行新token注意力计算
            prefill_query,  # 预填充查询
            key,  # 键
            value,  # 值
            return_lse=True,  # 返回LSE
        )
        lse_query = lse_query.transpose(0, 1).contiguous()  # 转置LSE维度并确保连续

        merge_attn_states(  # 合并上下文注意力和新token注意力的状态
            out,  # 输出张量
            output_context,  # 上下文注意力输出
            lse_context,  # 上下文LSE
            output_query,  # 新token注意力输出
            lse_query,  # 新token LSE
        )
        return out  # 返回合并后的输出


class FlashInferBackend(AttentionBackend):
    """FlashInfer注意力后端类，定义FlashInfer作为注意力计算后端的配置和接口。"""
    accept_output_buffer: bool = True  # 接受外部提供的输出缓冲区
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]  # 支持的数据类型列表
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [  # 支持的KV缓存数据类型列表
        "auto",  # 自动选择
        "bfloat16",  # BF16格式
        "fp8",  # FP8格式
        "fp8_e4m3",  # FP8 E4M3格式
        "fp8_e5m2",  # FP8 E5M2格式
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取FlashInfer支持的内核块大小列表。"""
        # Note: Not sure for all platforms, but on Blackwell,
        # only support a page size of 16, 32, 64.
        return [16, 32, 64]  # 返回支持的块大小：16、32、64

    @staticmethod
    def get_name() -> str:
        """获取后端名称。"""
        return "FLASHINFER"  # 返回后端名称字符串

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        """获取FlashInfer注意力实现类。"""
        return FlashInferImpl  # 返回FlashInfer实现类

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        """获取FlashInfer元数据构建器类。"""
        return FlashInferMetadataBuilder  # 返回FlashInfer元数据构建器类

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,  # 块数量
        block_size: int,  # 块大小
        num_kv_heads: int,  # KV头数量
        head_size: int,  # 头维度大小
        cache_dtype_str: str = "auto",  # 缓存数据类型字符串
    ) -> tuple[int, ...]:
        """获取KV缓存的形状。"""
        return (num_blocks, 2, block_size, num_kv_heads, head_size)  # 返回形状：(块数, 2(K和V), 块大小, KV头数, 头维度)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,  # 是否包含层数维度
    ) -> tuple[int, ...]:
        """获取KV缓存的步长顺序，用于将逻辑形状映射到实际内存布局。"""
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()  # 获取当前KV缓存布局类型
        if cache_layout == "NHD" and include_num_layers_dimension:  # NHD布局且包含层维度
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)  # 返回NHD多层布局的步长顺序
        elif cache_layout == "NHD":  # NHD布局不包含层维度
            stride_order = (0, 1, 2, 3, 4)  # NHD标准步长顺序
        elif cache_layout == "HND" and include_num_layers_dimension:  # HND布局且包含层维度
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)  # 返回HND多层布局的步长顺序
        elif cache_layout == "HND":  # HND布局不包含层维度
            stride_order = (0, 1, 3, 2, 4)  # HND标准步长顺序
        else:  # 未知布局类型
            raise ValueError(f"Unknown cache layout format {cache_layout}.")  # 抛出未知布局错误
        return stride_order  # 返回步长顺序

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        """根据KV缓存数据类型字符串获取对应的PyTorch FP8数据类型。"""
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):  # 如果是fp8或fp8_e4m3
            return torch.float8_e4m3fn  # 返回E4M3格式
        elif kv_cache_dtype == "fp8_e5m2":  # 如果是fp8_e5m2
            return torch.float8_e5m2  # 返回E5M2格式
        else:  # 无法识别的FP8类型
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")  # 抛出错误

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """获取FlashInfer支持的注意力头维度大小列表。"""
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256]  # 返回支持的头维度：64、128、256

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持给定的GPU计算能力。"""
        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(  # 支持计算能力7.5到12.1
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """检查FlashInfer是否支持注意力下沉（sink）。仅在TRTLLM注意力可用时（SM100架构）支持。"""
        """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
        from vllm.utils.flashinfer import (  # 延迟导入TRTLLM注意力辅助函数
            force_use_trtllm_attention,  # 强制使用TRTLLM注意力
            supports_trtllm_attention,  # 是否支持TRTLLM注意力
        )

        # Respect explicit disable flag (e.g.,
        # --attention-config.use_trtllm_attention=0)
        if force_use_trtllm_attention() is False:  # 如果显式禁用了TRTLLM注意力
            return False  # 不支持sink

        # Check if TRTLLM is supported on this platform
        return supports_trtllm_attention()  # 返回平台是否支持TRTLLM注意力

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        """获取FlashInfer要求的KV缓存布局类型。SM100（Blackwell）架构要求HND布局。"""
        capability = current_platform.get_device_capability()  # 获取当前设备的计算能力
        if capability is not None and capability.major == 10:  # 如果是SM100（Blackwell）架构
            return "HND"  # 返回HND布局要求
        return None  # 其他架构无特殊要求

    forward_includes_kv_cache_update: bool = False  # 前向传播不包含KV缓存更新


@dataclass
class FIPrefill:
    """FlashInfer原生预填充路径的元数据。包含用于非TRTLLM预填充的包装器。"""
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper  # 预填充包装器（分页KV或DCP模式）


@dataclass
class FIDecode:
    """FlashInfer原生解码路径的元数据。包含用于非TRTLLM解码的包装器。"""
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper  # 解码包装器


@dataclass
class TRTLLMPrefill:
    """TRTLLM预填充路径的元数据。包含块表、序列长度等预填充所需的信息。"""
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor  # 预填充请求的块表张量
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor  # 预填充请求的序列长度张量
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor  # 查询的累积序列长度
    cum_seq_lens_kv: torch.Tensor  # KV的累积序列长度

    max_q_len: int  # 预填充请求中最大的查询长度
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int  # KV缓存的最大序列长度
    """The maximum sequence length for KV Cache."""


@dataclass
class TRTLLMDecode:
    """TRTLLM解码路径的元数据。包含块表、序列长度等解码所需的信息。"""
    """Metadata for the TRTLLM decode pathway."""

    block_tables: torch.Tensor  # 解码请求的块表张量
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor  # 解码请求的序列长度张量
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int  # KV缓存的最大序列长度
    """The maximum sequence length for KV Cache."""


@dataclass
class FlashInferMetadata:
    """FlashInfer注意力元数据类，存储一次注意力计算所需的所有元信息，包括预填充、解码和级联注意力的配置。"""
    num_actual_tokens: int  # 批次中的实际token数量（不含填充）
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor  # 用于将K/V写入缓存的槽位映射张量
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    q_data_type: torch.dtype  # 查询的数据类型

    num_decodes: int  # 解码请求的数量
    num_decode_tokens: int  # 解码token的数量
    num_prefills: int  # 预填充请求的数量
    num_prefill_tokens: int  # 预填充token的数量

    prefill: FIPrefill | TRTLLMPrefill | None  # 预填充路径的元数据（FI原生或TRTLLM或None）
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | TRTLLMDecode | None  # 解码路径的元数据（FI原生或TRTLLM或None）
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    # --- Special Case: Cascade Attention ---

    use_cascade: bool  # 是否使用级联注意力
    """
    If True, the entire batch is a cascade attention call, and the
    `prefill` and `decode` fields will both be None.
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None  # 级联注意力包装器（不使用时为None）


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    """FlashInfer注意力元数据构建器，负责根据批次信息构建FlashInfer所需的元数据，包括预填充和解码的规划。"""
    reorder_batch_threshold: int = 1  # 批次重排序阈值

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,  # KV缓存规格
        layer_names: list[str],  # 注意力层名称列表
        vllm_config: VllmConfig,  # vLLM配置
        device: torch.device,  # 目标设备
    ):
        """初始化FlashInfer元数据构建器，设置缓存配置、模型配置、包装器和持久缓冲区。"""
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)  # 调用父类初始化
        self.cache_config = vllm_config.cache_config  # 缓存配置
        self.model_config = vllm_config.model_config  # 模型配置
        self.attention_config = vllm_config.attention_config  # 注意力配置
        self._workspace_buffer = None  # 工作空间缓冲区（延迟初始化）
        self._prefill_wrapper: (  # 预填充包装器类型注解
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append  # 预填充/追加包装器（延迟初始化）
        self._decode_wrapper = None  # Wrapper for decode (general shape)  # 解码包装器（通用形状，延迟初始化）

        if vllm_is_batch_invariant():  # 如果启用批次不变性模式
            self.decode_fixed_split_size = 2048  # 解码固定分割大小设为2048
            self.prefill_fixed_split_size = 4096  # 预填充固定分割大小设为4096
            self.disable_split_kv = True  # 禁用KV分割
        else:  # 非批次不变性模式
            self.decode_fixed_split_size = -1  # 解码不使用固定分割
            self.prefill_fixed_split_size = -1  # 预填充不使用固定分割
            self.disable_split_kv = False  # 不禁用KV分割

        self.compilation_config = vllm_config.compilation_config  # 编译配置
        max_num_pages_per_req = cdiv(  # 计算每个请求的最大页数
            self.model_config.max_model_len, self.kv_cache_spec.block_size  # 最大模型长度除以块大小（向上取整）
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs  # 最大并发请求数
        max_num_pages = max_num_reqs * max_num_pages_per_req  # 最大总页数
        speculative_config = vllm_config.speculative_config  # 推测解码配置
        num_spec_tokens = (  # 推测token数量
            speculative_config.num_speculative_tokens  # 如果配置了推测解码则取其值
            if speculative_config is not None
            else 0  # 否则为0
        )
        self.enable_cuda_graph = (  # 判断是否启用CUDA图
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL  # 解码模式为FULL时启用
        )
        if self.enable_cuda_graph:  # 如果启用了CUDA图
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[  # CUDA图模式下每个批次大小对应一个解码包装器
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs  # CUDA图最大批次大小
            if self.compilation_config.max_cudagraph_capture_size is not None:  # 如果设置了CUDA图捕获大小上限
                self._decode_cudagraph_max_bs = min(  # 取较小值
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        try:  # 尝试获取DCP分布式组信息
            self.dcp_world_size = get_dcp_group().world_size  # DCP组大小
            self.dcp_rank = get_dcp_group().rank_in_group  # DCP组内排名
            self.dcp_kv_cache_interleave_size = (  # DCP KV缓存交错大小
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:  # DCP未初始化（可能在测试环境中）
            # DCP might not be initialized in testing
            self.dcp_world_size = 1  # 默认DCP组大小为1
            self.dcp_rank = 0  # 默认DCP排名为0
            self.dcp_kv_cache_interleave_size = 1  # 默认交错大小为1
        self.use_dcp = self.dcp_world_size > 1  # 判断是否使用DCP（组大小>1时使用）
        self.dcp_a2a = (  # 判断是否使用DCP全对全通信
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"  # 使用DCP且通信后端为a2a
        )

        self.num_qo_heads = self.model_config.get_num_attention_heads(  # 获取查询/输出注意力头数
            self.vllm_config.parallel_config  # 传入并行配置
        )

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads  # KV注意力头数
        self.head_dim = self.kv_cache_spec.head_size  # 注意力头维度
        self.page_size = self.kv_cache_spec.block_size  # 页大小（等于块大小）

        self.cache_dtype = self.cache_config.cache_dtype  # 缓存数据类型字符串
        if self.cache_dtype.startswith("fp8"):  # 如果缓存类型是FP8
            self.kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(  # 获取FlashInfer对应的FP8类型
                self.cache_dtype
            )
        else:  # 非FP8缓存类型
            assert self.kv_cache_spec.dtype == self.model_config.dtype  # 断言KV缓存类型与模型类型一致
            self.kv_cache_dtype = self.kv_cache_spec.dtype  # 使用KV缓存规格中的数据类型

        # Use model dtype as q dtype when TRTLLM attn is not supported, or
        # --attention-config.disable_flashinfer_q_quantization is set to 1. Otherwise,
        # try to use fp8 q if kv cache is fp8, and will fall back to model dtype
        # if TRTLLM attention kernel is not used when building attn metadata
        can_use_trtllm = can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads)  # 检查是否可以使用TRTLLM注意力
        if (  # 如果可以使用TRTLLM且未禁用查询量化
            can_use_trtllm
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        ):
            self.q_data_type = self.kv_cache_dtype  # 查询类型与KV缓存类型一致（可能是FP8）
        else:  # 否则
            self.q_data_type = self.model_config.dtype  # 查询类型使用模型原始数据类型

        # Prefer TRTLLM attention for decoding in all cases.
        # This allows us to use AttentionCGSupport.UNIFORM_BATCH mode.
        self.use_trtllm_decode_attention = can_use_trtllm  # 解码是否使用TRTLLM注意力
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=can_use_trtllm)  # 初始化批次重排序阈值

        self._cascade_wrapper = None  # Wrapper for cascade attention  # 级联注意力包装器（延迟初始化）

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        self.global_hyperparameters = infer_global_hyperparameters(  # 推断所有注意力层共享的全局超参数
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl)  # 获取每层参数
        )
        self.sm_scale = self.global_hyperparameters.sm_scale  # softmax缩放因子
        self.window_left = self.global_hyperparameters.window_left  # 滑动窗口左边界
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap  # logits软上限
        self.has_sinks = self.global_hyperparameters.has_sinks  # 是否有注意力下沉
        if self.has_sinks and not can_use_trtllm:  # 如果有注意力下沉但不支持TRTLLM
            raise NotImplementedError(  # 抛出未实现错误
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        # Preparing persistent buffers
        # Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
        # reused CPU buffers to avoid a race condition between step N async copies to
        # GPU and step N+1 buffer updates.
        self.pin_memory = (  # 判断是否使用锁页内存
            not envs.VLLM_USE_V2_MODEL_RUNNER and is_pin_memory_available()  # V2模型运行器不使用锁页内存以避免竞态
        )
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)  # 创建分页KV索引指针缓冲区
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(  # 创建分页KV索引指针的额外CPU缓冲区
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory  # 用于CUDA图模式下的可变数据
        )  # Extra buffer for mutable paged_kv_indptr.cpu in cuda graph mode
        self.paged_kv_indices = self._make_buffer(max_num_pages)  # 创建分页KV索引缓冲区
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)  # 创建最后一页长度缓冲区

        if self.head_dim == 256 and current_platform.is_device_capability_family(100):  # 如果头维度为256且在Blackwell架构上
            # https://github.com/flashinfer-ai/flashinfer/issues/1993 reports that
            # head size 256 and block size 16 is not supported on blackwell.
            assert kv_cache_spec.block_size != 16, (  # 断言块大小不为16（已知bug）
                "There is a bug in FlashInfer "
                "block_size 16 head size 256 support. Please avoid this combination by "
                "passing --block-size 32 or --block-size 64."
            )

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype = torch.int32  # 缓冲区大小和数据类型参数
    ) -> CpuGpuBuffer:
        """创建一个CPU-GPU双缓冲区，用于高效的主机到设备数据传输。"""
        return CpuGpuBuffer(  # 返回CPU-GPU双缓冲区
            *size,
            dtype=dtype,  # 数据类型
            device=self.device,  # GPU设备
            pin_memory=self.pin_memory,  # 是否使用锁页内存
            with_numpy=True,  # 包含numpy视图
        )

    @override  # type: ignore[misc]  # 重写父类方法
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],  # 类方法的类型参数
        vllm_config: VllmConfig,  # vLLM配置
        kv_cache_spec: AttentionSpec,  # KV缓存规格
    ) -> AttentionCGSupport:
        """获取FlashInfer注意力的CUDA图支持级别。取决于是否所有KV缓存规格都支持TRTLLM注意力。"""
        """Get the cudagraph support level for FlashInfer attention.

        This depends on whether we can use TRTLLM attention for decodes, since we can
        only do UNIFORM_SINGLE_TOKEN_DECODE if it is unavailable.
        To check this, we must call can_use_trtllm_attention with the number of KV
        heads from the kv_cache_spec. We check all available KV cache specs and
        only return UNIFORM_BATCH if all of them support TRTLLM attention.
        """
        # For UniformTypeKVCacheSpecs, check all contained specs
        kv_specs = (  # 获取所有KV缓存规格
            kv_cache_spec.kv_cache_specs.values()  # 如果是统一类型规格则获取所有子规格
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]  # 否则作为单个规格处理
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(  # 获取查询/输出头数
            vllm_config.parallel_config  # 传入并行配置
        )
        has_trtllm_support: bool = len(kv_specs) > 0  # 初始假设支持TRTLLM
        for spec in kv_specs:  # 遍历所有KV缓存规格
            if not isinstance(spec, AttentionSpec):  # 如果不是注意力规格（如Mamba）
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue  # 跳过非注意力规格
            if not can_use_trtllm_attention(  # 检查是否支持TRTLLM注意力
                num_qo_heads=num_qo_heads,  # 查询头数
                num_kv_heads=spec.num_kv_heads,  # KV头数
            ):
                has_trtllm_support = False  # 标记不支持TRTLLM
                break  # 只要有一个不支持就退出

        if has_trtllm_support:  # 如果所有规格都支持TRTLLM
            return AttentionCGSupport.UNIFORM_BATCH  # 返回统一批次支持级别
        else:  # 否则
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE  # 返回仅统一单token解码支持级别

    def _get_workspace_buffer(self):
        """获取或创建FlashInfer的工作空间缓冲区。"""
        if self._workspace_buffer is None:  # 如果工作空间缓冲区尚未创建
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE  # 使用环境变量指定的缓冲区大小
            if vllm_is_batch_invariant():  # 如果启用批次不变性模式
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT  # 使用更大的缓冲区（2GB）
            self._workspace_buffer = torch.zeros(  # 创建零初始化的GPU缓冲区
                buffer_size, dtype=torch.uint8, device=self.device  # uint8类型，分配在指定设备上
            )
        return self._workspace_buffer  # 返回工作空间缓冲区

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        """设置外部提供的工作空间缓冲区。"""
        self._workspace_buffer = workspace_buffer  # 直接赋值工作空间缓冲区

    def _get_prefill_wrapper(
        self,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        """获取或创建预填充包装器，根据是否使用DCP选择不同的包装器类型。"""
        if self._prefill_wrapper is None:  # 如果预填充包装器尚未创建
            if self.use_dcp:  # 如果使用DCP
                self._prefill_wrapper = BatchDCPPrefillWrapper(  # 创建DCP预填充包装器
                    workspace_buffer=self._get_workspace_buffer(),  # 传入工作空间缓冲区
                    dcp_a2a=self.dcp_a2a,  # 传入是否使用全对全通信
                )
            else:  # 不使用DCP
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(  # 创建标准预填充包装器
                    self._get_workspace_buffer(), get_kv_cache_layout()  # 传入工作空间和KV缓存布局
                )
        assert self._prefill_wrapper is not None  # 断言包装器已创建
        return self._prefill_wrapper  # 返回预填充包装器

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        """获取或创建解码包装器。CUDA图模式下为每个批次大小维护单独的包装器。"""
        if use_cudagraph:  # 如果使用CUDA图
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)  # 从CUDA图包装器字典中查找
        else:  # 不使用CUDA图
            decode_wrapper = self._decode_wrapper  # 使用通用解码包装器

        if decode_wrapper is None:  # 如果包装器不存在，需要创建
            if use_cudagraph:  # CUDA图模式
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]  # 截取GPU上的索引指针
                paged_kv_indices = self.paged_kv_indices.gpu  # GPU上的页索引
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]  # 截取GPU上的最后页长度
            else:  # 非CUDA图模式
                paged_kv_indptr = None  # 不预分配索引指针
                paged_kv_indices = None  # 不预分配页索引
                paged_kv_last_page_len = None  # 不预分配最后页长度
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(  # 创建批量解码包装器
                self._get_workspace_buffer(),  # 工作空间缓冲区
                get_kv_cache_layout(),  # KV缓存布局
                use_cuda_graph=use_cudagraph,  # 是否使用CUDA图
                paged_kv_indptr_buffer=paged_kv_indptr,  # 预分配的索引指针缓冲区
                paged_kv_indices_buffer=paged_kv_indices,  # 预分配的页索引缓冲区
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,  # 预分配的最后页长度缓冲区
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,  # 默认启用张量核心以获得最佳性能
            )

            # save the decode wrapper
            if use_cudagraph:  # CUDA图模式
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper  # 保存到CUDA图包装器字典
            else:  # 非CUDA图模式
                self._decode_wrapper = decode_wrapper  # 保存为通用解码包装器

        return decode_wrapper  # 返回解码包装器

    def _get_cascade_wrapper(self):
        """获取或创建级联注意力包装器（2级级联）。"""
        if self._cascade_wrapper is None:  # 如果级联包装器尚未创建
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(  # 创建2级级联注意力包装器
                2, self._get_workspace_buffer(), get_kv_cache_layout()  # 2级，工作空间缓冲区，KV缓存布局
            )
        return self._cascade_wrapper  # 返回级联包装器

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,  # 每个请求的块数（numpy数组）
        seq_lens_np: np.ndarray,  # 每个请求的序列长度（numpy数组）
        block_table_tensor: torch.Tensor,  # 块表张量
        num_reqs: int,  # 请求数量
        page_size: int,  # 页大小
    ) -> torch.Tensor:
        """计算FlashInfer注意力所需的分页KV元数据：索引指针、页索引和最后页长度。返回GPU上的页索引张量。"""
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(  # 计算块数的累积和，生成索引指针
            num_blocks_np,  # 每个请求的块数
            dtype=np.int32,  # 输出类型为int32
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],  # 原地写入索引指针（跳过第0个位置，因为始终为0）
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_reqs + 1] = self.paged_kv_indptr.cpu[  # 拷贝到额外缓冲区避免竞态条件
            : num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]  # 获取GPU上的索引指针切片
        paged_kv_indptr.copy_(  # 异步拷贝CPU缓冲区到GPU
            self.paged_kv_indptr_cpu_buffer[: num_reqs + 1], non_blocking=True  # 非阻塞拷贝
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]  # 获取实际总页数
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]  # 获取GPU上的页索引切片
        _copy_page_indices_kernel[(num_reqs,)](  # 启动Triton内核拷贝页索引
            paged_kv_indices,  # 输出页索引
            block_table_tensor,  # 块表张量
            block_table_tensor.stride(0),  # 块表第0维步长
            paged_kv_indptr,  # 索引指针
            BLOCK_SIZE=1024,  # Triton块大小
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size  # 计算最后一页的有效长度（序列长度对页大小取模）
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(  # 处理整除情况：如果余数为0且序列非空，则最后页满
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),  # 条件：余数为0且序列长度不为0
            page_size,  # 为真时设为页大小（满页）
            paged_kv_last_page_len_np,  # 为假时使用计算的余数
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(  # 异步拷贝最后页长度到GPU
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True  # 非阻塞拷贝
        )
        return paged_kv_indices  # 返回GPU上的页索引

    def build(
        self,
        common_prefix_len: int,  # 公共前缀长度（用于级联注意力）
        common_attn_metadata: CommonAttentionMetadata,  # 通用注意力元数据
        fast_build: bool = False,  # 是否快速构建
    ) -> FlashInferMetadata:
        """构建FlashInfer注意力元数据，根据批次内容选择预填充和解码的执行路径（FI原生或TRTLLM）。"""
        num_reqs = common_attn_metadata.num_reqs  # 请求总数
        num_actual_tokens = common_attn_metadata.num_actual_tokens  # 实际token总数
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (  # 分离解码和预填充请求
            split_decodes_and_prefills(
                common_attn_metadata,  # 通用注意力元数据
                decode_threshold=self.reorder_batch_threshold,  # 解码阈值
                require_uniform=True,  # 要求统一
            )
        )

        page_size = self.page_size  # 页大小
        max_seq_len = common_attn_metadata.max_seq_len  # 最大序列长度
        seq_lens = common_attn_metadata.seq_lens  # 序列长度张量
        block_table_tensor = common_attn_metadata.block_table_tensor  # 块表张量
        qo_indptr = common_attn_metadata.query_start_loc  # 查询起始位置索引指针
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu  # 查询起始位置索引指针（CPU版）

        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native or TRTLLM)
        use_cascade = common_prefix_len > 0  # 公共前缀长度>0时使用级联注意力
        uses_spec_reorder = self.reorder_batch_threshold > 1  # 判断是否使用推测重排序
        prefill_use_trtllm = use_trtllm_attention(  # 判断预填充是否使用TRTLLM
            self.num_qo_heads,  # 查询头数
            self.num_kv_heads,  # KV头数
            num_prefill_tokens,  # 预填充token数
            max_seq_len,  # 最大序列长度
            self.dcp_world_size,  # DCP组大小
            self.cache_dtype,  # 缓存数据类型
            self.q_data_type,  # 查询数据类型
            is_prefill=True,  # 标记为预填充
            force_use_trtllm=self.attention_config.use_trtllm_attention,  # 强制使用TRTLLM配置
            has_sinks=self.has_sinks,  # 是否有注意力下沉
            has_spec=uses_spec_reorder,  # 是否使用推测
        )
        decode_use_trtllm = (  # 判断解码是否使用TRTLLM
            self.use_trtllm_decode_attention and self.dcp_world_size <= 1  # 支持TRTLLM且DCP组大小<=1
        )

        all_uses_trtllm = (num_prefills == 0 or prefill_use_trtllm) and (  # 判断是否所有路径都使用TRTLLM
            num_decodes == 0 or decode_use_trtllm  # 无解码或解码使用TRTLLM
        )
        is_only_trtllm_decode = num_prefills == 0 and (  # 判断是否仅有TRTLLM解码
            num_decodes > 0 and decode_use_trtllm  # 无预填充，有解码且使用TRTLLM
        )

        if not all_uses_trtllm:  # 如果不是所有路径都使用TRTLLM
            if self.has_sinks:  # 如果有注意力下沉
                raise NotImplementedError(  # 抛出未实现错误
                    "FlashInfer backend currently does not support attention "
                    "sinks, please use trtllm on blackwell or flash attention "
                    "on earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:  # 如果各层的滑动窗口不一致
                raise ValueError(  # 抛出值错误
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (  # 断言所有层的超参数一致
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

            # The q quantization is not supported for non-trtllm attention,
            # fall back to model dtype.
            self.q_data_type = self.model_config.dtype  # 非TRTLLM不支持查询量化，回退到模型数据类型

        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(  # 创建FlashInfer元数据对象
            num_actual_tokens=num_actual_tokens,  # 实际token数
            slot_mapping=common_attn_metadata.slot_mapping,  # 槽位映射
            q_data_type=self.q_data_type,  # 查询数据类型
            num_decodes=num_decodes,  # 解码请求数
            num_decode_tokens=num_decode_tokens,  # 解码token数
            num_prefills=num_prefills,  # 预填充请求数
            num_prefill_tokens=num_prefill_tokens,  # 预填充token数
            use_cascade=use_cascade,  # 是否使用级联注意力
            prefill=None,  # 预填充元数据（稍后填充）
            decode=None,  # 解码元数据（稍后填充）
            cascade_wrapper=None,  # 级联包装器（稍后填充）
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        needs_seq_lens_cpu = self.use_dcp or use_cascade or not is_only_trtllm_decode  # 判断是否需要CPU上的序列长度
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None  # 按需获取CPU序列长度
        seq_lens_np = seq_lens_cpu.numpy() if seq_lens_cpu is not None else None  # 转换为numpy数组
        num_blocks_np = (  # 计算每个请求的块数
            (seq_lens_np + (page_size - 1)) // page_size  # 向上取整
            if seq_lens_np is not None
            else None  # 如果不需要则为None
        )

        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:  # 如果使用DCP
            assert seq_lens_cpu is not None  # 断言CPU序列长度存在
            if num_prefills > 0:  # 如果有预填充请求
                qo_indptr_prefill_cpu = (  # 计算预填充部分的查询索引指针
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]  # 减去解码部分的偏移
                )
                query_lens_prefill_cpu = (  # 计算预填充的查询长度
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]  # 相邻指针之差
                )
                seq_lens_cpu[num_decodes:] = (  # 调整序列长度：减去查询长度（DCP中只需上下文长度）
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(  # 获取DCP本地的序列长度
                seq_lens_cpu,  # 全局序列长度
                self.dcp_world_size,  # DCP组大小
                self.dcp_rank,  # DCP组内排名
                self.dcp_kv_cache_interleave_size,  # KV缓存交错大小
            )

        # Adjust num_block_np for cascade attention
        if use_cascade:  # 如果使用级联注意力
            assert num_blocks_np is not None  # 断言块数数组存在
            assert common_prefix_len % page_size == 0  # 断言公共前缀长度是页大小的整数倍
            num_common_kv_blocks = common_prefix_len // page_size  # 计算公共KV块数
            num_blocks_np -= num_common_kv_blocks  # 减去公共块数

        # Compute paged_kv_indices if necessary
        needs_paged_kv_indices = use_cascade or not is_only_trtllm_decode  # 判断是否需要计算分页KV索引
        if needs_paged_kv_indices:  # 如果需要
            assert num_blocks_np is not None  # 断言块数数组存在
            assert seq_lens_np is not None  # 断言序列长度数组存在
            paged_kv_indices = self._compute_flashinfer_kv_metadata(  # 计算FlashInfer KV元数据
                num_blocks_np,  # 每个请求的块数
                seq_lens_np,  # 每个请求的序列长度
                block_table_tensor,  # 块表张量
                num_reqs,  # 请求数
                page_size,  # 页大小
            )
        else:  # 不需要计算
            paged_kv_indices = None  # 设为None

        # Early-out for cascade attention
        if use_cascade:  # 如果使用级联注意力
            assert num_blocks_np is not None  # 断言块数数组存在
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size  # 计算公共前缀的KV块数

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(  # 创建共享查询索引指针（CPU）
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"  # 从0到总token数
            )
            shared_kv_page_indptr_cpu = torch.tensor(  # 创建共享KV页索引指针（CPU）
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"  # 从0到公共块数
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]  # 从第一个请求获取共享前缀的页索引
            shared_kv_last_page_len_cpu = torch.tensor(  # 共享部分最后一页长度为满页
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]  # 从块表中移除共享前缀的块
            num_blocks_np -= num_common_kv_blocks  # 更新块数

            assert paged_kv_indices is not None  # 断言页索引已计算
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]  # 获取CPU上的索引指针
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]  # 获取CPU上的最后页长度

            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()  # 获取级联注意力包装器
            attn_metadata.cascade_wrapper.plan(  # 规划级联注意力操作
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],  # 两级查询索引指针数组
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],  # 两级KV索引指针数组
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],  # 两级页索引数组
                paged_kv_last_page_len=[  # 两级最后页长度
                    shared_kv_last_page_len_cpu,  # 共享前缀的最后页长度
                    paged_kv_last_page_len_cpu,  # 各请求独立部分的最后页长度
                ],
                num_qo_heads=self.num_qo_heads,  # 查询头数
                num_kv_heads=self.num_kv_heads,  # KV头数
                head_dim=self.head_dim,  # 头维度
                page_size=self.page_size,  # 页大小
                causal=True,  # 因果注意力
                sm_scale=self.sm_scale,  # softmax缩放因子
                window_left=self.window_left,  # 滑动窗口左边界
                logits_soft_cap=self.logits_soft_cap,  # logits软上限
                q_data_type=self.q_data_type,  # 查询数据类型
                kv_data_type=self.kv_cache_dtype,  # KV数据类型
            )
            return attn_metadata  # 返回级联注意力元数据

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:  # 如果有预填充请求
            # Slices for shared prefill metadata
            prefill_start = num_decodes  # 预填充起始索引（解码请求之后）
            qo_indptr_prefill_cpu = (  # 计算预填充部分的查询索引指针
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]  # 减去起始偏移使其从0开始
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1  # 断言索引指针长度正确

            if prefill_use_trtllm:  # 如果预填充使用TRTLLM
                # Create GPU versions
                qo_indptr_prefill_gpu = (  # 创建GPU上的预填充查询索引指针
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]  # 减去起始偏移
                )
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[  # 获取GPU上的预填充KV索引指针
                    prefill_start : num_reqs + 1
                ]
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (  # 计算每个预填充请求的查询长度
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]  # 相邻指针之差
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())  # 获取最大查询长度
                attn_metadata.prefill = TRTLLMPrefill(  # 创建TRTLLM预填充元数据
                    block_tables=block_table_tensor[prefill_start:],  # 预填充部分的块表
                    seq_lens=seq_lens[prefill_start:],  # 预填充部分的序列长度
                    cum_seq_lens_q=qo_indptr_prefill_gpu,  # 查询的累积序列长度
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,  # KV的累积序列长度
                    max_q_len=max_q_len_prefill,  # 最大查询长度
                    max_seq_len=max_seq_len,  # 最大序列长度
                )
            else:  # 预填充使用FlashInfer原生路径
                prefill_wrapper = self._get_prefill_wrapper()  # 获取预填充包装器
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[  # 获取预填充部分的最后页长度
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills  # 断言长度正确
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[  # 获取预填充部分的KV索引指针
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1  # 断言长度正确
                if self.use_dcp:  # 如果使用DCP
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)  # 断言是DCP包装器类型
                    prefill_wrapper.plan(  # 规划DCP预填充操作
                        qo_indptr_cpu=qo_indptr_prefill_cpu,  # 查询索引指针
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,  # KV索引指针
                        paged_kv_indices=paged_kv_indices,  # 页索引
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,  # 最后页长度
                        page_size=self.page_size,  # 页大小
                        num_qo_heads=self.num_qo_heads,  # 查询头数
                        dcp_world_size=self.dcp_world_size,  # DCP组大小
                        num_kv_heads=self.num_kv_heads,  # KV头数
                        head_dim=self.head_dim,  # 头维度
                        sm_scale=self.sm_scale,  # softmax缩放因子
                        window_left=self.window_left,  # 滑动窗口左边界
                        logits_soft_cap=self.logits_soft_cap,  # logits软上限
                        q_data_type=self.q_data_type,  # 查询数据类型
                        kv_cache_dtype=self.kv_cache_dtype,  # KV缓存数据类型
                        prefill_fixed_split_size=self.prefill_fixed_split_size,  # 预填充固定分割大小
                        disable_split_kv=self.disable_split_kv,  # 是否禁用KV分割
                    )
                else:  # 不使用DCP
                    assert isinstance(  # 断言是标准预填充包装器类型
                        prefill_wrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                    )
                    prefill_wrapper.plan(  # 规划标准预填充操作
                        qo_indptr=qo_indptr_prefill_cpu,  # 查询索引指针
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,  # KV索引指针
                        paged_kv_indices=paged_kv_indices,  # 页索引
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,  # 最后页长度
                        num_qo_heads=self.num_qo_heads,  # 查询头数
                        num_kv_heads=self.num_kv_heads,  # KV头数
                        head_dim_qk=self.head_dim,  # QK头维度
                        page_size=self.page_size,  # 页大小
                        causal=True,  # 因果注意力
                        sm_scale=self.sm_scale,  # softmax缩放因子
                        window_left=self.window_left,  # 滑动窗口左边界
                        logits_soft_cap=self.logits_soft_cap,  # logits软上限
                        q_data_type=self.q_data_type,  # 查询数据类型
                        kv_data_type=self.kv_cache_dtype,  # KV数据类型
                        o_data_type=self.model_config.dtype,  # 输出数据类型
                        fixed_split_size=self.prefill_fixed_split_size,  # 固定分割大小
                        disable_split_kv=self.disable_split_kv,  # 是否禁用KV分割
                    )
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)  # 设置FI原生预填充元数据

        ## DECODE PATHWAY
        if num_decodes > 0:  # 如果有解码请求
            if decode_use_trtllm:  # 如果解码使用TRTLLM
                assert num_decode_tokens % num_decodes == 0, (  # 断言解码token数能被解码请求数整除
                    "TRTLLM decode requires uniform query lengths per request. "
                    f"Got {num_decode_tokens=} and {num_decodes=}."
                )
                attn_metadata.decode = TRTLLMDecode(  # 创建TRTLLM解码元数据
                    block_tables=block_table_tensor[:num_decodes],  # 解码部分的块表
                    seq_lens=seq_lens[:num_decodes],  # 解码部分的序列长度
                    max_seq_len=max_seq_len,  # 最大序列长度
                )
            else:  # 解码使用FlashInfer原生路径
                assert seq_lens_cpu is not None  # 断言CPU序列长度存在
                pure_decode = num_prefills == 0  # 判断是否纯解码批次
                use_cudagraph = (  # 判断是否使用CUDA图
                    self.enable_cuda_graph  # 启用了CUDA图
                    and pure_decode  # 且是纯解码批次
                    and num_decode_tokens <= self._decode_cudagraph_max_bs  # 且token数不超过CUDA图最大批次
                )
                num_input_tokens = num_decode_tokens  # 输入token数等于解码token数

                decode_wrapper = self._get_decode_wrapper(  # 获取解码包装器
                    num_input_tokens, use_cudagraph  # 传入token数和是否使用CUDA图
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                fast_plan_decode(  # 执行快速解码规划
                    decode_wrapper,  # 解码包装器
                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],  # KV索引指针（CPU）
                    indices=paged_kv_indices,  # 页索引
                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[  # 最后页长度（CPU）
                        :num_input_tokens
                    ],
                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,  # 查询头数乘以DCP并行度
                    num_kv_heads=self.num_kv_heads,  # KV头数
                    head_dim=self.head_dim,  # 头维度
                    page_size=self.page_size,  # 页大小
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",  # 禁用FlashInfer位置编码，使用vLLM的RoPE
                    sm_scale=self.sm_scale,  # softmax缩放因子
                    window_left=self.window_left,  # 滑动窗口左边界
                    logits_soft_cap=self.logits_soft_cap,  # logits软上限
                    q_data_type=self.q_data_type,  # 查询数据类型
                    kv_data_type=self.kv_cache_dtype,  # KV数据类型
                    o_data_type=self.model_config.dtype,  # 输出数据类型
                    fixed_split_size=self.decode_fixed_split_size,  # 固定分割大小
                    disable_split_kv=self.disable_split_kv,  # 是否禁用KV分割
                )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)  # 设置FI原生解码元数据
        return attn_metadata  # 返回构建完成的注意力元数据

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        """判断是否使用级联注意力。当KV缓存类型与模型类型不一致时不支持，目前级联注意力已被禁用。"""
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:  # 如果KV缓存类型与模型类型不一致
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False  # 不支持级联注意力
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False  # 级联注意力当前已禁用


class FlashInferImpl(AttentionImpl):
    """FlashInfer注意力实现类，实现基于FlashInfer的注意力前向计算，支持预填充、解码、级联注意力、DCP分布式并行等多种模式。"""
    can_return_lse_for_decode: bool = True  # 解码时可以返回log-sum-exp值

    def __init__(
        self,
        num_heads: int,  # 注意力头数量
        head_size: int,  # 注意力头维度
        scale: float,  # 注意力缩放因子
        num_kv_heads: int,  # KV注意力头数量
        alibi_slopes: list[float] | None,  # ALiBi位置编码斜率
        sliding_window: int | None,  # 滑动窗口大小
        kv_cache_dtype: str,  # KV缓存数据类型字符串
        logits_soft_cap: float | None = None,  # logits软上限
        attn_type: AttentionType = AttentionType.DECODER,  # 注意力类型（默认解码器）
        kv_sharing_target_layer_name: int | None = None,  # KV共享目标层名称
        sinks: torch.Tensor | None = None,  # 注意力下沉张量
    ) -> None:
        """初始化FlashInfer注意力实现，配置头数、维度、缩放因子、滑动窗口、KV缓存类型等参数。"""
        self.num_heads = num_heads  # 注意力头数量
        self.head_size = head_size  # 注意力头维度
        self.scale = float(scale)  # 注意力缩放因子
        self.num_kv_heads = num_kv_heads  # KV注意力头数量
        if alibi_slopes is not None:  # 如果提供了ALiBi斜率
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)  # 转换为float32张量
        self.alibi_slopes = alibi_slopes  # 保存ALiBi斜率
        if sliding_window is None:  # 如果没有滑动窗口
            self.sliding_window = (-1, -1)  # 设为(-1, -1)表示无窗口
        else:  # 有滑动窗口
            self.sliding_window = (sliding_window - 1, 0)  # 设置窗口范围
        self.window_left = (  # 提取滑动窗口左边界
            self.sliding_window[0] if self.sliding_window is not None else -1  # 有窗口取第一个值，否则为-1
        )
        self.kv_cache_dtype = kv_cache_dtype  # KV缓存数据类型字符串
        self.logits_soft_cap = logits_soft_cap  # logits软上限
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name  # KV共享目标层名称

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads  # 每个KV头对应的查询头数（GQA比率）

        if attn_type != AttentionType.DECODER:  # 如果不是解码器注意力类型
            raise NotImplementedError(  # 抛出未实现错误
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        self.sinks: torch.Tensor | None = None  # 初始化注意力下沉张量
        if sinks is not None:  # 如果提供了注意力下沉
            if sinks.shape[0] != num_heads:  # 如果下沉张量的头数与注意力头数不匹配
                raise ValueError(  # 抛出值错误
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks  # 保存注意力下沉张量

        self.support_trtllm_attn = can_use_trtllm_attention(num_heads, num_kv_heads)  # 检查是否支持TRTLLM注意力
        vllm_config = get_current_vllm_config_or_none()  # 获取当前vLLM配置
        self.supports_quant_query_input = (  # 判断是否支持量化查询输入
            self.support_trtllm_attn  # 支持TRTLLM注意力
            and vllm_config is not None  # 且配置存在
            and not vllm_config.attention_config.disable_flashinfer_q_quantization  # 且未禁用查询量化
        )
        self.bmm1_scale: float | None = None  # BMM1（Q*K）的缩放因子（延迟初始化）
        self.bmm2_scale: float | None = None  # BMM2（Attn*V）的缩放因子（延迟初始化）
        self.o_sf_scale: float | None = None  # 输出缩放因子（延迟初始化）

        dcp_a2a = (  # 判断是否使用DCP全对全通信
            vllm_config is not None  # 配置存在
            and vllm_config.parallel_config.decode_context_parallel_size > 1  # DCP并行度>1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"  # 通信后端为a2a
        )
        if dcp_a2a:  # 如果使用全对全通信
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)  # 设置DCP合并函数为全对全LSE归约
        else:  # 否则使用聚合-归约通信
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)  # 设置DCP合并函数为聚合输出归约

    def fused_output_quant_supported(self, quant_key: QuantKey):
        """检查是否支持注意力输出与量化的融合。仅在TRTLLM注意力+FP8 KV缓存+特定量化类型时支持。"""
        return (  # 返回是否支持融合输出量化
            self.support_trtllm_attn  # 支持TRTLLM注意力
            and self.kv_cache_dtype.startswith("fp8")  # KV缓存是FP8类型
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)  # 量化类型为FP8静态或NVFP4动态
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """加载权重后处理：将注意力下沉张量转换为float32（FlashInfer要求）。"""
        if self.sinks is not None and self.sinks.dtype != torch.float32:  # 如果下沉张量存在且不是float32
            self.sinks = self.sinks.to(torch.float32)  # 转换为float32

    def forward(
        self,
        layer: torch.nn.Module,  # 注意力层模块
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        value: torch.Tensor,  # 值张量
        kv_cache: torch.Tensor,  # KV缓存张量
        attn_metadata: FlashInferMetadata,  # FlashInfer注意力元数据
        output: torch.Tensor | None = None,  # 输出张量
        output_scale: torch.Tensor | None = None,  # 输出缩放因子（用于注意力+量化融合）
        output_block_scale: torch.Tensor | None = None,  # 输出块缩放因子（用于NVFP4）
    ) -> torch.Tensor:
        """FlashInfer注意力前向传播，根据元数据选择级联注意力、预填充或解码路径执行计算。"""
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor with different possible shapes:
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."  # 断言输出张量必须提供

        if attn_metadata is None:  # 如果元数据为空（性能分析运行）
            # Profiling run.
            return output.fill_(0)  # 返回全零输出

        # Ensure query dtype matches the expected dtype from attention metadata
        assert attn_metadata.q_data_type == query.dtype, (  # 断言查询数据类型与元数据中预期的一致
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        if self.bmm1_scale is None:  # 如果BMM1缩放因子尚未初始化
            self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale  # 计算BMM1缩放因子（Q缩放*K缩放*注意力缩放）

        if self.bmm2_scale is None:  # 如果BMM2缩放因子尚未初始化
            self.bmm2_scale = layer._v_scale_float  # BMM2缩放因子为V缩放

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)  # 判断预填充是否使用TRTLLM
        decode_use_trtllm = isinstance(attn_metadata.decode, TRTLLMDecode)  # 判断解码是否使用TRTLLM

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:  # 如果没有提供输出缩放因子
            assert output_block_scale is None, (  # 断言也不应有块缩放因子
                "output_block_scale is not supported when fusion has not happened"
            )
        else:  # 提供了输出缩放因子（注意力+量化融合模式）
            assert attn_metadata.q_data_type == FP8_DTYPE, (  # 断言查询必须是FP8类型
                "Query must be FP8 when attn+quant fusion happened."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (  # 断言必须使用TRTLLM注意力
                attn_metadata.num_decodes == 0 or decode_use_trtllm
            ), "Must use TRT-LLM attn"

            if output.dtype == FP8_DTYPE:  # 如果输出是FP8类型
                assert output_block_scale is None, (  # FP8输出不需要块缩放因子
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:  # 如果输出是FP4类型
                assert output_block_scale is not None, (  # FP4输出需要块缩放因子
                    "output_block_scale is required for nvfp4 output"
                )
            else:  # 不支持的输出类型
                raise ValueError(f"Unsupported output dtype: {output.dtype}")  # 抛出错误

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:  # 如果输出缩放浮点值尚未缓存
                layer._o_scale_float = output_scale.cpu().item()  # 将GPU上的缩放因子拷贝到CPU并转为标量
                if output.dtype == FP8_DTYPE:  # 如果输出是FP8
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float  # 将输出缩放融合到BMM2缩放中
                elif output.dtype == FP4_DTYPE:  # 如果输出是FP4
                    self.o_sf_scale = layer._o_scale_float  # 保存输出缩放因子

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens  # 获取实际token数量

        # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
        # to process the cache when the kv_cache_dtype is fp8
        if self.kv_sharing_target_layer_name is None and self.kv_cache_dtype.startswith(  # 如果不共享KV且缓存是FP8
            "fp8"
        ):
            torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(  # 获取FlashInfer对应的FP8类型
                self.kv_cache_dtype
            )
            kv_cache = kv_cache.view(torch_dtype)  # 将KV缓存视图转换为FP8类型

        # Inputs and outputs may be padded for CUDA graphs
        query = query[:num_actual_tokens]  # 截取实际的查询（去除CUDA图填充）
        key = key[:num_actual_tokens]  # 截取实际的键
        value = value[:num_actual_tokens]  # 截取实际的值
        output_padded = output  # 保存带填充的输出引用
        output = output[:num_actual_tokens]  # 截取实际的输出

        if attn_metadata.use_cascade:  # 如果使用级联注意力
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None  # 断言级联包装器存在
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))  # 执行级联注意力并拷贝结果
            return output  # 返回输出

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        num_decode_tokens = attn_metadata.num_decode_tokens  # 获取解码token数
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # 获取预填充token数

        stride_order = FlashInferBackend.get_kv_cache_stride_order()  # 获取KV缓存步长顺序
        kv_cache_permute = kv_cache.permute(*stride_order)  # 按步长顺序重排KV缓存维度

        use_dcp = self.dcp_world_size > 1  # 判断是否使用DCP

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:  # 如果有预填充token
            prefill_query = query[num_decode_tokens:]  # 获取预填充查询（在解码token之后）
            assert prefill_query.shape[0] == num_prefill_tokens  # 断言预填充查询数量正确

            if not prefill_use_trtllm:  # 如果预填充不使用TRTLLM
                assert isinstance(attn_metadata.prefill, FIPrefill)  # 断言是FI原生预填充元数据
                prefill_wrapper = attn_metadata.prefill.wrapper  # 获取预填充包装器
                assert prefill_wrapper is not None  # 断言包装器存在
                if use_dcp:  # 如果使用DCP
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)  # 断言是DCP包装器
                    assert prefill_wrapper._context._window_left == self.window_left  # 断言上下文窗口左边界一致
                    assert prefill_wrapper._context._logits_soft_cap == (  # 断言上下文logits软上限一致
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._context._sm_scale == self.scale  # 断言上下文softmax缩放一致
                    assert not prefill_wrapper._context._causal  # 断言上下文运行非因果
                    assert prefill_wrapper._new_tokens._window_left == self.window_left  # 断言新token窗口左边界一致
                    assert prefill_wrapper._new_tokens._logits_soft_cap == (  # 断言新token logits软上限一致
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._new_tokens._sm_scale == self.scale  # 断言新token softmax缩放一致
                    assert prefill_wrapper._new_tokens._causal  # 断言新token运行因果

                    prefill_wrapper.run(  # 执行DCP预填充注意力
                        layer,  # 注意力层
                        prefill_query,  # 预填充查询
                        kv_cache_permute,  # 重排后的KV缓存
                        key[num_decode_tokens:],  # 预填充部分的键
                        value[num_decode_tokens:],  # 预填充部分的值
                        out=output[num_decode_tokens:],  # 预填充部分的输出
                    )
                else:  # 不使用DCP
                    assert isinstance(  # 断言是标准预填充包装器
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left  # 断言窗口左边界一致
                    assert prefill_wrapper._logits_soft_cap == (  # 断言logits软上限一致
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale  # 断言softmax缩放一致
                    assert prefill_wrapper._causal  # 断言因果注意力
                    prefill_wrapper.run(  # 执行标准预填充注意力
                        prefill_query,  # 预填充查询
                        kv_cache_permute,  # 重排后的KV缓存
                        k_scale=layer._k_scale_float,  # K缩放因子
                        v_scale=layer._v_scale_float,  # V缩放因子
                        out=output[num_decode_tokens:],  # 预填充部分的输出
                    )
            else:  # 预填充使用TRTLLM
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)  # 断言是TRTLLM预填充元数据
                # prefill_query may be non-contiguous or have degenerate strides
                # First ensure memory contiguity, then fix degenerate strides
                # with reshape. contiguous() alone doesn't fix degenerate
                # strides when a dimension has size 1.
                prefill_query = prefill_query.contiguous().reshape(prefill_query.shape)  # 确保查询连续并修复退化步长
                workspace_buffer = _get_trtllm_gen_workspace_buffer()  # 获取TRTLLM工作空间缓冲区
                block_tables_prefill = attn_metadata.prefill.block_tables  # 获取预填充块表
                seq_lens_prefill = attn_metadata.prefill.seq_lens  # 获取预填充序列长度

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"  # 断言KV缓存布局为HND
                assert is_strictly_contiguous(prefill_query)  # 断言查询严格连续
                assert is_strictly_contiguous(kv_cache_permute)  # 断言KV缓存严格连续
                assert is_strictly_contiguous(workspace_buffer)  # 断言工作空间严格连续
                assert is_strictly_contiguous(block_tables_prefill)  # 断言块表严格连续
                assert is_strictly_contiguous(seq_lens_prefill)  # 断言序列长度严格连续

                if output.dtype == FP4_DTYPE:  # 如果输出是FP4类型
                    assert self.o_sf_scale is not None  # 断言输出缩放因子存在
                    out = FP4Tensor(  # 创建FP4张量包装
                        data=output[num_decode_tokens:],  # 预填充部分的输出数据
                        scale=output_block_scale,  # 块缩放因子
                        scale_start_index=num_decode_tokens,  # 缩放起始索引
                        original_shape=prefill_query.shape,  # 原始形状
                    )
                else:  # 非FP4输出
                    assert self.o_sf_scale is None  # 断言不需要输出缩放因子
                    out = output[num_decode_tokens:]  # 预填充部分的输出

                if (  # 如果查询不是FP8但KV缓存是FP8
                    attn_metadata.q_data_type != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(  # 反量化FP8 KV缓存
                        kv_cache_permute,  # 重排后的KV缓存
                        block_tables_prefill,  # 预填充块表
                        layer._k_scale,  # K缩放因子张量
                        layer._v_scale,  # V缩放因子张量
                        attn_metadata.q_data_type,  # 查询数据类型作为反量化目标类型
                    )
                else:  # 不需要反量化
                    mock_kv_cache = kv_cache_permute  # 直接使用原始KV缓存
                    mock_block_table = block_tables_prefill  # 直接使用原始块表

                trtllm_batch_context_with_kv_cache(  # 调用TRTLLM批量上下文预填充内核
                    query=prefill_query,  # 预填充查询
                    kv_cache=mock_kv_cache,  # KV缓存（可能是反量化后的模拟缓存）
                    workspace_buffer=workspace_buffer,  # 工作空间缓冲区
                    block_tables=mock_block_table,  # 块表（可能是模拟块表）
                    seq_lens=seq_lens_prefill,  # 序列长度
                    max_q_len=attn_metadata.prefill.max_q_len,  # 最大查询长度
                    max_kv_len=attn_metadata.prefill.max_seq_len,  # 最大KV长度
                    bmm1_scale=self.bmm1_scale,  # BMM1缩放因子
                    bmm2_scale=self.bmm2_scale,  # BMM2缩放因子
                    batch_size=attn_metadata.num_prefills,  # 预填充批次大小
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,  # 查询累积序列长度
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,  # KV累积序列长度
                    window_left=self.window_left,  # 滑动窗口左边界
                    sinks=self.sinks,  # 注意力下沉
                    o_sf_scale=self.o_sf_scale,  # 输出缩放因子
                    out=out,  # 输出张量
                )

        if num_decode_tokens > 0:  # 如果有解码token
            decode_query = query[:num_decode_tokens]  # 获取解码查询（在前面）
            assert decode_query.shape[0] == num_decode_tokens  # 断言解码查询数量正确

            if not decode_use_trtllm:  # 如果解码不使用TRTLLM
                assert isinstance(attn_metadata.decode, FIDecode)  # 断言是FI原生解码元数据
                decode_wrapper = attn_metadata.decode.wrapper  # 获取解码包装器
                assert decode_wrapper is not None  # 断言包装器存在
                assert decode_wrapper._window_left == self.window_left  # 断言窗口左边界一致
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)  # 断言logits软上限一致
                assert decode_wrapper._sm_scale == self.scale  # 断言softmax缩放一致

                if use_dcp:  # 如果使用DCP
                    decode_query = get_dcp_group().all_gather(  # 跨DCP组全收集解码查询
                        decode_query.contiguous(), dim=-2  # 在倒数第二维（头维度）上聚合
                    )
                    output_tmp = torch.empty_like(decode_query)  # 创建临时输出张量
                    lse = torch.empty(  # 创建LSE张量
                        (decode_query.size(0), decode_query.size(1)),  # 形状为(token数, 头数)
                        dtype=torch.float32,  # float32类型
                        device=decode_query.device,  # 与查询相同设备
                    )
                    decode_wrapper.run(  # 执行解码注意力
                        decode_query,  # 聚合后的解码查询
                        kv_cache_permute,  # 重排后的KV缓存
                        k_scale=layer._k_scale_float,  # K缩放因子
                        v_scale=layer._v_scale_float,  # V缩放因子
                        out=output_tmp,  # 临时输出
                        lse=lse,  # LSE输出
                        return_lse=True,  # 返回LSE
                    )
                    output[:num_decode_tokens] = self.dcp_combine(  # DCP合并解码结果
                        output_tmp,  # 临时输出
                        lse,  # LSE
                        get_dcp_group(),  # DCP通信组
                    )
                else:  # 不使用DCP
                    decode_wrapper.run(  # 执行解码注意力
                        decode_query,  # 解码查询
                        kv_cache_permute,  # 重排后的KV缓存
                        k_scale=layer._k_scale_float,  # K缩放因子
                        v_scale=layer._v_scale_float,  # V缩放因子
                        out=output[:num_decode_tokens],  # 解码部分的输出
                    )
            else:  # 解码使用TRTLLM
                # decode_query may be non-contiguous or have degenerate strides
                assert isinstance(attn_metadata.decode, TRTLLMDecode)  # 断言是TRTLLM解码元数据
                # First ensure memory contiguity, then fix degenerate strides
                # with reshape. contiguous() alone doesn't fix degenerate
                # strides when a dimension has size 1.
                decode_query = decode_query.contiguous().reshape(decode_query.shape)  # 确保查询连续并修复退化步长
                workspace_buffer = _get_trtllm_gen_workspace_buffer()  # 获取TRTLLM工作空间缓冲区
                block_tables_decode = attn_metadata.decode.block_tables  # 获取解码块表
                seq_lens_decode = attn_metadata.decode.seq_lens  # 获取解码序列长度

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"  # 断言KV缓存布局为HND
                assert is_strictly_contiguous(decode_query)  # 断言查询严格连续
                assert is_strictly_contiguous(kv_cache_permute)  # 断言KV缓存严格连续
                assert is_strictly_contiguous(workspace_buffer)  # 断言工作空间严格连续
                assert is_strictly_contiguous(block_tables_decode)  # 断言块表严格连续
                assert is_strictly_contiguous(seq_lens_decode)  # 断言序列长度严格连续

                if output.dtype == FP4_DTYPE:  # 如果输出是FP4类型
                    assert self.o_sf_scale is not None  # 断言输出缩放因子存在
                    out = FP4Tensor(  # 创建FP4张量包装
                        data=output[:num_decode_tokens],  # 解码部分的输出数据
                        scale=output_block_scale,  # 块缩放因子
                        scale_start_index=0,  # 缩放起始索引为0
                        original_shape=decode_query.shape,  # 原始形状
                    )
                else:  # 非FP4输出
                    assert self.o_sf_scale is None  # 断言不需要输出缩放因子
                    out = output[:num_decode_tokens]  # 解码部分的输出

                if num_decode_tokens % attn_metadata.num_decodes != 0:  # 如果token数不能被请求数整除
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1  # 每个请求的查询长度设为1（dummy运行时的回退）
                else:  # 能整除
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes  # 计算每个请求的查询长度

                trtllm_batch_decode_with_kv_cache(  # 调用TRTLLM批量解码内核
                    query=decode_query,  # 解码查询
                    kv_cache=kv_cache_permute,  # 重排后的KV缓存
                    workspace_buffer=workspace_buffer,  # 工作空间缓冲区
                    block_tables=block_tables_decode,  # 解码块表
                    seq_lens=seq_lens_decode,  # 解码序列长度
                    max_seq_len=attn_metadata.decode.max_seq_len,  # 最大序列长度
                    bmm1_scale=self.bmm1_scale,  # BMM1缩放因子
                    bmm2_scale=self.bmm2_scale,  # BMM2缩放因子
                    window_left=self.window_left,  # 滑动窗口左边界
                    sinks=self.sinks,  # 注意力下沉
                    o_sf_scale=self.o_sf_scale,  # 输出缩放因子
                    out=out,  # 输出张量
                    q_len_per_req=q_len_per_req,  # 每个请求的查询长度
                )
        return output_padded  # 返回带填充的输出张量

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,  # 注意力层模块
        key: torch.Tensor,  # 键张量
        value: torch.Tensor,  # 值张量
        kv_cache: torch.Tensor,  # KV缓存张量
        slot_mapping: torch.Tensor,  # 槽位映射张量
    ) -> None:
        """执行KV缓存更新，将新的键值对写入KV缓存的对应槽位。如果KV共享则跳过更新。"""
        if self.kv_sharing_target_layer_name is None:  # 如果没有KV共享目标层
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            torch.ops._C_cache_ops.reshape_and_cache_flash(  # 调用自定义算子将KV写入缓存
                key,  # 键张量（可能带填充）
                value,  # 值张量（可能带填充）
                kv_cache[:, 0],  # K缓存
                kv_cache[:, 1],  # V缓存
                slot_mapping,  # 槽位映射（决定写入位置）
                self.kv_cache_dtype,  # KV缓存数据类型
                layer._k_scale,  # K缩放因子
                layer._v_scale,  # V缩放因子
            )


def fast_plan_decode(
    self,  # decode wrapper  # 解码包装器实例
    indptr_cpu: torch.Tensor,  # 索引指针（CPU张量）
    indices: torch.Tensor,  # 页索引张量
    last_page_len_cpu: torch.Tensor,  # 最后页长度（CPU张量）
    num_qo_heads: int,  # 查询/输出头数
    num_kv_heads: int,  # KV头数
    head_dim: int,  # 头维度
    page_size: int,  # 页大小
    pos_encoding_mode: str = "NONE",  # 位置编码模式
    window_left: int = -1,  # 滑动窗口左边界
    logits_soft_cap: float | None = None,  # logits软上限
    q_data_type: str | torch.dtype | None = "float16",  # 查询数据类型
    kv_data_type: str | torch.dtype | None = None,  # KV数据类型
    o_data_type: str | torch.dtype | None = None,  # 输出数据类型
    data_type: str | torch.dtype | None = None,  # 通用数据类型
    sm_scale: float | None = None,  # softmax缩放因子
    rope_scale: float | None = None,  # RoPE缩放因子
    rope_theta: float | None = None,  # RoPE theta参数
    non_blocking: bool = True,  # 是否非阻塞传输
    fixed_split_size: int = -1,  # 固定分割大小
    disable_split_kv: bool = False,  # 是否禁用KV分割
) -> None:
    """快速解码规划函数，用于CUDA图捕获/回放时避免不必要的设备间数据拷贝。首次调用使用完整plan初始化，后续使用fast_decode_plan仅做必要的H2D拷贝。"""
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):  # 非CUDA图模式或首次调用
        self.plan(  # 使用完整的plan函数
            indptr=indptr_cpu,  # 索引指针
            indices=indices,  # 页索引
            last_page_len=last_page_len_cpu,  # 最后页长度
            num_qo_heads=num_qo_heads,  # 查询头数
            num_kv_heads=num_kv_heads,  # KV头数
            head_dim=head_dim,  # 头维度
            page_size=page_size,  # 页大小
            pos_encoding_mode=pos_encoding_mode,  # 位置编码模式
            window_left=window_left,  # 滑动窗口左边界
            logits_soft_cap=logits_soft_cap,  # logits软上限
            q_data_type=q_data_type,  # 查询数据类型
            kv_data_type=kv_data_type,  # KV数据类型
            o_data_type=o_data_type,  # 输出数据类型
            data_type=data_type,  # 通用数据类型
            sm_scale=sm_scale,  # softmax缩放因子
            rope_scale=rope_scale,  # RoPE缩放因子
            rope_theta=rope_theta,  # RoPE theta
            non_blocking=non_blocking,  # 非阻塞传输
            block_tables=None,  # 块表（使用已有缓冲区）
            seq_lens=None,  # 序列长度（使用已有缓冲区）
            fixed_split_size=fixed_split_size,  # 固定分割大小
            disable_split_kv=disable_split_kv,  # 是否禁用KV分割
        )
        self.vllm_first_call = False  # 标记首次调用已完成
        return  # 返回

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"  # 断言此处只应在CUDA图模式下执行

    fast_decode_plan(  # 调用FlashInfer的快速解码规划（仅做必要的H2D拷贝）
        self,  # 解码包装器
        indptr=indptr_cpu,  # 索引指针
        indices=indices,  # 页索引
        last_page_len=last_page_len_cpu,  # 最后页长度
        num_qo_heads=num_qo_heads,  # 查询头数
        num_kv_heads=num_kv_heads,  # KV头数
        head_dim=head_dim,  # 头维度
        page_size=page_size,  # 页大小
        pos_encoding_mode=pos_encoding_mode,  # 位置编码模式
        window_left=window_left,  # 滑动窗口左边界
        logits_soft_cap=logits_soft_cap,  # logits软上限
        q_data_type=q_data_type,  # 查询数据类型
        kv_data_type=kv_data_type,  # KV数据类型
        data_type=data_type,  # 通用数据类型
        sm_scale=sm_scale,  # softmax缩放因子
        rope_scale=rope_scale,  # RoPE缩放因子
        rope_theta=rope_theta,  # RoPE theta
        non_blocking=non_blocking,  # 非阻塞传输
        fixed_split_size=fixed_split_size,  # 固定分割大小
        disable_split_kv=disable_split_kv,  # 是否禁用KV分割
    )


@triton.jit  # Triton即时编译装饰器
def _copy_page_indices_kernel(
    page_indices,  # 输出页索引张量
    block_table,  # 块表张量
    block_table_stride,  # 块表步长
    cu_num_blocks,  # 累积块数指针
    BLOCK_SIZE: tl.constexpr,  # Triton块大小（编译时常量）
):
    """Triton内核：从块表中拷贝页索引到连续的页索引数组。每个线程块处理一个请求的所有页。"""
    req_idx = tl.program_id(0)  # 获取当前请求索引
    row_ptr = block_table + req_idx * block_table_stride  # 计算当前请求在块表中的起始地址
    start_idx = tl.load(cu_num_blocks + req_idx)  # 加载当前请求的累积起始索引
    end_idx = tl.load(cu_num_blocks + req_idx + 1)  # 加载下一个请求的累积起始索引
    num_blocks = end_idx - start_idx  # 计算当前请求的块数

    offset = tl.arange(0, BLOCK_SIZE)  # 生成0到BLOCK_SIZE的偏移数组
    for i in tl.range(0, num_blocks, BLOCK_SIZE):  # 按BLOCK_SIZE步长遍历所有块
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)  # 从块表加载块ID（带掩码防越界）
        tl.store(  # 存储块ID到输出页索引数组
            page_indices + start_idx + i + offset,  # 目标地址
            block_ids,  # 块ID值
            mask=i + offset < num_blocks,  # 掩码防止越界写入
        )
