# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings."""
# 旋转位置编码模块

from typing import Any  # 导入Any类型，用于类型注解

import torch  # 导入PyTorch深度学习框架

from .base import RotaryEmbedding  # 导入基础旋转位置编码类
from .deepseek_scaling_rope import DeepseekScalingRotaryEmbedding  # 导入DeepSeek缩放旋转编码
from .dual_chunk_rope import DualChunkRotaryEmbedding  # 导入双块旋转编码
from .dynamic_ntk_alpha_rope import DynamicNTKAlphaRotaryEmbedding  # 导入动态NTK Alpha旋转编码
from .dynamic_ntk_scaling_rope import DynamicNTKScalingRotaryEmbedding  # 导入动态NTK缩放旋转编码
from .fope import FourierRotaryEmbedding  # 导入傅里叶旋转位置编码
from .linear_scaling_rope import LinearScalingRotaryEmbedding  # 导入线性缩放旋转编码
from .llama3_rope import Llama3RotaryEmbedding  # 导入Llama3旋转编码
from .llama4_vision_rope import Llama4VisionRotaryEmbedding  # 导入Llama4视觉旋转编码
from .mrope import MRotaryEmbedding  # 导入多模态旋转编码
from .mrope_interleaved import MRotaryEmbeddingInterleaved  # 导入交错多模态旋转编码
from .ntk_scaling_rope import NTKScalingRotaryEmbedding  # 导入NTK缩放旋转编码
from .phi3_long_rope_scaled_rope import Phi3LongRoPEScaledRotaryEmbedding  # 导入Phi3长距离缩放旋转编码
from .xdrope import XDRotaryEmbedding  # 导入XD多维旋转编码
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding  # 导入YaRN缩放旋转编码

_ROPE_DICT: dict[tuple[Any, ...], RotaryEmbedding] = {}  # 全局缓存字典，避免重复创建旋转编码实例


def get_rope(
    head_size: int,  # 注意力头的维度大小
    max_position: int,  # 最大位置编码长度
    is_neox_style: bool = True,  # 是否使用NeoX风格的旋转编码
    rope_parameters: dict[str, Any] | None = None,  # 旋转编码参数字典
    dtype: torch.dtype | None = None,  # 数据类型
    dual_chunk_attention_config: dict[str, Any] | None = None,  # 双块注意力配置
) -> RotaryEmbedding:
    """获取旋转位置编码实例，带有缓存机制以避免重复创建。"""
    if dtype is None:  # 如果未指定数据类型
        dtype = torch.get_default_dtype()  # 使用默认数据类型
    if rope_parameters is not None:  # 如果提供了旋转编码参数
        # Transforms every value that is a list into a tuple for caching calls
        rope_parameters_tuple = {  # 将列表值转换为元组以用于缓存键
            k: tuple(v) if isinstance(v, list) else v  # 列表转元组，其他类型保持不变
            for k, v in rope_parameters.items()  # 遍历参数字典
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())  # 转换为可哈希的元组
    else:
        rope_parameters_args = None  # 无参数时设为None

    if dual_chunk_attention_config is not None:  # 如果提供了双块注意力配置
        dual_chunk_attention_tuple = {  # 将列表值转换为元组
            k: tuple(v) if isinstance(v, list) else v  # 列表转元组
            for k, v in dual_chunk_attention_config.items()  # 遍历配置字典
            if k != "sparse_attention_config"  # 排除稀疏注意力配置
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())  # 转换为可哈希元组
    else:
        dual_chunk_attention_args = None  # 无配置时设为None

    rope_parameters = rope_parameters or {}  # 如果为None则使用空字典
    base = rope_parameters.get("rope_theta", 10000)  # 获取基础频率，默认10000
    scaling_type = rope_parameters.get("rope_type", "default")  # 获取缩放类型，默认"default"
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)  # 获取部分旋转因子

    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:  # 检查因子范围
        raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")  # 抛出范围错误
    rotary_dim = int(head_size * partial_rotary_factor)  # 计算旋转维度

    key = (  # 构建缓存键
        head_size,  # 头大小
        rotary_dim,  # 旋转维度
        max_position,  # 最大位置
        is_neox_style,  # NeoX风格标志
        rope_parameters_args,  # 旋转编码参数
        dual_chunk_attention_args,  # 双块注意力参数
        dtype,  # 数据类型
    )
    if key in _ROPE_DICT:  # 检查缓存中是否已存在
        return _ROPE_DICT[key]  # 直接返回缓存实例

    if dual_chunk_attention_config is not None:  # 双块注意力模式
        extra_kwargs = {  # 提取额外关键字参数
            k: v  # 键值对
            for k, v in dual_chunk_attention_config.items()  # 遍历配置
            if k in ("chunk_size", "local_size")  # 仅保留块大小和局部大小
        }
        rotary_emb = DualChunkRotaryEmbedding(  # 创建双块旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            max_position,  # 最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            dtype,  # 数据类型
            **extra_kwargs,  # 额外参数
        )
    elif scaling_type == "default":  # 默认缩放类型
        if "mrope_section" in rope_parameters:  # 多模态旋转编码
            rotary_emb = MRotaryEmbedding(  # 创建多模态旋转编码实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                max_position,  # 最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                dtype,  # 数据类型
                mrope_section=rope_parameters["mrope_section"],  # 多模态旋转编码分段
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),  # 是否交错
            )
        elif "use_fope" in rope_parameters and rope_parameters["use_fope"]:  # 使用傅里叶旋转编码
            extra_kwargs = {  # 提取FoPE额外参数
                k: v  # 键值对
                for k, v in rope_parameters.items()  # 遍历参数
                if k  # 检查键名
                in (  # 允许的参数列表
                    "num_key_value_heads",  # KV头数量
                    "num_inv_freq",  # 逆频率数量
                    "fope_sep_head",  # FoPE分离头标志
                    "fope_init_factor",  # FoPE初始化因子
                )
            }
            extra_kwargs["init_cache"] = False  # 不初始化缓存
            rotary_emb = FourierRotaryEmbedding(  # 创建傅里叶旋转编码实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                max_position,  # 最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                dtype,  # 数据类型
                **extra_kwargs,  # 额外参数
            )
        else:
            rotary_emb = RotaryEmbedding(  # 创建标准旋转编码实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                max_position,  # 最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                dtype,  # 数据类型
            )
    elif scaling_type == "llama3":  # Llama3缩放类型
        scaling_factor = rope_parameters["factor"]  # 获取缩放因子
        low_freq_factor = rope_parameters["low_freq_factor"]  # 获取低频因子
        high_freq_factor = rope_parameters["high_freq_factor"]  # 获取高频因子
        original_max_position = rope_parameters["original_max_position_embeddings"]  # 获取原始最大位置
        rotary_emb = Llama3RotaryEmbedding(  # 创建Llama3旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            max_position,  # 最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            dtype,  # 数据类型
            scaling_factor,  # 缩放因子
            low_freq_factor,  # 低频因子
            high_freq_factor,  # 高频因子
            original_max_position,  # 原始最大位置
        )
    elif scaling_type == "mllama4":  # Llama4视觉模型缩放类型
        rotary_emb = Llama4VisionRotaryEmbedding(  # 创建Llama4视觉旋转编码实例
            head_size, rotary_dim, max_position, base, is_neox_style, dtype  # 基本参数
        )
    elif scaling_type == "linear":  # 线性缩放类型
        scaling_factor = rope_parameters["factor"]  # 获取缩放因子
        rotary_emb = LinearScalingRotaryEmbedding(  # 创建线性缩放旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            max_position,  # 最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            scaling_factor,  # 缩放因子
            dtype,  # 数据类型
        )
    elif scaling_type == "ntk":  # NTK缩放类型
        scaling_factor = rope_parameters["factor"]  # 获取缩放因子
        mixed_b = rope_parameters.get("mixed_b")  # 获取混合参数b
        rotary_emb = NTKScalingRotaryEmbedding(  # 创建NTK缩放旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            max_position,  # 最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            scaling_factor,  # 缩放因子
            dtype,  # 数据类型
            mixed_b,  # 混合参数
        )
    elif scaling_type == "dynamic":  # 动态缩放类型
        if "alpha" in rope_parameters:  # 使用alpha参数的动态NTK
            scaling_alpha = rope_parameters["alpha"]  # 获取缩放alpha值
            rotary_emb = DynamicNTKAlphaRotaryEmbedding(  # 创建动态NTK Alpha实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                max_position,  # 最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                scaling_alpha,  # 缩放alpha
                dtype,  # 数据类型
            )
        elif "factor" in rope_parameters:  # 使用factor参数的动态NTK
            scaling_factor = rope_parameters["factor"]  # 获取缩放因子
            rotary_emb = DynamicNTKScalingRotaryEmbedding(  # 创建动态NTK缩放实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                max_position,  # 最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                scaling_factor,  # 缩放因子
                dtype,  # 数据类型
            )
        else:
            raise ValueError(  # 缺少必要参数时抛出错误
                "Dynamic rope scaling must contain either 'alpha' or 'factor' field"
            )
    elif scaling_type == "xdrope":  # XD旋转编码类型
        scaling_alpha = rope_parameters["alpha"]  # 获取缩放alpha值
        rotary_emb = XDRotaryEmbedding(  # 创建XD旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            max_position,  # 最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            scaling_alpha,  # 缩放alpha
            dtype,  # 数据类型
            xdrope_section=rope_parameters["xdrope_section"],  # XD旋转编码分段
        )
    elif scaling_type == "yarn":  # YaRN缩放类型
        scaling_factor = rope_parameters["factor"]  # 获取缩放因子
        original_max_position = rope_parameters["original_max_position_embeddings"]  # 获取原始最大位置
        extra_kwargs = {  # 提取YaRN额外参数
            k: v  # 键值对
            for k, v in rope_parameters.items()  # 遍历参数
            if k  # 检查键名
            in (  # 允许的参数列表
                "extrapolation_factor",  # 外推因子
                "attn_factor",  # 注意力因子
                "beta_fast",  # 快速beta参数
                "beta_slow",  # 慢速beta参数
                "apply_yarn_scaling",  # 是否应用YaRN缩放
                "truncate",  # 是否截断
            )
        }
        if "mrope_section" in rope_parameters:  # 带有多模态分段的YaRN
            extra_kwargs.pop("apply_yarn_scaling", None)  # 移除YaRN缩放标志
            rotary_emb = MRotaryEmbedding(  # 创建多模态旋转编码实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                original_max_position,  # 原始最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                dtype,  # 数据类型
                mrope_section=rope_parameters["mrope_section"],  # 多模态分段
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),  # 是否交错
                scaling_factor=scaling_factor,  # 缩放因子
                **extra_kwargs,  # 额外参数
            )
        else:
            rotary_emb = YaRNScalingRotaryEmbedding(  # 创建YaRN缩放旋转编码实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                original_max_position,  # 原始最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                scaling_factor,  # 缩放因子
                dtype,  # 数据类型
                **extra_kwargs,  # 额外参数
            )
    elif scaling_type in ["deepseek_yarn", "deepseek_llama_scaling"]:  # DeepSeek缩放类型
        scaling_factor = rope_parameters["factor"]  # 获取缩放因子
        original_max_position = rope_parameters["original_max_position_embeddings"]  # 获取原始最大位置
        # assert max_position == original_max_position * scaling_factor
        extra_kwargs = {  # 提取DeepSeek额外参数
            k: v  # 键值对
            for k, v in rope_parameters.items()  # 遍历参数
            if k  # 检查键名
            in (  # 允许的参数列表
                "extrapolation_factor",  # 外推因子
                "attn_factor",  # 注意力因子
                "beta_fast",  # 快速beta参数
                "beta_slow",  # 慢速beta参数
                "mscale",  # 幅度缩放
                "mscale_all_dim",  # 全维度幅度缩放
            )
        }
        rotary_emb = DeepseekScalingRotaryEmbedding(  # 创建DeepSeek缩放旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            original_max_position,  # 原始最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            scaling_factor,  # 缩放因子
            dtype,  # 数据类型
            **extra_kwargs,  # 额外参数
        )
    elif scaling_type == "longrope":  # 长距离RoPE缩放类型
        short_factor = rope_parameters["short_factor"]  # 获取短距离因子
        long_factor = rope_parameters["long_factor"]  # 获取长距离因子
        original_max_position = rope_parameters["original_max_position_embeddings"]  # 获取原始最大位置
        extra_kwargs = {  # 提取额外参数
            k: v  # 键值对
            for k, v in rope_parameters.items()  # 遍历参数
            if k in ("short_mscale", "long_mscale")  # 短距离和长距离幅度缩放
        }
        rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(  # 创建Phi3长距离缩放旋转编码实例
            head_size,  # 头大小
            rotary_dim,  # 旋转维度
            max_position,  # 最大位置
            original_max_position,  # 原始最大位置
            base,  # 基础频率
            is_neox_style,  # NeoX风格
            dtype,  # 数据类型
            short_factor,  # 短距离因子
            long_factor,  # 长距离因子
            **extra_kwargs,  # 额外参数
        )
    elif scaling_type == "openpangu":  # OpenPangu缩放类型
        mrope_interleaved = rope_parameters.get("mrope_interleaved", False)  # 获取交错标志
        if "mrope_section" in rope_parameters and mrope_interleaved:  # 需要交错多模态旋转编码
            rotary_emb = MRotaryEmbeddingInterleaved(  # 创建交错多模态旋转编码实例
                head_size,  # 头大小
                rotary_dim,  # 旋转维度
                max_position,  # 最大位置
                base,  # 基础频率
                is_neox_style,  # NeoX风格
                dtype,  # 数据类型
                mrope_section=rope_parameters["mrope_section"],  # 多模态分段
                mrope_interleaved=mrope_interleaved,  # 交错标志
            )
        else:
            raise ValueError("Pangu mrope lacks necessary parameters.")  # 缺少必要参数
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")  # 未知的缩放类型
    _ROPE_DICT[key] = rotary_emb  # 将实例存入缓存
    return rotary_emb  # 返回旋转编码实例
