# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
from typing import Any  # 导入Any类型，表示任意类型

from pydantic import ConfigDict  # 导入pydantic的配置字典类型
from pydantic.dataclasses import dataclass  # 导入pydantic的dataclass装饰器

from vllm.logger import init_logger  # 导入vLLM日志初始化函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))  # 允许任意类型的pydantic数据类
class ModelArchitectureConfig:
    """
    vLLM运行时所需的模型架构配置类。
    """

    architectures: list[str] | None  # 模型架构类名列表（例如 ['LlamaForCausalLM']），可以为None
    """List of model architecture class names (e.g., ['LlamaForCausalLM']).
       It can be None upon calling `vllm_config.with_hf_config(config.text_config)`"""

    model_type: str  # 模型类型标识符（例如 'llama', 'gpt_oss'）
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    text_model_type: str | None  # 文本模型类型标识符（例如 'llama4_text'），可以为None
    """Text model type identifier (e.g., 'llama4_text')."""

    hidden_size: int  # 模型的隐藏层大小
    """Hidden size of the model."""

    total_num_hidden_layers: int  # 模型的隐藏层总数
    """Number of hidden layers in the model."""

    total_num_attention_heads: int  # 模型的注意力头总数
    """Number of attention heads in the model."""

    head_size: int  # 注意力头的维度大小
    """Head dimension of the model."""

    vocab_size: int  # 模型的词汇表大小
    """Vocabulary size of the model."""

    total_num_kv_heads: int  # 键值注意力头的总数
    """Number of key value heads in the model."""

    num_experts: int  # MoE模型中专家的数量
    """Number of experts in the model."""

    quantization_config: dict[str, Any] | None  # 量化配置字典，包含量化参数
    """Quantization configuration dictionary containing quantization parameters."""

    is_deepseek_mla: bool  # 是否为DeepSeek MLA模型
    """Whether the model is a DeepSeek MLA model."""

    derived_max_model_len_and_key: tuple[float, str | None]  # 从HF配置推导出的最大模型长度及其对应的键名
    """Derived maximum model length and key from the hf config."""
