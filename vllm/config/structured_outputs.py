# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from typing import Any, Literal  # 导入Any和Literal类型

from pydantic import model_validator  # 导入pydantic模型验证器
from typing_extensions import Self  # 导入Self类型，用于返回类型标注

from vllm.config.utils import config  # 导入config装饰器
from vllm.utils.hashing import safe_hash  # 导入安全哈希函数

StructuredOutputsBackend = Literal[  # 结构化输出后端类型定义
    "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"
]


@config  # 使用config装饰器创建pydantic数据类
class StructuredOutputsConfig:
    """引擎的结构化输出配置数据类。"""

    backend: StructuredOutputsBackend = "auto"  # 结构化输出使用的引擎，默认"auto"自动选择
    """Which engine will be used for structured outputs (e.g. JSON schema,
    regex, etc) by default. With "auto", we will make opinionated choices
    based on request contents and what the backend libraries currently support,
    so the behavior is subject to change in each release."""
    disable_any_whitespace: bool = False  # 如果为True，JSON输出将始终紧凑无空格
    """If `True`, json output will always be compact without any whitespace.
    If `False`, the model may generate whitespace between JSON fields,
    which is still valid JSON. This is only supported for xgrammar
    and guidance backends."""
    disable_additional_properties: bool = False  # 如果为True，guidance后端不使用additionalProperties
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""
    reasoning_parser: str = ""  # 推理解析器名称，用于将推理内容解析为OpenAI API格式
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""
    reasoning_parser_plugin: str = ""  # 动态加载的推理解析器插件路径
    """Path to a dynamically reasoning parser plugin that can be dynamically
    loaded and registered."""
    enable_in_reasoning: bool = False  # 是否在推理过程中使用结构化输入
    """Whether to use structured input for reasoning."""

    def compute_hash(self) -> str:
        """
        计算唯一标识此配置的哈希值。

        警告：每当向此配置添加新字段时，如果影响计算图，
        请确保将其包含在因子列表中。
        """
        # 无需考虑任何因子
        # 此配置不会影响计算图
        factors: list[Any] = []  # 空因子列表
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()  # 计算哈希值
        return hash_str  # 返回哈希字符串

    @model_validator(mode="after")  # 模型验证器，在初始化后运行
    def _validate_structured_output_config(self) -> Self:
        """验证结构化输出配置的一致性。"""
        if self.disable_any_whitespace and self.backend not in ("xgrammar", "guidance"):  # 检查空格禁用选项兼容性
            raise ValueError(
                "disable_any_whitespace is only supported for "
                "xgrammar and guidance backends."
            )
        if self.disable_additional_properties and self.backend != "guidance":  # 检查附加属性禁用选项兼容性
            raise ValueError(
                "disable_additional_properties is only supported "
                "for the guidance backend."
            )
        return self  # 返回验证后的实例
