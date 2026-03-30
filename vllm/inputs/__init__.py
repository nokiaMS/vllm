# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from .data import (  # 从data模块导入数据类型定义
    DataPrompt,  # 数据提示类型
    DecoderOnlyInputs,  # 仅解码器输入类型
    EmbedsInputs,  # 嵌入输入类型
    EmbedsPrompt,  # 嵌入提示类型
    EncoderDecoderInputs,  # 编码器-解码器输入类型
    ExplicitEncoderDecoderPrompt,  # 显式编码器-解码器提示类型
    ProcessorInputs,  # 处理器输入类型
    PromptType,  # 提示类型别名
    SingletonInputs,  # 单例输入类型
    SingletonPrompt,  # 单例提示类型
    TextPrompt,  # 文本提示类型
    TokenInputs,  # 令牌输入类型
    TokensPrompt,  # 令牌提示类型
    embeds_inputs,  # 嵌入输入构造函数
    token_inputs,  # 令牌输入构造函数
)

__all__ = [  # 模块公开接口列表
    "DataPrompt",  # 数据提示
    "TextPrompt",  # 文本提示
    "TokensPrompt",  # 令牌提示
    "PromptType",  # 提示类型
    "SingletonPrompt",  # 单例提示
    "ExplicitEncoderDecoderPrompt",  # 显式编码器-解码器提示
    "TokenInputs",  # 令牌输入
    "EmbedsInputs",  # 嵌入输入
    "EmbedsPrompt",  # 嵌入提示
    "token_inputs",  # 令牌输入构造器
    "embeds_inputs",  # 嵌入输入构造器
    "DecoderOnlyInputs",  # 仅解码器输入
    "EncoderDecoderInputs",  # 编码器-解码器输入
    "ProcessorInputs",  # 处理器输入
    "SingletonInputs",  # 单例输入
]
