# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

from .protocol import TokenizerLike  # 从protocol模块导入TokenizerLike协议类
from .registry import (  # 从registry模块导入以下组件
    TokenizerRegistry,  # 分词器注册表类
    cached_get_tokenizer,  # 带缓存的分词器获取函数
    cached_tokenizer_from_config,  # 从配置获取缓存分词器的函数
    get_tokenizer,  # 获取分词器的函数
)

__all__ = [  # 定义模块公开的接口列表
    "TokenizerLike",  # 分词器协议类
    "TokenizerRegistry",  # 分词器注册表
    "cached_get_tokenizer",  # 带缓存的分词器获取函数
    "get_tokenizer",  # 分词器获取函数
    "cached_tokenizer_from_config",  # 从配置获取缓存分词器
]
