# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import BaseRenderer  # 导入基础渲染器类
from .params import ChatParams, TokenizeParams, merge_kwargs  # 导入聊天参数、分词参数和合并关键字参数工具
from .registry import RendererRegistry, renderer_from_config  # 导入渲染器注册表和从配置创建渲染器的函数

__all__ = [  # 定义模块公开接口
    "BaseRenderer",  # 基础渲染器
    "RendererRegistry",  # 渲染器注册表
    "renderer_from_config",  # 从配置创建渲染器
    "ChatParams",  # 聊天参数
    "TokenizeParams",  # 分词参数
    "merge_kwargs",  # 合并关键字参数
]
