# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .hasher import MultiModalHasher  # 导入多模态哈希计算器
from .inputs import (  # 从inputs模块导入多模态输入相关类型
    BatchedTensorInputs,  # 批量张量输入类型
    ModalityData,  # 模态数据类型
    MultiModalDataBuiltins,  # 多模态数据内置类型定义
    MultiModalDataDict,  # 多模态数据字典类型
    MultiModalKwargsItems,  # 多模态关键字参数项
    MultiModalPlaceholderDict,  # 多模态占位符字典类型
    MultiModalUUIDDict,  # 多模态UUID字典类型
    NestedTensors,  # 嵌套张量类型
)
from .registry import MultiModalRegistry  # 导入多模态注册表

MULTIMODAL_REGISTRY = MultiModalRegistry()  # 创建全局多模态注册表实例
"""
The global [`MultiModalRegistry`][vllm.multimodal.registry.MultiModalRegistry]
is used by model runners to dispatch data processing according to the target
model.
全局多模态注册表，模型运行器使用它根据目标模型分发数据处理。

Info:
    [mm_processing](../../../design/mm_processing.md)
"""

__all__ = [  # 定义模块的公开接口列表
    "BatchedTensorInputs",  # 批量张量输入
    "ModalityData",  # 模态数据
    "MultiModalDataBuiltins",  # 多模态数据内置类型
    "MultiModalDataDict",  # 多模态数据字典
    "MultiModalHasher",  # 多模态哈希器
    "MultiModalKwargsItems",  # 多模态关键字参数项
    "MultiModalPlaceholderDict",  # 多模态占位符字典
    "MultiModalUUIDDict",  # 多模态UUID字典
    "NestedTensors",  # 嵌套张量
    "MULTIMODAL_REGISTRY",  # 多模态注册表实例
    "MultiModalRegistry",  # 多模态注册表类
]
