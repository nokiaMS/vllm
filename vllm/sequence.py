# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence and its related classes."""  # 序列及其相关类的模块

from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Any  # 导入类型检查标志和Any类型

import torch  # 导入PyTorch库

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环导入
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput  # 导入KV连接器输出类型
else:
    KVConnectorOutput = Any  # 运行时使用Any替代，避免导入开销


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """中间张量数据结构，用于流水线并行中各阶段之间传递隐藏状态。

    对于除最后一个之外的所有流水线阶段，需要将隐藏状态和残差返回
    并发送到下一个阶段。此数据结构包含请求的隐藏状态和残差。

    每个阶段还需要处理自己的kv_connector_output。

    Attributes:
        tensors: 字符串到张量的字典，存储隐藏状态等中间结果。
        kv_connector_output: KV连接器的输出，可选。
    """

    tensors: dict[str, torch.Tensor]  # 存储中间张量的字典，键为名称，值为张量
    kv_connector_output: KVConnectorOutput | None  # KV连接器输出，流水线并行时使用，可为None

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None:
        """初始化中间张量对象。

        手动定义此函数而非使用dataclass自动生成，
        以便Dynamo能够知道IntermediateTensors()来自此文件。
        否则dataclass会通过执行字符串来生成此函数，导致丢失源文件信息。

        Args:
            tensors: 字符串到张量的字典，存储隐藏状态等中间结果。
            kv_connector_output: KV连接器输出，默认为None。
        """
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors  # 保存中间张量字典
        self.kv_connector_output = kv_connector_output  # 保存KV连接器输出

    def __getitem__(self, key: str | slice):
        """通过键名或切片获取中间张量。

        Args:
            key: 字符串键名获取单个张量，切片获取所有张量的对应切片。

        Returns:
            字符串键时返回对应的张量，切片时返回新的IntermediateTensors对象。
        """
        if isinstance(key, str):  # 如果键为字符串
            return self.tensors[key]  # 返回对应名称的张量
        elif isinstance(key, slice):  # 如果键为切片
            return self.__class__({k: v[key] for k, v in self.tensors.items()})  # 对所有张量应用切片，返回新对象

    def __setitem__(self, key: str, value: torch.Tensor):
        """通过键名设置中间张量。

        Args:
            key: 张量的名称。
            value: 要设置的张量值。
        """
        self.tensors[key] = value  # 设置指定名称的张量

    def items(self):
        """返回张量字典的键值对视图。

        Returns:
            张量字典的items视图。
        """
        return self.tensors.items()  # 返回张量字典的items视图

    def __len__(self):
        """返回中间张量的数量。

        Returns:
            张量字典中的元素数量。
        """
        return len(self.tensors)  # 返回张量字典的长度

    def __eq__(self, other: object):
        """判断两个IntermediateTensors对象是否相等。

        Args:
            other: 要比较的对象。

        Returns:
            如果两个对象的所有张量都相等则返回True，否则返回False。
        """
        if not isinstance(other, self.__class__):  # 如果类型不同
            return False  # 返回False
        if self.tensors.keys() != other.tensors.keys():  # 如果键集合不同
            return False  # 返回False
        return all(torch.equal(self.tensors[k], other.tensors[k]) for k in self.tensors)  # 逐个比较所有张量是否相等

    def __repr__(self) -> str:
        """返回IntermediateTensors对象的字符串表示。

        Returns:
            对象的字符串表示形式。
        """
        return f"IntermediateTensors(tensors={self.tensors})"  # 返回包含张量字典信息的字符串表示
