# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass, field  # 导入数据类装饰器和字段函数
from pathlib import Path  # 导入路径处理类
from typing import Any, Generic, TypeVar  # 导入类型标注工具

import numpy as np  # 导入NumPy数值计算库

_T = TypeVar("_T")  # 定义泛型类型变量


@dataclass  # 数据类装饰器
class MediaWithBytes(Generic[_T]):
    """
    Wrapper that couples a media object with its original encoded bytes.
    将媒体对象与其原始编码字节耦合的包装类。

    This ensures the raw bytes and media object remain synchronized,
    preventing cache corruption from in-place modifications.
    这确保原始字节和媒体对象保持同步，防止原地修改导致的缓存损坏。

    The wrapper delegates attribute access to the underlying media object,
    making it behave transparently like the wrapped type (e.g., PIL.Image).
    包装器将属性访问委托给底层媒体对象，使其行为透明地像被包装的类型。

    NOTE: Currently, this wrapper is used only for the image modality.
    注意：目前此包装器仅用于图像模态。
    """

    media: _T  # 底层媒体对象
    original_bytes: bytes = field(repr=False)  # 原始编码字节，不包含在repr中

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Allow np.array(obj) to return np.array(obj.media).
        允许np.array(obj)返回np.array(obj.media)。
        """
        return np.array(self.media, *args, **kwargs)  # 将底层媒体对象转为numpy数组

    def __getstate__(self):
        """获取对象状态，用于序列化。"""
        return self.__dict__.copy()  # 返回实例字典的拷贝

    def __setstate__(self, state: dict[str, Any]):
        """设置对象状态，用于反序列化。"""
        self.__dict__.update(state)  # 从状态字典恢复实例

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying media object.
        将属性访问委托给底层媒体对象。
        """
        return getattr(self.media, name)  # 从底层媒体对象获取属性


class MediaIO(ABC, Generic[_T]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    媒体IO抽象基类。配置值可以通过--media-io-kwargs或运行时API字段"media_io_kwargs"由用户提供。
    确保正确的验证和错误处理。
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge config-level kwargs and request-level kwargs.
        合并配置级别和请求级别的关键字参数。

        By default this performs a shallow merge where runtime kwargs override
        keys in default kwargs. Subclasses may override to apply modality-
        specific behavior.
        默认执行浅合并，运行时参数覆盖默认参数中的键。子类可以重写以应用特定模态的行为。
        """
        merged = dict(default_kwargs or {})  # 从默认参数创建字典
        if runtime_kwargs:  # 如果有运行时参数
            merged.update(runtime_kwargs)  # 用运行时参数覆盖默认值
        return merged  # 返回合并后的字典

    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        """从字节数据加载媒体对象。"""
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """
        List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        从base64编码的字符串加载媒体对象。
        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        """从文件路径加载媒体对象。"""
        raise NotImplementedError  # 子类必须实现
