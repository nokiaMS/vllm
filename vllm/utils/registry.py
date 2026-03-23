# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, TypeVar

_T = TypeVar("_T", bound=type)


# 可插拔扩展类的注册管理器，实现了简单的插件系统模式：
# 通过装饰器 register() 注册实现类，通过 load() 按名称实例化，
# 支持运行时动态切换不同的实现
class ExtensionManager:
    """
    A registry for managing pluggable extension classes.

    This class provides a simple mechanism to register and instantiate
    extension classes by name. It is commonly used to implement plugin
    systems where different implementations can be swapped at runtime.

    Examples:
        Basic usage with a registry instance:

        >>> FOO_REGISTRY = ExtensionManager()
        >>> @FOO_REGISTRY.register("my_foo_impl")
        ... class MyFooImpl(Foo):
        ...     def __init__(self, value):
        ...         self.value = value
        >>> foo_impl = FOO_REGISTRY.load("my_foo_impl", value=123)

    """

    # 初始化空的注册表，name2class 字典存储名称到类的映射
    def __init__(self) -> None:
        """
        Initialize an empty extension registry.
        """
        self.name2class: dict[str, type] = {}

    # 返回一个装饰器，将被装饰的类以指定名称注册到管理器中
    def register(self, name: str):
        """
        Decorator to register a class with the given name.
        """

        def wrap(cls_to_register: _T) -> _T:
            self.name2class[name] = cls_to_register
            return cls_to_register

        return wrap

    # 根据名称查找已注册的类并实例化，找不到时抛出断言错误
    def load(self, cls_name: str, *args, **kwargs) -> Any:
        """
        Instantiate and return a registered extension class by name.
        """
        cls = self.name2class.get(cls_name)
        assert cls is not None, f"Extension class {cls_name} not found"
        return cls(*args, **kwargs)
