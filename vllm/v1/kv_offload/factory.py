# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib  # 导入动态模块导入工具
from collections.abc import Callable  # 导入可调用类型
from typing import TYPE_CHECKING  # 导入类型检查标志

from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.v1.kv_offload.spec import OffloadingSpec  # 导入卸载规格基类

if TYPE_CHECKING:  # 仅在类型检查时导入以下模块
    from vllm.config import VllmConfig  # vLLM 配置类
    from vllm.v1.kv_cache_interface import KVCacheConfig  # KV 缓存配置类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class OffloadingSpecFactory:
    """卸载规格工厂类，用于注册和创建不同类型的卸载规格实例。"""

    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}  # 已注册规格的字典，名称 -> 延迟加载器

    @classmethod  # 类方法装饰器
    def register_spec(cls, name: str, module_path: str, class_name: str) -> None:  # 注册一个卸载规格
        """Register a spec with a lazy-loading module and class name.
        使用延迟加载的方式注册一个卸载规格。"""
        if name in cls._registry:  # 检查是否已注册同名规格
            raise ValueError(f"Connector '{name}' is already registered.")  # 抛出重复注册错误

        def loader() -> type[OffloadingSpec]:  # 定义延迟加载函数
            module = importlib.import_module(module_path)  # 动态导入模块
            return getattr(module, class_name)  # 从模块中获取类

        cls._registry[name] = loader  # 将加载器注册到注册表中

    @classmethod  # 类方法装饰器
    def create_spec(  # 根据配置创建卸载规格实例
        cls,
        config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> OffloadingSpec:
        """根据配置创建对应的卸载规格实例。"""
        kv_transfer_config = config.kv_transfer_config  # 获取 KV 传输配置
        assert kv_transfer_config is not None  # 断言 KV 传输配置不为空
        extra_config = kv_transfer_config.kv_connector_extra_config  # 获取连接器额外配置
        spec_name = extra_config.get("spec_name", "CPUOffloadingSpec")  # 获取规格名称，默认为 CPU 卸载规格
        if spec_name in cls._registry:  # 如果规格已注册
            spec_cls = cls._registry[spec_name]()  # 通过延迟加载器获取规格类
        else:  # 如果规格未注册
            spec_module_path = extra_config.get("spec_module_path")  # 从额外配置中获取模块路径
            if spec_module_path is None:  # 如果模块路径为空
                raise ValueError(f"Unsupported spec type: {spec_name}")  # 抛出不支持的规格类型错误
            spec_module = importlib.import_module(spec_module_path)  # 动态导入模块
            spec_cls = getattr(spec_module, spec_name)  # 从模块中获取规格类
        assert issubclass(spec_cls, OffloadingSpec)  # 断言获取的类是 OffloadingSpec 的子类
        logger.info("Creating offloading spec with name: %s", spec_name)  # 记录创建信息
        return spec_cls(config, kv_cache_config)  # 创建并返回规格实例


# Register various specs here.
# 在此处注册各种卸载规格。
OffloadingSpecFactory.register_spec(  # 注册 CPU 卸载规格
    "CPUOffloadingSpec", "vllm.v1.kv_offload.cpu", "CPUOffloadingSpec"
)
