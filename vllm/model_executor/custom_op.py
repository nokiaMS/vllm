# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools  # 导入函数工具模块，用于高阶函数操作
import inspect  # 导入检查模块，用于获取函数签名信息

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块

from vllm.config import get_cached_compilation_config  # 导入获取缓存编译配置的函数
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.model_executor.utils import maybe_disable_graph_partition  # 导入可能禁用图分区的工具函数
from vllm.platforms import current_platform  # 导入当前平台信息

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

# Dictionary of all custom ops (classes, indexed by registered name).
# To check if an op with a name is enabled, call .enabled() on the class.
# Examples:
# - MyOp.enabled()
# - op_registry["my_op"].enabled()
op_registry: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}  # 所有自定义算子的注册表（按名称索引的类字典）
op_registry_oot: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}  # 树外(out-of-tree)自定义算子的注册表


def get_oot_class_by_name(class_name: str) -> type | None:
    """根据类名获取树外(out-of-tree)注册的类。"""
    if class_name in op_registry_oot:  # 如果类名在树外注册表中
        return op_registry_oot[class_name]  # 返回对应的类
    return None  # 未找到则返回None


class PluggableLayer(nn.Module):
    """
    Base class for pluggable layers.

    A PluggableLayer is a *module-composing* abstraction: it may instantiate other
    ``torch.nn.Module`` objects as sub-layers, and its functionality depends on
    these sub-layers following a generalized invocation sequence. Also, it is stateful
    and may hold parameters or buffers.

    Unlike :class:`CustomOp`, PluggableLayer does NOT provide per-platform
    ``forward_*`` dispatch. Instead, it supports out-of-tree (OOT) replacement
    of the entire layer class at instantiation time, allowing customized
    initialization and submodule composition.
    """
    # 可插拔层的基类。
    # PluggableLayer是一种模块组合抽象：它可以实例化其他torch.nn.Module对象作为子层，
    # 其功能依赖于这些子层遵循通用调用序列。它是有状态的，可以持有参数或缓冲区。
    # 与CustomOp不同，PluggableLayer不提供按平台的forward_*调度，
    # 而是支持在实例化时对整个层类进行树外(OOT)替换。

    def __new__(cls, *args, **kwargs):
        """创建PluggableLayer实例，支持树外替换。"""
        try:
            layer_class_name = cls.__name__  # 获取层的类名
        except AttributeError:
            raise TypeError(  # 如果无法获取名称属性，抛出类型错误
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@PluggableLayer.register, or it's the PluggableLayer itself."
            ) from None

        if layer_class_name not in op_registry_oot:  # 如果不在树外注册表中
            layer_cls_to_instantiate = cls  # 使用原始类
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]  # 使用树外替换类
            logger.debug(  # 记录调试日志
                "Instantiating pluggable layer: %s using %s",
                layer_class_name,
                str(layer_cls_to_instantiate),
            )
        return super().__new__(layer_cls_to_instantiate)  # 创建并返回实例

    # Decorator to register pluggable layers.
    @classmethod
    def register(cls, name: str):
        """注册可插拔层的装饰器。"""
        def decorator(op_cls):
            """内部装饰器函数，将层类注册到注册表中。"""
            assert name not in op_registry, f"Duplicate op name: {name}"  # 确保名称不重复
            op_cls.name = name  # 设置层的名称
            op_registry[name] = op_cls  # 将层注册到全局注册表
            return op_cls  # 返回原始类

        return decorator  # 返回装饰器

    # Decorator to register out-of-tree(oot) pluggable layers.
    # For OOT pluggable layers:
    #   if in-tree layer class is registered with an oot_custom_layer,
    #   the oot_custom_layer will be used instead.
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        """注册树外(out-of-tree)可插拔层的装饰器，用于替换树内同名层。"""
        def decorator(layer_cls):
            """内部装饰器函数，将树外层类注册到树外注册表中。"""
            reg_name = name if name is not None else cls.__name__  # 使用指定名称或类名
            assert reg_name not in op_registry_oot, f"Duplicate layer name: {reg_name}"  # 确保名称不重复
            layer_cls.name = reg_name  # 设置层的名称
            op_registry_oot[reg_name] = layer_cls  # 注册到树外注册表
            return layer_cls  # 返回原始类

        if _decorated_layer_cls is None:  # 带括号调用
            # Called with parentheses: @PluggableLayer.register_oot()
            # or @PluggableLayer.register_oot(name="...")
            return decorator  # 返回装饰器函数
        elif isinstance(_decorated_layer_cls, type):  # Check if it's a class  # 不带括号调用
            # Called without parentheses: @PluggableLayer.register_oot
            return decorator(_decorated_layer_cls)  # 直接应用装饰器
        else:
            raise TypeError("Decorator can only be applied to classes.")  # 装饰器只能应用于类


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """
    # 自定义算子的基类。将前向方法分派到适当的后端执行。

    def __new__(cls, *args, **kwargs):
        """创建CustomOp实例，支持树外替换。"""
        try:
            op_name = cls.__name__  # 获取算子名称
        except AttributeError:
            raise TypeError(  # 如果无法获取名称属性，抛出类型错误
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register, or it's the CustomOp base class itself."
            ) from None

        if op_name not in op_registry_oot:  # 如果不在树外注册表中
            op_cls_to_instantiate = cls  # 使用原始类
        else:
            op_cls_to_instantiate = op_registry_oot[op_name]  # 使用树外替换类
            logger.debug(  # 记录调试日志
                "Instantiating custom op: %s using %s",
                op_name,
                str(op_cls_to_instantiate),
            )
        return super().__new__(op_cls_to_instantiate)  # 创建并返回实例

    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        """初始化自定义算子，设置强制启用标志和编译原生标志。"""
        super().__init__()  # 调用父类初始化
        self._enforce_enable = enforce_enable  # 是否强制启用此算子
        self._forward_method = self.dispatch_forward(compile_native=compile_native)  # 根据平台分派前向方法

    def forward(self, *args, **kwargs):
        """前向传播，调用分派后的前向方法。"""
        return self._forward_method(*args, **kwargs)  # 调用已分派的前向方法

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        # PyTorch原生前向实现。此方法可选，可用于torch.compile或PyTorch XLA等编译器，也可用于测试。
        raise NotImplementedError  # 需要子类实现

    def forward_cuda(self, *args, **kwargs):
        """CUDA平台的前向实现。"""
        raise NotImplementedError  # 需要子类实现

    def forward_hip(self, *args, **kwargs):
        """HIP/ROCm平台的前向实现，默认兼容CUDA实现。"""
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)  # 默认调用CUDA前向方法

    def forward_xpu(self, *args, **kwargs):
        """XPU平台的前向实现，默认兼容PyTorch原生实现。"""
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)  # 默认调用原生前向方法

    def forward_cpu(self, *args, **kwargs):
        """CPU平台的前向实现，默认兼容PyTorch原生实现。"""
        # By default, we assume that CPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)  # 默认调用原生前向方法

    def forward_tpu(self, *args, **kwargs):
        """TPU平台的前向实现，默认兼容PyTorch原生实现。"""
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)  # 默认调用原生前向方法

    def forward_oot(self, *args, **kwargs):
        """树外(out-of-tree)平台的前向实现，默认兼容PyTorch原生实现。"""
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)  # 默认调用原生前向方法

    def dispatch_forward(self, compile_native: bool):
        """根据当前平台分派前向方法到对应的后端实现。"""
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        compilation_config = get_cached_compilation_config()  # 获取缓存的编译配置

        # NOTE(shen-shanshan): CustomOp object can be enforce enabled, e.g.,
        # enable device-specific kernels in ViT models when enabling graph
        # mode. By default, it will follow the compilation_config to determine
        # whether enable itself.
        # This enforce_enable mechanism will be removed after we adding a
        # separate compilation_config for multi-modal part.
        enabled = self._enforce_enable or self.enabled()  # 检查算子是否启用
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])  # 将算子加入已启用列表
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])  # 将算子加入已禁用列表

        if not enabled:  # 如果算子未启用
            # Compile forward_native to avoid eager torch ops if inside
            # opaque torch custom op (e.g. fused_moe, unified_attention, etc.)
            return self.maybe_compile(self.forward_native, enable=compile_native)  # 可能编译原生前向方法

        if current_platform.is_rocm():  # 如果是ROCm平台
            return self.forward_hip  # 返回HIP前向方法
        elif current_platform.is_cpu():  # 如果是CPU平台
            return self.forward_cpu  # 返回CPU前向方法
        elif current_platform.is_tpu():  # 如果是TPU平台
            return self.forward_tpu  # 返回TPU前向方法
        elif current_platform.is_xpu():  # 如果是XPU平台
            return self.forward_xpu  # 返回XPU前向方法
        elif current_platform.is_out_of_tree():  # 如果是树外平台
            return self.forward_oot  # 返回树外前向方法
        else:
            return self.forward_cuda  # 默认返回CUDA前向方法

    def maybe_compile(self, fn, *, enable: bool = True):
        """
        Compile fn if compilation enabled.
        Useful for CustomOp instances called from within a torch custom op,
        meaning the forward call is hidden from the model-level torch.compile.

        NOTE: this does not enable fusion across ops, so opaque custom ops
        should still be unwrapped wherever possible.
        """
        # 如果编译已启用，则编译函数fn。适用于从torch自定义算子内部调用的CustomOp实例。
        from vllm.config.compilation import CompilationMode  # 导入编译模式枚举

        # Do not compile if compilation disabled
        if not enable:  # 如果编译未启用
            return fn  # 直接返回原函数

        # Do not compile if global compilation disabled
        compilation_config = get_cached_compilation_config()  # 获取编译配置
        if compilation_config.mode == CompilationMode.NONE:  # 如果编译模式为NONE
            return fn  # 直接返回原函数

        # If eager backend is used, do not compile either
        if compilation_config.backend == "eager":  # 如果使用eager后端
            return fn  # 直接返回原函数

        compile_options = maybe_disable_graph_partition(  # 获取编译选项，可能禁用图分区
            current_platform.simple_compile_backend
        )
        backend = current_platform.simple_compile_backend  # 获取当前平台的简单编译后端

        dynamic_arg_dims = getattr(self.__class__, "_dynamic_arg_dims", None)  # 获取动态参数维度配置
        if dynamic_arg_dims is not None:  # 如果设置了动态参数维度
            compiled_fn = torch.compile(  # 使用torch.compile编译函数
                fn,
                dynamic=False,  # 不使用动态形状
                backend=backend,  # 使用指定后端
                options=compile_options,  # 使用编译选项
            )
            sig = inspect.signature(fn)  # 获取函数签名

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                """包装函数，在调用编译函数前标记动态维度。"""
                bound = sig.bind(*args, **kwargs)  # 绑定参数
                bound.apply_defaults()  # 应用默认值
                for name, dims in dynamic_arg_dims.items():  # 遍历动态参数维度
                    arg = bound.arguments.get(name)  # 获取对应参数
                    if arg is not None and isinstance(arg, torch.Tensor):  # 如果是张量
                        dims_list = [dims] if isinstance(dims, int) else dims  # 确保维度为列表
                        for d in dims_list:  # 遍历每个维度
                            real_d = arg.ndim + d if d < 0 else d  # 处理负数维度索引
                            torch._dynamo.mark_dynamic(arg, real_d)  # 标记动态维度
                return compiled_fn(*args, **kwargs)  # 调用编译后的函数

            return wrapper  # 返回包装函数

        # dynamic=True to avoid recompilations
        return torch.compile(  # 使用动态形状编译以避免重新编译
            fn,
            dynamic=True,  # 使用动态形状
            backend=backend,  # 使用指定后端
            options=compile_options,  # 使用编译选项
        )

    @classmethod
    def enabled(cls) -> bool:
        """检查此自定义算子是否启用。"""
        # if no name, then it was not registered
        compilation_config = get_cached_compilation_config()  # 获取编译配置
        custom_ops = compilation_config.custom_ops  # 获取自定义算子配置
        if not hasattr(cls, "name"):  # 如果没有注册名称
            logger.warning_once(  # 记录一次性警告
                "Custom op %s was not registered, which means it won't appear "
                "in the op registry. It will be enabled/disabled based on the "
                "global settings.",
                cls.__name__,
            )
            return CustomOp.default_on()  # 根据全局设置决定

        enabled = f"+{cls.name}" in custom_ops  # 检查是否显式启用
        disabled = f"-{cls.name}" in custom_ops  # 检查是否显式禁用
        assert not (enabled and disabled), f"Cannot enable and disable {cls.name}"  # 不能同时启用和禁用

        return (CustomOp.default_on() or enabled) and not disabled  # 返回最终启用状态

    @staticmethod
    def default_on() -> bool:
        """
        Behavior controlled by `CompilationConfig.custom_ops`: On by default if
        'all', off by default if 'none'.
        When PyTorch Inductor is used, 'none' is the default value,
        otherwise 'all'.
        """
        # 由CompilationConfig.custom_ops控制：'all'时默认开启，'none'时默认关闭。
        compilation_config = get_cached_compilation_config()  # 获取编译配置
        count_none = compilation_config.custom_ops.count("none")  # 统计'none'的数量
        count_all = compilation_config.custom_ops.count("all")  # 统计'all'的数量
        assert count_none + count_all == 1  # 确保只有一个全局设置

        return not count_none > 0 or count_all > 0  # 如果没有'none'或有'all'则返回True

    # Decorator to register custom ops.
    @classmethod
    def register(
        cls,
        name: str,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    ):
        """注册自定义算子的装饰器。"""
        def decorator(op_cls):
            """内部装饰器函数，将算子类注册到全局注册表。"""
            assert name not in op_registry, f"Duplicate op name: {name}"  # 确保名称不重复
            op_cls.name = name  # 设置算子名称
            op_cls._dynamic_arg_dims = dynamic_arg_dims  # 设置动态参数维度
            op_registry[name] = op_cls  # 注册到全局注册表
            return op_cls  # 返回原始类

        return decorator  # 返回装饰器

    # Decorator to register out-of-tree(oot) custom ops.
    # For OOT custom ops:
    #   if in-tree layer class is registered with an oot_custom_op layer,
    #   the oot_custom_op layer will be used instead.
    # Example:
    # - @UnquantizedFusedMoEMethod.register_oot
    #   class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod)
    # or
    # - @CustomOP.register_oot(name="UnquantizedFusedMoEMethod")
    @classmethod
    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):
        """注册树外(out-of-tree)自定义算子的装饰器，用于替换树内同名算子。"""
        def decorator(op_cls):
            """内部装饰器函数，将算子类注册到树外注册表。"""
            reg_name = name if name is not None else cls.__name__  # 使用指定名称或类名
            assert reg_name not in op_registry_oot, f"Duplicate op name: {reg_name}"  # 确保名称不重复
            op_cls.name = reg_name  # 设置算子名称
            op_registry_oot[reg_name] = op_cls  # 注册到树外注册表
            return op_cls  # 返回原始类

        if _decorated_op_cls is None:  # 带括号调用
            # Called with parentheses: @CustomOP.register_oot()
            # or @CustomOP.register_oot(name="...")
            # So, _decorated_op_cls is None.
            # We return the actual decorator function.
            return decorator  # 返回装饰器函数
        elif isinstance(_decorated_op_cls, type):  # Check if it's a class  # 不带括号调用
            # Called without parentheses: @CustomOP.register_oot
            # The first argument is the class itself.
            # We call the 'decorator' function immediately with the class.
            return decorator(_decorated_op_cls)  # 直接应用装饰器
        else:
            # Handle other unexpected cases if necessary
            raise TypeError("Decorator can only be applied to classes.")  # 装饰器只能应用于类
