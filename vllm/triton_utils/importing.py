# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

import os  # 导入操作系统接口模块
import types  # 导入类型工具模块，用于创建模块类型的占位符
from importlib.util import find_spec  # 导入模块查找工具，用于检测包是否已安装

from vllm.logger import init_logger  # 从vLLM日志模块导入日志初始化函数

logger = init_logger(__name__)  # 为当前模块初始化日志记录器

HAS_TRITON = (  # 检测Triton是否已安装
    find_spec("triton") is not None  # 检查标准triton包是否存在
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible  # 检查XPU版本的triton（不兼容）
)
if HAS_TRITON:  # 如果检测到Triton已安装
    try:  # 尝试进一步验证Triton的可用性
        from triton.backends import backends  # 导入Triton后端注册表

        # It's generally expected that x.driver exists and has
        # an is_active method.
        # The `x.driver and` check adds a small layer of safety.
        active_drivers = [  # 获取所有处于活跃状态的驱动程序列表
            x.driver for x in backends.values() if x.driver and x.driver.is_active()  # 过滤出有活跃驱动的后端
        ]

        # Check if we're in a distributed environment where CUDA_VISIBLE_DEVICES
        # might be temporarily empty (e.g., Ray sets it to "" during actor init)
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")  # 获取CUDA可见设备环境变量
        is_distributed_env = (  # 判断是否处于分布式环境中
            cuda_visible_devices is not None and len(cuda_visible_devices.strip()) == 0  # 环境变量存在但为空字符串
        )

        # Apply lenient driver check for distributed environments
        if is_distributed_env and len(active_drivers) == 0:  # 分布式环境中允许0个活跃驱动
            # Allow 0 drivers in distributed environments - they may become
            # active later when CUDA context is properly initialized
            logger.debug(  # 记录调试日志
                "Triton found 0 active drivers in distributed environment. "  # 提示在分布式环境中未找到活跃驱动
                "This is expected during initialization."  # 说明这在初始化期间是正常的
            )
        elif not is_distributed_env and len(active_drivers) != 1:  # 非分布式环境中要求恰好1个活跃驱动
            # Strict check for non-distributed environments
            logger.info(  # 记录信息日志
                "Triton is installed but %d active driver(s) found "  # 提示找到的活跃驱动数量
                "(expected 1). Disabling Triton to prevent runtime errors.",  # 预期为1个，禁用Triton以防止运行时错误
                len(active_drivers),  # 传入实际的活跃驱动数量
            )
            HAS_TRITON = False  # 将Triton可用标志设为False
    except ImportError:  # 捕获导入错误异常
        # This can occur if Triton is partially installed or triton.backends
        # is missing.
        logger.warning(  # 记录警告日志
            "Triton is installed, but `triton.backends` could not be imported. "  # 提示Triton已安装但后端模块导入失败
            "Disabling Triton."  # 禁用Triton
        )
        HAS_TRITON = False  # 将Triton可用标志设为False
    except Exception as e:  # 捕获所有其他异常
        # Catch any other unexpected errors during the check.
        logger.warning(  # 记录警告日志
            "An unexpected error occurred while checking Triton active drivers:"  # 提示检查Triton活跃驱动时发生意外错误
            " %s. Disabling Triton.",  # 输出错误信息并禁用Triton
            e,  # 传入异常对象
        )
        HAS_TRITON = False  # 将Triton可用标志设为False

if not HAS_TRITON:  # 如果Triton最终不可用
    logger.info(  # 记录信息日志
        "Triton not installed or not compatible; certain GPU-related"  # 提示Triton未安装或不兼容
        " functions will not be available."  # 某些GPU相关功能将不可用
    )


class TritonPlaceholder(types.ModuleType):  # 定义Triton主模块的占位符类，继承自ModuleType
    """Triton模块的占位符类。

    当Triton未安装时，提供一个模拟的triton模块，
    使得依赖triton装饰器的代码在导入时不会报错。
    所有装饰器都会变成空操作（no-op）。
    """
    def __init__(self):  # 占位符初始化方法
        """初始化Triton占位符模块。

        设置模块名称、版本号，并创建常用装饰器的空操作替代。
        """
        super().__init__("triton")  # 调用父类初始化，设置模块名为"triton"
        self.__version__ = "3.4.0"  # 设置模拟的版本号
        self.jit = self._dummy_decorator("jit")  # 创建jit装饰器的空操作替代
        self.autotune = self._dummy_decorator("autotune")  # 创建autotune装饰器的空操作替代
        self.heuristics = self._dummy_decorator("heuristics")  # 创建heuristics装饰器的空操作替代
        self.Config = self._dummy_decorator("Config")  # 创建Config装饰器的空操作替代
        self.language = TritonLanguagePlaceholder()  # 创建Triton语言模块的占位符实例

    def _dummy_decorator(self, name):  # 创建空操作装饰器的工厂方法
        """创建一个空操作装饰器。

        Args:
            name: 装饰器名称（仅用于标识，不影响功能）。

        Returns:
            一个装饰器函数，直接返回被装饰的函数，不做任何修改。
        """
        def decorator(*args, **kwargs):  # 定义装饰器函数，接受任意参数
            if args and callable(args[0]):  # 如果第一个参数是可调用对象（直接作为装饰器使用）
                return args[0]  # 直接返回被装饰的函数
            return lambda f: f  # 否则返回一个恒等函数（带参数的装饰器用法）

        return decorator  # 返回装饰器函数


class TritonLanguagePlaceholder(types.ModuleType):  # 定义Triton语言模块的占位符类
    """Triton语言模块(triton.language)的占位符类。

    当Triton未安装时，提供一个模拟的triton.language模块，
    将常用的类型和函数属性设置为None，避免属性访问错误。
    """
    def __init__(self):  # 占位符初始化方法
        """初始化Triton语言占位符模块。

        将所有常用的triton.language属性设置为None。
        """
        super().__init__("triton.language")  # 调用父类初始化，设置模块名为"triton.language"
        self.constexpr = None  # 编译期常量类型占位
        self.dtype = None  # 数据类型占位
        self.int64 = None  # 64位整数类型占位
        self.int32 = None  # 32位整数类型占位
        self.tensor = None  # 张量类型占位
        self.exp = None  # 指数函数占位
        self.log = None  # 自然对数函数占位
        self.log2 = None  # 以2为底对数函数占位
