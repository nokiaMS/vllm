# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
import logging  # 导入日志模块
import traceback  # 导入堆栈追踪模块
from itertools import chain  # 从itertools导入chain函数，用于串联迭代器
from typing import TYPE_CHECKING  # 导入类型检查标志

from vllm import envs  # 导入vLLM环境变量配置
from vllm.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group  # 导入平台插件组名和插件加载函数
from vllm.utils.import_utils import resolve_obj_by_qualname  # 导入通过全限定名解析对象的工具函数
from vllm.utils.torch_utils import supports_xccl  # 导入检测XCCL支持的工具函数

from .interface import CpuArchEnum, Platform, PlatformEnum  # 从接口模块导入CPU架构枚举、平台基类和平台枚举

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def vllm_version_matches_substr(substr: str) -> bool:
    """
    检查vLLM版本字符串是否包含指定的子字符串。

    参数:
        substr: 要匹配的子字符串

    返回:
        如果vLLM版本包含该子字符串则返回True，否则返回False
    """
    from importlib.metadata import PackageNotFoundError, version  # 延迟导入包元数据工具

    try:
        vllm_version = version("vllm")  # 获取vLLM的安装版本号
    except PackageNotFoundError as e:  # 捕获包未找到异常
        logger.warning(  # 记录警告日志
            "The vLLM package was not found, so its version could not be "
            "inspected. This may cause platform detection to fail."
        )
        raise e  # 重新抛出异常
    return substr in vllm_version  # 返回子字符串是否在版本号中


def tpu_platform_plugin() -> str | None:
    """
    检测TPU平台是否可用的插件函数。

    返回:
        如果TPU可用则返回TPU平台类的全限定名，否则返回None
    """
    logger.debug("Checking if TPU platform is available.")  # 调试日志：检查TPU平台

    # Check for Pathways TPU proxy  # 检查Pathways TPU代理
    if envs.VLLM_TPU_USING_PATHWAYS:  # 如果配置了通过Pathways使用TPU
        logger.debug("Confirmed TPU platform is available via Pathways proxy.")  # 确认TPU可通过Pathways代理使用
        return "tpu_inference.platforms.tpu_platform.TpuPlatform"  # 返回Pathways TPU平台类路径

    # Check for libtpu installation  # 检查libtpu库是否已安装
    try:
        # While it's technically possible to install libtpu on a  # 虽然技术上可以在非TPU机器上安装libtpu
        # non-TPU machine, this is a very uncommon scenario. Therefore,  # 但这种情况非常罕见，因此
        # we assume that libtpu is installed only if the machine  # 我们假设只有当机器有TPU时才会安装libtpu
        # has TPUs.

        import libtpu  # noqa: F401  # 尝试导入libtpu库

        logger.debug("Confirmed TPU platform is available.")  # 确认TPU平台可用
        return "vllm.platforms.tpu.TpuPlatform"  # 返回vLLM内置的TPU平台类路径
    except Exception as e:  # 捕获所有异常
        logger.debug("TPU platform is not available because: %s", str(e))  # 记录TPU不可用的原因
        return None  # 返回None表示TPU不可用


def cuda_platform_plugin() -> str | None:
    """
    检测CUDA平台是否可用的插件函数。

    返回:
        如果CUDA可用则返回CUDA平台类的全限定名，否则返回None
    """
    is_cuda = False  # 初始化CUDA可用标志为False
    logger.debug("Checking if CUDA platform is available.")  # 调试日志：检查CUDA平台
    try:
        from vllm.utils.import_utils import import_pynvml  # 延迟导入pynvml工具函数

        pynvml = import_pynvml()  # 导入pynvml库
        pynvml.nvmlInit()  # 初始化NVML库
        try:
            # NOTE: Edge case: vllm cpu build on a GPU machine.  # 注意：边缘情况：GPU机器上的vllm CPU构建
            # Third-party pynvml can be imported in cpu build,  # 第三方pynvml可以在CPU构建中导入
            # we need to check if vllm is built with cpu too.  # 我们还需要检查vllm是否是CPU构建
            # Otherwise, vllm will always activate cuda plugin  # 否则vllm将始终在GPU机器上激活CUDA插件
            # on a GPU machine, even if in a cpu build.  # 即使是在CPU构建中
            is_cuda = (  # 判断是否为CUDA平台
                pynvml.nvmlDeviceGetCount() > 0  # GPU设备数量大于0
                and not vllm_version_matches_substr("cpu")  # 且vLLM不是CPU构建版本
            )
            if pynvml.nvmlDeviceGetCount() <= 0:  # 如果没有找到GPU设备
                logger.debug("CUDA platform is not available because no GPU is found.")  # 记录未找到GPU
            if vllm_version_matches_substr("cpu"):  # 如果是CPU构建版本
                logger.debug(  # 记录因CPU构建而不可用
                    "CUDA platform is not available because vLLM is built with CPU."
                )
            if is_cuda:  # 如果确认是CUDA平台
                logger.debug("Confirmed CUDA platform is available.")  # 确认CUDA可用
        finally:
            pynvml.nvmlShutdown()  # 关闭NVML库，释放资源
    except Exception as e:  # 捕获所有异常
        logger.debug("Exception happens when checking CUDA platform: %s", str(e))  # 记录检查异常
        if "nvml" not in e.__class__.__name__.lower():  # 如果错误与NVML无关
            # If the error is not related to NVML, re-raise it.  # 如果不是NVML相关错误，重新抛出
            raise e

        # CUDA is supported on Jetson, but NVML may not be.  # Jetson支持CUDA但可能不支持NVML
        import os  # 导入os模块

        def cuda_is_jetson() -> bool:
            """检测当前设备是否为NVIDIA Jetson平台。"""
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(  # 检查Jetson特有的文件路径
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():  # 如果是Jetson平台
            logger.debug("Confirmed CUDA platform is available on Jetson.")  # 确认Jetson上CUDA可用
            is_cuda = True  # 设置CUDA可用标志
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))  # 记录CUDA不可用原因

    return "vllm.platforms.cuda.CudaPlatform" if is_cuda else None  # 返回CUDA平台类路径或None


def rocm_platform_plugin() -> str | None:
    """
    检测ROCm平台是否可用的插件函数。

    返回:
        如果ROCm可用则返回ROCm平台类的全限定名，否则返回None
    """
    is_rocm = False  # 初始化ROCm可用标志为False
    logger.debug("Checking if ROCm platform is available.")  # 调试日志：检查ROCm平台
    try:
        import amdsmi  # 导入AMD系统管理接口库

        amdsmi.amdsmi_init()  # 初始化AMDSMI
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:  # 如果找到AMD处理器
                is_rocm = True  # 设置ROCm可用标志
                logger.debug("Confirmed ROCm platform is available.")  # 确认ROCm可用
            else:
                logger.debug("ROCm platform is not available because no GPU is found.")  # 未找到AMD GPU
        finally:
            amdsmi.amdsmi_shut_down()  # 关闭AMDSMI，释放资源
    except Exception as e:  # 捕获所有异常
        logger.debug("ROCm platform is not available because: %s", str(e))  # 记录ROCm不可用原因

    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None  # 返回ROCm平台类路径或None


def xpu_platform_plugin() -> str | None:
    """
    检测Intel XPU平台是否可用的插件函数。

    返回:
        如果XPU可用则返回XPU平台类的全限定名，否则返回None
    """
    is_xpu = False  # 初始化XPU可用标志为False
    logger.debug("Checking if XPU platform is available.")  # 调试日志：检查XPU平台
    try:
        import torch  # 导入PyTorch

        if supports_xccl():  # 如果支持XCCL通信后端
            dist_backend = "xccl"  # 设置分布式通信后端为xccl
            from vllm.platforms.xpu import XPUPlatform  # 导入XPU平台类

            XPUPlatform.dist_backend = dist_backend  # 设置XPU平台的分布式后端
            logger.debug("Confirmed %s backend is available.", XPUPlatform.dist_backend)  # 确认通信后端可用

        if hasattr(torch, "xpu") and torch.xpu.is_available():  # 如果PyTorch支持XPU且XPU设备可用
            is_xpu = True  # 设置XPU可用标志
            logger.debug("Confirmed XPU platform is available.")  # 确认XPU可用
    except Exception as e:  # 捕获所有异常
        logger.debug("XPU platform is not available because: %s", str(e))  # 记录XPU不可用原因

    return "vllm.platforms.xpu.XPUPlatform" if is_xpu else None  # 返回XPU平台类路径或None


def cpu_platform_plugin() -> str | None:
    """
    检测CPU平台是否可用的插件函数。

    返回:
        如果CPU平台可用则返回CPU平台类的全限定名，否则返回None
    """
    is_cpu = False  # 初始化CPU可用标志为False
    logger.debug("Checking if CPU platform is available.")  # 调试日志：检查CPU平台
    try:
        is_cpu = vllm_version_matches_substr("cpu")  # 检查vLLM版本是否为CPU构建
        if is_cpu:  # 如果是CPU构建
            logger.debug(  # 记录因CPU构建而确认CPU平台可用
                "Confirmed CPU platform is available because vLLM is built with CPU."
            )
        if not is_cpu:  # 如果不是CPU构建
            import sys  # 导入sys模块

            is_cpu = sys.platform.startswith("darwin")  # 检查是否在macOS上运行
            if is_cpu:  # 如果是macOS
                logger.debug(  # 记录因macOS而确认CPU平台可用
                    "Confirmed CPU platform is available because the machine is MacOS."
                )

    except Exception as e:  # 捕获所有异常
        logger.debug("CPU platform is not available because: %s", str(e))  # 记录CPU不可用原因

    return "vllm.platforms.cpu.CpuPlatform" if is_cpu else None  # 返回CPU平台类路径或None


builtin_platform_plugins = {  # 内置平台插件字典，映射平台名称到检测函数
    "tpu": tpu_platform_plugin,  # TPU平台插件
    "cuda": cuda_platform_plugin,  # CUDA平台插件
    "rocm": rocm_platform_plugin,  # ROCm平台插件
    "xpu": xpu_platform_plugin,  # XPU平台插件
    "cpu": cpu_platform_plugin,  # CPU平台插件
}


def resolve_current_platform_cls_qualname() -> str:
    """
    解析当前平台类的全限定名。

    通过检测所有内置和外部平台插件，确定当前运行环境的硬件平台，
    并返回对应平台类的全限定名。

    返回:
        当前平台类的全限定名字符串
    """
    platform_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)  # 加载所有外部平台插件

    activated_plugins = []  # 已激活的插件列表

    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):  # 遍历所有内置和外部插件
        try:
            assert callable(func)  # 确保插件函数可调用
            platform_cls_qualname = func()  # 调用插件检测函数
            if platform_cls_qualname is not None:  # 如果检测到平台可用
                activated_plugins.append(name)  # 添加到已激活列表
        except Exception:  # 捕获所有异常，静默处理
            pass

    activated_builtin_plugins = list(  # 获取已激活的内置插件列表
        set(activated_plugins) & set(builtin_platform_plugins.keys())
    )
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))  # 获取已激活的外部插件列表

    if len(activated_oot_plugins) >= 2:  # 如果有多个外部插件被激活
        raise RuntimeError(  # 抛出运行时错误
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}"
        )
    elif len(activated_oot_plugins) == 1:  # 如果只有一个外部插件被激活
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()  # 获取该插件的平台类路径
        logger.info("Platform plugin %s is activated", activated_oot_plugins[0])  # 记录激活的平台插件
    elif len(activated_builtin_plugins) >= 2:  # 如果有多个内置插件被激活
        raise RuntimeError(  # 抛出运行时错误
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}"
        )
    elif len(activated_builtin_plugins) == 1:  # 如果只有一个内置插件被激活
        platform_cls_qualname = builtin_platform_plugins[activated_builtin_plugins[0]]()  # 获取该插件的平台类路径
        logger.debug(  # 调试日志：记录自动检测到的平台
            "Automatically detected platform %s.", activated_builtin_plugins[0]
        )
    else:  # 如果没有检测到任何平台
        platform_cls_qualname = "vllm.platforms.interface.UnspecifiedPlatform"  # 使用未指定平台类
        logger.debug("No platform detected, vLLM is running on UnspecifiedPlatform")  # 记录未检测到平台
    return platform_cls_qualname  # 返回平台类全限定名


_current_platform = None  # 当前平台实例的全局变量，初始为None
_init_trace: str = ""  # 平台初始化的堆栈追踪信息

if TYPE_CHECKING:  # 仅在类型检查时
    current_platform: Platform  # 声明current_platform的类型为Platform


def __getattr__(name: str):
    """
    模块级别的__getattr__方法，用于惰性初始化current_platform。

    参数:
        name: 请求的属性名称

    返回:
        请求的属性值

    说明:
        1. 外部平台插件需要 `from vllm.platforms import Platform` 来继承Platform类，
           因此不能在导入 `vllm.platforms` 时就解析 `current_platform`。
        2. 当用户使用外部平台插件时，某些vllm内部代码可能在导入时访问 `current_platform`，
           需要确保 `current_platform` 在插件加载后才解析。
    """
    if name == "current_platform":  # 如果请求的是current_platform属性
        # lazy init current_platform.  # 惰性初始化current_platform
        # 1. out-of-tree platform plugins need `from vllm.platforms import  # 外部平台插件需要导入Platform类
        #    Platform` so that they can inherit `Platform` class. Therefore,  # 因此不能在导入时解析current_platform
        #    we cannot resolve `current_platform` during the import of
        #    `vllm.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run  # 用户使用外部插件时可能会运行import vllm
        #    `import vllm`, some vllm internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform  # 声明使用全局变量
        if _current_platform is None:  # 如果当前平台尚未初始化
            platform_cls_qualname = resolve_current_platform_cls_qualname()  # 解析平台类全限定名
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()  # 实例化平台对象
            global _init_trace  # 声明使用全局变量
            _init_trace = "".join(traceback.format_stack())  # 记录初始化时的堆栈追踪
        return _current_platform  # 返回当前平台实例
    elif name in globals():  # 如果属性存在于全局命名空间
        return globals()[name]  # 返回该全局变量
    else:  # 如果属性不存在
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")  # 抛出属性错误


def __setattr__(name: str, value):
    """
    模块级别的__setattr__方法，用于设置模块属性。

    参数:
        name: 要设置的属性名称
        value: 要设置的值
    """
    if name == "current_platform":  # 如果设置的是current_platform
        global _current_platform  # 声明使用全局变量
        _current_platform = value  # 更新当前平台实例
    elif name in globals():  # 如果属性存在于全局命名空间
        globals()[name] = value  # 更新全局变量
    else:  # 如果属性不存在
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")  # 抛出属性错误


__all__ = ["Platform", "PlatformEnum", "current_platform", "CpuArchEnum", "_init_trace"]  # 模块公开的符号列表
