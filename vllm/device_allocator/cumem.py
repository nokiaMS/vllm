# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# 基于cumem的PyTorch可插拔分配器，用于实现休眠模式。
# other approaches tried but failed:
# 其他尝试过但失败的方法：
# - cuda-python package binding
# - cuda-python包绑定
# - custom libcuda driver ctypes wrapper
# - 自定义libcuda驱动ctypes包装器
# both of them failed because of cuda context mismatch.
# 两种方法都因CUDA上下文不匹配而失败。
# not sure why, they are created from a different context.
# 原因不明，它们是从不同的上下文创建的。
# the only successful approach is to call cuda driver API in C.
# 唯一成功的方法是在C中调用CUDA驱动API。
import dataclasses  # 导入dataclasses模块，用于数据类装饰器
import gc  # 导入gc模块，用于垃圾回收
import os  # 导入os模块，用于操作系统相关功能
from collections.abc import Callable, Iterator  # 从collections.abc导入Callable和Iterator类型
from contextlib import contextmanager  # 导入contextmanager装饰器，用于创建上下文管理器
from typing import Any  # 导入Any类型注解

import torch  # 导入PyTorch库

from vllm.logger import init_logger  # 从vllm.logger导入日志初始化函数
from vllm.utils.platform_utils import is_pin_memory_available  # 导入检测锁页内存是否可用的工具函数
from vllm.utils.system_utils import find_loaded_library  # 导入查找已加载库的工具函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


cumem_available = False  # 标记cumem分配器是否可用，默认为False
libcudart: Any = None  # CUDA运行时库的引用，初始为None
try:
    from vllm.cumem_allocator import (  # 尝试从cumem_allocator模块导入核心函数
        init_module,  # 初始化模块函数
        python_create_and_map,  # Python层创建并映射内存函数
        python_unmap_and_release,  # Python层取消映射并释放内存函数
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary  # 导入CUDA运行时库包装类

    lib_name = find_loaded_library("cumem_allocator")  # 查找已加载的cumem_allocator库路径
    libcudart = CudaRTLibrary()  # 创建CUDA运行时库实例
    cumem_available = True  # 标记cumem分配器可用
except ModuleNotFoundError:
    # only cuda and rocm platforms support cumem allocator
    # 只有CUDA和ROCm平台支持cumem分配器
    init_module = None  # 设置init_module为None
    python_create_and_map = None  # 设置python_create_and_map为None
    python_unmap_and_release = None  # 设置python_unmap_and_release为None
    lib_name = None  # 设置lib_name为None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
# 分别表示：设备ID、对齐后的大小、设备内存指针、内存句柄指针
HandleType = tuple[int, int, int, int]  # 定义内存分配句柄的类型别名


@dataclasses.dataclass
class AllocationData:
    """内存分配数据类，存储单次内存分配的相关信息。"""

    handle: HandleType  # 内存分配句柄，包含设备、大小、内存指针和句柄指针
    tag: str  # 内存分配的标签，用于分类管理
    cpu_backup_tensor: torch.Tensor | None = None  # CPU备份张量，休眠时用于保存GPU数据


def create_and_map(allocation_handle: HandleType) -> None:
    """创建并映射内存分配。

    根据给定的分配句柄，在GPU上创建并映射物理内存。

    Args:
        allocation_handle: 内存分配句柄，包含设备、大小、内存指针和句柄指针。
    """
    python_create_and_map(*allocation_handle)  # 调用底层C扩展函数创建并映射内存


def unmap_and_release(allocation_handle: HandleType) -> None:
    """取消映射并释放内存分配。

    根据给定的分配句柄，取消GPU上的内存映射并释放物理内存。

    Args:
        allocation_handle: 内存分配句柄，包含设备、大小、内存指针和句柄指针。
    """
    python_unmap_and_release(*allocation_handle)  # 调用底层C扩展函数取消映射并释放内存


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],  # Python层内存分配回调函数
    python_free_func: Callable[[int], HandleType],  # Python层内存释放回调函数
) -> torch.cuda.memory.CUDAPluggableAllocator:
    """获取CUDA可插拔分配器实例。

    初始化模块并创建一个CUDA可插拔分配器。

    Args:
        python_malloc_fn: 内存分配时调用的Python回调函数。
        python_free_func: 内存释放时调用的Python回调函数。

    Returns:
        CUDA可插拔分配器实例。
    """
    init_module(python_malloc_fn, python_free_func)  # 使用回调函数初始化C扩展模块
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(  # 创建CUDA可插拔分配器
        lib_name, "my_malloc", "my_free"  # 指定库名称和分配/释放函数名
    )
    return new_alloc  # 返回创建的分配器实例


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],  # Python层内存分配回调函数
    python_free_func: Callable[[int], HandleType],  # Python层内存释放回调函数
) -> Iterator[
    tuple[torch.cuda.memory.MemPool, torch.cuda.memory.CUDAPluggableAllocator]
]:
    """使用自定义分配器创建并切换到内存池的上下文管理器。

    在上下文中，所有CUDA内存分配都将通过自定义分配器进行。

    Args:
        python_malloc_fn: 内存分配时调用的Python回调函数。
        python_free_func: 内存释放时调用的Python回调函数。

    Yields:
        包含内存池和CUDA可插拔分配器的元组。
    """
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)  # 获取可插拔分配器
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)  # 基于分配器创建内存池
    with torch.cuda.memory.use_mem_pool(mem_pool):  # 切换到该内存池
        yield mem_pool, new_alloc  # 返回内存池和分配器供调用方使用


class CuMemAllocator:
    """CuMem内存分配器，用于管理CUDA张量的内存池。

    这是一个单例类，管理CUDA张量的内存池。该内存池中的内存可以在分配器休眠时
    被卸载到CPU或丢弃。

    在 `use_memory_pool(tag)` 上下文中创建的所有张量都将在内存池中分配，
    并具有与传入上下文的tag相同的标签。

    当调用 `sleep` 时，具有指定标签的所有张量将被卸载到CPU内存，
    其余张量将被丢弃。当调用 `wake_up` 时，之前被卸载的所有张量
    将被加载回GPU内存，其余张量将拥有空的内存。

    为什么需要单例模式？
    当已分配的张量被垃圾回收时，PyTorch会调用释放回调函数，
    该函数会调用 `python_free_callback` 方法。C扩展使用全局变量
    来存储此类实例的函数。如果创建多个实例，全局变量将被覆盖，
    释放回调将无法正常工作。
    """

    instance: "CuMemAllocator | None" = None  # 单例实例，初始为None
    default_tag: str = "default"  # 默认内存分配标签

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """获取CuMemAllocator的单例实例。

        CuMemAllocator是单例类，不能直接调用构造函数。
        必须通过此方法获取实例。

        Returns:
            CuMemAllocator的唯一实例。

        Raises:
            AssertionError: 如果cumem分配器不可用。
        """
        assert cumem_available, "cumem allocator is not available"  # 断言cumem分配器可用
        if CuMemAllocator.instance is None:  # 如果实例尚未创建
            CuMemAllocator.instance = CuMemAllocator()  # 创建新实例
        return CuMemAllocator.instance  # 返回单例实例

    def __init__(self):
        """初始化CuMemAllocator实例。

        检查环境配置兼容性，并初始化内部数据结构。

        Raises:
            AssertionError: 如果PYTORCH_CUDA_ALLOC_CONF中包含expandable_segments:True。
        """
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")  # 获取PyTorch CUDA分配器配置
        assert "expandable_segments:True" not in conf, (  # 断言未启用可扩展段，因为与内存池不兼容
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}  # 内存指针到分配数据的映射字典
        self.current_tag: str = CuMemAllocator.default_tag  # 当前内存分配标签
        self.allocator_and_pools: dict[str, Any] = {}  # 分配器和内存池的引用字典
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # 在此处创建两个回调函数的强引用，防止这些临时绑定方法对象被垃圾回收。
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback  # 保存内存分配回调的强引用
        self.python_free_callback = self._python_free_callback  # 保存内存释放回调的强引用

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """内存分配回调函数，在内存池中分配内存时被调用。

        将分配数据存储到内部字典中，以便后续管理。

        Args:
            allocation_handle: 内存分配句柄，包含设备、大小、内存指针和句柄指针。
        """
        py_d_mem = allocation_handle[2]  # 获取设备内存指针
        self.pointer_to_data[py_d_mem] = AllocationData(  # 将分配数据存入字典
            allocation_handle, self.current_tag  # 记录句柄和当前标签
        )
        logger.debug(  # 记录调试日志
            "Allocated %s bytes for %s with address %s from cumem allocator",  # 日志格式：分配字节数、标签、地址
            allocation_handle[1],  # 分配的字节数
            self.current_tag,  # 当前标签
            py_d_mem,  # 设备内存地址
        )
        return  # 返回

    def _python_free_callback(self, ptr: int) -> HandleType:
        """内存释放回调函数，在内存池中释放内存时被调用。

        从内部字典中查找并移除分配数据。

        Args:
            ptr: 要释放的设备内存指针。

        Returns:
            对应的内存分配句柄。
        """
        data = self.pointer_to_data.pop(ptr)  # 从字典中弹出并获取分配数据
        if data.cpu_backup_tensor is not None:  # 如果存在CPU备份张量
            data.cpu_backup_tensor = None  # 清除CPU备份张量引用
        logger.debug(  # 记录调试日志
            "Freed %s bytes for %s with address %s from cumem allocator",  # 日志格式：释放字节数、标签、地址
            data.handle[1],  # 释放的字节数
            data.tag,  # 分配标签
            ptr,  # 设备内存地址
        )
        return data.handle  # 返回内存分配句柄

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """将分配器置入休眠模式。

        指定标签的内存数据将被卸载到CPU内存，其余数据将被丢弃。

        Args:
            offload_tags: 需要卸载的内存分配标签。可以是元组、字符串或None。
                         如果为None，则使用默认标签。其余的内存分配将被丢弃。
        """
        if offload_tags is None:  # 如果未指定卸载标签
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            # 默认情况下，分配的张量在分配器休眠时会被卸载
            offload_tags = (CuMemAllocator.default_tag,)  # 使用默认标签
        elif isinstance(offload_tags, str):  # 如果传入的是字符串
            offload_tags = (offload_tags,)  # 转换为元组

        assert isinstance(offload_tags, tuple)  # 断言offload_tags是元组类型

        total_bytes = 0  # 总字节数计数器
        backup_bytes = 0  # 备份字节数计数器

        for ptr, data in self.pointer_to_data.items():  # 遍历所有内存分配
            handle = data.handle  # 获取内存分配句柄
            total_bytes += handle[1]  # 累加总字节数
            if data.tag in offload_tags:  # 如果该分配的标签在卸载标签列表中
                backup_bytes += handle[1]  # 累加备份字节数
                size_in_bytes = handle[1]  # 获取分配大小（字节）
                cpu_backup_tensor = torch.empty(  # 创建CPU上的备份张量
                    size_in_bytes,  # 张量大小
                    dtype=torch.uint8,  # 使用uint8类型以按字节存储
                    device="cpu",  # 存储在CPU上
                    pin_memory=is_pin_memory_available(),  # 如果可用则使用锁页内存加速传输
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()  # 获取CPU张量的数据指针
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)  # 从GPU复制数据到CPU
                data.cpu_backup_tensor = cpu_backup_tensor  # 保存CPU备份张量引用
            unmap_and_release(handle)  # 取消映射并释放GPU内存

        logger.info(  # 记录信息日志
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "  # 日志：休眠释放的总内存
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "  # 日志：CPU备份量和丢弃量
            "directly.",
            total_bytes / 1024**3,  # 总字节数转换为GiB
            backup_bytes / 1024**3,  # 备份字节数转换为GiB
            (total_bytes - backup_bytes) / 1024**3,  # 丢弃字节数转换为GiB
        )

        gc.collect()  # 触发Python垃圾回收
        torch.cuda.empty_cache()  # 清空CUDA缓存

    def wake_up(self, tags: list[str] | None = None) -> None:
        """从休眠模式唤醒分配器。

        之前被卸载的数据将被加载回GPU内存，
        其余数据将拥有空的内存（已重新映射但无数据）。

        Args:
            tags: 需要加载回GPU内存的内存分配标签列表。
                  如果为None，则加载所有内存分配。
        """
        for ptr, data in self.pointer_to_data.items():  # 遍历所有内存分配
            if tags is None or data.tag in tags:  # 如果未指定标签或该分配的标签匹配
                handle = data.handle  # 获取内存分配句柄
                create_and_map(handle)  # 重新创建并映射GPU内存
                if data.cpu_backup_tensor is not None:  # 如果存在CPU备份张量
                    cpu_backup_tensor = data.cpu_backup_tensor  # 获取CPU备份张量
                    if cpu_backup_tensor is not None:  # 再次检查备份张量是否存在
                        size_in_bytes = (  # 计算备份数据的字节大小
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()  # 元素数量乘以每个元素的字节大小
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()  # 获取CPU张量的数据指针
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)  # 从CPU复制数据回GPU
                        data.cpu_backup_tensor = None  # 清除CPU备份张量引用，释放CPU内存

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """使用内存池的上下文管理器。

        在此上下文中创建的所有内存分配都将在内存池中分配，
        并具有指定的标签。

        Args:
            tag: 内存分配的标签。如果为None，将使用默认标签。

        Yields:
            None
        """
        if tag is None:  # 如果未指定标签
            tag = CuMemAllocator.default_tag  # 使用默认标签

        assert isinstance(tag, str)  # 断言标签是字符串类型

        old_tag = self.current_tag  # 保存当前标签
        self.current_tag = tag  # 设置新的当前标签
        with use_memory_pool_with_allocator(  # 使用自定义分配器的内存池上下文
            self.python_malloc_callback, self.python_free_callback  # 传入分配和释放回调函数
        ) as data:
            # start to hit another PyTorch bug in PyTorch 2.6,
            # 在PyTorch 2.6中开始遇到另一个PyTorch bug，
            # possibly because of gc-related issue w.r.t. the allocator and
            # 可能是因为与分配器和内存池相关的垃圾回收问题。
            # the memory pool.
            # to avoid the issue, we keep a reference of the data.
            # 为了避免该问题，我们保持对数据的引用。
            # see https://github.com/pytorch/pytorch/issues/146431 .
            self.allocator_and_pools[tag] = data  # 保存分配器和内存池的引用以防止被垃圾回收
            yield  # 让出控制权给调用方
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # PyTorch的bug，调用torch.cuda.empty_cache()会在使用可插拔分配器时报错，
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # 如果我们有一些内存被分配然后又被释放，
            # the memory will not be released, e.g. in online quantization,
            # 内存不会被释放，例如在在线量化中，
            # where the model is created in higher precision, and then
            # 模型以更高精度创建，然后
            # quantized in lower precision.
            # 以更低精度进行量化。
            # Find all unused allocations and manually release them.
            # 找到所有未使用的分配并手动释放它们。
            # TODO: we should expose `empty_cache` method in the memory pool.
            # TODO: 我们应该在内存池中暴露`empty_cache`方法。
            # TODO: ask for help from PyTorch team to expose this method.
            # TODO: 请求PyTorch团队帮助暴露此方法。
            allocations = data[0].snapshot()  # 获取内存池的快照，列出所有分配
            for allocation in allocations:  # 遍历所有分配
                if allocation["allocated_size"] == 0:  # 如果分配的大小为0（已释放但未回收）
                    handle = self._python_free_callback(allocation["address"])  # 调用释放回调获取句柄
                    unmap_and_release(handle)  # 取消映射并释放内存
            self.current_tag = old_tag  # 恢复之前的标签

    def get_current_usage(self) -> int:
        """获取内存池中当前已分配的总字节数。

        Returns:
            内存池中所有分配的总字节数。
        """
        sum_bytes: int = 0  # 总字节数累加器
        for ptr, data in self.pointer_to_data.items():  # 遍历所有内存分配
            handle = data.handle  # 获取内存分配句柄
            sum_bytes += handle[1]  # 累加分配的字节数
        return sum_bytes  # 返回总字节数
