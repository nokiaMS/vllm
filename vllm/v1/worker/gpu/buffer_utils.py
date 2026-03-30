# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence  # 导入可迭代对象和序列抽象基类
from functools import partial  # 导入偏函数工具

import numpy as np  # 导入 NumPy 数值计算库
import torch  # 导入 PyTorch 深度学习框架

from vllm.triton_utils import tl, triton  # 导入 Triton JIT 编译工具
from vllm.utils.platform_utils import is_uva_available  # 导入 UVA 可用性检查函数
from vllm.utils.torch_utils import (  # 导入 PyTorch 工具函数
    async_tensor_h2d,  # 异步 CPU 到 GPU 张量传输
    get_accelerator_view_from_cpu_tensor,  # 从 CPU 张量获取加速器（GPU）视图
)


# 将 CPU 张量或 NumPy 数组异步拷贝到 GPU。
# 不使用显式 pin_memory() 以避免高并发下 CUDA 驱动争用导致的偶发卡顿。
def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,  # 输入的 CPU 张量或 NumPy 数组
    out: torch.Tensor | None = None,  # 可选的 GPU 输出张量
    device: torch.device | None = None,  # 目标设备
) -> torch.Tensor:
    if isinstance(x, np.ndarray):  # 如果输入是 NumPy 数组
        x = torch.from_numpy(x)  # 将 NumPy 数组转换为 PyTorch 张量
    assert x.is_cpu  # 断言输入张量在 CPU 上

    if out is None:  # 如果没有提供输出张量
        assert device is not None  # 断言必须指定目标设备
        out = torch.empty_like(x, device=device)  # 在目标设备上分配同形状的空张量

    # Copy directly to GPU — explicit pin_memory() causes sporadic stalls
    # under high concurrency due to CUDA driver contention. The driver
    # handles the transfer efficiently without manual pinning.
    return out.copy_(x, non_blocking=True)  # 非阻塞地将数据从 CPU 拷贝到 GPU


# UVA（统一虚拟地址）缓冲区：分配锁页 CPU 内存，并获取其 GPU 端可访问的视图。
# GPU 可以通过 UVA 直接读取 CPU 数据，无需显式的 H2D 拷贝，适用于小量数据的低延迟访问。
class UvaBuffer:
    # 初始化 UVA 缓冲区，分配锁页内存并创建 GPU 视图
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():  # 检查 UVA 是否可用
            raise RuntimeError("UVA is not available")  # 不可用则抛出运行时错误
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)  # 分配锁页 CPU 内存并初始化为零
        self.np = self.cpu.numpy()  # 创建 NumPy 视图以便高效的 CPU 端操作
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)  # 获取 GPU 可直接访问的 UVA 视图


# UVA 缓冲区池：维护多个 UVA 缓冲区实例并以轮转（round-robin）方式分配，
# 支持并发的 CPU->GPU 数据传输而不产生写冲突。
class UvaBufferPool:
    # 初始化 UVA 缓冲区池，创建多个缓冲区实例
    def __init__(
        self,
        size: int | Sequence[int],  # 每个缓冲区的大小
        dtype: torch.dtype,  # 数据类型
        max_concurrency: int = 2,  # 最大并发数（缓冲区数量）
    ):
        self.size = size  # 保存缓冲区大小
        self.dtype = dtype  # 保存数据类型
        self.max_concurrency = max_concurrency  # 保存最大并发数

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]  # 创建多个 UVA 缓冲区实例
        # Current buffer index
        self._curr = 0  # 初始化当前缓冲区索引为 0

    # 将数据拷贝到下一个 UVA 缓冲区（CPU 端写入），并返回 GPU 可访问的 UVA 视图。
    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency  # 轮转到下一个缓冲区
        buf = self._uva_bufs[self._curr]  # 获取当前缓冲区
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np  # 根据输入类型选择目标：张量用 cpu，其他用 numpy
        n = len(x)  # 获取数据长度
        dst[:n] = x  # 将数据拷贝到缓冲区的 CPU 端
        return buf.uva[:n]  # 返回对应长度的 UVA 视图

    # 先拷贝到 UVA 缓冲区，再从 UVA 拷贝到 GPU 显存（两阶段传输）。
    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,  # 输入数据
        out: torch.Tensor | None = None,  # 可选的 GPU 输出张量
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)  # 第一阶段：拷贝到 UVA 缓冲区
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)  # 第二阶段：从 UVA 拷贝到 GPU


# UVA 支撑的张量：CPU 端为数据源（支持 NumPy 操作），GPU 端通过 UVA 视图直接访问。
# 适用于需要在 CPU 上频繁更新、同时需要 GPU 读取的元数据（如块表中的块数量）。
class UvaBackedTensor:
    # 初始化 UVA 支撑张量，创建 CPU 数据源和 UVA 缓冲区池
    def __init__(
        self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int = 2
    ):
        self.dtype = dtype  # 保存数据类型

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)  # 分配 CPU 张量作为数据源（不锁页，因为 UVA 池有自己的锁页内存）
        self.np = self.cpu.numpy()  # 创建 NumPy 视图以便高效操作

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)  # 创建 UVA 缓冲区池
        self.gpu = self.pool.copy_to_uva(self.np)  # 初始化 GPU 端的 UVA 视图

    # 将 CPU 数据源拷贝到 UVA 缓冲区，更新 GPU 端视图
    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)  # 将 CPU 数据拷贝到新的 UVA 缓冲区并更新 GPU 视图
        return self.gpu  # 返回更新后的 GPU 视图


# 分阶段写入张量：支持在 CPU 端暂存多次稀疏写入操作（行索引+偏移+内容），
# 然后通过单次 Triton 内核调用批量应用到 GPU 张量上。
# 设计动机：避免每次写入都启动一个 CUDA 内核，减少内核启动开销。
# 支持 GPU 张量或 UVA 后端两种模式（后者用于大而不常访问的数据以节省显存）。
class StagedWriteTensor:
    # 初始化分阶段写入张量，分配 GPU 或 UVA 存储和暂存缓冲区
    def __init__(
        self,
        size: int | Sequence[int],  # 张量大小
        dtype: torch.dtype,  # 数据类型
        device: torch.device,  # 计算设备
        max_concurrency: int = 2,  # 最大并发数
        uva_instead_of_gpu: bool = False,  # 是否使用 UVA 替代 GPU 存储
    ):
        supported_dtypes = [torch.int32, torch.int64, torch.float32]  # 支持的数据类型列表
        if dtype not in supported_dtypes:  # 如果数据类型不在支持列表中
            raise ValueError(  # 抛出值错误
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]  # 获取行数
        self.dtype = dtype  # 保存数据类型
        self.device = device  # 保存计算设备
        self.max_concurrency = max_concurrency  # 保存最大并发数

        if not uva_instead_of_gpu:  # 如果使用 GPU 存储（默认）
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)  # 在 GPU 上分配零初始化张量
        else:  # 如果使用 UVA 存储
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)  # 创建 UVA 缓冲区
            self.gpu = self._uva_buf.uva  # 使用 UVA 视图作为 GPU 端张量

        self._staged_write_indices: list[int] = []  # 暂存的写入行索引列表
        self._staged_write_starts: list[int] = []  # 暂存的写入起始列位置列表
        self._staged_write_contents: list[int | float] = []  # 暂存的写入内容列表（扁平化）
        self._staged_write_cu_lens: list[int] = []  # 暂存的写入累积长度列表

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)  # 创建 UVA 缓冲区池的偏函数

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)  # 为写入索引分配 UVA 缓冲区池
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)  # 为写入起始位置分配 UVA 缓冲区池
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)  # 为累积长度分配 UVA 缓冲区池

    # 暂存一次写入操作：将内容 x 写入第 index 行、从 start 列开始的位置。
    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0  # 断言行索引非负
        assert start >= 0  # 断言起始位置非负
        if not x:  # 如果内容为空
            return  # 直接返回
        self._staged_write_indices.append(index)  # 记录写入的行索引
        self._staged_write_starts.append(start)  # 记录写入的起始列位置
        self._staged_write_contents.extend(x)  # 将写入内容追加到扁平列表
        self._staged_write_cu_lens.append(len(self._staged_write_contents))  # 记录当前累积内容长度

    # 暂存单个元素的写入操作（简化版，start 固定为 0）。
    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0  # 断言行索引非负
        self._staged_write_indices.append(index)  # 记录写入的行索引
        self._staged_write_starts.append(0)  # 起始位置为 0
        self._staged_write_contents.append(x)  # 追加单个元素到内容列表
        self._staged_write_cu_lens.append(len(self._staged_write_contents))  # 记录累积长度

    # 将所有暂存的写入操作通过 Triton 内核批量写入 GPU 张量，然后清空暂存区。
    def apply_write(self) -> None:
        n = len(self._staged_write_indices)  # 获取暂存的写入操作数量
        if n == 0:  # 如果没有暂存的写入
            return  # 直接返回

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)  # 将行索引拷贝到 UVA 缓冲区
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)  # 将起始位置拷贝到 UVA 缓冲区
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)  # 将累积长度拷贝到 UVA 缓冲区

        # Special handling for write_contents
        write_contents = async_tensor_h2d(  # 异步将写入内容从 CPU 传输到 GPU
            self._staged_write_contents, self.dtype, self.device, pin_memory=True
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](  # 启动 Triton 内核执行批量写入
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()  # 清空暂存区

    # 清空所有暂存的写入数据。
    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()  # 清空行索引列表
        self._staged_write_starts.clear()  # 清空起始位置列表
        self._staged_write_contents.clear()  # 清空内容列表
        self._staged_write_cu_lens.clear()  # 清空累积长度列表


# Triton 内核：将暂存的稀疏写入操作批量应用到 2D 输出张量上。
# 每个 program 处理一个写入操作，根据累积长度（cu_lens）定位内容在扁平缓冲区中的位置，
# 然后写入到 output[row_idx, start_idx:start_idx+content_len]。
@triton.jit  # Triton JIT 编译装饰器
def _apply_write_kernel(
    output_ptr,  # 输出张量的数据指针
    output_stride,  # 输出张量的行步幅
    write_indices_ptr,  # 写入行索引数组的指针
    write_starts_ptr,  # 写入起始位置数组的指针
    write_contents_ptr,  # 写入内容数组的指针
    write_cu_lens_ptr,  # 累积长度数组的指针
    BLOCK_SIZE: tl.constexpr,  # Triton 块大小常量
):
    pid = tl.program_id(0)  # 获取当前 program 的 ID（对应一个写入操作）
    row_idx = tl.load(write_indices_ptr + pid)  # 加载要写入的行索引
    start_idx = tl.load(write_starts_ptr + pid)  # 加载写入的起始列位置

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0  # 计算当前写入内容在扁平缓冲区中的起始位置
    cu_end = tl.load(write_cu_lens_ptr + pid)  # 获取当前写入内容的结束位置
    content_len = cu_end - cu_start  # 计算写入内容的长度

    for i in range(0, content_len, BLOCK_SIZE):  # 按块大小遍历写入内容
        block = i + tl.arange(0, BLOCK_SIZE)  # 计算块内偏移
        mask = block < content_len  # 创建有效元素掩码
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)  # 从扁平缓冲区加载内容
        tl.store(  # 将内容写入输出张量的指定位置
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )
