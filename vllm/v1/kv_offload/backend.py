# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes  # 导入 C 类型库，用于定义与 C 兼容的数据结构
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections.abc import Iterable  # 导入可迭代类型

from vllm.v1.core.kv_cache_utils import BlockHash  # 导入块哈希类型
from vllm.v1.kv_offload.abstract import LoadStoreSpec  # 导入加载/存储规格抽象类


class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:
    单个 KV 数据块的卸载状态。包含以下信息：

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    ref_cnt - 当前使用此块作为源的传输数量。值为 -1 表示块尚未准备好被读取。
    load_store_spec - backend-specific information on how to actually
        read/write the block.
    load_store_spec - 后端特定的信息，描述如何实际读/写此块。
    """

    _fields_ = [("ref_cnt", ctypes.c_int32)]  # 定义 C 结构体字段：引用计数，32 位整数

    def __init__(self):  # 初始化块状态
        super().__init__()  # 调用父类初始化
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1  # 初始化块为"未就绪"状态（ref_cnt = -1）

    @property  # 属性装饰器
    def is_ready(self) -> bool:  # 检查块是否已准备好被读取
        """
        Returns whether the block is ready to be read.
        返回块是否已准备好被读取。
        """
        return self.ref_cnt >= 0  # ref_cnt >= 0 表示块已就绪


class Backend(ABC):
    """
    An abstract class for allocating and returning specs for writing
    KV blocks to some backend.
    抽象类，用于分配和返回将 KV 块写入某个后端的规格。
    """

    def __init__(self, block_size: int, medium: str):  # 初始化后端
        self.block_size = block_size  # 块大小
        self.medium = medium  # 存储介质类型

    @abstractmethod  # 抽象方法装饰器
    def get_num_free_blocks(self):  # 获取当前可分配的空闲块数量
        """
        Returns the number of current number of blocks that can be allocated.
        返回当前可分配的块数量。
        """
        pass  # 子类必须实现此方法

    @abstractmethod  # 抽象方法装饰器
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:  # 分配块空间
        """
        Allocate space for writing blocks.
        This method assumes there is enough space for allocation.
        It is unsafe to use without checking get_num_free_blocks beforehand.
        为写入块分配空间。此方法假设有足够的空间用于分配。
        在不先检查 get_num_free_blocks 的情况下使用是不安全的。

        Args:
            block_hashes: the hashes identifying the blocks to be written.
            block_hashes: 标识要写入的块的哈希值。

        Returns:
            A list of BlockStatus for the allocated blocks.
            The ref_cnt of each returned item will be -1, meaning the block
            is not yet ready to be read.
            返回已分配块的 BlockStatus 列表。每个返回项的 ref_cnt 为 -1，
            表示块尚未准备好被读取。
        """
        pass  # 子类必须实现此方法

    @abstractmethod  # 抽象方法装饰器
    def free(self, block: BlockStatus):  # 释放之前分配的块
        """
        Free a previously allocated block.
        You should only call this function with blocks returned by
        allocate_blocks, and only once per each block.
        释放之前分配的块。只应使用 allocate_blocks 返回的块调用此函数，
        且每个块只能调用一次。

        Args:
            block: The block to be freed.
            block: 要释放的块。
        """
        pass  # 子类必须实现此方法

    def get_load_store_spec(  # 获取后端特定的读写规格
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        """
        Get backend-specific information on how to read/write blocks.
        获取后端特定的块读写信息。

        Args:
            block_hashes: the list of block hashes identifying the blocks.
            block_hashes: 标识块的哈希值列表。
            blocks: the list of blocks.
            blocks: 块列表。

        Returns:
            A LoadStoreSpec that can be used by a worker
            to read/write the blocks.
            返回一个 LoadStoreSpec，工作线程可用其读写块。
        """
        raise NotImplementedError  # 默认抛出未实现异常，子类可覆写
