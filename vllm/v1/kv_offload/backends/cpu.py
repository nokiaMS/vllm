# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes  # 导入 C 类型库，用于定义与 C 兼容的数据结构
from collections.abc import Iterable  # 导入可迭代类型

from vllm.v1.core.kv_cache_utils import BlockHash  # 导入块哈希类型
from vllm.v1.kv_offload.abstract import LoadStoreSpec  # 导入加载/存储规格抽象类
from vllm.v1.kv_offload.backend import Backend, BlockStatus  # 导入后端基类和块状态类
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec  # 导入 CPU 加载/存储规格


class CPUBlockStatus(BlockStatus):
    """CPU 块状态类，继承自 BlockStatus，增加了 block_id 字段用于标识 CPU 端的块编号。"""

    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]  # type: ignore  # 在父类字段基础上增加块 ID 字段，64 位整数

    def __init__(self, block_id: int):  # 初始化 CPU 块状态
        """初始化 CPU 块状态，设置块 ID。"""
        super().__init__()  # 调用父类初始化，ref_cnt 设为 -1
        self.block_id = block_id  # 设置块 ID


class CPUBackend(Backend):
    """CPU 后端类，管理 CPU 端 KV 缓存块的分配和释放。"""

    def __init__(self, block_size: int, num_blocks: int):  # 初始化 CPU 后端
        """初始化 CPU 后端，设置块大小和总块数。"""
        super().__init__(block_size=block_size, medium=CPULoadStoreSpec.medium())  # 调用父类初始化，设置介质为 CPU

        self.num_blocks: int = num_blocks  # CPU 端可用的总块数
        self.num_allocated_blocks: int = 0  # 当前已分配的块数量
        self.allocated_blocks_free_list: list[int] = []  # 已释放块的空闲列表，存储可复用的块 ID

    def get_num_free_blocks(self):  # 获取当前可用的空闲块数量
        """获取当前可用的空闲块数量，包括空闲列表中的块和尚未分配过的块。"""
        return (  # 返回空闲块总数
            len(self.allocated_blocks_free_list)  # 空闲列表中已回收的块数
            + self.num_blocks  # 总块数
            - self.num_allocated_blocks  # 减去已分配的块数
        )

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:  # 分配指定数量的块
        """分配指定数量的块，优先分配新块，不够时从空闲列表中复用。"""
        num_fresh_blocks = min(  # 计算需要全新分配的块数
            len(block_hashes), self.num_blocks - self.num_allocated_blocks  # 取请求数和剩余新块数的较小值
        )
        num_reused_blocks = len(block_hashes) - num_fresh_blocks  # 需要从空闲列表中复用的块数
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks  # 断言空闲列表中有足够的块可复用

        # allocate fresh blocks
        blocks: list[BlockStatus] = []  # 初始化分配结果列表
        for _ in range(num_fresh_blocks):  # 遍历分配新块
            blocks.append(CPUBlockStatus(self.num_allocated_blocks))  # 创建新的 CPU 块状态，使用递增的块 ID
            self.num_allocated_blocks += 1  # 递增已分配块计数

        # allocate reused blocks
        for _ in range(num_reused_blocks):  # 遍历复用空闲块
            block_id = self.allocated_blocks_free_list.pop()  # 从空闲列表中取出一个块 ID
            blocks.append(CPUBlockStatus(block_id))  # 创建复用的 CPU 块状态

        return blocks  # 返回分配的块列表

    def free(self, block: BlockStatus):  # 释放一个已分配的块
        """释放一个已分配的块，将其块 ID 加入空闲列表。"""
        assert isinstance(block, CPUBlockStatus)  # 断言块类型为 CPUBlockStatus
        self.allocated_blocks_free_list.append(block.block_id)  # 将块 ID 加入空闲列表

    def get_load_store_spec(  # 获取 CPU 端的加载/存储规格
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        """获取 CPU 端的加载/存储规格，包含所有块的 block_id 列表。"""
        return CPULoadStoreSpec([block.block_id for block in blocks])  # 创建 CPULoadStoreSpec，提取所有块的 ID
