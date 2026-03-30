# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict  # 导入有序字典，用于实现 LRU 缓存
from collections.abc import Iterable  # 导入可迭代类型

from vllm.v1.core.kv_cache_utils import BlockHash  # 导入块哈希类型
from vllm.v1.kv_offload.abstract import (  # 导入卸载相关的抽象类和数据类
    LoadStoreSpec,  # 加载/存储规格
    OffloadingEvent,  # 卸载事件
    OffloadingManager,  # 卸载管理器抽象基类
    PrepareStoreOutput,  # 准备存储操作的输出
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus  # 导入后端基类和块状态类


class LRUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
    基于 LRU（最近最少使用）策略的卸载管理器，支持可插拔后端，按 LRU 顺序驱逐块。
    """

    def __init__(self, backend: Backend, enable_events: bool = False):  # 初始化 LRU 卸载管理器
        """初始化 LRU 卸载管理器，设置后端和事件记录。"""
        self.backend: Backend = backend  # 存储后端实例
        # block_hash -> BlockStatus
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()  # 有序字典，存储块哈希到块状态的映射，用于 LRU 排序
        self.events: list[OffloadingEvent] | None = [] if enable_events else None  # 事件列表，启用时记录卸载事件

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:  # 查找已卸载的连续块数量
        """查找从第一个块开始的最大连续已卸载块数量。"""
        hit_count = 0  # 命中计数器
        for block_hash in block_hashes:  # 遍历块哈希
            block = self.blocks.get(block_hash)  # 查找块状态
            if block is None or not block.is_ready:  # 如果块不存在或未就绪
                break  # 中断查找
            hit_count += 1  # 命中计数加一
        return hit_count  # 返回命中数量

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:  # 准备加载指定块
        """准备加载指定块，增加引用计数以防止被驱逐。"""
        blocks = []  # 初始化块列表
        for block_hash in block_hashes:  # 遍历块哈希
            block = self.blocks[block_hash]  # 获取块状态
            assert block.is_ready  # 断言块已就绪
            block.ref_cnt += 1  # 增加引用计数，防止被驱逐
            blocks.append(block)  # 添加到块列表

        return self.backend.get_load_store_spec(block_hashes, blocks)  # 返回加载/存储规格

    def touch(self, block_hashes: Iterable[BlockHash]):  # 将指定块标记为最近使用
        """将指定块移到有序字典末尾，标记为最近使用。"""
        for block_hash in reversed(list(block_hashes)):  # 逆序遍历块哈希
            if self.blocks.get(block_hash):  # 如果块存在
                self.blocks.move_to_end(block_hash)  # 移到有序字典末尾（标记为最近使用）

    def complete_load(self, block_hashes: Iterable[BlockHash]):  # 完成加载操作
        """完成加载操作，减少引用计数。"""
        for block_hash in block_hashes:  # 遍历块哈希
            block = self.blocks[block_hash]  # 获取块状态
            assert block.ref_cnt > 0  # 断言引用计数大于 0
            block.ref_cnt -= 1  # 减少引用计数

    def prepare_store(  # 准备存储指定块
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """准备存储指定块到后端，必要时驱逐 LRU 块以腾出空间。"""
        block_hashes_list = list(block_hashes)  # 将可迭代对象转为列表

        # filter out blocks that are already stored
        block_hashes_to_store = [  # 过滤掉已存储的块
            block_hash  # 块哈希
            for block_hash in block_hashes_list  # 遍历所有块哈希
            if block_hash not in self.blocks  # 仅保留尚未存储的块
        ]

        num_blocks_to_evict = (  # 计算需要驱逐的块数
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()  # 需要存储的块数减去可用空闲块数
        )

        # build list of blocks to evict
        to_evict = []  # 初始化待驱逐块列表
        if num_blocks_to_evict > 0:  # 如果需要驱逐块
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(block_hashes_list)  # 原始输入中的块不能被驱逐
            for block_hash, block in self.blocks.items():  # 按 LRU 顺序遍历块（最旧的在前）
                if block.ref_cnt == 0 and block_hash not in protected:  # 如果块无引用且不受保护
                    to_evict.append(block_hash)  # 加入驱逐列表
                    num_blocks_to_evict -= 1  # 减少需驱逐数量
                    if num_blocks_to_evict == 0:  # 如果已找到足够的块
                        break  # 停止遍历
            else:  # 遍历完毕仍未找到足够的块（for-else 语法）
                # we could not evict enough blocks
                return None  # 无法驱逐足够的块，返回 None

        # evict blocks
        for block_hash in to_evict:  # 遍历待驱逐块
            self.backend.free(self.blocks.pop(block_hash))  # 从映射中移除并释放后端资源

        if to_evict and self.events is not None:  # 如果有驱逐发生且事件记录已启用
            self.events.append(  # 记录驱逐事件
                OffloadingEvent(
                    block_hashes=to_evict,  # 被驱逐的块哈希
                    block_size=self.backend.block_size,  # 块大小
                    medium=self.backend.medium,  # 存储介质
                    removed=True,  # 标记为移除
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)  # 在后端分配新块
        assert len(blocks) == len(block_hashes_to_store)  # 断言分配的块数与请求数一致

        for block_hash, block in zip(block_hashes_to_store, blocks):  # 将新块添加到映射中
            self.blocks[block_hash] = block  # 建立块哈希到块状态的映射

        # build store specs for allocated blocks
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)  # 获取分配块的存储规格

        return PrepareStoreOutput(  # 返回准备存储的输出结果
            block_hashes_to_store=block_hashes_to_store,  # 需要存储的块哈希
            store_spec=store_spec,  # 存储规格
            block_hashes_evicted=to_evict,  # 被驱逐的块哈希
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):  # 完成存储操作
        """完成存储操作。成功时将块标记为就绪，失败时释放块。"""
        stored_block_hashes: list[BlockHash] = []  # 初始化成功存储的块哈希列表
        if success:  # 如果存储成功
            for block_hash in block_hashes:  # 遍历块哈希
                block = self.blocks[block_hash]  # 获取块状态
                if not block.is_ready:  # 如果块尚未就绪
                    block.ref_cnt = 0  # 将 ref_cnt 设为 0，标记为就绪
                    stored_block_hashes.append(block_hash)  # 加入成功存储列表
        else:  # 如果存储失败
            for block_hash in block_hashes:  # 遍历块哈希
                block = self.blocks[block_hash]  # 获取块状态
                if not block.is_ready:  # 如果块尚未就绪
                    self.backend.free(block)  # 释放后端资源
                    del self.blocks[block_hash]  # 从映射中删除

        if stored_block_hashes and self.events is not None:  # 如果有成功存储的块且事件记录已启用
            self.events.append(  # 记录存储事件
                OffloadingEvent(
                    block_hashes=stored_block_hashes,  # 成功存储的块哈希
                    block_size=self.backend.block_size,  # 块大小
                    medium=self.backend.medium,  # 存储介质
                    removed=False,  # 标记为存储（非移除）
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:  # 获取并清除卸载事件
        """获取自上次调用以来的所有卸载事件，并清空事件列表。"""
        if self.events is not None:  # 如果事件记录已启用
            yield from self.events  # 生成所有事件
            self.events.clear()  # 清空事件列表
