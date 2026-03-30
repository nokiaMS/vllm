# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reuse-frequency gating for CPU KV-cache offload stores.
CPU KV 缓存卸载存储的重用频率门控机制。

FilterReusedOffloadingManager — OffloadingManager decorator that skips
    storing blocks that have not yet been seen enough times.
FilterReusedOffloadingManager — OffloadingManager 装饰器，跳过尚未被看到
    足够多次的块的存储操作。
"""

from collections import OrderedDict  # 导入有序字典，用于实现 LRU 跟踪
from collections.abc import Iterable  # 导入可迭代类型

from vllm.v1.core.kv_cache_utils import BlockHash  # 导入块哈希类型
from vllm.v1.kv_offload.abstract import (  # 导入卸载相关的抽象类和数据类
    LoadStoreSpec,  # 加载/存储规格
    OffloadingEvent,  # 卸载事件
    OffloadingManager,  # 卸载管理器抽象基类
    PrepareStoreOutput,  # 准备存储操作的输出
)


class FilterReusedOffloadingManager(OffloadingManager):
    """An :class:`OffloadingManager` decorator that skips storing blocks
    whose reuse frequency is below *store_threshold*.
    一个 OffloadingManager 装饰器，跳过重用频率低于 store_threshold 的块的存储。

    All methods are delegated to the *backing* manager.  Two methods are
    intercepted:
    所有方法都委托给底层管理器。以下两个方法被拦截：

    * ``lookup`` — records each visited block hash in an internal LRU counter.
    * ``lookup`` — 在内部 LRU 计数器中记录每个访问的块哈希。
    * ``prepare_store`` — filters out block hashes that have not yet
      crossed the threshold *before* calling the backing
      ``prepare_store``.
    * ``prepare_store`` — 在调用底层 prepare_store 之前，过滤掉尚未达到阈值的块哈希。

    Args:
        backing: The underlying ``OffloadingManager`` to delegate to.
        backing: 要委托的底层 OffloadingManager。
        store_threshold: A block must be seen at least this many times in
            ``lookup()`` before it is eligible for offloading.  Must be >= 2
            (a value of 1 would be equivalent to no filtering).
        store_threshold: 块必须在 lookup() 中被看到至少这么多次才有资格被卸载。
            必须 >= 2（值为 1 等同于不过滤）。
        max_tracker_size: Maximum entries in the internal tracker's LRU table.
        max_tracker_size: 内部跟踪器 LRU 表中的最大条目数。
    """

    def __init__(  # 初始化重用过滤卸载管理器
        self,
        backing: OffloadingManager,  # 底层卸载管理器
        store_threshold: int = 2,  # 存储阈值，默认为 2
        max_tracker_size: int = 64_000,  # 跟踪器最大容量，默认 64000
    ):
        """初始化重用过滤卸载管理器，设置阈值和跟踪器参数。"""
        if store_threshold < 2:  # 检查阈值是否合法
            raise ValueError(  # 抛出值错误
                "FilterReusedOffloadingManager store_threshold must be >= 2, "
                f"got {store_threshold}"
            )
        if max_tracker_size < 1:  # 检查跟踪器大小是否合法
            raise ValueError(  # 抛出值错误
                "FilterReusedOffloadingManager max_tracker_size must be >= 1, "
                f"got {max_tracker_size}"
            )
        self._backing = backing  # 存储底层管理器引用
        self.store_threshold = store_threshold  # 存储阈值
        self.max_tracker_size = max_tracker_size  # 跟踪器最大容量
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[BlockHash, int] = OrderedDict()  # 有序字典，记录每个块哈希的访问计数，支持 O(1) LRU 驱逐

    # ------------------------------------------------------------------
    # Intercepted methods
    # 拦截的方法
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:  # 查找并记录块哈希访问
        """Record each hash, then delegate lookup to backing manager.
        记录每个哈希的访问次数，然后委托给底层管理器进行查找。"""
        block_hashes = list(block_hashes)  # 将可迭代对象转为列表
        for block_hash in block_hashes:  # 遍历块哈希
            if block_hash in self.counts:  # 如果块哈希已在计数器中
                self.counts.move_to_end(block_hash)  # 移到末尾（标记为最近使用）
                self.counts[block_hash] += 1  # 访问计数加一
            else:  # 如果块哈希不在计数器中
                if len(self.counts) >= self.max_tracker_size:  # 如果计数器已满
                    self.counts.popitem(last=False)  # 驱逐最旧的条目（LRU）
                self.counts[block_hash] = 1  # 初始化计数为 1
        return self._backing.lookup(block_hashes)  # 委托给底层管理器进行查找

    def prepare_store(  # 准备存储，过滤低频块
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """Filter out blocks below threshold, then delegate to backing.
        过滤掉低于阈值的块，然后委托给底层管理器。

        Filtering is evaluated *before* calling the backing manager's
        ``prepare_store`` so that blocks that would be skipped do not
        consume any CPU offload capacity.
        过滤在调用底层管理器的 prepare_store 之前执行，
        这样被跳过的块不会占用任何 CPU 卸载容量。
        """
        block_hashes = list(block_hashes)  # 将可迭代对象转为列表
        eligible = [  # 筛选出符合阈值条件的块哈希
            bh for bh in block_hashes if self.counts.get(bh, 0) >= self.store_threshold  # 仅保留访问次数达到阈值的块
        ]

        # Delegate to the backing manager with only the eligible hashes.
        # Passing an empty list is intentional and safe — both
        # LRUOffloadingManager and ARCOffloadingManager handle it correctly,
        # returning a PrepareStoreOutput with empty lists.
        return self._backing.prepare_store(eligible)  # 仅将符合条件的块哈希传递给底层管理器

    # ------------------------------------------------------------------
    # Delegated methods
    # 委托的方法
    # ------------------------------------------------------------------

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:  # 委托准备加载操作
        """委托给底层管理器执行准备加载操作。"""
        return self._backing.prepare_load(block_hashes)  # 直接委托给底层管理器

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:  # 委托标记最近使用操作
        """委托给底层管理器执行标记最近使用操作。"""
        return self._backing.touch(block_hashes)  # 直接委托给底层管理器

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:  # 委托完成加载操作
        """委托给底层管理器执行完成加载操作。"""
        return self._backing.complete_load(block_hashes)  # 直接委托给底层管理器

    def complete_store(  # 委托完成存储操作
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        """委托给底层管理器执行完成存储操作。"""
        return self._backing.complete_store(block_hashes, success)  # 直接委托给底层管理器

    def take_events(self) -> Iterable[OffloadingEvent]:  # 委托获取卸载事件
        """委托给底层管理器获取卸载事件。"""
        return self._backing.take_events()  # 直接委托给底层管理器
