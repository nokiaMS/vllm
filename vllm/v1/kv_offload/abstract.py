# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OffloadingManager class for managing KV data offloading in vLLM v1
OffloadingManager 类用于管理 vLLM v1 中的 KV 数据卸载

This class runs in the scheduler, tracks which blocks are offloaded
and their address.
此类运行在调度器中，跟踪哪些块已被卸载及其地址。

The class provides the following primitives:
该类提供以下基本操作：
    lookup() - find the length of the maximal series of blocks,
        starting from the first one, that are all offloaded.
    lookup() - 查找从第一个块开始、所有块都已卸载的最大连续块序列长度。
    prepare_load() - prepare given blocks to be read.
        The given blocks will be protected from eviction.
        This function returns a LoadSpec which encapsulates
        information required for performing the load.
    prepare_load() - 准备读取给定的块。给定的块将受到保护，不会被驱逐。
        此函数返回一个 LoadSpec，封装了执行加载所需的信息。
    touch() - marks the give blocks as recently used. Can be used
        to track block's LRU. This function is separated from the
        prepare_load function to allow setting block recency even
        for blocks which do not need reading from the cache, such as
        blocks that are cached by the GPU prefix cache.
    touch() - 将给定块标记为最近使用。可用于跟踪块的 LRU。
        此函数与 prepare_load 分开，以允许即使不需要从缓存读取的块
        （如 GPU 前缀缓存中的块）也能设置块的近期性。
    complete_load() - mark blocks which were previously prepared to be
        loaded as done loading. This is to re-allow their eviction.
    complete_load() - 将之前准备加载的块标记为加载完成。
        这将重新允许它们被驱逐。
    prepare_store() - prepare the given blocks to be written.
        Returns a StoreSpec encapsulating offloading information,
        as well as a list of blocks that were evicted as a result.
    prepare_store() - 准备写入给定的块。
        返回一个封装卸载信息的 StoreSpec，以及由此被驱逐的块列表。
    complete_store() - marks a previous store as completed.
        Following this call, the given blocks will become loadable.
    complete_store() - 将之前的存储标记为完成。
        调用此方法后，给定的块将变为可加载状态。
"""

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections.abc import Iterable  # 导入可迭代类型
from dataclasses import dataclass  # 导入数据类装饰器

from vllm.v1.core.kv_cache_utils import BlockHash  # 导入块哈希类型


class LoadStoreSpec(ABC):
    """
    Abstract metadata that encapsulates information allowing a worker
    to load, and optionally also to store, blocks of KV data.
    抽象元数据类，封装了允许工作线程加载（以及可选地存储）KV 数据块的信息。
    """

    @staticmethod  # 静态方法装饰器
    @abstractmethod  # 抽象方法装饰器
    def medium() -> str:  # 返回存储介质类型的字符串表示
        """
        Returns a string representation of the medium type
        this store/load targets.
        返回此存储/加载操作目标的存储介质类型的字符串表示。
        """
        pass  # 子类必须实现此方法


@dataclass  # 数据类装饰器，自动生成 __init__ 等方法
class PrepareStoreOutput:
    """准备存储操作的输出结果，包含要存储的块哈希、存储规格和被驱逐的块哈希。"""
    block_hashes_to_store: list[BlockHash]  # 需要存储的块哈希列表
    store_spec: LoadStoreSpec  # 存储规格，包含存储位置等信息
    block_hashes_evicted: list[BlockHash]  # 因腾出空间而被驱逐的块哈希列表


@dataclass  # 数据类装饰器
class OffloadingEvent:
    """卸载事件，记录块的存储或移除操作。"""
    block_hashes: list[BlockHash]  # 相关的块哈希列表
    block_size: int  # 块大小
    medium: str  # 存储介质类型
    # True if blocks are removed, False if stored
    removed: bool  # True 表示块被移除，False 表示块被存储


class OffloadingManager(ABC):
    """卸载管理器抽象基类，定义了管理 KV 数据卸载的接口。"""

    @abstractmethod  # 抽象方法装饰器
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:  # 查找已卸载的连续块数量
        """
        Finds the length of the maximal series of blocks, starting from the
        first one, that are all offloaded.
        查找从第一个块开始、所有块都已卸载的最大连续序列长度。

        Args:
            block_hashes: the hashes identifying the blocks to lookup.
            block_hashes: 用于查找的块标识哈希。

        Returns:
            An integer representing the maximal number of blocks that
            are currently offloaded, or None if the lookup should be retried
            later. Returning None will delay the request handling by the vLLM
            scheduler.
            返回当前已卸载的最大连续块数，如果查找应稍后重试则返回 None。
            返回 None 将延迟 vLLM 调度器的请求处理。
        """
        pass  # 子类必须实现此方法

    @abstractmethod  # 抽象方法装饰器
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:  # 准备加载指定块
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.
        准备读取给定的块。给定的块将受到保护，直到调用 complete_load 之前不会被驱逐。
        假设所有给定的块都已被卸载。

        Args:
            block_hashes: the hashes identifying the blocks.
            block_hashes: 标识块的哈希值。

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
            返回一个 LoadStoreSpec，工作线程可用其定位和加载实际的卸载 KV 数据。
        """
        pass  # 子类必须实现此方法

    def touch(self, block_hashes: Iterable[BlockHash]):  # 将指定块标记为最近使用
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.
        将给定块标记为最近使用。实际上可能意味着将它们移到 LRU 列表的末尾。

        Args:
            block_hashes: the hashes identifying the blocks.
            block_hashes: 标识块的哈希值。
        """
        return  # 默认实现为空操作

    def complete_load(self, block_hashes: Iterable[BlockHash]):  # 标记加载完成
        """
        Marks previous blocks that were prepared to load as done loading.
        将之前准备加载的块标记为加载完成。

        Args:
            block_hashes: the hashes identifying the blocks.
            block_hashes: 标识块的哈希值。
        """
        return  # 默认实现为空操作

    @abstractmethod  # 抽象方法装饰器
    def prepare_store(  # 准备存储指定块
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.
        准备将给定的块卸载。给定的块将受到保护，直到调用 complete_store 之前不会被驱逐。

        Args:
            block_hashes: the hashes identifying the blocks.
            block_hashes: 标识块的哈希值。

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
            返回 PrepareStoreOutput，指示哪些块需要存储、存储位置（LoadStoreSpec）
            以及因此被驱逐的块列表。如果无法存储这些块则返回 None。
        """
        pass  # 子类必须实现此方法

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):  # 标记存储完成
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If if_success is False, blocks that were not marked as stored will be
        removed.
        将之前准备存储的块标记为已存储。调用此方法后，这些块变为可加载状态。
        如果 success 为 False，未标记为已存储的块将被移除。

        Args:
            block_hashes: the hashes identifying the blocks.
            block_hashes: 标识块的哈希值。
            success: whether the blocks were stored successfully.
            success: 块是否存储成功。
        """
        return  # 默认实现为空操作

    def take_events(self) -> Iterable[OffloadingEvent]:  # 获取卸载事件
        """
        Take the offloading events from the manager.
        从管理器中获取卸载事件。

        Yields:
            New OffloadingEvents collected since the last call.
            自上次调用以来收集的新卸载事件。
        """
        return ()  # 默认返回空元组
