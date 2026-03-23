# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# LRU 缓存模块，基于 cachetools.LRUCache 扩展，提供缓存命中率统计、
# 项目固定（pin）防止被淘汰、以及有序视图访问等增强功能。
# 主要用于 vLLM 中需要容量受限且可追踪使用情况的缓存场景。

from collections import UserDict
from collections.abc import Callable, Hashable, Iterator, KeysView, Mapping
from types import MappingProxyType
from typing import NamedTuple, TypeVar, cast, overload

import cachetools

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")
_T = TypeVar("_T")


# 哨兵类型，用于区分"未找到非固定项"和 None 值的情况
class _Sentinel: ...


# 全局哨兵实例，在 popitem 中表示所有缓存项均被固定、无法淘汰
ALL_PINNED_SENTINEL = _Sentinel()


# 缓存的有序只读视图，迭代时按 LRU 顺序返回键，
# 用于在不暴露内部数据结构的前提下提供有序访问能力
class _MappingOrderCacheView(UserDict[_K, _V]):
    def __init__(self, data: Mapping[_K, _V], ordered_keys: Mapping[_K, None]):
        super().__init__(data)
        self.ordered_keys = ordered_keys

    def __iter__(self) -> Iterator[_K]:
        return iter(self.ordered_keys)

    def keys(self) -> KeysView[_K]:
        return KeysView(self.ordered_keys)


# 缓存统计信息的不可变数据结构，记录命中次数和总查询次数，
# 支持计算命中率和通过减法运算获取增量统计
class CacheInfo(NamedTuple):
    hits: int
    total: int

    @property
    def hit_ratio(self) -> float:
        if self.total == 0:
            return 0

        return self.hits / self.total

    def __sub__(self, other: "CacheInfo"):
        return CacheInfo(
            hits=self.hits - other.hits,
            total=self.total - other.total,
        )


# 增强版 LRU 缓存，在 cachetools.LRUCache 基础上扩展了以下功能：
# 1. 命中率统计（hits/total）及增量查询（delta 模式）
# 2. 项目固定（pin）机制——被固定的项不会被 LRU 淘汰
# 3. 有序视图属性（cache/order）用于按 LRU 顺序遍历
# 4. 可覆写的 _on_remove 回调，在项目被删除时触发自定义逻辑
class LRUCache(cachetools.LRUCache[_K, _V]):
    def __init__(self, capacity: float, getsizeof: Callable[[_V], float] | None = None):
        super().__init__(capacity, getsizeof)

        self.pinned_items = set[_K]()

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def __getitem__(self, key: _K, *, update_info: bool = True) -> _V:
        value = super().__getitem__(key)

        if update_info:
            self._hits += 1
            self._total += 1

        return value

    def __delitem__(self, key: _K) -> None:
        run_on_remove = key in self
        value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]
        super().__delitem__(key)
        if key in self.pinned_items:
            # Todo: add warning to inform that del pinned item
            self._unpin(key)
        if run_on_remove:
            self._on_remove(key, value)

    @property
    def cache(self) -> Mapping[_K, _V]:
        """Return the internal cache dictionary in order (read-only)."""
        return _MappingOrderCacheView(
            self._Cache__data,  # type: ignore
            self.order,
        )

    @property
    def order(self) -> Mapping[_K, None]:
        """Return the internal order dictionary (read-only)."""
        return MappingProxyType(self._LRUCache__order)  # type: ignore

    @property
    def capacity(self) -> float:
        return self.maxsize

    @property
    def usage(self) -> float:
        if self.maxsize == 0:
            return 0

        return self.currsize / self.maxsize

    def stat(self, *, delta: bool = False) -> CacheInfo:
        """
        Gets the cumulative number of hits and queries against this cache.

        If `delta=True`, instead gets these statistics
        since the last call that also passed `delta=True`.
        """
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info

    def touch(self, key: _K) -> None:
        try:
            self._LRUCache__order.move_to_end(key)  # type: ignore
        except KeyError:
            self._LRUCache__order[key] = None  # type: ignore

    @overload
    def get(self, key: _K, /) -> _V | None: ...

    @overload
    def get(self, key: _K, /, default: _V | _T) -> _V | _T: ...

    def get(self, key: _K, /, default: _V | _T | None = None) -> _V | _T | None:
        value: _V | _T | None
        if key in self:
            value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]

            self._hits += 1
        else:
            value = default

        self._total += 1
        return value

    @overload
    def pop(self, key: _K) -> _V: ...

    @overload
    def pop(self, key: _K, default: _V | _T) -> _V | _T: ...

    def pop(self, key: _K, default: _V | _T | None = None) -> _V | _T | None:
        value: _V | _T | None
        if key not in self:
            return default

        value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]
        self.__delitem__(key)
        return value

    def put(self, key: _K, value: _V) -> None:
        self.__setitem__(key, value)

    def pin(self, key: _K) -> None:
        """
        Pins a key in the cache preventing it from being
        evicted in the LRU order.
        """
        if key not in self:
            raise ValueError(f"Cannot pin key: {key} not in cache.")
        self.pinned_items.add(key)

    def _unpin(self, key: _K) -> None:
        """
        Unpins a key in the cache allowing it to be
        evicted in the LRU order.
        """
        self.pinned_items.remove(key)

    def _on_remove(self, key: _K, value: _V | None) -> None:
        pass

    def remove_oldest(self, *, remove_pinned: bool = False) -> None:
        if len(self) == 0:
            return

        self.popitem(remove_pinned=remove_pinned)

    def _remove_old_if_needed(self) -> None:
        while self.currsize > self.capacity:
            self.remove_oldest()

    def popitem(self, remove_pinned: bool = False):
        """Remove and return the `(key, value)` pair least recently used."""
        if not remove_pinned:
            # pop the oldest item in the cache that is not pinned
            lru_key = next(
                (key for key in self.order if key not in self.pinned_items),
                ALL_PINNED_SENTINEL,
            )
            if lru_key is ALL_PINNED_SENTINEL:
                raise RuntimeError(
                    "All items are pinned, cannot remove oldest from the cache."
                )
        else:
            lru_key = next(iter(self.order))
        value = self.pop(cast(_K, lru_key))
        return (lru_key, value)

    def clear(self) -> None:
        while len(self) > 0:
            self.remove_oldest(remove_pinned=True)

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)
