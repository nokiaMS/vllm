# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
"""KV cache metrics tracking."""  # KV 缓存指标追踪模块

import random  # 导入随机数模块（用于采样）
import time  # 导入时间模块（用于记录时间戳）
from collections import deque  # 导入双端队列（用于有界访问历史）
from typing import TYPE_CHECKING  # 导入类型检查标志

if TYPE_CHECKING:  # 仅在类型检查时导入
    from vllm.v1.core.kv_cache_utils import KVCacheBlock  # 导入 KVCacheBlock 类型

from vllm.v1.metrics.stats import KVCacheEvictionEvent  # 导入 KV 缓存驱逐事件类


# [中文注释] 单个 KV cache block 的生命周期指标。
#   记录 block 的诞生时间 (birth_time_ns)、最后访问时间 (last_access_ns)
#   和最近 4 次访问的时间戳 (access_history)。
#   用于在 block 被驱逐时生成 KVCacheEvictionEvent，包含：
#     - lifetime_seconds: block 从创建到驱逐的总存活时长
#     - idle_seconds: 最后一次访问到驱逐的空闲时长
#     - reuse_gaps_seconds: 连续两次访问之间的间隔列表
class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    def __init__(self):
        """初始化 block 指标状态，记录诞生时间"""
        now_ns = time.monotonic_ns()  # 获取当前单调时间（纳秒）
        self.birth_time_ns = now_ns  # block 诞生时间
        self.last_access_ns = now_ns  # 最后访问时间
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4)  # 最近 4 次访问时间戳

    def record_access(self) -> None:
        """记录一次访问事件"""
        now_ns = time.monotonic_ns()  # 获取当前时间
        self.last_access_ns = now_ns  # 更新最后访问时间
        self.access_history.append(now_ns)  # 添加到访问历史

    def get_lifetime_seconds(self) -> float:
        """获取 block 从创建到现在的存活时长（秒）"""
        now_ns = time.monotonic_ns()  # 获取当前时间
        return (now_ns - self.birth_time_ns) / 1e9  # 计算存活时长

    def get_idle_time_seconds(self) -> float:
        """获取 block 从最后访问到现在的空闲时长（秒）"""
        now_ns = time.monotonic_ns()  # 获取当前时间
        return (now_ns - self.last_access_ns) / 1e9  # 计算空闲时长

    def get_reuse_gaps_seconds(self) -> list[float]:
        """获取连续两次访问之间的时间间隔列表（秒）"""
        if len(self.access_history) < 2:  # 不足 2 次访问
            return []  # 无间隔可计算
        history = list(self.access_history)  # 转为列表
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]  # 计算相邻间隔


# [中文注释] KV Cache 驻留指标的采样收集器。
#   通过 sample_rate 控制采样比例（默认 1%），避免对所有 block 追踪指标的开销。
#   核心流程：
#     on_block_allocated → 以 sample_rate 概率开始追踪该 block
#     on_block_accessed  → 更新已追踪 block 的访问记录
#     on_block_evicted   → 生成 KVCacheEvictionEvent 并加入 _eviction_events
#     drain_events()     → 取走并清空事件队列，供 metrics reporter 上报
class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling."""

    def __init__(self, sample_rate: float = 0.01):
        """初始化指标收集器，设置采样率"""
        assert 0 < sample_rate <= 1.0, (  # 确保采样率在有效范围内
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate  # 保存采样率

        self.block_metrics: dict[int, BlockMetricsState] = {}  # block_id → 指标状态的映射

        self._eviction_events: list[KVCacheEvictionEvent] = []  # 驱逐事件队列

    def should_sample_block(self) -> bool:
        """判断是否应该对当前 block 进行采样追踪"""
        return random.random() < self.sample_rate  # 以采样率概率返回 True

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        """block 被分配时调用，以采样率概率开始追踪"""
        if self.should_sample_block():  # 如果被采样命中
            self.block_metrics[block.block_id] = BlockMetricsState()  # 创建指标状态

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        """block 被访问时调用，更新已追踪 block 的访问记录"""
        metrics = self.block_metrics.get(block.block_id)  # 查找该 block 的指标
        if metrics:  # 如果正在追踪
            metrics.record_access()  # 记录访问

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        """block 被驱逐时调用，生成驱逐事件"""
        metrics = self.block_metrics.pop(block.block_id, None)  # 弹出并移除指标
        if not metrics:  # 如果未被追踪
            return  # 直接返回

        lifetime = metrics.get_lifetime_seconds()  # 计算存活时长
        idle_time = metrics.get_idle_time_seconds()  # 计算空闲时长
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())  # 计算重用间隔

        self._eviction_events.append(  # 创建并添加驱逐事件
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,  # 存活时长
                idle_seconds=idle_time,  # 空闲时长
                reuse_gaps_seconds=reuse_gaps,  # 重用间隔
            )
        )

    def reset(self) -> None:
        """重置所有状态（缓存重置时调用）"""
        self.block_metrics.clear()  # 清空指标映射
        self._eviction_events.clear()  # 清空事件队列

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        """取走并清空事件队列，供 metrics reporter 上报"""
        events = self._eviction_events  # 保存当前事件列表
        self._eviction_events = []  # 重置为空列表
        return events  # 返回事件列表
