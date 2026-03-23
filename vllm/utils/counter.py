# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 计数器模块，提供非线程安全和线程安全两种计数器实现，
# 用于生成递增的唯一 ID（如请求 ID、序列 ID 等）。

import threading


# 简单的非线程安全计数器，实现 __next__ 协议使其可用于 next() 调用。
# 每次调用返回当前值并自增，适用于单线程场景下的顺序 ID 生成。
class Counter:
    def __init__(self, start: int = 0) -> None:
        super().__init__()

        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


# 原子计数器，使用 threading.Lock 保证多线程环境下的增减操作是原子性的。
# 适用于多线程共享的计数场景，如并发请求计数、资源使用量追踪等。
class AtomicCounter:
    """An atomic, thread-safe counter"""

    def __init__(self, initial: int = 0) -> None:
        """Initialize a new atomic counter to given initial value"""
        super().__init__()

        self._value = initial
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def inc(self, num: int = 1) -> int:
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    def dec(self, num: int = 1) -> int:
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value
