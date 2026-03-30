# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq  # 导入堆队列（优先队列）模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections import deque  # 导入双端队列
from collections.abc import Iterable, Iterator  # 导入可迭代和迭代器类型
from enum import Enum  # 导入枚举类型

from vllm.v1.request import Request  # 导入请求对象


# [中文注释] 调度策略枚举：FCFS（先来先服务）和 PRIORITY（优先级调度）。
class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"  # 先来先服务策略
    PRIORITY = "priority"  # 优先级调度策略


# [中文注释] 请求队列抽象基类，定义调度器所需的队列操作接口。
#   子类需实现：add_request、pop_request、peek_request、prepend_request 等方法。
class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


# [中文注释] FCFS（先来先服务）请求队列，继承自 deque 和 RequestQueue。
#   按请求到达顺序调度，prepend_request 用于将被抢占的请求放回队首。
class FCFSRequestQueue(deque[Request], RequestQueue):
    """先来先服务请求队列，基于双端队列实现。"""

    def add_request(self, request: Request) -> None:
        """按先来先服务策略将请求添加到队列尾部。"""
        self.append(request)  # 追加到队尾

    def pop_request(self) -> Request:
        """按先来先服务策略从队列头部弹出请求。"""
        return self.popleft()  # 从队首弹出

    def peek_request(self) -> Request:
        """查看队列头部的请求但不移除。"""
        if not self:  # 队列为空时抛出异常
            raise IndexError("peek from an empty queue")
        return self[0]  # 返回队首元素

    def prepend_request(self, request: Request) -> None:
        """将请求插入队列头部（用于被抢占的请求恢复）。"""
        self.appendleft(request)  # 插入队首

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)  # 将另一队列的请求按逆序插入队首

    def remove_request(self, request: Request) -> None:
        """从队列中移除指定请求。"""
        self.remove(request)  # 移除第一个匹配的请求

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列中批量移除多个请求。"""
        requests_to_remove = set(requests)  # 转换为集合以加速查找
        filtered_requests = [req for req in self if req not in requests_to_remove]  # 过滤掉需移除的请求
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()  # 清空队列
        self.extend(filtered_requests)  # 用过滤后的请求重新填充

    def __bool__(self) -> bool:
        """检查队列是否非空。"""
        return len(self) > 0  # 长度大于0返回True

    def __len__(self) -> int:
        """返回队列中的请求数量。"""
        return super().__len__()  # 调用父类的长度方法

    def __iter__(self) -> Iterator[Request]:
        """按先来先服务顺序迭代队列。"""
        return super().__iter__()  # 调用父类的迭代器


# [中文注释] 优先级请求队列，基于最小堆实现。
#   按 Request 的 (priority, arrival_time) 排序，priority 值越小优先级越高。
#   prepend_request 等效于 add_request（优先级队列没有"队首"概念）。
class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        """初始化优先级队列，创建空的最小堆。"""
        self._heap: list[Request] = []  # 内部最小堆存储

    def add_request(self, request: Request) -> None:
        """按优先级策略将请求加入队列。"""
        heapq.heappush(self._heap, request)  # 将请求推入堆中

    def pop_request(self) -> Request:
        """按优先级弹出最高优先级的请求。"""
        if not self._heap:  # 堆为空时抛出异常
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)  # 弹出堆顶（最小优先级值）元素

    def peek_request(self) -> Request:
        """查看最高优先级的请求但不移除。"""
        if not self._heap:  # 堆为空时抛出异常
            raise IndexError("peek from empty heap")
        return self._heap[0]  # 返回堆顶元素

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)  # 优先级队列无队首概念，等价于add_request

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将另一个队列的所有请求按优先级加入（优先级队列无队首概念）。"""
        for request in requests:  # 遍历源队列中的所有请求
            self.add_request(request)  # 逐个加入当前优先级队列

    def remove_request(self, request: Request) -> None:
        """从队列中移除指定请求。"""
        self._heap.remove(request)  # 从堆列表中移除请求
        heapq.heapify(self._heap)  # 重新调整堆结构

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列中批量移除多个请求。"""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)  # 确保为集合类型
        self._heap = [r for r in self._heap if r not in requests_to_remove]  # 过滤掉需移除的请求
        heapq.heapify(self._heap)  # 重新调整堆结构

    def __bool__(self) -> bool:
        """检查队列是否非空。"""
        return bool(self._heap)  # 堆非空返回True

    def __len__(self) -> int:
        """返回队列中的请求数量。"""
        return len(self._heap)  # 返回堆的长度

    def __iter__(self) -> Iterator[Request]:
        """按优先级顺序迭代队列（使用堆副本，不影响原堆）。"""
        heap_copy = self._heap[:]  # 复制堆避免修改原数据
        while heap_copy:  # 持续弹出直到为空
            yield heapq.heappop(heap_copy)  # 按优先级顺序逐个弹出


# [中文注释] 工厂函数：根据调度策略创建对应的请求队列实例。
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """根据调度策略创建对应的请求队列实例。"""
    if policy == SchedulingPolicy.PRIORITY:  # 优先级策略
        return PriorityRequestQueue()  # 返回优先级队列
    elif policy == SchedulingPolicy.FCFS:  # 先来先服务策略
        return FCFSRequestQueue()  # 返回FCFS队列
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")  # 未知策略抛出异常
