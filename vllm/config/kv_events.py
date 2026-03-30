# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明


from typing import Literal  # 导入Literal类型，用于限定字面量类型

from pydantic import Field  # 导入pydantic的Field，用于字段配置

from vllm.config.utils import config  # 导入config装饰器，用于创建pydantic数据类


@config  # 使用config装饰器创建pydantic数据类
class KVEventsConfig:
    """KV缓存事件发布的配置类。"""

    enable_kv_cache_events: bool = False  # 是否启用KV缓存事件跟踪，用于记录块的存储和移除
    """If True, enable KV cache events for tracking block storage and removal.
    Events can be published externally by zmq using the event publisher config.
    """

    publisher: Literal["null", "zmq"] = Field(default=None)  # 事件发布器类型，支持"null"和"zmq"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"  # ZMQ事件发布的端点地址
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: str | None = None  # ZMQ事件重放的端点地址
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000  # 重放端点缓存的步数，仅保存最近N步的事件
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000  # ZMQ发布器的高水位标记，超过后消费者跟不上时会丢弃事件
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000  # 等待发布时排队的最大事件数
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""  # 事件发布器使用的主题，消费者可以订阅此主题接收事件
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    def __post_init__(self):
        """初始化后处理：根据是否启用KV缓存事件自动设置发布器类型。"""
        if self.publisher is None:  # 如果未指定发布器类型
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"  # 启用事件则用zmq，否则用null
