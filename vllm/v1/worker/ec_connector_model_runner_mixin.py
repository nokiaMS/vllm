# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# ECConnectorModelRunnerMixin：编码器缓存（Encoder Cache）连接器的 Mixin 类
# 设计目的：为 GPU/TPU 等不同后端的 ModelRunner 提供统一的编码器缓存传输接口
# 核心功能：
#   1. 将编码器输出缓存保存到 EC 连接器（用于跨节点/跨设备共享）
#   2. 查询已完成的缓存传输（发送端和接收端）
#   3. 通过上下文管理器封装完整的 EC 连接器生命周期（绑定元数据→加载缓存→清理）
# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
class ECConnectorModelRunnerMixin:
    # 将编码器缓存保存到 EC 连接器，供其他节点消费
    # 如果 EC 传输未初始化则跳过（非 EC 部署场景）
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        if not has_ec_transfer():
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    # 查询已完成的 EC 传输，返回已完成发送和接收的请求 ID 集合
    @staticmethod
    def get_finished_ec_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[set[str] | None, set[str] | None]:
        if has_ec_transfer():
            return get_ec_transfer().get_finished(scheduler_output.finished_req_ids)
        return None, None

    # 条件性获取 EC 连接器输出的上下文管理器
    # 当 EC 传输可用时返回完整的连接器生命周期管理器；否则返回空上下文
    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    # EC 连接器生命周期的核心上下文管理器
    # 流程：绑定调度器元数据 → 启动缓存加载（消费端）→ yield 执行模型前向 →
    #       收集已完成传输 → 清理元数据
    # 必须在活跃的 forward 上下文中使用
    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        # Load caches for consumer or both roles
        if ec_connector.is_consumer:
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )

            ec_connector.clear_connector_metadata()
