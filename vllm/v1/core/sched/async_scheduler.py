# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.v1.core.sched.output import SchedulerOutput  # 导入调度器输出数据类
from vllm.v1.core.sched.scheduler import Scheduler  # 导入同步调度器基类
from vllm.v1.request import Request, RequestStatus  # 导入请求对象和请求状态枚举

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# [中文注释] 异步调度器（继承自 Scheduler），用于异步调度模式。
#   与同步 Scheduler 的主要区别：
#     1. _update_after_schedule(): 在调度后立即为每个请求预留 1 + num_spec_tokens 个
#        output placeholder，使用占位 spec_token_ids（-1），实际值在 worker 中更新。
#     2. _update_request_with_output(): 处理模型输出时减少 placeholder 计数，
#        并在此时（而非调度时）执行 cache_blocks，因为异步模式下调度和执行是解耦的。
#     3. 支持 discard_latest_async_tokens: 当 reset_prefix_cache 强制抢占时，
#        丢弃最近的异步 token 以避免重复输出。
class AsyncScheduler(Scheduler):
    """异步调度器，继承自 Scheduler，用于调度与执行解耦的异步推理模式。"""

    def __init__(self, *args, **kwargs) -> None:
        """初始化异步调度器，创建投机解码占位符列表。"""
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens  # 预分配的投机解码占位符列表，值为-1

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """调度完成后更新状态：为每个请求预留输出占位符（包括投机解码 token）。"""
        super()._update_after_schedule(scheduler_output)  # 调用父类的调度后更新
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens  # 获取本步投机解码 token 映射
        for req_id in scheduler_output.num_scheduled_tokens:  # 遍历本步调度的所有请求
            request = self.requests[req_id]  # 根据请求ID获取请求对象
            if request.is_prefill_chunk:  # 如果请求仍在预填充阶段则跳过
                continue

            scheduler_output.pending_structured_output_tokens |= (  # 更新是否有待处理的结构化输出 token
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))  # 获取当前请求的投机解码 token 数量
            request.num_output_placeholders += 1 + cur_num_spec_tokens  # 增加输出占位符计数（1个正常token + 投机token数）
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders  # 设置占位符spec_token_ids，实际值在worker中更新

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """处理模型输出：减少占位符计数，缓存KV block，返回新token和是否停止。"""
        if request.discard_latest_async_tokens:  # 如果请求被标记为丢弃最新异步token
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False  # 重置丢弃标记
            return [], False  # 返回空token列表，未停止

        status_before_update = request.status  # 记录更新前的请求状态
        new_token_ids, stopped = super()._update_request_with_output(  # 调用父类方法处理输出token
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)  # 减少占位符计数（已生成的token数量）
        assert request.num_output_placeholders >= 0  # 断言占位符计数不为负

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:  # 仅对运行中的请求执行缓存
            self.kv_cache_manager.cache_blocks(  # 缓存KV block到前缀缓存
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped  # 返回新token列表和是否停止
