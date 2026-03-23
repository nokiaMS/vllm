# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio  # 导入异步IO库，用于事件循环和异步操作
from collections import defaultdict, deque  # 导入默认字典和双端队列数据结构
from collections.abc import Iterable  # 导入可迭代类型抽象基类
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, cast  # 导入类型提示工具

import numpy as np  # 导入NumPy数组库，用于路由专家数据
import torch  # 导入PyTorch张量库，用于池化输出和嵌入

from vllm.lora.request import LoRARequest  # 导入LoRA请求类，表示LoRA适配器请求
from vllm.outputs import (  # 导入各种输出类型
    STREAM_FINISHED,  # 流式输出完成的哨兵值
    CompletionOutput,  # 单个补全输出
    PoolingOutput,  # 池化输出（用于嵌入/分类等任务）
    PoolingRequestOutput,  # 池化请求的完整输出
    RequestOutput,  # 生成请求的完整输出
)
from vllm.sampling_params import RequestOutputKind  # 导入请求输出类型枚举（DELTA/FINAL_ONLY等）
from vllm.tokenizers import TokenizerLike  # 导入分词器类型接口
from vllm.tracing import (  # 导入分布式追踪相关工具
    SpanAttributes,  # 追踪跨度属性常量
    SpanKind,  # 追踪跨度类型枚举
    extract_trace_context,  # 从请求头中提取追踪上下文
    instrument_manual,  # 手动创建追踪跨度
)
from vllm.utils import length_from_prompt_token_ids_or_embeds  # 导入从token ID或嵌入计算长度的工具函数
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason  # 导入引擎核心输出、请求和完成原因类型
from vllm.v1.engine.detokenizer import IncrementalDetokenizer  # 导入增量反分词器，将token ID转换为文本
from vllm.v1.engine.logprobs import LogprobsProcessor  # 导入对数概率处理器
from vllm.v1.engine.parallel_sampling import ParentRequest  # 导入并行采样的父请求类
from vllm.v1.metrics.stats import (  # 导入统计指标相关类
    IterationStats,  # 单次迭代统计
    LoRARequestStates,  # LoRA请求状态追踪
    RequestStateStats,  # 单个请求的统计数据
    SchedulerStats,  # 调度器统计数据
)

# 共享的空CPU张量，用作池化输出的占位符
# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")  # 创建一个空的CPU张量常量


# [中文注释] 请求输出收集器：每个请求对应一个实例，作为生产者-消费者队列：
#   - 生产者（output_handler）通过 put() 推入 RequestOutput
#   - 消费者（generate() 协程）通过 get()/get_nowait() 取出并 yield
#   - DELTA 模式下自动合并：若消费者未及时消费，多个 output 会被聚合
#   - 使用 asyncio.Event 实现无锁阻塞/唤醒
class RequestOutputCollector:  # 请求输出收集器类，作为生产者-消费者桥梁
    """收集每个请求的流式 RequestOutput，作为生产者和消费者之间的桥梁。

    当以流式增量模式输出时，如果生产者速度超过消费者，
    多个 RequestOutput 会被自动合并。

    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):  # 构造函数，接收输出类型和请求ID
        """初始化请求输出收集器。

        Args:
            output_kind: 输出类型，决定是否启用聚合模式
            request_id: 请求的唯一标识符
        """
        self.aggregate = output_kind == RequestOutputKind.DELTA  # 判断是否为增量模式，增量模式下启用聚合
        self.request_id = request_id  # 保存请求ID
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None  # 当前缓存的输出，可能是请求输出、异常或空
        self.ready = asyncio.Event()  # 异步事件，用于通知消费者有新输出可用

        self._input_stream_task: asyncio.Task | None = None  # 关联的流式输入任务引用，用于取消操作

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:  # 放入一个输出到收集器
        """非阻塞地放入一个输出。

        如果当前没有缓存输出或收到异常，直接设置；
        否则将新输出合并到已有输出中。

        Args:
            output: 要放入的输出对象或异常
        """
        if self.output is None or isinstance(output, Exception):  # 如果当前无输出，或者新输出是异常
            self.output = output  # 直接设置输出
            self.ready.set()  # 设置事件，通知消费者
        elif isinstance(self.output, RequestOutput) and isinstance(  # 如果当前输出和新输出都是RequestOutput
            output, RequestOutput  # 检查新输出是否也是RequestOutput类型
        ):  # 两个条件都满足时进入合并分支
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)  # 将新输出合并到现有输出中（支持并行采样的多个索引）
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(  # 如果当前输出和新输出都是PoolingRequestOutput
            output, PoolingRequestOutput  # 检查新输出是否也是PoolingRequestOutput类型
        ):  # 两个条件都满足时进入池化输出替换分支
            self.output = output  # 池化输出直接替换（不需要合并）

    async def get(self) -> RequestOutput | PoolingRequestOutput:  # 异步阻塞式获取输出
        """阻塞式获取输出，等待直到有输出可用。

        Returns:
            可用的请求输出

        Raises:
            Exception: 如果收到的是异常则抛出
        """
        while (output := self.output) is None:  # 循环等待直到输出不为空
            await self.ready.wait()  # 异步等待事件被设置
        self.output = None  # 清空当前缓存的输出
        self.ready.clear()  # 重置事件状态
        if isinstance(output, Exception):  # 如果输出是异常
            raise output  # 抛出异常
        return output  # 返回正常输出

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:  # 非阻塞式获取输出
        """非阻塞式获取输出，如果没有可用输出则返回None。

        Returns:
            可用的请求输出，或None

        Raises:
            Exception: 如果收到的是异常则抛出
        """
        output = self.output  # 获取当前输出
        if output is not None:  # 如果有输出可用
            self.output = None  # 清空当前缓存
            self.ready.clear()  # 重置事件状态
        if isinstance(output, Exception):  # 如果输出是异常
            raise output  # 抛出异常
        return output  # 返回输出（可能为None）

    def close(self):  # 关闭收集器方法
        """关闭收集器，取消关联的流式输入任务。"""
        if self._input_stream_task is not None:  # 如果有关联的流式输入任务
            self._input_stream_task.cancel()  # 取消该任务
        self._input_stream_task = None  # 清空任务引用

    def __del__(self):  # 析构函数，垃圾回收时调用
        """析构函数，确保在对象被垃圾回收时取消关联的任务。"""
        if (task := self._input_stream_task) is not None:  # 如果有关联的流式输入任务
            task.get_loop().call_soon_threadsafe(task.cancel)  # 线程安全地在事件循环中取消任务
            self._input_stream_task = None  # 清空任务引用


@dataclass  # 数据类装饰器，自动生成__init__等方法
class OutputProcessorOutput:  # 输出处理器结果数据类
    """输出处理器的处理结果数据类。

    包含处理后的请求输出列表和需要中止的请求ID列表。

    Attributes:
        request_outputs: 处理后的请求输出列表
        reqs_to_abort: 需要在引擎核心中中止的请求ID列表
    """
    request_outputs: list[RequestOutput | PoolingRequestOutput]  # 处理后的请求输出列表
    reqs_to_abort: list[str]  # 需要中止的请求ID列表（因stop string提前终止的请求）


@dataclass
class StreamingUpdate:
    """流式输入更新的数据类。

    包含增量提示数据，当当前子请求完成时应用于请求状态。

    Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.

    Attributes:
        prompt: 增量提示文本
        prompt_token_ids: 增量提示的token ID列表
        arrival_time: 该更新的到达时间
        final: 是否为最终更新
    """

    prompt: str | None  # 增量提示文本，可能为空
    prompt_token_ids: list[int] | None  # 增量提示的token ID列表，可能为空
    arrival_time: float  # 该更新的到达时间戳
    final: bool = False  # 是否为最终更新，默认为False


# [中文注释] 每个请求在 OutputProcessor 中的状态。包含：
#   - detokenizer — 增量 detokenizer 实例
#   - logprobs_processor — logprobs 处理器
#   - output_kind — DELTA（流式增量）、FINAL_ONLY（仅最终结果）等
#   - parent_req — 并行采样时的父请求引用
#   - queue — 对应的 RequestOutputCollector
#   - stats — 请求级统计（到达时间、首 token 延迟等）
#   - stream_interval — 流式输出间隔（每 N 个 token 发送一次）
#   - streaming_input — 是否支持流式输入（resumable 请求）
class RequestState:
    """请求状态类，维护每个请求在输出处理器中的完整状态。

    包括反分词器、对数概率处理器、输出模式、并行采样父请求、
    统计信息以及流式输入队列等。
    """

    def __init__(
        self,
        request_id: str,  # 内部请求ID
        external_req_id: str,  # 外部请求ID（用户可见的ID）
        parent_req: ParentRequest | None,  # 并行采样时的父请求引用
        request_index: int,  # 请求在并行采样中的索引
        lora_request: LoRARequest | None,  # LoRA适配器请求
        output_kind: RequestOutputKind,  # 输出类型（DELTA/FINAL_ONLY等）
        prompt: str | None,  # 提示文本
        prompt_token_ids: list[int] | None,  # 提示的token ID列表
        prompt_embeds: torch.Tensor | None,  # 提示的嵌入向量
        logprobs_processor: LogprobsProcessor | None,  # 对数概率处理器
        detokenizer: IncrementalDetokenizer | None,  # 增量反分词器
        max_tokens_param: int | None,  # 最大生成token数参数
        arrival_time: float,  # 请求到达时间
        queue: RequestOutputCollector | None,  # 输出收集器队列
        log_stats: bool,  # 是否记录统计信息
        stream_interval: int,  # 流式输出间隔
        top_p: float | None = None,  # top-p采样参数
        n: int | None = None,  # 并行采样数
        temperature: float | None = None,  # 温度参数
        stream_input: bool = False,  # 是否启用流式输入
    ):
        """初始化请求状态。

        Args:
            request_id: 内部请求ID
            external_req_id: 外部请求ID
            parent_req: 并行采样的父请求
            request_index: 并行采样中的请求索引
            lora_request: LoRA请求
            output_kind: 输出类型
            prompt: 提示文本
            prompt_token_ids: 提示token ID列表
            prompt_embeds: 提示嵌入向量
            logprobs_processor: 对数概率处理器
            detokenizer: 反分词器
            max_tokens_param: 最大token数
            arrival_time: 到达时间
            queue: 输出收集器
            log_stats: 是否记录统计
            stream_interval: 流式间隔
            top_p: top-p参数
            n: 并行采样数
            temperature: 温度参数
            stream_input: 是否流式输入
        """
        self.request_id = request_id  # 保存内部请求ID
        self.external_req_id = external_req_id  # 保存外部请求ID
        self.parent_req = parent_req  # 保存父请求引用（并行采样时使用）
        self.request_index = request_index  # 保存请求在并行采样中的索引
        self.lora_request = lora_request  # 保存LoRA适配器请求
        self.lora_name = lora_request.lora_name if lora_request is not None else None  # 提取LoRA名称，无LoRA时为None
        self.output_kind = output_kind  # 保存输出类型（DELTA/FINAL_ONLY等）
        self.prompt = prompt  # 保存提示文本
        self.prompt_token_ids = prompt_token_ids  # 保存提示的token ID列表
        self.prompt_embeds = prompt_embeds  # 保存提示的嵌入向量
        self.prompt_len = length_from_prompt_token_ids_or_embeds(  # 计算提示长度（从token ID或嵌入向量）
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor  # 保存对数概率处理器
        self.detokenizer = detokenizer  # 保存增量反分词器
        self.max_tokens_param = max_tokens_param  # 保存最大生成token数参数
        self.top_p = top_p  # 保存top-p采样参数
        self.n = n  # 保存并行采样数
        self.temperature = temperature  # 保存温度参数
        self.is_prefilling = True  # 标记是否处于预填充阶段，初始为True
        self.queue = queue  # 保存输出收集器队列
        self.num_cached_tokens = 0  # 缓存的token数量，初始为0

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None  # 如果启用统计则创建统计对象，否则为None

        # Stream Interval
        self.stream_interval = stream_interval  # 保存流式输出间隔（每N个token发送一次）
        self.sent_tokens_offset = 0  # Offset of sent tokens  # 已发送token的偏移量，用于增量模式

        # Streaming input queue
        self.streaming_input = stream_input  # 标记是否启用流式输入
        self.input_chunk_queue: deque[StreamingUpdate] | None = (  # 流式输入更新队列
            deque() if stream_input else None  # 如果启用流式输入则创建队列，否则为None
        )

    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        """应用流式输入更新到请求状态。

        将增量提示数据合并到当前请求状态中，
        重新计算提示长度并重置预填充状态。

        Args:
            update: 流式输入更新数据
        """
        # Apply the update to the request state.
        self.streaming_input = not update.final  # 如果是最终更新则关闭流式输入标记
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        if update.prompt:  # 如果更新中包含提示文本
            self.prompt = (  # 将新提示追加到现有提示
                (self.prompt + update.prompt) if self.prompt else update.prompt  # 如果已有提示则拼接，否则直接使用新提示
            )
        if self.prompt_token_ids:  # 如果已有提示token ID
            self.prompt_token_ids.extend(update.prompt_token_ids or ())  # 追加新的token ID（如果有的话）
        else:  # 如果没有现有提示token ID
            self.prompt_token_ids = update.prompt_token_ids or []  # 使用新的token ID或空列表
        assert self.prompt_token_ids is not None  # 断言提示token ID不为空
        self.prompt_len = len(self.prompt_token_ids)  # 重新计算提示长度
        if self.stats is not None:  # 如果启用了统计
            self.stats.arrival_time = update.arrival_time  # 更新到达时间为最新更新的时间
        self.is_prefilling = True  # 重置为预填充状态

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,  # 分词器实例
        request: EngineCoreRequest,  # 引擎核心请求
        prompt: str | None,  # 提示文本
        parent_req: ParentRequest | None,  # 并行采样的父请求
        request_index: int,  # 请求索引
        queue: RequestOutputCollector | None,  # 输出收集器
        log_stats: bool,  # 是否记录统计
        stream_interval: int,  # 流式间隔
    ) -> "RequestState":
        """从新请求创建RequestState实例的工厂方法。

        根据请求类型（采样或池化）初始化相应的处理器。

        Args:
            tokenizer: 分词器实例
            request: 引擎核心请求
            prompt: 提示文本
            parent_req: 并行采样的父请求
            request_index: 请求索引
            queue: 输出收集器
            log_stats: 是否记录统计
            stream_interval: 流式间隔

        Returns:
            新创建的RequestState实例
        """
        if sampling_params := request.sampling_params:  # 如果请求包含采样参数（生成请求）
            if not sampling_params.detokenize:  # 如果不需要反分词
                tokenizer = None  # 将分词器设为None以跳过反分词
            output_kind = sampling_params.output_kind  # 获取输出类型
            logprobs_processor = LogprobsProcessor.from_new_request(  # 创建对数概率处理器
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(  # 创建增量反分词器
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens  # 获取最大token数参数
            top_p = sampling_params.top_p  # 获取top-p参数
            n = sampling_params.n  # 获取并行采样数
            temperature = sampling_params.temperature  # 获取温度参数
        else:  # 如果不是采样请求（是池化请求）
            logprobs_processor = None  # 池化请求不需要对数概率处理器
            detokenizer = None  # 池化请求不需要反分词器
            max_tokens_param = None  # 池化请求没有最大token数
            top_p = None  # 池化请求没有top-p参数
            n = None  # 池化请求没有并行采样数
            temperature = None  # 池化请求没有温度参数
            assert request.pooling_params is not None  # 断言池化参数不为空
            output_kind = request.pooling_params.output_kind  # 从池化参数获取输出类型

        assert request.external_req_id is not None  # 断言外部请求ID不为空
        return cls(  # 调用构造函数创建实例
            request_id=request.request_id,  # 内部请求ID
            external_req_id=request.external_req_id,  # 外部请求ID
            parent_req=parent_req,  # 父请求
            request_index=request_index,  # 请求索引
            lora_request=request.lora_request,  # LoRA请求
            output_kind=output_kind,  # 输出类型
            prompt=prompt,  # 提示文本
            prompt_token_ids=request.prompt_token_ids,  # 提示token ID
            prompt_embeds=request.prompt_embeds,  # 提示嵌入
            logprobs_processor=logprobs_processor,  # 对数概率处理器
            detokenizer=detokenizer,  # 反分词器
            max_tokens_param=max_tokens_param,  # 最大token数
            top_p=top_p,  # top-p参数
            n=n,  # 并行采样数
            temperature=temperature,  # 温度参数
            arrival_time=request.arrival_time,  # 到达时间
            queue=queue,  # 输出收集器
            log_stats=log_stats,  # 是否记录统计
            stream_interval=stream_interval,  # 流式间隔
            stream_input=request.resumable,  # 是否可恢复（流式输入）
        )

    def make_request_output(
        self,
        new_token_ids: list[int],  # 新生成的token ID列表
        pooling_output: torch.Tensor | None,  # 池化输出张量
        finish_reason: FinishReason | None,  # 完成原因
        stop_reason: int | str | None,  # 停止原因
        kv_transfer_params: dict[str, Any] | None = None,  # KV缓存传输参数
        routed_experts: np.ndarray | None = None,  # 路由专家信息
    ) -> RequestOutput | PoolingRequestOutput | None:
        """构造请求输出对象。

        根据输出类型、完成状态和流式间隔决定是否生成输出。

        Args:
            new_token_ids: 新生成的token ID列表
            pooling_output: 池化输出
            finish_reason: 完成原因
            stop_reason: 停止原因
            kv_transfer_params: KV缓存传输参数
            routed_experts: 路由专家信息

        Returns:
            请求输出对象，或None（如果此时不需要输出）
        """
        finished = finish_reason is not None  # 判断请求是否已完成
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY  # 判断是否为仅最终输出模式

        if not finished and final_only:  # 如果未完成且处于仅最终输出模式
            # Only the final output is required in FINAL_ONLY mode.
            return None  # 仅最终输出模式下，未完成时不返回输出

        if self.stream_interval > 1:  # 如果流式间隔大于1（不是每个token都发送）
            assert self.detokenizer is not None  # 断言反分词器存在

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (  # 检查是否满足发送条件
                finished  # 条件1：已完成
                or self.sent_tokens_offset == 0  # 条件2：第一个token
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset  # 条件3：达到流式间隔
                >= self.stream_interval
            ):
                return None  # 不满足任何发送条件，跳过本次输出

            if self.output_kind == RequestOutputKind.DELTA:  # 如果是增量输出模式
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[  # 从偏移位置截取新token
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()  # 更新已发送偏移量

        external_req_id = self.external_req_id  # 获取外部请求ID

        if pooling_output is not None:  # 如果有池化输出（嵌入/分类任务）
            return self._new_request_output(  # 创建池化请求输出
                external_req_id,
                [self._new_pooling_output(pooling_output)],  # 包装池化输出
                finished,
            )

        output = self._new_completion_output(  # 创建补全输出
            new_token_ids, finish_reason, stop_reason, routed_experts
        )

        if self.parent_req is None:  # 如果没有父请求（非并行采样）
            outputs = [output]  # 单个输出放入列表
        else:  # 如果有父请求（并行采样）
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)  # 从父请求获取聚合后的输出
            if not outputs:  # 如果没有输出（父请求还在等待其他子请求）
                return None  # 暂不返回输出
            external_req_id = self.parent_req.external_req_id  # 使用父请求的外部ID

        return self._new_request_output(  # 创建最终的请求输出
            external_req_id, outputs, finished, kv_transfer_params
        )

    def _new_request_output(
        self,
        external_req_id: str,  # 外部请求ID
        outputs: list[CompletionOutput] | list[PoolingOutput],  # 输出列表
        finished: bool,  # 是否已完成
        kv_transfer_params: dict[str, Any] | None = None,  # KV缓存传输参数
    ) -> RequestOutput | PoolingRequestOutput:
        """创建请求输出对象（内部方法）。

        处理提示嵌入的占位符token ID，并根据输出类型
        创建相应的RequestOutput或PoolingRequestOutput。

        Args:
            external_req_id: 外部请求ID
            outputs: 输出列表
            finished: 是否完成
            kv_transfer_params: KV传输参数

        Returns:
            RequestOutput或PoolingRequestOutput实例
        """
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids  # 获取提示token ID
        if prompt_token_ids is None and self.prompt_embeds is not None:  # 如果使用了嵌入而没有token ID
            prompt_token_ids = [0] * len(self.prompt_embeds)  # 创建与嵌入长度相同的占位符token ID
        assert prompt_token_ids is not None  # 断言提示token ID不为空

        first_output = outputs[0]  # 获取第一个输出
        if isinstance(first_output, PoolingOutput):  # 如果是池化输出
            assert len(outputs) == 1  # 断言池化输出只有一个
            return PoolingRequestOutput(  # 创建池化请求输出
                request_id=external_req_id,  # 外部请求ID
                outputs=first_output,  # 池化输出
                num_cached_tokens=self.num_cached_tokens,  # 缓存token数
                prompt_token_ids=prompt_token_ids,  # 提示token ID
                finished=finished,  # 是否完成
            )
        assert self.logprobs_processor is not None  # 断言对数概率处理器存在
        if self.output_kind == RequestOutputKind.DELTA:  # 如果是增量输出模式
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()  # 弹出并返回提示的对数概率（仅首次返回）
        else:  # 非增量模式
            prompt_logprobs = self.logprobs_processor.prompt_logprobs  # 每次都返回完整的提示对数概率

        return RequestOutput(  # 创建生成请求输出
            request_id=external_req_id,  # request_id is what was provided externally  # 外部请求ID
            lora_request=self.lora_request,  # LoRA请求
            prompt=self.prompt,  # 提示文本
            prompt_token_ids=prompt_token_ids,  # 提示token ID
            prompt_logprobs=prompt_logprobs,  # 提示对数概率
            outputs=cast(list[CompletionOutput], outputs),  # 补全输出列表（类型转换）
            finished=finished,  # 是否完成
            kv_transfer_params=kv_transfer_params,  # KV传输参数
            num_cached_tokens=self.num_cached_tokens,  # 缓存token数
            metrics=self.stats,  # 请求统计信息
        )

    def _new_completion_output(
        self,
        token_ids: list[int],  # token ID列表
        finish_reason: FinishReason | None,  # 完成原因
        stop_reason: int | str | None,  # 停止原因
        routed_experts: np.ndarray | None = None,  # 路由专家信息
    ) -> CompletionOutput:
        """创建单个补全输出。

        根据是否为增量模式准备文本、token ID和对数概率。

        Args:
            token_ids: 新生成的token ID列表
            finish_reason: 完成原因
            stop_reason: 停止原因
            routed_experts: 路由专家信息

        Returns:
            CompletionOutput实例
        """
        assert self.detokenizer is not None  # 断言反分词器存在
        assert self.logprobs_processor is not None  # 断言对数概率处理器存在
        finished = finish_reason is not None  # 判断是否已完成
        delta = self.output_kind == RequestOutputKind.DELTA  # 判断是否为增量模式

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)  # 获取下一段输出文本（增量或完整）
        if not delta:  # 如果不是增量模式
            token_ids = self.detokenizer.output_token_ids  # 使用完整的输出token ID列表

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs  # 获取对数概率
        if delta and logprobs:  # 如果是增量模式且有对数概率
            logprobs = logprobs[-len(token_ids) :]  # 只取与新token对应的对数概率

        return CompletionOutput(  # 创建补全输出
            index=self.request_index,  # 请求索引（并行采样中的第几个）
            text=text,  # 输出文本
            token_ids=token_ids,  # 输出token ID
            routed_experts=routed_experts,  # 路由专家信息
            logprobs=logprobs,  # 对数概率
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,  # 累积对数概率
            finish_reason=str(finish_reason) if finished else None,  # 完成原因字符串
            stop_reason=stop_reason if finished else None,  # 停止原因
        )

    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        """创建池化输出对象。

        Args:
            pooling_output: 池化输出张量

        Returns:
            PoolingOutput实例
        """
        return PoolingOutput(data=pooling_output)  # 用张量数据包装为PoolingOutput


# [中文注释] 输出处理器：将 Engine Core 返回的 EngineCoreOutput 转换为用户可见的 RequestOutput。
#   核心方法 process_outputs() 在单次循环中完成：
#     1. 统计计算（TTFT、吞吐量等）
#     2. Detokenize（token IDs → 文本）+ stop string 检查
#     3. Logprobs 计算
#     4. 构造 RequestOutput 并分发（AsyncLLM 放入 queue，LLMEngine 放入返回列表）
#     5. 清理已完成请求，返回需要 abort 的请求 ID（因 stop string 导致的提前终止）
#   设计原则：只在这一个循环中遍历所有输出，最小化 Python 循环开销
class OutputProcessor:
    """输出处理器，将 EngineCoreOutput 转换为 RequestOutput。

    核心方法 process_outputs() 在单次循环中完成统计计算、
    反分词、对数概率计算、输出构造和请求清理。

    Process EngineCoreOutputs into RequestOutputs.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,  # 分词器实例
        *,
        log_stats: bool,  # 是否记录统计信息
        stream_interval: int = 1,  # 流式输出间隔，默认每个token都发送
        tracing_enabled: bool = False,  # 是否启用分布式追踪
    ):
        """初始化输出处理器。

        Args:
            tokenizer: 分词器实例，用于反分词
            log_stats: 是否启用统计日志
            stream_interval: 流式输出间隔
            tracing_enabled: 是否启用追踪
        """
        self.log_stats = log_stats  # 保存是否记录统计的标志
        self.tokenizer = tokenizer  # 保存分词器引用
        self.stream_interval = stream_interval  # 保存流式输出间隔
        self.request_states: dict[str, RequestState] = {}  # 内部请求ID到请求状态的映射字典
        self.parent_requests: dict[str, ParentRequest] = {}  # 父请求ID到父请求的映射字典（并行采样）
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)  # 外部请求ID到内部请求ID列表的映射
        self.lora_states = LoRARequestStates(log_stats)  # LoRA请求状态追踪器
        self.tracing_enabled = tracing_enabled  # 保存是否启用追踪的标志

    def get_num_unfinished_requests(self):
        """获取未完成请求的数量。

        Returns:
            当前未完成的请求数量
        """
        return len(self.request_states)  # 返回请求状态字典的长度

    def has_unfinished_requests(self) -> bool:
        """检查是否有未完成的请求。

        Returns:
            如果有未完成的请求返回True
        """
        return len(self.request_states) > 0  # 判断请求状态字典是否非空

    def propagate_error(self, e: Exception):
        """将错误传播到所有 generate() 任务。

        当引擎遇到致命错误时调用，将异常推送到每个请求的队列中。

        Args:
            e: 要传播的异常
        """
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():  # 遍历所有请求状态
            assert state.queue is not None  # 断言队列存在
            state.queue.put(e)  # 将异常放入队列

    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """中止一组请求。

        请求ID可以是外部请求ID或内部请求ID。
        如果提供外部请求ID，所有关联的内部请求都会被中止。
        对于并行采样，中止父请求会同时中止所有子请求。

        Args:
            request_ids: 要中止的请求ID列表
            internal: 是否为内部请求ID

        Returns:
            实际被中止的内部请求ID列表

        Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        internal_req_ids = []  # 收集所有需要中止的内部请求ID
        for request_id in request_ids:  # 遍历输入的请求ID
            if internal:  # 如果输入的是内部ID
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)  # 直接添加到内部ID列表

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):  # 如果请求状态存在
                    external_req_id = req_state.external_req_id  # 获取对应的外部ID
                    internal_ids = self.external_req_ids[external_req_id]  # 获取外部ID对应的内部ID列表
                    internal_ids.remove(request_id)  # 从映射中移除该内部ID
                    if not internal_ids:  # 如果映射中没有更多内部ID
                        del self.external_req_ids[external_req_id]  # 删除整个外部ID映射
            elif internal_ids := self.external_req_ids.pop(request_id, []):  # 如果是外部ID，弹出所有关联的内部ID
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)  # 将所有关联的内部ID加入中止列表

        request_ids_to_abort = []  # 最终需要在引擎核心中中止的请求ID列表
        for request_id in internal_req_ids:  # 遍历所有需要中止的内部ID
            req_state = self.request_states.pop(request_id, None)  # 从状态字典中移除请求状态
            if req_state is not None:  # 如果请求状态存在
                self.lora_states.request_finished(request_id, req_state.lora_name)  # 更新LoRA状态
                request_ids_to_abort.append(request_id)  # 添加到中止列表
                # Produce final abort output.
                if req_state.queue is not None and (  # 如果有输出队列
                    request_output := req_state.make_request_output(  # 创建最终的中止输出
                        new_token_ids=[],  # 没有新的token
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR  # 如果没有反分词器（池化请求），使用空张量占位
                        if req_state.detokenizer is None
                        else None,  # 如果有反分词器（生成请求），设为None
                        finish_reason=FinishReason.ABORT,  # 完成原因为中止
                        stop_reason=None,  # 无停止原因
                        kv_transfer_params=None,  # 无KV传输参数
                    )
                ):
                    req_state.queue.put(request_output)  # 将中止输出放入队列
            elif parent := self.parent_requests.get(request_id):  # 如果是父请求ID
                # Abort children prior to removing the parent.
                if parent.child_requests:  # 如果父请求有子请求
                    child_reqs = list(parent.child_requests)  # 获取所有子请求ID
                    child_reqs = self.abort_requests(child_reqs, internal=True)  # 递归中止所有子请求
                    request_ids_to_abort.extend(child_reqs)  # 将子请求ID加入中止列表
                self.parent_requests.pop(request_id, None)  # 移除父请求
        return request_ids_to_abort  # 返回所有需要中止的请求ID

    def add_request(
        self,
        request: EngineCoreRequest,  # 引擎核心请求
        prompt: str | None,  # 提示文本
        parent_req: ParentRequest | None = None,  # 父请求（并行采样）
        request_index: int = 0,  # 请求索引
        queue: RequestOutputCollector | None = None,  # 输出收集器
    ) -> None:
        """添加新请求或更新已有的流式输入请求。

        如果请求ID已存在（流式输入场景），更新现有请求状态；
        否则创建新的请求状态。

        Args:
            request: 引擎核心请求
            prompt: 提示文本
            parent_req: 并行采样的父请求
            request_index: 请求索引
            queue: 输出收集器
        """
        request_id = request.request_id  # 获取请求ID
        req_state = self.request_states.get(request_id)  # 查找是否已有该请求的状态
        if req_state is not None:  # 如果请求状态已存在（流式输入的后续块）
            self._update_streaming_request_state(req_state, request, prompt)  # 更新流式请求状态
            return  # 直接返回，不创建新状态

        req_state = RequestState.from_new_request(  # 从新请求创建请求状态
            tokenizer=self.tokenizer,  # 分词器
            request=request,  # 请求
            prompt=prompt,  # 提示文本
            parent_req=parent_req,  # 父请求
            request_index=request_index,  # 请求索引
            queue=queue,  # 输出收集器
            log_stats=self.log_stats,  # 是否记录统计
            stream_interval=self.stream_interval,  # 流式间隔
        )
        self.request_states[request_id] = req_state  # 将新状态存入字典
        if parent_req:  # 如果有父请求
            self.parent_requests[parent_req.request_id] = parent_req  # 注册父请求

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)  # 记录外部ID到内部ID的映射

    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """更新流式输入请求的状态，将更新入队或直接应用。

        Args:
            req_state: 现有的请求状态
            request: 新的引擎核心请求（包含增量数据）
            prompt: 增量提示文本
        """
        """Queue a streaming update instead of immediately applying it."""
        if not request.resumable:  # 如果请求不可恢复（即最终请求）
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:  # 如果引擎已完成处理（队列已被设为None）
                # Engine already finished - emit final output and clean up.
                self._finish_request(req_state)  # 清理请求状态
                if req_state.queue is not None:  # 如果有输出队列
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    req_state.queue.put(STREAM_FINISHED)  # 放入流式完成标记以解除阻塞
            elif req_state.input_chunk_queue:  # 如果队列中还有待处理的更新
                req_state.input_chunk_queue[-1].final = True  # 将最后一个更新标记为最终更新
            else:  # 如果队列为空（当前块正在处理）
                req_state.streaming_input = False  # 直接关闭流式输入标记
            return  # 返回

        update = StreamingUpdate(  # 创建流式更新对象
            prompt=prompt,  # 增量提示文本
            prompt_token_ids=request.prompt_token_ids,  # 增量token ID
            arrival_time=request.arrival_time,  # 到达时间
        )

        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:  # 如果上一个输入已完成（队列为None）
            req_state.apply_streaming_update(update)  # 立即应用更新
            req_state.input_chunk_queue = deque()  # 重新创建队列
        else:  # 如果上一个输入还在处理中
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)  # 将更新加入队列等待

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],  # 引擎核心输出列表
        engine_core_timestamp: float | None = None,  # 引擎核心时间戳
        iteration_stats: IterationStats | None = None,  # 迭代统计对象
    ) -> OutputProcessorOutput:
        """处理引擎核心输出，转换为请求输出。

        在单次循环中完成以下步骤：
        1) 计算统计信息
        2) 反分词（token ID转文本）并检查停止字符串
        3) 计算对数概率
        4) 创建并分发RequestOutput对象
        5) 清理已完成请求，收集需要中止的请求

        注意：为最小化Python循环开销，这是唯一遍历所有输出的函数。

        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []  # 初始化请求输出列表
        reqs_to_abort: list[str] = []  # 初始化需要中止的请求ID列表
        for engine_core_output in engine_core_outputs:  # 遍历每个引擎核心输出
            req_id = engine_core_output.request_id  # 获取请求ID
            req_state = self.request_states.get(req_id)  # 查找请求状态
            if req_state is None:  # 如果请求状态不存在（已被中止的请求）
                # Ignore output for already-aborted request.
                continue  # 跳过该输出

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(  # 更新迭代统计信息
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids  # 获取新生成的token ID
            pooling_output = engine_core_output.pooling_output  # 获取池化输出
            finish_reason = engine_core_output.finish_reason  # 获取完成原因
            stop_reason = engine_core_output.stop_reason  # 获取停止原因
            kv_transfer_params = engine_core_output.kv_transfer_params  # 获取KV传输参数
            routed_experts = engine_core_output.routed_experts  # 获取路由专家信息
            req_state.num_cached_tokens = engine_core_output.num_cached_tokens  # 更新缓存token数
            req_state.is_prefilling = False  # 标记预填充已完成

            if pooling_output is None:  # 如果不是池化输出（是生成任务）
                assert req_state.detokenizer is not None  # 断言反分词器存在
                assert req_state.logprobs_processor is not None  # 断言对数概率处理器存在
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(  # 反分词并检查停止字符串
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:  # 如果检测到停止字符串
                    finish_reason = FinishReason.STOP  # 设置完成原因为停止
                    stop_reason = stop_string  # 设置停止原因为匹配到的字符串

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)  # 更新对数概率

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(  # 创建请求输出
                new_token_ids,  # 新token ID
                pooling_output,  # 池化输出
                finish_reason,  # 完成原因
                stop_reason,  # 停止原因
                kv_transfer_params,  # KV传输参数
                routed_experts,  # 路由专家
            ):
                if req_state.streaming_input:  # 如果是流式输入请求
                    request_output.finished = False  # 强制标记为未完成（还有更多输入块）

                if req_state.queue is not None:  # 如果有输出队列（AsyncLLM模式）
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)  # 将输出放入异步队列
                else:  # 如果没有队列（LLMEngine模式）
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)  # 添加到返回列表

            # Free completed requests.
            if finish_reason is not None:  # 如果请求已完成
                if req_state.streaming_input:  # 如果是流式输入请求
                    if req_state.input_chunk_queue:  # 如果队列中还有待处理的输入块
                        update = req_state.input_chunk_queue.popleft()  # 取出下一个更新
                        req_state.apply_streaming_update(update)  # 应用更新，开始处理下一个输入块
                    else:  # 如果没有更多输入块
                        req_state.input_chunk_queue = None  # 将队列设为None，标记引擎侧已完成
                else:  # 非流式输入请求
                    self._finish_request(req_state)  # 清理请求状态
                    if not engine_core_output.finished:  # 如果引擎核心认为请求未完成
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)  # 需要在引擎核心中中止（因反分词器检测到停止字符串）

                    # Track per-request stats
                    self._update_stats_from_finished(  # 更新完成请求的统计信息
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:  # 如果启用了追踪
                        self.do_tracing(engine_core_output, req_state, iteration_stats)  # 记录追踪跨度

        return OutputProcessorOutput(  # 返回处理结果
            request_outputs=request_outputs,  # 请求输出列表
            reqs_to_abort=reqs_to_abort,  # 需要中止的请求ID列表
        )

    def _finish_request(self, req_state: RequestState) -> None:
        """清理已完成请求的状态。

        从请求状态字典和外部ID映射中移除该请求，
        并在适当时清理父请求。

        Args:
            req_state: 要清理的请求状态
        """
        req_id = req_state.request_id  # 获取请求ID
        self.request_states.pop(req_id)  # 从请求状态字典中移除

        internal_ids = self.external_req_ids[req_state.external_req_id]  # 获取外部ID对应的内部ID列表
        internal_ids.remove(req_id)  # 从列表中移除该内部ID
        if not internal_ids:  # 如果列表为空
            del self.external_req_ids[req_state.external_req_id]  # 删除整个外部ID映射

        # Remove parent request if applicable.
        parent_req = req_state.parent_req  # 获取父请求引用
        if parent_req and not parent_req.child_requests:  # 如果有父请求且所有子请求都已完成
            self.parent_requests.pop(parent_req.request_id, None)  # 移除父请求

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        """更新调度器统计信息。

        Args:
            scheduler_stats: 调度器统计数据
        """
        self.lora_states.update_scheduler_stats(scheduler_stats)  # 将调度器统计传递给LoRA状态追踪器

    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,  # 引擎核心输出
        req_state: RequestState,  # 请求状态
        iteration_stats: IterationStats | None,  # 迭代统计
    ) -> None:
        """为已完成的请求创建分布式追踪跨度。

        记录请求的时延指标（TTFT、端到端延迟、排队时间等）
        和请求参数。

        Args:
            engine_core_output: 引擎核心输出
            req_state: 请求状态
            iteration_stats: 迭代统计
        """
        assert req_state.stats is not None  # 断言请求统计存在
        assert iteration_stats is not None  # 断言迭代统计存在

        metrics = req_state.stats  # 获取请求统计指标
        arrival_time_ns = int(metrics.arrival_time * 1e9)  # 将到达时间转换为纳秒
        trace_context = extract_trace_context(engine_core_output.trace_headers)  # 从输出头中提取追踪上下文
        prompt_length = length_from_prompt_token_ids_or_embeds(  # 计算提示长度
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time  # 计算端到端延迟
        queued_time = metrics.scheduled_ts - metrics.queued_ts  # 计算排队等待时间
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts  # 计算预填充时间
        decode_time = metrics.last_token_ts - metrics.first_token_ts  # 计算解码时间
        inference_time = metrics.last_token_ts - metrics.scheduled_ts  # 计算总推理时间

        # Build attributes dict
        attributes: dict[str, Any] = {  # 构建追踪属性字典
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (  # 首token延迟
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,  # 端到端延迟
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,  # 排队时间
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,  # 提示token数
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (  # 生成token数
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,  # 模型预填充时间
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,  # 模型解码时间
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,  # 模型推理总时间
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,  # 请求ID
        }

        # Add optional request parameters
        if req_state.top_p:  # 如果有top-p参数
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p  # 添加top-p属性
        if req_state.max_tokens_param:  # 如果有最大token数参数
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (  # 添加最大token数属性
                req_state.max_tokens_param
            )
        if req_state.temperature:  # 如果有温度参数
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (  # 添加温度属性
                req_state.temperature
            )
        if req_state.n:  # 如果有并行采样数参数
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n  # 添加并行采样数属性

        instrument_manual(  # 创建手动追踪跨度
            span_name="llm_request",  # 跨度名称
            start_time=arrival_time_ns,  # 起始时间（纳秒）
            attributes=attributes,  # 追踪属性
            context=trace_context,  # 追踪上下文
            kind=SpanKind.SERVER,  # 跨度类型为服务端
        )

    def _update_stats_from_output(
        self,
        req_state: RequestState,  # 请求状态
        engine_core_output: EngineCoreOutput,  # 引擎核心输出
        engine_core_timestamp: float | None,  # 引擎核心时间戳
        iteration_stats: IterationStats | None,  # 迭代统计
    ):
        """从引擎核心输出更新统计信息。

        在每次迭代中为每个请求调用，更新吞吐量和延迟统计。

        Args:
            req_state: 请求状态
            engine_core_output: 引擎核心输出
            engine_core_timestamp: 引擎核心时间戳
            iteration_stats: 迭代统计
        """
        if iteration_stats is None:  # 如果没有迭代统计对象
            return  # 直接返回，不做统计

        assert engine_core_timestamp is not None  # 断言引擎核心时间戳存在
        assert req_state.stats is not None  # 断言请求统计存在
        iteration_stats.update_from_output(  # 调用迭代统计的更新方法
            engine_core_output,  # 引擎核心输出
            engine_core_timestamp,  # 引擎核心时间戳
            req_state.is_prefilling,  # 是否正在预填充
            req_state.prompt_len,  # 提示长度
            req_state.stats,  # 请求统计
            self.lora_states,  # LoRA状态
            req_state.lora_name,  # LoRA名称
        )

    def _update_stats_from_finished(
        self,
        req_state: RequestState,  # 请求状态
        finish_reason: FinishReason | None,  # 完成原因
        iteration_stats: IterationStats | None,  # 迭代统计
    ):
        """从已完成请求更新统计信息。

        当请求完成时调用，记录最终统计数据包括完成原因和token数量。

        Args:
            req_state: 请求状态
            finish_reason: 完成原因
            iteration_stats: 迭代统计
        """
        if iteration_stats is None:  # 如果没有迭代统计对象
            return  # 直接返回

        assert finish_reason is not None  # 断言完成原因不为空
        assert req_state.stats is not None  # 断言请求统计存在
        iteration_stats.update_from_finished_request(  # 调用迭代统计的完成请求更新方法
            finish_reason=finish_reason,  # 完成原因
            num_prompt_tokens=req_state.prompt_len,  # 提示token数
            max_tokens_param=req_state.max_tokens_param,  # 最大token数参数
            req_stats=req_state.stats,  # 请求统计
            num_cached_tokens=req_state.num_cached_tokens,  # 缓存token数
        )
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)  # 更新LoRA完成状态

        ParentRequest.observe_finished_request(  # 通知父请求有子请求完成（用于并行采样统计）
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )
