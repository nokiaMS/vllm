# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy  # 导入浅拷贝函数，用于复制采样参数对象
from typing import cast  # 导入类型转换函数，用于类型提示中的强制转换

from vllm.outputs import CompletionOutput  # 导入补全输出类，表示单个生成结果
from vllm.sampling_params import RequestOutputKind, SamplingParams  # 导入请求输出类型枚举和采样参数类
from vllm.v1.engine import EngineCoreRequest  # 导入引擎核心请求类，表示用户的推理请求
from vllm.v1.metrics.stats import IterationStats  # 导入迭代统计类，用于记录每轮推理的统计信息


# [中文注释] 并行采样（n>1）的父请求管理器。当 sampling_params.n > 1 时：
#   1. 一个用户请求被拆分为 n 个子请求（child_request），每个子请求独立推理
#   2. ParentRequest 负责：
#      - 生成子请求的 ID（格式: "{index}_{parent_request_id}"）和 sampling_params（含不同 seed）
#      - 跟踪子请求完成状态 (child_requests set)
#      - FINAL_ONLY 模式下聚合所有子请求结果后一次性返回
#      - 流式模式下逐个返回子请求结果
#   3. cached_child_sampling_params — 无 seed 时复用同一份参数，避免重复拷贝
class ParentRequest:  # 并行采样父请求类，管理n>1时的多个子请求
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.

    并行采样请求的信息、状态和处理逻辑。
    存储父请求的ID和采样参数。
    负责生成子请求的采样参数，聚合子请求的输出结果。
    """

    request_id: str  # 父请求的唯一标识符
    external_req_id: str  # 外部请求ID，用于与客户端通信的标识符
    sampling_params: SamplingParams  # 父请求的采样参数，包含n、seed等配置

    # To track the completion of child requests
    child_requests: set[str]  # 子请求ID集合，用于跟踪尚未完成的子请求

    # To aggregate child completions when not streaming
    output_aggregator: list[CompletionOutput]  # 输出聚合列表，非流式模式下收集所有子请求的最终结果

    # To find the max number of generated tokens across all children
    max_num_generation_tokens: int  # 所有子请求中生成的最大token数量

    # To efficiently obtain child sampling params
    cached_child_sampling_params: SamplingParams | None  # 缓存的子请求采样参数，无seed时可复用同一份以提高效率

    def __init__(self, request: EngineCoreRequest) -> None:  # 构造函数，根据引擎核心请求初始化父请求
        """初始化父请求对象。

        根据引擎核心请求创建父请求，设置请求ID、采样参数，
        并初始化子请求跟踪集合和输出聚合器。

        Args:
            request: 引擎核心请求对象，包含请求ID和采样参数等信息。
        """
        assert request.external_req_id is not None  # 断言外部请求ID不为空，确保请求合法
        sampling_params = request.params  # 从请求中获取采样参数
        self.request_id = request.request_id  # 设置父请求ID
        self.external_req_id = request.external_req_id  # 设置外部请求ID
        self.sampling_params = sampling_params  # 保存采样参数

        self.child_requests = set()  # 初始化子请求集合为空集
        self.output_aggregator = (  # 初始化输出聚合器
            [cast(CompletionOutput, None)] * sampling_params.n  # 非流式模式：预分配n个槽位，初始值为None（通过cast绕过类型检查）
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)  # 判断是否为仅返回最终结果模式
            else []  # 流式模式：使用空列表，不需要聚合
        )
        self.max_num_generation_tokens = 0  # 初始化最大生成token数为0
        self.cached_child_sampling_params = None  # 初始化缓存的子请求采样参数为None

    def _get_child_sampling_params(  # 内部方法：高效获取子请求的采样参数
        self,  # 实例自身引用
        index: int,  # 子请求在n个并行请求中的索引
    ) -> SamplingParams:  # 返回子请求的采样参数实例
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.

        高效地获取子请求的采样参数。

        如果父请求的seed不为None，则每个子请求需要一个独立的采样参数副本，
        并且带有唯一的seed值。如果seed为None，则所有子请求共享同一个采样参数实例。

        Args:
            index: 子请求在n个并行请求中的索引。

        Returns:
            子请求的采样参数实例。
        """
        seed = self.sampling_params.seed  # 获取父请求的随机种子
        if self.cached_child_sampling_params:  # 如果已有缓存的子请求采样参数
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params  # 直接返回缓存的参数，避免重复创建
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)  # 浅拷贝父请求的采样参数，创建子请求的参数副本
        child_sampling_params.n = 1  # 将子请求的并行数设为1，因为每个子请求独立生成
        if seed is None:  # 如果没有设置随机种子
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params  # 缓存参数以供后续子请求复用
        else:  # 如果设置了随机种子
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index  # 为每个子请求设置唯一的种子值（父种子+索引）
        return child_sampling_params  # 返回子请求的采样参数

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:  # 获取子请求的ID和采样参数
        """Get child request ID and sampling params.

        Args:
          index: index within `n` child requests.

        Returns:
          (request ID, sampling_params) tuple

        获取子请求的ID和采样参数。

        根据索引生成子请求的唯一ID，并将其加入跟踪集合，
        同时返回对应的采样参数。

        Args:
            index: 子请求在n个并行请求中的索引。

        Returns:
            包含（子请求ID, 采样参数）的元组。
        """
        child_req_id = f"{index}_{self.request_id}"  # 生成子请求ID，格式为"索引_父请求ID"
        self.child_requests.add(child_req_id)  # 将子请求ID加入跟踪集合
        return child_req_id, self._get_child_sampling_params(index)  # 返回子请求ID和对应的采样参数

    @property  # 将n方法声明为只读属性
    def n(self) -> int:  # 获取并行采样数量n的属性方法
        """获取并行采样数量n。

        Returns:
            采样参数中配置的并行生成数量。
        """
        return self.sampling_params.n  # 返回采样参数中的并行数n

    def get_outputs(  # 获取并处理子请求的输出结果
        self,  # 实例自身引用
        child_request_id: str,  # 子请求的唯一标识符
        completion_output: CompletionOutput,  # 子请求的补全输出对象
    ) -> tuple[list[CompletionOutput], bool]:  # 返回（输出列表, 是否全部完成）元组
        """获取并处理子请求的输出结果。

        根据子请求的完成状态更新跟踪信息，并根据输出模式（流式/非流式）
        决定是立即返回还是聚合后返回。

        Args:
            child_request_id: 子请求的唯一标识符。
            completion_output: 子请求的补全输出结果。

        Returns:
            包含（输出列表, 是否所有子请求都已完成）的元组。
        """
        already_finished_and_returned: bool = False  # 标记该子请求是否已在之前完成并返回过
        if completion_output.finished():  # 如果当前输出表示子请求已完成
            if child_request_id in self.child_requests:  # 如果子请求ID仍在跟踪集合中
                self.child_requests.remove(child_request_id)  # 从跟踪集合中移除已完成的子请求
            else:  # 如果子请求ID不在跟踪集合中
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                already_finished_and_returned = True  # 标记为已完成并返回过（上一批次已处理）

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:  # 如果不是仅返回最终结果模式（即流式模式）
            # If streaming, just return the current output
            #
            # DO NOT output finished and already returned child request to client again
            outputs = [] if already_finished_and_returned else [completion_output]  # 流式模式：如果已返回过则返回空列表，否则返回当前输出
        else:  # 非流式模式（仅返回最终结果）
            # If not streaming, aggregate the n final outputs.
            self.output_aggregator[completion_output.index] = completion_output  # 将子请求结果放入聚合列表的对应索引位置
            outputs = [] if self.child_requests else self.output_aggregator  # 如果还有未完成的子请求返回空列表，否则返回聚合的全部结果

        finished = not self.child_requests  # 判断是否所有子请求都已完成（集合为空即全部完成）
        return outputs, finished  # 返回输出列表和完成状态

    def observe_num_generation_tokens(self, num_generation_tokens: int):  # 观察并更新子请求生成的最大token数
        """观察并更新子请求生成的最大token数量。

        将当前子请求的生成token数与已记录的最大值比较，
        保留较大的值。

        Args:
            num_generation_tokens: 当前子请求生成的token数量。

        Returns:
            所有子请求中生成的最大token数量。
        """
        self.max_num_generation_tokens = max(  # 更新最大生成token数
            num_generation_tokens, self.max_num_generation_tokens  # 取当前值和历史最大值中的较大者
        )
        return self.max_num_generation_tokens  # 返回更新后的最大生成token数

    @staticmethod  # 声明为静态方法，不需要实例引用
    def observe_finished_request(  # 观察已完成的请求并记录统计信息
        parent_req: "ParentRequest | None",  # 父请求对象，None表示非并行采样
        iteration_stats: IterationStats,  # 迭代统计对象，收集推理统计数据
        num_generation_tokens: int,  # 当前完成请求生成的token数量
    ):  # 无返回值
        """观察已完成的请求并记录统计信息。

        当子请求完成时调用此方法，更新最大生成token数，
        并在所有子请求都完成后将统计数据记录到迭代统计对象中。

        Args:
            parent_req: 父请求对象，如果为None表示非并行采样请求（n=1）。
            iteration_stats: 迭代统计对象，用于收集本轮推理的统计数据。
            num_generation_tokens: 当前完成的请求生成的token数量。
        """
        n_param = parent_req.n if parent_req is not None else 1  # 获取并行数n，如果没有父请求则为1

        if parent_req is not None:  # 如果存在父请求（即并行采样场景）
            num_generation_tokens = parent_req.observe_num_generation_tokens(  # 更新并获取所有子请求中的最大生成token数
                num_generation_tokens  # 传入当前子请求的生成token数
            )

        # Child requests finished, we can now record to iteration stats
        if parent_req is None or not parent_req.child_requests:  # 如果不是并行采样，或者所有子请求都已完成
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)  # 将最大生成token数追加到迭代统计中
            iteration_stats.n_params_iter.append(n_param)  # 将并行数n追加到迭代统计中
