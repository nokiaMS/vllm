# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools  # 导入迭代器工具模块
import time  # 导入时间模块
from collections import defaultdict, deque  # 导入默认字典和双端队列
from collections.abc import Iterable  # 导入可迭代类型
from dataclasses import replace  # 导入数据类替换函数
from typing import Any  # 导入Any类型注解

import numpy as np  # 导入NumPy数组库

from vllm import envs  # 导入vLLM环境变量
from vllm.compilation.cuda_graph import CUDAGraphStat  # 导入CUDA图统计信息
from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.distributed.ec_transfer.ec_connector.base import (  # 导入编码器缓存连接器基类
    ECConnectorMetadata,  # 编码器缓存连接器元数据
    ECConnectorRole,  # 编码器缓存连接器角色
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory  # 导入编码器缓存连接器工厂
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch  # 导入KV事件发布工厂和事件批次
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory  # 导入KV传输连接器工厂
from vllm.distributed.kv_transfer.kv_connector.v1 import (  # 导入KV传输连接器V1版本
    KVConnectorBase_V1,  # KV连接器V1基类
    KVConnectorRole,  # KV连接器角色（SCHEDULER/WORKER）
    SupportsHMA,  # 支持混合内存分配器接口
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata  # 导入KV连接器元数据
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats  # 导入KV连接器统计信息
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (  # 导入路由专家捕获器
    RoutedExpertsReader,  # 路由专家读取器
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 导入多模态注册表
from vllm.multimodal.encoder_budget import MultiModalBudget  # 导入多模态编码器预算
from vllm.v1.core.encoder_cache_manager import (  # 导入编码器缓存管理器
    EncoderCacheManager,  # 纯解码器模型的编码器缓存管理器
    EncoderDecoderCacheManager,  # 编码器-解码器模型的缓存管理器
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager  # 导入KV缓存块和管理器
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector  # 导入KV缓存指标收集器
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface  # 导入暂停状态和调度器接口
from vllm.v1.core.sched.output import (  # 导入调度器输出相关类
    CachedRequestData,  # 已缓存请求的数据
    GrammarOutput,  # 结构化输出语法结果
    NewRequestData,  # 新请求的数据
    SchedulerOutput,  # 调度器输出
)
from vllm.v1.core.sched.request_queue import (  # 导入请求队列相关
    RequestQueue,  # 请求队列
    SchedulingPolicy,  # 调度策略（FCFS/PRIORITY）
    create_request_queue,  # 创建请求队列工厂函数
)
from vllm.v1.core.sched.utils import check_stop, remove_all  # 导入停止条件检查和批量移除工具
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs  # 导入引擎核心输出类型
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig  # 导入注意力规格和KV缓存配置
from vllm.v1.metrics.perf import ModelMetrics, PerfStats  # 导入模型性能指标
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats  # 导入前缀缓存和调度器统计
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput  # 导入模型运行器输出类
from vllm.v1.request import Request, RequestStatus, StreamingUpdate  # 导入请求、请求状态和流式更新
from vllm.v1.spec_decode.metrics import SpecDecodingStats  # 导入推测解码统计
from vllm.v1.structured_output import StructuredOutputManager  # 导入结构化输出管理器
from vllm.v1.utils import record_function_or_nullcontext  # 导入性能记录上下文管理器

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# [中文注释] vLLM V1 核心调度器，实现 SchedulerInterface 接口。
#   调度算法核心思想：没有 prefill/decode 阶段划分。
#     每个请求有 num_computed_tokens 和 num_tokens_with_spec，
#     每步让 computed 追赶 total，统一处理 chunked prefill、prefix caching、
#     speculative decoding。
#   调度流程（schedule() 方法）：
#     1. 调度 RUNNING 请求：为每个请求分配新 token slot，不足时抢占低优先级请求
#     2. 调度 WAITING 请求：查找前缀缓存命中 → 分配 KV block → 移入 running 队列
#   关键组件：
#     kv_cache_manager — KV cache block 分配与管理
#     encoder_cache_manager — 多模态编码器缓存
#     connector — KV 传输连接器（P/D 分离）
#     structured_output_manager — 结构化输出语法约束
class Scheduler(SchedulerInterface):
    # [中文注释] __init__() — 调度器初始化方法。
    #   参数说明：
    #     vllm_config: vLLM全局配置
    #     kv_cache_config: KV缓存配置（块大小、块数量等）
    #     structured_output_manager: 结构化输出管理器（语法约束）
    #     block_size: KV缓存块大小
    #     mm_registry: 多模态注册表（管理视觉/音频等模态）
    #     include_finished_set: 是否包含已完成请求集合（多引擎场景）
    #     log_stats: 是否记录统计信息
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config  # 保存vLLM全局配置
        self.scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.cache_config = vllm_config.cache_config  # 缓存配置
        self.lora_config = vllm_config.lora_config  # LoRA适配器配置
        self.kv_cache_config = kv_cache_config  # KV缓存配置
        self.kv_events_config = vllm_config.kv_events_config  # KV事件配置
        self.parallel_config = vllm_config.parallel_config  # 并行配置
        self.log_stats = log_stats  # 是否记录统计信息
        self.observability_config = vllm_config.observability_config  # 可观测性配置
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None  # KV缓存指标收集器
        if self.observability_config.kv_cache_metrics:  # 如果启用了KV缓存指标
            self.kv_metrics_collector = KVCacheMetricsCollector(  # 创建KV缓存指标收集器
                self.observability_config.kv_cache_metrics_sample,  # 采样配置
            )
        self.structured_output_manager = structured_output_manager  # 结构化输出管理器
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder  # 是否为编码器-解码器模型

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        # 已完成请求ID字典，按客户端索引分组（多引擎场景使用）
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()  # 上一步调度的请求ID集合

        # Scheduling constraints.
        # 调度约束参数
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs  # 最大并发运行请求数
        self.max_num_scheduled_tokens = (  # 每步最大调度token数
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len  # 模型最大序列长度
        self.enable_kv_cache_events = (  # 是否启用KV缓存事件
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        # 创建KV连接器（用于P/D分离架构的KV缓存传输）
        self.connector = None  # KV传输连接器（调度器侧）
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None  # 连接器前缀缓存统计
        self.recompute_kv_load_failures = True  # KV加载失败时是否重新计算
        if self.vllm_config.kv_transfer_config is not None:  # 如果配置了KV传输
            assert not self.is_encoder_decoder, (  # 编码器-解码器模型暂不支持KV连接器
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(  # 创建KV连接器实例
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,  # 角色为调度器
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:  # 如果启用统计
                self.connector_prefix_cache_stats = PrefixCacheStats()  # 初始化前缀缓存统计
            kv_load_failure_policy = (  # 获取KV加载失败策略
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"  # 判断是否采用重计算策略

        self.kv_event_publisher = EventPublisherFactory.create(  # 创建KV事件发布器
            self.kv_events_config,
            self.parallel_config.data_parallel_index,  # 数据并行索引
        )
        self.ec_connector = None  # 编码器缓存连接器
        if self.vllm_config.ec_transfer_config is not None:  # 如果配置了编码器缓存传输
            self.ec_connector = ECConnectorFactory.create_connector(  # 创建编码器缓存连接器
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks  # 获取GPU块数量
        assert num_gpu_blocks is not None and num_gpu_blocks > 0  # 确保GPU块数量有效

        self.block_size = block_size  # KV缓存块大小
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size  # 解码上下文并行大小
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size  # 预填充上下文并行大小

        # req_id -> Request
        self.requests: dict[str, Request] = {}  # 请求ID到请求对象的映射
        # Scheduling policy
        # 调度策略（FCFS先来先服务 / PRIORITY优先级）
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)  # 解析调度策略
        except ValueError as e:
            raise ValueError(  # 未知调度策略时抛出异常
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)  # 等待队列（存放待调度的请求）
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)  # 跳过的等待队列（异步依赖或约束导致暂时无法调度的请求）
        self.running: list[Request] = []  # 正在运行的请求列表

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # 两步之间已完成的请求ID集合，用于通知worker释放缓存状态
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        # 等待流式输入的请求计数器
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        # KV连接器：异步KV加载/接收中的请求ID集合
        self.finished_recving_kv_req_ids: set[str] = set()  # 已完成接收KV的请求
        self.failed_recving_kv_req_ids: set[str] = set()  # 接收KV失败的请求

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # 编码器相关初始化：计算编码器缓存大小
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(  # 检查模型是否支持多模态输入
            vllm_config.model_config
        )
        mm_budget = (  # 创建多模态预算（管理编码器token预算和缓存大小）
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        # 注意：纯文本编码器-解码器模型也通过多模态接口实现
        if self.is_encoder_decoder:  # 编码器-解码器模型至多只能有一种模态
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (  # 编码器输入token的最大数量
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0  # 编码器缓存大小
        self.encoder_cache_manager = (  # 创建编码器缓存管理器（根据模型类型选择不同管理器）
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        # 推测解码配置
        speculative_config = vllm_config.speculative_config  # 获取推测解码配置
        self.use_eagle = False  # 是否使用EAGLE推测解码
        self.num_spec_tokens = self.num_lookahead_tokens = 0  # 推测token数和前瞻token数
        if speculative_config:  # 如果启用了推测解码
            self.num_spec_tokens = speculative_config.num_speculative_tokens  # 推测token数量
            if speculative_config.use_eagle():  # 如果使用EAGLE算法
                self.use_eagle = True  # 标记使用EAGLE
                self.num_lookahead_tokens = self.num_spec_tokens  # 前瞻token数等于推测token数
            if speculative_config.uses_draft_model():  # 如果使用草稿模型
                self.num_lookahead_tokens = self.num_spec_tokens  # 设置前瞻token数

        # Create the KV cache manager.
        # 创建KV缓存管理器
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,  # KV缓存配置
            max_model_len=self.max_model_len,  # 最大模型长度
            enable_caching=self.cache_config.enable_prefix_caching,  # 是否启用前缀缓存
            use_eagle=self.use_eagle,  # 是否使用EAGLE
            log_stats=self.log_stats,  # 是否记录统计
            enable_kv_cache_events=self.enable_kv_cache_events,  # 是否启用KV缓存事件
            dcp_world_size=self.dcp_world_size,  # 解码上下文并行大小
            pcp_world_size=self.pcp_world_size,  # 预填充上下文并行大小
            hash_block_size=self.block_size,  # 哈希块大小
            metrics_collector=self.kv_metrics_collector,  # 指标收集器
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1  # 是否使用流水线并行
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER  # 是否使用V2模型运行器

        self.has_mamba_layers = kv_cache_config.has_mamba_layers  # 是否包含Mamba层
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing  # 是否需要KV缓存清零
        self.need_mamba_block_aligned_split = (  # 是否需要Mamba块对齐分割
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None  # 性能指标
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:  # 如果启用MFU指标
            self.perf_metrics = ModelMetrics(vllm_config)  # 创建模型性能指标

        # 路由专家（MoE混合专家模型）相关配置
        if self.vllm_config.model_config.enable_return_routed_experts:  # 如果启用返回路由专家信息
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (  # 路由专家不支持上下文并行
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()  # 创建路由专家读取器

            assert len(kv_cache_config.kv_cache_groups) > 0, (  # 至少需要一个KV缓存组
                "enable_return_routed_experts requires at least one kv cache group"
            )
            # Find the attention group for routed experts indexing.
            # 找到用于路由专家索引的注意力组
            self.routed_experts_attn_gid = 0  # 路由专家使用的注意力组ID
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):  # 遍历KV缓存组
                if isinstance(group.kv_cache_spec, AttentionSpec):  # 找到注意力规格的组
                    self.routed_experts_attn_gid = gid  # 记录该组ID
                    break
            min_block_size = min(  # 计算所有缓存组中的最小块大小
                [
                    group.kv_cache_spec.block_size
                    for group in kv_cache_config.kv_cache_groups
                ]
            )
            num_groups = len(kv_cache_config.kv_cache_groups)  # 缓存组数量
            self.max_num_kv_tokens = (  # 计算最大KV token数
                kv_cache_config.num_blocks // num_groups
            ) * min_block_size
            dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size  # 解码上下文并行大小
            pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size  # 预填充上下文并行大小
            if pcp_size * dcp_size > 1:  # 如果使用了上下文并行
                self.max_num_kv_tokens *= pcp_size * dcp_size  # 按并行度扩大最大KV token数

            self.routed_experts_reader.attach_buffer(  # 挂载路由专家缓冲区
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED  # 初始化暂停状态为未暂停

    # [中文注释] _mamba_block_aligned_split() — Mamba层块对齐分割。
    #   对于包含Mamba层的混合模型，确保每次调度的新token数量是block_size的倍数，
    #   以便Mamba状态可以按块缓存。
    #   参数：
    #     request: 请求对象
    #     num_new_tokens: 计划调度的新token数
    #     num_new_local_computed_tokens: 本地新计算的token数
    #     num_external_computed_tokens: 外部计算的token数
    #   返回：调整后的新token数
    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (  # 外部KV连接器尚未验证
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (  # 计算总的已计算token数
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        # 在预填充阶段执行块对齐分割
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # 为了启用Mamba状态的块对齐缓存，num_new_tokens必须是block_size的倍数
            block_size = self.cache_config.block_size  # 获取块大小
            last_cache_position = request.num_tokens - request.num_tokens % block_size  # 计算最后一个可缓存位置
            # eagle prune
            if self.use_eagle:  # EAGLE模式下调整最后缓存位置
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens  # 调度后的总token数
            if num_computed_tokens_after_sched < last_cache_position:  # 未到达最后缓存位置
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size  # 对齐到块大小的倍数
            elif (  # 跨越最后缓存位置
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens  # 强制在最后缓存位置处截断
            else:
                # prefill the last few tokens
                pass  # 预填充最后几个token，无需调整
        return num_new_tokens  # 返回调整后的新token数

    # [中文注释] schedule() — 单步调度入口（Engine Core 主循环每步调用一次）。
    #   输出 SchedulerOutput，告诉 model runner 本步处理哪些请求的多少 token。
    #   两阶段调度：
    #     Phase 1: 遍历 running 队列，为每个请求计算 num_new_tokens 并分配 KV block
    #     Phase 2: 遍历 waiting 队列，查前缀缓存 → 分配 slot → 移入 running
    #   约束：max_num_running_reqs（最大并发数）、max_num_scheduled_tokens（token 预算）
    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []  # 本步新调度的请求列表
        scheduled_resumed_reqs: list[Request] = []  # 本步恢复调度的请求列表
        scheduled_running_reqs: list[Request] = []  # 本步正在运行的请求列表
        preempted_reqs: list[Request] = []  # 本步被抢占的请求列表

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}  # 请求ID到新分配KV块的映射
        num_scheduled_tokens: dict[str, int] = {}  # 请求ID到本步调度token数的映射
        token_budget = self.max_num_scheduled_tokens  # 本步token预算
        if self._pause_state == PauseState.PAUSED_ALL:  # 如果调度器已暂停
            # Do not schedule any requests when paused.
            token_budget = 0  # 将token预算设为0，不调度任何请求

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}  # 请求ID到待调度编码器输入索引的映射
        encoder_compute_budget = self.max_num_encoder_input_tokens  # 编码器计算预算
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}  # 请求ID到推测解码token的映射

        # For logging.
        scheduled_timestamp = time.monotonic()  # 记录调度时间戳

        self.kv_cache_manager.new_step_starts()  # 通知KV缓存管理器新的调度步骤开始

        # First, schedule the RUNNING requests.
        # 第一阶段：调度正在运行的请求
        req_index = 0  # 请求索引
        while req_index < len(self.running) and token_budget > 0:  # 遍历running队列直到预算耗尽
            request = self.running[req_index]  # 获取当前请求

            if (  # 检查异步调度下是否已达最大token数，避免多余的调度步骤
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                # 异步调度：确信上一步已达max_tokens时跳过
                req_index += 1  # 跳到下一个请求
                continue

            # 计算请求需要的新token数 = 总token数(含推测) + 输出占位符 - 已计算token数
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:  # 长预填充阈值限制
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)  # 不超过token预算

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            # 确保输入位置不超过模型最大长度（推测解码时需要）
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            # 调度编码器输入（多模态场景）
            encoder_inputs_to_schedule = None  # 待调度的编码器输入列表
            external_load_encoder_input: list[int] = []  # 外部加载的编码器输入
            new_encoder_compute_budget = encoder_compute_budget  # 新的编码器计算预算
            if request.has_encoder_inputs:  # 如果请求有编码器输入
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(  # 尝试调度编码器输入
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,  # EAGLE模式下偏移1
                )

            if self.need_mamba_block_aligned_split:  # 如果需要Mamba块对齐分割
                num_new_tokens = self._mamba_block_aligned_split(  # 执行块对齐分割
                    request, num_new_tokens
                )

            if num_new_tokens == 0:  # 如果没有新token可调度
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1  # 跳到下一个请求
                continue

            # Schedule newly needed KV blocks for the request.
            # 为请求分配新需要的KV缓存块
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:  # 循环尝试分配，分配失败则抢占低优先级请求
                    new_blocks = self.kv_cache_manager.allocate_slots(  # 尝试分配KV缓存槽位
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:  # 分配成功
                        # The request can be scheduled.
                        break  # 跳出循环

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    # 分配失败，需要抢占最低优先级的请求来释放内存
                    if self.policy == SchedulingPolicy.PRIORITY:  # 优先级调度策略
                        preempted_req = max(  # 找到优先级最低的请求（priority和arrival_time最大）
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)  # 从运行队列移除
                        if preempted_req in scheduled_running_reqs:  # 如果已被调度，回退调度状态
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)  # 从已调度列表移除
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)  # 归还token预算
                            req_to_new_blocks.pop(preempted_req_id)  # 移除块分配记录
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)  # 移除推测解码记录
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(  # 移除编码器输入记录
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:  # 如果被抢占请求有编码器输入
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                # 恢复编码器计算预算
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore  # 归还编码器预算
                            req_index -= 1  # 调整索引
                    else:
                        preempted_req = self.running.pop()  # FCFS策略：抢占队列末尾的请求

                    self._preempt_request(preempted_req, scheduled_timestamp)  # 执行抢占操作
                    preempted_reqs.append(preempted_req)  # 记录被抢占的请求
                    if preempted_req == request:  # 如果自己被抢占了
                        # No more request to preempt. Cannot schedule this request.
                        break  # 无法调度当前请求

            if new_blocks is None:  # 分配失败且无法通过抢占解决
                # Cannot schedule this request.
                break  # 停止调度更多running请求

            # Schedule the request.
            # 调度该请求：记录块分配和token数
            scheduled_running_reqs.append(request)  # 加入已调度运行请求列表
            request_id = request.request_id  # 获取请求ID
            req_to_new_blocks[request_id] = new_blocks  # 记录新分配的KV缓存块
            num_scheduled_tokens[request_id] = num_new_tokens  # 记录调度的token数
            token_budget -= num_new_tokens  # 扣减token预算
            req_index += 1  # 移到下一个请求

            # Speculative decode related.
            # 推测解码相关：记录本步调度的推测token
            if request.spec_token_ids:  # 如果请求有推测token
                num_scheduled_spec_tokens = (  # 计算本步调度的推测token数
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:  # 如果有推测token被调度
                    spec_token_ids = request.spec_token_ids  # 获取推测token ID列表
                    if len(spec_token_ids) > num_scheduled_spec_tokens:  # 截断到调度数量
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids  # 记录推测token

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                # 清空推测token，下一步会通过update_draft_token_ids重新设置
                request.spec_token_ids = []

            # Encoder-related.
            # 编码器相关：分配编码器缓存
            if encoder_inputs_to_schedule:  # 如果有编码器输入需要调度
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule  # 记录编码器输入
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:  # 遍历需要调度的编码器输入
                    self.encoder_cache_manager.allocate(request, i)  # 分配编码器缓存
                    if self.ec_connector is not None:  # 如果有编码器缓存连接器
                        self.ec_connector.update_state_after_alloc(request, i)  # 更新连接器状态
                encoder_compute_budget = new_encoder_compute_budget  # 更新编码器计算预算
            if external_load_encoder_input:  # 如果有外部加载的编码器输入
                for i in external_load_encoder_input:  # 遍历外部编码器输入
                    self.encoder_cache_manager.allocate(request, i)  # 分配编码器缓存
                    if self.ec_connector is not None:  # 如果有编码器缓存连接器
                        self.ec_connector.update_state_after_alloc(request, i)  # 更新连接器状态

        # Record the LoRAs in scheduled_running_reqs
        # 记录已调度运行请求中使用的LoRA适配器
        scheduled_loras: set[int] = set()  # 已调度的LoRA ID集合
        if self.lora_config:  # 如果启用了LoRA
            scheduled_loras = set(  # 收集所有使用LoRA的请求的LoRA ID
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras  # 确保不超过最大LoRA数量

        # Next, schedule the WAITING requests.
        # 第二阶段：调度等待中的请求（仅在没有被抢占且未暂停时执行）
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)  # 创建本步跳过的等待队列

            while (self.waiting or self.skipped_waiting) and token_budget > 0:  # 遍历等待队列直到预算耗尽
                if len(self.running) == self.max_num_running_reqs:  # 如果达到最大并发数
                    break  # 停止调度新请求

                request_queue = self._select_waiting_queue_for_scheduling()  # 选择要调度的等待队列
                assert request_queue is not None  # 确保队列存在

                request = request_queue.peek_request()  # 查看队列头部的请求（不弹出）
                request_id = request.request_id  # 获取请求ID

                # try to promote blocked statuses while traversing skipped queue.
                # 尝试提升被阻塞的等待请求状态（如等待FSM、远程KV、流式输入）
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):  # 如果无法提升
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 仍在等待远程KV
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()  # 弹出请求
                    step_skipped_waiting.prepend_request(request)  # 加入本步跳过队列
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                # 检查LoRA约束：是否已达最大LoRA数量
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    # 调度将超过max_loras限制，跳过该请求
                    request_queue.pop_request()  # 弹出请求
                    step_skipped_waiting.prepend_request(request)  # 加入跳过队列
                    continue

                num_external_computed_tokens = 0  # 外部计算的token数（KV传输）
                load_kv_async = False  # 是否异步加载KV
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0  # 连接器前缀缓存查询/命中计数

                # Get already-cached tokens.
                # 获取已缓存的token（前缀缓存命中）
                if request.num_computed_tokens == 0:  # 如果请求尚未计算任何token
                    # Get locally-cached tokens.
                    # 获取本地缓存命中的token
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)  # 查找本地前缀缓存
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    # 如果使用KV连接器，获取外部缓存的token
                    if self.connector is not None:  # 如果有KV传输连接器
                        ext_tokens, load_kv_async = (  # 获取外部匹配的token数和是否异步加载
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:  # 如果连接器无法确定匹配token数
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()  # 弹出请求
                            step_skipped_waiting.prepend_request(request)  # 加入跳过队列
                            continue

                        request.num_external_computed_tokens = ext_tokens  # 记录外部计算的token数
                        num_external_computed_tokens = ext_tokens  # 设置外部计算token数

                        connector_prefix_cache_queries = (  # 连接器前缀缓存查询数
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens  # 连接器前缀缓存命中数

                    # Total computed tokens (local + external).
                    # 计算总的已计算token数（本地 + 外部）
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens  # 确保不超过请求总token数
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    # KV传输：异步接收完成后，等待中的请求已有部分计算token
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks  # 空KV缓存块
                    num_new_local_computed_tokens = 0  # 没有新的本地计算token
                    num_computed_tokens = request.num_computed_tokens  # 使用已有的计算token数

                encoder_inputs_to_schedule = None  # 待调度的编码器输入
                external_load_encoder_input = []  # 外部加载的编码器输入
                new_encoder_compute_budget = encoder_compute_budget  # 新的编码器计算预算

                if load_kv_async:  # 如果需要异步加载KV
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    # KV传输：正在加载远程KV，不分配新工作
                    assert num_external_computed_tokens > 0  # 确保有外部计算token
                    num_new_tokens = 0  # 不调度新token
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    # 计算需要调度的token数（使用num_tokens而非num_prompt_tokens以支持恢复请求）
                    num_new_tokens = request.num_tokens - num_computed_tokens  # 新token数 = 总token数 - 已计算数
                    threshold = self.scheduler_config.long_prefill_token_threshold  # 长预填充阈值
                    if 0 < threshold < num_new_tokens:  # 如果超过阈值则截断
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    # 分块预填充必须显式启用才能对池化请求分块
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        # 如果未启用分块预填充且超出预算，停止调度
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)  # 不超过token预算
                    assert num_new_tokens > 0  # 确保有token要调度

                    # Schedule encoder inputs.
                    # 调度编码器输入（等待请求的多模态处理）
                    if request.has_encoder_inputs:  # 如果请求有编码器输入
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(  # 尝试调度编码器输入
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:  # 编码器预算不足
                            # The request cannot be scheduled.
                            break  # 停止调度

                if self.need_mamba_block_aligned_split:  # 如果需要Mamba块对齐分割
                    num_new_tokens = self._mamba_block_aligned_split(  # 执行块对齐分割
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:  # 对齐后无token可调度
                        break  # 停止调度

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                # 处理P/D分离与推测解码结合时的边界情况
                effective_lookahead_tokens = (  # 有效前瞻token数（新请求不使用前瞻）
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                # 确定是否需要分配交叉注意力块（编码器-解码器模型）
                num_encoder_tokens = 0  # 编码器token数
                if (
                    self.is_encoder_decoder  # 如果是编码器-解码器模型
                    and request.has_encoder_inputs  # 且有编码器输入
                    and encoder_inputs_to_schedule  # 且有待调度的编码器输入
                ):
                    num_encoder_tokens = sum(  # 计算编码器嵌入总数
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                # 为请求分配KV缓存槽位
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,  # 新token数
                    num_new_computed_tokens=num_new_local_computed_tokens,  # 新本地计算token数
                    new_computed_blocks=new_computed_blocks,  # 新计算的块
                    num_lookahead_tokens=effective_lookahead_tokens,  # 前瞻token数
                    num_external_computed_tokens=num_external_computed_tokens,  # 外部计算token数
                    delay_cache_blocks=load_kv_async,  # 是否延迟缓存块
                    num_encoder_tokens=num_encoder_tokens,  # 编码器token数
                )

                if new_blocks is None:  # 分配失败
                    # The request cannot be scheduled.
                    # 请求无法被调度

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    # 需要从编码器缓存管理器中释放请求
                    if request.has_encoder_inputs:  # 如果有编码器输入
                        self.encoder_cache_manager.free(request)  # 释放编码器缓存
                    break  # 停止调度等待请求

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                # KV传输：更新连接器状态，判断是否需要加载远程KV
                if self.connector is not None:  # 如果有KV传输连接器
                    self.connector.update_state_after_alloc(  # 更新分配后的连接器状态
                        request,
                        self.kv_cache_manager.get_blocks(request_id),  # 获取请求的KV缓存块
                        num_external_computed_tokens,  # 外部计算token数
                    )
                    if (  # 记录连接器前缀缓存统计
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(  # 记录缓存查询/命中
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()  # 从等待队列弹出请求
                if load_kv_async:  # 如果需要异步加载KV
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    # 异步加载：分配内存并将请求设为等待远程KV状态
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS  # 设置状态为等待远程KV
                    step_skipped_waiting.prepend_request(request)  # 加入跳过队列
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens  # 设置已计算token数（虽然KV尚未加载）
                    continue  # 继续调度下一个请求

                # 请求可以正常调度（非异步加载）
                self.running.append(request)  # 加入运行队列
                if self.log_stats:  # 如果启用统计
                    request.record_event(  # 记录调度事件
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:  # 如果是新等待的请求
                    scheduled_new_reqs.append(request)  # 加入新调度列表
                elif request.status == RequestStatus.PREEMPTED:  # 如果是被抢占后恢复的请求
                    scheduled_resumed_reqs.append(request)  # 加入恢复调度列表
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")  # 无效状态

                if self.lora_config and request.lora_request:  # 如果请求使用LoRA
                    scheduled_loras.add(request.lora_request.lora_int_id)  # 记录LoRA ID
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(  # 记录所有块（含新分配）
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens  # 记录调度token数
                token_budget -= num_new_tokens  # 扣减token预算
                request.status = RequestStatus.RUNNING  # 设置状态为运行中
                request.num_computed_tokens = num_computed_tokens  # 设置已计算token数
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:  # 如果尚未记录缓存token数
                    request.num_cached_tokens = num_computed_tokens  # 记录前缀缓存命中的token数
                # Encoder-related.
                # 编码器相关：分配编码器缓存
                if encoder_inputs_to_schedule:  # 如果有编码器输入需要调度
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule  # 记录编码器输入
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:  # 遍历编码器输入
                        self.encoder_cache_manager.allocate(request, i)  # 分配编码器缓存
                        if self.ec_connector is not None:  # 如果有编码器缓存连接器
                            self.ec_connector.update_state_after_alloc(request, i)  # 更新连接器状态
                    encoder_compute_budget = new_encoder_compute_budget  # 更新编码器预算
                # Allocate for external load encoder cache
                # 为外部加载的编码器缓存分配空间
                if external_load_encoder_input:  # 如果有外部编码器输入
                    for i in external_load_encoder_input:  # 遍历外部编码器输入
                        self.encoder_cache_manager.allocate(request, i)  # 分配缓存
                        if self.ec_connector is not None:  # 如果有连接器
                            self.ec_connector.update_state_after_alloc(request, i)  # 更新状态

            # re-queue requests skipped in this pass ahead of older skipped items.
            # 将本步跳过的请求重新加入跳过队列（排在旧跳过项之前）
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)  # 前置加入跳过队列

        # Check if the scheduling constraints are satisfied.
        # 检查调度约束是否满足
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())  # 计算总调度token数
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens  # 不超过最大调度token数

        assert token_budget >= 0  # token预算不能为负
        assert len(self.running) <= self.max_num_running_reqs  # 运行请求数不超过最大值
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        # 获取运行队列中所有请求的最长公共前缀块数（用于级联注意力）
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)  # 初始化公共前缀块数
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:  # 如果有运行中的请求
                any_request_id = self.running[0].request_id  # 取任意请求ID
                num_common_prefix_blocks = (  # 计算公共前缀块数
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        # 构建调度器输出
        if self.use_v2_model_runner:  # 如果使用V2模型运行器
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs  # 合并新请求和恢复请求
            scheduled_resumed_reqs = []  # 清空恢复请求列表
            new_reqs_data = [  # 构建新请求数据（V2版本包含all_token_ids）
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [  # 构建新请求数据
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(  # 构建已缓存请求数据
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        # 记录本步调度的请求ID（用于下一步判断是否需要发送完整token列表）
        self.prev_step_scheduled_req_ids.clear()  # 清空上一步的记录
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())  # 更新为本步的请求ID

        # 获取需要清零的新块ID（某些模型需要KV缓存清零）
        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        # 构建SchedulerOutput输出对象
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,  # 新请求数据
            scheduled_cached_reqs=cached_reqs_data,  # 已缓存请求数据
            num_scheduled_tokens=num_scheduled_tokens,  # 每个请求的调度token数
            total_num_scheduled_tokens=total_num_scheduled_tokens,  # 总调度token数
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,  # 推测解码token
            scheduled_encoder_inputs=scheduled_encoder_inputs,  # 编码器输入
            num_common_prefix_blocks=num_common_prefix_blocks,  # 公共前缀块数
            preempted_req_ids={req.request_id for req in preempted_reqs},  # 被抢占的请求ID
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,  # 已完成的请求ID（两步之间完成的）
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),  # 已释放的编码器多模态哈希
            new_block_ids_to_zero=new_block_ids_to_zero,  # 需要清零的新块ID
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        # KV连接器元数据构建：规划KV缓存存储，封装KV加载/保存操作
        if self.connector is not None:  # 如果有KV传输连接器
            meta: KVConnectorMetadata = self.connector.build_connector_meta(  # 构建连接器元数据
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta  # 设置到调度输出中

        # Build the connector meta for ECConnector
        # 构建编码器缓存连接器元数据
        if self.ec_connector is not None:  # 如果有编码器缓存连接器
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(  # 构建EC连接器元数据
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta  # 设置到调度输出中

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)  # 调度后更新内部状态
        return scheduler_output  # 返回调度输出

    # [中文注释] _preempt_request() — 抢占请求并放回等待队列。
    #   释放KV缓存和编码器缓存，重置计算状态，增加抢占计数。
    #   注意：调用者需要先从running队列中移除请求。
    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (  # 只有运行中的请求可以被抢占
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)  # 释放KV缓存
        self.encoder_cache_manager.free(request)  # 释放编码器缓存
        request.status = RequestStatus.PREEMPTED  # 设置状态为已抢占
        request.num_computed_tokens = 0  # 重置已计算token数
        if request.spec_token_ids:  # 如果有推测token
            request.spec_token_ids = []  # 清空推测token
        request.num_preemptions += 1  # 增加抢占计数
        if self.log_stats:  # 如果启用统计
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)  # 记录抢占事件

        # Put the request back to the waiting queue.
        # 将请求放回等待队列头部
        self.waiting.prepend_request(request)

    # [中文注释] _update_after_schedule() — 调度完成后更新请求的已计算token数。
    #   在调度输出构建完成后调用，将num_computed_tokens推进，
    #   以便下一步可以立即重新调度预填充请求。
    #   如果推测token后来被拒绝，num_computed_tokens会在update_from_output中调整。
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 调度后推进请求的已计算token数
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 获取调度token数映射
        for req_id, num_scheduled_token in num_scheduled_tokens.items():  # 遍历所有调度的请求
            request = self.requests[req_id]  # 获取请求对象
            request.num_computed_tokens += num_scheduled_token  # 推进已计算token数
            request.is_prefill_chunk = request.num_computed_tokens < (  # 判断是否仍在预填充阶段
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (  # 标记是否有结构化输出请求
                request.use_structured_output and not request.is_prefill_chunk
            )

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            # 释放不再需要的编码器输入（编码器输入是prompt的一部分，不受推测解码影响）
            if request.has_encoder_inputs:  # 如果请求有编码器输入
                self._free_encoder_inputs(request)  # 释放已处理完的编码器输入

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        # 注意：这里不能用clear()，因为会影响scheduler_output中的引用
        self.finished_req_ids = set()  # 创建新的空集合替代

    # [中文注释] _update_request_as_session() — 更新流式输入会话。
    #   用新的StreamingUpdate更新等待中的会话请求：
    #     丢弃上一输入块的最后采样输出token，
    #     将已计算的输出token合并到prompt中，
    #     追加新的prompt token和多模态特征。
    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        # 当前流式输入行为：保留已计算的输出token（丢弃最后采样的输出token）
        num_computed_tokens = session.num_computed_tokens  # 获取已计算token数
        kept_output_tokens = session._all_token_ids[  # 提取保留的输出token
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]  # 删除未计算的token
        session._output_token_ids.clear()  # 清空输出token列表
        assert session.prompt_token_ids is not None  # 确保prompt token存在
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)  # 将保留的输出token合并到prompt中

        if update.mm_features:  # 如果有新的多模态特征
            base = session.num_tokens  # 当前token数作为偏移基础
            for mm_feature in update.mm_features:  # 调整多模态特征的位置偏移
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)  # 追加多模态特征

        session._all_token_ids.extend(update.prompt_token_ids or ())  # 追加新的prompt token到all_token_ids
        session.prompt_token_ids.extend(update.prompt_token_ids or ())  # 追加到prompt_token_ids
        # Update block hashes for the new tokens.
        session.update_block_hashes()  # 更新新token的块哈希（用于前缀缓存）
        session.num_prompt_tokens = len(session.prompt_token_ids)  # 更新prompt token数
        session.arrival_time = update.arrival_time  # 更新到达时间
        session.sampling_params = update.sampling_params  # 更新采样参数
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:  # 如果之前在等待流式输入
            self.num_waiting_for_streaming_input -= 1  # 递减等待流式输入计数
        session.status = RequestStatus.WAITING  # 设置状态为等待调度

        if self.log_stats:  # 如果启用统计
            session.record_event(EngineCoreEventType.QUEUED)  # 记录入队事件

    # [中文注释] _make_cached_request_data() — 构建已缓存请求的数据。
    #   将running和resumed请求的信息打包为CachedRequestData，
    #   供model runner使用。包含token ID、新块ID、计算状态等。
    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []  # 请求ID列表
        new_token_ids: list[list[int]] = []  # 新token ID列表（PP场景使用）
        new_block_ids: list[tuple[list[int], ...] | None] = []  # 新分配的块ID列表
        all_token_ids: dict[str, list[int]] = {}  # 完整token ID映射（首次调度时发送）
        num_computed_tokens: list[int] = []  # 已计算token数列表
        num_output_tokens: list[int] = []  # 输出token数列表
        resumed_req_ids = set()  # 恢复请求ID集合

        num_running_reqs = len(running_reqs)  # 运行请求数量
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):  # 遍历running和resumed请求
            req_id = req.request_id  # 获取请求ID
            req_ids.append(req_id)  # 添加到请求ID列表
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            # PP场景下（非异步调度）需要发送采样token回调度器
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                num_tokens = num_scheduled_tokens[req_id] - len(  # 计算非推测token数
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[  # 提取新token ID
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)  # 添加到新token列表
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids  # 是否在上一步调度过
            if idx >= num_running_reqs:  # 如果是恢复请求
                assert not scheduled_in_prev_step  # 恢复请求不应在上一步调度过
                resumed_req_ids.add(req_id)  # 添加到恢复请求ID集合
            if not scheduled_in_prev_step:  # 如果不是上一步调度的（首次出现）
                all_token_ids[req_id] = req.all_token_ids.copy()  # 发送完整token列表
            new_block_ids.append(  # 记录新分配的块ID
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)  # 记录已计算token数
            num_output_tokens.append(  # 记录输出token数（含占位符）
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(  # 返回构建的缓存请求数据
            req_ids=req_ids,  # 请求ID列表
            resumed_req_ids=resumed_req_ids,  # 恢复请求ID集合
            new_token_ids=new_token_ids,  # 新token ID列表
            all_token_ids=all_token_ids,  # 完整token ID映射
            new_block_ids=new_block_ids,  # 新块ID列表
            num_computed_tokens=num_computed_tokens,  # 已计算token数列表
            num_output_tokens=num_output_tokens,  # 输出token数列表
        )

    # [中文注释] _try_schedule_encoder_inputs() — 尝试调度编码器输入。
    #   确定当前步骤需要调度哪些编码器输入（多模态场景），
    #   并相应更新num_new_tokens和编码器token预算。
    #   返回：(待调度编码器输入列表, 调整后的新token数, 新编码器预算, 外部加载编码器输入列表)
    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:  # 无新token或无编码器输入时直接返回
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []  # 待调度的编码器输入索引列表
        mm_features = request.mm_features  # 获取多模态特征列表
        assert mm_features is not None  # 确保存在多模态特征
        assert len(mm_features) > 0  # 确保至少有一个多模态特征
        external_load_encoder_input = []  # 外部加载的编码器输入列表

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        # 调度器在请求级别操作，需要在编码器输入级别进行临时跟踪
        mm_hashes_to_schedule = set()  # 待调度的多模态哈希集合（避免重复调度同一编码器输入）
        num_embeds_to_schedule = 0  # 待调度的嵌入数量
        for i, mm_feature in enumerate(mm_features):  # 遍历所有多模态特征
            start_pos = mm_feature.mm_position.offset  # 编码器输入在序列中的起始位置
            num_encoder_tokens = mm_feature.mm_position.length  # 编码器token数量
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()  # 编码器嵌入数量
            item_identifier = mm_feature.identifier  # 多模态项标识符

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            # 判断编码器输出是否在当前调度范围内（两个区间是否重叠）
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break  # 编码器输入不在本步范围内，停止遍历

            if self.is_encoder_decoder and num_computed_tokens > 0:  # 编码器-解码器模型且已有计算
                assert start_pos == 0, (  # 编码器输入应在序列开头处理
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # 编码器输入已经计算过（编码器-解码器模型中，一旦有decoder token计算，
                # 说明编码器输入已处理完毕）
                continue  # 跳过已计算的编码器输入
            elif start_pos + num_encoder_tokens <= num_computed_tokens:  # 编码器输入已在解码器KV缓存中
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue  # 跳过已存储的编码器输入

            if not self.is_encoder_decoder:  # 非编码器-解码器模型
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:  # 同一编码器输入已在本步调度
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue  # 跳过重复调度

                if self.encoder_cache_manager.check_and_update_cache(request, i):  # 编码器输入已在缓存中
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue  # 跳过已缓存的编码器输入

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            # 如果不允许编码器输入分块，且调度范围只覆盖部分多模态输入，回退到多模态项之前
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(  # 回退新token数到多模态项之前
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break  # 停止遍历
            if not self.encoder_cache_manager.can_allocate(  # 检查编码器缓存是否可分配
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0  # 本步不调度任何token
                break  # 停止遍历

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            # 计算当前调度范围内的嵌入数量
            start_idx_rel = max(0, num_computed_tokens - start_pos)  # 相对起始索引
            end_idx_rel = min(  # 相对结束索引
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (  # 获取当前范围内的嵌入索引
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:  # 当前范围内没有嵌入
                continue  # 跳过无嵌入的范围

            if self.ec_connector is not None and self.ec_connector.has_cache_item(  # 如果编码器缓存连接器有该项
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)  # 记录多模态哈希
                external_load_encoder_input.append(i)  # 加入外部加载列表
                num_embeds_to_schedule += num_encoder_embeds  # 累加嵌入数
                continue  # 跳过本地计算

            num_embeds_to_schedule += num_encoder_embeds  # 累加待调度的嵌入数
            encoder_compute_budget -= num_encoder_embeds  # 扣减编码器计算预算
            mm_hashes_to_schedule.add(item_identifier)  # 记录多模态哈希
            encoder_inputs_to_schedule.append(i)  # 加入待调度列表

        return (  # 返回调度结果
            encoder_inputs_to_schedule,  # 待调度的编码器输入索引
            num_new_tokens,  # 调整后的新token数
            encoder_compute_budget,  # 剩余编码器计算预算
            external_load_encoder_input,  # 外部加载的编码器输入索引
        )

    # [中文注释] get_grammar_bitmask() — 获取结构化输出的语法位掩码。
    #   收集使用结构化输出的已调度请求，生成对应的语法位掩码，
    #   用于在采样时约束token生成。
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # 收集使用结构化输出的已调度请求ID列表
        if not scheduler_output.has_structured_output_requests:  # 如果没有结构化输出请求
            return None  # 直接返回

        structured_output_request_ids = [  # 筛选使用结构化输出且非预填充阶段的请求
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:  # 如果没有符合条件的请求
            return None  # 直接返回

        bitmask = self.structured_output_manager.grammar_bitmask(  # 生成语法位掩码
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)  # 返回语法输出

    # [中文注释] update_from_output() — 模型推理完成后更新调度器状态。
    #   处理每个请求的生成结果：
    #     - 追加生成的 token 到请求
    #     - 处理 speculative decoding 的 token 接受/拒绝
    #     - 检查停止条件（EOS、stop_token、max_tokens、重复检测）
    #     - 更新结构化输出的语法状态
    #     - 释放已完成请求的资源
    #   返回 dict[client_index → EngineCoreOutputs]，供前端按客户端分发。
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids  # 采样得到的token ID
        logprobs = model_runner_output.logprobs  # 对数概率
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict  # prompt对数概率字典
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 调度的token数映射
        pooler_outputs = model_runner_output.pooler_output  # 池化输出
        num_nans_in_logits = model_runner_output.num_nans_in_logits  # logits中的NaN数量
        kv_connector_output = model_runner_output.kv_connector_output  # KV连接器输出
        cudagraph_stats = model_runner_output.cudagraph_stats  # CUDA图统计

        perf_stats: PerfStats | None = None  # 性能统计
        if self.perf_metrics and self.perf_metrics.is_enabled():  # 如果启用性能指标
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)  # 获取每GPU性能统计

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)  # 按客户端索引分组的输出
        spec_decoding_stats: SpecDecodingStats | None = None  # 推测解码统计
        kv_connector_stats: KVConnectorStats | None = (  # KV连接器统计
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:  # 如果有连接器统计
            kv_stats = self.connector.get_kv_connector_stats()  # 获取连接器统计
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)  # 聚合统计

        failed_kv_load_req_ids = None  # KV加载失败的请求ID
        if kv_connector_output and kv_connector_output.invalid_block_ids:  # 如果有无效块
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(  # 处理无效块，返回受影响请求ID
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck.
        # 遍历所有调度的请求处理输出（可能有1K+个请求，是性能瓶颈）
        stopped_running_reqs: set[Request] = set()  # 已停止的运行请求集合
        stopped_preempted_reqs: set[Request] = set()  # 已停止的被抢占请求集合
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():  # 遍历每个调度的请求
            assert num_tokens_scheduled > 0  # 确保调度了至少一个token
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:  # 如果是KV加载失败的请求
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)  # 获取请求对象
            if request is None or request.is_finished():  # 如果请求已完成或不存在
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it.
                # 请求已完成（可能在模型执行期间被中止，如PP或异步调度场景）
                continue  # 跳过

            req_index = model_runner_output.req_id_to_index[req_id]  # 获取请求在输出中的索引
            generated_token_ids = (  # 获取生成的token ID
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (  # 获取本步调度的推测token
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:  # 如果有推测解码
                num_draft_tokens = len(scheduled_spec_token_ids)  # 草稿token数
                num_accepted = len(generated_token_ids) - 1  # 被接受的token数（生成数-1）
                num_rejected = num_draft_tokens - num_accepted  # 被拒绝的token数
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:  # 减去被拒绝的token数
                    request.num_computed_tokens -= num_rejected  # 回退已计算token数
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:  # 异步调度下也调整输出占位符
                    request.num_output_placeholders -= num_rejected  # 回退输出占位符数
                spec_decoding_stats = self.make_spec_decoding_stats(  # 创建推测解码统计
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False  # 是否已停止
            new_logprobs = None  # 新的对数概率
            new_token_ids = generated_token_ids  # 新生成的token ID
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None  # 池化输出
            kv_transfer_params = None  # KV传输参数
            status_before_stop = request.status  # 停止前的状态

            # Check for stop and update request status.
            # 检查停止条件并更新请求状态
            if new_token_ids:  # 如果有新生成的token
                new_token_ids, stopped = self._update_request_with_output(  # 追加token并检查停止条件
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:  # 如果是池化请求且有输出
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED  # 池化请求有输出即完成
                stopped = True

            routed_experts = None  # 路由专家信息
            finish_reason = None  # 完成原因
            if stopped:  # 如果请求已停止
                routed_experts = self._get_routed_experts(request)  # 获取路由专家信息

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                # 在处理停止请求之前捕获完成原因（流式请求可能重置状态为WAITING）
                finish_reason = request.get_finished_reason()  # 获取完成原因
                finished = self._handle_stopped_request(request)  # 处理停止的请求
                if finished:  # 如果请求真正完成（非可恢复的流式请求）
                    kv_transfer_params = self._free_request(request)  # 释放请求资源

                if status_before_stop == RequestStatus.RUNNING:  # 根据停止前状态分类
                    stopped_running_reqs.add(request)  # 加入已停止的运行请求集合
                else:
                    stopped_preempted_reqs.add(request)  # 加入已停止的被抢占请求集合

            # Extract sample logprobs if needed.
            # 如果需要，提取采样对数概率
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None  # 请求了logprobs
                and logprobs  # 模型输出了logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))  # 切片提取该请求的logprobs

            # 更新结构化输出的语法状态
            if new_token_ids and self.structured_output_manager.should_advance(request):  # 如果需要推进语法
                struct_output_request = request.structured_output_request  # 获取结构化输出请求
                assert struct_output_request is not None  # 确保存在
                assert struct_output_request.grammar is not None  # 确保有语法
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)  # 让语法接受新token
                if not ok:  # 如果语法拒绝了token
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:  # 如果有NaN记录
                request.num_nans_in_logits = num_nans_in_logits[req_id]  # 更新NaN计数

            # Get prompt logprobs for this request.
            # 获取该请求的prompt对数概率
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)  # 获取prompt logprobs
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                # 为该请求构建引擎核心输出
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,  # 请求ID
                        new_token_ids=new_token_ids,  # 新生成的token ID
                        finish_reason=finish_reason,  # 完成原因
                        new_logprobs=new_logprobs,  # 新的对数概率
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,  # prompt对数概率张量
                        pooling_output=pooler_output,  # 池化输出
                        stop_reason=request.stop_reason,  # 停止原因
                        events=request.take_events(),  # 事件列表
                        kv_transfer_params=kv_transfer_params,  # KV传输参数
                        trace_headers=request.trace_headers,  # 追踪头
                        num_cached_tokens=request.num_cached_tokens,  # 缓存token数
                        num_external_computed_tokens=request.num_external_computed_tokens,  # 外部计算token数
                        routed_experts=routed_experts,  # 路由专家信息
                        num_nans_in_logits=request.num_nans_in_logits,  # logits中NaN数
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors  # 不变量：引擎核心不返回部分预填充输出

        # Remove the stopped requests from the running and waiting queues.
        # 从运行和等待队列中移除已停止的请求
        if stopped_running_reqs:  # 如果有停止的运行请求
            self.running = remove_all(self.running, stopped_running_reqs)  # 批量移除
        if stopped_preempted_reqs:  # 如果有停止的被抢占请求（罕见情况）
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)  # 从等待队列移除

        # 处理KV加载失败的请求（非重计算策略时直接报错）
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        # KV连接器：更新已完成KV传输的状态
        if kv_connector_output:  # 如果有KV连接器输出
            self._update_from_kv_xfer_finished(kv_connector_output)  # 更新KV传输完成状态

        # collect KV cache events from KV cache manager
        # 收集KV缓存事件
        events = self.kv_cache_manager.take_events()  # 从KV缓存管理器获取事件

        # collect KV cache events from connector
        # 从连接器收集KV缓存事件
        if self.connector is not None:  # 如果有KV传输连接器
            connector_events = self.connector.take_events()  # 获取连接器事件
            if connector_events:  # 如果有事件
                if events is None:
                    events = list(connector_events)  # 初始化事件列表
                else:
                    events.extend(connector_events)  # 追加事件

        # publish collected KV cache events
        # 发布收集的KV缓存事件
        if events:  # 如果有事件
            batch = KVEventBatch(ts=time.time(), events=events)  # 创建事件批次
            self.kv_event_publisher.publish(batch)  # 发布事件

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        # 为本步有输出的每个客户端创建EngineCoreOutputs
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        # 包含自上次输出以来完成的请求ID（多引擎场景）
        finished_req_ids = self.finished_req_ids_dict  # 获取已完成请求字典
        if finished_req_ids:  # 如果有已完成的请求
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():  # 遍历每个客户端
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:  # 如果已有输出
                    eco.finished_requests = finished_set  # 设置已完成请求集合
                else:  # 否则创建新的输出对象
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()  # 清空已完成请求字典

        # 生成统计信息并返回给前端
        if (
            stats := self.make_stats(  # 生成调度器统计信息
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            # 只返回统计信息给其中一个前端
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                # 即使本步没有请求输出也要返回统计信息
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats  # 设置调度器统计

        return engine_core_outputs  # 返回引擎核心输出

    # [中文注释] _is_blocked_waiting_status() — 判断请求状态是否为阻塞等待状态。
    #   阻塞等待状态包括：等待FSM（有限状态机）、等待远程KV、等待流式输入。
    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (  # 判断是否为阻塞等待状态
            RequestStatus.WAITING_FOR_FSM,  # 等待有限状态机（结构化输出语法初始化）
            RequestStatus.WAITING_FOR_REMOTE_KVS,  # 等待远程KV传输
            RequestStatus.WAITING_FOR_STREAMING_REQ,  # 等待流式输入
        )

    # [中文注释] _enqueue_waiting_request() — 将请求加入等待队列。
    #   阻塞状态的请求加入skipped_waiting队列，正常状态加入waiting队列。
    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):  # 如果是阻塞状态
            self.skipped_waiting.add_request(request)  # 加入跳过的等待队列
        else:
            self.waiting.add_request(request)  # 加入正常等待队列

    # [中文注释] _select_waiting_queue_for_scheduling() — 选择用于调度的等待队列。
    #   FCFS策略：优先从skipped_waiting取，然后从waiting取。
    #   PRIORITY策略：比较两个队列头部请求的优先级，选择更高优先级的队列。
    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:  # 先来先服务策略
            return self.skipped_waiting or self.waiting or None  # 优先skipped队列

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        # 优先级模式：两个队列都非空时比较头部请求
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()  # 查看waiting队列头部
            skipped_req = self.skipped_waiting.peek_request()  # 查看skipped队列头部
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting  # 选择优先级更高的

        return self.waiting or self.skipped_waiting or None  # 返回非空的队列

    # [中文注释] _handle_stopped_request() — 处理已停止的请求。
    #   如果请求不可恢复，返回True表示完成。
    #   如果是可恢复的流式请求，检查streaming_queue中是否有下一个输入块。
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:  # 如果请求不可恢复
            return True  # 直接完成

        if request.streaming_queue:  # 如果流式队列有数据
            update = request.streaming_queue.popleft()  # 取出下一个流式更新
            if update is None:  # 如果是完成标记
                # Streaming request finished.
                return True  # 流式请求完成
            self._update_request_as_session(request, update)  # 用新数据更新请求会话
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ  # 等待流式输入
            self.num_waiting_for_streaming_input += 1  # 递增等待计数

        self._enqueue_waiting_request(request)  # 将请求重新加入等待队列
        return False  # 请求未完成（可恢复）

    # [中文注释] _get_routed_experts() — 获取请求的路由专家信息。
    #   对于MoE模型，返回每个token被路由到的专家信息。
    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:  # 未启用则返回None
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)  # 获取KV缓存块
        block_ids = kv_blocks.get_block_ids()[self.routed_experts_attn_gid]  # 获取注意力组的块ID
        num_tokens = request.num_tokens - 1  # token数量（减1是因为最后一个token无路由信息）

        # compute slot mapping using attention group's block_size
        # 使用注意力组的block_size计算槽位映射
        block_ids_array = np.array(block_ids, dtype=np.int32)  # 块ID数组
        num_blocks = len(block_ids)  # 块数量
        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]  # 注意力组
        block_size = attn_group.kv_cache_spec.block_size  # 块大小

        # generate block offsets
        block_offsets = np.arange(0, block_size)  # 生成块内偏移量

        # compute slot mapping: slot = block_id * block_size + offset
        # 计算槽位映射：slot = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]  # 展平并截断到token数量

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)  # 读取路由专家信息

    # [中文注释] _update_request_with_output() — 用生成结果更新请求。
    #   逐个追加生成的token并检查停止条件。
    #   返回：(实际使用的token列表, 是否停止)
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop.
        # 追加生成的token并检查停止条件
        stopped = False  # 是否停止标志
        for num_new, output_token_id in enumerate(new_token_ids, 1):  # 遍历新token
            request.append_output_token_ids(output_token_id)  # 追加到请求的输出token列表

            # Check for stop and update request state.
            # 检查停止条件（EOS、stop_token、max_tokens等）
            stopped = check_stop(request, self.max_model_len)  # 检查是否触发停止
            if stopped:  # 如果停止
                del new_token_ids[num_new:]  # 截断后续token
                break
        return new_token_ids, stopped  # 返回实际token和停止标志

    # [中文注释] _free_encoder_inputs() — 释放不再需要的编码器输入缓存。
    #   当编码器输出已被处理并存入解码器KV缓存后，释放编码器缓存。
    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(  # 获取已缓存的编码器输入ID
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:  # 如果没有缓存的编码器输入，直接返回
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        # 使用list(set)避免迭代时修改集合
        for input_id in list(cached_encoder_input_ids):  # 遍历已缓存的编码器输入
            mm_feature = request.mm_features[input_id]  # 获取多模态特征
            start_pos = mm_feature.mm_position.offset  # 编码器输入起始位置
            num_tokens = mm_feature.mm_position.length  # 编码器token数量
            if self.is_encoder_decoder and request.num_computed_tokens > 0:  # 编码器-解码器模型
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input.
                # Whisper等模型：生成一个token后即可释放编码器输入
                self.encoder_cache_manager.free_encoder_input(request, input_id)  # 释放编码器输入缓存
            elif start_pos + num_tokens <= request.num_computed_tokens:  # 编码器输出已处理完毕
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)  # 释放编码器输入缓存

    # [中文注释] update_draft_token_ids() — 更新请求的草稿token ID（推测解码）。
    #   将草稿模型生成的推测token设置到对应请求中。
    #   预填充阶段的请求忽略草稿token。
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(  # 遍历草稿token
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)  # 获取请求
            if request is None or request.is_finished():  # 请求已完成则跳过
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:  # 预填充阶段忽略草稿token
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []  # 清空推测token
                continue

            # Add newly generated spec token ids to the request.
            # 将新生成的推测token添加到请求
            if self.structured_output_manager.should_advance(request):  # 如果有结构化输出约束
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]  # 用语法验证推测token
            request.spec_token_ids = spec_token_ids  # 设置推测token

    # [中文注释] update_draft_token_ids_in_output() — 在调度输出中更新草稿token（异步推测解码）。
    #   将草稿模型的token替换到调度输出的占位符中，验证语法并填充无效token。
    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}  # 无效推测token数映射

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens  # 调度输出中的推测token
        for req_id, spec_token_ids in zip(  # 遍历草稿token
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)  # 获取请求
            if request is None or request.is_finished():  # 请求已完成则跳过
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)  # 获取占位符推测token
            if not placeholder_spec_tokens:  # 如果没有占位符
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)  # 原始推测token数量
            # Trim drafts to scheduled number of spec tokens
            # 截断草稿到调度的推测token数量（分块预填充场景需要）
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            # 过滤不符合语法的推测token
            if self.structured_output_manager.should_advance(request):  # 如果有结构化输出约束
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # 语法验证
            # Pad to original number of spec tokens.
            # 填充到原始推测token数量（无效token用-1标记）
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)  # 计算无效token数
            if num_invalid_tokens:  # 如果有无效token
                spec_token_ids.extend([-1] * num_invalid_tokens)  # 用-1填充
                num_invalid_spec_tokens[req_id] = num_invalid_tokens  # 记录无效数量

            sched_spec_tokens[req_id] = spec_token_ids  # 更新调度输出中的推测token

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens  # 设置无效推测token统计

    # [中文注释] get_request_counts() — 获取运行和等待请求数量。
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)  # 返回(运行数, 等待数)

    # [中文注释] add_request() — 添加新请求到调度器。
    #   支持流式输入：同一 request_id 的后续调用会追加到 streaming_queue。
    #   新请求根据状态加入 waiting 或 skipped_waiting 队列。
    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)  # 检查是否已存在同ID请求
        if existing is not None:  # 如果已存在（流式输入场景）
            update = StreamingUpdate.from_request(request)  # 从请求创建流式更新
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:  # 如果请求不在等待流式输入
                assert existing.streaming_queue is not None, "duplicate request id"  # 确保有流式队列
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)  # 将更新加入流式队列
            elif update is not None:  # 如果有新的输入块
                # Commence next input chunk.
                self._update_request_as_session(existing, update)  # 开始处理新输入块
            else:  # 流式输入结束
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)  # 中止请求
        else:  # 新请求
            if request.resumable:  # 如果请求可恢复（流式输入）
                request.streaming_queue = deque()  # 初始化流式队列
            self._enqueue_waiting_request(request)  # 将请求加入等待队列
            self.requests[request.request_id] = request  # 注册请求
            if self.log_stats:  # 如果启用统计
                request.record_event(EngineCoreEventType.QUEUED)  # 记录入队事件

    # [中文注释] finish_requests() — 外部终止请求（如客户端断开、前端检测到 stop string）。
    #   从 running/waiting 队列移除请求，释放 KV cache 和编码器缓存。
    #   返回被终止的 (req_id, client_index) 列表。
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)  # 确保是已完成状态
        if isinstance(request_ids, str):  # 如果传入单个ID字符串
            request_ids = (request_ids,)  # 转为元组
        elif request_ids is not None:  # 如果传入多个ID
            request_ids = set(request_ids)  # 转为集合
        else:  # 如果为None，终止所有请求
            request_ids = self.requests.keys()

        running_requests_to_remove = set()  # 待从running队列移除的请求
        waiting_requests_to_remove = []  # 待从waiting队列移除的请求
        valid_requests = []  # 有效请求列表

        # First pass: collect requests to remove from queues
        # 第一遍遍历：收集需要从队列移除的请求
        for req_id in request_ids:  # 遍历请求ID
            request = self.requests.get(req_id)  # 获取请求
            if request is None or request.is_finished():  # 无效请求ID或已完成
                # Invalid request ID.
                continue

            valid_requests.append(request)  # 加入有效请求列表
            if request.status == RequestStatus.RUNNING:  # 如果在运行中
                running_requests_to_remove.add(request)  # 加入待移除集合
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:  # 如果在等待流式输入
                    self.num_waiting_for_streaming_input -= 1  # 递减等待计数
                waiting_requests_to_remove.append(request)  # 加入待移除列表

        # Remove all requests from queues at once for better efficiency
        # 批量从队列移除请求以提高效率
        if running_requests_to_remove:  # 如果有需要从running移除的请求
            self.running = remove_all(self.running, running_requests_to_remove)  # 批量移除
        if waiting_requests_to_remove:  # 如果有需要从waiting移除的请求
            self.waiting.remove_requests(waiting_requests_to_remove)  # 从waiting移除
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)  # 从skipped_waiting移除

        # Second pass: set status and free requests
        # 第二遍遍历：设置状态并释放请求资源
        for request in valid_requests:  # 遍历有效请求
            delay_free_blocks = False  # 是否延迟释放块
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 如果在等待远程KV
                delay_free_blocks = (  # 如果尚未完成接收则延迟释放
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)  # 从接收完成集合移除
                self.failed_recving_kv_req_ids.discard(request.request_id)  # 从接收失败集合移除

            request.status = finished_status  # 设置完成状态
            self._free_request(request, delay_free_blocks=delay_free_blocks)  # 释放请求资源

        return [(r.request_id, r.client_index) for r in valid_requests]  # 返回(请求ID, 客户端索引)列表

    # [中文注释] _free_request() — 释放已完成请求的资源。
    #   通知KV连接器、释放编码器缓存、记录完成ID、释放KV块。
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()  # 确保请求已完成

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)  # 通知连接器请求完成
        self.encoder_cache_manager.free(request)  # 释放编码器缓存
        request_id = request.request_id  # 获取请求ID
        self.finished_req_ids.add(request_id)  # 添加到已完成请求ID集合
        if self.finished_req_ids_dict is not None:  # 多引擎场景
            self.finished_req_ids_dict[request.client_index].add(request_id)  # 按客户端记录

        delay_free_blocks |= connector_delay_free_blocks  # 合并延迟释放标志
        if not delay_free_blocks:  # 如果不需要延迟释放
            self._free_blocks(request)  # 立即释放KV缓存块

        return kv_xfer_params  # 返回KV传输参数

    # [中文注释] _free_blocks() — 释放请求的KV缓存块并删除请求记录。
    def _free_blocks(self, request: Request):
        assert request.is_finished()  # 确保请求已完成
        self.kv_cache_manager.free(request)  # 释放KV缓存
        del self.requests[request.request_id]  # 从请求字典中删除

    # [中文注释] pause_state — 获取调度器暂停状态。
    @property
    def pause_state(self) -> PauseState:
        return self._pause_state  # 返回暂停状态

    # [中文注释] set_pause_state() — 设置调度器暂停状态。
    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state  # 设置暂停状态

    # [中文注释] get_num_unfinished_requests() — 获取未完成请求数量。
    #   暂停所有时返回0，暂停新请求时只计运行中的。
    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:  # 如果暂停所有
            return 0  # 返回0
        if self._pause_state == PauseState.PAUSED_NEW:  # 如果只暂停新请求
            return len(self.running)  # 只返回运行中的数量
        num_waiting = (  # 计算等待中的请求数（排除等待流式输入的）
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)  # 返回等待 + 运行

    # [中文注释] has_finished_requests() — 是否有已完成的请求。
    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0  # 判断是否有完成的请求

    # [中文注释] reset_prefix_cache() — 重置KV前缀缓存。
    #   如果reset_running_requests为True，先抢占所有运行中的请求以确保重置成功。
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:  # 如果需要重置运行中的请求
            # For logging.
            timestamp = time.monotonic()  # 记录时间戳
            # 抢占所有运行中的请求以将KV块引用计数降为0
            while self.running:  # 逆序抢占以保持FIFO顺序
                request = self.running.pop()  # 从运行队列弹出
                self._preempt_request(request, timestamp)  # 执行抢占
                request.num_output_placeholders = 0  # 重置输出占位符（异步调度需要）
                request.discard_latest_async_tokens = True  # 标记丢弃最新异步token

            # 清空上一步调度的请求ID缓存（强制抢占+恢复同步进行）
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()  # 重置前缀缓存
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:  # 如果需要重置连接器缓存
            reset_successful = self.reset_connector_cache() and reset_successful  # 重置连接器缓存

        return reset_successful  # 返回重置是否成功

    # [中文注释] reset_connector_cache() — 重置KV连接器缓存。
    def reset_connector_cache(self) -> bool:
        if self.connector is None:  # 如果没有配置KV连接器
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:  # 尝试重置连接器缓存
            return False  # 重置失败

        if self.log_stats:  # 如果启用统计
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True  # 标记统计已重置

        return True  # 重置成功

    # [中文注释] reset_encoder_cache() — 重置编码器缓存。
    #   当模型权重更新时调用，确保不复用过期的视觉嵌入。
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()  # 重置编码器缓存管理器

    # [中文注释] make_stats() — 创建调度器统计信息。
    #   收集KV缓存使用率、前缀缓存命中率、推测解码统计等。
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:  # 如果未启用统计
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()  # 生成前缀缓存统计
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None  # 连接器前缀缓存统计
        if self.connector_prefix_cache_stats is not None:  # 如果有连接器统计
            connector_prefix_cache_stats = self.connector_prefix_cache_stats  # 获取当前统计
            self.connector_prefix_cache_stats = PrefixCacheStats()  # 重置统计
        eviction_events = (  # 获取KV缓存驱逐事件
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats  # 推测解码统计
        connector_stats_payload = (  # 连接器统计数据
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(  # 返回调度器统计
            num_running_reqs=len(self.running),  # 运行中请求数
            num_waiting_reqs=len(self.waiting) + len(self.skipped_waiting),  # 等待中请求数
            kv_cache_usage=self.kv_cache_manager.usage,  # KV缓存使用率
            encoder_cache_usage=self._get_encoder_cache_usage(),  # 编码器缓存使用率
            prefix_cache_stats=prefix_cache_stats,  # 前缀缓存统计
            connector_prefix_cache_stats=connector_prefix_cache_stats,  # 连接器前缀缓存统计
            kv_cache_eviction_events=eviction_events,  # 缓存驱逐事件
            spec_decoding_stats=spec_stats,  # 推测解码统计
            kv_connector_stats=connector_stats_payload,  # KV连接器统计
            cudagraph_stats=cudagraph_stats,  # CUDA图统计
            perf_stats=perf_stats,  # 性能统计
        )

    # [中文注释] _get_encoder_cache_usage() — 获取编码器缓存使用率（0.0 到 1.0）。
    def _get_encoder_cache_usage(self) -> float:
        """Get encoder cache usage as a fraction (0.0 to 1.0)."""
        ecm = self.encoder_cache_manager  # 获取编码器缓存管理器
        if ecm.cache_size == 0:  # 如果缓存大小为0
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots  # 计算已使用槽位
        return used_slots / ecm.cache_size  # 返回使用率

    # [中文注释] make_spec_decoding_stats() — 创建推测解码统计信息。
    #   记录草稿token数和被接受的token数。
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:  # 如果未启用统计或无草稿token
            return None
        if spec_decoding_stats is None:  # 如果统计对象不存在则创建
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:  # 如果有无效推测token
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)  # 减去无效token数
        spec_decoding_stats.observe_draft(  # 记录草稿观测
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats  # 返回统计

    # [中文注释] shutdown() — 关闭调度器，释放资源。
    def shutdown(self) -> None:
        if self.kv_event_publisher:  # 如果有KV事件发布器
            self.kv_event_publisher.shutdown()  # 关闭事件发布器
        if self.connector is not None:  # 如果有KV连接器
            self.connector.shutdown()  # 关闭连接器

    ########################################################################
    # KV Connector Related Methods
    # KV连接器相关方法
    ########################################################################

    # [中文注释] get_kv_connector() — 获取KV连接器实例。
    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector  # 返回KV连接器

    # [中文注释] _connector_finished() — 通知KV连接器请求已完成。
    #   返回：(是否延迟释放块, KV传输参数)
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:  # 如果没有KV连接器
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        # 在将块表交给连接器之前，释放超出窗口的前缀块
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)  # 获取请求的块ID

        if not isinstance(self.connector, SupportsHMA):  # 如果连接器不支持混合内存分配器
            assert len(self.kv_cache_config.kv_cache_groups) == 1  # 确保只有一个缓存组
            return self.connector.request_finished(request, block_ids[0])  # 通知连接器请求完成

        return self.connector.request_finished_all_groups(request, block_ids)  # 通知所有组请求完成

    # [中文注释] _update_waiting_for_remote_kv() — 异步KV接收完成后更新请求状态。
    #   将成功接收的KV块缓存，处理加载失败的情况。
    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None  # 确保有KV连接器

        if request.request_id in self.failed_recving_kv_req_ids:  # 如果请求有KV加载失败
            # Request had KV load failures
            # 请求有KV加载失败，num_computed_tokens已在_update_requests_with_invalid_blocks中更新
            if request.num_computed_tokens:  # 如果有有效的已计算token
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)  # 缓存有效token
            else:  # 没有有效token
                # No valid computed tokens, release allocated blocks.
                self.kv_cache_manager.free(request)  # 释放已分配的块

            self.failed_recving_kv_req_ids.remove(request.request_id)  # 从失败集合移除
        else:  # 接收成功
            # Now that the blocks are ready, actually cache them.
            # 块已准备好，执行缓存
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)  # 缓存块

            # on a full prompt hit, we need to re-compute the last token
            # 完全命中时需要重新计算最后一个token以采样下一个token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1  # 回退一个token

            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:  # 如果尚未记录缓存token数
                request.num_cached_tokens = request.num_computed_tokens  # 记录缓存命中数

        self.finished_recving_kv_req_ids.remove(request.request_id)  # 从接收完成集合移除

    # [中文注释] _try_promote_blocked_waiting_request() — 尝试提升阻塞等待请求的状态。
    #   根据不同的阻塞原因（等待远程KV、等待FSM、等待流式输入）尝试恢复请求。
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 等待远程KV
            # finished_recving_kv_req_ids在update_from_output()中根据worker连接器信号填充
            if request.request_id not in self.finished_recving_kv_req_ids:  # 如果尚未完成接收
                return False  # 无法提升
            self._update_waiting_for_remote_kv(request)  # 更新KV接收状态
            if request.num_preemptions:  # 如果之前被抢占过
                request.status = RequestStatus.PREEMPTED  # 设为已抢占状态
            else:
                request.status = RequestStatus.WAITING  # 设为等待状态
            return True  # 提升成功

        if request.status == RequestStatus.WAITING_FOR_FSM:  # 等待FSM（结构化输出语法）
            structured_output_req = request.structured_output_request  # 获取结构化输出请求
            if not (structured_output_req and structured_output_req.grammar):  # 语法尚未就绪
                return False  # 无法提升
            request.status = RequestStatus.WAITING  # 语法就绪，恢复等待状态
            return True  # 提升成功

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:  # 等待流式输入
            assert not request.streaming_queue  # 确保流式队列为空
            return False  # 无法提升（需要外部输入）

        raise AssertionError(  # 不应到达此处
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    # [中文注释] _update_from_kv_xfer_finished() — 更新已完成KV传输的调度器状态。
    #   处理已完成的接收（加入finished_recving集合）和发送（释放块）。
    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:  # 如果有KV连接器
            self.connector.update_connector_output(kv_connector_output)  # 更新连接器输出

        # KV Connector:: update recv and send status from last step.
        # KV连接器：更新上一步的接收和发送状态
        for req_id in kv_connector_output.finished_recving or ():  # 遍历已完成接收的请求
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests  # 确保请求存在
            req = self.requests[req_id]  # 获取请求
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 如果仍在等待远程KV
                self.finished_recving_kv_req_ids.add(req_id)  # 标记接收完成
            else:
                assert RequestStatus.is_finished(req.status)  # 请求已完成
                self._free_blocks(self.requests[req_id])  # 释放块（延迟释放的情况）
        for req_id in kv_connector_output.finished_sending or ():  # 遍历已完成发送的请求
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests  # 确保请求存在
            self._free_blocks(self.requests[req_id])  # 释放块

    # [中文注释] _update_requests_with_invalid_blocks() — 更新受无效KV缓存块影响的请求。
    #   扫描请求，检测哪些请求包含无效块，将其num_computed_tokens调整到最长有效前缀。
    #   返回：(受影响请求ID集合, 需要重计算的token总数, 需要驱逐的块ID集合)
    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()  # 受影响的请求ID集合
        total_affected_tokens = 0  # 需要重计算的token总数
        blocks_to_evict: set[int] = set()  # 需要驱逐的块ID集合
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute it.
        # 如果无效块被多个请求共享，只有第一个请求需要重新计算
        marked_invalid_block_ids: set[int] = set()  # 已标记为需要重新计算的无效块ID
        for request in requests:  # 遍历请求
            is_affected = False  # 请求是否受影响
            marked_invalid_block = False  # 是否已标记无效块
            req_id = request.request_id  # 请求ID
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)  # 获取请求的块ID列表
            # We iterate only over blocks that may contain externally computed
            # tokens
            # 只遍历可能包含外部计算token的块
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 异步加载
                # Async loading. num_computed_tokens does not include new tokens
                req_num_computed_tokens = request.num_computed_tokens  # 不含新token
            else:  # 同步加载
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens  # 含新token

            req_num_computed_blocks = (  # 计算已计算token对应的块数
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):  # 遍历块
                if block_id not in invalid_block_ids:  # 如果不是无效块
                    continue  # 跳过

                is_affected = True  # 标记请求受影响

                if block_id in marked_invalid_block_ids:  # 如果无效块已被其他请求标记
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue  # 该块将由之前的请求重新计算，当前请求仍可认为该块有效

                marked_invalid_block_ids.add(block_id)  # 标记该块需要重新计算

                if marked_invalid_block:  # 如果当前请求已标记过无效块
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue  # 已标记过，跳过

                marked_invalid_block = True  # 标记当前请求已处理无效块
                # Truncate the computed tokens at the first failed block
                # 在第一个失败块处截断已计算token
                request.num_computed_tokens = idx * self.block_size  # 回退已计算token数
                num_affected_tokens = (  # 计算受影响的token数
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens  # 累加受影响token数
                request.num_external_computed_tokens -= num_affected_tokens  # 减少外部计算token数
                # collect invalid block and all downstream dependent blocks
                # 收集无效块及所有下游依赖块
                if evict_blocks:  # 如果需要驱逐块
                    blocks_to_evict.update(req_block_ids[idx:])  # 添加该块及后续所有块

            if is_affected:  # 如果请求受影响
                if not marked_invalid_block:  # 如果所有无效块都由其他请求处理
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # 所有无效块都与其他请求共享，将由其他请求重新计算
                    total_affected_tokens += (  # 累加受影响token数
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens  # 回退到只保留缓存token

                affected_req_ids.add(request.request_id)  # 添加到受影响集合

        return affected_req_ids, total_affected_tokens, blocks_to_evict  # 返回结果

    # [中文注释] _handle_invalid_blocks() — 处理无效KV缓存块。
    #   识别受影响的请求，根据策略选择重计算或报错。
    #   返回需要跳过的请求ID集合。
    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures  # 是否应该直接失败（非重计算策略）

        # handle async KV loads (not cached yet, evict_blocks=False)
        # 处理异步KV加载（尚未缓存，不需要驱逐块）
        async_load_reqs = (  # 获取异步加载中的请求
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (  # 更新异步加载失败的请求
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)  # 异步失败请求数
        total_failed_tokens = num_failed_tokens  # 异步失败token数

        # handle sync loads (may be cached, collect blocks for eviction)
        # 处理同步KV加载（可能已缓存，收集需要驱逐的块）
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (  # 更新同步加载失败的请求
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)  # 累加同步失败请求数
        total_failed_tokens += num_failed_tokens  # 累加同步失败token数

        if not total_failed_requests:  # 如果没有失败的请求
            return set()  # 返回空集合

        # evict invalid blocks and downstream dependent blocks from cache
        # 驱逐无效块及下游依赖块（仅在非重计算策略时）
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)  # 驱逐块

        if should_fail:  # 如果策略是直接失败
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids  # 合并所有失败请求
            logger.error(  # 记录错误日志
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids  # 返回所有失败请求ID

        # 重计算策略：记录警告并安排重新调度
        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        # 标记异步请求的KV加载失败，等待加载完成后重试
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        # 返回同步受影响的ID，以便在update_from_output中跳过
        return sync_failed_req_ids
