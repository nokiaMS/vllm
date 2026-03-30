# SPDX-License-Identifier: Apache-2.0  # 指定开源许可证为 Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明，归 vLLM 项目贡献者所有

import functools  # 导入 functools 模块，提供高阶函数工具（如缓存装饰器等）
import gc  # 导入垃圾回收模块，用于手动控制内存回收
import itertools  # 导入 itertools 模块，提供高效的迭代器工具
import threading  # 导入线程模块，用于多线程同步操作
import time  # 导入时间模块，用于计时和延迟操作
from collections import defaultdict  # 从 collections 导入 defaultdict，提供带默认值的字典
from collections.abc import Iterable, Iterator, Sequence  # 导入抽象基类，用于类型提示（可迭代、迭代器、序列）
from contextlib import contextmanager  # 导入上下文管理器装饰器，用于创建上下文管理器
from copy import copy, deepcopy  # 导入浅拷贝和深拷贝函数
from dataclasses import dataclass, replace  # 导入数据类装饰器和替换函数
from functools import reduce  # 导入 reduce 函数，用于累积操作
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast  # 导入类型提示相关工具

import numpy as np  # 导入 NumPy 库，用于高效的数值计算
import torch  # 导入 PyTorch 深度学习框架
import torch.distributed  # 导入 PyTorch 分布式通信模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from tqdm import tqdm  # 导入 tqdm 进度条库，用于显示循环进度

import vllm.envs as envs  # 导入 vLLM 环境变量配置模块
from vllm.compilation.counter import compilation_counter  # 导入编译计数器，用于跟踪编译状态
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper  # 导入 CUDA Graph 统计和包装器类
from vllm.compilation.monitor import set_cudagraph_capturing_enabled  # 导入设置 CUDA Graph 捕获状态的函数
from vllm.config import (  # 从 vLLM 配置模块导入核心配置类
    CompilationMode,  # 编译模式枚举
    CUDAGraphMode,  # CUDA Graph 模式枚举
    VllmConfig,  # vLLM 全局配置类
    get_layers_from_vllm_config,  # 从配置中获取模型层信息的函数
    set_current_vllm_config,  # 设置当前活跃的 vLLM 配置
    update_config,  # 更新配置的函数
)
from vllm.config.cache import CacheConfig  # 导入缓存配置类，管理 KV 缓存参数
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer  # 导入专家缓存传输相关函数
from vllm.distributed.eplb.eplb_state import EplbState  # 导入专家并行负载均衡状态类
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group  # 导入 KV 缓存传输组相关函数
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks  # 导入 KV 块拷贝工具函数
from vllm.distributed.parallel_state import (  # 从并行状态模块导入分布式相关函数
    get_dcp_group,  # 获取解码上下文并行组
    get_pp_group,  # 获取流水线并行组
    get_tp_group,  # 获取张量并行组
    graph_capture,  # CUDA Graph 捕获上下文管理器
    is_global_first_rank,  # 判断是否为全局第一个 rank
    prepare_communication_buffer_for_model,  # 为模型准备通信缓冲区
)
from vllm.forward_context import (  # 从前向上下文模块导入
    BatchDescriptor,  # 批次描述符类
    set_forward_context,  # 设置前向传播上下文的函数
)
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.lora.layers import LoRAMapping, LoRAMappingType  # 导入 LoRA 映射类和映射类型
from vllm.model_executor.layers.attention import Attention, MLAAttention  # 导入标准注意力和多头潜在注意力类
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase  # 导入注意力层基类
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (  # 导入路由专家捕获器
    RoutedExpertsCapturer,  # 用于 CUDA Graph 中捕获 MoE 专家层的类
)
from vllm.model_executor.layers.rotary_embedding import (  # 导入旋转位置编码相关类
    MRotaryEmbedding,  # 多维旋转位置编码（M-RoPE，用于 Qwen2-VL 等多模态模型）
    XDRotaryEmbedding,  # 扩展维度旋转位置编码（XD-RoPE，用于 HunYuan-VL 等）
)
from vllm.model_executor.model_loader import get_model_loader  # 导入模型加载器获取函数
from vllm.model_executor.model_loader.reload import (  # 导入分层重载相关函数
    finalize_layerwise_reload,  # 完成分层重载
    initialize_layerwise_reload,  # 初始化分层重载
)
from vllm.model_executor.models.interfaces import (  # 导入模型接口和类型检查函数
    MultiModalEmbeddings,  # 多模态嵌入接口
    SupportsMRoPE,  # 支持 M-RoPE 的接口
    SupportsMultiModal,  # 支持多模态的接口
    SupportsXDRoPE,  # 支持 XD-RoPE 的接口
    is_mixture_of_experts,  # 检查模型是否为 MoE（专家混合）架构
    supports_eagle3,  # 检查是否支持 EAGLE3 投机解码
    supports_mrope,  # 检查是否支持 M-RoPE
    supports_multimodal_pruning,  # 检查是否支持多模态剪枝
    supports_realtime,  # 检查是否支持实时处理
    supports_transcription,  # 检查是否支持语音转录
    supports_xdrope,  # 检查是否支持 XD-RoPE
)
from vllm.model_executor.models.interfaces_base import (  # 导入模型基础接口
    VllmModelForPooling,  # 用于池化任务的 vLLM 模型接口
    is_pooling_model,  # 检查是否为池化模型
    is_text_generation_model,  # 检查是否为文本生成模型
)
from vllm.model_executor.offloader import (  # 导入模型权重卸载器相关函数
    create_offloader,  # 创建卸载器实例
    get_offloader,  # 获取当前卸载器
    set_offloader,  # 设置当前卸载器
)
from vllm.multimodal import MULTIMODAL_REGISTRY  # 导入多模态注册表全局单例
from vllm.multimodal.encoder_budget import MultiModalBudget  # 导入多模态编码器预算管理类
from vllm.multimodal.inputs import (  # 导入多模态输入相关类型
    BatchedTensorInputs,  # 批量张量输入类型
    MultiModalKwargsItem,  # 多模态关键字参数项
    PlaceholderRange,  # 占位符范围（用于标记多模态 token 位置）
)
from vllm.multimodal.utils import group_and_batch_mm_kwargs  # 导入多模态参数分组和批处理工具函数
from vllm.platforms import current_platform  # 导入当前平台信息（GPU 类型等）
from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.sampling_params import SamplingType  # 导入采样类型枚举
from vllm.sequence import IntermediateTensors  # 导入中间张量类（流水线并行中传递）
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask  # 导入任务类型定义
from vllm.tracing import instrument  # 导入追踪装饰器，用于性能分析
from vllm.utils import length_from_prompt_token_ids_or_embeds  # 导入从 prompt token 或嵌入获取长度的工具函数
from vllm.utils.math_utils import cdiv, round_up  # 导入向上取整除法和向上对齐函数
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib  # 导入设备内存分析器和 GiB 格式化函数
from vllm.utils.nvtx_pytorch_hooks import PytHooks  # 导入 NVTX PyTorch 钩子，用于 Nsight 性能分析
from vllm.utils.platform_utils import is_pin_memory_available, num_compute_units  # 导入内存锁页可用检查和计算单元数量函数
from vllm.utils.torch_utils import (  # 导入 PyTorch 工具函数
    get_dtype_size,  # 获取数据类型的字节大小
    kv_cache_dtype_str_to_dtype,  # 将 KV 缓存数据类型字符串转为 torch dtype
)
from vllm.v1.attention.backend import (  # 导入 v1 注意力后端相关类
    AttentionBackend,  # 注意力后端基类
    AttentionCGSupport,  # 注意力 CUDA Graph 支持信息
    AttentionMetadata,  # 注意力元数据基类
    AttentionMetadataBuilder,  # 注意力元数据构建器基类
    AttentionType,  # 注意力类型枚举
    CommonAttentionMetadata,  # 通用注意力元数据
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder  # 导入 GDN 注意力元数据构建器
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder  # 导入 Mamba2 注意力元数据构建器
from vllm.v1.attention.backends.utils import (  # 导入注意力后端工具函数
    create_fast_prefill_custom_backend,  # 创建快速预填充自定义后端
    get_dcp_local_seq_lens,  # 获取解码上下文并行的本地序列长度
    reorder_batch_to_split_decodes_and_prefills,  # 重排批次以分离解码和预填充请求
)
from vllm.v1.core.sched.output import NewRequestData  # 导入新请求数据类
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher  # 导入 CUDA Graph 调度器
from vllm.v1.kv_cache_interface import (  # 导入 KV 缓存接口定义
    AttentionSpec,  # 注意力规格基类
    ChunkedLocalAttentionSpec,  # 分块本地注意力规格
    CrossAttentionSpec,  # 交叉注意力规格
    EncoderOnlyAttentionSpec,  # 纯编码器注意力规格
    FullAttentionSpec,  # 全注意力规格
    KVCacheConfig,  # KV 缓存配置
    KVCacheGroupSpec,  # KV 缓存组规格
    KVCacheSpec,  # KV 缓存规格基类
    MambaSpec,  # Mamba 模型规格
    SlidingWindowSpec,  # 滑动窗口注意力规格
    UniformTypeKVCacheSpecs,  # 统一类型 KV 缓存规格
)
from vllm.v1.outputs import (  # 导入 v1 输出相关类
    EMPTY_MODEL_RUNNER_OUTPUT,  # 空模型运行器输出常量
    AsyncModelRunnerOutput,  # 异步模型运行器输出基类
    DraftTokenIds,  # 草稿 token ID 类型
    ECConnectorOutput,  # 专家缓存连接器输出
    KVConnectorOutput,  # KV 缓存连接器输出
    LogprobsLists,  # 日志概率列表类型
    LogprobsTensors,  # 日志概率张量类型
    ModelRunnerOutput,  # 模型运行器输出
    PoolerOutput,  # 池化器输出类型
    SamplerOutput,  # 采样器输出类型
    make_empty_encoder_model_runner_output,  # 创建空编码器模型运行器输出的函数
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates  # 导入池化元数据和池化状态类
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs  # 导入 logits 处理器和构建函数
from vllm.v1.sample.logits_processor.interface import LogitsProcessor  # 导入 logits 处理器接口
from vllm.v1.sample.metadata import SamplingMetadata  # 导入采样元数据类
from vllm.v1.sample.rejection_sampler import RejectionSampler  # 导入拒绝采样器（投机解码验证用）
from vllm.v1.sample.sampler import Sampler  # 导入采样器类
from vllm.v1.spec_decode.draft_model import DraftModelProposer  # 导入草稿模型提议器（投机解码）
from vllm.v1.spec_decode.eagle import EagleProposer  # 导入 EAGLE 投机解码提议器
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer  # 导入隐藏状态提取提议器
from vllm.v1.spec_decode.medusa import MedusaProposer  # 导入 Medusa 投机解码提议器
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata  # 导入投机解码元数据
from vllm.v1.spec_decode.ngram_proposer_gpu import (  # 导入 GPU N-gram 提议器相关
    NgramProposerGPU,  # GPU 上的 N-gram 提议器
    copy_num_valid_draft_tokens,  # 拷贝有效草稿 token 数量的函数
    update_ngram_gpu_tensors_incremental,  # 增量更新 N-gram GPU 张量的函数
    update_scheduler_for_invalid_drafts,  # 为无效草稿更新调度器的函数
)
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer  # 导入后缀解码提议器
from vllm.v1.structured_output.utils import apply_grammar_bitmask  # 导入应用语法位掩码的函数（结构化输出）
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext  # 导入 CPU-GPU 缓冲区和记录函数上下文
from vllm.v1.worker import mamba_utils  # 导入 Mamba 模型工具模块
from vllm.v1.worker.cp_utils import (  # 导入上下文并行工具函数
    check_attention_cp_compatibility,  # 检查注意力与上下文并行的兼容性
    get_total_cp_world_size,  # 获取上下文并行总 world size
)
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp  # 导入跨数据并行协调批次的函数
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin  # 导入专家缓存连接器模型运行器混入类
from vllm.v1.worker.gpu.pool.late_interaction_runner import LateInteractionRunner  # 导入晚交互运行器（ColBERT 等模型）
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch  # 导入缓存请求状态和输入批次类
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper  # 导入微批次包装器
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin  # 导入 KV 连接器模型运行器混入类
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin  # 导入 LoRA 模型运行器混入类
from vllm.v1.worker.ubatch_utils import (  # 导入微批次工具函数
    UBatchSlices,  # 微批次切片类
    check_ubatch_thresholds,  # 检查微批次阈值
    maybe_create_ubatch_slices,  # 可能创建微批次切片
    split_attn_metadata,  # 拆分注意力元数据
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp  # 导入检查残差是否为序列并行散布的函数
from vllm.v1.worker.workspace import lock_workspace  # 导入工作空间锁函数

from .utils import (  # 从当前包的 utils 模块导入
    AttentionGroup,  # 注意力组类
    KVBlockZeroer,  # KV 块清零器
    add_kv_sharing_layers_to_kv_cache_groups,  # 将 KV 共享层添加到 KV 缓存组的函数
    bind_kv_cache,  # 绑定 KV 缓存到模型层的函数
    prepare_kernel_block_sizes,  # 准备内核块大小的函数
    sanity_check_mm_encoder_outputs,  # 多模态编码器输出的健全性检查函数
)

if TYPE_CHECKING:  # 仅在类型检查时导入以下模块（避免运行时循环导入）
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 导入语法输出和调度器输出类型
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer  # 导入 N-gram 提议器类型

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]  # 定义类型别名：注意力元数据字典，键为层名，值为注意力元数据
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict  # 定义类型别名：每层注意力元数据，微批次模式下为列表


# 异步 GPU 模型输出包装器，用于支持异步调度下的执行与输出拷贝重叠。
# 在独立的 CUDA 流上将采样 token ID 和 logprobs 异步拷贝到 CPU，
# 调用 get_output() 时等待拷贝完成并解析最终结果（支持投机解码的拒绝采样解析）。
# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    """异步 GPU 模型运行器输出类。
    在异步调度模式下，将 GPU 上的采样结果（token ID 和 logprobs）
    通过独立 CUDA 流异步拷贝到 CPU，实现计算与数据传输的重叠，
    从而减少总体延迟。调用 get_output() 时同步等待拷贝完成。
    """
    def __init__(  # 构造函数
        self,
        model_runner_output: ModelRunnerOutput,  # 模型运行器输出对象
        sampled_token_ids: torch.Tensor,  # GPU 上采样得到的 token ID 张量
        logprobs_tensors: LogprobsTensors | None,  # GPU 上的日志概率张量（可选）
        invalid_req_indices: list[int],  # 无效请求的索引列表（需清空其输出）
        async_output_copy_stream: torch.cuda.Stream,  # 用于异步拷贝的 CUDA 流
        vocab_size: int,  # 词汇表大小
    ):
        self._model_runner_output = model_runner_output  # 保存模型运行器输出引用
        self._invalid_req_indices = invalid_req_indices  # 保存无效请求索引

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()  # 创建事件对象，用于同步异步拷贝操作

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids  # 保持对 GPU 采样 token 张量的引用，防止被回收
        self.vocab_size = vocab_size  # 保存词汇表大小
        self._logprobs_tensors = logprobs_tensors  # 保持对 GPU logprobs 张量的引用

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()  # 获取当前默认 CUDA 流
        with torch.cuda.stream(async_output_copy_stream):  # 切换到异步拷贝 CUDA 流
            async_output_copy_stream.wait_stream(default_stream)  # 等待默认流上的计算完成
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(  # 将采样 token ID 异步拷贝到 CPU
                "cpu", non_blocking=True  # 使用非阻塞拷贝
            )
            self._logprobs_tensors_cpu = (  # 将 logprobs 张量异步拷贝到 CPU（如果存在）
                self._logprobs_tensors.to_cpu_nonblocking()  # 调用非阻塞拷贝方法
                if self._logprobs_tensors  # 如果 logprobs 张量存在
                else None  # 否则为 None
            )
            self.async_copy_ready_event.record()  # 在拷贝流上记录事件，标记拷贝完成点

    def get_output(self) -> ModelRunnerOutput:
        """将设备张量拷贝到主机并返回 ModelRunnerOutput。
        此函数会阻塞等待直到拷贝完成。
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]  # 获取最大生成长度（最后一维大小）
        self.async_copy_ready_event.synchronize()  # 阻塞等待异步拷贝完成

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors  # 释放 GPU 上的 logprobs 张量引用
        del self._sampled_token_ids  # 释放 GPU 上的采样 token 张量引用
        if max_gen_len == 1:  # 如果最大生成长度为 1（非投机解码模式）
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()  # 将 CPU 张量转为 Python 列表
            for i in self._invalid_req_indices:  # 遍历无效请求索引
                valid_sampled_token_ids[i].clear()  # 清空无效请求的采样结果
            logprobs_lists = None  # 初始化 logprobs 列表为 None
            if self._logprobs_tensors_cpu is not None:  # 如果存在 CPU 上的 logprobs 张量
                logprobs_lists = self._logprobs_tensors_cpu.tolists()  # 转换为嵌套 Python 列表
        else:  # 投机解码模式（生成多个 token）
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(  # 使用拒绝采样器解析输出
                self.sampled_token_ids_cpu,  # CPU 上的采样 token ID
                self.vocab_size,  # 词汇表大小
                self._invalid_req_indices,  # 无效请求索引
                logprobs_tensors=self._logprobs_tensors_cpu,  # CPU 上的 logprobs 张量
            )

        output = self._model_runner_output  # 获取之前保存的模型运行器输出
        output.sampled_token_ids = valid_sampled_token_ids  # 设置有效的采样 token ID
        output.logprobs = logprobs_lists  # 设置 logprobs 列表
        return output  # 返回最终的模型运行器输出


# 将池化模型的输出从 GPU 异步拷贝到 CPU。
# 支持部分完成的请求（通过 finished_mask 过滤），
# 仅拷贝已完成的请求结果以减少不必要的数据传输。
def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]  # 接收原始池化输出和完成掩码
) -> list[torch.Tensor | None]:  # 返回每个请求的 CPU 张量列表（未完成的为 None）
    """将池化模型的原始输出从 GPU 异步拷贝到 CPU。
    仅拷贝 finished_mask 为 True 的请求结果，减少不必要的数据传输。
    支持张量和列表两种输出格式。
    """
    num_reqs = len(finished_mask)  # 获取请求总数

    if isinstance(raw_pooler_output, torch.Tensor):  # 如果池化输出是一个张量
        if raw_pooler_output.shape[0] != num_reqs:  # 检查批次大小是否匹配
            raise ValueError(  # 不匹配则抛出错误
                "Pooler output batch size does not match finished mask size: "  # 错误消息
                f"{raw_pooler_output.shape[0]} != {num_reqs}."  # 显示实际大小
            )

        num_finished = sum(finished_mask)  # 计算已完成请求的数量
        if num_finished == 0:  # 如果没有请求完成
            return [None] * num_reqs  # 返回全 None 列表
        if num_finished == num_reqs:  # 如果所有请求都完成了
            return list(raw_pooler_output.to("cpu", non_blocking=True))  # 将整个张量异步拷贝到 CPU 并转为列表

        # partial finished
        finished_indices = [i for i, include in enumerate(finished_mask) if include]  # 获取已完成请求的索引列表
        index_tensor = torch.tensor(  # 创建索引张量
            finished_indices, device=raw_pooler_output.device, dtype=torch.long  # 放在与输出相同的设备上
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(  # 根据索引选取已完成的输出并拷贝到 CPU
            "cpu", non_blocking=True  # 使用非阻塞拷贝
        )
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs  # 初始化结果列表，全为 None
        for i, out in zip(finished_indices, finished_outputs):  # 遍历已完成的索引和对应输出
            partial_pooler_output[i] = out  # 将输出放到对应位置
        return partial_pooler_output  # 返回部分完成的结果列表

    assert isinstance(raw_pooler_output, list)  # 断言池化输出是列表类型
    if len(raw_pooler_output) != num_reqs:  # 检查列表长度是否匹配
        raise ValueError(  # 不匹配则抛出错误
            "Pooler output batch size does not match finished mask size: "  # 错误消息
            f"{len(raw_pooler_output)} != {num_reqs}."  # 显示实际大小
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs  # 初始化结果列表，全为 None
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):  # 遍历每个输出和对应掩码
        if include and out is not None:  # 如果该请求已完成且输出不为 None
            pooler_output[i] = out.to("cpu", non_blocking=True)  # 异步拷贝到 CPU
    return pooler_output  # 返回结果列表


# 异步 GPU 池化模型输出包装器，类似 AsyncGPUModelRunnerOutput，
# 但专门处理池化模型（如 embedding 模型）的输出拷贝。
class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    """异步 GPU 池化模型运行器输出类。
    专门用于池化模型（如嵌入模型），在独立的 CUDA 流上
    将池化结果异步拷贝到 CPU，实现计算与数据传输的重叠。
    """
    def __init__(  # 构造函数
        self,
        model_runner_output: ModelRunnerOutput,  # 模型运行器输出对象
        raw_pooler_output: PoolerOutput,  # GPU 上的原始池化输出
        finished_mask: list[bool],  # 请求完成掩码
        async_output_copy_stream: torch.cuda.Stream,  # 用于异步拷贝的 CUDA 流
    ):
        self._model_runner_output = model_runner_output  # 保存模型运行器输出引用

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()  # 创建事件对象用于同步

        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output  # 保持对 GPU 池化输出的引用防止被回收

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()  # 获取当前默认 CUDA 流
        with torch.cuda.stream(async_output_copy_stream):  # 切换到异步拷贝 CUDA 流
            async_output_copy_stream.wait_stream(default_stream)  # 等待默认流上的计算完成
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(  # 将池化输出异步拷贝到 CPU
                raw_pooler_output=self._raw_pooler_output,  # 传入原始池化输出
                finished_mask=finished_mask,  # 传入完成掩码
            )
            self.async_copy_ready_event.record()  # 在拷贝流上记录事件

    def get_output(self) -> ModelRunnerOutput:
        """将设备张量拷贝到主机并返回 ModelRunnerOutput。
        此函数会阻塞等待直到拷贝完成。
        """
        self.async_copy_ready_event.synchronize()  # 阻塞等待异步拷贝完成

        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output  # 释放 GPU 上的池化输出引用
        return self._model_runner_output  # 返回包含 CPU 数据的模型运行器输出


# execute_model() 和 sample_tokens() 之间传递的临时状态。
# 在异步调度模式下，execute_model() 完成前向传播后返回 None，
# 将中间结果（logits、隐藏状态、投机解码元数据等）缓存在此结构中，
# 供后续的 sample_tokens() 调用使用。
class ExecuteModelState(NamedTuple):
    """execute_model() 和 sample_tokens() 之间传递的临时缓存状态。
    在异步调度模式下，execute_model() 完成前向传播后返回 None，
    将 logits、隐藏状态、投机解码元数据等中间结果缓存于此，
    供后续 sample_tokens() 使用。
    """

    scheduler_output: "SchedulerOutput"  # 调度器输出，包含本步的请求和调度信息
    logits: torch.Tensor  # 模型输出的 logits 张量
    spec_decode_metadata: SpecDecodeMetadata | None  # 投机解码元数据（可选）
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None  # 投机解码的通用注意力元数据（可选）
    hidden_states: torch.Tensor  # 模型最后一层的隐藏状态
    sample_hidden_states: torch.Tensor  # 用于采样的隐藏状态
    aux_hidden_states: list[torch.Tensor] | None  # 辅助隐藏状态列表（EAGLE3 等需要）
    ec_connector_output: ECConnectorOutput | None  # 专家缓存连接器输出（可选）
    cudagraph_stats: CUDAGraphStat | None  # CUDA Graph 统计信息（可选）
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None  # 槽位映射（KV 缓存写入位置）


# GPU 模型运行器，vLLM v1 引擎的核心推理组件。
# 职责：管理模型加载、输入准备（token 拼接、位置编码、注意力元数据构建）、
# 前向传播执行、采样/池化、投机解码（N-gram/EAGLE/Medusa/Draft Model）、
# 多模态编码器执行与缓存、KV 缓存初始化与管理、CUDA Graph 捕获与调度、
# 以及 EPLB（专家并行负载均衡）。
# 采用持久化批次（InputBatch）模式，步间增量更新以最小化开销。
# 通过混入类获得 LoRA、KV 连接器和 EC 连接器的能力。
class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    """GPU 模型运行器类，vLLM v1 引擎的核心推理组件。
    负责管理模型加载、输入准备（token 拼接、位置编码、注意力元数据构建）、
    前向传播执行、采样/池化、投机解码、多模态编码器执行与缓存、
    KV 缓存初始化与管理、CUDA Graph 捕获与调度、以及 EPLB。
    通过 LoRAModelRunnerMixin、KVConnectorModelRunnerMixin 和
    ECConnectorModelRunnerMixin 混入类获得 LoRA、KV 连接器和 EC 连接器能力。
    """
    def __init__(  # 构造函数
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置对象
        device: torch.device,  # 运行设备（GPU）
    ):
        self.vllm_config = vllm_config  # 保存 vLLM 全局配置
        self.model_config = vllm_config.model_config  # 提取模型配置
        self.cache_config = vllm_config.cache_config  # 提取缓存配置
        self.offload_config = vllm_config.offload_config  # 提取权重卸载配置
        self.compilation_config = vllm_config.compilation_config  # 提取编译配置
        self.lora_config = vllm_config.lora_config  # 提取 LoRA 配置
        self.load_config = vllm_config.load_config  # 提取模型加载配置
        self.parallel_config = vllm_config.parallel_config  # 提取并行配置
        self.scheduler_config = vllm_config.scheduler_config  # 提取调度器配置
        self.speculative_config = vllm_config.speculative_config  # 提取投机解码配置
        self.observability_config = vllm_config.observability_config  # 提取可观测性配置

        model_config = self.model_config  # 本地引用模型配置，方便后续使用
        cache_config = self.cache_config  # 本地引用缓存配置
        scheduler_config = self.scheduler_config  # 本地引用调度器配置
        parallel_config = self.parallel_config  # 本地引用并行配置
        self.device = device  # 保存设备信息
        self.pin_memory = is_pin_memory_available()  # 检查是否可用页锁定内存（加速 CPU-GPU 数据传输）
        self.dtype = self.model_config.dtype  # 获取模型数据类型（如 float16, bfloat16）

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(  # 将 KV 缓存数据类型字符串转为 torch dtype
            cache_config.cache_dtype, self.model_config  # 从缓存配置和模型配置获取
        )

        self.is_pooling_model = model_config.runner_type == "pooling"  # 判断是否为池化模型（如嵌入模型）
        self.enable_prompt_embeds = model_config.enable_prompt_embeds  # 是否启用 prompt 嵌入输入
        self.is_multimodal_raw_input_only_model = (  # 是否为仅接受原始多模态输入的模型
            model_config.is_multimodal_raw_input_only_model  # 从模型配置获取
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False  # 多模态剪枝是否启用，将在 load_model() 中覆盖
        # Set to True after init_routed_experts_capturer() completes.
        # Prevents routed experts code from running during profiling/dummy run.
        self.routed_experts_initialized = False  # 路由专家是否已初始化，防止在 profiling 时运行
        self.max_model_len = model_config.max_model_len  # 模型支持的最大序列长度

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales  # 是否计算 KV 缓存缩放因子
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size  # 解码上下文并行的 world size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group  # 当前进程在解码上下文并行组中的 rank
        self.max_num_tokens = scheduler_config.max_num_batched_tokens  # 每步最大批处理 token 数
        self.max_num_reqs = scheduler_config.max_num_seqs  # 每步最大并发请求数

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping micro-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (  # 是否在流水线并行 rank 之间广播输出
            self.parallel_config.distributed_executor_backend == "external_launcher"  # 使用外部启动器（torchrun）
            and len(get_pp_group().ranks) > 1  # 且流水线并行组大小大于 1
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)  # 获取当前并行配置下的查询头数量
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()  # 获取输入嵌入维度大小
        self.attention_chunk_size = model_config.attention_chunk_size  # 获取注意力分块大小
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi  # 是否使用 ALiBi 位置编码（如 MPT 模型）

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn  # 是否启用级联注意力
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm  # 是否为多模态前缀语言模型

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY  # 保存多模态注册表引用
        self.uses_mrope = model_config.uses_mrope  # 是否使用多维旋转位置编码（M-RoPE）
        self.uses_xdrope_dim = model_config.uses_xdrope_dim  # 使用的 XD-RoPE 维度数（0 表示不使用）
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(  # 检查模型是否支持多模态输入
            model_config  # 传入模型配置
        )

        if self.model_config.is_encoder_decoder:  # 如果是编码器-解码器模型
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens  # 设置编码器最大输入长度
        else:  # 如果不是编码器-解码器模型
            self.max_encoder_len = 0  # 编码器最大长度为 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling  # 是否启用异步调度

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)  # 初始化采样器，设置 logprobs 模式

        self.eplb_state: EplbState | None = None  # EPLB 状态，模型加载后懒初始化
        # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
        self.eep_eplb_suppressed = False  # 标志位：在扩缩容期间临时禁用 EPLB
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []  # KV 缓存张量列表，在 initialize_kv_cache 中初始化
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None  # 交叉注意力层 KV 缓存张量
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None  # 交叉注意力层使用的注意力后端类型
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []  # 注意力组列表，按 KV 缓存组和注意力组索引

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}  # 多模态编码器缓存：哈希值 -> 编码器输出张量
        self.late_interaction_runner = LateInteractionRunner()  # 晚交互运行器实例（用于 ColBERT 等模型）

        self.use_aux_hidden_state_outputs = False  # 是否使用辅助隐藏状态输出（EAGLE3 等需要）
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:  # 如果启用了投机解码且当前是最后一个流水线 rank
            self.drafter: (  # 声明草稿提议器的类型注解
                NgramProposer  # noqa: F823  # N-gram 提议器
                | NgramProposerGPU  # GPU N-gram 提议器
                | SuffixDecodingProposer  # 后缀解码提议器
                | EagleProposer  # EAGLE 提议器
                | DraftModelProposer  # 草稿模型提议器
                | MedusaProposer  # Medusa 提议器
                | ExtractHiddenStatesProposer  # 隐藏状态提取提议器
            )
            if self.speculative_config.method == "ngram":  # 如果投机解码方法为 N-gram
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer  # 延迟导入 N-gram 提议器

                self.drafter = NgramProposer(self.vllm_config)  # 创建 N-gram 提议器实例
            elif self.speculative_config.uses_draft_model():  # 如果使用草稿模型
                self.drafter = DraftModelProposer(  # 创建草稿模型提议器
                    vllm_config=self.vllm_config,  # 传入全局配置
                    device=self.device,  # 传入设备
                    runner=self,  # 传入当前运行器引用
                )
            elif self.speculative_config.use_ngram_gpu():  # 如果使用 GPU N-gram 方法
                self.drafter = NgramProposerGPU(self.vllm_config, self.device, self)  # 创建 GPU N-gram 提议器
                self.num_tokens_no_spec_gpu = torch.zeros(  # 创建不使用投机解码的 token 数量张量
                    self.max_num_reqs, dtype=torch.int32, device=device  # 形状为最大请求数
                )
                self.token_ids_gpu_tensor = torch.zeros(  # 创建 GPU 上的 token ID 张量
                    self.max_num_reqs,  # 行数为最大请求数
                    self.max_model_len,  # 列数为最大模型长度
                    dtype=torch.int32,  # 整型
                    device=device,  # 放在 GPU 上
                )
                self._ngram_pinned_idx_buf = torch.zeros(  # 创建页锁定内存的索引缓冲区
                    self.max_num_reqs, dtype=torch.long, pin_memory=True  # 用于 N-gram 异步传输
                )
                self._ngram_pinned_val_buf = torch.zeros(  # 创建页锁定内存的值缓冲区
                    self.max_num_reqs, dtype=torch.int32, pin_memory=True  # 用于 N-gram 异步传输
                )
            elif self.speculative_config.method == "suffix":  # 如果投机解码方法为后缀解码
                self.drafter = SuffixDecodingProposer(self.vllm_config)  # 创建后缀解码提议器
            elif self.speculative_config.use_eagle():  # 如果使用 EAGLE 投机解码
                self.drafter = EagleProposer(self.vllm_config, self.device, self)  # 创建 EAGLE 提议器
                if self.speculative_config.method == "eagle3":  # 如果是 EAGLE3 方法
                    self.use_aux_hidden_state_outputs = (  # 设置是否使用辅助隐藏状态
                        self.drafter.eagle3_use_aux_hidden_state  # 从提议器获取配置
                    )
            elif self.speculative_config.method == "medusa":  # 如果投机解码方法为 Medusa
                self.drafter = MedusaProposer(  # 创建 Medusa 提议器
                    vllm_config=self.vllm_config, device=self.device  # 传入配置和设备
                )
            elif self.speculative_config.method == "extract_hidden_states":  # 如果使用隐藏状态提取方法
                self.drafter = ExtractHiddenStatesProposer(  # 创建隐藏状态提取提议器
                    vllm_config=self.vllm_config, device=self.device  # 传入配置和设备
                )
                self.use_aux_hidden_state_outputs = True  # 启用辅助隐藏状态输出
            else:  # 未知的投机解码方法
                raise ValueError(  # 抛出错误
                    "Unknown speculative decoding method: "  # 错误消息前缀
                    f"{self.speculative_config.method}"  # 显示未知的方法名
                )
            self.rejection_sampler = RejectionSampler(self.sampler)  # 创建拒绝采样器，传入基础采样器

        self.num_spec_tokens = 0  # 初始化投机 token 数量为 0
        if self.speculative_config:  # 如果启用了投机解码
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens  # 获取投机 token 数量
            draft_config = self.speculative_config.draft_model_config  # 获取草稿模型配置
            if draft_config is not None and draft_config.max_model_len is not None:  # 如果草稿模型配置和最大长度都存在
                self.effective_drafter_max_model_len = draft_config.max_model_len  # 使用草稿模型的最大长度
            else:  # 否则
                self.effective_drafter_max_model_len = self.max_model_len  # 使用主模型的最大长度

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}  # 请求状态字典：请求 ID -> 缓存的请求状态
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}  # prompt logprobs 数量字典：仅包含预填充阶段的请求
        self.comm_stream = torch.cuda.Stream()  # 创建通信专用 CUDA 流

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors  # 获取模型配置中的 logits 处理器列表
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (  # 创建自定义 logits 处理器序列
            tuple(logits_processors) if logits_processors is not None else ()  # 存在则转为元组，否则空元组
        )
        placeholder_block_size = (  # 占位符块大小
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE  # 从配置获取或使用默认值
        )
        self._init_block_sizes = [placeholder_block_size]  # 保存初始块大小列表
        self._init_kernel_block_sizes = [placeholder_block_size]  # 保存初始内核块大小列表
        self.input_batch = InputBatch(  # 创建输入批次对象
            max_num_reqs=self.max_num_reqs,  # 最大请求数
            # We need to use the encoder length for encoder-decoder
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),  # 取模型长度和编码器长度的最大值
            max_num_batched_tokens=self.max_num_tokens,  # 最大批处理 token 数
            device=self.device,  # 设备
            pin_memory=self.pin_memory,  # 是否使用页锁定内存
            vocab_size=self.model_config.get_vocab_size(),  # 词汇表大小
            block_sizes=[placeholder_block_size],  # 块大小列表
            kernel_block_sizes=[placeholder_block_size],  # 内核块大小列表
            is_spec_decode=bool(self.vllm_config.speculative_config),  # 是否为投机解码模式
            logitsprocs=build_logitsprocs(  # 构建 logits 处理器
                self.vllm_config,  # 全局配置
                self.device,  # 设备
                self.pin_memory,  # 页锁定内存
                self.is_pooling_model,  # 是否为池化模型
                custom_logitsprocs,  # 自定义处理器
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),  # 保守设置：自定义处理器是否需要输出 token ID
            is_pooling_model=self.is_pooling_model,  # 是否为池化模型
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,  # 上下文并行 KV 缓存交错大小
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None  # 异步输出拷贝流（异步调度时使用）
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None  # 输入准备事件（异步调度时同步 CPU 张量复用）
        if self.use_async_scheduling:  # 如果启用异步调度
            self.async_output_copy_stream = torch.cuda.Stream()  # 创建异步输出拷贝流
            self.prepare_inputs_event = torch.Event()  # 创建输入准备事件

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (  # 如果配置了 CUDA Graph 捕获尺寸且未禁用 CUDA Graph
            self.compilation_config.cudagraph_capture_sizes  # 存在捕获尺寸列表
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE  # 且 CUDA Graph 模式不为 NONE
        ):
            self.cudagraph_batch_sizes = sorted(  # 对捕获尺寸排序（升序）
                self.compilation_config.cudagraph_capture_sizes  # 从编译配置获取
            )
        else:  # 否则
            self.cudagraph_batch_sizes = []  # 空列表表示不使用 CUDA Graph

        # Cache the device properties.
        self._init_device_properties()  # 初始化并缓存设备属性

        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}  # 编码器计时注册表（用于可观测性）
        self._encoder_timing_lock = threading.Lock()  # 编码器计时锁（线程安全）

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)  # 创建输入 token ID 持久缓冲区
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)  # 创建位置编码持久缓冲区
        self.query_start_loc = self._make_buffer(  # 创建查询起始位置持久缓冲区
            self.max_num_reqs + 1, dtype=torch.int32  # 大小为最大请求数 + 1
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)  # 创建序列长度持久缓冲区
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)  # 创建编码器序列长度持久缓冲区
        if self.dcp_world_size > 1:  # 如果解码上下文并行 world size 大于 1
            self.dcp_local_seq_lens = self._make_buffer(  # 创建 DCP 本地序列长度缓冲区
                self.max_num_reqs, dtype=torch.int32  # 大小为最大请求数
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(  # 创建输入嵌入持久缓冲区
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False  # 不创建 numpy 版本（bfloat16 不兼容 numpy）
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)  # 创建 token ID 标记持久缓冲区
        self.discard_request_mask = self._make_buffer(  # 创建丢弃请求掩码持久缓冲区
            self.max_num_reqs, dtype=torch.bool  # 布尔类型
        )
        self.num_decode_draft_tokens = self._make_buffer(  # 创建解码草稿 token 数量持久缓冲区
            self.max_num_reqs, dtype=torch.int32  # 整型
        )
        self.num_accepted_tokens = self._make_buffer(  # 创建已接受 token 数量持久缓冲区
            self.max_num_reqs, dtype=torch.int64  # 64 位整型
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:  # 如果模型支持多模态输入
            # Double buffer to avoid race condition: previous iteration's async
            # copy may still be reading from CPU while current iteration writes.
            self.is_mm_embed_buffers = [  # 创建双缓冲区避免竞争条件
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),  # 缓冲区 0
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),  # 缓冲区 1
            ]
            self.is_mm_embed_idx = 0  # 当前使用的缓冲区索引

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:  # 如果使用 M-RoPE（如 Qwen2-VL）
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(  # 创建 M-RoPE 位置编码缓冲区（3D）
                (3, self.max_num_tokens + 1), dtype=torch.int64  # 3 个维度，额外 1 个虚拟位置使其非连续
            )

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:  # 如果使用 XD-RoPE（如 HunYuan-VL）
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(  # 创建 XD-RoPE 位置编码缓冲区
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64  # 维度数 x (最大 token 数 + 1)
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None  # 中间张量（流水线并行中各 rank 间传递）

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(  # 预缓存 numpy 整数范围数组（避免每步重建）
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),  # 取三者最大值
            dtype=np.int64,  # 使用 int64 避免长上下文溢出
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}  # 交叉层 KV 共享映射：层名 -> 共享源层名
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()  # 可使用快速预填充的 KV 共享层集合

        self.kv_sharing_fast_prefill_logits_indices = None  # KV 共享快速预填充的 logits 索引
        if self.cache_config.kv_sharing_fast_prefill:  # 如果启用 KV 共享快速预填充
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(  # 创建 logits 索引张量
                self.max_num_tokens, dtype=torch.int32, device=self.device  # 放在 GPU 上
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens  # 统一解码查询长度 = 1 + 投机 token 数

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)  # 创建 CUDA Graph 调度器

        self.mm_budget = (  # 多模态编码器预算管理器
            MultiModalBudget(self.vllm_config, self.mm_registry)  # 如果支持多模态则创建
            if self.supports_mm_inputs  # 条件判断
            else None  # 否则为 None
        )

        self.reorder_batch_threshold: int | None = None  # 批次重排阈值（分离解码和预填充）

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()  # 仅存在于运行器 KVCacheConfig 中的注意力层集合

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None  # 缓存的草稿 token ID
        # N-gram GPU path: async D2H buffer/event for per-request valid draft counts.
        self._num_valid_draft_tokens: torch.Tensor | None = None  # GPU 上有效草稿 token 数量张量
        self._num_valid_draft_tokens_cpu: torch.Tensor | None = None  # CPU 上有效草稿 token 数量张量
        self._num_valid_draft_tokens_event: torch.cuda.Event | None = None  # 异步拷贝事件
        self._num_valid_draft_tokens_copy_stream: torch.cuda.Stream | None = None  # 异步拷贝流
        if (  # 如果启用了 GPU N-gram 投机解码
            self.speculative_config is not None  # 投机配置存在
            and self.speculative_config.use_ngram_gpu()  # 且使用 GPU N-gram
        ):
            self._num_valid_draft_tokens_cpu = torch.empty(  # 创建 CPU 端有效草稿 token 数量张量
                self.max_num_reqs, dtype=torch.int32, pin_memory=self.pin_memory  # 页锁定内存
            )
            self._num_valid_draft_tokens_event = torch.cuda.Event()  # 创建 CUDA 事件
            self._num_valid_draft_tokens_copy_stream = torch.cuda.Stream()  # 创建 CUDA 流

        self._draft_token_req_ids: list[str] | None = None  # 草稿 token 对应的请求 ID 列表
        self.transfer_event = torch.Event()  # 创建传输事件
        self.sampled_token_ids_pinned_cpu = torch.empty(  # 创建 CPU 端页锁定的采样 token ID 张量
            (self.max_num_reqs, 1),  # 形状为 (最大请求数, 1)
            dtype=torch.int64,  # 64 位整型
            device="cpu",  # CPU 设备
            pin_memory=self.pin_memory,  # 使用页锁定内存
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None  # 有效采样 token 计数事件
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None  # 有效采样 token 计数拷贝流
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None  # 草稿 token ID 拷贝事件
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None  # 草稿 token ID 拷贝流
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None  # CPU 端有效采样 token 计数
        self.draft_token_ids_cpu: torch.Tensor | None = None  # CPU 端草稿 token ID
        self.num_accepted_tokens_event: torch.Event | None = None  # 已接受 token 数量事件
        if self.num_spec_tokens:  # 如果有投机 token
            self.draft_token_ids_event = torch.Event()  # 创建草稿 token ID 事件
            self.num_accepted_tokens_event = torch.Event()  # 创建已接受 token 数量事件
            self.draft_token_ids_copy_stream = torch.cuda.Stream()  # 创建草稿 token ID 拷贝流
            self.draft_token_ids_cpu = torch.empty(  # 创建 CPU 端草稿 token ID 张量
                (self.max_num_reqs, self.num_spec_tokens),  # 形状为 (最大请求数, 投机 token 数)
                dtype=torch.int64,  # 64 位整型
                device="cpu",  # CPU 设备
                pin_memory=self.pin_memory,  # 使用页锁定内存
            )
            if self.use_async_scheduling:  # 如果启用异步调度
                self.valid_sampled_token_count_event = torch.Event()  # 创建有效采样计数事件
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()  # 创建有效采样计数拷贝流
                self.valid_sampled_token_count_cpu = torch.empty(  # 创建 CPU 端有效采样计数张量
                    self.max_num_reqs,  # 大小为最大请求数
                    dtype=torch.int64,  # 64 位整型
                    device="cpu",  # CPU 设备
                    pin_memory=self.pin_memory,  # 使用页锁定内存
                )

        # Model weight offloader
        # Make sure this is called before any get_offloader call
        set_offloader(create_offloader(self.offload_config))  # 创建并设置模型权重卸载器

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None  # execute_model 与 sample_tokens 间的临时状态
        self.kv_connector_output: KVConnectorOutput | None = None  # KV 连接器输出
        self.mamba_state_idx: dict[str, int] = {}  # Mamba 状态索引字典
        self._mamba_copy_bufs: mamba_utils.MambaCopyBuffers | None = None  # Mamba 拷贝缓冲区
        self.layerwise_nvtx_hooks_registered = False  # 是否已注册逐层 NVTX 钩子
    # 更新模型支持的最大序列长度
    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len  # 设置最大模型长度
        if self.speculative_config:  # 如果启用了投机解码配置
            draft_config = self.speculative_config.draft_model_config  # 获取草稿模型配置
            if draft_config is None or draft_config.max_model_len is None:  # 如果草稿配置或其最大长度为空
                self.effective_drafter_max_model_len = self.max_model_len  # 使用主模型的最大长度作为草稿模型的有效最大长度

    # 重置多模态缓存（在性能分析后不再需要）
    def reset_mm_cache(self) -> None:
        """
        重置多模态缓存。
        清除在性能分析期间使用的多模态缓存，推理时不再需要。
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        if self.mm_budget:  # 如果存在多模态预算对象
            self.mm_budget.reset_cache()  # 重置多模态缓存
        self.late_interaction_runner.clear()  # 清除延迟交互运行器

    # 清除GPU端的编码器缓存（存储视觉嵌入向量）
    def reset_encoder_cache(self) -> None:
        """
        清除GPU端的编码器缓存，用于存储视觉嵌入向量。
        当模型权重更新时应调用此方法，以确保不会重用旧权重计算的过期嵌入。
        Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()  # 清除编码器缓存
        self.late_interaction_runner.clear()  # 清除延迟交互运行器

    # 从休眠唤醒后重新初始化KV缓存和FP8缩放因子
    @torch.inference_mode()  # 使用推理模式装饰器，禁用梯度计算
    def init_fp8_kv_scales(self) -> None:
        """
        从休眠唤醒后重新初始化KV缓存和FP8缩放因子。
        1. 将KV缓存张量清零以移除重新分配后的垃圾数据。
        2. 将注意力层缩放因子(_k_scale, _v_scale)重置为1.0。
           如果这些值保持0.0（唤醒后的默认值），所有KV缓存值实际上变为零，导致输出乱码。
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):  # 如果缓存数据类型不是fp8，直接返回
            return

        kv_caches = getattr(self, "kv_caches", [])  # 获取KV缓存张量列表
        for cache_tensor in kv_caches:  # 遍历每个KV缓存张量
            if cache_tensor is not None:  # 如果缓存张量非空
                cache_tensor.zero_()  # 将缓存张量清零，移除脏数据

        k_attr_names = ("_k_scale", "k_scale")  # K缩放因子的可能属性名
        v_attr_names = ("_v_scale", "v_scale")  # V缩放因子的可能属性名

        attn_layers = self.compilation_config.static_forward_context  # 获取所有注意力层
        for name, module in attn_layers.items():  # 遍历每个注意力层模块
            if isinstance(module, (Attention, MLAAttention)):  # 如果是Attention或MLAAttention类型
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0  # 设置K和V的缩放值为1.0

                # Processing K Scale
                for attr in k_attr_names:  # 遍历K缩放因子的属性名
                    if hasattr(module, attr):  # 如果模块有该属性
                        param = getattr(module, attr)  # 获取属性值
                        if isinstance(param, torch.Tensor):  # 如果是张量类型
                            param.fill_(k_scale_val)  # 填充为K缩放值

                # Processing V Scale
                for attr in v_attr_names:  # 遍历V缩放因子的属性名
                    if hasattr(module, attr):  # 如果模块有该属性
                        param = getattr(module, attr)  # 获取属性值
                        if isinstance(param, torch.Tensor):  # 如果是张量类型
                            param.fill_(v_scale_val)  # 填充为V缩放值

    # 根据token数量获取对应的位置编码张量（支持M-RoPE、XD-RoPE和普通位置编码）
    def _get_positions(self, num_tokens: Any):
        """
        根据token数量获取对应的位置编码张量。
        支持M-RoPE（多维旋转位置编码）、XD-RoPE（跨维旋转位置编码）和普通位置编码三种模式。
        num_tokens可以是整数（切片模式）或索引张量（高级索引模式）。
        """
        if isinstance(num_tokens, int):  # 如果num_tokens是整数（切片模式）
            if self.uses_mrope:  # 如果使用M-RoPE位置编码
                return self.mrope_positions.gpu[:, :num_tokens]  # 返回M-RoPE位置的GPU切片
            if self.uses_xdrope_dim > 0:  # 如果使用XD-RoPE位置编码
                return self.xdrope_positions.gpu[:, :num_tokens]  # 返回XD-RoPE位置的GPU切片
            return self.positions.gpu[:num_tokens]  # 返回普通位置编码的GPU切片
        else:  # 如果num_tokens是索引张量（高级索引模式）
            if self.uses_mrope:  # 如果使用M-RoPE位置编码
                return self.mrope_positions.gpu[:, num_tokens]  # 按索引获取M-RoPE位置
            if self.uses_xdrope_dim > 0:  # 如果使用XD-RoPE位置编码
                return self.xdrope_positions.gpu[:, num_tokens]  # 按索引获取XD-RoPE位置
            return self.positions.gpu[num_tokens]  # 按索引获取普通位置编码

    # 创建一个CPU-GPU双缓冲区对象
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True  # 接受大小、数据类型和是否创建numpy视图的参数
    ) -> CpuGpuBuffer:
        """
        创建一个CPU-GPU双缓冲区对象。
        该缓冲区同时维护CPU和GPU端的数据副本，支持高效的数据传输。
        """
        return CpuGpuBuffer(  # 返回新创建的CPU-GPU缓冲区
            *size,
            dtype=dtype,  # 数据类型
            device=self.device,  # 目标设备
            pin_memory=self.pin_memory,  # 是否锁页内存
            with_numpy=numpy,  # 是否创建numpy视图
        )

    # 获取或懒初始化Mamba状态复制缓冲区
    def _get_mamba_copy_bufs(self) -> mamba_utils.MambaCopyBuffers:
        """
        获取或懒初始化Mamba状态复制缓冲区。
        如果缓冲区尚未创建则创建之，否则直接返回已有的缓冲区。
        """
        if self._mamba_copy_bufs is None:  # 如果缓冲区尚未创建
            self._mamba_copy_bufs = mamba_utils.MambaCopyBuffers.create(  # 创建Mamba复制缓冲区
                self.max_num_reqs,  # 最大请求数
                self.kv_cache_config,  # KV缓存配置
                self.model.get_mamba_state_copy_func(),  # 获取Mamba状态复制函数
                self._make_buffer,  # 缓冲区创建工厂方法
            )
        return self._mamba_copy_bufs  # 返回Mamba复制缓冲区

    # 初始化模型前向传播的额外关键字参数（主要用于池化模型的token_type_ids）
    def _init_model_kwargs(self):
        """
        初始化模型前向传播的额外关键字参数。
        主要用于池化模型，处理token_type_ids的生成和传递。
        对于非池化模型直接返回空字典。
        """
        model_kwargs = dict[str, Any]()  # 创建空的模型参数字典

        if not self.is_pooling_model:  # 如果不是池化模型
            return model_kwargs  # 直接返回空字典

        num_reqs = self.input_batch.num_reqs  # 获取当前批次中的请求数
        pooling_params = self.input_batch.get_pooling_params()  # 获取池化参数列表

        token_type_id_requests = dict[int, Any]()  # 存储需要token_type_ids的请求索引
        for i, param in enumerate(pooling_params):  # 遍历每个请求的池化参数
            if (
                param.extra_kwargs is not None  # 如果存在额外参数
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))  # 获取压缩的token类型ID
                is not None
            ):
                token_type_id_requests[i] = token_types  # 记录该请求的token类型ID

        if len(token_type_id_requests) == 0:  # 如果没有请求需要token_type_ids
            return model_kwargs  # 直接返回空字典

        seq_lens = self.seq_lens.gpu[:num_reqs]  # 获取GPU上的序列长度
        token_type_ids = []  # 用于存储每个请求的token类型ID列表

        for i in range(num_reqs):  # 遍历每个请求
            pos = token_type_id_requests.get(i, seq_lens[i])  # 获取分界位置，默认为序列长度
            ids = (torch.arange(seq_lens[i]) >= pos).int()  # 生成token类型ID（分界位置之前为0，之后为1）
            token_type_ids.append(ids)  # 添加到列表

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(  # 拼接所有token类型ID并转移到GPU
            device=self.device
        )
        return model_kwargs  # 返回包含token_type_ids的模型参数字典

    # 根据注意力后端的需要重新排序批次中的请求（例如MLA需要分离计算密集型和内存密集型请求）
    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        根据注意力后端的需要重新排序批次中的请求。
        例如，某些注意力后端（如MLA）可能希望根据注意力计算是计算密集型还是
        内存密集型来分离请求。

        参数:
            scheduler_output: 调度器输出。

        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:  # 如果没有KV缓存组（无注意力模型）
            return  # 直接返回，无需重排序

        if self.reorder_batch_threshold is not None:  # 如果设置了重排序阈值
            reorder_batch_to_split_decodes_and_prefills(  # 将解码和预填充请求分开排列
                self.input_batch,  # 输入批次
                scheduler_output,  # 调度器输出
                decode_threshold=self.reorder_batch_threshold,  # 解码阈值
            )

    # 一次性预计算KV缓存清零所需的元数据
    def _init_kv_zero_meta(self) -> None:
        """
        一次性预计算KV缓存清零所需的元数据。
        委托给KVBlockZeroer.init_meta处理runner的状态。
        从gpu_worker.py在CuMem池上下文之外调用。

        One-time precomputation for _zero_block_ids.

        Delegates to KVBlockZeroer.init_meta with the runner's state.
        Called from gpu_worker.py outside the CuMem pool context.
        """
        self._kv_block_zeroer = KVBlockZeroer(self.device, self.pin_memory)  # 创建KV缓存块清零器
        self._kv_block_zeroer.init_meta(  # 初始化清零器的元数据
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),  # 注意力组迭代器
            kernel_block_sizes=self._kernel_block_sizes,  # 内核块大小
            cache_dtype=self.cache_config.cache_dtype,  # 缓存数据类型
            runner_only_attn_layers=self.runner_only_attn_layers,  # 仅runner使用的注意力层
            static_forward_context=(self.compilation_config.static_forward_context),  # 静态前向上下文
        )

    # 将指定block ID对应的KV缓存内存清零
    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """
        将指定block ID对应的KV缓存内存清零。
        Zero the KV cache memory for the given block IDs.
        """
        if hasattr(self, "_kv_block_zeroer"):  # 如果清零器已初始化
            self._kv_block_zeroer.zero_block_ids(block_ids)  # 执行清零操作

    # Note: used for model runner override.
    # 初始化设备属性（如SM数量）
    def _init_device_properties(self) -> None:
        """
        初始化设备属性，从torch.cuda.get_device_properties获取。
        主要获取GPU的计算单元（SM）数量。
        Initialize attributes from torch.cuda.get_device_properties
        """

        self.num_sms = num_compute_units(self.device.index)  # 获取GPU的计算单元（SM）数量

    # Note: used for model runner override.
    # 同步GPU设备，等待所有异步操作完成
    def _sync_device(self) -> None:
        """
        同步GPU设备，等待所有异步操作完成。
        用于确保所有GPU操作已执行完毕。
        """
        torch.accelerator.synchronize()  # 同步加速器设备

    # 根据调度器输出更新持久化批次和缓存状态。
    # 处理已完成请求的清理、新请求的添加、运行中请求的状态更新（block ID、token 等），
    # 投机解码 token 的同步，以及批次的压缩和重排序。
    # 这是每步推理开始时的关键状态同步步骤。
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """
        根据调度器输出更新缓存状态和持久化批次。
        更新后的状态被_prepare_inputs函数用于创建模型的输入GPU张量。
        如果批次中有新的/恢复的/暂停的/完成的请求，则更新SamplingMetadata并复制到GPU。

        Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:  # 遍历已完成的请求ID
            self.requests.pop(req_id, None)  # 从缓存状态中移除已完成请求
            self.num_prompt_logprobs.pop(req_id, None)  # 移除对应的prompt logprobs记录
        self.late_interaction_runner.on_requests_finished(  # 通知延迟交互运行器请求已完成
            scheduler_output.finished_req_ids
        )
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:  # 再次遍历已完成请求ID
            self.input_batch.remove_request(req_id)  # 从持久化批次中移除已完成请求

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:  # 如果有新分配的需要清零的缓存块
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)  # 将这些缓存块清零

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:  # 遍历需要释放的编码器缓存哈希
            self.encoder_cache.pop(mm_hash, None)  # 从编码器缓存中移除对应条目

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()  # 获取本次调度的请求ID集合
        cached_req_ids = self.input_batch.req_id_to_index.keys()  # 获取持久化批次中的请求ID集合
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids  # 获取从抢占恢复的请求ID集合
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)  # 计算未调度的请求ID集合
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:  # 遍历未调度的请求
            self.input_batch.remove_request(req_id)  # 从持久化批次中移除（但保留缓存状态）

        is_ngram_gpu = (  # 判断是否使用GPU上的n-gram投机解码
            self.speculative_config is not None  # 投机解码配置不为空
            and self.speculative_config.use_ngram_gpu()  # 且启用了GPU n-gram
        )
        if is_ngram_gpu:  # 如果使用GPU n-gram
            ngram_gpu_new_reqs: list[CachedRequestState] = []  # 初始化新请求列表，用于n-gram张量更新

        reqs_to_add: list[CachedRequestState] = []  # 待添加到持久化批次的请求列表
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:  # 遍历调度器分配的新请求
            req_id = new_req_data.req_id  # 获取请求ID
            if req_id in self.requests:  # 如果请求ID已存在于缓存中
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)  # 更新流式会话请求
                reqs_to_add.append(req_state)  # 添加到待加入列表
                continue  # 跳到下一个请求

            sampling_params = new_req_data.sampling_params  # 获取采样参数
            pooling_params = new_req_data.pooling_params  # 获取池化参数

            if (
                sampling_params  # 如果采样参数存在
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED  # 且采样类型为带种子的随机采样
            ):
                generator = torch.Generator(device=self.device)  # 在指定设备上创建随机数生成器
                generator.manual_seed(sampling_params.seed)  # 用给定种子初始化生成器
            else:
                generator = None  # 不需要自定义生成器

            if self.is_pooling_model:  # 如果是池化模型
                assert pooling_params is not None  # 确保池化参数存在
                task = pooling_params.task  # 获取池化任务类型
                assert task is not None, "You did not set `task` in the API"  # 确保设置了任务

                model = cast(VllmModelForPooling, self.get_model())  # 将模型转换为池化模型类型
                to_update = model.pooler.get_pooling_updates(task)  # 获取池化更新配置
                to_update.apply(pooling_params)  # 将更新应用到池化参数

            req_state = CachedRequestState(  # 创建新的缓存请求状态
                req_id=req_id,  # 请求ID
                prompt_token_ids=new_req_data.prompt_token_ids,  # 提示token ID列表
                prompt_embeds=new_req_data.prompt_embeds,  # 提示嵌入向量
                mm_features=new_req_data.mm_features,  # 多模态特征
                sampling_params=sampling_params,  # 采样参数
                pooling_params=pooling_params,  # 池化参数
                generator=generator,  # 随机数生成器
                block_ids=new_req_data.block_ids,  # KV缓存块ID列表
                num_computed_tokens=new_req_data.num_computed_tokens,  # 已计算的token数
                output_token_ids=[],  # 初始化输出token ID为空列表
                lora_request=new_req_data.lora_request,  # LoRA请求信息
            )
            self.requests[req_id] = req_state  # 将请求状态存入缓存字典
            self.late_interaction_runner.register_request(req_id, pooling_params)  # 向延迟交互运行器注册请求

            if sampling_params and sampling_params.prompt_logprobs is not None:  # 如果需要计算prompt logprobs
                self.num_prompt_logprobs[req_id] = (  # 记录该请求的prompt logprobs数量
                    self.input_batch.vocab_size  # 如果值为-1，使用完整词汇表大小
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs  # 否则使用指定值
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:  # 如果使用M-RoPE位置编码
                self._init_mrope_positions(req_state)  # 初始化M-RoPE位置

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:  # 如果使用XD-RoPE位置编码
                self._init_xdrope_positions(req_state)  # 初始化XD-RoPE位置

            reqs_to_add.append(req_state)  # 将新请求添加到待加入列表
            # Track new requests for ngram_gpu full tensor copy
            if is_ngram_gpu:  # 如果使用GPU n-gram
                ngram_gpu_new_reqs.append(req_state)  # 追踪新请求以进行n-gram张量复制

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank  # 判断当前是否为流水线并行的最后一个阶段
        req_data = scheduler_output.scheduled_cached_reqs  # 获取已缓存请求的调度数据
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens  # 获取调度的投机解码token

        # Save scheduler-allocated spec lengths before trimming so
        # prev_num_draft_len keeps the optimistic count for rejection correction.
        original_num_spec_per_req: dict[str, int] = {}  # 保存每个请求原始的投机token数量
        if (
            self.speculative_config is not None  # 如果启用了投机解码
            and self.speculative_config.use_ngram_gpu()  # 且使用GPU n-gram
        ):
            for req_id, toks in scheduled_spec_tokens.items():  # 遍历调度的投机token
                original_num_spec_per_req[req_id] = len(toks)  # 记录原始投机token长度
            update_scheduler_for_invalid_drafts(  # 根据无效草稿更新调度器
                self._num_valid_draft_tokens_event,  # 有效草稿token数量的CUDA事件
                self._num_valid_draft_tokens_cpu,  # 有效草稿token数量的CPU张量
                scheduler_output,  # 调度器输出
                self.input_batch.req_id_to_index,  # 请求ID到索引的映射
            )

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()  # 等待并获取有效采样token计数

        for i, req_id in enumerate(req_data.req_ids):  # 遍历已缓存的运行/恢复请求
            req_state = self.requests[req_id]  # 获取请求的缓存状态
            num_computed_tokens = req_data.num_computed_tokens[i]  # 获取已计算的token数
            new_block_ids = req_data.new_block_ids[i]  # 获取新分配的缓存块ID
            resumed_from_preemption = req_id in req_data.resumed_req_ids  # 判断是否从抢占中恢复
            num_output_tokens = req_data.num_output_tokens[i]  # 获取输出token数量
            req_index = self.input_batch.req_id_to_index.get(req_id)  # 获取请求在批次中的索引

            if req_state.prev_num_draft_len and self.use_async_scheduling:  # 如果有之前的草稿长度且使用异步调度
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # first step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step doesn't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:  # 如果请求不在当前批次中
                    req_state.prev_num_draft_len = 0  # 重置之前的草稿长度
                else:  # 请求在批次中
                    assert self.input_batch.prev_req_id_to_index is not None  # 确保上一步的索引映射存在
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]  # 获取上一步的请求索引
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1  # 计算被接受的草稿token数（减1因为包含了原始token）
                    num_rejected = req_state.prev_num_draft_len - num_accepted  # 计算被拒绝的草稿token数
                    num_computed_tokens -= num_rejected  # 从已计算token数中减去被拒绝的数量
                    req_state.output_token_ids.extend([-1] * num_accepted)  # 用占位符-1扩展输出token列表

                    if is_ngram_gpu and num_accepted > 0 and req_index is not None:  # 如果使用GPU n-gram且有被接受的token
                        self.input_batch.num_tokens_no_spec[req_index] += num_accepted  # 更新不含投机token的token计数

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens  # 更新缓存状态中的已计算token数

            if not is_last_rank:  # 如果不是流水线的最后阶段
                if not req_data.new_token_ids:  # 如果没有新的token ID
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []  # 异步调度的PP：采样token通过GPU广播传播，此处为空列表
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]  # 获取调度器发回的新token ID
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (  # 计算新增token的数量
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:  # 如果只有一个新token（最常见情况）
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])  # 直接追加最后一个token
                    elif num_new_tokens > 0:  # 如果有多个新token
                        req_state.output_token_ids.extend(  # 扩展输出token列表
                            new_token_ids[-num_new_tokens:]  # 取最后num_new_tokens个token
                        )
            elif num_output_tokens < len(req_state.output_token_ids):  # 如果输出token数少于缓存中的（sync-KV-load失败导致丢弃）
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]  # 截断输出token列表以对齐
                if req_index is not None:  # 如果请求在批次中
                    end_idx = (  # 计算结束索引
                        self.input_batch.num_prompt_tokens[req_index]  # 提示token数
                        + num_output_tokens  # 加上输出token数
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx  # 更新不含投机token的计数

            # Update the block IDs.
            if not resumed_from_preemption:  # 如果不是从抢占中恢复
                if new_block_ids is not None:  # 如果有新的缓存块ID
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):  # 遍历每个KV缓存组的块ID
                        block_ids.extend(new_ids)  # 将新块ID追加到现有块ID列表
            else:  # 从抢占中恢复的情况
                assert req_index is None  # 确保请求不在批次中（已被移除）
                assert new_block_ids is not None  # 确保有新的块ID
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids  # 用新的块ID完全替换旧的

            if req_index is None:  # 如果请求不在持久化批次中
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:  # 如果使用异步调度且有输出token
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]  # 获取恢复请求的所有token ID
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]  # 恢复输出token ID

                reqs_to_add.append(req_state)  # 将请求添加到待加入列表
                # Track resumed requests for ngram_gpu full tensor copy
                if is_ngram_gpu:  # 如果使用GPU n-gram
                    ngram_gpu_new_reqs.append(req_state)  # 追踪恢复的请求以进行n-gram张量复制
                continue  # 跳到下一个请求

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens  # 更新持久化批次中的已计算token数
            if new_block_ids is not None:  # 如果有新的缓存块ID
                self.input_batch.block_table.append_row(new_block_ids, req_index)  # 将新块追加到块表中

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:  # 如果不是最后阶段
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens  # 起始token索引
                end_token_index = num_computed_tokens + len(new_token_ids)  # 结束token索引
                self.input_batch.token_ids_cpu[  # 将新token ID写入CPU token缓存
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index  # 更新不含投机token的计数

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)  # 将投机解码token写入CPU token缓存
            # Restore scheduler-side draft count after ngram trimming.
            if original_num_spec_per_req:  # 如果保存了原始投机token数量
                orig = original_num_spec_per_req.get(req_id, 0)  # 获取原始投机token数
                if orig != req_state.prev_num_draft_len:  # 如果被修剪后不一致
                    req_state.prev_num_draft_len = orig  # 恢复为调度器端的原始草稿计数

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:  # 遍历所有待添加的新请求或恢复的请求
            self.input_batch.add_request(request)  # 将请求添加到持久化批次（优先填充较小的空索引）
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)  # 更新投机解码token

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()  # 压缩批次状态，消除移除请求留下的空隙
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)  # 允许注意力后端重排序批次
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()  # 刷新批次元数据

        # Incrementally update ngram_gpu tensors after batch is stable
        if is_ngram_gpu:  # 如果使用GPU n-gram
            update_ngram_gpu_tensors_incremental(  # 在批次稳定后增量更新n-gram GPU张量
                self.input_batch,  # 输入批次
                self.token_ids_gpu_tensor,  # GPU上的token ID张量
                self.num_tokens_no_spec_gpu,  # 不含投机token的GPU计数
                ngram_gpu_new_reqs,  # 新请求列表
                self.device,  # 目标设备
                _pinned_idx_buf=self._ngram_pinned_idx_buf,  # 锁页索引缓冲区
                _pinned_val_buf=self._ngram_pinned_val_buf,  # 锁页值缓冲区
            )

    # 模型执行后更新缓存状态（用于MTP/EAGLE混合模型的状态管理）
    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"  # 输出token ID张量和调度器输出
    ) -> None:
        """
        模型执行后更新缓存状态。
        用于MTP/EAGLE混合模型，因为在线性注意力中只保留最后一个token的状态。
        在MTP/EAGLE中，草稿token的状态会被保留直到确定每个序列接受了多少token，
        然后在下一次迭代中根据接受的token数进行移位操作。

        Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.speculative_config or not self.model_config.is_hybrid:  # 如果没有投机解码或不是混合模型
            return  # 直接返回

        # Find the number of accepted tokens for each sequence.
        num_reqs = output_token_ids.size(0)  # 获取请求数量（输出张量的第一维大小）
        self.num_accepted_tokens.gpu[:num_reqs] = (  # 计算每个序列被接受的token数
            (
                torch.cat(  # 在输出token后拼接一列-1作为哨兵值
                    [
                        output_token_ids,  # 原始输出token ID
                        torch.full(  # 创建全-1的哨兵列
                            (num_reqs, 1),  # 形状：(请求数, 1)
                            -1,  # 填充值-1
                            device=output_token_ids.device,  # 与输出相同设备
                        ),
                    ],
                    dim=1,  # 沿列方向拼接
                )
                == -1  # 找到第一个-1的位置（即无效token的位置）
            )
            .int()  # 转为整数
            .argmax(-1)  # 取每行第一个True的索引，即被接受的token数
        )
        if self.cache_config.mamba_cache_mode == "align":  # 如果Mamba缓存模式为对齐模式
            for i, num_tokens in enumerate(  # 遍历每个请求被接受的token数
                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()  # 从GPU复制到CPU
            ):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens  # 更新CPU端的接受token计数

            mamba_utils.postprocess_mamba(  # 对Mamba状态进行后处理
                scheduler_output,  # 调度器输出
                self.kv_cache_config,  # KV缓存配置
                self.input_batch,  # 输入批次
                self.requests,  # 请求缓存
                self.mamba_state_idx,  # Mamba状态索引
                self.compilation_config.static_forward_context,  # 静态前向上下文
                self.model.get_mamba_state_copy_func(),  # Mamba状态复制函数
                self._get_mamba_copy_bufs(),  # Mamba复制缓冲区
            )
        else:  # 非对齐模式
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(  # 将GPU上的接受token数异步复制到CPU张量
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None  # 确保CUDA事件已创建
            self.num_accepted_tokens_event.record()  # 记录CUDA事件用于后续同步

    # 更新流式会话请求的状态
    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData  # 请求ID和新请求数据
    ) -> CachedRequestState:
        """
        从scheduled_new_reqs更新流式会话请求。
        从InputBatch中移除请求（如果存在），更新缓存状态，并准备将其重新添加到批次中。
        注意：prompt_token_ids包含中间输出token——即之前生成但现在作为输入上下文（提示的一部分）的token。

        Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        self.input_batch.remove_request(req_id)  # 从输入批次中移除请求（如果存在）
        req_state = self.requests[req_id]  # 获取请求的缓存状态

        req_state.prompt_token_ids = new_req_data.prompt_token_ids  # 更新提示token ID（包含之前生成的中间输出token）
        req_state.mm_features = new_req_data.mm_features  # 更新多模态特征
        req_state.prompt_embeds = new_req_data.prompt_embeds  # 更新提示嵌入向量
        req_state.sampling_params = new_req_data.sampling_params  # 更新采样参数
        req_state.pooling_params = new_req_data.pooling_params  # 更新池化参数
        self.late_interaction_runner.register_request(req_id, req_state.pooling_params)  # 重新注册延迟交互
        req_state.block_ids = new_req_data.block_ids  # 更新KV缓存块ID
        req_state.num_computed_tokens = new_req_data.num_computed_tokens  # 更新已计算token数
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(  # 更新提示token数量
            req_state.prompt_token_ids, req_state.prompt_embeds  # 从token ID或嵌入计算长度
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()  # 清空输出token列表（之前的输出现在已成为提示的一部分）

        if self.uses_mrope:  # 如果使用M-RoPE
            self._init_mrope_positions(req_state)  # 重新初始化M-RoPE位置编码

        return req_state  # 返回更新后的请求状态

    # 初始化M-RoPE（多维旋转位置编码）的位置信息
    def _init_mrope_positions(self, req_state: CachedRequestState):
        """
        初始化M-RoPE（多维旋转位置编码）的位置信息。
        通过模型的get_mrope_input_positions方法计算位置和偏移量。
        """
        model = self.get_model()  # 获取模型实例
        assert supports_mrope(model), "M-RoPE support is not implemented."  # 确保模型支持M-RoPE
        assert req_state.prompt_token_ids is not None, (  # 确保提示token ID存在
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)  # 将模型转换为支持M-RoPE的类型

        req_state.mrope_positions, req_state.mrope_position_delta = (  # 计算并存储M-RoPE位置和位置偏移量
            mrope_model.get_mrope_input_positions(  # 调用模型方法计算M-RoPE输入位置
                req_state.prompt_token_ids,  # 提示token ID
                req_state.mm_features,  # 多模态特征
            )
        )

    # 初始化XD-RoPE（跨维旋转位置编码）的位置信息
    def _init_xdrope_positions(self, req_state: CachedRequestState):
        """
        初始化XD-RoPE（跨维旋转位置编码）的位置信息。
        通过模型的get_xdrope_input_positions方法计算位置。
        """
        model = self.get_model()  # 获取模型实例
        xdrope_model = cast(SupportsXDRoPE, model)  # 将模型转换为支持XD-RoPE的类型
        assert req_state.prompt_token_ids is not None, (  # 确保提示token ID存在
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."  # 确保模型支持XD-RoPE

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(  # 计算并存储XD-RoPE位置
            req_state.prompt_token_ids,  # 提示token ID
            req_state.mm_features,  # 多模态特征
        )

    # 从调度器输出中提取多模态关键字参数（用于原始输入的多模态模型）
    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出
    ) -> BatchedTensorInputs:
        """
        从调度器输出中提取多模态关键字参数。
        仅用于原始输入的多模态模型，将各模态的特征数据分组并批处理。
        """
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:  # 如果调度器输出为空或非原始输入多模态模型
            return {}  # 返回空字典

        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()  # 创建多模态参数列表
        for req in scheduler_output.scheduled_new_reqs:  # 遍历调度的新请求
            for feature in req.mm_features:  # 遍历每个请求的多模态特征
                if feature.data is not None:  # 如果特征数据存在
                    mm_kwargs.append((feature.modality, feature.data))  # 添加模态名和数据的元组

        # Input all modalities at once
        mm_kwargs_combined: BatchedTensorInputs = {}  # 创建合并后的多模态参数字典
        for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(  # 按模态分组并批处理多模态参数
            mm_kwargs,  # 多模态参数列表
            device=self.device,  # 目标设备
            pin_memory=self.pin_memory,  # 是否锁页内存
        ):
            mm_kwargs_combined.update(mm_kwargs_batch)  # 合并到结果字典中

        return mm_kwargs_combined  # 返回合并后的多模态参数

    # 创建多模态模型的虚拟批次参数（用于性能分析和CUDA图捕获）
    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        """
        创建多模态模型的虚拟批次参数。
        用于性能分析和CUDA图捕获时生成虚拟输入。
        """
        if not self.is_multimodal_raw_input_only_model:  # 如果不是原始输入多模态模型
            return {}  # 返回空字典

        mm_budget = self.mm_budget  # 获取多模态预算
        assert mm_budget is not None  # 确保预算存在

        if not mm_budget.mm_max_toks_per_item:  # 如果没有每项最大token数（仅嵌入模式）
            return {}  # No tower modalities (embed-only mode) # 无塔模态（仅嵌入模式）

        dummy_modality = mm_budget.get_modality_with_max_tokens()  # 获取占用最多token的模态
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)  # 返回该模态的虚拟批次

    # 计算给定数组的累积和以及批量arange（用于高效构建token索引）
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,  # 每个请求的token数量数组
        cumsum_dtype: np.dtype | None = None,  # 累积和的数据类型
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定数组的累积和和批量arange。
        例如：[2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        等价于但快于：np.concatenate([np.arange(n) for n in num_tokens])

        Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)  # 计算累积和
        total_num_tokens = cu_num_tokens[-1]  # 获取总token数
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)  # 将每个请求的起始偏移重复对应的token次数
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets  # 用预分配的arange减去偏移得到每个请求内的局部索引

        return cu_num_tokens, arange  # 返回累积和与批量arange

    # 准备 input_ids GPU 张量。
    # 常规模式直接从 CPU 拷贝；异步调度模式下需要特殊处理：
    # 上一步的采样结果可能还在 GPU 上，需用 scatter 操作将其写入正确位置，
    # 同时处理投机解码的 draft token 拷贝。
    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出
        total_num_scheduled_tokens: int,  # 本次调度的总token数
        cu_num_tokens: np.ndarray,  # 累积token数数组
    ) -> None:
        """
        为当前批次准备输入ID。
        仔细处理prev_sampled_token_ids，这些token可能从上一次引擎迭代中缓存在GPU上，
        需要被复制到input_ids的对应位置中。

        Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.
        """

        if self.input_batch.prev_sampled_token_ids is None:  # 如果没有上一步缓存的采样token（常规调度情况）
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)  # 将input_ids从CPU拷贝到GPU
            if self.enable_prompt_embeds:  # 如果启用了提示嵌入
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)  # 将嵌入拷贝到GPU
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)  # 将token ID标记拷贝到GPU
            return  # 返回

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index  # 获取上一步的请求ID到索引映射
        assert prev_req_id_to_index is not None  # 确保映射存在
        sample_flattened_indices: list[int] = []  # 采样token在展平input_ids中的目标索引
        spec_flattened_indices: list[int] = []  # 投机token在展平input_ids中的目标索引
        prev_common_req_indices: list[int] = []  # 与上一步共有请求在上一步中的索引
        prev_draft_token_indices: list[int] = []  # 上一步草稿token在展平草稿张量中的索引
        indices_match = True  # 标记索引是否完全匹配（用于优化路径）
        max_flattened_index = -1  # 记录最大展平索引
        total_num_spec_tokens = 0  # 投机token总数
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens  # 获取调度的投机token

        for req_id, cur_index in self.input_batch.req_id_to_index.items():  # 遍历当前批次中的每个请求
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:  # 如果该请求在上一步中也存在
                prev_common_req_indices.append(prev_index)  # 记录上一步的索引
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))  # 获取该请求的草稿token长度
                total_num_spec_tokens += draft_len  # 累加投机token总数
                flattened_index = cu_num_tokens[cur_index].item() - 1  # 计算该请求最后一个token的展平索引
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)  # 采样token位置 = 最后位置 - 草稿长度
                spec_flattened_indices.extend(  # 投机token的目标位置范围
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens  # 计算上一步草稿token的起始索引
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))  # 记录要复制的草稿token索引
                indices_match &= prev_index == flattened_index  # 检查索引是否一一匹配
                max_flattened_index = max(max_flattened_index, flattened_index)  # 更新最大展平索引
        num_common_tokens = len(sample_flattened_indices)  # 计算共有请求的采样token数
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens  # 不含投机token的总调度数
        if num_common_tokens < total_without_spec:  # 如果不是所有请求都是上一步的解码请求
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)  # 先将CPU input_ids拷贝到GPU
            if self.enable_prompt_embeds:  # 如果启用了提示嵌入
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)  # 拷贝嵌入到GPU
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)  # 拷贝token ID标记到GPU
        if num_common_tokens == 0:  # 如果没有与上一步共有的请求
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return  # CPU已包含所有input_ids，直接返回
        if indices_match and max_flattened_index == (num_common_tokens - 1):  # 如果索引完全匹配（常见优化路径）
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(  # 直接用切片拷贝上一步的采样token
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,  # 异步拷贝
            )
            if self.enable_prompt_embeds:  # 如果启用了提示嵌入
                self.is_token_ids.gpu[:num_common_tokens] = True  # 标记为token ID
            return  # 返回
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(  # 创建采样token的目标索引张量
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)  # 异步传输到GPU
        prev_common_req_indices_tensor = torch.tensor(  # 创建上一步共有请求的索引张量
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)  # 异步传输到GPU
        self.input_ids.gpu.scatter_(  # 使用scatter操作将采样token写入input_ids的正确位置
            dim=0,  # 沿第0维散射
            index=sampled_tokens_index_tensor,  # 目标索引
            src=self.input_batch.prev_sampled_token_ids[  # 源数据：上一步的采样token
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:  # 如果没有草稿token或无需散射的索引
            return  # 直接返回

        assert isinstance(self._draft_token_ids, torch.Tensor)  # 确保草稿token ID是张量类型
        draft_tokens_index_tensor = torch.tensor(  # 创建草稿token的目标索引张量
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)  # 异步传输到GPU
        prev_draft_token_indices_tensor = torch.tensor(  # 创建上一步草稿token的源索引张量
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)  # 异步传输到GPU

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)  # 将草稿token ID转换为int32以匹配input_ids

        self.input_ids.gpu.scatter_(  # 使用scatter操作将草稿token写入input_ids的正确位置
            dim=0,  # 沿第0维散射
            index=draft_tokens_index_tensor,  # 目标索引
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],  # 源数据：展平后按索引取出的草稿token
        )

    # 获取编码器序列长度（用于交叉注意力模型，如编码器-解码器架构）
    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],  # 每个请求的调度token数
        kv_cache_spec: KVCacheSpec,  # KV缓存规格
        num_reqs: int,  # 请求数量
        for_cudagraph_capture: bool = False,  # 是否用于CUDA图捕获
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        """
        获取编码器序列长度。
        用于交叉注意力模型（如编码器-解码器架构），构建请求索引到编码器长度的映射。
        """
        if not isinstance(kv_cache_spec, CrossAttentionSpec):  # 如果不是交叉注意力规格
            return None, None  # 返回None

        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0  # 将编码器序列长度缓冲区清零

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:  # 遍历本批次调度的请求
            req_index = self.input_batch.req_id_to_index[req_id]  # 获取请求在批次中的索引
            req_state = self.requests[req_id]  # 获取请求的缓存状态
            if req_state.mm_features is None:  # 如果没有多模态特征
                self.encoder_seq_lens.np[req_index] = 0  # 编码器序列长度设为0
                continue  # 跳到下一个请求
            encoder_input_tokens = sum(  # 计算编码器输入token的总数
                feature.mm_position.length for feature in req_state.mm_features  # 累加每个多模态特征的长度
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens  # 设置编码器序列长度
        if for_cudagraph_capture:  # 如果用于CUDA图捕获
            # During CUDA graph capture, we need to use realistic encoder lengths  # 在CUDA图捕获期间，需要使用真实的编码器长度
            # so that max_seqlen_k is captured with the correct value.  # 以确保max_seqlen_k以正确的值被捕获
            max_encoder_len = getattr(  # 获取最大编码器长度
                self.model_config.hf_config,  # 从HuggingFace配置中获取
                "max_source_positions",  # 最大源位置属性
                self.max_encoder_len,  # 默认值为self.max_encoder_len
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len  # 所有请求使用最大编码器长度

        self.encoder_seq_lens.copy_to_gpu(num_reqs)  # 将编码器序列长度拷贝到GPU
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]  # 获取GPU上的编码器序列长度切片
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]  # 获取CPU上的编码器序列长度切片

        return encoder_seq_lens, encoder_seq_lens_cpu  # 返回GPU和CPU上的编码器序列长度

    # 准备模型前向传播所需的 GPU 输入张量。
    # 核心步骤：计算位置编码、构建 token 索引映射、拷贝 input_ids 到 GPU、
    # 提交 block table 和 slot mapping、构建 query_start_loc 和 seq_lens、
    # 处理 M-RoPE/XD-RoPE 位置、激活 LoRA 适配器。
    # 对于投机解码，还计算 draft token 的 logits 索引。
    def _prepare_inputs(  # 定义准备输入的方法
        self,  # 实例自身引用
        scheduler_output: "SchedulerOutput",  # 调度器输出对象
        num_scheduled_tokens: np.ndarray,  # 每个请求被调度的token数量数组
    ) -> tuple[  # 返回一个元组
        torch.Tensor,  # logits索引张量
        SpecDecodeMetadata | None,  # 投机解码元数据（可选）
    ]:
        """
        准备模型前向传播所需的输入数据，包括位置编码、token索引、block table等。
        :return: tuple[
            logits_indices, spec_decode_metadata,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取总调度token数量
        assert total_num_scheduled_tokens > 0  # 断言总调度token数大于0
        num_reqs = self.input_batch.num_reqs  # 获取当前批次中的请求数量
        assert num_reqs > 0  # 断言请求数大于0

        # OPTIMIZATION: Start copying the block table first.  # 优化：首先开始拷贝block table
        # This way, we can overlap the copy with the following CPU operations.  # 这样可以将拷贝与后续CPU操作重叠
        self.input_batch.block_table.commit_block_table(num_reqs)  # 提交block table到GPU

        # Get request indices.  # 获取请求索引
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]  # 例如将每个请求的token数展开为请求索引
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)  # 根据每个请求的调度token数重复请求索引

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]  # 累积和示例
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]  # 每个请求内的局部递增索引
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)  # 计算累积和与局部递增索引

        # Get positions.  # 获取位置编码
        positions_np = self.positions.np[:total_num_scheduled_tokens]  # 获取位置数组的切片
        np.add(  # 执行numpy加法运算
            self.input_batch.num_computed_tokens_cpu[req_indices],  # 每个token对应请求的已计算token数
            arange,  # 加上局部递增索引
            out=positions_np,  # 结果存入positions_np
        )

        # Calculate M-RoPE positions.  # 计算M-RoPE位置编码
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)  # 仅适用于使用M-RoPE的模型（如Qwen2-VL）
        if self.uses_mrope:  # 如果模型使用M-RoPE
            self._calc_mrope_positions(scheduler_output)  # 计算M-RoPE位置

        # Calculate XD-RoPE positions.  # 计算XD-RoPE位置编码
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)  # 仅适用于使用XD-RoPE的模型（如HunYuan-VL）
        if self.uses_xdrope_dim > 0:  # 如果模型使用XD-RoPE
            self._calc_xdrope_positions(scheduler_output)  # 计算XD-RoPE位置

        # Get token indices.  # 获取token索引
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]  # 局部位置
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]  # 映射到全局token_ids数组中的索引
        # where M is the max_model_len.  # 其中M是最大模型长度
        token_indices = (  # 计算token在二维token_ids数组中的一维索引
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]  # 位置 + 请求索引 * 每行长度
        )
        token_indices_tensor = torch.from_numpy(token_indices)  # 将numpy数组转为PyTorch张量

        # NOTE(woosuk): We use torch.index_select instead of np.take here  # 注意：这里使用torch.index_select而不是np.take
        # because torch.index_select is much faster than np.take for large  # 因为对大张量torch.index_select比np.take快得多
        # tensors.
        torch.index_select(  # 使用索引选择从展平的token_ids中提取input_ids
            self.input_batch.token_ids_cpu_tensor.flatten(),  # 展平的token_ids CPU张量
            0,  # 在第0维上选择
            token_indices_tensor,  # 选择的索引
            out=self.input_ids.cpu[:total_num_scheduled_tokens],  # 输出到input_ids的CPU缓冲区
        )
        if self.enable_prompt_embeds:  # 如果启用了提示嵌入
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()  # 展平的is_token_ids标记张量
            torch.index_select(  # 使用索引选择对应的is_token_ids标记
                is_token_ids,  # 输入张量
                0,  # 在第0维上选择
                token_indices_tensor,  # 选择的索引
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],  # 输出到is_token_ids的CPU缓冲区
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on  # 因为没有在InputBatch上预分配大的prompt_embeds CPU张量
        # the InputBatch, we need to fill in the prompt embeds into the expected  # 所以需要将prompt嵌入填入GpuModelRunner预分配的prompt_embeds张量的对应位置
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:  # 如果有请求的提示嵌入数据
            output_idx = 0  # 初始化输出索引
            for req_idx in range(num_reqs):  # 遍历每个请求
                num_sched = num_scheduled_tokens[req_idx]  # 获取该请求调度的token数

                # Skip if this request doesn't have embeddings  # 如果该请求没有嵌入则跳过
                if req_idx not in self.input_batch.req_prompt_embeds:  # 判断请求是否有prompt嵌入
                    output_idx += num_sched  # 跳过对应数量的输出位置
                    continue  # 继续下一个请求

                # Skip if no tokens scheduled  # 如果没有调度的token则跳过
                if num_sched <= 0:  # 判断调度的token数是否为正
                    output_idx += num_sched  # 跳过对应数量的输出位置
                    continue  # 继续下一个请求

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]  # 获取该请求的提示嵌入数据
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]  # 获取该请求已计算的token数作为起始位置

                # Skip if trying to read beyond available embeddings  # 如果起始位置超出可用嵌入范围则跳过
                if start_pos >= req_embeds.shape[0]:  # 判断起始位置是否超出嵌入长度
                    output_idx += num_sched  # 跳过对应数量的输出位置
                    continue  # 继续下一个请求

                # Copy available embeddings  # 拷贝可用的嵌入
                end_pos = start_pos + num_sched  # 计算结束位置
                actual_end = min(end_pos, req_embeds.shape[0])  # 实际结束位置不超过嵌入长度
                actual_num_sched = actual_end - start_pos  # 实际需要拷贝的token数

                if actual_num_sched > 0:  # 如果有需要拷贝的数据
                    self.inputs_embeds.cpu[  # 目标：GpuModelRunner的inputs_embeds CPU缓冲区
                        output_idx : output_idx + actual_num_sched  # 输出范围
                    ].copy_(req_embeds[start_pos:actual_end])  # 从请求嵌入中拷贝数据

                output_idx += num_sched  # 更新输出索引

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)  # 计算slot映射，将位置映射到KV cache的slot
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)  # 提交slot映射到GPU

        # Prepare the attention metadata.  # 准备注意力元数据
        self.query_start_loc.np[0] = 0  # query起始位置数组的第一个元素设为0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens  # 设置每个请求的query累积起始位置
        # Note: pad query_start_loc to be non-decreasing, as kernels  # 注意：填充query_start_loc为非递减序列
        # like FlashAttention requires that  # 因为FlashAttention等kernel要求如此
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])  # 用最后一个累积值填充剩余位置
        self.query_start_loc.copy_to_gpu()  # 将query_start_loc拷贝到GPU
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]  # 获取GPU上的query_start_loc切片

        self.seq_lens.np[:num_reqs] = (  # 计算每个请求的序列长度
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens  # 已计算token数 + 本次调度token数
        )
        # Fill unused with 0 for full cuda graph mode.  # 为full CUDA图模式将未使用位置填0
        self.seq_lens.np[num_reqs:].fill(0)  # 将超出请求数的位置填0
        self.seq_lens.copy_to_gpu()  # 将seq_lens拷贝到GPU

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]  # 获取每个请求的总token数列表
        num_tokens_np = np.array(num_tokens, dtype=np.int32)  # 转为numpy数组

        # Record which requests should not be sampled,  # 记录哪些请求不应进行采样
        # so that we could clear the sampled tokens before returning  # 以便在返回前清除已采样的token
        self.discard_request_mask.np[:num_reqs] = (  # 设置丢弃请求掩码
            self.seq_lens.np[:num_reqs] < num_tokens_np  # 当前序列长度小于总token数的请求需要丢弃
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)  # 将丢弃掩码拷贝到GPU

        # Copy the tensors to the GPU.  # 将张量拷贝到GPU
        self._prepare_input_ids(  # 调用准备input_ids的方法
            scheduler_output,  # 调度器输出
            total_num_scheduled_tokens,  # 总调度token数
            cu_num_tokens,  # 累积token数
        )

        if self.uses_mrope:  # 如果使用M-RoPE
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)  # 仅适用于M-RoPE模型
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(  # 将M-RoPE位置从CPU拷贝到GPU
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],  # 源：CPU上的M-RoPE位置
                non_blocking=True,  # 异步拷贝
            )
        elif self.uses_xdrope_dim > 0:  # 如果使用XD-RoPE
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)  # 仅适用于XD-RoPE模型
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(  # 将XD-RoPE位置从CPU拷贝到GPU
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],  # 源：CPU上的XD-RoPE位置
                non_blocking=True,  # 异步拷贝
            )
        else:  # 否则使用普通1D位置编码
            # Common case (1D positions)  # 通常情况（一维位置编码）
            self.positions.copy_to_gpu(total_num_scheduled_tokens)  # 将一维位置编码拷贝到GPU

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0  # 判断是否使用投机解码
        if not use_spec_decode:  # 如果不使用投机解码
            # NOTE(woosuk): Due to chunked prefills, the batch may contain  # 注意：由于分块预填充，批次可能包含
            # partial requests. While we should not sample any token  # 部分请求。虽然不应从这些部分请求中采样token
            # from these partial requests, we do so for simplicity.  # 但为简便起见仍进行采样
            # We will ignore the sampled tokens from the partial requests.  # 后续会忽略部分请求的采样结果
            # TODO: Support prompt logprobs.  # TODO: 支持prompt的logprobs
            logits_indices = query_start_loc[1:] - 1  # logits索引为每个请求最后一个token的位置
            spec_decode_metadata = None  # 不使用投机解码元数据
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)  # 每个请求采样1个token
        else:  # 如果使用投机解码
            # Get the number of draft tokens for each request.  # 获取每个请求的draft token数量
            # Iterate over the dictionary rather than all requests since not all  # 遍历字典而非所有请求，因为不是所有请求都有draft token
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)  # 初始化draft token数数组为0
            # For chunked prefills, use -1 as mask rather than 0, as guided  # 对分块预填充使用-1作为掩码而非0
            # decoding may rollback speculative tokens.  # 因为引导解码可能会回滚投机token
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)  # 初始化decode draft token数数组为-1
            for (  # 遍历调度的投机解码token
                req_id,  # 请求ID
                draft_token_ids,  # draft token的ID列表
            ) in scheduler_output.scheduled_spec_decode_tokens.items():  # 从调度输出的投机解码token字典中遍历
                req_idx = self.input_batch.req_id_to_index[req_id]  # 获取请求在批次中的索引
                num_draft_tokens[req_idx] = len(draft_token_ids)  # 设置该请求的draft token数量
                if (  # 如果该请求已完成预填充阶段
                    self.input_batch.num_computed_tokens_cpu[req_idx]  # 已计算的token数
                    >= self.input_batch.num_prompt_tokens[req_idx]  # 大于等于prompt token数
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)  # 设置decode阶段的draft token数
            spec_decode_metadata = self._calc_spec_decode_metadata(  # 计算投机解码元数据
                num_draft_tokens, cu_num_tokens  # 传入draft token数和累积token数
            )
            logits_indices = spec_decode_metadata.logits_indices  # 获取logits索引
            num_sampled_tokens = num_draft_tokens + 1  # 采样的token数 = draft token数 + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).  # 用于某些注意力后端（如GDN）的decode-only CUDA图
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens  # 设置decode draft token数
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)  # 超出请求数的位置填-1
            self.num_decode_draft_tokens.copy_to_gpu()  # 拷贝到GPU

        # Hot-Swap lora model  # 热交换LoRA模型
        if self.lora_config:  # 如果配置了LoRA
            assert (  # 断言采样token总数不超过最大批处理token数
                np.sum(num_sampled_tokens)  # 所有请求的采样token数之和
                <= self.vllm_config.scheduler_config.max_num_batched_tokens  # 最大批处理token数限制
            )
            self.set_active_loras(  # 设置当前活跃的LoRA适配器
                self.input_batch, num_scheduled_tokens, num_sampled_tokens  # 传入批次信息和token数信息
            )

        return (  # 返回结果
            logits_indices,  # logits索引张量
            spec_decode_metadata,  # 投机解码元数据
        )

    # 为每个 KV 缓存组和注意力组构建注意力元数据。
    # 支持混合架构（Transformer + Mamba）、级联注意力、编码器-解码器交叉注意力、
    # 微批次切分以及 CUDA Graph 捕获模式。
    # 通过缓存构建结果并仅更新 block table 来避免重复计算。
    def _build_attention_metadata(  # 定义构建注意力元数据的方法
        self,  # 实例自身引用
        num_tokens: int,  # token总数
        num_reqs: int,  # 请求总数
        max_query_len: int,  # 最大查询长度
        num_tokens_padded: int | None = None,  # 填充后的token数（可选）
        num_reqs_padded: int | None = None,  # 填充后的请求数（可选）
        ubatch_slices: UBatchSlices | None = None,  # 微批次切片（可选）
        logits_indices: torch.Tensor | None = None,  # logits索引（可选）
        use_spec_decode: bool = False,  # 是否使用投机解码
        for_cudagraph_capture: bool = False,  # 是否用于CUDA图捕获
        num_scheduled_tokens: dict[str, int] | None = None,  # 调度的token数字典（可选）
        cascade_attn_prefix_lens: list[list[int]] | None = None,  # 级联注意力前缀长度（可选）
        slot_mappings: dict[int, torch.Tensor] | None = None,  # slot映射字典（可选）
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:  # 返回类型：每层注意力元数据和公共注意力元数据
        """
        构建每一层的注意力元数据，支持混合架构、级联注意力和CUDA图捕获模式。
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models  # 无注意力模型不需要注意力元数据
        if len(self.kv_cache_config.kv_cache_groups) == 0:  # 如果没有KV缓存组
            return {}, None  # 返回空字典和None

        num_tokens_padded = num_tokens_padded or num_tokens  # 如果未指定填充token数则使用原始值
        num_reqs_padded = num_reqs_padded or num_reqs  # 如果未指定填充请求数则使用原始值
        assert num_reqs_padded is not None and num_tokens_padded is not None  # 断言填充值不为None

        attn_metadata: PerLayerAttnMetadata = {}  # 初始化每层注意力元数据字典
        if ubatch_slices is not None:  # 如果有微批次切片
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]  # 为每个微批次创建一个空字典

        if for_cudagraph_capture:  # 如果用于CUDA图捕获
            # For some attention backends (e.g. FA) with sliding window models we need  # 对于某些带滑动窗口模型的注意力后端
            # to make sure the backend see a max_seq_len that is larger to the sliding  # 需要确保后端看到的max_seq_len大于滑动窗口大小
            # window size when capturing to make sure the correct kernel is selected.  # 以确保捕获时选择正确的kernel
            max_seq_len = self.max_model_len  # 使用最大模型长度
        else:  # 否则
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()  # 使用当前批次中的最大序列长度

        if use_spec_decode:  # 如果使用投机解码
            if self.num_accepted_tokens_event is not None:  # 如果有已接受token数的事件
                self.num_accepted_tokens_event.synchronize()  # 同步等待事件完成
            self.num_accepted_tokens.np[:num_reqs] = (  # 设置已接受token数
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]  # 从输入批次的CPU数据中获取
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)  # 超出请求数的位置填1
            self.num_accepted_tokens.copy_to_gpu()  # 拷贝到GPU

        kv_cache_groups = self.kv_cache_config.kv_cache_groups  # 获取KV缓存组配置

        def _get_block_table(kv_cache_gid: int):  # 定义获取block table的内部函数
            """获取指定KV缓存组的block table张量。"""
            assert num_reqs_padded is not None and num_tokens_padded is not None  # 断言填充值不为None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec  # 获取KV缓存规格
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):  # 如果是仅编码器注意力规格
                blk_table_tensor = torch.zeros(  # 创建全零的block table张量
                    (num_reqs_padded, 1),  # 形状为(填充请求数, 1)
                    dtype=torch.int32,  # 数据类型为int32
                    device=self.device,  # 放在当前设备上
                )
            else:  # 否则
                blk_table = self.input_batch.block_table[kv_cache_gid]  # 获取对应KV缓存组的block table
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)  # 获取设备上的block table张量

            # Fill unused with -1. Needed for reshape_and_cache in full cuda  # 将未使用位置填-1，full CUDA图模式下reshape_and_cache需要
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID  # block table中-1与mamba的PAD_SLOT_ID匹配
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)  # 将超出实际请求数的位置填-1
            return blk_table_tensor  # 返回block table张量

        assert slot_mappings is not None  # 断言slot映射不为None
        block_table_gid_0 = _get_block_table(0)  # 获取第0个KV缓存组的block table
        slot_mapping_gid_0 = slot_mappings[0]  # 获取第0个KV缓存组的slot映射

        if self.routed_experts_initialized:  # 如果路由专家已初始化
            attn_gid = self.routed_experts_attn_gid  # 获取路由专家的注意力组ID
            slot_mapping_attn = slot_mappings[attn_gid]  # 获取对应的slot映射
            self.slot_mapping = slot_mapping_attn[:num_tokens].cpu().numpy()  # 将slot映射转为CPU numpy数组
        cm_base = CommonAttentionMetadata(  # 创建公共注意力元数据基础对象
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],  # 查询起始位置（GPU）
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],  # 查询起始位置（CPU）
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],  # 序列长度（GPU）
            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],  # 序列长度（CPU）
            _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[  # 已计算token数（CPU张量）
                :num_reqs_padded  # 截取到填充请求数
            ],
            num_reqs=num_reqs_padded,  # 填充后的请求数
            num_actual_tokens=num_tokens_padded,  # 填充后的实际token数
            max_query_len=max_query_len,  # 最大查询长度
            max_seq_len=max_seq_len,  # 最大序列长度
            block_table_tensor=block_table_gid_0,  # block table张量
            slot_mapping=slot_mapping_gid_0,  # slot映射张量
            causal=True,  # 使用因果注意力
        )

        if self.dcp_world_size > 1:  # 如果数据中心并行世界大小大于1
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(  # 计算DCP本地序列长度
                self.seq_lens.cpu[:num_reqs],  # 序列长度
                self.dcp_world_size,  # DCP世界大小
                self.dcp_rank,  # DCP排名
                self.parallel_config.cp_kv_cache_interleave_size,  # KV缓存交错大小
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)  # 超出请求数的位置填0
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)  # 拷贝到GPU

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]  # 设置DCP本地序列长度（GPU）
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[  # 设置DCP本地序列长度（CPU）
                :num_reqs_padded  # 截取到填充请求数
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:  # 如果有logits索引且启用KV共享快速预填充
            cm_base.num_logits_indices = logits_indices.size(0)  # 设置logits索引数量
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(  # 准备KV共享快速预填充的logits索引
                logits_indices  # 传入logits索引
            )

        # Cache attention metadata builds across hybrid KV-cache groups  # 在混合KV缓存组之间缓存注意力元数据构建结果
        # The only thing that changes between different hybrid KV-cache groups when the  # 当使用相同的元数据构建器和KVCacheSpec时
        # same metadata builder and KVCacheSpec is the same is the block table, so we  # 不同混合KV缓存组之间唯一变化的是block table
        # can cache the attention metadata builds and just update the block table using  # 所以可以缓存构建结果并使用update_block_table更新
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[  # 定义缓存的注意力元数据字典
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata  # 键为(KVCacheSpec, 构建器类型)，值为注意力元数据
        ] = {}

        def _build_attn_group_metadata(  # 定义构建注意力组元数据的内部函数
            kv_cache_gid: int,  # KV缓存组ID
            attn_gid: int,  # 注意力组ID
            common_attn_metadata: CommonAttentionMetadata,  # 公共注意力元数据
            ubid: int | None = None,  # 微批次ID（可选）
        ) -> None:  # 无返回值
            """为指定的KV缓存组和注意力组构建注意力元数据。"""
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]  # 获取注意力组对象
            builder = attn_group.get_metadata_builder(ubid or 0)  # 获取元数据构建器
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec  # 获取KV缓存规格
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):  # 如果是统一类型KV缓存规格
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]  # 获取第一层的具体规格
            cache_key = (kv_cache_spec, type(builder))  # 构建缓存键

            cascade_attn_prefix_len = (  # 获取级联注意力前缀长度
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]  # 从级联注意力前缀长度数组中获取
                if cascade_attn_prefix_lens  # 如果提供了级联注意力前缀长度
                else 0  # 否则为0
            )

            extra_attn_metadata_args = {}  # 初始化额外的注意力元数据参数
            if use_spec_decode and isinstance(  # 如果使用投机解码且构建器是Mamba2或GDN类型
                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)  # 判断构建器类型
            ):
                assert ubid is None, "UBatching not supported with GDN yet"  # 断言微批次不支持GDN
                extra_attn_metadata_args = dict(  # 设置额外参数
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],  # 已接受token数（GPU）
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[  # decode draft token数（CPU）
                        :num_reqs_padded  # 截取到填充请求数
                    ],
                )

            if for_cudagraph_capture:  # 如果用于CUDA图捕获
                attn_metadata_i = builder.build_for_cudagraph_capture(  # 使用CUDA图捕获模式构建元数据
                    common_attn_metadata  # 传入公共注意力元数据
                )
            elif (  # 否则如果缓存中已有且构建器支持更新block table
                cache_key in cached_attn_metadata  # 检查缓存键是否存在
                and builder.supports_update_block_table  # 且构建器支持更新block table
            ):
                attn_metadata_i = builder.update_block_table(  # 仅更新block table
                    cached_attn_metadata[cache_key],  # 使用缓存的元数据
                    common_attn_metadata.block_table_tensor,  # 新的block table张量
                    common_attn_metadata.slot_mapping,  # 新的slot映射
                )
            else:  # 否则从头构建
                attn_metadata_i = builder.build(  # 构建注意力元数据
                    common_prefix_len=cascade_attn_prefix_len,  # 级联注意力公共前缀长度
                    common_attn_metadata=common_attn_metadata,  # 公共注意力元数据
                    **extra_attn_metadata_args,  # 额外参数
                )
                if builder.supports_update_block_table:  # 如果构建器支持更新block table
                    cached_attn_metadata[cache_key] = attn_metadata_i  # 缓存构建结果

            if ubid is None:  # 如果没有微批次ID
                assert isinstance(attn_metadata, dict)  # 断言attn_metadata是字典
                attn_metadata_dict = attn_metadata  # 直接使用attn_metadata
            else:  # 否则使用微批次对应的字典
                assert isinstance(attn_metadata, list)  # 断言attn_metadata是列表
                attn_metadata_dict = attn_metadata[ubid]  # 获取对应微批次的字典

            for layer_name in attn_group.layer_names:  # 遍历注意力组中的所有层名
                attn_metadata_dict[layer_name] = attn_metadata_i  # 将构建的元数据赋给每一层

        # Prepare the attention metadata for each KV cache group and make layers  # 为每个KV缓存组准备注意力元数据
        # in the same group share the same metadata.  # 并使同一组中的层共享相同的元数据
        spec_decode_common_attn_metadata = None  # 初始化投机解码的公共注意力元数据为None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):  # 遍历每个KV缓存组
            cm = copy(cm_base)  # shallow copy  # 浅拷贝公共注意力元数据基础对象

            # Basically only the encoder seq_lens, block_table and slot_mapping change  # 基本上只有编码器序列长度、block table和slot映射会变化
            # for each kv_cache_group.  # 对于每个KV缓存组
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(  # 获取编码器序列长度
                num_scheduled_tokens or {},  # 调度的token数
                kv_cache_group.kv_cache_spec,  # KV缓存规格
                num_reqs_padded,  # 填充后的请求数
                for_cudagraph_capture=for_cudagraph_capture,  # 是否用于CUDA图捕获
            )
            if kv_cache_gid > 0:  # 如果不是第一个KV缓存组
                cm.block_table_tensor = _get_block_table(kv_cache_gid)  # 获取该组的block table
                cm.slot_mapping = slot_mappings[kv_cache_gid]  # 获取该组的slot映射

            if self.speculative_config and spec_decode_common_attn_metadata is None:  # 如果有投机配置且尚未设置
                if isinstance(self.drafter, EagleProposer):  # 如果drafter是Eagle提议者
                    if self.drafter.kv_cache_gid == kv_cache_gid:  # 如果当前组匹配drafter的KV缓存组
                        spec_decode_common_attn_metadata = cm  # 设置投机解码的公共注意力元数据
                else:  # 否则
                    spec_decode_common_attn_metadata = cm  # 直接设置

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):  # 遍历每个注意力组
                if ubatch_slices is not None:  # 如果有微批次切片
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):  # 遍历切分后的注意力元数据
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)  # 为每个微批次构建注意力组元数据

                else:  # 否则不使用微批次
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)  # 直接构建注意力组元数据

        if self.is_mm_prefix_lm:  # 如果是多模态前缀语言模型
            req_doc_ranges = {}  # 初始化请求文档范围字典
            for req_id in self.input_batch.req_ids:  # 遍历每个请求ID
                image_doc_ranges = []  # 初始化图像文档范围列表
                req_state = self.requests[req_id]  # 获取请求状态
                for mm_feature in req_state.mm_features:  # 遍历请求的多模态特征
                    pos_info = mm_feature.mm_position  # 获取多模态位置信息
                    img_doc_range = pos_info.extract_embeds_range()  # 提取嵌入范围
                    image_doc_ranges.extend(img_doc_range)  # 添加到图像文档范围列表
                req_idx = self.input_batch.req_id_to_index[req_id]  # 获取请求在批次中的索引
                req_doc_ranges[req_idx] = image_doc_ranges  # 存储请求的文档范围

            if isinstance(attn_metadata, list):  # 如果注意力元数据是列表（微批次模式）
                for ub_metadata in attn_metadata:  # 遍历每个微批次的元数据
                    for _metadata in ub_metadata.values():  # 遍历每层的元数据
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]  # 设置多模态前缀范围
            else:  # 否则是字典模式
                for _metadata in attn_metadata.values():  # 遍历每层的元数据
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]  # 设置多模态前缀范围

        if spec_decode_common_attn_metadata is not None and (  # 如果投机解码的公共注意力元数据存在且有填充
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded  # 实际数量与填充后数量不同
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies  # 当前drafter仍使用分段CUDA图
            # the attention metadata in directly), and therefore does not want to use  # 并直接修改注意力元数据
            # padded attention metadata.  # 因此不需要使用填充后的注意力元数据
            spec_decode_common_attn_metadata = (  # 将投机解码的注意力元数据恢复为未填充版本
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)  # 获取未填充的版本
            )

        return attn_metadata, spec_decode_common_attn_metadata  # 返回注意力元数据和投机解码公共注意力元数据

    def _compute_cascade_attn_prefix_lens(  # 定义计算级联注意力前缀长度的方法
        self,  # 实例自身引用
        num_scheduled_tokens: np.ndarray,  # 每个请求调度的token数
        num_computed_tokens: np.ndarray,  # 每个请求已计算的token数
        num_common_prefix_blocks: list[int],  # 每个KV缓存组的公共前缀block数
    ) -> list[list[int]] | None:  # 返回二维列表或None
        """
        计算级联注意力的公共前缀长度。返回二维数组[kv_cache_group_id][attn_group_idx]。
        如果不应使用级联注意力则返回None。
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False  # 初始化级联注意力标志为False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)  # 获取KV缓存组数量
        cascade_attn_prefix_lens: list[list[int]] = [  # 初始化级联注意力前缀长度二维列表
            [] for _ in range(num_kv_cache_groups)  # 为每个KV缓存组创建空列表
        ]

        for kv_cache_gid in range(num_kv_cache_groups):  # 遍历每个KV缓存组
            for attn_group in self.attn_groups[kv_cache_gid]:  # 遍历每个注意力组
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):  # 如果是仅编码器注意力规格
                    cascade_attn_prefix_len = 0  # 前缀长度为0
                else:  # 否则
                    # 0 if cascade attention should not be used  # 如果不应使用级联注意力则为0
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(  # 计算级联注意力前缀长度
                        num_scheduled_tokens,  # 调度的token数
                        num_computed_tokens,  # 已计算的token数
                        num_common_prefix_blocks[kv_cache_gid],  # 该组的公共前缀block数
                        attn_group.kv_cache_spec,  # KV缓存规格
                        attn_group.get_metadata_builder(),  # 获取元数据构建器
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)  # 添加前缀长度到列表
                use_cascade_attn |= cascade_attn_prefix_len > 0  # 如果任何前缀长度大于0则标记使用级联注意力

        return cascade_attn_prefix_lens if use_cascade_attn else None  # 如果使用级联注意力则返回前缀长度，否则返回None

    # 计算级联注意力的公共前缀长度。
    # 级联注意力将注意力计算分为两个 kernel：公共前缀（双向注意力）和请求独立部分。
    # 前缀长度必须是 block_size 的倍数，且不超过最小 num_computed_tokens，
    # 以避免需要 masking 的边界情况。
    def _compute_cascade_attn_prefix_len(  # 定义计算单个级联注意力前缀长度的方法
        self,  # 实例自身引用
        num_scheduled_tokens: np.ndarray,  # 每个请求调度的token数
        num_computed_tokens: np.ndarray,  # 每个请求已计算的token数
        num_common_prefix_blocks: int,  # 公共前缀block数
        kv_cache_spec: KVCacheSpec,  # KV缓存规格
        attn_metadata_builder: AttentionMetadataBuilder,  # 注意力元数据构建器
    ) -> int:  # 返回整数前缀长度
        """Compute the length of the common prefix for cascade attention.
        计算级联注意力的公共前缀长度。

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size  # 公共前缀长度 = 公共前缀block数 * block大小
        if common_prefix_len == 0:  # 如果公共前缀长度为0
            # Common case.  # 常见情况
            return 0  # 直接返回0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one  # 注意：级联注意力使用两个注意力kernel
        # for the common prefix and the other for the rest. For the first  # 一个用于公共前缀，另一个用于其余部分
        # kernel, we concatenate all the query tokens (possibly from  # 对于第一个kernel，我们拼接所有查询token
        # different requests) and treat them as if they are from the same  # （可能来自不同请求）并将它们视为来自同一请求
        # request. Then, we use bi-directional attention to process the  # 然后使用双向注意力处理KV缓存中的公共前缀
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.  # 重要的是，第一个kernel不做任何masking

        # Consider the following example:  # 考虑以下示例
        # Request 1's input query: [D, E, X]  # 请求1的输入查询
        # Request 1's kv cache: [A, B, C, D, E, X]  # 请求1的KV缓存
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])  # 请求1已计算的token数
        # Request 2's input query: [E, Y]  # 请求2的输入查询
        # Request 2's kv cache: [A, B, C, D, E, Y]  # 请求2的KV缓存
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])  # 请求2已计算的token数

        # If we use [A, B, C, D, E] as the common prefix, then the  # 如果使用[A,B,C,D,E]作为公共前缀
        # first kernel will compute the bi-directional attention between  # 第一个kernel将计算双向注意力
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].  # 输入查询和公共前缀之间
        # However, this is wrong because D in Request 1 should not attend to  # 但这是错的，因为请求1中的D不应关注公共前缀中的E
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.  # 为避免这个问题，应使用[A,B,C,D]作为公共前缀
        # That is, the common prefix should be capped by the minimum  # 即公共前缀应被限制为最小的num_computed_tokens
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.  # 加一以包含查询的第一个token

        # In practice, we use [A, B, C] as the common prefix, instead of  # 实际上我们使用[A,B,C]作为公共前缀而非[A,B,C,D]
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).  # 即公共前缀被限制为最小num_computed_tokens，不加一
        # This is because of an implementation detail: We want to always  # 这是因为一个实现细节：我们希望始终使用两个kernel
        # use two kernels for cascade attention. Let's imagine:  # 考虑以下情况

        # Request 3's input query: [D]  # 请求3的输入查询：[D]
        # Request 3's kv cache: [A, B, C, D]  # 请求3的KV缓存
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])  # 请求3已计算的token数：3
        # If we use [A, B, C, D] as the common prefix for Request 1-3,  # 如果使用[A,B,C,D]作为请求1-3的公共前缀
        # then Request 3 will be processed only by the first kernel,  # 请求3将只由第一个kernel处理
        # and the second kernel will get an empty input. While this is not  # 第二个kernel将得到空输入
        # a fundamental problem, our current implementation does not support  # 虽然这不是根本性问题，但当前实现不支持这种情况
        # this case.
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())  # 将公共前缀长度限制为最小已计算token数
        # common_prefix_len should be a multiple of the block size.  # 公共前缀长度应是block大小的倍数
        common_prefix_len = (  # 向下取整到block大小的倍数
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size  # 整除后再乘以block大小
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (  # 判断是否使用滑动窗口
            isinstance(kv_cache_spec, FullAttentionSpec)  # 或者是全注意力规格
            and kv_cache_spec.sliding_window is not None  # 且设置了滑动窗口
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (  # 判断是否使用局部注意力
            isinstance(kv_cache_spec, FullAttentionSpec)  # 或者是全注意力规格
            and kv_cache_spec.attention_chunk_size is not None  # 且设置了注意力块大小
        )
        assert isinstance(kv_cache_spec, AttentionSpec)  # 断言KV缓存规格是注意力规格类型
        use_cascade = attn_metadata_builder.use_cascade_attention(  # 调用构建器判断是否使用级联注意力
            common_prefix_len=common_prefix_len,  # 公共前缀长度
            query_lens=num_scheduled_tokens,  # 查询长度（调度的token数）
            num_query_heads=self.num_query_heads,  # 查询头数量
            num_kv_heads=kv_cache_spec.num_kv_heads,  # KV头数量
            use_alibi=self.use_alibi,  # 是否使用ALiBi位置编码
            use_sliding_window=use_sliding_window,  # 是否使用滑动窗口
            use_local_attention=use_local_attention,  # 是否使用局部注意力
            num_sms=self.num_sms,  # GPU SM数量
            dcp_world_size=self.dcp_world_size,  # DCP世界大小
        )
        return common_prefix_len if use_cascade else 0  # 如果使用级联注意力则返回前缀长度，否则返回0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):  # 定义计算M-RoPE位置的方法
        """计算M-RoPE（多维旋转位置编码）位置，适用于Qwen2-VL等多模态模型。"""
        mrope_pos_ptr = 0  # 初始化M-RoPE位置指针
        for index, req_id in enumerate(self.input_batch.req_ids):  # 遍历批次中的每个请求
            req = self.requests[req_id]  # 获取请求对象
            assert req.mrope_positions is not None  # 断言请求有M-RoPE位置数据

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]  # 获取已计算的token数
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]  # 获取调度的token数
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(  # 计算prompt token数量
                req.prompt_token_ids, req.prompt_embeds  # 从prompt token ID或嵌入中获取长度
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:  # 如果计算+调度超过prompt长度
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)  # prompt部分长度
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)  # 补全部分长度
            else:  # 否则全部都是prompt部分
                prompt_part_len = num_scheduled_tokens  # prompt部分长度等于调度的token数
                completion_part_len = 0  # 补全部分长度为0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len  # 断言两部分之和等于调度token数

            if prompt_part_len > 0:  # 如果有prompt部分
                # prompt's mrope_positions are pre-computed  # prompt的M-RoPE位置是预计算的
                dst_start = mrope_pos_ptr  # 目标起始位置
                dst_end = mrope_pos_ptr + prompt_part_len  # 目标结束位置
                src_start = num_computed_tokens  # 源起始位置
                src_end = num_computed_tokens + prompt_part_len  # 源结束位置

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[  # 将预计算的M-RoPE位置拷贝到CPU缓冲区
                    :, src_start:src_end  # 从请求的M-RoPE位置中切片
                ]
                mrope_pos_ptr += prompt_part_len  # 更新指针位置

            if completion_part_len > 0:  # 如果有补全部分
                # compute completion's mrope_positions on-the-fly  # 动态计算补全部分的M-RoPE位置
                dst_start = mrope_pos_ptr  # 目标起始位置
                dst_end = mrope_pos_ptr + completion_part_len  # 目标结束位置

                assert req.mrope_position_delta is not None  # 断言请求有M-RoPE位置增量
                MRotaryEmbedding.get_next_input_positions_tensor(  # 调用M-RoPE嵌入获取下一个输入位置
                    out=self.mrope_positions.np,  # 输出数组
                    out_offset=dst_start,  # 输出偏移量
                    mrope_position_delta=req.mrope_position_delta,  # M-RoPE位置增量
                    context_len=num_computed_tokens + prompt_part_len,  # 上下文长度
                    num_new_tokens=completion_part_len,  # 新token数量
                )

                mrope_pos_ptr += completion_part_len  # 更新指针位置

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):  # 定义计算XD-RoPE位置的方法
        """计算XD-RoPE（扩展维度旋转位置编码）位置，适用于HunYuan-VL等模型。"""
        xdrope_pos_ptr = 0  # 初始化XD-RoPE位置指针
        for index, req_id in enumerate(self.input_batch.req_ids):  # 遍历批次中的每个请求
            req = self.requests[req_id]  # 获取请求对象
            assert req.xdrope_positions is not None  # 断言请求有XD-RoPE位置数据

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]  # 获取已计算的token数
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]  # 获取调度的token数
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(  # 计算prompt token数量
                req.prompt_token_ids, req.prompt_embeds  # 从prompt token ID或嵌入中获取长度
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:  # 如果计算+调度超过prompt长度
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)  # prompt部分长度
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)  # 补全部分长度
            else:  # 否则全部都是prompt部分
                prompt_part_len = num_scheduled_tokens  # prompt部分长度等于调度的token数
                completion_part_len = 0  # 补全部分长度为0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len  # 断言两部分之和等于调度token数

            if prompt_part_len > 0:  # 如果有prompt部分
                # prompt's xdrope_positions are pre-computed  # prompt的XD-RoPE位置是预计算的
                dst_start = xdrope_pos_ptr  # 目标起始位置
                dst_end = xdrope_pos_ptr + prompt_part_len  # 目标结束位置
                src_start = num_computed_tokens  # 源起始位置
                src_end = num_computed_tokens + prompt_part_len  # 源结束位置

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[  # 将预计算的XD-RoPE位置拷贝到CPU缓冲区
                    :, src_start:src_end  # 从请求的XD-RoPE位置中切片
                ]
                xdrope_pos_ptr += prompt_part_len  # 更新指针位置

            if completion_part_len > 0:  # 如果有补全部分
                # compute completion's xdrope_positions on-the-fly  # 动态计算补全部分的XD-RoPE位置
                dst_start = xdrope_pos_ptr  # 目标起始位置
                dst_end = xdrope_pos_ptr + completion_part_len  # 目标结束位置

                XDRotaryEmbedding.get_next_input_positions_tensor(  # 调用XD-RoPE嵌入获取下一个输入位置
                    out=self.xdrope_positions.np,  # 输出数组
                    out_offset=dst_start,  # 输出偏移量
                    context_len=num_computed_tokens + prompt_part_len,  # 上下文长度
                    num_new_tokens=completion_part_len,  # 新token数量
                )

                xdrope_pos_ptr += completion_part_len  # 更新指针位置

    # 计算投机解码的元数据：logits 索引、目标/奖励 logits 索引、累积 draft token 数等。
    # 这些索引用于从模型输出中提取验证 token 和奖励 token 对应的 logits。
    def _calc_spec_decode_metadata(  # 定义计算投机解码元数据的方法
        self,  # 实例自身引用
        num_draft_tokens: np.ndarray,  # 每个请求的draft token数
        cu_num_scheduled_tokens: np.ndarray,  # 累积调度token数
    ) -> SpecDecodeMetadata:  # 返回投机解码元数据
        """
        计算投机解码所需的各种索引和元数据，包括logits索引、target logits索引、
        bonus logits索引和draft token ID。
        """
        # Inputs:  # 输入示例
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]  # 累积调度token数
        # num_draft_tokens:         [  3,   0,   2,   0,   1]  # 每个请求的draft token数
        # Outputs:  # 输出示例
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]  # 累积draft token数
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,  # logits索引
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]  # 目标logits索引
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]  # 奖励logits索引

        # Compute the logits indices.  # 计算logits索引
        # [4, 1, 3, 1, 2]  # 每个请求的采样token数
        num_sampled_tokens = num_draft_tokens + 1  # 采样token数 = draft token数 + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]  # 步骤1：计算累积采样token数
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]  # 局部递增索引
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(  # 计算累积和与局部递增索引
            num_sampled_tokens, cumsum_dtype=np.int32  # 累积和使用int32类型
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]  # 步骤2：基础偏移
        logits_indices = np.repeat(  # 重复每个请求的起始偏移
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens  # 每个请求的起始偏移 = 累积调度数 - 采样数
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]  # 步骤3：加上局部偏移
        logits_indices += arange  # 加上局部递增索引得到最终logits索引

        # Compute the bonus logits indices.  # 计算奖励logits索引
        bonus_logits_indices = cu_num_sampled_tokens - 1  # 奖励logits索引 = 累积采样token数 - 1

        # Compute the draft logits indices.  # 计算draft logits索引
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]  # 累积draft token数
        # arange: [0, 1, 2, 0, 1, 0]  # 局部递增索引
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(  # 计算draft token的累积和与局部递增索引
            num_draft_tokens, cumsum_dtype=np.int32  # 累积和使用int32类型
        )
        # [0, 0, 0, 5, 5, 9]  # 基础偏移
        target_logits_indices = np.repeat(  # 重复每个请求的目标起始偏移
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens  # 每个请求在采样token中的起始位置
        )
        # [0, 1, 2, 5, 6, 9]  # 最终目标logits索引
        target_logits_indices += arange  # 加上局部递增索引

        # TODO: Optimize the CPU -> GPU copy.  # TODO: 优化CPU到GPU的拷贝
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(  # 将累积draft token数转为GPU张量
            self.device, non_blocking=True  # 异步传输到设备
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(  # 将累积采样token数转为GPU张量
            self.device, non_blocking=True  # 异步传输到设备
        )
        logits_indices = torch.from_numpy(logits_indices).to(  # 将logits索引转为GPU张量
            self.device, non_blocking=True  # 异步传输到设备
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(  # 将目标logits索引转为GPU张量
            self.device, non_blocking=True  # 异步传输到设备
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(  # 将奖励logits索引转为GPU张量
            self.device, non_blocking=True  # 异步传输到设备
        )

        # Compute the draft token ids.  # 计算draft token的ID
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]  # draft token的索引
        draft_token_ids = self.input_ids.gpu[logits_indices]  # 从GPU上的input_ids中提取logits位置的token
        draft_token_ids = draft_token_ids[target_logits_indices + 1]  # 通过目标索引+1获取draft token ID

        return SpecDecodeMetadata(  # 返回投机解码元数据对象
            draft_token_ids=draft_token_ids,  # draft token的ID
            num_draft_tokens=num_draft_tokens.tolist(),  # draft token数量列表
            cu_num_draft_tokens=cu_num_draft_tokens,  # 累积draft token数（GPU张量）
            cu_num_sampled_tokens=cu_num_sampled_tokens,  # 累积采样token数（GPU张量）
            target_logits_indices=target_logits_indices,  # 目标logits索引（GPU张量）
            bonus_logits_indices=bonus_logits_indices,  # 奖励logits索引（GPU张量）
            logits_indices=logits_indices,  # logits索引（GPU张量）
        )

    def _prepare_kv_sharing_fast_prefill(  # 定义准备KV共享快速预填充的方法
        self,  # 实例自身引用
        logits_indices: torch.Tensor,  # logits索引张量
    ) -> torch.Tensor:  # 返回填充后的logits索引张量
        """准备KV共享快速预填充的logits索引，填充并确保索引有效。"""
        assert self.kv_sharing_fast_prefill_logits_indices is not None  # 断言KV共享快速预填充logits索引已初始化
        num_logits = logits_indices.shape[0]  # 获取logits索引的数量
        assert num_logits > 0  # 断言logits索引数量大于0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)  # 拷贝logits索引到预分配缓冲区
        # There might have leftover indices in logits_indices[num_logits:]  # logits_indices[num_logits:]可能残留之前的索引
        # from previous iterations, whose values may be greater than the  # 这些值可能大于当前迭代的批次大小
        # batch size in the current iteration. To ensure indices are always  # 为确保索引始终有效
        # valid, we fill the padded indices with the last index.  # 用最后一个索引填充填充部分
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(  # 用最后一个有效索引填充剩余位置
            logits_indices[-1].item()  # 获取最后一个索引的值
        )
        # Dispatch for the decoder portion of the model.  # 为模型的解码器部分分派
        _, batch_desc = self.cudagraph_dispatcher.dispatch(  # 调用CUDA图调度器分派
            num_logits, invalid_modes={CUDAGraphMode.FULL}  # 传入logits数量，排除FULL模式
        )
        num_logits_padded = batch_desc.num_tokens  # 获取填充后的logits数量
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[  # 获取填充后的logits索引切片
            :num_logits_padded  # 截取到填充数量
        ]
        return logits_indices_padded  # 返回填充后的logits索引

    def _batch_mm_inputs_from_scheduler(  # 定义从调度器批量获取多模态输入的方法
        self,  # 实例自身引用
        scheduler_output: "SchedulerOutput",  # 调度器输出对象
    ) -> tuple[  # 返回一个元组
        list[str],  # 多模态哈希列表
        list[tuple[str, MultiModalKwargsItem]],  # 多模态kwargs列表
        list[tuple[str, PlaceholderRange]],  # 多模态LoRA引用列表
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.
        从调度的编码器输入中批量获取多模态输入。

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.  调度器输出，包含调度的编码器输入

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item  多模态哈希列表
            - mm_kwargs: List of multimodal kwargs for each item  多模态kwargs列表
            - mm_lora_refs: List of (req_id, placeholder_range) for each item  多模态LoRA引用列表
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs  # 获取调度的编码器输入
        if not scheduled_encoder_inputs:  # 如果没有调度的编码器输入
            return [], [], []  # 返回空列表

        mm_hashes = list[str]()  # 初始化多模态哈希列表
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()  # 初始化多模态kwargs列表
        # Multimodal LoRA reference info to map each multimodal item  # 多模态LoRA引用信息，用于将每个多模态项
        # back to its request & position  # 映射回其请求和位置
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()  # 初始化多模态LoRA引用列表
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():  # 遍历调度的编码器输入
            req_state = self.requests[req_id]  # 获取请求状态

            for mm_input_id in encoder_input_ids:  # 遍历编码器输入ID
                mm_feature = req_state.mm_features[mm_input_id]  # 获取多模态特征
                if mm_feature.data is None:  # 如果多模态特征数据为空
                    continue  # 跳过

                mm_hashes.append(mm_feature.identifier)  # 添加多模态标识符到哈希列表
                mm_kwargs.append((mm_feature.modality, mm_feature.data))  # 添加模态名和数据到kwargs列表
                mm_lora_refs.append((req_id, mm_feature.mm_position))  # 添加请求ID和位置到LoRA引用列表

        return mm_hashes, mm_kwargs, mm_lora_refs  # 返回三个列表

    # 执行多模态编码器（视觉/音频等）的前向传播。
    # 按模态分批处理输入，支持 LoRA 的塔式/连接器映射，
    # 将编码结果缓存到 encoder_cache 中供后续的注意力层使用。
    def _execute_mm_encoder(  # 定义执行多模态编码器的方法
        self, scheduler_output: "SchedulerOutput"  # 调度器输出对象
    ) -> list[torch.Tensor]:  # 返回编码器输出张量列表
        """执行多模态编码器前向传播，处理视觉/音频等多模态输入并返回编码结果。"""
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(  # 从调度器获取批量多模态输入
            scheduler_output  # 传入调度器输出
        )

        if not mm_kwargs:  # 如果没有多模态输入
            return []  # 返回空列表

        should_time = bool(  # 判断是否需要计时
            self.observability_config  # 存在可观测性配置
            and self.observability_config.enable_mm_processor_stats  # 且启用了多模态处理器统计
            and scheduler_output.scheduled_encoder_inputs  # 且有调度的编码器输入
        )
        # multiple modalities or a different modality than the previous one,  # 如果请求包含多种模态或与前一个模态不同
        # we process it separately to preserve item order.  # 我们将其单独处理以保持项目顺序
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities  # FIXME: 这是处理同一批次中多种模态的临时方案
        # in the same batch while still being able to benefit from batching  # 同时仍然能从批处理多模态输入中获益
        # multimodal inputs. The proper solution should be reordering the  # 正确的解决方案应该是重新排序
        # encoder outputs.  # 编码器输出
        model = cast(SupportsMultiModal, self.model)  # 将模型转换为支持多模态的类型

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():  # 如果启用了LoRA配置且LoRA管理器支持塔连接器LoRA
            # Build LoRA mappings independently for encoder inputs  # 为编码器输入独立构建LoRA映射
            # (encoder batch structure is different from main batch)  # （编码器批次结构与主批次不同）
            prompt_lora_mapping = []  # 初始化提示级别的LoRA映射列表
            token_lora_mapping = []  # 初始化token级别的LoRA映射列表
            lora_requests = set()  # 初始化LoRA请求集合
            encoder_token_counts = []  # 初始化编码器token计数列表

            for req_id, pos_info in mm_lora_refs:  # 遍历多模态LoRA引用中的请求ID和位置信息
                req_idx = self.input_batch.req_id_to_index[req_id]  # 根据请求ID获取在批次中的索引
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])  # 获取该请求对应的LoRA适配器ID

                # Prefer pos_info.get_num_embeds to count precise MM embedding tokens.  # 优先使用pos_info.get_num_embeds来精确计算多模态嵌入token数
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]  # 获取多模态编码器的token数量
                    pos_info.get_num_embeds()  # 传入嵌入数量
                )
                prompt_lora_mapping.append(lora_id)  # 将LoRA ID添加到提示映射
                token_lora_mapping.extend([lora_id] * num_tokens)  # 将LoRA ID扩展到每个token的映射
                encoder_token_counts.append(num_tokens)  # 记录编码器token计数

                if lora_id > 0:  # 如果LoRA ID有效（大于0）
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)  # 获取对应的LoRA请求对象
                    if lora_request is not None:  # 如果LoRA请求存在
                        lora_requests.add(lora_request)  # 将其添加到LoRA请求集合

            # Set tower adapter mapping  # 设置塔适配器映射
            tower_mapping = LoRAMapping(  # 创建LoRA映射对象
                tuple(token_lora_mapping),  # token级别的LoRA映射元组
                tuple(prompt_lora_mapping),  # 提示级别的LoRA映射元组
                is_prefill=True,  # 设置为预填充模式
                type=LoRAMappingType.TOWER,  # 映射类型为TOWER（塔）
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)  # 激活对应的LoRA适配器

            if hasattr(self.model, "get_num_mm_connector_tokens"):  # 如果模型有获取多模态连接器token数的方法
                post_op_counts = [  # 计算后处理操作的token数
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]  # 获取连接器token数量
                    for num_tokens in encoder_token_counts  # 遍历编码器token计数
                ]

                connector_token_mapping = np.repeat(  # 重复创建连接器token映射数组
                    np.array(prompt_lora_mapping, dtype=np.int32),  # 将提示LoRA映射转为numpy数组
                    np.array(post_op_counts, dtype=np.int32),  # 按后处理token数重复
                )
                connector_mapping = LoRAMapping(  # 创建连接器LoRA映射
                    index_mapping=tuple(connector_token_mapping.tolist()),  # 索引映射元组
                    prompt_mapping=tuple(prompt_lora_mapping),  # 提示映射元组
                    is_prefill=True,  # 设置为预填充模式
                    type=LoRAMappingType.CONNECTOR,  # 映射类型为CONNECTOR（连接器）
                )

                self.lora_manager.set_active_adapters(  # 激活连接器的LoRA适配器
                    lora_requests,  # LoRA请求集合
                    connector_mapping,  # 连接器映射
                )

        encoder_outputs: list[torch.Tensor] = []  # 初始化编码器输出列表
        # Track the current index in mm_kwargs/mm_lora_refs to map groups to request IDs  # 跟踪mm_kwargs/mm_lora_refs中的当前索引以将组映射到请求ID
        current_item_idx = 0  # 初始化当前项目索引
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(  # 遍历分组和批处理后的多模态参数
            mm_kwargs,  # 多模态关键字参数
            device=self.device,  # 目标设备
            pin_memory=self.pin_memory,  # 是否固定内存
        ):
            batch_outputs: MultiModalEmbeddings  # 声明批次输出类型为多模态嵌入

            # EVS-related change.  # EVS相关的更改
            # (ekhvedchenia): Temporary hack to limit peak memory usage when  # 临时方案：限制处理多模态数据时的峰值内存使用
            # processing multimodal data. This solves the issue with scheduler  # 解决调度器将过多视频样本放入单个批次的问题
            # putting too many video samples into a single batch. Scheduler  # 调度器使用剪枝后的视觉token数与计算预算比较
            # uses pruned vision tokens count to compare it versus compute  # 这是不正确的
            # budget which is incorrect (Either input media size or non-pruned  # （应该考虑输入媒体大小或未剪枝的输出视觉token数）
            # output vision tokens count should be considered)
            # TODO(ywang96): Fix memory profiling to take EVS into account and  # TODO: 修复内存分析以考虑EVS并移除此临时方案
            # remove this hack.
            if (  # 如果满足以下条件
                self.is_multimodal_pruning_enabled  # 多模态剪枝已启用
                and modality == "video"  # 且当前模态是视频
                and num_items > 1  # 且有多个视频项目
            ):
                batch_outputs_lst = list[torch.Tensor]()  # 初始化批次输出列表
                for video_idx in range(num_items):  # 逐个处理每个视频
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]  # 获取当前视频的多模态参数
                    with self.timed_encoder_operation(  # 使用计时上下文管理器
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1  # 传入计时参数
                    ):
                        _, _, micro_batch_mm_inputs = next(  # 获取微批次的多模态输入
                            group_and_batch_mm_kwargs(  # 对单个视频进行分组和批处理
                                [video_mm_kwargs_item],  # 单个视频项
                                device=self.device,  # 目标设备
                                pin_memory=self.pin_memory,  # 是否固定内存
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(  # 对微批次执行多模态编码
                            **micro_batch_mm_inputs  # 传入微批次多模态输入
                        )

                        batch_outputs_lst.extend(micro_batch_outputs)  # 将微批次输出添加到列表

                batch_outputs = batch_outputs_lst  # 将所有微批次输出作为批次输出
            else:
                # Run the encoder.  # 运行编码器
                # `batch_outputs` is either of the following:  # batch_outputs可能是以下之一：
                # 1. A tensor of shape (num_items, feature_size, hidden_size)  # 1. 形状为(num_items, feature_size, hidden_size)的张量（特征大小固定时）
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,  # 2. 长度为num_items的张量列表或元组
                # each of shape (feature_size, hidden_size) in case the feature  # 每个张量形状为(feature_size, hidden_size)（特征大小动态时）
                # size is dynamic depending on the input multimodal items.

                with self.timed_encoder_operation(  # 使用计时上下文管理器记录编码器操作时间
                    should_time, mm_lora_refs, current_item_idx, num_items  # 传入计时参数和当前批次信息
                ):
                    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)  # 对整个批次执行多模态编码

            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)  # 检查编码器输出的正确性
            encoder_outputs.extend(batch_outputs)  # 将批次输出添加到总输出列表

            current_item_idx += num_items  # 更新当前项目索引

        # Cache the encoder outputs by mm_hash  # 按mm_hash缓存编码器输出
        for mm_hash, output in zip(mm_hashes, encoder_outputs):  # 遍历哈希值和对应的编码器输出
            self.encoder_cache[mm_hash] = output  # 将输出存入编码器缓存
            logger.debug("Finish execute for mm hash %s", mm_hash)  # 记录完成执行的调试日志
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)  # 可能将编码器缓存保存到连接器

        return encoder_outputs  # 返回所有编码器输出

    # 从编码器缓存中收集当前步骤所需的多模态嵌入。
    # 根据每个请求的已计算 token 数和调度 token 数，
    # 确定需要替换为编码器输出的位置，生成 is_mm_embed 掩码。
    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出对象
        shift_computed_tokens: int = 0,  # 已计算token的偏移量，默认为0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:  # 返回多模态嵌入列表和多模态嵌入掩码张量
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取总调度token数

        # Swap to the other buffer to avoid race condition with previous  # 切换到另一个缓冲区以避免与前一次迭代异步拷贝的竞争条件
        # iteration's async copy that may still be reading from CPU.  # 前一次迭代的异步拷贝可能仍在从CPU读取
        self.is_mm_embed_idx = 1 - self.is_mm_embed_idx  # 切换双缓冲索引（0和1之间切换）
        is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]  # 获取当前活跃的缓冲区

        mm_embeds = list[torch.Tensor]()  # 初始化多模态嵌入列表
        is_mm_embed = is_mm_embed_buf.cpu  # 获取CPU端的多模态嵌入掩码
        is_mm_embed[:total_num_scheduled_tokens] = False  # 将掩码初始化为False

        req_start_idx = 0  # 初始化请求在批次中的起始索引
        should_sync_mrope_positions = False  # 是否需要同步mrope位置编码的标志
        should_sync_xdrope_positions = False  # 是否需要同步xdrope位置编码的标志

        for req_id in self.input_batch.req_ids:  # 遍历批次中的每个请求ID
            mm_embeds_req: list[torch.Tensor] = []  # 初始化当前请求的多模态嵌入列表

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]  # 获取当前请求的调度token数
            req_state = self.requests[req_id]  # 获取当前请求的状态
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens  # 计算已处理的token总数（加上偏移）

            for mm_feature in req_state.mm_features:  # 遍历当前请求的多模态特征
                pos_info = mm_feature.mm_position  # 获取多模态位置信息
                start_pos = pos_info.offset  # 获取多模态数据在序列中的起始位置
                num_encoder_tokens = pos_info.length  # 获取编码器输出的token数量

                # The encoder output is needed if the two ranges overlap:  # 如果以下两个范围重叠则需要编码器输出
                # [num_computed_tokens,  # [已计算token数,
                #  num_computed_tokens + num_scheduled_tokens) and  #  已计算token数 + 调度token数) 和
                # [start_pos, start_pos + num_encoder_tokens)  # [起始位置, 起始位置 + 编码器token数)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:  # 如果编码器输出在当前步骤之后
                    # The encoder output is not needed in this step.  # 此步骤不需要该编码器输出
                    break  # 跳出循环
                if start_pos + num_encoder_tokens <= num_computed_tokens:  # 如果编码器输出已经在之前的步骤中处理
                    # The encoder output is already processed and stored  # 编码器输出已被处理并存储
                    # in the decoder's KV cache.  # 在解码器的KV缓存中
                    continue  # 继续处理下一个多模态特征

                start_idx = max(num_computed_tokens - start_pos, 0)  # 计算当前步骤中编码器输出的起始切片索引
                end_idx = min(  # 计算当前步骤中编码器输出的结束切片索引
                    num_computed_tokens - start_pos + num_scheduled_tokens,  # 已计算token偏移加上调度token数
                    num_encoder_tokens,  # 不超过编码器token总数
                )
                assert start_idx < end_idx  # 断言切片范围有效
                curr_embeds_start, curr_embeds_end = (  # 获取当前范围内嵌入的起止索引
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)  # 根据切片范围获取嵌入索引
                )
                # If there are no embeddings in the current range, we skip  # 如果当前范围内没有嵌入
                # gathering the embeddings.  # 则跳过嵌入收集
                if curr_embeds_start == curr_embeds_end:  # 如果嵌入起止索引相同（无嵌入）
                    continue  # 跳过

                mm_hash = mm_feature.identifier  # 获取多模态特征的哈希标识符
                encoder_output = self.encoder_cache.get(mm_hash, None)  # 从编码器缓存中获取输出
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."  # 断言缓存命中

                if (is_embed := pos_info.is_embed) is not None:  # 如果存在嵌入掩码（部分位置需要嵌入）
                    is_embed = is_embed[start_idx:end_idx]  # 切片获取当前范围的嵌入掩码
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]  # 从编码器输出中切片获取嵌入
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]  # 直接从编码器输出中切片获取嵌入

                req_start_pos = req_start_idx + start_pos - num_computed_tokens  # 计算在批次中的相对起始位置
                # OR mask for overlapping mm_features (use_audio_in_video)  # 对重叠的多模态特征进行OR掩码操作（用于视频中的音频）
                if is_embed is None:  # 如果没有嵌入掩码（全部位置都是嵌入）
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (  # 将对应范围设置为True
                        True
                    )
                else:
                    is_mm_embed[  # 使用OR运算合并嵌入掩码
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed  # 按位或操作以处理重叠区域
                mm_embeds_req.append(mm_embeds_item)  # 将当前嵌入项添加到请求的嵌入列表

            if self.is_multimodal_pruning_enabled and self.uses_mrope:  # 如果启用了多模态剪枝且使用mrope位置编码
                assert req_state.mrope_positions is not None  # 断言mrope位置不为空
                should_sync_mrope_positions = True  # 标记需要同步mrope位置
                mm_embeds_req, new_mrope_positions, new_delta = (  # 重新计算mrope位置
                    self.model.recompute_mrope_positions(  # 调用模型的mrope位置重计算方法
                        input_ids=req_state.prompt_token_ids,  # 传入提示token ID
                        multimodal_embeddings=mm_embeds_req,  # 传入多模态嵌入
                        mrope_positions=req_state.mrope_positions,  # 传入当前mrope位置
                        num_computed_tokens=req_state.num_computed_tokens,  # 传入已计算token数
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)  # 更新请求状态中的mrope位置
                req_state.mrope_position_delta = new_delta  # 更新mrope位置增量

            mm_embeds.extend(mm_embeds_req)  # 将当前请求的嵌入添加到总嵌入列表
            req_start_idx += num_scheduled_tokens  # 更新请求起始索引

        is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)  # 将多模态嵌入掩码拷贝到GPU

        if should_sync_mrope_positions:  # 如果需要同步mrope位置
            self._calc_mrope_positions(scheduler_output)  # 计算mrope位置
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)  # 将mrope位置拷贝到GPU

        if should_sync_xdrope_positions:  # 如果需要同步xdrope位置
            self._calc_xdrope_positions(scheduler_output)  # 计算xdrope位置
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)  # 将xdrope位置拷贝到GPU

        return mm_embeds, is_mm_embed  # 返回多模态嵌入列表和掩码

    def get_model(self) -> nn.Module:  # 获取底层模型对象
        """获取底层的原始模型对象，如果模型被包装（如CUDAGraph包装器），则解包返回原始模型。"""
        if not hasattr(self, "model"):  # 如果模型尚未初始化
            raise ValueError("Cannot get model before model has been initialized")  # 抛出错误：模型未初始化前不能获取
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):  # 如果模型被CUDAGraph或UBatch包装
            # get raw model out of the cudagraph wrapper.  # 从CUDAGraph包装器中获取原始模型
            return self.model.unwrap()  # 解包并返回原始模型
        return self.model  # 直接返回模型

    def get_supported_generation_tasks(self) -> list[GenerationTask]:  # 获取模型支持的生成任务列表
        """获取模型支持的所有生成任务类型（如generate、transcription、realtime等）。"""
        model = self.get_model()  # 获取底层模型
        supported_tasks = list[GenerationTask]()  # 初始化支持的任务列表

        if is_text_generation_model(model):  # 如果模型支持文本生成
            supported_tasks.append("generate")  # 添加"generate"任务

        if supports_transcription(model):  # 如果模型支持语音转写
            if model.supports_transcription_only:  # 如果模型只支持转写（不支持其他任务）
                return ["transcription"]  # 只返回转写任务

            supported_tasks.append("transcription")  # 添加"transcription"任务

        if supports_realtime(model):  # 如果模型支持实时推理
            supported_tasks.append("realtime")  # 添加"realtime"任务

        return supported_tasks  # 返回支持的任务列表

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:  # 获取模型支持的池化任务列表
        """获取模型支持的所有池化任务类型（如embed、classify、score等）。"""
        model = self.get_model()  # 获取底层模型
        if not is_pooling_model(model):  # 如果模型不是池化模型
            return []  # 返回空列表

        supported_tasks = list(model.pooler.get_supported_tasks())  # 获取池化器支持的任务列表

        if "score" in supported_tasks:  # 如果支持评分任务
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)  # 获取模型配置中的标签数量
            if num_labels != 1:  # 如果标签数量不为1
                supported_tasks.remove("score")  # 移除评分任务
                logger.debug_once("Score API is only enabled for num_labels == 1.")  # 记录调试日志：评分API仅在num_labels==1时启用

        return supported_tasks  # 返回支持的池化任务列表

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取所有支持的任务
        """获取模型支持的所有任务类型的元组，包括生成任务和池化任务。"""
        tasks = list[SupportedTask]()  # 初始化任务列表

        if self.model_config.runner_type == "generate":  # 如果运行器类型为生成
            tasks.extend(self.get_supported_generation_tasks())  # 添加所有支持的生成任务
        if self.model_config.runner_type == "pooling":  # 如果运行器类型为池化
            tasks.extend(self.get_supported_pooling_tasks())  # 添加所有支持的池化任务

        return tuple(tasks)  # 返回任务元组

    def sync_and_slice_intermediate_tensors(  # 同步和切片中间张量
        self,
        num_tokens: int,  # token数量
        intermediate_tensors: IntermediateTensors | None,  # 中间张量（来自前一流水线阶段）
        sync_self: bool,  # 是否同步自身的中间张量
    ) -> IntermediateTensors:  # 返回切片后的中间张量
        """同步并切片流水线并行中间张量，处理序列并行下residual张量的分片。"""
        assert self.intermediate_tensors is not None  # 断言中间张量缓冲区已初始化

        tp = self.vllm_config.parallel_config.tensor_parallel_size  # 获取张量并行大小
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)  # 判断residual是否在序列并行中被分散

        # When sequence parallelism is enabled, the "residual" tensor is sharded  # 启用序列并行时，residual张量跨张量并行rank分片
        # across tensor parallel ranks, so each rank only needs its own slice.  # 每个rank只需要自己的切片
        if sync_self:  # 如果需要同步自身
            assert intermediate_tensors is not None  # 断言中间张量不为空
            for k, v in intermediate_tensors.items():  # 遍历中间张量的每个键值对
                is_scattered = k == "residual" and is_rs  # 判断当前张量是否为分散的residual
                copy_len = num_tokens // tp if is_scattered else num_tokens  # 计算需要拷贝的长度
                self.intermediate_tensors[k][:copy_len].copy_(  # 将中间张量拷贝到缓冲区
                    v[:copy_len], non_blocking=True  # 使用非阻塞拷贝
                )

        return IntermediateTensors(  # 返回切片后的中间张量
            {
                k: v[: num_tokens // tp]  # 如果是分散的residual，按TP大小切片
                if k == "residual" and is_rs
                else v[:num_tokens]  # 否则按token数切片
                for k, v in self.intermediate_tensors.items()  # 遍历所有中间张量
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:  # EPLB步进方法
        """
        EPLB（专家并行负载均衡）状态的步进更新。
        用于在每一步更新专家负载均衡状态，支持MoE模型的专家路由优化。
        """
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:  # 如果未启用EPLB或被抑制
            return  # 直接返回

        assert self.eplb_state is not None  # 断言EPLB状态已初始化
        model = self.get_model()  # 获取底层模型
        assert is_mixture_of_experts(model)  # 断言模型是混合专家模型
        self.eplb_state.step(  # 执行EPLB状态步进
            is_dummy,  # 是否为虚拟步骤
            is_profile,  # 是否为性能分析步骤
            log_stats=self.parallel_config.eplb_config.log_balancedness,  # 是否记录均衡性统计
        )

    def setup_eplb_from_mapping(  # 从映射设置EPLB状态
        self,
        expanded_physical_to_logical: torch.Tensor,  # 扩展的物理到逻辑专家映射张量
        old_num_physical_experts: int,  # 旧的物理专家数量
    ) -> None:  # 无返回值
        """从给定的物理到逻辑专家映射初始化EPLB状态，用于专家并行负载均衡。"""
        model = self.get_model()  # 获取底层模型
        assert is_mixture_of_experts(model)  # 断言模型是混合专家模型

        self.eplb_state = EplbState.from_mapping(  # 从映射创建EPLB状态
            model=model,  # 模型
            model_config=self.model_config,  # 模型配置
            device=self.device,  # 设备
            parallel_config=self.parallel_config,  # 并行配置
            expanded_physical_to_logical=expanded_physical_to_logical,  # 物理到逻辑的专家映射
            num_valid_physical_experts=old_num_physical_experts,  # 有效物理专家数量
        )

    # 执行池化模型的后处理：调用模型的 pooler 层对隐藏状态进行池化，
    # 处理 late interaction（如 ColBERT）的后处理，并构建输出结果。
    def _pool(  # 执行池化操作
        self,
        hidden_states: torch.Tensor,  # 模型隐藏状态张量
        num_scheduled_tokens: int,  # 调度的token数量
        num_scheduled_tokens_np: np.ndarray,  # 每个请求的调度token数numpy数组
        kv_connector_output: KVConnectorOutput | None,  # KV连接器输出
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:  # 返回模型运行器输出
        """对模型隐藏状态执行池化操作，构建池化输出结果，支持同步和异步模式。"""
        num_reqs = self.input_batch.num_reqs  # 获取批次中的请求数量
        assert num_reqs == len(self.input_batch.pooling_params), (  # 断言所有请求都是池化请求
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]  # 截取有效的隐藏状态
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]  # 获取CPU上的序列长度

        pooling_metadata = self.input_batch.get_pooling_metadata()  # 获取池化元数据
        pooling_metadata.build_pooling_cursor(  # 构建池化游标
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device  # 传入token数、序列长度和设备
        )

        model = cast(VllmModelForPooling, self.model)  # 将模型转换为池化模型类型
        raw_pooler_output: PoolerOutput = model.pooler(  # 执行池化层前向传播
            hidden_states=hidden_states, pooling_metadata=pooling_metadata  # 传入隐藏状态和池化元数据
        )

        finished_mask = [  # 创建完成掩码列表
            seq_len == prompt_len  # 判断序列长度是否等于提示长度（即是否已完成）
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)  # 遍历序列长度和提示长度
        ]
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(  # 对池化输出进行后处理（如ColBERT的late interaction）
            raw_pooler_output=raw_pooler_output,  # 原始池化输出
            pooling_params=pooling_metadata.pooling_params,  # 池化参数
            req_ids=self.input_batch.req_ids,  # 请求ID列表
            finished_mask=finished_mask,  # 完成掩码
        )

        model_runner_output = ModelRunnerOutput(  # 创建模型运行器输出对象
            req_ids=self.input_batch.req_ids.copy(),  # 复制请求ID列表
            req_id_to_index=self.input_batch.req_id_to_index.copy(),  # 复制请求ID到索引的映射
            kv_connector_output=kv_connector_output,  # KV连接器输出
        )

        if raw_pooler_output is None or not any(finished_mask):  # 如果池化输出为空或没有完成的请求
            model_runner_output.pooler_output = [None] * num_reqs  # 将池化输出设为None列表
            return model_runner_output  # 返回模型运行器输出

        if self.use_async_scheduling:  # 如果使用异步调度
            return AsyncGPUPoolingModelRunnerOutput(  # 返回异步GPU池化模型运行器输出
                model_runner_output=model_runner_output,  # 模型运行器输出
                raw_pooler_output=raw_pooler_output,  # 原始池化输出
                finished_mask=finished_mask,  # 完成掩码
                async_output_copy_stream=self.async_output_copy_stream,  # 异步输出拷贝流
            )

        model_runner_output.pooler_output = _copy_pooler_output_to_cpu(  # 将池化输出拷贝到CPU
            raw_pooler_output=raw_pooler_output,  # 原始池化输出
            finished_mask=finished_mask,  # 完成掩码
        )
        self._sync_device()  # 同步设备（等待GPU操作完成）

        return model_runner_output  # 返回模型运行器输出

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:  # 为序列并行进行padding
        """当启用序列并行的集合融合时，将token数量padding为张量并行大小的倍数。"""
        # Pad tokens to multiple of tensor_parallel_size when  # 当启用SP的集合融合时
        # enabled collective fusion for SP  # 将token数padding为tensor_parallel_size的倍数
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size  # 获取张量并行大小
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:  # 如果启用了序列并行且TP大于1
            return round_up(num_scheduled_tokens, tp_size)  # 向上取整为TP大小的倍数
        return num_scheduled_tokens  # 否则不进行padding

    def _prepare_mm_inputs(  # 准备多模态输入
        self, num_tokens: int  # token数量
    ) -> tuple[torch.Tensor | None, torch.Tensor]:  # 返回可选的input_ids和inputs_embeds
        """准备多模态模型的输入，返回原始token ID（如果模型需要）和输入嵌入。"""
        if self.model.requires_raw_input_tokens:  # 如果模型需要原始输入token
            input_ids = self.input_ids.gpu[:num_tokens]  # 获取GPU上的输入token ID
        else:
            input_ids = None  # 不需要原始输入token

        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]  # 获取GPU上的输入嵌入
        return input_ids, inputs_embeds  # 返回input_ids和inputs_embeds

    def _preprocess(  # 执行模型前向传播前的预处理
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出
        num_input_tokens: int,  # Padded  # 填充后的输入token数量
        intermediate_tensors: IntermediateTensors | None = None,  # 中间张量（流水线并行时使用）
    ) -> tuple[  # 返回预处理结果元组
        torch.Tensor | None,  # input_ids（可选）
        torch.Tensor | None,  # inputs_embeds（可选）
        torch.Tensor,  # positions位置编码
        IntermediateTensors | None,  # 中间张量
        dict[str, Any],  # 模型额外关键字参数
        ECConnectorOutput | None,  # EC连接器输出
    ]:
        """预处理模型输入：处理多模态编码、嵌入计算、位置编码和流水线并行中间张量。"""
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取总调度token数
        is_first_rank = get_pp_group().is_first_rank  # 判断是否为流水线并行的第一个rank
        is_encoder_decoder = self.model_config.is_encoder_decoder  # 判断是否为编码器-解码器模型

        # _prepare_inputs may reorder the batch, so we must gather multi  # _prepare_inputs可能重排批次，所以必须在之后收集多模态输出
        # modal outputs after that to ensure the correct order  # 以确保正确的顺序
        ec_connector_output = None  # 初始化EC连接器输出

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:  # 如果支持多模态输入、是第一个rank且非编码器-解码器模型
            # Run the multimodal encoder if any.  # 运行多模态编码器（如果有的话）
            with self.maybe_get_ec_connector_output(  # 使用EC连接器输出上下文管理器
                scheduler_output,  # 调度器输出
                encoder_cache=self.encoder_cache,  # 编码器缓存
            ) as ec_connector_output:  # 获取EC连接器输出
                self._execute_mm_encoder(scheduler_output)  # 执行多模态编码器
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)  # 收集多模态嵌入

            # NOTE(woosuk): To unify token ids and soft tokens (vision  # 注意：为统一token id和软token（视觉嵌入）
            # embeddings), we always use embeddings (rather than token ids)  # 我们始终使用嵌入（而非token id）作为多模态模型的输入
            # as input to the multimodal model, even when the input is text.  # 即使输入是纯文本也是如此
            inputs_embeds_scheduled = self.model.embed_input_ids(  # 将输入token ID转换为嵌入
                self.input_ids.gpu[:num_scheduled_tokens],  # GPU上的输入token ID
                multimodal_embeddings=mm_embeds,  # 多模态嵌入
                is_multimodal=is_mm_embed,  # 多模态掩码
            )

            # TODO(woosuk): Avoid the copy. Optimize.  # TODO: 避免拷贝，优化性能
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)  # 将嵌入拷贝到GPU缓冲区

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)  # 准备多模态输入
            model_kwargs = {  # 构建模型关键字参数
                **self._init_model_kwargs(),  # 基础模型参数
                **self._extract_mm_kwargs(scheduler_output),  # 多模态相关参数
            }
        elif self.enable_prompt_embeds and is_first_rank:  # 如果启用了提示嵌入且是第一个rank
            # Get the input embeddings for the tokens that are not input embeds,  # 获取非输入嵌入token的嵌入
            # then put them into the appropriate positions.  # 然后将它们放到正确的位置
            # TODO(qthequartermasterman): Since even when prompt embeds are  # TODO: 即使启用了提示嵌入
            # enabled, (a) not all requests will use prompt embeds, and (b)  # (a) 不是所有请求都使用提示嵌入
            # after the initial prompt is processed, the rest of the generated  # (b) 初始提示处理后，生成的token都是token id
            # tokens will be token ids, it is not desirable to have the  # 不建议始终将嵌入层放在CUDA图之外
            # embedding layer outside of the CUDA graph all the time. The v0  # v0引擎通过"双编译"CUDA图来避免此问题
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer  # 如果批次只有token id，将嵌入层包含在CUDA图中更高效
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (  # 获取需要转换为嵌入的token id的索引
                self.is_token_ids.gpu[:num_scheduled_tokens]  # 获取GPU上的token id掩码
                .nonzero(as_tuple=False)  # 找到非零元素的索引
                .squeeze(1)  # 压缩维度
            )
            # Some tokens ids may need to become embeds  # 某些token id可能需要转换为嵌入
            if token_ids_idx.numel() > 0:  # 如果有需要转换的token id
                token_ids = self.input_ids.gpu[token_ids_idx]  # 获取对应的token id
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)  # 将token id转换为嵌入
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds  # 将嵌入写入对应位置

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]  # 获取填充后的输入嵌入
            model_kwargs = self._init_model_kwargs()  # 初始化模型关键字参数
            input_ids = None  # 不使用token id作为输入
        else:
            # For text-only models, we use token ids as input.  # 对于纯文本模型，使用token id作为输入
            # While it is possible to use embeddings as input just like the  # 虽然可以像多模态模型一样使用嵌入作为输入
            # multimodal models, it is not desirable for performance since  # 但这对性能不利
            # then the embedding layer is not included in the CUDA graph.  # 因为嵌入层不会被包含在CUDA图中
            input_ids = self.input_ids.gpu[:num_input_tokens]  # 获取GPU上的输入token ID
            inputs_embeds = None  # 不使用输入嵌入
            model_kwargs = self._init_model_kwargs()  # 初始化模型关键字参数

        if self.uses_mrope:  # 如果使用mrope位置编码
            positions = self.mrope_positions.gpu[:, :num_input_tokens]  # 获取mrope位置编码（多维）
        elif self.uses_xdrope_dim > 0:  # 如果使用xdrope位置编码
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]  # 获取xdrope位置编码（多维）
        else:
            positions = self.positions.gpu[:num_input_tokens]  # 获取标准位置编码（一维）

        if is_first_rank:  # 如果是流水线并行的第一个rank
            intermediate_tensors = None  # 第一个rank不需要中间张量
        else:
            assert intermediate_tensors is not None  # 断言中间张量不为空
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(  # 同步和切片中间张量
                num_input_tokens, intermediate_tensors, True  # 传入token数、中间张量和同步标志
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:  # 如果是编码器-解码器模型且有调度的编码器输入
            # Run the encoder, just like we do with other multimodal inputs.  # 运行编码器，与其他多模态输入的处理方式相同
            # For an encoder-decoder model, our processing here is a bit  # 对于编码器-解码器模型，这里的处理更简单
            # simpler, because the outputs are just passed to the decoder.  # 因为输出直接传递给解码器
            # We are not doing any prompt replacement. We also will only  # 我们不做任何提示替换
            # ever have a single encoder input.  # 而且只会有一个编码器输入
            encoder_outputs = self._execute_mm_encoder(scheduler_output)  # 执行多模态编码器
            model_kwargs.update({"encoder_outputs": encoder_outputs})  # 将编码器输出添加到模型参数

        return (  # 返回预处理结果
            input_ids,  # 输入token ID
            inputs_embeds,  # 输入嵌入
            positions,  # 位置编码
            intermediate_tensors,  # 中间张量
            model_kwargs,  # 模型关键字参数
            ec_connector_output,  # EC连接器输出
        )

    def _sample(  # 执行采样
        self,
        logits: torch.Tensor | None,  # 模型输出的logits张量
        spec_decode_metadata: SpecDecodeMetadata | None,  # 推测解码元数据
    ) -> SamplerOutput:  # 返回采样器输出
        """对模型输出的logits进行采样，生成下一个token。支持常规采样和推测解码的拒绝采样。"""
        # Sample the next token and get logprobs if needed.  # 采样下一个token并在需要时获取logprobs
        sampling_metadata = self.input_batch.sampling_metadata  # 获取采样元数据
        # Update output token ids with tokens sampled in last step  # 用上一步采样的token更新输出token id
        # if async scheduling and required by current sampling params.  # 如果使用异步调度且当前采样参数需要
        self.input_batch.update_async_output_token_ids()  # 更新异步输出token ID
        if spec_decode_metadata is None:  # 如果没有推测解码元数据（常规采样）
            return self.sampler(  # 调用常规采样器
                logits=logits,  # 传入logits
                sampling_metadata=sampling_metadata,  # 传入采样元数据
            )

        # Update spec_token_ids with real draft tokens from pre step only when  # 仅在需要output_token_ids时（使用惩罚或禁用词）
        # output_token_ids is needed (penalties or bad_words are in use).  # 用预步骤的真实草稿token更新spec_token_ids
        if self.use_async_scheduling and self._draft_token_req_ids is not None:  # 如果使用异步调度且有草稿token请求ID
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()  # 获取CPU上的草稿token ID
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)  # 更新异步推测token ID

        sampler_output = self.rejection_sampler(  # 调用拒绝采样器
            spec_decode_metadata,  # 推测解码元数据
            None,  # draft_probs（草稿概率，此处为None）
            logits,  # 模型logits
            sampling_metadata,  # 采样元数据
        )
        return sampler_output  # 返回采样器输出

    def _bookkeeping_sync(  # 执行簿记同步操作
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出
        sampler_output: SamplerOutput,  # 采样器输出
        logits: torch.Tensor | None,  # logits张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        num_scheduled_tokens: int,  # 调度的token数量
        spec_decode_metadata: SpecDecodeMetadata | None,  # 推测解码元数据
    ) -> tuple[  # 返回多个簿记结果的元组
        dict[str, int],  # logits中NaN的数量字典
        LogprobsLists | None,  # logprobs列表
        list[list[int]],  # 有效的采样token ID列表
        dict[str, LogprobsTensors | None],  # 提示logprobs字典
        list[str],  # 请求ID列表副本
        dict[str, int],  # 请求ID到索引映射副本
        list[int],  # 无效的请求索引列表
    ]:
        """执行采样后的簿记操作：处理NaN检测、token缓存、logprobs计算等。"""
        num_nans_in_logits = {}  # 初始化logits中NaN数量的字典
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:  # 如果配置了计算logits中的NaN
            num_nans_in_logits = self._get_nans_in_logits(logits)  # 计算并获取NaN数量

        num_reqs = self.input_batch.num_reqs  # 获取批次中的请求数量
        discard_sampled_tokens_req_indices = np.nonzero(  # 找到需要丢弃采样token的请求索引
            self.discard_request_mask.np[:num_reqs]  # 从丢弃掩码中获取有效部分
        )[0]
        for i in discard_sampled_tokens_req_indices:  # 遍历需要丢弃的请求索引
            gen = self.input_batch.generators.get(int(i))  # 获取对应的随机数生成器
            if gen is not None:  # 如果生成器存在
                gen.set_offset(gen.get_offset() - 4)  # 回退生成器的偏移量（撤销采样消耗的随机数）

        # Copy some objects so they don't get modified after returning.  # 复制对象以防返回后被修改
        # This is important when using async scheduling.  # 这在使用异步调度时很重要
        req_ids_output_copy = self.input_batch.req_ids.copy()  # 复制请求ID列表
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()  # 复制请求ID到索引的映射

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]  # 获取采样token的数量
        sampled_token_ids = sampler_output.sampled_token_ids  # 获取采样的token ID张量
        logprobs_tensors = sampler_output.logprobs_tensors  # 获取logprobs张量
        invalid_req_indices = []  # 初始化无效请求索引列表
        logprobs_lists = None  # 初始化logprobs列表
        if not self.use_async_scheduling:  # 如果不使用异步调度（同步模式）
            # Get the valid generated tokens.  # 获取有效的生成token
            max_gen_len = sampled_token_ids.shape[-1]  # 获取最大生成长度
            if max_gen_len == 1:  # 如果最大生成长度为1（无推测解码token）
                # No spec decode tokens.  # 没有推测解码token
                valid_sampled_token_ids = self._to_list(sampled_token_ids)  # 将采样token ID转为列表
                # Mask out the sampled tokens that should not be sampled.  # 屏蔽不应该被采样的token
                for i in discard_sampled_tokens_req_indices:  # 遍历需要丢弃的请求索引
                    valid_sampled_token_ids[int(i)].clear()  # 清空对应请求的采样结果

                if logprobs_tensors is not None:  # 如果有logprobs张量
                    logprobs_lists = logprobs_tensors.tolists()  # 将logprobs张量转为列表
            else:
                # Includes spec decode tokens.  # 包含推测解码token
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(  # 解析拒绝采样的输出
                    sampled_token_ids,  # 采样的token ID
                    self.input_batch.vocab_size,  # 词汇表大小
                    discard_sampled_tokens_req_indices,  # 需要丢弃的请求索引
                    logprobs_tensors=logprobs_tensors,  # logprobs张量
                )
        else:
            valid_sampled_token_ids = []  # 异步模式下初始化为空列表
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()  # 将无效请求索引转为列表
            invalid_req_indices_set = set(invalid_req_indices)  # 转为集合以加速查找

            # Cache the sampled tokens on the GPU and avoid CPU sync.  # 将采样token缓存在GPU上避免CPU同步
            # These will be copied into input_ids in the next step  # 这些将在下一步准备输入时拷贝到input_ids
            # when preparing inputs.  # 准备输入时使用
            # With spec decoding, this is done in propose_draft_token_ids().  # 推测解码时在propose_draft_token_ids()中完成
            if self.input_batch.prev_sampled_token_ids is None:  # 如果没有之前的采样token ID
                assert sampled_token_ids.shape[-1] == 1  # 断言采样token维度为1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids  # 缓存采样token ID到GPU
            self.input_batch.prev_req_id_to_index = {  # 更新前一步的请求ID到索引映射
                req_id: i  # 请求ID映射到索引
                for i, req_id in enumerate(self.input_batch.req_ids)  # 遍历请求ID
                if i not in invalid_req_indices_set  # 排除无效请求
            }

        # Cache the sampled tokens in the model runner, so that the scheduler  # 在模型运行器中缓存采样token
        # doesn't need to send them back.  # 这样调度器不需要将它们发回
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends  # 注意：使用PP时例外，调度器会发回采样token
        # the sampled tokens back, because there's no direct communication  # 因为第一阶段worker和最后阶段worker之间没有直接通信
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids  # 获取请求ID列表
        for req_idx in range(num_sampled_tokens):  # 遍历所有采样token
            if self.use_async_scheduling:  # 如果使用异步调度
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None  # 有效请求用-1占位，无效请求设为None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]  # 获取同步模式下的有效采样token

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0  # 计算采样token数量

            if not sampled_ids:  # 如果没有采样token
                continue  # 跳过

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]  # 获取该请求的非推测token起始位置
            end_idx = start_idx + num_sampled_ids  # 计算结束位置
            assert end_idx <= self.max_model_len, (  # 断言不超过最大模型长度
                "Sampled token IDs exceed the max model length. "  # 采样token ID超出最大模型长度
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids  # 将采样token写入CPU token ID缓冲区
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True  # 标记这些位置为token ID（非嵌入）
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx  # 更新非推测token数量

            req_id = req_ids[req_idx]  # 获取请求ID
            req_state = self.requests[req_id]  # 获取请求状态
            req_state.output_token_ids.extend(sampled_ids)  # 将采样token添加到请求的输出token列表

        # Compute prompt logprobs if needed.  # 如果需要，计算提示logprobs
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(  # 获取提示logprobs字典
            hidden_states[:num_scheduled_tokens],  # 截取有效的隐藏状态
            scheduler_output.num_scheduled_tokens,  # 每个请求的调度token数
        )

        return (  # 返回簿记结果
            num_nans_in_logits,  # logits中NaN的数量
            logprobs_lists,  # logprobs列表
            valid_sampled_token_ids,  # 有效的采样token ID
            prompt_logprobs_dict,  # 提示logprobs字典
            req_ids_output_copy,  # 请求ID列表副本
            req_id_to_index_output_copy,  # 请求ID到索引映射副本
            invalid_req_indices,  # 无效请求索引列表
        )

    @contextmanager  # 上下文管理器装饰器
    def synchronize_input_prep(self):  # 同步输入准备的上下文管理器
        """同步输入准备操作，确保前一步完成后才开始使用CPU张量，防止异步调度中的数据竞争。"""
        if self.prepare_inputs_event is None:  # 如果没有输入准备事件
            yield  # 直接执行
            return  # 返回

        # Ensure prior step has finished with reused CPU tensors.  # 确保前一步已完成对复用CPU张量的使用
        # This is required in the async scheduling case because  # 在异步调度情况下这是必需的
        # the CPU->GPU transfer happens async.  # 因为CPU到GPU的传输是异步的
        self.prepare_inputs_event.synchronize()  # 同步等待事件完成
        try:
            yield  # 执行上下文中的代码
        finally:
            self.prepare_inputs_event.record()  # 记录事件（标记当前操作完成）

    def _model_forward(  # 模型前向传播辅助方法
        self,
        input_ids: torch.Tensor | None = None,  # 输入token ID（可选）
        positions: torch.Tensor | None = None,  # 位置编码（可选）
        intermediate_tensors: IntermediateTensors | None = None,  # 中间张量（可选）
        inputs_embeds: torch.Tensor | None = None,  # 输入嵌入（可选）
        **model_kwargs: dict[str, Any],  # 额外的模型参数
    ) -> Any:  # 返回模型输出
        """模型前向传播的辅助方法，可被子类重写以自定义模型执行逻辑。

        调用模型的forward方法，传入input_ids、positions、intermediate_tensors、
        inputs_embeds等参数。子类可重写此方法以检查模型执行而非整个execute_model。
        """
        return self.model(  # 调用模型的前向传播
            input_ids=input_ids,  # 传入输入token ID
            positions=positions,  # 传入位置编码
            intermediate_tensors=intermediate_tensors,  # 传入中间张量
            inputs_embeds=inputs_embeds,  # 传入输入嵌入
            **model_kwargs,  # 传入额外的模型参数
        )

    # 判断当前批次是否为"统一 decode"模式（所有请求的调度 token 数相同），
    # 这是 CUDA Graph 调度和注意力后端优化的关键判断条件。
    @staticmethod  # 静态方法装饰器
    def _is_uniform_decode(  # 判断是否为统一解码模式
        max_num_scheduled_tokens: int,  # 最大调度token数
        uniform_decode_query_len: int,  # 统一解码查询长度
        num_tokens: int,  # 总token数
        num_reqs: int,  # 请求数量
        force_uniform_decode: bool | None = None,  # 强制统一解码标志
    ) -> bool:  # 返回是否为统一解码
        """
        判断当前批次是否为统一解码模式，即所有请求具有相同的调度token数。
        这是CUDA Graph调度和注意力后端优化的关键判断条件。
        """
        return (  # 返回判断结果
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)  # 最大调度token数等于统一解码查询长度
                and (num_tokens == max_num_scheduled_tokens * num_reqs)  # 总token数等于最大调度token数乘以请求数
            )
            if force_uniform_decode is None  # 如果未强制指定
            else force_uniform_decode  # 否则使用强制指定的值
        )

    # 确定批次的执行策略：CUDA Graph 模式选择、padding 到匹配的 graph 大小、
    # 微批次切分决策，以及序列并行的 token padding。
    # 返回运行时 CUDA Graph 模式、批次描述符、padding 后的尺寸和微批次切片。
    def _determine_batch_execution_and_padding(  # 确定批次执行策略和padding
        self,
        num_tokens: int,  # 总token数
        num_reqs: int,  # 请求数量
        num_scheduled_tokens_np: np.ndarray,  # 每个请求的调度token数numpy数组
        max_num_scheduled_tokens: int,  # 最大调度token数
        use_cascade_attn: bool,  # 是否使用级联注意力
        allow_microbatching: bool = True,  # 是否允许微批次处理
        force_eager: bool = False,  # 是否强制使用eager模式（不使用CUDA图）
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will  # 用于CUDA图捕获，TODO: 重构CUDA图捕获方式
        # be improved in model runner v2)  # 将在模型运行器v2中改进
        force_uniform_decode: bool | None = None,  # 强制统一解码模式
        force_has_lora: bool | None = None,  # 强制LoRA状态
        force_num_active_loras: int | None = None,  # 强制活跃LoRA数量
        num_encoder_reqs: int = 0,  # 编码器请求数量
    ) -> tuple[  # 返回执行策略结果元组
        CUDAGraphMode,  # CUDA图模式
        BatchDescriptor,  # 批次描述符
        bool,  # 是否需要微批次
        torch.Tensor | None,  # 跨数据并行的token数张量
        CUDAGraphStat | None,  # CUDA图统计信息
    ]:
        """确定批次的CUDA图执行模式、padding策略、微批次切分和序列并行padding。"""
        uniform_decode = self._is_uniform_decode(  # 判断是否为统一解码模式
            max_num_scheduled_tokens=max_num_scheduled_tokens,  # 最大调度token数
            uniform_decode_query_len=self.uniform_decode_query_len,  # 统一解码查询长度
            num_tokens=num_tokens,  # 总token数
            num_reqs=num_reqs,  # 请求数量
            force_uniform_decode=force_uniform_decode,  # 强制统一解码标志
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output  # 编码器-解码器模型仅在decoder_step>0时支持CUDA图
        # is present). Also, chunked-prefill is disabled, so batch are uniform.  # 同时分块预填充被禁用，所以批次是统一的
        has_encoder_output = (  # 判断是否有编码器输出
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0  # 是编码器-解码器模型且有编码器请求
        )

        # Compute LoRA state for cudagraph dispatch  # 计算CUDA图调度的LoRA状态
        num_active_loras = (  # 获取活跃LoRA数量
            force_num_active_loras  # 如果强制指定则使用强制值
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)  # 否则从输入批次获取
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora  # 判断是否有LoRA

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)  # 为序列并行进行padding

        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):  # 定义CUDA图调度辅助函数
            return self.cudagraph_dispatcher.dispatch(  # 调用CUDA图调度器
                num_tokens=num_tokens,  # token数量
                has_lora=has_lora,  # 是否有LoRA
                uniform_decode=uniform_decode,  # 是否统一解码
                num_active_loras=num_active_loras,  # 活跃LoRA数量
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,  # 有效模式集合
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,  # 无效模式集合
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(  # 执行CUDA图调度
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output  # 级联注意力或有编码器输出时禁用FULL模式
        )
        num_tokens_padded = batch_descriptor.num_tokens  # 获取调度后的padding token数
        if self.compilation_config.pass_config.enable_sp:  # 如果启用了序列并行
            assert (  # 断言token数是TP大小的倍数
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "  # 序列并行要求token数是张量并行大小的倍数
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate  # 数据并行运行时需要额外的协调
        # across ranks  # 跨rank协调
        should_ubatch, num_tokens_across_dp = False, None  # 初始化微批次标志和跨DP的token数
        if self.vllm_config.parallel_config.data_parallel_size > 1:  # 如果数据并行大小大于1
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (  # 协调跨数据并行的批次
                coordinate_batch_across_dp(  # 调用跨DP批次协调函数
                    num_tokens_unpadded=num_tokens,  # 未padding的token数
                    parallel_config=self.parallel_config,  # 并行配置
                    allow_microbatching=allow_microbatching,  # 是否允许微批次
                    num_tokens_padded=num_tokens_padded,  # padding后的token数
                    uniform_decode=uniform_decode,  # 是否统一解码
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,  # 每请求调度token数
                    cudagraph_mode=cudagraph_mode.value,  # CUDA图模式值
                )
            )

            # Extract DP-synced values  # 提取DP同步后的值
            if num_tokens_across_dp is not None:  # 如果跨DP的token数不为空
                dp_rank = self.parallel_config.data_parallel_rank  # 获取数据并行rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())  # 获取当前rank的padding token数
                # Re-dispatch with DP padding so we have the correct batch_descriptor  # 使用DP padding重新调度以获取正确的批次描述符
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(  # 重新调度CUDA图
                    num_tokens_padded,  # 使用DP同步后的token数
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},  # 使用同步后的CUDA图模式
                )
                # Assert to make sure the agreed upon token count is correct otherwise  # 断言确保约定的token数正确
                # num_tokens_across_dp will no-longer be valid  # 否则num_tokens_across_dp将不再有效
                assert batch_descriptor.num_tokens == num_tokens_padded  # 断言描述符中的token数与padding数一致

        cudagraph_stats = None  # 初始化CUDA图统计
        if self.vllm_config.observability_config.cudagraph_metrics:  # 如果启用了CUDA图指标
            cudagraph_stats = CUDAGraphStat(  # 创建CUDA图统计对象
                num_unpadded_tokens=num_tokens,  # 未padding的token数
                num_padded_tokens=batch_descriptor.num_tokens,  # padding后的token数
                num_paddings=batch_descriptor.num_tokens - num_tokens,  # padding数量
                runtime_mode=str(cudagraph_mode),  # 运行时CUDA图模式
            )

        return (  # 返回执行策略结果
            cudagraph_mode,  # CUDA图执行模式
            batch_descriptor,  # 批次描述符
            should_ubatch,  # 是否需要微批次处理
            num_tokens_across_dp,  # 跨数据并行的token数量
            cudagraph_stats,  # CUDA图统计信息
        )  # 返回批次执行和填充参数的元组

    def _register_layerwise_nvtx_hooks(self) -> None:  # 注册逐层NVTX跟踪钩子的方法
        """
        注册逐层NVTX钩子。当启用 --enable-layerwise-nvtx-tracing 时，
        为模型中每一层或模块添加NVTX标记，用于追踪每一层的详细执行信息。
        """

        if (  # 检查是否满足注册NVTX钩子的条件
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing  # 检查是否启用了逐层NVTX跟踪配置
            and not self.layerwise_nvtx_hooks_registered  # 并且钩子尚未注册
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:  # 如果CUDA图模式不是NONE
                logger.debug_once(  # 输出一次性调试日志
                    "layerwise NVTX tracing is not supported when CUDA graph is "  # 提示CUDA图模式下不支持逐层NVTX跟踪
                    "turned off; you may observe part or all of the model "  # 可能会缺少部分或全部NVTX标记
                    "missing NVTX markers"  # 缺少NVTX标记的警告信息
                )

            # In STOCK_TORCH_COMPILE mode, after registering hooks here,
            # the __call__ function of nn.module will be recompiled with
            # fullgraph=True. Since nvtx.range_push/pop are not traceable
            # by torch dynamo, we can't register hook functions here
            # because hook functions will also be traced by torch dynamo.
            if (  # 检查编译模式是否为STOCK_TORCH_COMPILE
                self.vllm_config.compilation_config.mode  # 获取编译配置的模式
                == CompilationMode.STOCK_TORCH_COMPILE  # 与STOCK_TORCH_COMPILE模式比较
            ):
                logger.debug_once(  # 输出一次性调试日志
                    "layerwise NVTX tracing is not supported when "  # 提示STOCK_TORCH_COMPILE模式不支持逐层NVTX跟踪
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "  # 跳过钩子注册
                    "function hooks registration"  # 不注册函数钩子
                )
            else:  # 其他编译模式下正常注册钩子
                pyt_hooks = PytHooks()  # 创建PyTorch钩子实例
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)  # 为模型注册NVTX跟踪钩子
                self.layerwise_nvtx_hooks_registered = True  # 标记钩子已注册

    def _get_slot_mappings(  # 获取KV缓存槽位映射的方法
        self,
        num_tokens_padded: int,  # 填充后的token总数
        num_reqs_padded: int,  # 填充后的请求数
        num_tokens_unpadded: int,  # 未填充的实际token数
        ubatch_slices: "UBatchSlices | None" = None,  # 可选的微批次切片信息
    ) -> tuple[  # 返回类型为元组
        dict[int, torch.Tensor] | None,  # 按组ID索引的槽位映射字典或None
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,  # 按层名索引的槽位映射或其列表
    ]:
        """
        构建系统所需的两种格式的槽位映射。

        参数:
            num_tokens_padded: 填充后的token总数
            num_reqs_padded: 填充后的请求总数
            num_tokens_unpadded: 实际的token数量（未填充）
            ubatch_slices: 可选的DBO微批次切片信息

        返回:
            一个元组包含:
            - slot_mappings_by_gid: dict[int, torch.Tensor] 用于注意力元数据
            - slot_mappings_by_layer: dict[str, torch.Tensor] 或 list 用于前向上下文
        """
        if not (  # 检查是否具有有效的KV缓存配置
            hasattr(self, "kv_cache_config")  # 检查是否有kv_cache_config属性
            and self.kv_cache_config is not None  # 并且该配置不为None
            and len(self.kv_cache_config.kv_cache_groups) > 0  # 并且KV缓存组数量大于0
        ):
            return None, None  # 没有KV缓存配置时返回None

        def _get_slot_mapping(kv_cache_gid: int):  # 获取单个KV缓存组的槽位映射的内部函数
            """获取指定KV缓存组ID的槽位映射张量。"""
            assert num_reqs_padded is not None and num_tokens_padded is not None  # 断言填充参数不为None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[  # 获取指定组的KV缓存规格
                kv_cache_gid  # 使用组ID索引
            ].kv_cache_spec  # 获取KV缓存规格对象
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):  # 如果是仅编码器注意力规格
                slot_mapping = torch.zeros(  # 创建全零的槽位映射张量
                    (num_tokens_padded,),  # 形状为填充后的token数
                    dtype=torch.int64,  # 数据类型为64位整数
                    device=self.device,  # 放在当前设备上
                )
            else:  # 非编码器注意力规格的情况
                blk_table = self.input_batch.block_table[kv_cache_gid]  # 获取该组的块表
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]  # 从GPU块表中截取填充长度的槽位映射

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)  # 将未使用的槽位填充为-1

            return slot_mapping  # 返回构建好的槽位映射

        slot_mappings_by_gid = {  # 构建按组ID索引的槽位映射字典
            gid: _get_slot_mapping(gid)  # 为每个组生成对应的槽位映射
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)  # 遍历所有KV缓存组
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}  # 初始化按层名索引的槽位映射字典
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):  # 遍历所有KV缓存组
            slot_mapping = slot_mappings_by_gid[gid]  # 获取该组的槽位映射
            for layer_name in kv_cache_group.layer_names:  # 遍历该组中的所有层名
                slot_mappings_by_layer[layer_name] = slot_mapping  # 将槽位映射关联到层名

        if ubatch_slices is not None:  # 如果提供了微批次切片信息
            result: list[dict[str, torch.Tensor]] = []  # 初始化结果列表
            for ubatch in ubatch_slices:  # 遍历每个微批次切片
                sliced_mappings: dict[str, torch.Tensor] = {}  # 初始化当前切片的映射字典
                for layer_name, slot_mapping in slot_mappings_by_layer.items():  # 遍历所有层的映射
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]  # 按token切片截取槽位映射
                result.append(sliced_mappings)  # 将切片结果添加到列表
            return slot_mappings_by_gid, result  # 返回按组和按微批次切片的映射

        return slot_mappings_by_gid, slot_mappings_by_layer  # 返回按组和按层的槽位映射

    # 模型前向执行的主入口。流程：
    # 1. 更新持久化批次状态（_update_states）
    # 2. 准备输入张量（_prepare_inputs）
    # 3. 执行多模态编码器和嵌入收集
    # 4. 确定 CUDA Graph 调度策略和 padding
    # 5. 构建注意力元数据
    # 6. 执行模型前向传播（_model_forward）
    # 7. 计算 logits
    # 在异步调度模式下返回 None，将状态缓存到 execute_model_state 供 sample_tokens 使用。
    # 在流水线并行的非末级 rank 返回 IntermediateTensors。
    @torch.inference_mode()  # 使用推理模式装饰器，禁用梯度计算以节省内存
    def execute_model(  # 模型执行的主入口方法
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出，包含本次调度的请求和token信息
        intermediate_tensors: IntermediateTensors | None = None,  # 流水线并行中间张量，非首阶段时不为None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:  # 返回模型运行输出或中间张量
        """
        模型前向执行的主入口函数。负责从调度器输出中准备输入、执行前向传播、
        计算logits，并将状态缓存到execute_model_state供后续采样使用。
        """
        if self.execute_model_state is not None:  # 检查上一次执行的状态是否已被消费
            raise RuntimeError(  # 如果未消费则抛出运行时错误
                "State error: sample_tokens() must be called "  # 错误提示：必须先调用sample_tokens
                "after execute_model() returns None."  # 在execute_model返回None之后
            )

        if self.routed_experts_initialized:  # 如果路由专家已初始化（MoE模型）
            capturer = RoutedExpertsCapturer.get_instance()  # 获取路由专家捕获器的单例实例
            if capturer is not None:  # 如果捕获器存在
                capturer.clear_buffer()  # noqa  # 清除捕获器缓冲区
            else:  # 如果捕获器不存在
                logger.error("RoutedExpertsCapturer not initialized.")  # 记录错误日志

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (  # 检查是否使用GPU版N-gram投机解码
            self.speculative_config is not None  # 投机解码配置不为空
            and self.speculative_config.use_ngram_gpu()  # 并且使用GPU N-gram方法
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()  # 浅拷贝调度token数字典
            spec_decode_tokens_copy = (  # 拷贝投机解码token信息
                scheduler_output.scheduled_spec_decode_tokens.copy()  # 浅拷贝投机解码token字典
            )
            scheduler_output = replace(  # 使用dataclass的replace创建新的调度器输出副本
                scheduler_output,  # 基于原始调度器输出
                num_scheduled_tokens=num_scheduled_tokens_copy,  # 替换为拷贝的调度token数
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,  # 替换为拷贝的投机解码token
            )

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():  # 如果有被抢占的请求且存在KV传输组
            get_kv_transfer_group().handle_preemptions(  # 处理KV传输组中的抢占事件
                scheduler_output.preempted_req_ids  # 传入被抢占的请求ID列表
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取本次调度的总token数
        with (  # 进入预处理上下文
            record_function_or_nullcontext("gpu_model_runner: preprocess"),  # 记录预处理性能分析区间
            self.synchronize_input_prep(),  # 同步输入准备操作
        ):
            # Update persistent batch states.
            self._update_states(scheduler_output)  # 更新持久化批次状态（添加/移除/恢复请求）

            if has_ec_transfer() and not get_ec_transfer().is_consumer:  # 如果有EC传输且当前节点不是消费者
                with self.maybe_get_ec_connector_output(  # 获取EC连接器输出上下文
                    scheduler_output,  # 传入调度器输出
                    encoder_cache=self.encoder_cache,  # 传入编码器缓存
                ) as ec_connector_output:  # 作为EC连接器输出
                    self._execute_mm_encoder(scheduler_output)  # 执行多模态编码器
                    return make_empty_encoder_model_runner_output(scheduler_output)  # 返回空的编码器模型运行输出

            if not num_scheduled_tokens:  # 如果没有调度任何token
                if (  # 检查是否为外部启动器+数据并行的特殊情况
                    self.parallel_config.distributed_executor_backend  # 获取分布式执行器后端
                    == "external_launcher"  # 是否为外部启动器模式
                    and self.parallel_config.data_parallel_size > 1  # 并且数据并行度大于1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)  # 执行虚拟运行以确保DP同步
                if not has_kv_transfer_group():  # 如果没有KV传输组
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT  # 返回空的模型运行输出
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)  # 无前向传播的KV连接器处理

            if self.cache_config.kv_sharing_fast_prefill:  # 如果启用了KV共享快速预填充
                assert not self.num_prompt_logprobs, (  # 断言不能同时使用prompt logprobs
                    "--kv-sharing-fast-prefill produces incorrect "  # 快速预填充会产生不正确的
                    "logprobs for prompt tokens, tokens, please disable "  # prompt token的logprobs
                    "it when the requests need prompt logprobs"  # 需要prompt logprobs时请禁用
                )

            num_reqs = self.input_batch.num_reqs  # 获取当前批次中的请求数
            req_ids = self.input_batch.req_ids  # 获取当前批次中的请求ID列表
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]  # 获取每个请求的调度token数
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)  # 转为numpy数组
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())  # 获取单个请求的最大调度token数
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens  # 获取未填充的总token数

            logits_indices, spec_decode_metadata = self._prepare_inputs(  # 准备输入张量并获取logits索引
                scheduler_output,  # 传入调度器输出
                num_scheduled_tokens_np,  # 传入每个请求的调度token数
            )

            cascade_attn_prefix_lens = None  # 初始化级联注意力前缀长度为None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:  # 如果启用级联注意力且未使用微批处理
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(  # 预计算级联注意力前缀长度
                    num_scheduled_tokens_np,  # 每个请求的调度token数
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],  # 每个请求已计算的token数
                    scheduler_output.num_common_prefix_blocks,  # 公共前缀块数量
                )

            (  # 解包批次执行和填充参数
                cudagraph_mode,  # CUDA图执行模式
                batch_desc,  # 批次描述符
                should_ubatch,  # 是否应该微批处理
                num_tokens_across_dp,  # 跨数据并行的token数
                cudagraph_stats,  # CUDA图统计信息
            ) = self._determine_batch_execution_and_padding(  # 确定批次执行策略和填充方案
                num_tokens=num_tokens_unpadded,  # 未填充的token数
                num_reqs=num_reqs,  # 请求数
                num_scheduled_tokens_np=num_scheduled_tokens_np,  # 每个请求的调度token数
                max_num_scheduled_tokens=max_num_scheduled_tokens,  # 最大调度token数
                use_cascade_attn=cascade_attn_prefix_lens is not None,  # 是否使用级联注意力
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),  # 编码器请求数
            )

            logger.debug(  # 输出调试日志记录批次执行参数
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "  # 日志格式字符串
                "should_ubatch: %s, num_tokens_across_dp: %s",  # 包含微批次和DP信息
                cudagraph_mode,  # CUDA图模式值
                batch_desc,  # 批次描述符值
                should_ubatch,  # 微批次标志值
                num_tokens_across_dp,  # 跨DP的token数值
            )

            num_tokens_padded = batch_desc.num_tokens  # 获取填充后的token数量
            num_reqs_padded = (  # 获取填充后的请求数量
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs  # 如果批次描述符有值则使用，否则用实际值
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(  # 创建微批次切片
                should_ubatch,  # 是否需要微批处理
                num_scheduled_tokens_np,  # 每个请求的调度token数
                num_tokens_padded,  # 填充后的token数
                num_reqs_padded,  # 填充后的请求数
                self.parallel_config.num_ubatches,  # 微批次数量
            )

            logger.debug(  # 输出调试日志记录微批次切片信息
                "ubatch_slices: %s, ubatch_slices_padded: %s",  # 日志格式字符串
                ubatch_slices,  # 微批次切片值
                ubatch_slices_padded,  # 填充后的微批次切片值
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(  # 检查是否有注意力后端需要单独更新KV缓存
                all(  # 检查组内所有后端
                    g.backend.forward_includes_kv_cache_update  # 前向传播是否包含KV缓存更新
                    for g in self.attn_groups[id]  # 遍历该组的所有注意力组
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)  # 遍历所有KV缓存组
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)  # 排除仅编码器注意力
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL  # 完全CUDA图模式下需要填充注意力

            if self.cache_config.mamba_cache_mode == "align":  # 如果Mamba缓存模式为对齐模式
                mamba_utils.preprocess_mamba(  # 预处理Mamba状态
                    scheduler_output,  # 调度器输出
                    self.kv_cache_config,  # KV缓存配置
                    self.cache_config,  # 缓存配置
                    self.mamba_state_idx,  # Mamba状态索引
                    self.input_batch,  # 输入批次
                    self.requests,  # 请求字典
                    self.compilation_config.static_forward_context,  # 静态前向上下文
                    self.model.get_mamba_state_copy_func(),  # 获取Mamba状态拷贝函数
                    self._get_mamba_copy_bufs(),  # 获取Mamba拷贝缓冲区
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0  # 判断是否使用投机解码
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices  # 根据是否填充选择注意力用的微批次切片

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(  # 获取槽位映射
                num_tokens_padded=num_tokens_padded  # 填充后的token数
                if pad_attn or has_separate_kv_update  # 如果需要填充注意力或单独KV更新
                else num_tokens_unpadded,  # 否则使用未填充的token数
                num_reqs_padded=(  # 填充后的请求数
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs  # 根据条件选择填充或实际请求数
                ),
                num_tokens_unpadded=num_tokens_unpadded,  # 未填充的token数
                ubatch_slices=ubatch_slices_padded,  # 填充后的微批次切片
            )

            attn_metadata, spec_decode_common_attn_metadata = (  # 构建注意力元数据
                self._build_attention_metadata(  # 调用注意力元数据构建方法
                    num_tokens=num_tokens_unpadded,  # 未填充的token数
                    num_tokens_padded=num_tokens_padded if pad_attn else None,  # 填充后的token数（如果需要）
                    num_reqs=num_reqs,  # 请求数
                    num_reqs_padded=num_reqs_padded if pad_attn else None,  # 填充后的请求数（如果需要）
                    max_query_len=max_num_scheduled_tokens,  # 最大查询长度
                    ubatch_slices=ubatch_slices_attn,  # 注意力用的微批次切片
                    logits_indices=logits_indices,  # logits索引
                    use_spec_decode=use_spec_decode,  # 是否使用投机解码
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,  # 调度token数字典
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,  # 级联注意力前缀长度
                    slot_mappings=slot_mappings_by_group,  # 按组的槽位映射
                )
            )

            (  # 解包预处理结果
                input_ids,  # 输入token ID张量
                inputs_embeds,  # 输入嵌入张量
                positions,  # 位置编码张量
                intermediate_tensors,  # 中间张量（流水线并行）
                model_kwargs,  # 模型额外关键字参数
                ec_connector_output,  # EC连接器输出
            ) = self._preprocess(  # 执行预处理
                scheduler_output, num_tokens_padded, intermediate_tensors  # 传入调度器输出、填充token数和中间张量
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # 如果需要计算KV缩放因子
            cudagraph_mode = CUDAGraphMode.NONE  # 禁用CUDA图模式（KV缩放计算包含动态操作）
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # 标记KV缩放因子已计算完成

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)  # 获取编码器请求数量
        has_encoder_input = (  # 判断是否有编码器输入
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0  # 是编码器-解码器模型且有编码器请求
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None  # 投机解码时延迟KV连接器最终化
        with (  # 进入模型前向传播的上下文管理器
            set_forward_context(  # 设置前向传播上下文
                attn_metadata,  # 注意力元数据
                self.vllm_config,  # vLLM配置
                num_tokens=num_tokens_padded,  # 填充后的token数
                num_tokens_across_dp=num_tokens_across_dp,  # 跨DP的token数
                cudagraph_runtime_mode=cudagraph_mode,  # CUDA图运行时模式
                batch_descriptor=batch_desc,  # 批次描述符
                ubatch_slices=ubatch_slices_padded,  # 填充后的微批次切片
                slot_mapping=slot_mappings,  # 槽位映射
                skip_compiled=has_encoder_input,  # 有编码器输入时跳过编译
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),  # 记录前向传播性能分析区间
            self.maybe_get_kv_connector_output(  # 获取KV连接器输出
                scheduler_output,  # 调度器输出
                defer_finalize=defer_kv_connector_finalize,  # 是否延迟最终化
            ) as kv_connector_output,  # 作为KV连接器输出
        ):
            model_output = self._model_forward(  # 执行模型前向传播
                input_ids=input_ids,  # 输入token ID
                positions=positions,  # 位置编码
                intermediate_tensors=intermediate_tensors,  # 中间张量
                inputs_embeds=inputs_embeds,  # 输入嵌入
                **model_kwargs,  # 其他模型参数
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):  # 进入后处理性能分析区间
            if self.use_aux_hidden_state_outputs:  # 如果使用辅助隐藏状态输出（如EAGLE 3）
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output  # 解包主隐藏状态和辅助隐藏状态
            else:  # 通常情况
                # Common case.
                hidden_states = model_output  # 模型输出即为隐藏状态
                aux_hidden_states = None  # 无辅助隐藏状态

            if not self.broadcast_pp_output:  # 如果不广播流水线并行输出（常见情况）
                # Common case.
                if not get_pp_group().is_last_rank:  # 如果不是流水线并行的最后一个rank
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)  # 断言隐藏状态是中间张量类型
                    hidden_states.kv_connector_output = kv_connector_output  # 附加KV连接器输出
                    self.kv_connector_output = kv_connector_output  # 保存KV连接器输出到实例
                    return hidden_states  # 返回中间张量给下一个流水线阶段

                if self.is_pooling_model:  # 如果是池化模型（如嵌入模型）
                    # Return the pooling output.
                    return self._pool(  # 执行池化操作并返回
                        hidden_states,  # 隐藏状态
                        num_scheduled_tokens,  # 调度的token数
                        num_scheduled_tokens_np,  # numpy格式的调度token数
                        kv_connector_output,  # KV连接器输出
                    )

                sample_hidden_states = hidden_states[logits_indices]  # 按logits索引提取用于采样的隐藏状态
                logits = self.model.compute_logits(sample_hidden_states)  # 计算logits（词汇表上的分数）
            else:  # 广播PP输出的情况（较少见）
                # Rare case.
                assert not self.is_pooling_model  # 断言不是池化模型

                sample_hidden_states = hidden_states[logits_indices]  # 按logits索引提取采样隐藏状态
                if not get_pp_group().is_last_rank:  # 如果不是最后一个PP rank
                    all_gather_tensors = {  # 需要全局收集的张量字典
                        "residual": not is_residual_scattered_for_sp(  # 检查残差是否已为序列并行分散
                            self.vllm_config, num_tokens_padded  # 传入配置和填充token数
                        )
                    }
                    get_pp_group().send_tensor_dict(  # 发送张量字典给最后一个rank
                        hidden_states.tensors,  # 中间张量
                        all_gather_group=get_tp_group(),  # 使用TP组进行全局收集
                        all_gather_tensors=all_gather_tensors,  # 需要全局收集的张量
                    )
                    logits = None  # 非最后rank不计算logits
                else:  # 最后一个PP rank
                    logits = self.model.compute_logits(sample_hidden_states)  # 计算logits

                model_output_broadcast_data: dict[str, Any] = {}  # 初始化广播数据字典
                if logits is not None:  # 如果有logits
                    model_output_broadcast_data["logits"] = logits.contiguous()  # 将logits添加到广播数据

                broadcasted = get_pp_group().broadcast_tensor_dict(  # 广播张量字典到所有PP rank
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1  # 从最后一个rank广播
                )
                assert broadcasted is not None  # 断言广播结果不为None
                logits = broadcasted["logits"]  # 从广播结果中获取logits

        self.execute_model_state = ExecuteModelState(  # 缓存执行模型状态，供sample_tokens使用
            scheduler_output,  # 调度器输出
            logits,  # 计算得到的logits
            spec_decode_metadata,  # 投机解码元数据
            spec_decode_common_attn_metadata,  # 投机解码通用注意力元数据
            hidden_states,  # 隐藏状态
            sample_hidden_states,  # 用于采样的隐藏状态
            aux_hidden_states,  # 辅助隐藏状态
            ec_connector_output,  # EC连接器输出
            cudagraph_stats,  # CUDA图统计信息
            slot_mappings,  # 槽位映射
        )
        self.kv_connector_output = kv_connector_output  # 保存KV连接器输出
        return None  # 返回None表示需要调用sample_tokens完成采样

    # 采样步骤：从 execute_model 缓存的 logits 中采样 token。
    # 处理结构化输出的 grammar bitmask 应用、投机解码的拒绝采样、
    # prompt logprobs 计算、异步输出拷贝、以及草稿 token 的生成。
    # 返回 ModelRunnerOutput（包含采样 token ID、logprobs、KV 连接器输出等）。
    @torch.inference_mode  # 推理模式装饰器（注意：此处是函数引用而非调用）
    def sample_tokens(  # 从缓存的logits中采样token的方法
        self, grammar_output: "GrammarOutput | None"  # 语法输出（结构化输出的bitmask）
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:  # 返回模型运行输出
        """
        采样步骤：从execute_model缓存的logits中采样token。
        处理结构化输出bitmask应用、投机解码拒绝采样、prompt logprobs计算等。
        """
        if self.execute_model_state is None:  # 如果没有缓存的执行状态
            kv_connector_output = self.kv_connector_output  # 获取KV连接器输出
            self.kv_connector_output = None  # 清除KV连接器输出引用
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and get_pp_group().world_size > 1:  # 异步调度且多PP阶段时
                self._pp_receive_prev_sampled_token_ids_to_input_batch()  # 从最后一个PP rank接收采样token ID
            if not kv_connector_output:  # 如果没有KV连接器输出
                return None  # type: ignore[return-value]  # 返回None

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():  # 如果KV连接器输出为空
                return EMPTY_MODEL_RUNNER_OUTPUT  # 返回空的模型运行输出

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)  # 浅拷贝空的模型运行输出
            output.kv_connector_output = kv_connector_output  # 附加KV连接器输出
            return output  # 返回带KV连接器输出的结果

        # Unpack ephemeral state.
        (  # 解包临时状态
            scheduler_output,  # 调度器输出
            logits,  # logits张量
            spec_decode_metadata,  # 投机解码元数据
            spec_decode_common_attn_metadata,  # 投机解码通用注意力元数据
            hidden_states,  # 隐藏状态
            sample_hidden_states,  # 用于采样的隐藏状态
            aux_hidden_states,  # 辅助隐藏状态
            ec_connector_output,  # EC连接器输出
            cudagraph_stats,  # CUDA图统计信息
            slot_mappings,  # 槽位映射
        ) = self.execute_model_state  # 从缓存的执行状态中解包
        # Clear ephemeral state.
        self.execute_model_state = None  # 清除临时执行状态

        # Apply structured output bitmasks if present.
        if grammar_output is not None:  # 如果有结构化输出的语法约束
            apply_grammar_bitmask(  # 应用语法bitmask到logits上
                scheduler_output, grammar_output, self.input_batch, logits  # 传入调度输出、语法输出、批次和logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):  # 进入采样性能分析区间
            sampler_output = self._sample(logits, spec_decode_metadata)  # 执行采样操作

        self._update_states_after_model_execute(  # 模型执行后更新状态
            sampler_output.sampled_token_ids, scheduler_output  # 传入采样的token ID和调度器输出
        )
        if self.use_async_scheduling:  # 如果使用异步调度
            pp = get_pp_group()  # 获取流水线并行组
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:  # 非广播模式、多PP阶段、最后rank
                self._pp_broadcast_prev_sampled_token_ids(  # 广播上一步采样的token ID
                    sampler_output.sampled_token_ids  # 传入采样的token ID张量
                )

        self._draft_token_ids = None  # 清除草稿token ID
        self._draft_token_req_ids = None  # 清除草稿token对应的请求ID
        self.input_batch.prev_sampled_token_ids = None  # 清除上一步采样的token ID

        def propose_draft_token_ids(sampled_token_ids):  # 定义提议草稿token ID的内部函数
            """提议草稿token ID用于投机解码。"""
            assert spec_decode_common_attn_metadata is not None  # 断言投机解码注意力元数据存在
            with record_function_or_nullcontext("gpu_model_runner: draft"):  # 进入草稿生成性能分析区间
                self._draft_token_ids = self.propose_draft_token_ids(  # 调用草稿token提议方法
                    scheduler_output,  # 调度器输出
                    sampled_token_ids,  # 主模型采样的token ID
                    self.input_batch.sampling_metadata,  # 采样元数据
                    hidden_states,  # 隐藏状态
                    sample_hidden_states,  # 用于采样的隐藏状态
                    aux_hidden_states,  # 辅助隐藏状态
                    spec_decode_metadata,  # 投机解码元数据
                    spec_decode_common_attn_metadata,  # 投机解码通用注意力元数据
                    slot_mappings,  # 槽位映射
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)  # 将草稿token ID异步拷贝到CPU

        spec_config = self.speculative_config  # 获取投机解码配置
        propose_drafts_after_bookkeeping = False  # 初始化标志：是否在簿记后提议草稿
        if spec_config is not None:  # 如果启用了投机解码
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (  # 检查输入是否适合草稿模型
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens  # 最大序列长度加投机token数
                <= self.effective_drafter_max_model_len  # 不超过草稿模型的有效最大长度
            )
            use_gpu_toks = (  # 检查是否可以使用GPU上的采样token
                spec_config.use_eagle()  # EAGLE方法
                or spec_config.uses_draft_model()  # 草稿模型方法
                or spec_config.uses_extract_hidden_states()  # 隐藏状态提取方法
            ) and not spec_config.disable_padded_drafter_batch  # 且未禁用填充草稿批次
            if use_gpu_toks:  # 如果使用GPU token
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(  # 断言草稿器是正确的类型
                    self.drafter,  # 草稿器实例
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,  # 支持的草稿器类型
                )
                sampled_token_ids = sampler_output.sampled_token_ids  # 获取GPU上的采样token ID
                if input_fits_in_drafter:  # 如果输入适合草稿模型
                    propose_draft_token_ids(sampled_token_ids)  # 立即提议草稿token
                elif self.valid_sampled_token_count_event is not None:  # 如果有有效采样计数事件
                    assert spec_decode_common_attn_metadata is not None  # 断言元数据存在
                    next_token_ids, valid_sampled_tokens_count = (  # 准备填充的下一个token ID
                        self.drafter.prepare_next_token_ids_padded(  # 调用草稿器的填充准备方法
                            spec_decode_common_attn_metadata,  # 投机解码注意力元数据
                            sampled_token_ids,  # 采样的token ID
                            self.requests,  # 请求字典
                            self.input_batch,  # 输入批次
                            self.discard_request_mask.gpu,  # GPU上的丢弃请求掩码
                        )
                    )
                    self._copy_valid_sampled_token_count(  # 异步拷贝有效采样计数到CPU
                        next_token_ids, valid_sampled_tokens_count  # 传入token ID和计数
                    )
                    self._draft_token_ids = torch.zeros(  # 创建全零的草稿token ID张量
                        1, device=self.device, dtype=torch.int32  # 形状为[1]，后续expand
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)  # 扩展为[num_reqs, num_spec_tokens]
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)  # 拷贝零值草稿token到CPU
            elif (  # 否则检查是否使用GPU N-gram方法
                spec_config.use_ngram_gpu()  # 使用GPU N-gram
                and not spec_config.disable_padded_drafter_batch  # 且未禁用填充草稿批次
            ):
                assert isinstance(self.drafter, NgramProposerGPU)  # 断言草稿器是GPU N-gram类型
                sampled_token_ids = sampler_output.sampled_token_ids  # 获取GPU上的采样token ID
                if input_fits_in_drafter:  # 如果输入适合草稿模型
                    propose_draft_token_ids(sampled_token_ids)  # 立即提议草稿token
                elif self.valid_sampled_token_count_event is not None:  # 如果有有效采样计数事件
                    assert spec_decode_common_attn_metadata is not None  # 断言元数据存在
                    next_token_ids, valid_sampled_tokens_count, _ = (  # 更新GPU N-gram的token ID
                        self.drafter.update_token_ids_ngram(  # 调用N-gram token更新方法
                            sampled_token_ids,  # 采样的token ID
                            self.input_batch,  # 输入批次
                            self.token_ids_gpu_tensor,  # GPU上的token ID张量
                            self.num_tokens_no_spec_gpu,  # GPU上的非投机token数
                            self.discard_request_mask.gpu,  # GPU上的丢弃请求掩码
                        )
                    )
                    self._copy_valid_sampled_token_count(  # 异步拷贝有效采样计数到CPU
                        next_token_ids, valid_sampled_tokens_count  # 传入token ID和计数
                    )
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(  # 创建全零的草稿token ID张量
                        1, device=self.device, dtype=torch.int32  # 形状为[1]，后续expand
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)  # 扩展为[num_reqs, num_spec_tokens]
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)  # 拷贝零值草稿token到CPU
            else:  # 其他投机解码方法（如CPU N-gram、suffix等）
                propose_drafts_after_bookkeeping = input_fits_in_drafter  # 在簿记后提议草稿

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):  # 进入簿记性能分析区间
            (  # 解包簿记同步结果
                num_nans_in_logits,  # logits中的NaN数量
                logprobs_lists,  # logprobs列表
                valid_sampled_token_ids,  # 有效的采样token ID（CPU列表）
                prompt_logprobs_dict,  # prompt logprobs字典
                req_ids_output_copy,  # 输出用的请求ID拷贝
                req_id_to_index_output_copy,  # 输出用的请求ID到索引映射拷贝
                invalid_req_indices,  # 无效请求索引
            ) = self._bookkeeping_sync(  # 执行簿记同步操作
                scheduler_output,  # 调度器输出
                sampler_output,  # 采样器输出
                logits,  # logits张量
                hidden_states,  # 隐藏状态
                scheduler_output.total_num_scheduled_tokens,  # 总调度token数
                spec_decode_metadata,  # 投机解码元数据
            )

        if propose_drafts_after_bookkeeping:  # 如果需要在簿记后提议草稿
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)  # 使用CPU上的有效token ID提议草稿

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if spec_config is not None:  # 如果启用了投机解码
            self.finalize_kv_connector()  # 最终化KV连接器（等待保存完成并清除元数据）

        with record_function_or_nullcontext("gpu_model_runner: eplb"):  # 进入EPLB（专家并行负载均衡）性能分析区间
            self.eplb_step()  # 执行EPLB步骤

        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output  # 获取可能在草稿阶段修改过的KV连接器输出
        self.kv_connector_output = None  # 清除KV连接器输出引用

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):  # 进入构建输出性能分析区间
            if self.routed_experts_initialized:  # 如果路由专家已初始化
                capturer = RoutedExpertsCapturer.get_instance()  # 获取路由专家捕获器实例
                if capturer is not None:  # 如果捕获器存在
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa  # 保存捕获的专家信息
                else:  # 捕获器不存在
                    logger.error("RoutedExpertsCapturer not initialized.")  # 记录错误日志

            output = ModelRunnerOutput(  # 构建模型运行输出对象
                req_ids=req_ids_output_copy,  # 请求ID列表
                req_id_to_index=req_id_to_index_output_copy,  # 请求ID到索引的映射
                sampled_token_ids=valid_sampled_token_ids,  # 有效的采样token ID
                logprobs=logprobs_lists,  # logprobs列表
                prompt_logprobs_dict=prompt_logprobs_dict,  # prompt logprobs字典
                kv_connector_output=kv_connector_output,  # KV连接器输出
                ec_connector_output=ec_connector_output  # EC连接器输出（仅多模态模型）
                if self.supports_mm_inputs  # 如果支持多模态输入
                else None,  # 否则为None
                num_nans_in_logits=num_nans_in_logits,  # logits中的NaN数量
                cudagraph_stats=cudagraph_stats,  # CUDA图统计信息
            )

        if not self.use_async_scheduling:  # 如果不使用异步调度
            return output  # 直接返回同步输出

        with record_function_or_nullcontext(  # 进入异步输出构建性能分析区间
            "gpu_model_runner: AsyncGPUModelRunnerOutput"  # 性能分析标签
        ):
            async_output = AsyncGPUModelRunnerOutput(  # 构建异步GPU模型运行输出
                model_runner_output=output,  # 基础模型运行输出
                sampled_token_ids=sampler_output.sampled_token_ids,  # GPU上的采样token ID
                logprobs_tensors=sampler_output.logprobs_tensors,  # logprobs张量
                invalid_req_indices=invalid_req_indices,  # 无效请求索引
                async_output_copy_stream=self.async_output_copy_stream,  # 异步输出拷贝流
                vocab_size=self.input_batch.vocab_size,  # 词汇表大小
            )
        with record_function_or_nullcontext(  # 进入异步采样token设置性能分析区间
            "gpu_model_runner: set_async_sampled_token_ids"  # 性能分析标签
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(  # 设置异步采样token ID
                async_output.sampled_token_ids_cpu,  # CPU上的采样token ID
                async_output.async_copy_ready_event,  # 异步拷贝就绪事件
            )

        return async_output  # 返回异步模型运行输出

    def _pp_broadcast_prev_sampled_token_ids(  # 广播上一步采样token ID的方法
        self, sampled_token_ids: torch.Tensor  # 采样的token ID张量
    ) -> None:  # 无返回值
        """从最后一个PP阶段广播采样的token ID（GPU张量）到所有PP阶段。"""
        pp = get_pp_group()  # 获取流水线并行组
        assert pp.is_last_rank  # 断言当前是最后一个PP rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (  # 断言形状为[num_reqs, 1]
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"  # 错误提示信息
        )
        torch.distributed.broadcast(  # 使用PyTorch分布式广播
            sampled_token_ids, src=pp.rank, group=pp.device_group  # 从当前rank广播到PP设备组
        )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:  # 接收上一步采样token ID的方法
        """从最后一个PP阶段接收广播的采样token ID。"""
        pp = get_pp_group()  # 获取流水线并行组
        assert not pp.is_last_rank  # 断言当前不是最后一个PP rank
        num_reqs = self.input_batch.num_reqs  # 获取当前批次请求数
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)  # 创建接收缓冲区张量
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)  # 从最后一个rank接收广播
        self.input_batch.prev_sampled_token_ids = recv  # 保存接收到的采样token ID

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]  # 获取需要丢弃的请求索引
        discard_req_indices_set = set(discard_req_indices)  # 转换为集合以加速查找
        prev_req_id_to_index: dict[str, int] = {}  # 初始化请求ID到索引的映射字典
        for i, req_id in enumerate(self.input_batch.req_ids):  # 遍历批次中的所有请求
            if i in discard_req_indices_set:  # 如果该请求在丢弃集合中
                continue  # 跳过丢弃的请求
            prev_req_id_to_index[req_id] = i  # 建立请求ID到索引的映射
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:  # 如果请求状态存在
                req_state.output_token_ids.append(-1)  # 追加占位符token ID以推进输出长度
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index  # 保存映射到输入批次

    def take_draft_token_ids(self) -> DraftTokenIds | None:  # 获取并消费草稿token ID的方法
        """获取草稿token ID，如果没有投机token或草稿请求ID则返回None。"""
        if not self.num_spec_tokens or not self._draft_token_req_ids:  # 如果没有投机token数或草稿请求ID
            return None  # 返回None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()  # 获取CPU上的草稿token ID
        return DraftTokenIds(req_ids, draft_token_ids)  # 返回包含请求ID和草稿token的对象

    def _copy_draft_token_ids_to_cpu(  # 将草稿token ID异步拷贝到CPU的方法
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False  # 调度器输出和是否仅零值标志
    ) -> None:  # 无返回值
        """将草稿token ID从GPU异步拷贝到CPU，或者仅将CPU张量清零。"""
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (  # 异步调度时检查是否真的需要拷贝
            scheduler_output.has_structured_output_requests  # 有结构化输出请求
            or self.input_batch.sampling_metadata.output_token_ids  # 或有需要输出token ID的采样参数
        ):
            return  # 不需要拷贝时直接返回
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()  # 拷贝请求ID列表

        draft_token_ids: torch.Tensor = self._draft_token_ids  # 获取草稿token ID张量
        if not torch.is_tensor(draft_token_ids):  # 如果不是张量类型（可能是列表）
            return  # 直接返回
        assert self.draft_token_ids_event is not None  # 断言CUDA事件存在
        assert self.draft_token_ids_copy_stream is not None  # 断言拷贝流存在
        assert self.draft_token_ids_cpu is not None  # 断言CPU缓冲区存在
        default_stream = torch.cuda.current_stream()  # 获取当前默认CUDA流
        num_reqs = draft_token_ids.shape[0]  # 获取请求数
        with torch.cuda.stream(self.draft_token_ids_copy_stream):  # 在专用拷贝流上执行
            if not zeros_only:  # 如果不是仅零值模式
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)  # 等待默认流完成
                self.draft_token_ids_cpu[:num_reqs].copy_(  # 异步拷贝GPU张量到CPU
                    draft_token_ids, non_blocking=True  # 非阻塞拷贝
                )
            else:  # 仅零值模式
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0  # 将CPU张量对应区域清零
            self.draft_token_ids_event.record()  # 记录CUDA事件以便后续同步

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:  # 获取CPU上草稿token ID的方法
        """等待异步拷贝完成并返回CPU上的草稿token ID和对应请求ID。"""
        if isinstance(self._draft_token_ids, list):  # 如果草稿token已经是列表格式
            return self._draft_token_ids, self.input_batch.req_ids  # 直接返回
        req_ids = self._draft_token_req_ids  # 获取草稿token对应的请求ID
        if req_ids is None:  # 如果没有请求ID
            return [], []  # 返回空列表
        assert self.draft_token_ids_event is not None  # 断言CUDA事件存在
        assert self.draft_token_ids_cpu is not None  # 断言CPU缓冲区存在
        self.draft_token_ids_event.synchronize()  # 同步等待异步拷贝完成
        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids  # 将CPU张量转为列表并返回

    def _copy_valid_sampled_token_count(  # 异步拷贝有效采样token计数到CPU的方法
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor  # 下一个token ID和有效计数张量
    ) -> None:  # 无返回值
        """将有效采样token计数从GPU异步拷贝到CPU，并更新上一步采样token ID。"""
        if self.valid_sampled_token_count_event is None:  # 如果没有有效采样计数事件
            return  # 直接返回

        default_stream = torch.cuda.current_stream()  # 获取当前默认CUDA流
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):  # 在专用拷贝流上执行
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore  # 等待默认流完成
            counts = valid_sampled_tokens_count  # 获取有效采样token计数张量
            counts_cpu = self.valid_sampled_token_count_cpu  # 获取CPU缓冲区
            assert counts_cpu is not None  # 断言CPU缓冲区存在
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)  # 异步拷贝计数到CPU
            self.valid_sampled_token_count_event.record()  # 记录CUDA事件以便后续同步

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)  # 将下一个token ID扩展维度并保存为上一步采样ID

    def _get_valid_sampled_token_count(self) -> list[int]:  # 获取有效采样token计数的方法
        """等待异步拷贝完成并返回CPU上的有效采样token计数列表。"""
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids  # 获取上一步采样token ID
        sampled_count_event = self.valid_sampled_token_count_event  # 获取CUDA同步事件
        if sampled_count_event is None or prev_sampled_token_ids is None:  # 如果事件或token ID不存在
            return []  # 返回空列表

        counts_cpu = self.valid_sampled_token_count_cpu  # 获取CPU缓冲区
        assert counts_cpu is not None  # 断言CPU缓冲区存在
        sampled_count_event.synchronize()  # 同步等待异步拷贝完成
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()  # 将CPU张量转为列表并返回

    # 投机解码的 draft token 生成入口。
    # 根据配置的投机解码方法（N-gram/EAGLE/Medusa/Draft Model 等），
    # 利用主模型的隐藏状态或输出 token 提议候选 draft token。
    def propose_draft_token_ids(  # 提议草稿token ID的方法
        self,
        scheduler_output: "SchedulerOutput",  # 调度器输出
        sampled_token_ids: torch.Tensor | list[list[int]],  # 主模型采样的token ID（GPU张量或CPU列表）
        sampling_metadata: SamplingMetadata,  # 采样元数据
        hidden_states: torch.Tensor,  # 主模型的隐藏状态
        sample_hidden_states: torch.Tensor,  # 用于采样的隐藏状态子集
        aux_hidden_states: list[torch.Tensor] | None,  # 辅助隐藏状态（EAGLE 3使用）
        spec_decode_metadata: SpecDecodeMetadata | None,  # 投机解码元数据
        common_attn_metadata: CommonAttentionMetadata,  # 通用注意力元数据
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,  # 槽位映射
    ) -> list[list[int]] | torch.Tensor:  # 返回草稿token ID列表或张量
        """
        投机解码的草稿token生成入口。根据配置的投机解码方法，
        利用主模型的隐藏状态或输出token提议候选草稿token。
        """
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取总调度token数
        spec_config = self.speculative_config  # 获取投机解码配置
        assert spec_config is not None  # 断言投机解码配置存在
        if spec_config.method == "ngram":  # 如果使用CPU N-gram方法
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer  # 导入N-gram提议器

            assert isinstance(sampled_token_ids, list)  # 断言采样token ID为列表格式
            assert isinstance(self.drafter, NgramProposer)  # 断言草稿器为NgramProposer类型
            draft_token_ids = self.drafter.propose(  # 调用N-gram提议方法
                sampled_token_ids,  # 采样的token ID列表
                self.input_batch.num_tokens_no_spec,  # 非投机token数
                self.input_batch.token_ids_cpu,  # CPU上的token ID
                slot_mappings=slot_mappings,  # 槽位映射
            )
        elif spec_config.use_ngram_gpu():  # 如果使用GPU N-gram方法
            assert isinstance(self.drafter, NgramProposerGPU)  # 断言草稿器为GPU N-gram类型
            (  # 解包GPU N-gram更新结果
                next_token_ids,  # 下一个token ID
                valid_sampled_tokens_count,  # 有效采样token计数
                valid_sampled_token_ids_gpu,  # GPU上的有效采样token ID
            ) = self.drafter.update_token_ids_ngram(  # 更新GPU N-gram的token ID
                sampled_token_ids,  # 采样的token ID
                self.input_batch,  # 输入批次
                self.token_ids_gpu_tensor,  # GPU上的token ID张量
                self.num_tokens_no_spec_gpu,  # GPU上的非投机token数
                self.discard_request_mask.gpu,  # GPU上的丢弃请求掩码
            )
            self._copy_valid_sampled_token_count(  # 异步拷贝有效采样计数到CPU
                next_token_ids, valid_sampled_tokens_count  # 传入token ID和计数
            )

            batch_size = next_token_ids.shape[0]  # 获取批次大小

            draft_token_ids, num_valid_draft_tokens = self.drafter.propose(  # 调用GPU N-gram提议方法
                self.num_tokens_no_spec_gpu[:batch_size],  # 截取批次大小的非投机token数
                self.token_ids_gpu_tensor[:batch_size],  # 截取批次大小的token ID张量
                valid_sampled_token_ids_gpu,  # GPU上的有效采样token ID
                valid_sampled_tokens_count,  # 有效采样token计数
            )

            # Cache valid draft counts for scheduler-side trimming.
            self._num_valid_draft_tokens = num_valid_draft_tokens  # 缓存有效草稿token数，供调度器裁剪使用

            # Async D2H copy on a dedicated stream.
            copy_num_valid_draft_tokens(  # 异步将有效草稿token数从GPU拷贝到CPU
                self._num_valid_draft_tokens_cpu,  # CPU缓冲区
                self._num_valid_draft_tokens_copy_stream,  # 专用拷贝流
                self._num_valid_draft_tokens_event,  # CUDA同步事件
                self._num_valid_draft_tokens,  # GPU上的有效草稿token数
                self.input_batch.num_reqs,  # 请求数量
            )
        elif spec_config.method == "suffix":  # 如果使用后缀解码方法
            assert isinstance(sampled_token_ids, list)  # 断言采样token ID为列表格式
            assert isinstance(self.drafter, SuffixDecodingProposer)  # 断言草稿器为后缀解码类型
            draft_token_ids = self.drafter.propose(  # 调用后缀解码提议方法
                self.input_batch, sampled_token_ids, slot_mappings=slot_mappings  # 传入批次、token ID和槽位映射
            )
        elif spec_config.method == "medusa":  # 如果使用Medusa方法
            assert isinstance(sampled_token_ids, list)  # 断言采样token ID为列表格式
            assert isinstance(self.drafter, MedusaProposer)  # 断言草稿器为Medusa类型

            if sample_hidden_states.shape[0] == len(sampled_token_ids):  # 如果隐藏状态数等于采样token数
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states  # 直接使用采样隐藏状态
            else:  # 隐藏状态包含草稿token对应的部分
                indices = []  # 初始化索引列表
                offset = 0  # 初始化偏移量为0，用于追踪token位置
                assert spec_decode_metadata is not None, (  # 断言投机解码元数据不为None
                    "No spec decode metadata for medusa"  # 错误信息：medusa模式缺少投机解码元数据
                )
                for num_draft, tokens in zip(  # 遍历每个请求的draft token数量和已采样的token
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids  # 从投机解码元数据和采样结果中获取
                ):
                    indices.append(offset + len(tokens) - 1)  # 将每个请求最后一个token的索引加入列表
                    offset += num_draft + 1  # 偏移量增加draft token数加1（包含原始token）
                indices = torch.tensor(indices, device=self.device)  # 将索引列表转换为GPU张量
                hidden_states = sample_hidden_states[indices]  # 根据索引从隐藏状态中提取对应的状态

            draft_token_ids = self.drafter.propose(  # 调用drafter模型生成草稿token
                target_hidden_states=hidden_states,  # 传入目标隐藏状态
                sampling_metadata=sampling_metadata,  # 传入采样元数据
                slot_mappings=slot_mappings,  # 传入slot映射
            )
        elif spec_config.uses_extract_hidden_states():  # 如果使用提取隐藏状态的投机解码方法
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)  # 断言drafter是ExtractHiddenStatesProposer类型
            assert isinstance(sampled_token_ids, torch.Tensor), (  # 断言采样token ID是torch张量
                "sampled_token_ids should be a torch.Tensor for "  # 错误信息部分1
                "extract_hidden_states method."  # 错误信息部分2：提取隐藏状态方法需要torch张量
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:  # 如果未启用辅助隐藏状态输出或其为None
                raise ValueError(  # 抛出值错误
                    "aux_hidden_states are required when using `extract_hidden_states`"  # 使用extract_hidden_states时需要辅助隐藏状态
                )
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]  # 截取每个辅助隐藏状态到已调度token数量

            draft_token_ids, drafter_kv_connector_output = self.drafter.propose(  # 调用drafter生成草稿token和KV连接器输出
                sampled_token_ids=sampled_token_ids,  # 传入已采样的token ID
                target_hidden_states=target_hidden_states,  # 传入目标隐藏状态
                common_attn_metadata=common_attn_metadata,  # 传入通用注意力元数据
                scheduler_output=scheduler_output,  # 传入调度器输出
                slot_mappings=slot_mappings,  # 传入slot映射
            )
            # Combine KVConnectorOutputs or select the non-empty one
            if self.kv_connector_output and drafter_kv_connector_output:  # 如果两个KV连接器输出都存在
                self.kv_connector_output = KVConnectorOutput.merge(  # 合并两个KV连接器输出
                    self.kv_connector_output, drafter_kv_connector_output  # 传入两个待合并的输出
                )
            else:  # 否则选择非空的那个
                self.kv_connector_output = (  # 设置KV连接器输出
                    self.kv_connector_output or drafter_kv_connector_output  # 取非None的那个
                )

            next_token_ids, valid_sampled_tokens_count = (  # 获取下一个token ID和有效采样token计数
                self.drafter.prepare_next_token_ids_padded(  # 调用drafter准备填充后的下一个token ID
                    common_attn_metadata,  # 传入通用注意力元数据
                    sampled_token_ids,  # 传入已采样的token ID
                    self.requests,  # 传入请求字典
                    self.input_batch,  # 传入输入批次
                    self.discard_request_mask.gpu,  # 传入丢弃请求掩码（GPU端）
                )
            )
            self._copy_valid_sampled_token_count(  # 复制有效采样token计数
                next_token_ids, valid_sampled_tokens_count  # 传入下一个token ID和有效计数
            )

        elif spec_config.use_eagle() or spec_config.uses_draft_model():  # 如果使用EAGLE或draft模型方法
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)  # 断言drafter是Eagle或DraftModel类型

            if spec_config.disable_padded_drafter_batch:  # 如果禁用了填充的drafter批次
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (  # 断言采样token ID是Python列表
                    "sampled_token_ids should be a python list when"  # 错误信息：禁用填充批次时应为列表
                    "padded-batch is disabled."  # 错误信息续
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(  # 在CPU端准备下一个token ID
                    sampled_token_ids,  # 传入已采样的token ID列表
                    self.requests,  # 传入请求字典
                    self.input_batch,  # 传入输入批次
                    scheduler_output.num_scheduled_tokens,  # 传入调度的token数量
                )
            else:  # 否则使用填充批次模式
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (  # 断言采样token ID是torch张量
                    "sampled_token_ids should be a torch.Tensor when"  # 错误信息：启用填充批次时应为张量
                    "padded-batch is enabled."  # 错误信息续
                )
                next_token_ids, valid_sampled_tokens_count = (  # 获取下一个token ID和有效采样token计数
                    self.drafter.prepare_next_token_ids_padded(  # 调用drafter准备填充后的下一个token ID
                        common_attn_metadata,  # 传入通用注意力元数据
                        sampled_token_ids,  # 传入已采样的token ID
                        self.requests,  # 传入请求字典
                        self.input_batch,  # 传入输入批次
                        self.discard_request_mask.gpu,  # 传入丢弃请求掩码（GPU端）
                    )
                )
                self._copy_valid_sampled_token_count(  # 复制有效采样token计数
                    next_token_ids, valid_sampled_tokens_count  # 传入下一个token ID和有效计数
                )

            num_rejected_tokens_gpu = None  # 初始化被拒绝token数量GPU变量为None
            if spec_decode_metadata is None:  # 如果投机解码元数据为None（首次运行，无draft token）
                token_indices_to_sample = None  # 设置要采样的token索引为None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]  # 获取目标token ID（截取到已调度数量）
                target_positions = self._get_positions(num_scheduled_tokens)  # 获取目标位置信息
                if self.use_aux_hidden_state_outputs:  # 如果使用辅助隐藏状态输出
                    assert aux_hidden_states is not None  # 断言辅助隐藏状态不为None
                    target_hidden_states = torch.cat(  # 拼接所有辅助隐藏状态
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1  # 在最后一个维度拼接
                    )
                else:  # 否则直接使用隐藏状态
                    target_hidden_states = hidden_states[:num_scheduled_tokens]  # 截取隐藏状态到已调度数量
            else:  # 如果有投机解码元数据（后续迭代，有draft token）
                if spec_config.disable_padded_drafter_batch:  # 如果禁用了填充的drafter批次
                    token_indices_to_sample = None  # 设置要采样的token索引为None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(  # 准备drafter输入并获取token索引
                        common_attn_metadata,  # 传入通用注意力元数据
                        sampled_token_ids,  # 传入已采样的token ID
                        spec_decode_metadata.num_draft_tokens,  # 传入draft token数量
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]  # 根据索引获取目标token ID
                    target_positions = self._get_positions(token_indices)  # 根据索引获取位置信息
                    if self.use_aux_hidden_state_outputs:  # 如果使用辅助隐藏状态输出
                        assert aux_hidden_states is not None  # 断言辅助隐藏状态不为None
                        target_hidden_states = torch.cat(  # 拼接辅助隐藏状态
                            [h[token_indices] for h in aux_hidden_states], dim=-1  # 根据索引选取并在最后维度拼接
                        )
                    else:  # 否则直接使用隐藏状态
                        target_hidden_states = hidden_states[token_indices]  # 根据索引选取隐藏状态
                else:  # 使用填充批次模式
                    (
                        common_attn_metadata,  # 更新后的通用注意力元数据
                        token_indices_to_sample,  # 需要采样的token索引
                        num_rejected_tokens_gpu,  # GPU端被拒绝的token数量
                    ) = self.drafter.prepare_inputs_padded(  # 准备填充的drafter输入
                        common_attn_metadata,  # 传入通用注意力元数据
                        spec_decode_metadata,  # 传入投机解码元数据
                        valid_sampled_tokens_count,  # 传入有效采样token计数
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens  # 获取实际token总数
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]  # 截取目标token ID到总token数
                    target_positions = self._get_positions(total_num_tokens)  # 获取位置信息
                    if self.use_aux_hidden_state_outputs:  # 如果使用辅助隐藏状态输出
                        assert aux_hidden_states is not None  # 断言辅助隐藏状态不为None
                        target_hidden_states = torch.cat(  # 拼接辅助隐藏状态
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1  # 截取并在最后维度拼接
                        )
                    else:  # 否则直接使用隐藏状态
                        target_hidden_states = hidden_states[:total_num_tokens]  # 截取隐藏状态到总token数

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:  # 如果目标模型和drafter都支持多模态输入
                mm_embed_inputs = self._gather_mm_embeddings(  # 收集多模态嵌入输入
                    scheduler_output,  # 传入调度器输出
                    shift_computed_tokens=1,  # 偏移已计算的token数为1
                )
            else:  # 否则不使用多模态嵌入
                mm_embed_inputs = None  # 设置多模态嵌入输入为None

            draft_token_ids = self.drafter.propose(  # 调用drafter生成草稿token ID
                target_token_ids=target_token_ids,  # 传入目标token ID
                target_positions=target_positions,  # 传入目标位置
                target_hidden_states=target_hidden_states,  # 传入目标隐藏状态
                next_token_ids=next_token_ids,  # 传入下一个token ID
                token_indices_to_sample=token_indices_to_sample,  # 传入需要采样的token索引
                sampling_metadata=sampling_metadata,  # 传入采样元数据
                common_attn_metadata=common_attn_metadata,  # 传入通用注意力元数据
                mm_embed_inputs=mm_embed_inputs,  # 传入多模态嵌入输入
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,  # 传入被拒绝的token数量
                slot_mappings=slot_mappings,  # 传入slot映射
            )

        return draft_token_ids  # 返回草稿token ID

    def update_config(self, overrides: dict[str, Any]) -> None:  # 更新模型运行器配置的方法
        """更新模型运行器的配置，仅允许修改指定的配置项"""
        allowed_config_names = {"load_config", "model_config"}  # 定义允许修改的配置名称集合
        for config_name, config_overrides in overrides.items():  # 遍历所有覆盖配置
            assert config_name in allowed_config_names, (  # 断言配置名在允许列表中
                f"Config `{config_name}` not supported. "  # 错误信息：不支持的配置
                f"Allowed configs: {allowed_config_names}"  # 显示允许的配置列表
            )
            config = getattr(self, config_name)  # 获取当前配置对象
            new_config = update_config(config, config_overrides)  # 使用覆盖值更新配置
            setattr(self, config_name, new_config)  # 设置更新后的配置

    # 加载模型权重和相关组件。流程：
    # 通过模型加载器加载权重（支持张量并行分片和量化），
    # 初始化 LoRA 层、投机解码 draft 模型、EPLB 状态，
    # 设置 KV 缓存共享层、中间张量缓冲区，以及 CUDA Graph 包装器。
    @instrument(span_name="Loading (GPU)")  # 使用追踪装饰器标记加载阶段
    def load_model(self, load_dummy_weights: bool = False) -> None:  # 加载模型权重的方法
        """
        加载模型权重到GPU，支持真实权重和虚拟权重加载。
        包括LoRA、draft模型、EPLB、CUDA Graph等初始化。

        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        logger.info_once(  # 记录模型加载开始的日志
            "Starting to load model %s...",  # 日志消息模板
            self.model_config.model,  # 模型名称
            scope="global",  # 全局作用域
        )

        if self.parallel_config.enable_eplb:  # 如果启用了专家并行负载均衡（EPLB）
            self.eplb_state = EplbState(self.parallel_config, self.device)  # 创建EPLB状态对象
            eplb_models = 0  # 初始化EPLB模型计数器

        try:  # 尝试加载模型
            with DeviceMemoryProfiler() as m:  # 使用设备内存分析器追踪内存使用
                time_before_load = time.perf_counter()  # 记录加载开始时间
                if load_dummy_weights:  # 如果加载虚拟权重
                    self.load_config.load_format = "dummy"  # 设置加载格式为dummy
                model_loader = get_model_loader(self.load_config)  # 获取模型加载器
                self.model = model_loader.load_model(  # 加载模型权重
                    vllm_config=self.vllm_config, model_config=self.model_config  # 传入vllm配置和模型配置
                )
                if self.lora_config:  # 如果配置了LoRA
                    self.model = self.load_lora_model(  # 加载LoRA模型
                        self.model, self.vllm_config, self.device  # 传入基础模型、配置和设备
                    )
                if hasattr(self, "drafter"):  # 如果存在drafter（投机解码）
                    logger.info_once("Loading drafter model...")  # 记录drafter加载日志
                    self.drafter.load_model(self.model)  # 加载drafter模型
                    if (  # 检查drafter是否为MoE模型且启用了EPLB
                        hasattr(self.drafter, "model")  # drafter有model属性
                        and is_mixture_of_experts(self.drafter.model)  # 且是混合专家模型
                        and self.parallel_config.enable_eplb  # 且启用了EPLB
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (  # 断言未启用弹性EP
                            "Elastic EP is not supported with drafter model."  # 弹性EP不支持drafter模型
                        )
                        spec_config = self.vllm_config.speculative_config  # 获取投机解码配置
                        assert spec_config is not None  # 断言投机配置不为None
                        assert spec_config.draft_model_config is not None  # 断言draft模型配置不为None
                        logger.info_once(  # 记录EPLB启用日志
                            "EPLB is enabled for drafter model %s.",  # 日志消息模板
                            spec_config.draft_model_config.model,  # drafter模型名称
                        )
                        if self.eplb_state is None:  # 如果EPLB状态未初始化
                            self.eplb_state = EplbState(  # 创建EPLB状态
                                self.parallel_config, self.device  # 传入并行配置和设备
                            )
                        self.eplb_state.add_model(  # 将drafter模型添加到EPLB状态
                            self.drafter.model,  # drafter模型
                            spec_config.draft_model_config,  # draft模型配置
                        )
                        eplb_models += 1  # EPLB模型计数加1

                if self.use_aux_hidden_state_outputs:  # 如果使用辅助隐藏状态输出（如EAGLE3）
                    if not supports_eagle3(self.get_model()):  # 如果模型不支持EAGLE3
                        raise RuntimeError(  # 抛出运行时错误
                            "Model does not support EAGLE3 interface but "  # 错误信息：模型不支持EAGLE3
                            "aux_hidden_state_outputs was requested"  # 但请求了辅助隐藏状态输出
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()  # 从配置获取EAGLE3辅助层索引
                    if aux_layers:  # 如果从配置获取到了辅助层
                        logger.info(  # 记录日志
                            "Using auxiliary layers from speculative config: %s",  # 日志消息模板
                            aux_layers,  # 辅助层索引
                        )
                    else:  # 否则使用模型默认的辅助层
                        aux_layers = (  # 获取默认辅助层
                            self.model.get_eagle3_default_aux_hidden_state_layers()  # 调用模型的默认方法
                        )

                    self.model.set_aux_hidden_state_layers(aux_layers)  # 设置辅助隐藏状态层
                time_after_load = time.perf_counter()  # 记录加载结束时间
            self.model_memory_usage = m.consumed_memory  # 记录模型内存使用量
        except torch.cuda.OutOfMemoryError as e:  # 捕获CUDA内存不足错误
            msg = (  # 构建错误消息
                "Failed to load model - not enough GPU memory. "  # 加载失败：GPU内存不足
                "Try lowering --gpu-memory-utilization to free memory for weights, "  # 建议：降低GPU内存利用率
                "increasing --tensor-parallel-size, or using --quantization. "  # 或增加张量并行或使用量化
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "  # 参考文档链接
                "for more tips."  # 更多提示
            )
            combined_msg = f"{msg} (original error: {e})"  # 组合原始错误信息
            logger.error(combined_msg)  # 记录错误日志
            raise e  # 重新抛出异常
        logger.info_once(  # 记录模型加载完成的日志
            "Model loading took %s GiB memory and %.6f seconds",  # 日志消息模板
            format_gib(self.model_memory_usage),  # 内存使用量（GiB）
            time_after_load - time_before_load,  # 加载耗时（秒）
            scope="local",  # 本地作用域
        )
        if not load_dummy_weights:  # 如果不是加载虚拟权重
            prepare_communication_buffer_for_model(self.model)  # 为模型准备通信缓冲区
            if (drafter := getattr(self, "drafter", None)) and (  # 如果存在drafter
                drafter_model := getattr(drafter, "model", None)  # 且drafter有model属性
            ):
                prepare_communication_buffer_for_model(drafter_model)  # 为drafter模型准备通信缓冲区
        mm_config = self.model_config.multimodal_config  # 获取多模态配置
        self.is_multimodal_pruning_enabled = (  # 设置多模态剪枝是否启用
            supports_multimodal_pruning(self.get_model())  # 检查模型是否支持多模态剪枝
            and mm_config is not None  # 且多模态配置存在
            and mm_config.is_multimodal_pruning_enabled()  # 且多模态剪枝已启用
        )

        if (  # 检查是否需要为主模型启用EPLB
            is_mixture_of_experts(self.model)  # 模型是混合专家模型
            and self.parallel_config.enable_eplb  # 且启用了EPLB
            and not load_dummy_weights  # 且不是加载虚拟权重
        ):
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)  # 记录EPLB启用日志
            assert self.eplb_state is not None  # 断言EPLB状态已初始化
            self.eplb_state.add_model(  # 将主模型添加到EPLB状态
                self.model,  # 主模型
                self.model_config,  # 模型配置
            )
            if self.eplb_state.is_async:  # 如果EPLB使用异步模式
                self.eplb_state.start_async_loop()  # 启动异步事件循环

        if (  # 检查是否使用原生PyTorch编译模式
            self.vllm_config.compilation_config.mode  # 获取编译模式
            == CompilationMode.STOCK_TORCH_COMPILE  # 判断是否为原生torch.compile模式
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)  # 初始化编译后端
            compilation_counter.stock_torch_compile_count += 1  # 增加编译计数器
            self.model.compile(fullgraph=True, backend=backend)  # 使用torch.compile编译模型
            return  # 编译完成后直接返回
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWrapper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode  # 获取CUDA图模式
        assert cudagraph_mode is not None  # 断言CUDA图模式不为None
        if (  # 检查是否需要完整CUDA图包装且不使用微批次
            cudagraph_mode.has_full_cudagraphs()  # 启用了完整CUDA图
            and not self.parallel_config.use_ubatching  # 且未使用微批次
        ):
            self.model = CUDAGraphWrapper(  # 用CUDA图包装器包装模型
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL  # 设置为完整CUDA图运行时模式
            )
        elif self.parallel_config.use_ubatching:  # 如果使用微批次
            if cudagraph_mode.has_full_cudagraphs():  # 如果启用了完整CUDA图
                self.model = UBatchWrapper(  # 用微批次包装器包装模型（完整CUDA图模式）
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device  # 传入完整模式和设备
                )
            else:  # 否则不使用CUDA图
                self.model = UBatchWrapper(  # 用微批次包装器包装模型（无CUDA图模式）
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device  # 传入无CUDA图模式
                )

        get_offloader().post_init()  # 获取offloader并执行后初始化

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:  # 从配置中获取EAGLE3辅助层索引
        """
        从投机解码配置中提取Eagle3辅助层索引。
        这些索引指定基础模型的哪些隐藏状态应作为Eagle3 drafter模型的辅助输入。

        Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):  # 如果没有投机配置或draft模型配置
            return None  # 返回None

        hf_config = self.speculative_config.draft_model_config.hf_config  # 获取draft模型的HuggingFace配置
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):  # 如果配置没有辅助层ID属性
            return None  # 返回None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids  # 获取辅助层ID列表
        if layer_ids and isinstance(layer_ids, (list, tuple)):  # 如果层ID存在且是列表或元组
            return tuple(layer_ids)  # 返回元组形式的层ID

        return None  # 否则返回None

    # 热重载模型权重：支持在运行时从新的检查点加载权重，
    # 可选择性地更新模型配置中的生成参数（如 temperature）。
    def reload_weights(  # 热重载模型权重的方法
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,  # 可选的权重迭代器
        weights_path: str | None = None,  # 可选的权重路径
        is_checkpoint_format: bool = True,  # 是否为检查点格式
    ) -> None:
        """
        从权重迭代器或磁盘重新加载模型权重。

        Reload weights from a weights iterator or from disk

        :param weights_iterator: weights to load into model
        :param weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        :param is_checkpoint_format: set to False if weights have already been processed
            into kernel format (repacking, renaming, etc.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:  # 如果没有提供权重迭代器但标记为非检查点格式
            logger.warning(  # 记录警告日志
                "Reloading from disk means that weights will be in checkpoint format. "  # 从磁盘加载意味着权重是检查点格式
                "Please use `is_checkpoint_format=True` "  # 建议使用is_checkpoint_format=True
                "to avoid weight reloading errors"  # 以避免权重重载错误
            )

        model = self.get_model()  # 获取基础模型
        weights_to_load = {name for name, _ in model.named_parameters()}  # 获取所有需要加载的参数名集合
        counter_before_reloading = time.perf_counter()  # 记录重载开始时间

        # load weights from disk if none are provided
        if weights_iterator is None:  # 如果没有提供权重迭代器
            model_loader = get_model_loader(self.load_config)  # 获取模型加载器
            if not hasattr(model_loader, "get_all_weights"):  # 如果加载器不支持get_all_weights方法
                raise NotImplementedError(  # 抛出未实现错误
                    f"Model reloading with `{self.load_config.load_format}` format"  # 该格式不支持重载
                )

            if weights_path is not None:  # 如果提供了权重路径
                self.model_config.model = weights_path  # 更新模型配置的模型路径
            weights_iterator = model_loader.get_all_weights(self.model_config, model)  # 获取所有权重的迭代器
            weights_iterator = cast(  # 类型转换
                Iterable[tuple[str, torch.Tensor]], weights_iterator  # 转为权重迭代器类型
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...", scope="local")  # 记录开始原地重载权重的日志
        load_device = (  # 确定加载设备
            self.vllm_config.load_config.device or self.vllm_config.device_config.device  # 优先使用加载配置的设备
        )
        with torch.device(load_device):  # 在指定设备上执行
            if is_checkpoint_format:  # 如果是检查点格式
                # load weights from checkpoint/ original model format
                initialize_layerwise_reload(model)  # 初始化逐层重载
                loaded_weights = model.load_weights(weights_iterator)  # 加载权重并获取已加载的权重集合
                finalize_layerwise_reload(model, self.model_config)  # 完成逐层重载的收尾工作

            else:  # 否则是已处理的内核格式
                # load weights from kernel format
                logger.warning_once(  # 记录警告日志
                    "Reloading with `is_checkpoint_format=True` requires that "  # 使用True需要权重已是内核格式
                    "weights be in kernel format and already sharded",  # 且已经分片
                    scope="local",  # 本地作用域
                )
                loaded_weights = set()  # 初始化已加载权重集合
                for name, loaded_weight in weights_iterator:  # 遍历权重迭代器
                    param = model.get_parameter(name)  # TODO: buffers?  # 获取模型参数
                    param.copy_(loaded_weight)  # 复制权重到参数
                    loaded_weights.add(name)  # 将参数名加入已加载集合

        # logging and validation
        counter_after_reloading = time.perf_counter()  # 记录重载结束时间
        diff_seconds = counter_after_reloading - counter_before_reloading  # 计算重载耗时
        logger.info_once(  # 记录重载耗时日志
            "Reloading and processing weights took %.2f seconds",  # 日志消息模板
            diff_seconds,  # 耗时秒数
            scope="local",  # 本地作用域
        )
        if self.model_config.quantization is None and loaded_weights is not None:  # 如果不是量化模型且有已加载权重
            weights_not_loaded = weights_to_load - loaded_weights  # 计算未加载的权重
            if weights_not_loaded:  # 如果有未加载的权重
                logger.warning(  # 记录警告日志
                    "Following weights were not loaded from checkpoint: %s",  # 以下权重未从检查点加载
                    weights_not_loaded,  # 未加载的权重名称
                )

    def _get_prompt_logprobs_dict(  # 获取提示词对数概率字典的方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        num_scheduled_tokens: dict[str, int],  # 每个请求调度的token数量字典
    ) -> dict[str, LogprobsTensors | None]:  # 返回请求ID到LogprobsTensors的字典
        """
        获取提示词（prompt）的对数概率。对于分块预填充，
        逐步收集各块的logprobs并在完成时一次性返回。
        """
        num_prompt_logprobs_dict = self.num_prompt_logprobs  # 获取需要prompt logprobs的请求字典
        if not num_prompt_logprobs_dict:  # 如果没有需要prompt logprobs的请求
            return {}  # 返回空字典

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu  # 获取正在进行的prompt logprobs字典
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}  # 初始化结果字典

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []  # 初始化已完成预填充的请求列表
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():  # 遍历每个需要prompt logprobs的请求
            num_tokens = num_scheduled_tokens.get(req_id)  # 获取该请求调度的token数
            if num_tokens is None:  # 如果该请求没有调度的token
                # This can happen if the request was preempted in prefill stage.
                continue  # 跳过（可能在预填充阶段被抢占）

            # Get metadata for this request.
            request = self.requests[req_id]  # 获取请求对象
            if request.prompt_token_ids is None:  # 如果提示词token ID为None
                # Prompt logprobs is incompatible with prompt embeddings
                continue  # 跳过（prompt logprobs与prompt嵌入不兼容）

            num_prompt_tokens = len(request.prompt_token_ids)  # 获取提示词token总数
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(  # 将提示词token ID转为GPU张量
                self.device, non_blocking=True  # 异步传输到设备
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)  # 获取该请求正在进行的logprobs张量
            if not logprobs_tensors:  # 如果还没有创建logprobs张量
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(  # 创建空的CPU logprobs张量
                    num_prompt_tokens - 1, num_prompt_logprobs + 1  # 大小为prompt长度-1，列数为logprobs数+1
                )
                in_progress_dict[req_id] = logprobs_tensors  # 保存到正在进行的字典中

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens  # 获取已计算的token起始索引
            start_tok = start_idx + 1  # 起始token位置（偏移1因为第一个token无logprob）
            num_remaining_tokens = num_prompt_tokens - start_tok  # 计算剩余需要计算logprob的token数
            if num_tokens <= num_remaining_tokens:  # 如果本次调度的token数不超过剩余数
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens  # 本次处理的logit数等于调度的token数（分块处理）
            else:  # 否则这是最后一个块
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens  # 处理剩余所有token
                completed_prefill_reqs.append(req_id)  # 将该请求标记为预填充完成
                prompt_logprobs_dict[req_id] = logprobs_tensors  # 将完成的logprobs加入结果字典

            if num_logits <= 0:  # 如果没有需要处理的logit
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue  # 跳过

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]  # 获取请求在批次中的索引
            offset = self.query_start_loc.np[req_idx].item()  # 获取该请求在批次中的起始偏移
            prompt_hidden_states = hidden_states[offset : offset + num_logits]  # 提取该请求对应的隐藏状态
            logits = self.model.compute_logits(prompt_hidden_states)  # 通过模型计算logits

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]  # 获取目标token ID（即下一个token）

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)  # 计算对数概率
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(  # 收集top-k logprobs
                logprobs, num_prompt_logprobs, tgt_token_ids  # 传入logprobs、数量和目标token
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)  # 计算当前块在整体中的切片
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(  # 异步复制token ID到CPU
                token_ids, non_blocking=True  # 非阻塞传输
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)  # 异步复制logprobs到CPU
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(  # 异步复制排名到CPU
                ranks, non_blocking=True  # 非阻塞传输
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:  # 遍历已完成预填充的请求
            del num_prompt_logprobs_dict[req_id]  # 从prompt logprobs字典中删除
            del in_progress_dict[req_id]  # 从正在进行的字典中删除

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:  # 如果有完成的prompt logprobs
            self._sync_device()  # 同步设备以确保异步传输完成

        return prompt_logprobs_dict  # 返回prompt logprobs字典

    def _get_nans_in_logits(  # 检查logits中NaN值的方法
        self,
        logits: torch.Tensor | None,  # 可选的logits张量
    ) -> dict[str, int]:  # 返回请求ID到NaN数量的字典
        """检测每个请求的logits中NaN值的数量"""
        try:  # 尝试检测NaN
            if logits is None:  # 如果logits为None
                return {req_id: 0 for req_id in self.input_batch.req_ids}  # 所有请求返回0个NaN

            num_nans_in_logits = {}  # 初始化NaN计数字典
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()  # 计算每行的NaN数量并转为numpy
            for req_id in self.input_batch.req_ids:  # 遍历所有请求ID
                req_index = self.input_batch.req_id_to_index[req_id]  # 获取请求在批次中的索引
                num_nans_in_logits[req_id] = (  # 记录该请求的NaN数量
                    int(num_nans_for_index[req_index])  # 从numpy数组获取NaN计数
                    if num_nans_for_index is not None and req_index < logits.shape[0]  # 前提是索引有效
                    else 0  # 否则为0
                )
            return num_nans_in_logits  # 返回NaN计数字典
        except IndexError:  # 捕获索引错误
            return {}  # 返回空字典

    @contextmanager  # 上下文管理器装饰器
    def maybe_randomize_inputs(  # 可选随机化输入的上下文管理器方法
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None  # 输入token ID和嵌入
    ):
        """
        如果设置了VLLM_RANDOMIZE_DP_DUMMY_INPUTS，则随机化input_ids。
        这有助于在以下场景中平衡专家选择：
         - profile_run期间
         - DP rank虚拟运行期间

        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size  # 获取数据并行大小
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1  # 判断是否需要随机化
        if not randomize_inputs:  # 如果不需要随机化
            yield  # 直接返回（不做任何修改）
        elif input_ids is not None:  # 如果input_ids不为None（文本模型）

            @functools.cache  # 缓存装饰器，避免重复生成
            def rand_input_ids() -> torch.Tensor:  # 生成随机input_ids的内部函数
                return torch.randint_like(  # 生成与input_ids相同形状的随机整数张量
                    self.input_ids.gpu,  # 参考GPU上的input_ids
                    low=0,  # 最小值0
                    high=self.model_config.get_vocab_size(),  # 最大值为词表大小
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")  # 记录调试日志
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)  # 复制随机ID到input_ids
            yield  # 执行上下文中的代码
            input_ids.fill_(0)  # 执行完毕后将input_ids清零
        else:  # 否则处理inputs_embeds（多模态或嵌入输入模型）

            @functools.cache  # 缓存装饰器
            def rand_inputs_embeds() -> torch.Tensor:  # 生成随机嵌入的内部函数
                return torch.randn_like(  # 生成与inputs_embeds相同形状的随机正态张量
                    self.inputs_embeds.gpu,  # 参考GPU上的inputs_embeds
                )

            assert inputs_embeds is not None  # 断言inputs_embeds不为None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")  # 记录调试日志
            inputs_embeds.copy_(  # 复制随机嵌入到inputs_embeds
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True  # 截取并异步复制
            )
            yield  # 执行上下文中的代码
            inputs_embeds.fill_(0)  # 执行完毕后将inputs_embeds清零

    def _get_mm_dummy_batch(  # 获取多模态虚拟批次的方法
        self,
        modality: str,  # 模态类型（如"image"、"audio"等）
        max_items_per_batch: int,  # 每批次最大项目数
    ) -> BatchedTensorInputs:  # 返回批量张量输入
        """
        用于profiling和预编译多模态模型的虚拟数据。

        Dummy data for profiling and precompiling multimodal models.
        """
        assert self.mm_budget is not None  # 断言多模态预算不为None

        # Don't use `max_items_per_batch` here to avoid redundant computation
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(  # 从多模态注册表获取虚拟输入
            self.model_config,  # 传入模型配置
            mm_counts={modality: 1},  # 设置模态计数为1
            cache=self.mm_budget.cache,  # 使用多模态预算缓存
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]  # 获取第一个虚拟多模态项

        # We use the cache so that the item is saved to the cache,
        # but not read from the cache
        assert dummy_mm_item is not None, "Item should not already be cached"  # 断言虚拟项不为None

        return next(  # 返回第一个批次化的多模态关键字参数
            mm_kwargs_batch  # 批次化的多模态kwargs
            for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(  # 分组并批次化多模态kwargs
                [(modality, dummy_mm_item)] * max_items_per_batch,  # 复制max_items_per_batch份
                device=self.device,  # 目标设备
                pin_memory=self.pin_memory,  # 是否锁页内存
            )
        )

    # 使用虚拟输入执行模型前向传播，用于 torch.compile 预热、
    # CUDA Graph 捕获、内存 profiling 以及内核调优。
    # 支持多种配置：统一 decode、微批次、LoRA 激活等。
    @torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
    def _dummy_run(  # 虚拟运行方法，用于预热和CUDA图捕获
        self,
        num_tokens: int,  # 要运行的token数量
        cudagraph_runtime_mode: CUDAGraphMode | None = None,  # 可选的CUDA图运行时模式
        force_attention: bool = False,  # 是否强制创建注意力元数据
        uniform_decode: bool = False,  # 是否为统一解码批次
        allow_microbatching: bool = True,  # 是否允许微批次
        skip_eplb: bool = False,  # 是否跳过EPLB状态更新
        is_profile: bool = False,  # 是否为profile运行
        create_mixed_batch: bool = False,  # 是否创建混合批次
        remove_lora: bool = True,  # 运行后是否移除LoRA
        is_graph_capturing: bool = False,  # 是否正在捕获CUDA图
        num_active_loras: int = 0,  # 活跃LoRA的数量
        profile_seq_lens: int | None = None,  # 可选的profile序列长度
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回隐藏状态和logits的元组
        """
        执行虚拟前向传播以预热/profile运行或捕获模型的CUDA图。

        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        mm_config = self.vllm_config.model_config.multimodal_config  # 获取多模态配置
        if mm_config and mm_config.mm_encoder_only:  # 如果是仅编码器的多模态模型
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])  # 返回空张量（跳过LM虚拟运行）

        assert (  # 断言CUDA图运行时模式有效
            cudagraph_runtime_mode is None  # 为None
            or cudagraph_runtime_mode.is_valid_runtime_mode()  # 或者是有效的运行时模式
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens  # 根据是否统一解码设置最大查询长度

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.max_num_tokens  # 断言token数不超过最大允许数
        max_num_reqs = self.scheduler_config.max_num_seqs  # 获取最大请求数
        if create_mixed_batch:  # 如果创建混合批次
            assert not uniform_decode  # 断言不是统一解码
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)  # 计算解码token数（最多为最大请求数-1和token数的一半）
            num_prefill_tokens = num_tokens - num_decode_tokens  # 计算预填充token数
            num_reqs = num_decode_tokens + 1  # 请求数为解码请求数+1（1个预填充请求）

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]  # 每个解码请求1个token，预填充请求多个
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens  # 将最大查询长度设为预填充token数
        elif uniform_decode:  # 如果是统一解码
            assert not create_mixed_batch  # 断言不是混合批次
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))  # 计算请求数（向上取整）
            num_scheduled_tokens_list = [max_query_len] * num_reqs  # 每个请求分配max_query_len个token
            if num_tokens % max_query_len != 0:  # 如果不能整除
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len  # 最后一个请求分配余数
        else:  # 默认情况
            num_reqs = min(num_tokens, max_num_reqs)  # 请求数为token数和最大请求数的较小值
            min_tokens_per_req = num_tokens // num_reqs  # 每个请求的最小token数
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs  # 均匀分配token
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs  # 最后一个请求分配余数

        assert sum(num_scheduled_tokens_list) == num_tokens  # 断言总token数正确
        assert len(num_scheduled_tokens_list) == num_reqs  # 断言请求数正确
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)  # 转为numpy数组
        num_tokens_unpadded = int(num_scheduled_tokens.sum())  # 计算未填充的总token数

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)  # 每个请求采样1个token

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (  # 确定批次执行和填充策略
            self._determine_batch_execution_and_padding(  # 调用确定批次执行和填充的方法
                num_tokens=num_tokens_unpadded,  # 未填充的token数
                num_reqs=num_reqs,  # 请求数
                num_scheduled_tokens_np=num_scheduled_tokens,  # 调度的token数数组
                max_num_scheduled_tokens=max_query_len,  # 最大调度token数
                use_cascade_attn=False,  # 不使用级联注意力
                allow_microbatching=allow_microbatching,  # 是否允许微批次
                force_eager=is_profile  # 强制eager模式（profile或无CUDA图时）
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),  # 或CUDA图模式为NONE
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,  # 强制统一解码标记
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,  # 是否强制LoRA（用于CUDA图捕获）
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,  # 强制活跃LoRA数量
            )
        )

        if cudagraph_runtime_mode is None:  # 如果未指定CUDA图运行时模式
            cudagraph_runtime_mode = _cudagraph_mode  # 使用自动确定的模式
        else:  # 否则验证模式匹配
            assert cudagraph_runtime_mode == _cudagraph_mode, (  # 断言模式匹配
                f"Cudagraph runtime mode mismatch in dummy_run. "  # 虚拟运行中CUDA图运行时模式不匹配
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."  # 显示期望和实际值
            )

        num_tokens_padded = batch_desc.num_tokens  # 获取填充后的token数
        num_reqs_padded = (  # 获取填充后的请求数
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs  # 如果batch_desc有num_reqs则使用，否则用原值
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(  # 创建微批次切片
            should_ubatch,  # 是否应使用微批次
            num_scheduled_tokens,  # 调度的token数
            num_tokens_padded,  # 填充后的token数
            num_reqs_padded,  # 填充后的请求数
            self.vllm_config.parallel_config.num_ubatches,  # 微批次数量
        )
        logger.debug(  # 记录调试日志
            "ubatch_slices: %s, ubatch_slices_padded: %s",  # 日志消息模板
            ubatch_slices,  # 微批次切片
            ubatch_slices_padded,  # 填充后的微批次切片
        )

        attn_metadata: PerLayerAttnMetadata | None = None  # 初始化注意力元数据为None

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(  # 获取slot映射
            num_tokens_padded=num_tokens,  # 填充的token数
            num_reqs_padded=num_reqs_padded,  # 填充的请求数
            num_tokens_unpadded=num_tokens_unpadded,  # 未填充的token数
            ubatch_slices=ubatch_slices_padded,  # 填充后的微批次切片
        )

        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # etc.) with execute_model.  It must participate in the same event
        # protocol so that back-to-back dummy/real steps don't overwrite
        # pinned memory while a prior non_blocking H2D DMA is still reading.
        with self.synchronize_input_prep():  # 同步输入准备（防止DMA冲突）
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:  # 如果强制注意力或完整CUDA图模式
                if profile_seq_lens is not None:  # 如果提供了profile序列长度
                    seq_lens = profile_seq_lens  # type: ignore[assignment]  # 使用profile序列长度
                elif create_mixed_batch:  # 如果创建混合批次
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]  # type: ignore[assignment]  # 解码请求长度1，预填充请求长度为token数+1
                else:  # 否则使用默认序列长度
                    seq_lens = max_query_len  # type: ignore[assignment]  # 序列长度等于最大查询长度
                self.seq_lens.np[:num_reqs] = seq_lens  # 设置前num_reqs个请求的序列长度
                self.seq_lens.np[num_reqs:] = 0  # 其余请求序列长度设为0
                self.seq_lens.copy_to_gpu()  # 复制序列长度到GPU

                cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)  # 计算token数的累积和及范围
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens  # 设置查询起始位置
                self.query_start_loc.copy_to_gpu()  # 复制查询起始位置到GPU

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL  # 是否填充注意力（完整CUDA图模式时）
                attn_metadata, _ = self._build_attention_metadata(  # 构建注意力元数据
                    num_tokens=num_tokens_unpadded,  # 未填充的token数
                    num_tokens_padded=num_tokens_padded if pad_attn else None,  # 填充的token数（仅完整CUDA图时）
                    num_reqs=num_reqs_padded,  # 填充的请求数
                    max_query_len=max_query_len,  # 最大查询长度
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),  # 微批次切片
                    for_cudagraph_capture=is_graph_capturing,  # 是否用于CUDA图捕获
                    slot_mappings=slot_mappings_by_group,  # slot映射
                    use_spec_decode=self.speculative_config is not None,  # 是否使用投机解码
                )

        with self.maybe_dummy_run_with_lora(  # 使用LoRA上下文管理器
            self.lora_config,  # LoRA配置
            num_scheduled_tokens,  # 调度的token数
            num_sampled_tokens,  # 采样的token数
            remove_lora,  # 是否在运行后移除LoRA
            num_active_loras,  # 活跃LoRA数量
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens  # 断言填充后的token数不超过最大值
            model_kwargs = self._init_model_kwargs()  # 初始化模型关键字参数
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:  # 如果支持多模态输入且非编码器-解码器
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)  # 准备多模态输入

                model_kwargs = {  # 更新模型关键字参数
                    **model_kwargs,  # 保留原有参数
                    **self._dummy_mm_kwargs(num_reqs),  # 添加虚拟多模态参数
                }
            elif self.enable_prompt_embeds:  # 如果启用了prompt嵌入
                input_ids = None  # 不使用input_ids
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]  # 截取prompt嵌入
                model_kwargs = self._init_model_kwargs()  # 重新初始化模型参数
            else:  # 默认情况（纯文本模型）
                input_ids = self.input_ids.gpu[:num_tokens_padded]  # 截取input_ids
                inputs_embeds = None  # 不使用嵌入

            if self.uses_mrope:  # 如果使用多旋转位置编码
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]  # 截取mROPE位置
            elif self.uses_xdrope_dim > 0:  # 如果使用xdRoPE
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]  # 截取xdRoPE位置
            else:  # 默认位置编码
                positions = self.positions.gpu[:num_tokens_padded]  # 截取标准位置

            if get_pp_group().is_first_rank:  # 如果是流水线并行的第一个rank
                intermediate_tensors = None  # 无需中间张量
            else:  # 非第一个rank
                if self.intermediate_tensors is None:  # 如果中间张量未初始化
                    self.intermediate_tensors = (  # 创建空的中间张量
                        self.model.make_empty_intermediate_tensors(  # 调用模型方法创建
                            batch_size=self.max_num_tokens,  # 批次大小为最大token数
                            dtype=self.model_config.dtype,  # 数据类型
                            device=self.device,  # 设备
                        )  # 创建空的中间张量，用于流水线并行中非首rank的前向传播
                    )  # 结束make_empty_intermediate_tensors的赋值

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(  # 同步并切片中间张量
                    num_tokens_padded, None, False  # 传入填充后的token数量，无切片信息，不需要广播
                )  # 结束sync_and_slice_intermediate_tensors调用

            if ubatch_slices_padded is not None:  # 如果存在填充后的微批次切片信息
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens  # 将填充token数设为第一个微批次的token数
                if num_tokens_across_dp is not None:  # 如果存在跨数据并行的token数信息
                    num_tokens_across_dp[:] = num_tokens_padded  # 更新所有数据并行rank的token数为填充后的值

            with (  # 进入上下文管理器块
                self.maybe_randomize_inputs(input_ids, inputs_embeds),  # 可能随机化输入（用于调试/测试）
                set_forward_context(  # 设置前向传播上下文，包含注意力元数据等运行时信息
                    attn_metadata,  # 注意力元数据
                    self.vllm_config,  # vLLM配置
                    num_tokens=num_tokens_padded,  # 填充后的token数量
                    num_tokens_across_dp=num_tokens_across_dp,  # 跨数据并行的token数
                    cudagraph_runtime_mode=cudagraph_runtime_mode,  # CUDA Graph运行时模式
                    batch_descriptor=batch_desc,  # 批次描述符
                    ubatch_slices=ubatch_slices_padded,  # 微批次切片信息
                    slot_mapping=slot_mappings,  # KV缓存的slot映射
                ),  # 结束set_forward_context参数
            ):  # 结束with语句的上下文管理器列表
                outputs = self.model(  # 执行模型的前向传播
                    input_ids=input_ids,  # 输入token ID
                    positions=positions,  # 位置编码
                    intermediate_tensors=intermediate_tensors,  # 流水线并行的中间张量
                    inputs_embeds=inputs_embeds,  # 输入嵌入（可选，用于多模态等场景）
                    **model_kwargs,  # 其他模型参数（如LoRA等）
                )  # 结束模型前向传播调用

            if self.use_aux_hidden_state_outputs:  # 如果使用辅助隐藏状态输出（如MoE模型的辅助损失）
                hidden_states, _ = outputs  # 解包输出，取隐藏状态，忽略辅助输出
            else:  # 否则（标准模型输出）
                hidden_states = outputs  # 直接将输出作为隐藏状态

            if self.speculative_config and (  # 如果启用了推测解码配置
                self.speculative_config.use_eagle()  # 使用Eagle推测解码
                or self.speculative_config.uses_draft_model()  # 或使用草稿模型
                or self.speculative_config.uses_extract_hidden_states()  # 或使用提取隐藏状态方式
            ):  # 结束推测解码条件判断
                assert isinstance(  # 断言drafter是支持的推测解码提议器类型之一
                    self.drafter,  # 当前的草稿模型提议器
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,  # 支持的提议器类型联合
                )  # 结束isinstance断言
                assert self.speculative_config is not None  # 断言推测解码配置不为空
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (  # 判断是否对草稿模型使用CUDA Graph
                    (  # 第一个条件组
                        is_graph_capturing  # 正在进行graph捕获
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE  # 且使用分段CUDA Graph模式
                    )  # 结束第一个条件组
                    or (  # 或第二个条件组
                        not is_graph_capturing  # 不在graph捕获阶段
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE  # 且CUDA Graph模式不是NONE
                    )  # 结束第二个条件组
                ) and not self.speculative_config.enforce_eager  # 且推测解码未强制使用eager模式

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if (  # 如果启用了LoRA特化的CUDA Graph
                    self.compilation_config.cudagraph_specialize_lora  # 配置了cudagraph的LoRA特化
                    and num_active_loras > 0  # 且当前有活跃的LoRA适配器
                ):  # 结束条件判断
                    use_cudagraphs = False  # 禁用CUDA Graph以避免LoRA兼容性问题

                self.drafter.dummy_run(  # 执行草稿模型的虚拟运行（用于预热或graph捕获）
                    num_tokens,  # token数量
                    use_cudagraphs=use_cudagraphs,  # 是否使用CUDA Graph
                    is_graph_capturing=is_graph_capturing,  # 是否正在进行graph捕获
                    slot_mappings=slot_mappings,  # KV缓存的slot映射
                )  # 结束草稿模型的虚拟运行

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()  # 注册逐层的NVTX性能分析钩子（在dynamo追踪完成后注册以避免graph中断）

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:  # 如果不跳过专家并行负载均衡（EPLB）
            self.eplb_step(is_dummy=True, is_profile=is_profile)  # 执行EPLB步骤（虚拟模式，用于数据并行同步）

        logit_indices = np.cumsum(num_scheduled_tokens) - 1  # 计算每个请求最后一个token的索引（用于提取logits）
        logit_indices_device = torch.from_numpy(logit_indices).to(  # 将logit索引转换为GPU上的张量
            self.device, non_blocking=True  # 异步传输到目标设备
        )  # 结束张量创建
        return hidden_states, hidden_states[logit_indices_device]  # 返回完整隐藏状态和每个请求最后token的隐藏状态

    @torch.inference_mode()  # 装饰器：禁用梯度计算以提高推理性能
    def _dummy_sampler_run(  # 定义虚拟采样器运行方法
        self,  # 实例自身
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
    ) -> torch.Tensor:  # 返回采样器输出张量
        """虚拟采样器运行：使用随机隐藏状态运行采样器，用于预热和内存分析。
        避免使用可能包含inf/nan的虚拟隐藏状态导致采样器出错。"""
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config  # 获取多模态配置
        if mm_config and mm_config.mm_encoder_only:  # 如果是纯多模态编码器模型
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])  # 返回空张量，因为编码器模型不需要采样

        hidden_states = torch.rand_like(hidden_states)  # 用随机值替换隐藏状态（避免特殊值）

        logits = self.model.compute_logits(hidden_states)  # 通过模型的LM头计算logits
        num_reqs = logits.size(0)  # 获取请求数量（即batch大小）

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)  # 创建辅助函数：生成填充指定值的张量

        dummy_metadata = SamplingMetadata(  # 创建虚拟的采样元数据
            temperature=dummy_tensors(0.5),  # 温度参数设为0.5
            all_greedy=False,  # 不全部使用贪心采样
            all_random=False,  # 不全部使用随机采样
            top_p=dummy_tensors(0.9),  # top-p采样参数设为0.9
            top_k=dummy_tensors(logits.size(1) - 1),  # top-k采样参数设为词表大小减1
            generators={},  # 空的随机数生成器字典
            max_num_logprobs=None,  # 不计算对数概率
            no_penalties=True,  # 不使用惩罚项
            prompt_token_ids=None,  # 无提示token ID
            frequency_penalties=dummy_tensors(0.1),  # 频率惩罚设为0.1
            presence_penalties=dummy_tensors(0.1),  # 存在惩罚设为0.1
            repetition_penalties=dummy_tensors(0.1),  # 重复惩罚设为0.1
            output_token_ids=[[] for _ in range(num_reqs)],  # 每个请求的输出token ID列表（空）
            spec_token_ids=[[] for _ in range(num_reqs)],  # 每个请求的推测token ID列表（空）
            allowed_token_ids_mask=None,  # 无允许token ID掩码
            bad_words_token_ids={},  # 空的禁用词token ID字典
            logitsprocs=LogitsProcessors(),  # 空的logits处理器
        )  # 结束SamplingMetadata创建
        try:  # 尝试执行采样
            sampler_output = self.sampler(  # 运行采样器
                logits=logits, sampling_metadata=dummy_metadata  # 传入logits和虚拟采样元数据
            )  # 结束采样器调用
        except RuntimeError as e:  # 捕获运行时错误
            if "out of memory" in str(e):  # 如果是GPU内存不足错误
                raise RuntimeError(  # 抛出更友好的错误信息
                    "CUDA out of memory occurred when warming up sampler with "  # 错误描述
                    f"{num_reqs} dummy requests. Please try lowering "  # 提示请求数
                    "`max_num_seqs` or `gpu_memory_utilization` when "  # 建议降低配置参数
                    "initializing the engine."  # 初始化引擎时
                ) from e  # 保留原始异常链
            else:  # 如果是其他运行时错误
                raise e  # 重新抛出原始异常
        if self.speculative_config:  # 如果启用了推测解码
            draft_token_ids = [[0] for _ in range(num_reqs)]  # 为每个请求创建虚拟的草稿token ID
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(  # 创建虚拟的推测解码元数据
                draft_token_ids, self.device  # 传入草稿token ID和设备
            )  # 结束虚拟推测解码元数据创建

            num_tokens = sum(len(ids) for ids in draft_token_ids)  # 计算草稿token总数
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None  # 草稿概率设为None（不使用概率校验）
            logits = torch.randn(  # 创建随机logits用于拒绝采样预热
                num_tokens + num_reqs,  # 行数为草稿token数加请求数
                logits.shape[-1],  # 列数为词表大小
                device=self.device,  # 在目标设备上创建
                dtype=logits.dtype,  # 使用相同数据类型
            )  # 结束随机logits创建
            self.rejection_sampler(  # 运行拒绝采样器（用于推测解码的验证）
                dummy_spec_decode_metadata,  # 虚拟推测解码元数据
                draft_probs,  # 草稿概率（None）
                logits,  # 随机logits
                dummy_metadata,  # 虚拟采样元数据
            )  # 结束拒绝采样器调用
        return sampler_output  # 返回采样器输出

    def _dummy_pooler_run_task(  # 定义单个池化任务的虚拟运行方法
        self,  # 实例自身
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        task: PoolingTask,  # 池化任务类型
    ) -> PoolerOutput:  # 返回池化器输出
        """对指定池化任务执行虚拟运行，用于预热池化器和检测OOM。"""
        num_tokens = hidden_states.shape[0]  # 获取token总数
        max_num_reqs = self.scheduler_config.max_num_seqs  # 获取最大请求数
        num_reqs = min(num_tokens, max_num_reqs)  # 实际请求数取token数和最大请求数的较小值
        min_tokens_per_req = num_tokens // num_reqs  # 计算每个请求的最小token数
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)  # 创建均匀分配的token数数组
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs  # 将剩余token分配给最后一个请求
        assert np.sum(num_scheduled_tokens_np) == num_tokens  # 断言token总数一致
        assert len(num_scheduled_tokens_np) == num_reqs  # 断言请求数一致

        req_num_tokens = num_tokens // num_reqs  # 每个请求的token数（用于构造虚拟输入）

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)  # 将numpy数组转为PyTorch张量作为虚拟提示长度
        dummy_token_ids = torch.zeros(  # 创建全零的虚拟token ID张量
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device  # 形状为(请求数, 每请求token数)
        )  # 结束虚拟token ID创建

        model = cast(VllmModelForPooling, self.get_model())  # 将模型转换为池化模型类型
        dummy_pooling_params = PoolingParams(task=task)  # 创建虚拟池化参数
        dummy_pooling_params.verify(self.model_config)  # 验证池化参数与模型配置的兼容性
        to_update = model.pooler.get_pooling_updates(task)  # 获取指定任务的池化更新配置
        to_update.apply(dummy_pooling_params)  # 将更新应用到虚拟池化参数

        dummy_metadata = PoolingMetadata(  # 创建虚拟池化元数据
            prompt_lens=dummy_prompt_lens,  # 提示长度
            prompt_token_ids=dummy_token_ids,  # 提示token ID
            pooling_params=[dummy_pooling_params] * num_reqs,  # 为每个请求复制池化参数
            pooling_states=[PoolingStates() for i in range(num_reqs)],  # 为每个请求创建池化状态
        )  # 结束PoolingMetadata创建

        dummy_metadata.build_pooling_cursor(  # 构建池化游标（用于定位每个请求的token范围）
            num_scheduled_tokens_np,  # 每个请求的调度token数
            seq_lens_cpu=dummy_prompt_lens,  # CPU上的序列长度
            device=hidden_states.device,  # 目标设备
        )  # 结束构建池化游标

        try:  # 尝试执行池化操作
            return model.pooler(  # 运行池化器
                hidden_states=hidden_states, pooling_metadata=dummy_metadata  # 传入隐藏状态和池化元数据
            )  # 结束池化器调用
        except RuntimeError as e:  # 捕获运行时错误
            if "out of memory" in str(e):  # 如果是GPU内存不足
                raise RuntimeError(  # 抛出更友好的错误信息
                    "CUDA out of memory occurred when warming up pooler "  # 错误描述
                    f"({task=}) with {num_reqs} dummy requests. Please try "  # 提示任务类型和请求数
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "  # 建议降低配置参数
                    "initializing the engine."  # 初始化引擎时
                ) from e  # 保留原始异常链
            else:  # 如果是其他运行时错误
                raise e  # 重新抛出原始异常

    @torch.inference_mode()  # 装饰器：禁用梯度计算以提高推理性能
    def _dummy_pooler_run(  # 定义虚拟池化器运行方法（对所有支持的池化任务进行预热）
        self,  # 实例自身
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
    ) -> PoolerOutput:  # 返回池化器输出
        """虚拟池化器运行：对所有支持的池化任务进行预热运行，
        找出输出最大的任务并返回其结果，确保后续步骤有足够的内存。"""
        mm_config = self.vllm_config.model_config.multimodal_config  # 获取多模态配置
        if mm_config and mm_config.mm_encoder_only:  # 如果是纯多模态编码器模型
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])  # 返回空张量

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()  # 获取模型支持的池化任务列表

        if not supported_pooling_tasks:  # 如果没有支持的池化任务
            raise RuntimeError(  # 抛出运行时错误
                f"Model {self.model_config.model} does not support "  # 错误描述包含模型名称
                "any pooling tasks. See "  # 提示参考文档
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "  # 文档链接
                "to learn more."  # 了解更多信息
            )  # 结束RuntimeError

        output_size = dict[PoolingTask, float]()  # 创建字典记录每个池化任务的输出大小
        for task in supported_pooling_tasks:  # 遍历所有支持的池化任务
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)  # 对每个任务运行虚拟池化
            output_size[task] = sum(o.nbytes for o in output if o is not None)  # 计算输出的字节数
            del output  # Allow GC  # 删除输出以允许垃圾回收

        max_task = max(output_size.items(), key=lambda x: x[1])[0]  # 找到输出最大的池化任务
        return self._dummy_pooler_run_task(hidden_states, max_task)  # 返回最大输出任务的运行结果

    # 内存 profiling 运行：使用最大 batch size 的虚拟输入执行前向传播，
    # 测量模型的峰值 GPU 内存使用量，包括多模态编码器的内存开销。
    def profile_run(self) -> None:  # 定义内存分析运行方法
        """执行内存分析运行：使用最大batch size的虚拟输入测量GPU内存峰值，
        包括多模态编码器和编码器缓存的内存开销。"""
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:  # 如果模型支持多模态输入
            mm_config = self.model_config.multimodal_config  # 获取多模态配置
            if mm_config is not None and mm_config.skip_mm_profiling:  # 如果配置跳过多模态分析
                logger.info(  # 记录日志
                    "Skipping memory profiling for multimodal encoder and "  # 跳过多模态编码器内存分析
                    "encoder cache."  # 和编码器缓存
                )  # 结束日志记录
            else:  # 否则（需要进行多模态内存分析）
                mm_budget = self.mm_budget  # 获取多模态预算
                assert mm_budget is not None  # 断言多模态预算不为空

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:  # 如果编码器预算大于0
                    if not mm_budget.mm_max_toks_per_item:  # 如果所有模态的最大token数为0（纯嵌入模式）
                        # All modality limits are 0 — embedding-only mode.
                        # Budget is non-zero for embedding storage, but
                        # there's no encoder to profile.
                        logger.info(  # 记录日志
                            "Skipping encoder profiling for embedding-only "  # 跳过编码器分析
                            "mode (all modality limits=0 with "  # 纯嵌入模式
                            "enable_mm_embeds=True).",  # 启用多模态嵌入
                        )  # 结束日志记录
                    else:  # 否则（有编码器需要分析）
                        # NOTE: Currently model is profiled with a single
                        # non-text modality with the max possible input
                        # tokens even when it supports multiple.
                        dummy_modality = mm_budget.get_modality_with_max_tokens()  # 获取token数最多的模态
                        max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[  # 获取每批次最大多模态项数
                            dummy_modality  # 使用token最多的模态
                        ]  # 结束获取最大项数

                        logger.info(  # 记录编码器缓存初始化信息
                            "Encoder cache will be initialized with a "  # 编码器缓存将初始化
                            "budget of %s tokens, and profiled with "  # 预算token数
                            "%s %s items of the maximum feature size.",  # 最大特征尺寸的项数
                            encoder_budget,  # 编码器预算值
                            max_mm_items_per_batch,  # 每批次最大项数
                            dummy_modality,  # 模态类型
                        )  # 结束日志记录

                        # Create dummy batch of multimodal inputs.
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(  # 创建虚拟的多模态输入批次
                            dummy_modality,  # 模态类型
                            max_mm_items_per_batch,  # 最大项数
                        )  # 结束虚拟多模态输入创建

                        # Run multimodal encoder.
                        dummy_encoder_outputs = self.model.embed_multimodal(  # 运行多模态编码器
                            **batched_dummy_mm_inputs  # 传入虚拟多模态输入
                        )  # 结束多模态编码器运行

                        sanity_check_mm_encoder_outputs(  # 对编码器输出进行合理性检查
                            dummy_encoder_outputs,  # 编码器输出
                            expected_num_items=max_mm_items_per_batch,  # 期望的输出项数
                        )  # 结束合理性检查
                        for i, output in enumerate(dummy_encoder_outputs):  # 遍历编码器输出
                            self.encoder_cache[f"tmp_{i}"] = output  # 将编码器输出存入缓存（临时键名）

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(  # 执行虚拟运行，获取隐藏状态
            self.max_num_tokens, is_profile=True  # 使用最大token数，标记为分析模式
        )  # 结束虚拟运行
        if get_pp_group().is_last_rank:  # 如果是流水线并行的最后一个rank
            if self.is_pooling_model:  # 如果是池化模型
                output = self._dummy_pooler_run(hidden_states)  # 执行虚拟池化器运行
            else:  # 否则（生成模型）
                output = self._dummy_sampler_run(last_hidden_states)  # 执行虚拟采样器运行
        else:  # 如果不是最后一个rank
            output = None  # 输出为空（中间rank不需要采样/池化）
        self._sync_device()  # 同步设备（等待所有操作完成）
        del hidden_states, output  # 删除隐藏状态和输出以释放内存
        self.encoder_cache.clear()  # 清空编码器缓存
        gc.collect()  # 强制执行垃圾回收

    def _init_minimal_kv_cache_for_profiling(self) -> None:  # 为CUDA Graph分析初始化最小KV缓存
        """为CUDA Graph内存分析初始化最小的KV缓存，
        只分配足够捕获CUDA Graph所需的最少block数量。"""
        from vllm.v1.core.kv_cache_utils import (  # 从KV缓存工具模块导入
            get_kv_cache_config_from_groups,  # 从KV缓存组获取配置的函数
            get_kv_cache_groups,  # 获取KV缓存组的函数
        )  # 结束导入

        kv_cache_spec = self.get_kv_cache_spec()  # 获取KV缓存规格
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)  # 根据配置获取KV缓存组
        min_blocks = self.compilation_config.max_cudagraph_capture_size or 1  # 最小block数为最大graph捕获大小或1

        # Temporarily change num_gpu_blocks_override to allocate a minimal KV cache
        saved_override = self.cache_config.num_gpu_blocks_override  # 保存原始的GPU block覆盖值
        self.cache_config.num_gpu_blocks_override = min_blocks  # 临时设置为最小block数
        minimal_config = get_kv_cache_config_from_groups(  # 使用最小block数生成KV缓存配置
            self.vllm_config, kv_cache_groups, available_memory=0  # 可用内存设为0（使用override值）
        )  # 结束最小KV缓存配置生成
        self.cache_config.num_gpu_blocks_override = saved_override  # 恢复原始的GPU block覆盖值

        self.initialize_kv_cache(minimal_config)  # 使用最小配置初始化KV缓存
        self.cache_config.num_gpu_blocks = minimal_config.num_blocks  # 设置GPU block数为最小配置的block数

        logger.debug("Initialized minimal KV cache for CUDA graph profiling")  # 记录调试日志

    @staticmethod  # 静态方法装饰器
    @contextmanager  # 上下文管理器装饰器
    def _freeze_gc():  # 定义GC冻结上下文管理器
        """冻结垃圾回收器的上下文管理器，在CUDA Graph捕获期间避免GC干扰。
        通过gc.freeze()将当前对象移到永久代，退出时解冻并回收。"""
        gc.collect()  # 先执行一次垃圾回收
        should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC  # 判断是否需要冻结GC（默认冻结）
        if should_freeze:  # 如果需要冻结
            gc.freeze()  # 冻结GC（将当前对象移到永久代）
        try:  # 尝试执行上下文中的代码
            yield  # 让出控制权给with块中的代码
        finally:  # 最终清理
            if should_freeze:  # 如果之前冻结了GC
                gc.unfreeze()  # 解冻GC
                gc.collect()  # 执行垃圾回收

    def _cleanup_profiling_kv_cache(self) -> None:  # 清理分析用的KV缓存
        """清理CUDA Graph分析期间创建的临时KV缓存和相关资源，释放GPU内存。"""
        torch.accelerator.synchronize()  # 同步加速器设备
        if hasattr(self, "kv_caches") and self.kv_caches:  # 如果存在KV缓存
            for i in range(len(self.kv_caches)):  # 遍历所有KV缓存
                self.kv_caches[i] = None  # type: ignore  # 将每个KV缓存设为None
            self.kv_caches.clear()  # 清空KV缓存列表
        if hasattr(self, "cross_layers_kv_cache"):  # 如果存在交叉层KV缓存
            self.cross_layers_kv_cache = None  # 将交叉层KV缓存设为None
            self.cross_layers_attn_backend = None  # 将交叉层注意力后端设为None
        if hasattr(self, "attn_groups"):  # 如果存在注意力组
            self.attn_groups.clear()  # 清空注意力组
        if hasattr(self, "kv_cache_config"):  # 如果存在KV缓存配置
            delattr(self, "kv_cache_config")  # 删除KV缓存配置属性
        self.cache_config.num_gpu_blocks = None  # 重置GPU block数为None

        for layer in self.compilation_config.static_forward_context.values():  # 遍历静态前向上下文中的所有层
            if hasattr(layer, "kv_cache"):  # 如果层有kv_cache属性
                layer.kv_cache = []  # 清空层的KV缓存

        gc.collect()  # 执行垃圾回收
        torch.accelerator.empty_cache()  # 清空加速器的缓存

        logger.debug("Cleaned up profiling KV cache and CUDA graphs")  # 记录调试日志

    @torch.inference_mode()  # 装饰器：禁用梯度计算以提高推理性能
    # 估算 CUDA Graph 捕获所需的 GPU 内存。
    # 通过对少量代表性 batch size 进行试捕获并测量内存增长，
    # 外推出全部 graph 的内存开销，用于更精确的 KV 缓存内存规划。
    def profile_cudagraph_memory(self) -> int:  # 定义CUDA Graph内存分析方法，返回估算的内存字节数
        """分析CUDA Graph捕获所需的GPU内存。
        通过对少量代表性batch size进行试捕获并测量内存增长，
        外推出全部graph的内存开销，用于更精确的KV缓存内存规划。"""
        with set_current_vllm_config(self.vllm_config):  # 设置当前vLLM配置上下文
            self._init_minimal_kv_cache_for_profiling()  # 初始化最小KV缓存用于分析

        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured  # 保存当前已捕获的CUDA Graph数量

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()  # 获取需要捕获的CUDA Graph描述列表

        total_graphs = sum(len(descs) for _, descs in capture_descs)  # 计算需要捕获的Graph总数
        if total_graphs == 0:  # 如果不需要捕获任何Graph
            logger.debug("No CUDA graphs will be captured, skipping profiling")  # 记录调试日志
            self._cleanup_profiling_kv_cache()  # 清理分析用KV缓存
            return 0  # 返回0字节

        logger.info(  # 记录信息日志
            "Profiling CUDA graph memory: %s",  # 分析CUDA Graph内存
            ", ".join(  # 将各模式的描述用逗号连接
                f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"  # 格式化每种模式的Graph数和最大token数
                for mode, descs in capture_descs  # 遍历所有捕获描述
                if descs  # 只包含非空的描述
            ),  # 结束join
        )  # 结束日志记录

        # Use a temporary pool for profiling to avoid fragmentation in the main pool.
        profiling_pool = current_platform.graph_pool_handle()  # 创建临时的Graph内存池句柄
        original_pools: dict[int, Any] = {}  # 保存原始内存池的字典
        for instance in list(CUDAGraphWrapper._all_instances):  # 遍历所有CUDAGraphWrapper实例
            original_pools[id(instance)] = instance.graph_pool  # 保存每个实例的原始内存池
            instance.graph_pool = profiling_pool  # 替换为分析用的临时内存池

        set_cudagraph_capturing_enabled(True)  # 全局启用CUDA Graph捕获
        with self._freeze_gc(), graph_capture(device=self.device):  # 冻结GC并进入Graph捕获上下文
            shared_memory_estimate = {}  # 共享内存估算字典（每种模式）
            per_graph_estimate = {}  # 每Graph内存估算字典（每种模式）
            torch.accelerator.synchronize()  # 同步加速器设备
            torch.accelerator.empty_cache()  # 清空缓存

            for mode, descs in capture_descs:  # 遍历每种Graph模式的描述
                profile_descs = descs[:2]  # 只取前2个描述进行分析（代表性样本）
                mem_samples: list[int] = []  # 内存样本列表

                for i, desc in enumerate(profile_descs):  # 遍历分析描述
                    mem_before = torch.cuda.mem_get_info()[0]  # 记录捕获前的可用GPU内存
                    self._warmup_and_capture(  # 预热并捕获CUDA Graph
                        desc,  # 批次描述符
                        cudagraph_runtime_mode=mode,  # CUDA Graph运行时模式
                        profile_seq_lens=(  # 分析用的序列长度
                            min(  # 取较小值
                                self.max_model_len,  # 最大模型长度
                                self.max_num_tokens // desc.num_tokens,  # 或最大token数除以当前描述的token数
                            )  # 结束min
                            if mode == CUDAGraphMode.FULL and i == 0  # 仅在FULL模式的第一次捕获时设置
                            else None  # 否则不设置
                        ),  # 结束profile_seq_lens
                    )  # 结束预热和捕获
                    torch.accelerator.synchronize()  # 同步加速器设备
                    free_after = torch.cuda.mem_get_info()[0]  # 记录捕获后的可用GPU内存
                    mem_samples.append(mem_before - free_after)  # 计算并记录内存增长量

                first_capture = mem_samples[0]  # 第一次捕获的内存消耗（包含共享开销）
                # Use at least 1 MiB per graph for driver overhead
                per_graph = max(mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20)  # 每Graph内存消耗（至少1MiB）

                shared_memory_estimate[mode] = first_capture  # 记录该模式的共享内存估算
                per_graph_estimate[mode] = per_graph * (len(descs) - 1)  # 估算剩余Graph的总内存

                logger.debug(  # 记录调试日志
                    "Estimated %s CUDA graph memory: "  # 估算的CUDA Graph内存
                    "%.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",  # 首次捕获加每Graph开销
                    mode.name,  # 模式名称
                    first_capture / (1 << 20),  # 首次捕获内存（MiB）
                    len(descs),  # Graph总数
                    per_graph / (1 << 20),  # 每Graph内存（MiB）
                )  # 结束调试日志

        set_cudagraph_capturing_enabled(False)  # 全局禁用CUDA Graph捕获
        CUDAGraphWrapper.clear_all_graphs()  # 清除所有已捕获的CUDA Graph
        for instance in list(CUDAGraphWrapper._all_instances):  # 遍历所有CUDAGraphWrapper实例
            if id(instance) in original_pools:  # 如果实例有保存的原始内存池
                instance.graph_pool = original_pools[id(instance)]  # 恢复原始内存池
        for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():  # 遍历所有CUDA Graph键集合
            key_set.clear()  # 清空键集合
        self.cudagraph_dispatcher.keys_initialized = False  # 重置键初始化标志
        self.maybe_remove_all_loras(self.lora_config)  # 可能移除所有LoRA适配器
        self._cleanup_profiling_kv_cache()  # 清理分析用的KV缓存
        compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured  # 恢复已捕获Graph数量计数

        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        total_estimate = max(shared_memory_estimate.values()) + sum(  # 总估算：取共享内存最大值加所有每Graph内存之和
            per_graph_estimate.values()  # 所有模式的每Graph内存估算
        )  # 结束总估算计算
        logger.info(  # 记录信息日志
            "Estimated CUDA graph memory: %.2f GiB total",  # 估算的CUDA Graph总内存
            total_estimate / (1 << 30),  # 转换为GiB
        )  # 结束日志记录

        return int(total_estimate)  # 返回总估算内存字节数

    # CUDA Graph 捕获：对配置的 batch size 列表（从大到小）捕获前向传播的 CUDA Graph。
    # 支持 piecewise（分段）和 full（完整）两种模式，以及微批次的 graph 捕获。
    # 返回实际消耗的 CUDA Graph 内存量。
    @instrument(span_name="Capture model")  # 装饰器：为CUDA Graph捕获添加性能追踪span
    def capture_model(self) -> int:  # 定义模型CUDA Graph捕获方法，返回消耗的GPU内存字节数
        """捕获模型的CUDA Graph：对配置的batch size列表从大到小捕获前向传播Graph，
        支持PIECEWISE和FULL两种模式，返回实际消耗的CUDA Graph内存量。"""
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:  # 如果CUDA Graph模式为NONE
            logger.warning(  # 记录警告日志
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "  # 跳过CUDA Graph捕获
                "ensure `cudagraph_mode` was not manually set to `NONE`"  # 提示检查配置
            )  # 结束警告日志
            return 0  # 返回0字节

        compilation_counter.num_gpu_runner_capture_triggers += 1  # 递增GPU运行器捕获触发计数

        start_time = time.perf_counter()  # 记录捕获开始时间

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)  # 全局启用CUDA Graph捕获
        with self._freeze_gc(), graph_capture(device=self.device):  # 冻结GC并进入Graph捕获上下文
            torch.accelerator.synchronize()  # 同步加速器设备
            torch.accelerator.empty_cache()  # 清空缓存
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]  # 记录捕获前的可用GPU内存

            for (  # 遍历所有需要捕获的模式和批次描述
                runtime_mode,  # CUDA Graph运行时模式
                batch_descs,  # 该模式下的批次描述列表
            ) in self.cudagraph_dispatcher.get_capture_descs():  # 获取捕获描述
                self._capture_cudagraphs(  # 捕获该模式的所有CUDA Graph
                    batch_descriptors=batch_descs,  # 批次描述列表
                    cudagraph_runtime_mode=runtime_mode,  # 运行时模式
                )  # 结束捕获调用
                torch.accelerator.synchronize()  # 同步加速器设备

            torch.accelerator.synchronize()  # 再次同步确保所有捕获完成
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]  # 记录捕获后的可用GPU内存

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)  # 全局禁用CUDA Graph捕获

        torch.accelerator.synchronize()  # 同步加速器设备
        torch.accelerator.empty_cache()  # 清空缓存

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()  # 锁定工作空间大小，防止执行期间重新分配

        end_time = time.perf_counter()  # 记录捕获结束时间
        elapsed_time = end_time - start_time  # 计算捕获耗时
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory  # 计算CUDA Graph消耗的GPU内存
        # This usually takes 5~20 seconds.
        logger.info_once(  # 记录一次性信息日志
            "Graph capturing finished in %.0f secs, took %.2f GiB",  # Graph捕获完成的耗时和内存
            elapsed_time,  # 耗时（秒）
            cuda_graph_size / (1 << 30),  # 内存消耗（GiB）
            scope="local",  # 仅在本地rank记录
        )  # 结束日志记录
        return cuda_graph_size  # 返回CUDA Graph消耗的GPU内存字节数

    def _warmup_and_capture(  # 定义预热并捕获单个CUDA Graph的方法
        self,  # 实例自身
        desc: BatchDescriptor,  # 批次描述符（包含token数等信息）
        cudagraph_runtime_mode: CUDAGraphMode,  # CUDA Graph运行时模式
        profile_seq_lens: int | None = None,  # 分析用的序列长度（可选）
        allow_microbatching: bool = False,  # 是否允许微批次处理
        num_warmups: int | None = None,  # 预热次数（可选）
    ):  # 无返回值
        """预热并捕获单个CUDA Graph：先执行指定次数的预热运行，
        然后执行一次Graph捕获运行。"""
        if num_warmups is None:  # 如果未指定预热次数
            num_warmups = self.compilation_config.cudagraph_num_of_warmups  # 使用配置中的默认预热次数
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL  # FULL模式需要强制执行注意力计算
        for _ in range(num_warmups):  # 执行指定次数的预热
            self._dummy_run(  # 执行虚拟运行作为预热
                desc.num_tokens,  # 使用描述符中的token数
                cudagraph_runtime_mode=CUDAGraphMode.NONE,  # 预热时不使用CUDA Graph
                force_attention=force_attention,  # 是否强制注意力计算
                uniform_decode=desc.uniform,  # 是否为统一解码
                allow_microbatching=allow_microbatching,  # 是否允许微批次
                skip_eplb=True,  # 跳过专家并行负载均衡
                remove_lora=False,  # 不移除LoRA
                num_active_loras=desc.num_active_loras,  # 活跃LoRA数量
            )  # 结束预热虚拟运行
        self._dummy_run(  # 执行实际的Graph捕获运行
            desc.num_tokens,  # 使用描述符中的token数
            cudagraph_runtime_mode=cudagraph_runtime_mode,  # 使用指定的CUDA Graph模式
            uniform_decode=desc.uniform,  # 是否为统一解码
            allow_microbatching=allow_microbatching,  # 是否允许微批次
            skip_eplb=True,  # 跳过专家并行负载均衡
            remove_lora=False,  # 不移除LoRA
            num_active_loras=desc.num_active_loras,  # 活跃LoRA数量
            is_graph_capturing=True,  # 标记为正在进行Graph捕获
            profile_seq_lens=profile_seq_lens,  # 分析用的序列长度
        )  # 结束Graph捕获运行

    def _capture_cudagraphs(  # 定义批量CUDA Graph捕获方法
        self,  # 实例自身
        batch_descriptors: list[BatchDescriptor],  # 批次描述符列表（从大到小排序）
        cudagraph_runtime_mode: CUDAGraphMode,  # CUDA Graph运行时模式
    ):  # 无返回值
        """批量捕获CUDA Graph：对给定的批次描述符列表逐个进行预热和捕获，
        支持微批次和LoRA特化。"""
        assert (  # 断言CUDA Graph模式有效
            cudagraph_runtime_mode != CUDAGraphMode.NONE  # 模式不能为NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()  # 且必须是有效的运行时模式
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"  # 错误信息

        if not batch_descriptors:  # 如果批次描述符列表为空
            return  # 直接返回

        uniform_decode = batch_descriptors[0].uniform  # 获取是否为统一解码模式

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():  # 如果是全局第一个rank
            batch_descriptors = tqdm(  # 用进度条包装批次描述符列表
                batch_descriptors,  # 原始列表
                disable=not self.load_config.use_tqdm_on_load,  # 根据配置决定是否禁用进度条
                desc="Capturing CUDA graphs ({}, {})".format(  # 进度条描述
                    "decode" if uniform_decode else "mixed prefill-decode",  # 解码类型
                    cudagraph_runtime_mode.name,  # Graph模式名称
                ),  # 结束format
            )  # 结束tqdm包装

        # We skip EPLB here since we don't want to record dummy metrics
        for batch_desc in batch_descriptors:  # 遍历每个批次描述符
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (  # 判断是否允许微批次处理
                self.parallel_config.use_ubatching  # 配置启用了微批次
                and cudagraph_runtime_mode == CUDAGraphMode.FULL  # 且使用FULL模式
                and uniform_decode  # 且是统一解码
                and check_ubatch_thresholds(  # 且token数超过微批次阈值
                    config=self.vllm_config.parallel_config,  # 并行配置
                    num_tokens=batch_desc.num_tokens,  # 当前token数
                    uniform_decode=uniform_decode,  # 是否统一解码
                )  # 结束阈值检查
            )  # 结束微批次判断
            self._warmup_and_capture(  # 预热并捕获该批次的CUDA Graph
                batch_desc,  # 批次描述符
                cudagraph_runtime_mode=cudagraph_runtime_mode,  # CUDA Graph模式
                allow_microbatching=allow_microbatching,  # 是否允许微批次
            )  # 结束预热和捕获
            torch.accelerator.synchronize()  # 同步加速器设备
        self.maybe_remove_all_loras(self.lora_config)  # 捕获完成后可能移除所有LoRA适配器

    # 初始化注意力后端：为每个 KV 缓存组选择合适的注意力后端实现，
    # 创建 AttentionGroup 并关联 KV 缓存规格和层名称。
    # 支持混合架构中不同层使用不同的注意力后端。
    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:  # 定义初始化注意力后端方法
        """初始化注意力后端和注意力元数据构建器。
        为每个KV缓存组选择合适的注意力后端，创建AttentionGroup并关联KV缓存规格和层名称。"""
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"  # 断言注意力后端未被重复初始化

        class AttentionGroupKey(NamedTuple):  # 定义注意力组键的命名元组类
            """注意力组键：用于唯一标识一个注意力后端和KV缓存规格的组合。"""
            attn_backend: type[AttentionBackend]  # 注意力后端类型
            kv_cache_spec: KVCacheSpec  # KV缓存规格

        def get_attn_backends_for_group(  # 定义获取KV缓存组的注意力后端的内部函数
            kv_cache_group_spec: KVCacheGroupSpec,  # KV缓存组规格
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:  # 返回注意力后端映射和后端集合
            """获取指定KV缓存组中每层对应的注意力后端，按后端类型和KV缓存规格分组。"""
            layer_type = cast(type[Any], AttentionLayerBase)  # 将注意力层基类转换为通用类型
            layers = get_layers_from_vllm_config(  # 从vLLM配置中获取指定层名的注意力层
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names  # 配置、层类型、层名列表
            )  # 结束获取层
            attn_backends = {}  # 注意力后端字典（键为(类名, KV缓存规格)）
            attn_backend_layers = defaultdict(list)  # 默认字典：每个注意力后端对应的层名列表
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:  # 遍历KV缓存组中的每个层名
                attn_backend = layers[layer_name].get_attn_backend()  # 获取该层的注意力后端

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:  # 如果该层支持KV共享快速预填充
                    attn_backend = create_fast_prefill_custom_backend(  # 创建快速预填充自定义后端
                        "FastPrefill",  # 后端名称前缀
                        attn_backend,  # type: ignore[arg-type]  # 原始注意力后端
                    )  # 结束创建快速预填充后端

                full_cls_name = attn_backend.full_cls_name()  # 获取注意力后端的完整类名
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec  # 获取KV缓存组的缓存规格
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):  # 如果是统一类型但每层不同的KV缓存规格
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]  # 获取该层特定的KV缓存规格
                key = (full_cls_name, layer_kv_cache_spec)  # 构建去重键（类名+KV缓存规格）
                attn_backends[key] = AttentionGroupKey(  # 存储注意力组键
                    attn_backend, layer_kv_cache_spec  # 注意力后端和KV缓存规格
                )  # 结束AttentionGroupKey创建
                attn_backend_layers[key].append(layer_name)  # 将层名添加到对应后端的列表中
            return (  # 返回元组
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},  # 将键从字符串替换为AttentionGroupKey
                set(group_key.attn_backend for group_key in attn_backends.values()),  # 所有唯一的注意力后端类型集合
            )  # 结束返回

        def create_attn_groups(  # 定义创建注意力组的内部函数
            attn_backends_map: dict[AttentionGroupKey, list[str]],  # 注意力后端到层名的映射
            kv_cache_group_id: int,  # KV缓存组ID
        ) -> list[AttentionGroup]:  # 返回注意力组列表
            """根据注意力后端映射创建AttentionGroup列表。"""
            attn_groups: list[AttentionGroup] = []  # 初始化注意力组列表
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():  # 遍历注意力后端映射
                attn_group = AttentionGroup(  # 创建注意力组
                    attn_backend,  # 注意力后端类型
                    layer_names,  # 属于该组的层名列表
                    kv_cache_spec,  # KV缓存规格
                    kv_cache_group_id,  # KV缓存组ID
                )  # 结束AttentionGroup创建

                attn_groups.append(attn_group)  # 将注意力组添加到列表
            return attn_groups  # 返回注意力组列表

        attention_backend_maps = []  # 所有KV缓存组的注意力后端映射列表
        attention_backend_list = []  # 所有KV缓存组的注意力后端类型列表
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:  # 遍历每个KV缓存组规格
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)  # 获取该组的注意力后端
            attention_backend_maps.append(attn_backends[0])  # 添加后端到层名的映射
            attention_backend_list.append(attn_backends[1])  # 添加后端类型集合

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(  # 在初始化元数据构建器前解析CUDA Graph模式
            attention_backend_list, kv_cache_config.kv_cache_groups  # 传入注意力后端列表和KV缓存组
        )  # 结束CUDA Graph模式检查和更新

        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)  # 检查注意力后端是否支持上下文并行等特性

        for i, attn_backend_map in enumerate(attention_backend_maps):  # 遍历每个KV缓存组的注意力后端映射
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))  # 创建注意力组并添加到列表

    def initialize_metadata_builders(  # 定义初始化元数据构建器方法
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]  # KV缓存配置和内核block大小列表
    ) -> None:  # 无返回值
        """为所有KV缓存组和注意力组创建元数据构建器。
        构建器负责在运行时构建注意力元数据，用于指导注意力计算内核。"""
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):  # 遍历每个KV缓存组
            for attn_group in self.attn_groups[kv_cache_group_id]:  # 遍历该KV缓存组中的每个注意力组
                attn_group.create_metadata_builders(  # 为注意力组创建元数据构建器
                    self.vllm_config,  # vLLM配置
                    self.device,  # 目标设备
                    kernel_block_sizes[kv_cache_group_id]  # 该KV缓存组的内核block大小
                    if kv_cache_group_id < len(kernel_block_sizes)  # 如果索引在范围内
                    else None,  # 否则为None
                    num_metadata_builders=1  # 元数据构建器数量
                    if not self.parallel_config.use_ubatching  # 如果不使用微批次则为1
                    else self.parallel_config.num_ubatches,  # 否则为微批次数
                )  # 结束创建元数据构建器
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()  # 计算重排序批次阈值

        # Initialize drafter attention backend
        if self.speculative_config and (  # 如果启用了推测解码
            self.speculative_config.use_eagle()  # 使用Eagle推测解码
            or self.speculative_config.uses_draft_model()  # 或使用草稿模型
        ):  # 结束推测解码条件判断
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)  # 断言草稿模型是支持的类型
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)  # 初始化草稿模型的注意力后端

    def _check_and_update_cudagraph_mode(  # 定义检查和更新CUDA Graph模式的方法
        self,  # 实例自身
        attention_backends: list[set[type[AttentionBackend]]],  # 注意力后端类型集合列表
        kv_cache_groups: list[KVCacheGroupSpec],  # KV缓存组规格列表
    ) -> None:  # 无返回值
        """解析CUDA Graph模式：当存在多个注意力组且可能有冲突的CUDA Graph支持时，
        解析并确定最终的cudagraph_mode，然后初始化cudagraph_dispatcher。"""
        min_cg_support = AttentionCGSupport.ALWAYS  # 初始化最小CUDA Graph支持级别为ALWAYS（最高级）
        min_cg_backend_name = None  # 初始化限制最严格的后端名称

        for attn_backend_set, kv_cache_group in zip(  # 同时遍历注意力后端集合和KV缓存组
            attention_backends, kv_cache_groups  # 两个列表的对应元素
        ):  # 结束zip
            for attn_backend in attn_backend_set:  # 遍历每个注意力后端
                builder_cls = attn_backend.get_builder_cls()  # 获取元数据构建器类

                cg_support = builder_cls.get_cudagraph_support(  # 获取该构建器对CUDA Graph的支持级别
                    self.vllm_config, kv_cache_group.kv_cache_spec  # 传入vllm配置和KV缓存规格，获取CUDAGraph支持级别
                )  # 获取该注意力后端对CUDAGraph的支持程度
                if cg_support.value < min_cg_support.value:  # 如果当前后端的支持级别低于已知最小值
                    min_cg_support = cg_support  # 更新最小CUDAGraph支持级别
                    min_cg_backend_name = attn_backend.__name__  # 记录支持级别最低的后端名称
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode  # 获取编译配置中的CUDAGraph模式
        assert cudagraph_mode is not None  # 断言CUDAGraph模式不为空
        # check cudagraph for mixed batch is supported
        if (  # 检查混合批次的CUDAGraph是否受支持
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL  # 如果混合模式为FULL
            and min_cg_support != AttentionCGSupport.ALWAYS  # 且最小支持级别不是ALWAYS
        ):
            msg = (  # 构建错误/警告消息
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "  # 提示当前CUDAGraph模式不受支持
                f"with {min_cg_backend_name} backend (support: "  # 指出不兼容的后端名称
                f"{min_cg_support})"  # 显示实际支持级别
            )
            if min_cg_support == AttentionCGSupport.NEVER:  # 如果完全不支持CUDAGraph
                # if not supported any full cudagraphs, just raise it.
                msg += (  # 追加建议信息
                    "; please try cudagraph_mode=PIECEWISE, and "  # 建议尝试PIECEWISE模式
                    "make sure compilation mode is VLLM_COMPILE"  # 并确保使用VLLM_COMPILE编译模式
                )
                raise ValueError(msg)  # 抛出值错误异常

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():  # 如果分割操作包含注意力操作
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"  # 追加提示信息，降级为FULL_AND_PIECEWISE
                cudagraph_mode = self.compilation_config.cudagraph_mode = (  # 更新CUDAGraph模式
                    CUDAGraphMode.FULL_AND_PIECEWISE  # 设置为FULL和PIECEWISE混合模式
                )
            else:  # 如果分割操作不包含注意力操作
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"  # 追加提示信息，降级为FULL_DECODE_ONLY
                cudagraph_mode = self.compilation_config.cudagraph_mode = (  # 更新CUDAGraph模式
                    CUDAGraphMode.FULL_DECODE_ONLY  # 设置为仅解码阶段使用FULL模式
                )
            logger.warning(msg)  # 输出警告日志

        # check that if we are doing decode full-cudagraphs it is supported
        if (  # 检查解码阶段的FULL CUDAGraph是否受支持
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL  # 如果解码模式为FULL
            and min_cg_support == AttentionCGSupport.NEVER  # 且最小支持级别为NEVER（完全不支持）
        ):
            msg = (  # 构建错误/警告消息
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "  # 提示当前模式不受支持
                f"with {min_cg_backend_name} backend (support: "  # 指出不兼容的后端
                f"{min_cg_support})"  # 显示支持级别
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (  # 如果编译模式为VLLM_COMPILE且
                self.compilation_config.splitting_ops_contain_attention()  # 分割操作包含注意力操作
                or self.compilation_config.use_inductor_graph_partition  # 或使用inductor图分区
            ):
                msg += (  # 追加提示信息
                    "; setting cudagraph_mode=PIECEWISE because "  # 降级为PIECEWISE
                    "attention is compiled piecewise"  # 因为注意力是分片编译的
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (  # 更新CUDAGraph模式
                    CUDAGraphMode.PIECEWISE  # 设置为PIECEWISE模式
                )
            else:  # 否则（注意力不是分片编译的）
                msg += (  # 追加提示信息
                    "; setting cudagraph_mode=NONE because "  # 降级为NONE
                    "attention is not compiled piecewise"  # 因为注意力未分片编译
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (  # 更新CUDAGraph模式
                    CUDAGraphMode.NONE  # 完全禁用CUDAGraph
                )
            logger.warning(msg)  # 输出警告日志

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (  # 检查推测解码 + FULL CUDAGraph是否兼容
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL  # 如果解码模式为FULL
            and self.uniform_decode_query_len > 1  # 且统一解码查询长度大于1（推测解码场景）
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value  # 且支持级别不足以处理统一批次
        ):
            msg = (  # 构建警告消息
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"  # 提示模式不受支持
                f" with spec-decode for attention backend "  # 在推测解码场景下
                f"{min_cg_backend_name} (support: {min_cg_support})"  # 显示后端名和支持级别
            )
            if self.compilation_config.splitting_ops_contain_attention():  # 如果分割操作包含注意力
                msg += "; setting cudagraph_mode=PIECEWISE"  # 降级为PIECEWISE
                cudagraph_mode = self.compilation_config.cudagraph_mode = (  # 更新CUDAGraph模式
                    CUDAGraphMode.PIECEWISE  # 设置为PIECEWISE模式
                )
            else:  # 否则
                msg += "; setting cudagraph_mode=NONE"  # 降级为NONE
                cudagraph_mode = self.compilation_config.cudagraph_mode = (  # 更新CUDAGraph模式
                    CUDAGraphMode.NONE  # 完全禁用CUDAGraph
                )
            logger.warning(msg)  # 输出警告日志

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (  # 最终检查：经过自动降级后，如果仍包含FULL CUDAGraph
            cudagraph_mode.has_full_cudagraphs()  # 当前模式包含FULL CUDAGraph
            and min_cg_support == AttentionCGSupport.NEVER  # 但完全不受支持
        ):
            raise ValueError(  # 抛出值错误异常
                f"CUDAGraphMode.{cudagraph_mode.name} is not "  # 提示模式不受支持
                f"supported with {min_cg_backend_name} backend ("  # 指出后端名称
                f"support:{min_cg_support}) "  # 显示支持级别
                "; please try cudagraph_mode=PIECEWISE, "  # 建议使用PIECEWISE
                "and make sure compilation mode is VLLM_COMPILE"  # 并使用VLLM_COMPILE编译模式
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (  # 如果有专用的解码CUDAGraph且启用了推测解码
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL  # 解码模式为FULL
            and cudagraph_mode.separate_routine()  # 且使用独立的例程
            and self.uniform_decode_query_len > 1  # 且统一解码查询长度大于1（推测解码）
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(  # 调整CUDAGraph捕获大小
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size  # 传入解码查询长度和张量并行大小
            )

        # If the model has Mamba layers and cudagraph mode includes FULL
        # decode, cap cudagraph capture sizes to the number of available
        # Mamba cache blocks. Each decode request needs one conv_state
        # cache line, so capture batch sizes cannot exceed num_blocks.
        # Only FULL decode graphs are affected because PIECEWISE captures
        # run GDN/Mamba ops eagerly (prefill path, no causal_conv1d_update).
        # See: https://github.com/vllm-project/vllm/issues/34094
        if cudagraph_mode.has_full_cudagraphs():  # 如果当前模式包含FULL CUDAGraph
            has_mamba = any(  # 检查是否存在Mamba层
                isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_groups  # 遍历KV缓存组，检查是否有MambaSpec
            )
            if has_mamba and self.kv_cache_config is not None:  # 如果有Mamba层且KV缓存配置不为空
                self.compilation_config.adjust_cudagraph_sizes_for_mamba_cache(  # 调整CUDAGraph大小以适应Mamba缓存
                    self.kv_cache_config.num_blocks  # 传入可用的缓存块数量
                )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.compilation_config.cudagraph_mode = cudagraph_mode  # 将最终确定的CUDAGraph模式写回编译配置
        self.cudagraph_dispatcher.initialize_cudagraph_keys(  # 初始化CUDAGraph调度键
            cudagraph_mode, self.uniform_decode_query_len  # 传入CUDAGraph模式和统一解码查询长度
        )

        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (  # 如果启用了推测解码配置
            self.speculative_config.use_eagle()  # 且使用EAGLE方法
            or self.speculative_config.uses_extract_hidden_states()  # 或使用提取隐藏状态方法
        ):
            assert isinstance(self.drafter, EagleProposer | ExtractHiddenStatesProposer)  # 断言drafter是正确的类型
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)  # 初始化草稿模型的CUDAGraph调度键

    def calculate_reorder_batch_threshold(self) -> None:  # 计算重排批次阈值的方法
        """计算重排批次阈值，从所有注意力组中选择最小值。
        后端应该能支持低于其请求值的阈值，但可能因为将解码当作预填充处理而有性能损失。"""
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)  # 定义lambda：取两值最小值，None视为最大

        reorder_batch_thresholds: list[int | None] = [  # 收集所有注意力组的重排批次阈值列表
            group.get_metadata_builder().reorder_batch_threshold  # 从每个组的元数据构建器获取阈值
            for group in self._attn_group_iterator()  # 遍历所有注意力组
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:  # 如果没有注意力组（无注意力模型）
            self.reorder_batch_threshold = None  # 禁用重排功能
            return  # 直接返回
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]  # 使用reduce取所有阈值的最小值

    def may_reinitialize_input_batch(  # 可能需要重新初始化输入批次的方法
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]  # 接收KV缓存配置和内核块大小列表
    ) -> None:  # 无返回值
        """重新初始化输入批次（如果块大小与初始创建时不同）。
        当最终块大小（模型加载后确定）与__init__中使用的占位符不同，
        或存在多个KV缓存组时会发生这种情况。

        参数:
            kv_cache_config: KV缓存配置。
            kernel_block_sizes: 每个KV缓存组的内核块大小。
        """
        block_sizes = []  # 初始化块大小列表
        max_num_blocks = []  # 初始化最大块数列表
        max_model_len = max(self.max_model_len, self.max_encoder_len)  # 取模型最大长度和编码器最大长度的较大值
        for kv_cache_group in kv_cache_config.kv_cache_groups:  # 遍历所有KV缓存组
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):  # 如果是仅编码器注意力规格，跳过
                continue  # 跳过仅编码器的注意力层
            block_size = kv_cache_group.kv_cache_spec.block_size  # 获取当前组的块大小
            block_sizes.append(block_size)  # 添加到块大小列表
            max_num_blocks_per_req = cdiv(  # 计算每个请求的最大块数（向上取整除法）
                max_model_len, block_size * get_total_cp_world_size()  # 最大模型长度除以（块大小乘以上下文并行总世界大小）
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):  # 如果是Mamba规格
                max_num_blocks_per_req = (  # 重新计算Mamba层的最大块数
                    max_num_blocks_per_req  # 基础块数
                    if self.cache_config.enable_prefix_caching  # 如果启用前缀缓存则使用计算值
                    else 1  # 否则只需1个块
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks  # 加上推测块数量
            max_num_blocks.append(max_num_blocks_per_req)  # 添加到最大块数列表

        if (  # 如果块大小或内核块大小与初始值不同
            block_sizes != self._init_block_sizes  # 块大小发生变化
            or kernel_block_sizes != self._init_kernel_block_sizes  # 或内核块大小发生变化
        ):
            assert self.offload_config.uva.cpu_offload_gb == 0, (  # 断言CPU权重卸载未启用
                "Cannot re-initialize the input batch when CPU weight "  # 启用CPU卸载时不能重新初始化
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501  # 参考链接
                "for more details."  # 更多细节
            )
            self._init_block_sizes = block_sizes  # 更新初始块大小记录
            self._init_kernel_block_sizes = kernel_block_sizes  # 更新初始内核块大小记录
            self.input_batch = InputBatch(  # 重新创建InputBatch对象
                max_num_reqs=self.max_num_reqs,  # 最大请求数
                max_model_len=max_model_len,  # 最大模型长度
                max_num_batched_tokens=self.max_num_tokens,  # 最大批处理token数
                device=self.device,  # 计算设备
                pin_memory=self.pin_memory,  # 是否使用固定内存
                vocab_size=self.model_config.get_vocab_size(),  # 词汇表大小
                block_sizes=block_sizes,  # 块大小列表
                kernel_block_sizes=kernel_block_sizes,  # 内核块大小列表
                max_num_blocks_per_req=max_num_blocks,  # 每个请求的最大块数
                is_spec_decode=bool(self.vllm_config.speculative_config),  # 是否为推测解码
                logitsprocs=self.input_batch.logitsprocs,  # 沿用原有的logits处理器
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,  # 沿用原有的logits处理器标志
                is_pooling_model=self.is_pooling_model,  # 是否为池化模型
            )

        assert self._init_block_sizes == block_sizes, (  # 断言块大小一致
            f"InputBatch block_sizes {self._init_block_sizes} != "  # 提示不匹配信息
            f"kv_cache block_sizes {block_sizes}"  # 显示实际值
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (  # 断言内核块大小一致
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "  # 提示不匹配信息
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"  # 显示实际值
        )

    def _allocate_kv_cache_tensors(  # 分配KV缓存张量的方法
        self, kv_cache_config: KVCacheConfig  # 接收KV缓存配置参数
    ) -> dict[str, torch.Tensor]:  # 返回层名到张量的字典
        """分配KV缓存的内存缓冲区，使用正确的大小进行初始化。
        缓冲区在被模型使用之前需要重塑为所需形状。

        参数:
            kv_cache_config: KV缓存配置
        返回:
            dict[str, torch.Tensor]: 层名到其对应KV缓存内存缓冲区的映射。
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}  # 初始化原始张量字典
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:  # 遍历所有KV缓存张量配置
            tensor = torch.zeros(  # 创建零初始化的张量
                kv_cache_tensor.size, dtype=torch.int8, device=self.device  # 指定大小、数据类型和设备
            )
            for layer_name in kv_cache_tensor.shared_by:  # 遍历共享该张量的所有层
                kv_cache_raw_tensors[layer_name] = tensor  # 将张量映射到层名

        layer_names = set()  # 初始化层名集合
        for group in kv_cache_config.kv_cache_groups:  # 遍历所有KV缓存组
            for layer_name in group.layer_names:  # 遍历组中的所有层名
                if layer_name in self.runner_only_attn_layers:  # 如果是仅runner的注意力层则跳过
                    continue  # 跳过
                layer_names.add(layer_name)  # 添加到层名集合
        assert layer_names == set(kv_cache_raw_tensors.keys()), (  # 断言所有层都已正确初始化
            "Some layers are not correctly initialized"  # 某些层未正确初始化的错误消息
        )
        return kv_cache_raw_tensors  # 返回原始张量字典

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:  # 注意力组迭代器方法
        """返回所有注意力组的扁平化迭代器。"""
        return itertools.chain.from_iterable(self.attn_groups)  # 将嵌套的注意力组列表展平为单一迭代器

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:  # KV缓存规格注意力组迭代器
        """返回有KV缓存规格的注意力组迭代器。"""
        if not self.kv_cache_config.kv_cache_groups:  # 如果没有KV缓存组
            return  # 直接返回空迭代器
        for attn_groups in self.attn_groups:  # 遍历注意力组列表
            yield from attn_groups  # 逐个产出注意力组

    def _reshape_kv_cache_tensors(  # 重塑KV缓存张量的方法
        self,  # self引用
        kv_cache_config: KVCacheConfig,  # KV缓存配置
        kv_cache_raw_tensors: dict[str, torch.Tensor],  # 原始KV缓存张量字典
        kernel_block_sizes: list[int],  # 内核块大小列表
    ) -> dict[str, torch.Tensor]:  # 返回重塑后的张量字典
        """将KV缓存张量重塑为所需的形状和数据类型。

        参数:
            kv_cache_config: KV缓存配置
            kv_cache_raw_tensors: 每层的KV缓存缓冲区，大小正确但形状未初始化。
            kernel_block_sizes: 每个KV缓存组的内核块大小。
        返回:
            Dict[str, torch.Tensor]: 层名到其对应KV缓存内存缓冲区的映射。
        """
        kv_caches: dict[str, torch.Tensor] = {}  # 初始化重塑后的缓存字典
        has_attn, has_mamba = False, False  # 初始化标志：是否存在注意力层和Mamba层
        for group in self._kv_cache_spec_attn_group_iterator():  # 遍历所有有KV缓存规格的注意力组
            kv_cache_spec = group.kv_cache_spec  # 获取KV缓存规格
            attn_backend = group.backend  # 获取注意力后端
            if group.kv_cache_group_id == len(kernel_block_sizes):  # 如果组ID等于内核块大小列表长度
                # There may be a last group for layers without kv cache.
                continue  # 跳过没有KV缓存的最后一组
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]  # 获取对应组的内核块大小
            for layer_name in group.layer_names:  # 遍历该组中的所有层
                if layer_name in self.runner_only_attn_layers:  # 如果是仅runner的注意力层
                    continue  # 跳过
                raw_tensor = kv_cache_raw_tensors[layer_name]  # 获取该层的原始张量
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0  # 断言张量元素数能被页大小整除
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes  # 计算块数量
                if isinstance(kv_cache_spec, AttentionSpec):  # 如果是注意力规格
                    has_attn = True  # 标记存在注意力层
                    num_blocks_per_kv_block = (  # 计算每个KV块包含的内核块数
                        kv_cache_spec.block_size // kernel_block_size  # KV块大小除以内核块大小
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block  # 计算内核块总数

                    kv_cache_shape = attn_backend.get_kv_cache_shape(  # 从注意力后端获取KV缓存形状
                        kernel_num_blocks,  # 内核块总数
                        kernel_block_size,  # 内核块大小
                        kv_cache_spec.num_kv_heads,  # KV头数量
                        kv_cache_spec.head_size,  # 每个头的维度大小
                        cache_dtype_str=self.cache_config.cache_dtype,  # 缓存数据类型字符串
                    )
                    dtype = kv_cache_spec.dtype  # 获取数据类型
                    try:  # 尝试获取步幅顺序
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()  # 从后端获取KV缓存步幅顺序
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)  # 断言步幅顺序长度与形状维度数一致
                    except (AttributeError, NotImplementedError):  # 如果后端不支持自定义步幅顺序
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))  # 使用默认顺序（0,1,2,...）
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(  # 按步幅顺序重排形状
                        kv_cache_shape[i] for i in kv_cache_stride_order  # 根据步幅顺序选取维度
                    )
                    # Maintain original KV shape view.
                    inv_order = [  # 计算逆置换顺序
                        kv_cache_stride_order.index(i)  # 找到每个维度在原始顺序中的位置
                        for i in range(len(kv_cache_stride_order))  # 遍历所有维度索引
                    ]
                    kv_caches[layer_name] = (  # 将重塑后的张量存入字典
                        kv_cache_raw_tensors[layer_name]  # 获取原始张量
                        .view(dtype)  # 转换为目标数据类型视图
                        .view(kv_cache_shape)  # 重塑为目标形状
                        .permute(*inv_order)  # 按逆置换恢复原始语义顺序
                    )
                elif isinstance(kv_cache_spec, MambaSpec):  # 如果是Mamba规格
                    has_mamba = True  # 标记存在Mamba层
                    raw_tensor = kv_cache_raw_tensors[layer_name]  # 获取该层的原始张量
                    state_tensors = []  # 初始化状态张量列表
                    storage_offset_bytes = 0  # 初始化存储偏移量（字节）
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):  # 遍历Mamba的每个状态张量的形状和类型
                        dtype_size = get_dtype_size(dtype)  # 获取数据类型的字节大小
                        num_element_per_page = (  # 计算每页的元素数量
                            kv_cache_spec.page_size_bytes // dtype_size  # 页大小除以元素字节大小
                        )
                        target_shape = (num_blocks, *shape)  # 目标形状：(块数, *状态形状)
                        stride = torch.empty(target_shape).stride()  # 获取目标形状的默认步幅
                        target_stride = (num_element_per_page, *stride[1:])  # 自定义步幅：第一维用每页元素数
                        assert storage_offset_bytes % dtype_size == 0  # 断言偏移量能被数据类型大小整除
                        tensor = torch.as_strided(  # 创建带自定义步幅的张量视图
                            raw_tensor.view(dtype),  # 将原始张量视为目标数据类型
                            size=target_shape,  # 指定目标大小
                            stride=target_stride,  # 指定自定义步幅
                            storage_offset=storage_offset_bytes // dtype_size,  # 指定存储偏移（元素为单位）
                        )
                        state_tensors.append(tensor)  # 添加到状态张量列表
                        storage_offset_bytes += stride[0] * dtype_size  # 更新存储偏移量

                    kv_caches[layer_name] = state_tensors  # 将Mamba状态张量列表存入字典
                else:  # 其他未知类型
                    raise NotImplementedError  # 抛出未实现异常

        if has_attn and has_mamba:  # 如果同时存在注意力层和Mamba层（混合架构）
            self._update_hybrid_attention_mamba_layout(kv_caches)  # 更新混合注意力-Mamba的布局

        return kv_caches  # 返回重塑后的KV缓存字典

    def _update_hybrid_attention_mamba_layout(  # 更新混合注意力-Mamba布局的方法
        self, kv_caches: dict[str, torch.Tensor]  # 接收KV缓存字典
    ) -> None:  # 无返回值
        """更新注意力层的布局，从(2, num_blocks, ...)变为(num_blocks, 2, ...)。

        参数:
            kv_caches: 每层的KV缓存缓冲区。
        """

        for group in self._kv_cache_spec_attn_group_iterator():  # 遍历所有有KV缓存规格的注意力组
            kv_cache_spec = group.kv_cache_spec  # 获取KV缓存规格
            for layer_name in group.layer_names:  # 遍历组中的所有层名
                kv_cache = kv_caches[layer_name]  # 获取该层的KV缓存张量
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:  # 如果是注意力规格且第一维为2（K和V）
                    assert kv_cache.shape[1] != 2, (  # 断言第二维不为2，否则无法区分布局
                        "Fail to determine whether the layout is "  # 无法确定布局的错误消息
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "  # 两种可能的布局
                        f"a tensor of shape {kv_cache.shape}"  # 显示实际形状
                    )
                    hidden_size = kv_cache.shape[2:].numel()  # 计算隐藏维度的元素数量
                    kv_cache.as_strided_(  # 原地修改张量的步幅（不改变数据）
                        size=kv_cache.shape,  # 保持原有大小
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),  # 重新定义步幅以交错存储K和V
                    )

    # 分配和初始化 KV 缓存张量，支持统一布局（跨层连续存储）和非统一布局。
    # 将张量绑定到各注意力层，处理 Mamba 状态初始化和 FP8 缩放因子。
    def initialize_kv_cache_tensors(  # 初始化KV缓存张量的方法
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]  # 接收KV缓存配置和内核块大小列表
    ) -> dict[str, torch.Tensor]:  # 返回层名到张量的字典
        """初始化KV缓存的内存缓冲区。

        参数:
            kv_cache_config: KV缓存配置
            kernel_block_sizes: 每个KV缓存组的内核块大小。

        返回:
            Dict[str, torch.Tensor]: 层名到其对应KV缓存内存缓冲区的映射。
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype  # 获取缓存数据类型
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):  # 如果可以使用统一KV缓存布局
            kv_caches, cross_layers_kv_cache, attn_backend = (  # 分配统一KV缓存
                self.allocate_uniform_kv_caches(  # 调用统一KV缓存分配方法
                    kv_cache_config,  # KV缓存配置
                    self.attn_groups,  # 注意力组
                    cache_dtype,  # 缓存数据类型
                    self.device,  # 计算设备
                    kernel_block_sizes,  # 内核块大小列表
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache  # 保存跨层KV缓存引用
            self.cross_layers_attn_backend = attn_backend  # 保存跨层注意力后端引用
        else:  # 否则回退到通用情况
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)  # 分配原始KV缓存张量

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(  # 将内存缓冲区重塑为所需形状
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes  # 传入配置、原始张量和内核块大小
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():  # 遍历共享KV缓存的层映射
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)  # 记录调试日志：哪个层复用了哪个层的缓存
            kv_caches[layer_name] = kv_caches[target_layer_name]  # 将源层的缓存引用赋给目标层

        num_attn_module = (  # 确定注意力模块数量
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1  # longcat_flash模型有2个注意力模块，其他为1
        )
        bind_kv_cache(  # 将KV缓存绑定到注意力层
            kv_caches,  # KV缓存字典
            self.compilation_config.static_forward_context,  # 静态前向上下文
            self.kv_caches,  # 模型runner的KV缓存引用
            num_attn_module,  # 注意力模块数量
        )
        return kv_caches  # 返回KV缓存字典

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(  # 可能将KV共享层添加到KV缓存组的方法
        self, kv_cache_config: KVCacheConfig  # 接收KV缓存配置
    ) -> None:  # 无返回值
        """将复用KV缓存的层添加到其目标层的KV缓存组中。
        KV缓存张量的映射在`initialize_kv_cache_tensors()`中完成。
        """
        if not self.shared_kv_cache_layers:  # 如果没有跨层KV共享
            # No cross-layer KV sharing, return
            return  # 直接返回

        add_kv_sharing_layers_to_kv_cache_groups(  # 调用辅助函数添加KV共享层
            self.shared_kv_cache_layers,  # 共享KV缓存的层映射
            kv_cache_config.kv_cache_groups,  # KV缓存组列表
            self.runner_only_attn_layers,  # 仅runner的注意力层集合
        )

        if self.cache_config.kv_sharing_fast_prefill:  # 如果启用了KV共享快速预填充
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)  # 获取所有注意力层
            for layer_name in reversed(attn_layers):  # 从最后一层向前遍历
                if layer_name in self.shared_kv_cache_layers:  # 如果该层使用共享KV缓存
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)  # 标记为可快速预填充提前退出的层
                else:  # 如果遇到不共享KV缓存的层
                    break  # 停止遍历（只标记末尾连续共享的层）

    # KV 缓存初始化的主入口：设置注意力后端、分配 KV 缓存张量、
    # 绑定缓存到注意力层、初始化 Mamba 状态（混合架构），
    # 以及重建 InputBatch（如果 block size 发生变化）。
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:  # KV缓存初始化的主入口方法
        """基于kv_cache_config初始化KV缓存。
        参数:
            kv_cache_config: KV缓存的配置，包括每层的KV缓存大小
        """
        kv_cache_config = deepcopy(kv_cache_config)  # 深拷贝配置以避免修改原始对象
        self.kv_cache_config = kv_cache_config  # 保存KV缓存配置
        self._mamba_copy_bufs = None  # 重置Mamba拷贝缓冲区
        self.may_add_encoder_only_layers_to_kv_cache_config()  # 可能添加仅编码器层到KV缓存配置
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)  # 可能添加KV共享层到缓存组
        self.initialize_attn_backend(kv_cache_config)  # 初始化注意力后端
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = prepare_kernel_block_sizes(  # 准备内核块大小
            kv_cache_config, self.attn_groups  # 基于KV缓存配置和注意力组
        )
        self._kernel_block_sizes = kernel_block_sizes  # 保存内核块大小

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)  # 创建元数据构建器

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)  # 如果需要则重新初始化输入批次
        kv_caches = self.initialize_kv_cache_tensors(  # 初始化KV缓存张量
            kv_cache_config, kernel_block_sizes  # 传入配置和内核块大小
        )

        if (  # 如果启用了推测解码
            self.speculative_config  # 存在推测解码配置
            and self.speculative_config.uses_extract_hidden_states()  # 且使用提取隐藏状态方法
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)  # 断言drafter类型正确
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)  # 验证所有草稿模型层属于同一个KV缓存组

        if has_kv_transfer_group():  # 如果存在KV传输组（用于分离式推理）
            kv_transfer_group = get_kv_transfer_group()  # 获取KV传输组
            if self.cross_layers_kv_cache is not None:  # 如果有跨层KV缓存
                assert self.cross_layers_attn_backend is not None  # 断言跨层注意力后端不为空
                kv_transfer_group.register_cross_layers_kv_cache(  # 注册跨层KV缓存到传输组
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend  # 传入缓存和后端
                )
            else:  # 否则
                kv_transfer_group.register_kv_caches(kv_caches)  # 注册普通KV缓存到传输组
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)  # 设置主机传输缓冲区操作

    def _get_attention_kv_cache_gid(self) -> int:  # 获取注意力层KV缓存组索引的方法
        """查找注意力层的KV缓存组索引。"""
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):  # 遍历所有KV缓存组
            if isinstance(group.kv_cache_spec, AttentionSpec):  # 如果是注意力规格
                return gid  # 返回该组的索引
        return 0  # 如果未找到，返回默认值0

    def init_routed_experts_capturer(self):  # 初始化路由专家捕获器的方法
        """初始化路由专家捕获器，用于记录MoE层的路由决策。"""
        logger.info(  # 记录信息日志
            "Initializing routed experts capturer, enable_return_routed_experts: %s",  # 日志格式
            self.model_config.enable_return_routed_experts,  # 是否启用返回路由专家信息
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()  # 创建路由专家捕获器实例
        self.routed_experts_attn_gid = self._get_attention_kv_cache_gid()  # 获取注意力KV缓存组ID
        min_block_size = min(  # 获取所有KV缓存组中的最小块大小
            [
                group.kv_cache_spec.block_size  # 每个组的块大小
                for group in self.kv_cache_config.kv_cache_groups  # 遍历所有KV缓存组
            ]
        )
        num_groups = len(self.kv_cache_config.kv_cache_groups)  # 获取KV缓存组的数量
        self.max_num_kv_tokens = (  # 计算最大KV token数量
            self.kv_cache_config.num_blocks // num_groups  # 总块数除以组数
        ) * min_block_size  # 乘以最小块大小
        dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size  # 获取解码上下文并行大小
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size  # 获取预填充上下文并行大小
        if pcp_size * dcp_size > 1:  # 如果使用了上下文并行
            self.max_num_kv_tokens *= pcp_size * dcp_size  # 按上下文并行倍数放大KV token数量

        routed_experts_capturer.init_buffer(  # 初始化捕获器的缓冲区
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,  # 最大批处理token数
            max_num_kv_tokens=self.max_num_kv_tokens,  # 最大KV token数
            vllm_config=self.vllm_config,  # vllm配置
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)  # 将捕获器绑定到模型的MoE层
        self.routed_experts_initialized = True  # 标记路由专家捕获器已初始化

    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:  # 绑定路由专家捕获器到MoE层的方法
        """将路由专家捕获器的回调函数绑定到所有FusedMoE层的路由器上。"""
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE  # 导入FusedMoE层类，用于混合专家模型
        from vllm.model_executor.layers.fused_moe.router.base_router import (  # 导入基础路由器类
            BaseRouter,  # MoE路由器的基类
        )

        for module in self.compilation_config.static_forward_context.values():  # 遍历静态前向上下文中的所有模块
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):  # 如果是FusedMoE模块且路由器是BaseRouter类型
                layer_id = module.layer_id  # 获取层ID

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):  # 定义捕获回调函数（使用默认参数捕获闭包变量）
                    _capturer.capture(_layer_id, topk_ids)  # 调用捕获器记录路由决策

                module.router.set_capture_fn(_capture_fn)  # 将捕获函数设置到路由器上

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:  # 可能添加仅编码器层到KV缓存配置的方法
        """将仅编码器的注意力层添加到KV缓存配置中。"""
        block_size = self.vllm_config.cache_config.block_size  # 获取缓存块大小
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)  # 初始化编码器注意力规格到层名列表的字典
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)  # 获取所有注意力层
        for layer_name, attn_module in attn_layers.items():  # 遍历所有注意力层
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:  # 如果是仅编码器类型的注意力
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(  # 创建仅编码器注意力规格
                    block_size=block_size,  # 块大小
                    num_kv_heads=attn_module.num_kv_heads,  # KV头数量
                    head_size=attn_module.head_size,  # 头维度大小
                    dtype=self.kv_cache_dtype,  # KV缓存数据类型
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)  # 将层名添加到对应规格的列表中
                self.runner_only_attn_layers.add(layer_name)  # 标记为仅runner的注意力层
        if len(encoder_only_attn_specs) > 0:  # 如果存在仅编码器的注意力层
            assert len(encoder_only_attn_specs) == 1, (  # 断言目前只支持一种编码器注意力规格
                "Only support one encoder-only attention spec now"  # 错误提示
            )
            spec, layer_names = encoder_only_attn_specs.popitem()  # 取出唯一的规格和层名列表
            self.kv_cache_config.kv_cache_groups.append(  # 将新的KV缓存组添加到配置中
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)  # 创建KV缓存组规格
            )

    # 获取每个注意力层的 KV 缓存规格，用于 KV 缓存管理器的初始化。
    # 遍历模型的所有注意力层，根据注意力类型（全注意力、滑动窗口、
    # 分块局部注意力、交叉注意力、Mamba 等）生成对应的 KVCacheSpec。
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:  # 获取KV缓存规格的方法
        """通过解析静态前向上下文中每个注意力模块的KV缓存格式来生成KVCacheSpec。
        返回:
            KVCacheSpec: 层名到其KV缓存格式的字典映射。不需要KV缓存的层不包含在内。
        """
        if has_ec_transfer() and not get_ec_transfer().is_consumer:  # 如果存在EC传输且不是消费者
            return {}  # 返回空字典（生产者不需要KV缓存规格）
        kv_cache_spec: dict[str, KVCacheSpec] = {}  # 初始化KV缓存规格字典
        layer_type = cast(type[Any], AttentionLayerBase)  # 将注意力层基类转换为Any类型
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)  # 获取所有注意力层
        for layer_name, attn_module in attn_layers.items():  # 遍历所有注意力层
            if isinstance(attn_module, Attention) and (  # 如果是Attention实例
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name  # 且设置了KV共享目标层名
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer  # 记录共享KV缓存的层映射关系
                continue  # 跳过该层，不创建独立的KV缓存规格
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):  # 获取该模块的KV缓存规格（如果有的话）
                kv_cache_spec[layer_name] = spec  # 将规格存入字典

        return kv_cache_spec  # 返回KV缓存规格字典

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:  # 将采样token ID张量转换为Python列表的方法
        """将GPU上的采样token ID张量异步拷贝到CPU固定内存，然后转为列表。
        使用CUDA事件同步代替流同步，避免阻塞其他CUDA流的拷贝操作。"""
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]  # 获取对应大小的固定内存缓冲区切片
        pinned.copy_(sampled_token_ids, non_blocking=True)  # 异步将GPU张量拷贝到CPU固定内存
        self.transfer_event.record()  # 记录CUDA事件
        self.transfer_event.synchronize()  # 同步等待CUDA事件完成（比流同步更精确）
        return pinned.tolist()  # 将固定内存中的张量转换为Python列表并返回

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:  # 获取编码器计时统计信息的方法
        """获取所有请求的编码器计时统计信息并清除注册表。

        返回:
            字典，将request_id映射到统计信息字典。
        """
        with self._encoder_timing_lock:  # 获取编码器计时锁（线程安全）
            stats = {  # 构建统计信息字典
                req_id: stats_obj.to_dict()  # 将每个请求的统计对象转换为字典
                for req_id, stats_obj in self.encoder_timing_registry.items()  # 遍历计时注册表
            }
            self.encoder_timing_registry.clear()  # 清除注册表
            return stats  # 返回统计信息

    @contextmanager  # 上下文管理器装饰器
    def timed_encoder_operation(  # 编码器操作计时的上下文管理器方法
        self,  # self引用
        should_time: bool,  # 是否启用计时
        group_lora_refs: list[tuple[str, Any]],  # (request_id, 位置信息)元组的完整列表
        current_item_idx: int,  # 当前组的起始索引
        num_items: int,  # 当前组中的项目数量
    ):
        """用于计时编码器前向操作的上下文管理器。

        参数:
            should_time: 是否启用计时
            group_lora_refs: (request_id, pos_info)元组的完整列表
            current_item_idx: 当前组的起始索引
            num_items: 当前组中的项目数
        """
        if not should_time:  # 如果不需要计时
            yield  # 直接执行包裹的代码块
            return  # 返回

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]  # 切片获取当前组的引用列表
        group_request_ids = {req_id for req_id, _ in group_refs}  # 提取当前组中的所有请求ID集合

        torch.accelerator.synchronize()  # 同步加速器，确保之前的操作完成
        start_time = time.perf_counter()  # 记录开始时间（高精度计时器）

        try:  # 尝试执行
            yield  # 让出控制权，执行with块中的编码器前向操作
        finally:  # 无论是否异常都执行
            torch.accelerator.synchronize()  # 同步加速器，确保编码器操作完成
            elapsed = time.perf_counter() - start_time  # 计算经过的时间

            per_request_time = elapsed / max(len(group_request_ids), 1)  # 计算每个请求的平均耗时

            with self._encoder_timing_lock:  # 获取编码器计时锁（线程安全）
                for req_id in group_request_ids:  # 遍历当前组的所有请求ID
                    if req_id not in self.encoder_timing_registry:  # 如果请求ID不在注册表中
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()  # 创建新的计时统计对象

                    stats = self.encoder_timing_registry[req_id]  # 获取该请求的计时统计对象
                    stats.encoder_forward_secs += per_request_time  # 累加编码器前向耗时
                    stats.num_encoder_calls += 1  # 累加编码器调用次数
