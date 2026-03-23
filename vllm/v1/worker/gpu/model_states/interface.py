# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


# 模型状态抽象基类，定义了模型在推理过程中需要实现的核心接口
# 设计思路：通过抽象接口统一不同模型架构（如标准LLM、Whisper等）的状态管理，
# 包括请求注册、多模态嵌入获取、输入准备和注意力元数据构建
class ModelState(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        raise NotImplementedError

    # 返回模型支持的生成任务类型，默认为文本生成
    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        return ("generate",)

    # 添加新请求时的钩子，子类可重写以执行额外初始化（如预计算位置编码）
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        return None

    # 将暂存的写操作应用到GPU，子类可重写以刷新缓冲数据
    def apply_staged_writes(self) -> None:
        return None

    # 抽象方法：获取多模态嵌入，由子类实现具体的编码和嵌入合并逻辑
    @abstractmethod
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    # 抽象方法：准备模型特有的额外输入（如位置编码、编码器输出等）
    @abstractmethod
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        raise NotImplementedError

    # 抽象方法：准备CUDA Graph捕获所需的虚拟输入
    @abstractmethod
    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    # 抽象方法：构建注意力计算所需的元数据（如KV缓存块表、序列长度等）
    @abstractmethod
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError
