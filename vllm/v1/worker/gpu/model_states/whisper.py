# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


# Whisper模型状态管理类，专为编码器-解码器架构的语音识别模型设计
# 与DefaultModelState的关键区别：编码器输出通过交叉注意力（cross-attention）
# 传递给解码器，而非合并到inputs_embeds中；首次prefill时写入KV缓存，
# 后续decode步骤直接从交叉注意力KV缓存读取
class WhisperModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.model_config.max_model_len
        self.device = device

        assert encoder_cache is not None
        self.encoder_cache = encoder_cache
        self.encoder_runner = EncoderRunner(
            model=self.model,
            max_num_tokens=self.max_num_tokens,
            hidden_size=self.model_config.get_inputs_embeds_size(),
            encoder_cache=self.encoder_cache,
            dtype=self.model_config.dtype,
            device=self.device,
        )

        self.max_encoder_len = getattr(
            self.model_config.hf_config,
            "max_source_positions",
            self.max_model_len,
        )
        self.encoder_seq_lens_gpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

        self.encoder_outputs: list[torch.Tensor] = []

    # Whisper模型仅支持转录（transcription）任务
    def get_supported_generation_tasks(self):
        return ("transcription",)

    # 获取音频编码器输出：与默认实现不同，不生成inputs_embeds
    # 编码器输出直接通过encoder_outputs传递给解码器的交叉注意力层
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        # Ensure encoder inputs are ordered consistently with input_batch.req_ids.
        encoder_inputs: dict[str, list[int]] = {}
        for req_id in input_batch.req_ids:
            req_encoder_inputs = scheduled_encoder_inputs.get(req_id, [])
            if req_encoder_inputs:
                encoder_inputs[req_id] = req_encoder_inputs
        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        if mm_kwargs:
            # Whisper consumes encoder outputs through `encoder_outputs`, not
            # `inputs_embeds`. Single modality (audio) so execute_mm_encoder
            # preserves request order; use its return value directly.
            # No need to store in encoder_cache: cross-attention K/V are written
            # to the KV cache on the first step; decode steps use the cache.
            self.encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        else:
            # Decode steps: encoder K/V are in cross-attention KV cache.
            self.encoder_outputs = []
        return None

    # 准备模型输入：将编码器输出传递给解码器，传递后清空以避免重复使用
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = {"encoder_outputs": self.encoder_outputs}
        self.encoder_outputs = []
        return model_inputs

    # 准备CUDA Graph捕获的虚拟输入，传空的编码器输出列表
    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"encoder_outputs": []}

    # 构建注意力元数据：除标准自注意力外，还需处理交叉注意力的编码器序列长度
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
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        encoder_seq_lens = self._get_encoder_seq_lens(
            input_batch.req_ids, attn_groups, for_capture
        )

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            encoder_seq_lens=encoder_seq_lens,
        )
        return attn_metadata

    # 获取编码器序列长度：用于交叉注意力的KV长度计算
    # 正常推理时使用实际编码长度，CUDA Graph捕获时使用最大编码长度
    # 返回按注意力组索引的编码器长度映射
    def _get_encoder_seq_lens(
        self,
        req_ids: list[str],
        attn_groups: list[list[AttentionGroup]],
        for_capture: bool,
    ) -> dict[int, tuple[torch.Tensor, np.ndarray]]:
        num_reqs = len(req_ids)
        encoder_seq_lens_np = np.zeros(num_reqs, dtype=np.int32)
        if not for_capture:
            # During normal execution, use actual encoder lengths.
            for i, req_id in enumerate(req_ids):
                mm_features = self.encoder_cache.mm_features.get(req_id, [])
                encoder_seq_lens_np[i] = sum(
                    feature.mm_position.get_num_embeds() for feature in mm_features
                )
        else:
            # During CUDA graph capture, use max encoder length so max_seqlen_k
            # is captured with the correct value for cross-attention.
            encoder_seq_lens_np[:] = self.max_encoder_len

        self.encoder_seq_lens_gpu[:num_reqs].copy_(
            torch.from_numpy(encoder_seq_lens_np), non_blocking=True
        )
        self.encoder_seq_lens_gpu[num_reqs:].fill_(0)
        encoder_seq_lens_gpu = self.encoder_seq_lens_gpu[:num_reqs]

        seq_lens_by_group: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        for kv_cache_group_idx, groups in enumerate(attn_groups):
            has_cross_attn = any(
                isinstance(attn_group.kv_cache_spec, CrossAttentionSpec)
                for attn_group in groups
            )
            if has_cross_attn:
                seq_lens_by_group[kv_cache_group_idx] = (
                    encoder_seq_lens_gpu,
                    encoder_seq_lens_np,
                )
        return seq_lens_by_group
