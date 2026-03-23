# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define LoRA functionality mixin for model runners.
"""

from contextlib import contextmanager
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch: TypeAlias = TPUInputBatch | GPUInputBatch

logger = init_logger(__name__)


# LoRA 功能混入类，为 GPUModelRunner 和 TPUModelRunner 提供 LoRA 适配器管理能力。
# 核心设计：通过 LRUCacheWorkerLoRAManager 管理 LoRA 适配器的加载/卸载/LRU 缓存，
# 在每步推理前根据批次中的请求动态激活对应的 LoRA 权重。
# 支持 CUDA Graph 预热时的 dummy LoRA 设置，以及视觉编码器/连接器的独立 LoRA 映射。
# Defined as a mixin for GPUModelRunner
class LoRAModelRunnerMixin:
    # 加载 LoRA 模型：创建 LoRA 管理器并将 LoRA 层注入到基础模型中。
    def load_lora_model(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> nn.Module:
        if not supports_lora(model):
            raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")

        # Add LoRA Manager to the Model Runner
        self.lora_manager = LRUCacheWorkerLoRAManager(
            vllm_config,
            device,
            model.embedding_modules,
        )
        return self.lora_manager.create_lora_manager(model, vllm_config)

    # 激活指定的 LoRA 适配器：构建 token→LoRA ID 映射并交给 LoRA 管理器设置。
    def _set_active_loras(
        self,
        prompt_lora_mapping: tuple[int, ...],
        token_lora_mapping: tuple[int, ...],
        lora_requests: set[LoRARequest],
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        self._ensure_lora_enabled()

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
        lora_mapping = LoRAMapping(
            token_lora_mapping,
            prompt_lora_mapping,
            is_prefill=True,
            type=mapping_type,
        )
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def _ensure_lora_enabled(self) -> None:
        if not hasattr(self, "lora_manager"):
            raise RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")

    # 根据输入批次中的 LoRA 映射关系激活 LoRA 适配器（推理时调用）。
    def set_active_loras(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        prompt_lora_mapping: tuple[int, ...]  # of size np.sum(num_sampled_tokens)
        token_lora_mapping: tuple[int, ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        prompt_lora_mapping, token_lora_mapping, lora_requests = (
            input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
        )
        return self._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
        )

    # 用于 CUDA Graph 捕获和预热的上下文管理器：
    # 创建虚拟 LoRA 适配器并添加到缓存中，退出后可选择移除。
    @contextmanager
    def maybe_setup_dummy_loras(
        self, lora_config: LoRAConfig | None, remove_lora: bool = True
    ):
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_loras = lora_config.max_loras
            lora_warmup_rank = (
                lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            )
            # Make dummy lora requests
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, num_loras + 1)
            }

            with self.lora_manager.dummy_lora_cache():
                # Add the dummy LoRAs here so _set_active_loras doesn't try to
                # load from disk.
                for lr in lora_requests:
                    self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)

                yield

            # __exit__ code
            if remove_lora:
                self.lora_manager.remove_all_adapters()

    # 为 CUDA Graph 捕获/预热选择虚拟 LoRA：按请求数循环分配 LoRA ID，
    # 模拟最坏情况下的 LoRA 批次配置以确保 graph 覆盖所有场景。
    @contextmanager
    def maybe_select_dummy_loras(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
        num_sampled_tokens: np.ndarray | None = None,
        num_active_loras: int = 0,
    ):
        """
        Context manager to select dummy LoRAs for capture/warmup.

        Args:
            lora_config: LoRA configuration, or None if LoRA is disabled.
            num_scheduled_tokens: Array of scheduled token counts per request.
            num_sampled_tokens: Array of sampled token counts per request.
            num_active_loras: Number of distinct active LoRAs to use.
                - 0: No LoRA active (set up zero mappings).
                - >0: Use exactly this many distinct LoRAs.
        """
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        # Skip LoRA setup entirely only if no LoRA config
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_reqs = len(num_scheduled_tokens)
            max_loras = lora_config.max_loras

            # Determine how many distinct LoRAs to use and whether to include
            # no-LoRA tokens (-1 entries).
            # When num_active_loras > max_loras (e.g., max_loras + 1), we need
            # to include -1 entries to simulate batches with both LoRA and
            # no-LoRA tokens. This ensures prepare_tensors computes the correct
            # num_active_loras that matches the cudagraph capture key.
            if num_active_loras == 0:
                # No LoRA active - use 0 mappings like the original code
                effective_num_loras = 0
                include_no_lora = False
            elif num_active_loras > max_loras:
                # num_active_loras > max_loras means we want max_loras adapters
                # PLUS no-LoRA tokens (-1). This is the max_loras + 1 case.
                effective_num_loras = max_loras
                include_no_lora = True
            else:
                # Specific number of active LoRAs requested
                effective_num_loras = min(num_active_loras, max_loras)
                include_no_lora = False

            # Make prompt lora mapping
            # Assign LoRA IDs cyclically to simulate a worst-case scenario.
            # LoRA IDs are 1-indexed (1 to max_loras) as required by LoRARequest.
            # convert_mapping() will convert these to 0-indexed slot indices.
            if effective_num_loras > 0:
                if include_no_lora:
                    # Include -1 (no-LoRA) entries by cycling through
                    # -1, 1, 2, ..., effective_num_loras
                    # This ensures prepare_tensors sees both LoRA and no-LoRA
                    # tokens, computing num_active_loras = effective_num_loras+1
                    cycle_values = np.array(
                        list(range(1, effective_num_loras + 1)),
                        dtype=np.int32,
                    )
                    prompt_lora_mapping = cycle_values[
                        np.arange(num_reqs, dtype=np.int32) % len(cycle_values)
                    ]
                else:
                    # Use 1 to effective_num_loras (1-indexed lora IDs)
                    prompt_lora_mapping = (
                        np.arange(num_reqs, dtype=np.int32) % effective_num_loras
                    ) + 1
            else:
                # No LoRA active - use 0 for all tokens (original behavior)
                prompt_lora_mapping = np.zeros(num_reqs, dtype=np.int32)

            # Make sample lora mapping
            sample_lora_mapping = np.repeat(prompt_lora_mapping, num_sampled_tokens)

            # Make token lora mapping
            token_lora_mapping = np.repeat(prompt_lora_mapping, num_scheduled_tokens)

            # Make dummy lora requests (only for the active LoRAs)
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, effective_num_loras + 1)
            }

            self._set_active_loras(
                tuple(sample_lora_mapping),
                tuple(token_lora_mapping),
                lora_requests,
                mapping_type,
            )

            yield

    # 组合上下文管理器：同时设置和选择虚拟 LoRA，用于 dummy run 和 graph 捕获。
    @contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        remove_lora: bool = True,
        num_active_loras: int = 0,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ):
        """
        Context manager for dummy runs with LoRA.

        Args:
            lora_config: LoRA configuration.
            num_scheduled_tokens: Array of scheduled token counts per request.
            num_sampled_tokens: Array of sampled token counts per request.
            remove_lora: Whether to remove LoRAs after the context exits.
            num_active_loras: Number of distinct active LoRAs to use.
                LoRA is activated when num_active_loras > 0.
        """
        with (
            self.maybe_setup_dummy_loras(lora_config, remove_lora),
            self.maybe_select_dummy_loras(
                lora_config,
                num_scheduled_tokens,
                mapping_type,
                num_sampled_tokens,
                num_active_loras,
            ),
        ):
            yield

    def maybe_remove_all_loras(self, lora_config: LoRAConfig | None):
        if lora_config is None:
            return
        self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        self._ensure_lora_enabled()
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        self._ensure_lora_enabled()
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        self._ensure_lora_enabled()
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        self._ensure_lora_enabled()
        return self.lora_manager.list_adapters()
