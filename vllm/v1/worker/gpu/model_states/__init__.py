# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


# 模型状态工厂函数：根据模型架构选择合适的ModelState实现
# 设计思路：采用工厂模式，针对不同模型架构（如Whisper的编码器-解码器结构）
# 返回对应的状态管理实例，默认使用通用的DefaultModelState
def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    if "WhisperForConditionalGeneration" in vllm_config.model_config.architectures:
        from vllm.v1.worker.gpu.model_states.whisper import WhisperModelState

        return WhisperModelState(vllm_config, model, encoder_cache, device)

    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(vllm_config, model, encoder_cache, device)
