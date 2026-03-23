# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec


# 多模态编码器缓存管理类
# 设计思路：维护两级缓存——请求级别的多模态特征信息（mm_features）和
# 基于哈希的编码器输出缓存（encoder_outputs），支持跨请求复用相同的编码器输出
class EncoderCache:
    def __init__(self):
        # req_id -> MM features
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash -> encoder outputs
        self.encoder_outputs: dict[str, torch.Tensor] = {}

    # 注册新请求的多模态特征信息
    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        self.mm_features[req_id] = mm_features

    # 移除指定请求的多模态特征信息
    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # TODO: Implement MM budget for encoder dummy run
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    # 释放指定哈希值对应的单个编码器缓存条目
    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)
