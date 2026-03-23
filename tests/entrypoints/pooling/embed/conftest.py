# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM pooling embed tests."""

# [嵌入测试 pytest 配置：在 ROCm 平台上禁用 Flash/MemEfficient SDP 以避免精度问题]

import warnings

import torch

from vllm.platforms import current_platform


# [测试收集钩子：根据平台配置 ROCm 特定的 SDP 后端设置]
def pytest_collection_modifyitems(config, items):
    """Configure ROCm-specific settings based on collected tests."""
    if not current_platform.is_rocm():
        return

    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    warnings.warn(
        "ROCm: Disabled flash_sdp and mem_efficient_sdp, enabled math_sdp "
        "to avoid HuggingFace Transformers accuracy issues",
        UserWarning,
        stacklevel=1,
    )
