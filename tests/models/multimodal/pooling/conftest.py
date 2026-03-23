# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 池化测试的 pytest 配置，提供 ROCm 平台的 SigLIP 注意力后端配置
"""Pytest configuration for vLLM pooling tests."""

import pytest

from vllm.platforms import current_platform


@pytest.fixture
def siglip_attention_config():
    """Return attention config for SigLIP tests on ROCm.

    On ROCm, SigLIP tests require FLEX_ATTENTION backend.
    """
    if current_platform.is_rocm():
        return {"backend": "FLEX_ATTENTION"}
    return None
