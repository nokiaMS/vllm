# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试 VLLM_PORT 环境变量解析：验证未设置、合法整数、非法值和 URI 格式的处理]

import os
from unittest.mock import patch

import pytest

from vllm.envs import get_vllm_port


# [测试当 VLLM_PORT 未设置时返回 None]
def test_get_vllm_port_not_set():
    """Test when VLLM_PORT is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_vllm_port() is None


# [测试当 VLLM_PORT 设为合法整数时返回正确的端口号]
def test_get_vllm_port_valid():
    """Test when VLLM_PORT is set to a valid integer."""
    with patch.dict(os.environ, {"VLLM_PORT": "5678"}, clear=True):
        assert get_vllm_port() == 5678


# [测试当 VLLM_PORT 设为非整数值时抛出 ValueError]
def test_get_vllm_port_invalid():
    """Test when VLLM_PORT is set to a non-integer value."""
    with (
        patch.dict(os.environ, {"VLLM_PORT": "abc"}, clear=True),
        pytest.raises(ValueError, match="must be a valid integer"),
    ):
        get_vllm_port()


# [测试当 VLLM_PORT 设为 URI 格式时抛出 ValueError 并提示格式错误]
def test_get_vllm_port_uri():
    """Test when VLLM_PORT is set to a URI."""
    with (
        patch.dict(os.environ, {"VLLM_PORT": "tcp://localhost:5678"}, clear=True),
        pytest.raises(ValueError, match="appears to be a URI"),
    ):
        get_vllm_port()
