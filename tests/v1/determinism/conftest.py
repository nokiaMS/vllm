# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 确定性测试的公共配置：自动启用批次不变性内核覆盖
import pytest

import vllm.model_executor.layers.batch_invariant as batch_invariant


# 自动为所有确定性测试启用批次不变性模式
@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Automatically enable batch invariant kernel overrides for all tests."""
    monkeypatch.setattr(batch_invariant, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
