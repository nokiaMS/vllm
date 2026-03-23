# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# 测试不支持的引擎配置是否正确抛出异常（如推测解码配置）
def test_unsupported_configs():
    with pytest.raises(ValueError):
        AsyncEngineArgs(
            model=MODEL,
            speculative_config={
                "model": MODEL,
            },
        ).create_engine_config()
