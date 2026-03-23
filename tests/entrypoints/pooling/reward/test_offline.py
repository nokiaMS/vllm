# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [离线奖励模型测试模块：验证 vLLM 离线 reward API 的配置和激活函数参数行为]

import weakref

import pytest
import torch

from tests.models.utils import softmax
from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "internlm/internlm2-1_8b-reward"

prompts = ["The chef prepared a delicious meal."]


# [测试夹具：创建 internlm2-1_8b-reward 奖励模型的 LLM 实例]
@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        trust_remote_code=True,
        seed=0,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


# [测试配置验证：确认奖励模型启用了前缀缓存和分块预填充]
@pytest.mark.skip_global_cleanup
def test_config(llm: LLM):
    vllm_config = llm.llm_engine.vllm_config
    assert vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.scheduler_config.enable_chunked_prefill


# [测试池化激活函数参数：验证默认启用激活函数、启用与禁用的一致性]
def test_pooling_params(llm: LLM):
    def get_outputs(use_activation):
        outputs = llm.reward(
            prompts,
            pooling_params=PoolingParams(use_activation=use_activation),
            use_tqdm=False,
        )
        return torch.cat([x.outputs.data for x in outputs])

    default = get_outputs(use_activation=None)
    w_activation = get_outputs(use_activation=True)
    wo_activation = get_outputs(use_activation=False)

    assert torch.allclose(default, w_activation, atol=1e-2), (
        "Default should use activation."
    )
    assert not torch.allclose(w_activation, wo_activation, atol=1e-2), (
        "wo_activation should not use activation."
    )
    assert torch.allclose(softmax(wo_activation), w_activation, atol=1e-2), (
        "w_activation should be close to activation(wo_activation)."
    )
