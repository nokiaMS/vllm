# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [回归测试：包含用户报告的已修复问题的回归测试，确保问题不再复现]
"""Containing tests that check for regressions in vLLM's behavior.

It should include tests that are reported by users and making sure they
will never happen again.

"""

import gc

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


@pytest.mark.skip(reason="In V1, we reject tokens > max_seq_len")
# [回归测试 #1655：验证超长提示不会导致重复的被忽略序列组]
def test_duplicated_ignored_sequence_group():
    """https://github.com/vllm-project/vllm/issues/1655"""

    sampling_params = SamplingParams(temperature=0.01, top_p=0.1, max_tokens=256)
    llm = LLM(
        model="distilbert/distilgpt2",
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
    )
    prompts = ["This is a short prompt", "This is a very long prompt " * 1000]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


# [测试 max_tokens 设为 None 时模型能正常生成输出]
def test_max_tokens_none():
    sampling_params = SamplingParams(temperature=0.01, top_p=0.1, max_tokens=None)
    llm = LLM(
        model="distilbert/distilgpt2",
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
    )
    prompts = ["Just say hello!"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


# [测试 LLM 对象删除后 GPU 内存能被正确释放（残留应低于 50MB）]
def test_gc():
    llm = LLM(model="distilbert/distilgpt2", enforce_eager=True)
    del llm

    gc.collect()
    torch.accelerator.empty_cache()

    # The memory allocated for model and KV cache should be released.
    # The memory allocated for PyTorch and others should be less than 50MB.
    # Usually, it's around 10MB.
    allocated = torch.cuda.memory_allocated()
    assert allocated < 50 * 1024 * 1024


# [测试从 ModelScope 加载模型并成功生成输出]
def test_model_from_modelscope(monkeypatch: pytest.MonkeyPatch):
    # model: https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        # Don't use HF_TOKEN for ModelScope repos, otherwise it will fail
        # with 400 Client Error: Bad Request.
        m.setenv("HF_TOKEN", "")
        attn_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"
        llm = LLM(model="qwen/Qwen1.5-0.5B-Chat", attention_backend=attn_backend)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == 4
