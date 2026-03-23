# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import numpy as np
import pytest
import torch
from transformers import AutoModelForTokenClassification

from tests.models.utils import softmax
from vllm.platforms import current_platform


# 测试夹具：固定所有随机数种子以确保测试结果的可复现性
@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all random number generators for reproducibility."""
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


@pytest.mark.parametrize(
    "model",
    [
        "boltuix/NeuroBERT-NER",
        "gyr66/Ernie-3.0-base-chinese-finetuned-ner",
    ],
)
# The float32 is required for this tiny model to pass the test.
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
# 测试 BERT 类模型的命名实体识别（NER）token 分类任务，验证 vLLM 与 HuggingFace 的输出一致性
def test_bert_like_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=None, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    # Use eager attention on ROCm to avoid HF Transformers flash attention
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    hf_model_kwargs = {}
    if current_platform.is_rocm():
        hf_model_kwargs["attn_implementation"] = "eager"

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
        model_kwargs=hf_model_kwargs,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, vllm_output, atol=3.2e-2, rtol=1e-3)


@pytest.mark.parametrize("model", ["disham993/electrical-ner-ModernBERT-base"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.flaky(reruns=3)
@torch.inference_mode
# 测试 ModernBERT 模型的 token 分类任务，使用 flaky 重试机制应对随机初始化权重导致的数值精度波动
def test_modernbert_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    # NOTE: https://github.com/vllm-project/vllm/pull/32403
    # `disham993/electrical-ner-ModernBERT-base` is a randomly initialized
    # model, which can cause numerical precision variance and edge cases.
    # We use @flaky(reruns=3) to mitigate intermittent failures.
    print(
        f"\n[NOTE] Testing {model} (randomly initialized weights) - "
        "flaky tolerance enabled due to numerical precision variance."
    )

    with vllm_runner(model, max_model_len=None, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    # Use eager attention on ROCm to avoid HF Transformers flash attention
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    hf_model_kwargs = {}
    if current_platform.is_rocm():
        hf_model_kwargs["attn_implementation"] = "eager"

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
        model_kwargs=hf_model_kwargs,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, vllm_output, atol=3.2e-2, rtol=1e-3)


@pytest.mark.parametrize("model", ["bd2lcco/Qwen3-0.6B-finetuned"])
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
# 测试自动转换功能：验证 Qwen3 模型自动转换为 token 分类模式后的输出正确性
def test_auto_conversion(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=1024, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForTokenClassification
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        assert torch.allclose(hf_output, vllm_output, atol=1e-2)
