# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试模型注册表功能，包括模型导入、属性检查、流水线并行支持和注册表覆盖率验证

import warnings

import pytest
import torch.cuda

from vllm.model_executor.models import (
    is_pooling_model,
    is_text_generation_model,
    supports_multimodal,
)
from vllm.model_executor.models.adapters import (
    as_embedding_model,
    as_seq_cls_model,
)
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
    ModelRegistry,
)
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
# 测试所有注册的模型架构能否正确导入并满足类型约束
def test_registry_imports(model_arch):
    # Skip if transformers version is incompatible
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )
    # Ensure all model classes can be imported successfully
    model_cls = ModelRegistry._try_load_model_cls(model_arch)
    assert model_cls is not None

    if model_arch in _SPECULATIVE_DECODING_MODELS:
        return  # Ignore these models which do not have a unified format

    if model_arch in _TEXT_GENERATION_MODELS or model_arch in _MULTIMODAL_MODELS:
        assert is_text_generation_model(model_cls)

    # All vLLM models should be convertible to a pooling model
    assert is_pooling_model(as_seq_cls_model(model_cls))
    assert is_pooling_model(as_embedding_model(model_cls))

    if model_arch in _MULTIMODAL_MODELS:
        assert supports_multimodal(model_cls)


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,is_mm,init_cuda,score_type",
    [
        ("LlamaForCausalLM", False, False, "bi-encoder"),
        ("LlavaForConditionalGeneration", True, True, "bi-encoder"),
        ("BertForSequenceClassification", False, False, "cross-encoder"),
        ("RobertaForSequenceClassification", False, False, "cross-encoder"),
        ("XLMRobertaForSequenceClassification", False, False, "cross-encoder"),
        ("GteNewModel", False, False, "bi-encoder"),
        ("GteNewForSequenceClassification", False, False, "cross-encoder"),
        ("HF_ColBERT", False, False, "late-interaction"),
    ],
)
# 测试模型注册表中的多模态支持和评分类型等属性是否正确
def test_registry_model_property(model_arch, is_mm, init_cuda, score_type):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_multimodal is is_mm
    assert model_info.score_type == score_type

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry._try_load_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2,
            )


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,is_pp,init_cuda",
    [
        # TODO(woosuk): Re-enable this once the MLP Speculator is supported
        # in V1.
        # ("MLPSpeculatorPreTrainedModel", False, False),
        ("DeepseekV2ForCausalLM", True, False),
        ("Qwen2VLForConditionalGeneration", True, True),
    ],
)
# 测试模型是否正确报告流水线并行（Pipeline Parallel）支持
def test_registry_is_pp(model_arch, is_pp, init_cuda):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_pp is is_pp

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry._try_load_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2,
            )


# 测试HuggingFace注册表是否覆盖了所有支持的模型架构
def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )
