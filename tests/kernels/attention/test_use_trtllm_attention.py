# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 测试TRTLLM注意力后端的可用性判断和使用条件检查逻辑

from unittest.mock import patch

import pytest
import torch

from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    supports_trtllm_attention,
    use_trtllm_attention,
)

MODEL_CONFIGS = {
    "Llama-3-70B": dict(num_qo_heads=64, num_kv_heads=8),
    "Llama-3-8B": dict(num_qo_heads=32, num_kv_heads=8),
    "Qwen2.5-0.5B": dict(num_qo_heads=14, num_kv_heads=2),
    "Mistral-7B": dict(num_qo_heads=32, num_kv_heads=8),
    "Gemma-2-9B": dict(num_qo_heads=8, num_kv_heads=4),
    "Falcon-40B": dict(num_qo_heads=128, num_kv_heads=8),
}


# 获取指定模型的注意力头配置
def get_config(model: str) -> dict:
    """Return the attention config for a model."""
    return MODEL_CONFIGS[model]


DEFAULT_KWARGS = dict(
    **get_config("Llama-3-70B"),
    num_tokens=128,
    max_seq_len=4096,
    dcp_world_size=1,
    kv_cache_dtype="auto",
    q_dtype=torch.bfloat16,
    is_prefill=False,
    force_use_trtllm=None,
    has_sinks=False,
    has_spec=False,
)


# 使用默认参数调用use_trtllm_attention并支持参数覆盖
def _call(**overrides) -> bool:
    kwargs = {**DEFAULT_KWARGS, **overrides}
    return use_trtllm_attention(**kwargs)


# 每个测试用例前清除supports_trtllm_attention的functools.cache
@pytest.fixture(autouse=True)
def _clear_supports_cache():
    """Clear functools.cache to ensure each test runs independently."""
    supports_trtllm_attention.cache_clear()


# supports_trtllm_attention


# 测试batch_invariant模式下TRTLLM注意力被禁用
@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=True)
def test_supports_batch_invariant_disables(_mock):
    assert supports_trtllm_attention() is False


# 测试SM100平台且有NVIDIA artifactory时TRTLLM注意力可用
@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=False)
@patch(
    "vllm.utils.flashinfer.current_platform.is_device_capability_family",
    return_value=True,
)
@patch("vllm.utils.flashinfer.has_nvidia_artifactory", return_value=True)
def test_supports_sm100_with_artifactory(_art, _cap, _bi):
    assert supports_trtllm_attention() is True


# 测试非SM100平台下TRTLLM注意力不可用
@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=False)
@patch(
    "vllm.utils.flashinfer.current_platform.is_device_capability_family",
    return_value=False,
)
def test_supports_non_sm100_platform(_cap, _bi):
    assert supports_trtllm_attention() is False


# 测试SM100平台但缺少NVIDIA artifactory时TRTLLM注意力不可用
@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=False)
@patch(
    "vllm.utils.flashinfer.current_platform.is_device_capability_family",
    return_value=True,
)
@patch("vllm.utils.flashinfer.has_nvidia_artifactory", return_value=False)
def test_supports_sm100_without_artifactory(_art, _cap, _bi):
    assert supports_trtllm_attention() is False


# can_use_trtllm_attention


# 测试强制禁用时can_use_trtllm_attention返回False
@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=False)
def test_can_use_force_disabled(_mock):
    cfg = get_config("Llama-3-70B")
    assert can_use_trtllm_attention(cfg["num_qo_heads"], cfg["num_kv_heads"]) is False


# 测试兼容头数配置时can_use_trtllm_attention返回True
@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=None)
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_can_use_compatible_heads(_sup, _force):
    cfg = get_config("Llama-3-70B")
    assert can_use_trtllm_attention(cfg["num_qo_heads"], cfg["num_kv_heads"]) is True


# 测试不兼容头数配置时can_use_trtllm_attention返回False
@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=None)
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_can_use_incompatible_heads(_sup, _force):
    assert can_use_trtllm_attention(40, 6) is False


# 测试平台不支持时所有模型配置的can_use_trtllm_attention均返回False
@pytest.mark.parametrize("model", list(MODEL_CONFIGS.keys()))
@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=None)
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=False)
def test_can_use_platform_unsupported(_sup, _force, model):
    cfg = get_config(model)
    assert can_use_trtllm_attention(cfg["num_qo_heads"], cfg["num_kv_heads"]) is False


# use_trtllm_attention


# 测试force_use_trtllm=False时use_trtllm_attention返回False
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_force_off(_mock):
    assert _call(force_use_trtllm=False) is False


# 测试DCP多卡（dcp_world_size>1）时回退到非TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_dcp_fallback(_mock):
    assert _call(dcp_world_size=2) is False


# 测试平台不支持时use_trtllm_attention返回False
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=False)
def test_use_platform_unsupported(_mock):
    assert _call() is False


# 测试平台不支持时即使force_on也返回False
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=False)
def test_use_platform_unsupported_force_on_still_false(_mock):
    assert _call(force_use_trtllm=True) is False


# 测试不兼容头数时use_trtllm_attention返回False
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_incompatible_heads(_mock):
    assert _call(num_qo_heads=40, num_kv_heads=6) is False


# 测试不兼容头数即使force_on也返回False
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_incompatible_heads_force_on_still_false(_mock):
    assert _call(num_qo_heads=40, num_kv_heads=6, force_use_trtllm=True) is False


# 测试投机解码场景下启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_spec_decode_enables(_mock):
    assert _call(has_spec=True, is_prefill=False) is True


# 测试FP8查询类型强制启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
@patch(
    "vllm.utils.flashinfer.current_platform.fp8_dtype",
    return_value=torch.float8_e4m3fn,
)
def test_use_fp8_query_forces_trtllm(_fp8, _sup):
    assert _call(q_dtype=torch.float8_e4m3fn) is True


# 测试有注意力sink时强制启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_sinks_force_trtllm(_mock):
    assert _call(has_sinks=True) is True


# 测试自动模式下预填充阶段KV缓存为auto时启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_prefill_kv_auto(_mock):
    assert _call(is_prefill=True, kv_cache_dtype="auto") is True


# 测试预填充阶段KV缓存为FP8时不启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_prefill_kv_fp8(_mock):
    assert _call(is_prefill=True, kv_cache_dtype="fp8") is False


# 测试解码阶段小batch时启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_decode_small_batch(_mock):
    assert _call(is_prefill=False, num_tokens=128, kv_cache_dtype="auto") is True


# 测试解码阶段大batch时不启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_decode_large_batch(_mock):
    assert _call(is_prefill=False, num_tokens=512, kv_cache_dtype="auto") is False


# 测试force_use_trtllm=True时强制启用TRTLLM注意力
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_force_on(_mock):
    assert _call(force_use_trtllm=True) is True
