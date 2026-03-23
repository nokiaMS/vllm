# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 测试AWQ（Activation-aware Weight Quantization）反量化和GEMM的opcheck验证

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


# 验证AWQ反量化操作的opcheck一致性
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "awq_dequantize"),
    reason="AWQ is not supported on this GPU type.",
)
def test_awq_dequantize_opcheck(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_TRITON_AWQ", "0")
        qweight = torch.randint(
            -2000000000, 2000000000, (8192, 256), device="cuda", dtype=torch.int32
        )
        scales = torch.rand((64, 2048), device="cuda", dtype=torch.float16)
        zeros = torch.empty((64, 256), device="cuda", dtype=torch.int32)
        split_k_iters = 0
        thx = 0
        thy = 0
        opcheck(
            torch.ops._C.awq_dequantize,
            (qweight, scales, zeros, split_k_iters, thx, thy),
        )


# 验证AWQ GEMM操作的opcheck一致性（当前已跳过待修复）
@pytest.mark.skip(reason="Not working; needs investigation.")
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "awq_gemm"),
    reason="AWQ is not supported on this GPU type.",
)
def test_awq_gemm_opcheck(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_TRITON_AWQ", "0")
        input = torch.rand((2, 8192), device="cuda", dtype=torch.float16)
        qweight = torch.randint(
            -2000000000, 2000000000, (8192, 256), device="cuda", dtype=torch.int32
        )
        scales = torch.empty((64, 2048), device="cuda", dtype=torch.float16)
        qzeros = torch.randint(
            -2000000000, 2000000000, (64, 256), device="cuda", dtype=torch.int32
        )
        split_k_iters = 8
        opcheck(torch.ops._C.awq_gemm, (input, qweight, scales, qzeros, split_k_iters))
