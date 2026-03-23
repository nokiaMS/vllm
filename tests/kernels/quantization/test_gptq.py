# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 测试GPTQ量化内核（shuffle和gemm操作）的opcheck验证

import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


# 验证GPTQ权重shuffle操作的opcheck一致性
def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    perm = torch.empty((0,), device="cuda", dtype=torch.int32)
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, perm, bit))


# 验证GPTQ GEMM操作（GPTQv1和GPTQv2格式）的opcheck一致性
def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    idx = torch.empty((0,), device="cuda", dtype=torch.int32)
    use_exllama = True
    bit = 4
    # Test both GPTQv1 and GPTQv2 format
    opcheck(
        torch.ops._C.gptq_gemm, (a, weight, zeros, scales, idx, use_exllama, True, bit)
    )
    opcheck(
        torch.ops._C.gptq_gemm, (a, weight, zeros, scales, idx, use_exllama, False, bit)
    )
