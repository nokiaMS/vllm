# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试列排列内核permute_cols的正确性，验证CUDA实现与PyTorch索引操作的一致性
import pytest
import torch

from tests.kernels.utils import opcheck
from vllm._custom_ops import permute_cols

if not hasattr(torch.ops._C, "permute_cols"):
    pytest.skip(reason="permute_cols is not supported on ROCm", allow_module_level=True)


@pytest.mark.parametrize("shape", [(1, 512), (544, 4096), (67, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# 测试permute_cols在不同矩阵形状下的列重排正确性和opcheck合规性
def test_permute_cols(shape, dtype):
    x = torch.randn(shape, dtype=dtype).cuda()
    perm = torch.randperm(x.shape[1]).to(torch.int).cuda()
    opcheck(torch.ops._C.permute_cols, (x, perm))
    y = permute_cols(x, perm)
    torch.testing.assert_close(y, x[:, perm])
