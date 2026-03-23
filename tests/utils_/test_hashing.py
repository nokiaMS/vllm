# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import pickle

import pytest

from vllm.utils.hashing import sha256


# [中文注释] 测试sha256哈希函数：验证不同输入的哈希结果一致性、非空性和唯一性
@pytest.mark.parametrize("input", [(), ("abc",), (None,), (None, bool, [1, 2, 3])])
def test_sha256(input: tuple):
    digest = sha256(input)
    assert digest is not None
    assert isinstance(digest, bytes)
    assert digest != b""

    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    assert digest == hashlib.sha256(input_bytes).digest()

    # hashing again, returns the same value
    assert digest == sha256(input)

    # hashing different input, returns different value
    assert digest != sha256(input + (1,))
