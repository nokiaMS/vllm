# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 提供注意力测试所需的KV缓存工厂fixture

import pytest

from vllm.utils.torch_utils import (
    create_kv_caches_with_random,
    create_kv_caches_with_random_flash,
)


# 提供用于测试的随机KV缓存工厂fixture
@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches_with_random


# 提供用于FlashInfer测试的随机KV缓存工厂fixture
@pytest.fixture()
def kv_cache_factory_flashinfer():
    return create_kv_caches_with_random_flash
