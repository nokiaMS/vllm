# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试随机种子行为：验证 seed_everything 能让 random、numpy、torch 生成可重复结果]
import random

import numpy as np
import torch

from vllm.platforms.interface import Platform


# [测试设置相同种子后，random/numpy/torch 三个库生成的随机数完全一致]
def test_seed_behavior():
    # Test with a specific seed
    Platform.seed_everything(42)
    random_value_1 = random.randint(0, 100)
    np_random_value_1 = np.random.randint(0, 100)
    torch_random_value_1 = torch.randint(0, 100, (1,)).item()

    Platform.seed_everything(42)
    random_value_2 = random.randint(0, 100)
    np_random_value_2 = np.random.randint(0, 100)
    torch_random_value_2 = torch.randint(0, 100, (1,)).item()

    assert random_value_1 == random_value_2
    assert np_random_value_1 == np_random_value_2
    assert torch_random_value_1 == torch_random_value_2
