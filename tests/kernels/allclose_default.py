# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 提供不同数据类型（float16/bfloat16/float32）的默认绝对误差和相对误差阈值，
# 用于内核测试中的数值精度比较
import torch

# Reference default values of atol and rtol are from
# https://github.com/pytorch/pytorch/blob/6d96beb6bec24d73ee3f080bac54d2104068f675/test/test_transformers.py#L67
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float: 1.3e-6}


# 根据输出张量的数据类型返回默认的绝对误差容忍度
def get_default_atol(output) -> float:
    return default_atol[output.dtype]


# 根据输出张量的数据类型返回默认的相对误差容忍度
def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]
