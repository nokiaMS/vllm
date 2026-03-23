# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试 IntermediateTensors 的相等性比较：验证不同类型、不同键、不同值的比较逻辑]

import torch

from vllm.sequence import IntermediateTensors


# [测试 IntermediateTensors 对象之间的相等性判断：类型不同、键不同、值不同时应不等，完全一致时应相等]
def test_sequence_intermediate_tensors_equal():
    class AnotherIntermediateTensors(IntermediateTensors):
        pass

    intermediate_tensors = IntermediateTensors({})
    another_intermediate_tensors = AnotherIntermediateTensors({})
    assert intermediate_tensors != another_intermediate_tensors

    empty_intermediate_tensors_1 = IntermediateTensors({})
    empty_intermediate_tensors_2 = IntermediateTensors({})
    assert empty_intermediate_tensors_1 == empty_intermediate_tensors_2

    different_key_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    difference_key_intermediate_tensors_2 = IntermediateTensors(
        {"2": torch.zeros([2, 4], dtype=torch.int32)}
    )
    assert different_key_intermediate_tensors_1 != difference_key_intermediate_tensors_2

    same_key_different_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    same_key_different_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 5], dtype=torch.int32)}
    )
    assert (
        same_key_different_value_intermediate_tensors_1
        != same_key_different_value_intermediate_tensors_2
    )

    same_key_same_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    same_key_same_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    assert (
        same_key_same_value_intermediate_tensors_1
        == same_key_same_value_intermediate_tensors_2
    )
