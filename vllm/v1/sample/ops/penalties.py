# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch  # 导入PyTorch张量库

from vllm.model_executor.layers.utils import apply_penalties  # 导入底层惩罚应用函数
from vllm.utils.platform_utils import is_pin_memory_available  # 导入固定内存可用性检查工具
from vllm.utils.torch_utils import make_tensor_with_pad  # 导入带填充的张量创建工具


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    对logits应用存在惩罚、频率惩罚和重复惩罚。

    Args:
        logits: 模型输出的logits张量。
        prompt_token_ids: 提示token ID张量。
        presence_penalties: 存在惩罚系数张量。
        frequency_penalties: 频率惩罚系数张量。
        repetition_penalties: 重复惩罚系数张量。
        output_token_ids: 每个请求已生成的输出token ID列表。

    Returns:
        应用惩罚后的logits张量。
    """
    _, vocab_size = logits.shape  # 获取词表大小
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)  # 将输出token列表转换为张量

    # 在异步调度场景中，不需要应用惩罚的行可能包含-1占位符token ID。
    # 必须将这些替换为有效的token ID，以确保apply_penalties中的scatter操作有效。
    # 注意：当前惩罚实现效率较低，后续将重构。
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)  # 将-1占位符替换为vocab_size

    return apply_penalties(  # 调用底层函数应用所有惩罚
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    将不同的列表数据结构转换为张量。

    Args:
        output_token_ids: 输出token ID的嵌套列表。
        vocab_size: 词表大小，用作填充值。
        device: 目标设备。

    Returns:
        转换后的张量，已移动到目标设备。
    """
    output_tokens_tensor = make_tensor_with_pad(  # 创建带填充的张量
        output_token_ids,
        # 使用vocab_size作为填充值，因为不存在该值对应的token ID
        pad=vocab_size,
        device="cpu",  # 先在CPU上创建
        dtype=torch.int64,  # 使用64位整数类型
        pin_memory=is_pin_memory_available(),  # 如果可用则使用固定内存加速传输
    )
    return output_tokens_tensor.to(device, non_blocking=True)  # 异步传输到目标设备
