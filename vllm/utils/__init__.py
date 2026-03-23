# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vllm/utils 包的初始化模块，提供通用的基础工具函数，
# 包括 UUID 生成和请求长度计算等核心辅助功能。

import uuid

import torch

# 64 位掩码常量，用于将 UUID 截断为 64 位整数，以生成紧凑的十六进制标识符
MASK_64_BITS = (1 << 64) - 1


# 生成一个 16 位十六进制字符的随机 UUID 字符串，
# 通过对标准 UUID 进行 64 位截断来获得更短的标识符
def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


# 根据 prompt_token_ids 或 prompt_embeds 计算请求的 token 长度。
# 设计思路：优先使用 token_ids 的长度，若不存在则回退到 embeds 的长度；
# 若两者都提供且长度不一致则抛出异常，确保数据一致性。
def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len
