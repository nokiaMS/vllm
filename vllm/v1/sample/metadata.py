# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass  # 导入数据类装饰器

import torch  # 导入PyTorch张量库

from vllm.v1.sample.logits_processor import LogitsProcessors  # 导入logits处理器集合类


@dataclass
class SamplingMetadata:
    """采样元数据，包含采样过程中所需的全部参数和状态信息。"""

    temperature: torch.Tensor | None  # 温度参数张量，用于控制采样随机性
    all_greedy: bool  # 是否所有请求都使用贪心采样
    all_random: bool  # 是否所有请求都使用随机采样

    top_p: torch.Tensor | None  # top-p（核采样）阈值张量
    top_k: torch.Tensor | None  # top-k采样阈值张量

    generators: dict[int, torch.Generator]  # 每个请求的随机数生成器字典（请求索引→生成器）

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None  # 最大logprobs数量，None表示不需要logprobs，0表示仅采样token的logprobs

    no_penalties: bool  # 是否不应用任何惩罚
    prompt_token_ids: torch.Tensor | None  # 提示token ID张量，用于计算重复惩罚
    frequency_penalties: torch.Tensor  # 频率惩罚张量
    presence_penalties: torch.Tensor  # 存在惩罚张量
    repetition_penalties: torch.Tensor  # 重复惩罚张量

    output_token_ids: list[list[int]]  # 每个请求已生成的输出token ID列表

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None  # 允许的token ID掩码，形状为(最大批次大小, 词表大小)

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]  # 需要屏蔽的坏词token ID映射（请求索引→坏词列表）

    # Loaded logits processors
    logitsprocs: LogitsProcessors  # 已加载的logits处理器集合

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None  # 推测解码的token ID列表
