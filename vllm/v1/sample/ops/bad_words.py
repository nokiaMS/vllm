# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch  # 导入PyTorch张量库

_SMALLEST_LOGIT = float("-inf")  # 定义最小logit值为负无穷，用于屏蔽禁用词


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    """
    对单个批次样本应用坏词过滤。

    检查已生成的token序列是否匹配坏词的前缀，
    如果匹配则将坏词的最后一个token的logit设为负无穷。

    Args:
        logits: 单个样本的logits张量。
        bad_words_token_ids: 需要屏蔽的坏词token ID列表，每个坏词是一个token ID序列。
        past_tokens_ids: 已生成的token ID列表。
    """
    for bad_word_ids in bad_words_token_ids:  # 遍历每个坏词序列
        if len(bad_word_ids) > len(past_tokens_ids) + 1:  # 如果坏词长度超过已生成序列+1，跳过
            continue

        prefix_length = len(bad_word_ids) - 1  # 计算坏词前缀长度（不含最后一个token）
        last_token_id = bad_word_ids[-1]  # 获取坏词的最后一个token ID
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []  # 提取已生成序列的尾部作为实际前缀
        expected_prefix = bad_word_ids[:prefix_length]  # 获取坏词的期望前缀

        assert len(actual_prefix) == len(expected_prefix)  # 断言前缀长度一致

        if actual_prefix == expected_prefix:  # 如果实际前缀匹配坏词前缀
            logits[last_token_id] = _SMALLEST_LOGIT  # 将坏词最后一个token的logit设为负无穷


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    """
    对整个批次应用坏词过滤。

    Args:
        logits: 批次logits张量，形状为(batch_size, vocab_size)。
        bad_words_token_ids: 请求索引到坏词列表的映射字典。
        past_tokens_ids: 每个请求已生成的token ID列表。
    """
    for i, bad_words_ids in bad_words_token_ids.items():  # 遍历需要应用坏词过滤的请求
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])  # 对每个请求单独应用


def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> None:
    """
    对带有推测解码草稿token的批次应用坏词过滤。

    Args:
        logits: 批次logits张量。
        bad_words_token_ids: 请求索引到坏词列表的映射字典。
        past_tokens_ids: 每个请求已生成的token ID列表。
        num_draft_tokens: 每个请求的草稿token数量列表。
    """
    start_idx = 0  # 当前请求在logits中的起始索引
    remaining = len(bad_words_token_ids)  # 剩余需要处理的请求数
    for i, n in enumerate(num_draft_tokens):  # 遍历每个请求及其草稿token数
        if (bad_words_ids := bad_words_token_ids.get(i)) is not None:  # 如果该请求有坏词需要过滤
            for draft_idx in range(start_idx, start_idx + n):  # 遍历该请求的所有草稿位置
                _apply_bad_words_single_batch(  # 对每个草稿位置应用坏词过滤
                    logits[draft_idx],
                    bad_words_ids,
                    past_tokens_ids[draft_idx],
                )
            remaining -= 1  # 已处理请求数减一
            if not remaining:  # 如果所有需要处理的请求都已完成
                break  # 提前退出循环
        start_idx += n  # 更新起始索引到下一个请求
