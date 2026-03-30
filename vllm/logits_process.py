# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence  # 导入Callable和Sequence抽象基类
from typing import TypeAlias  # 导入TypeAlias用于类型别名定义

import torch  # 导入PyTorch库

from vllm.tokenizers import TokenizerLike  # 导入分词器类型接口

LogitsProcessor: TypeAlias = (  # 定义LogitsProcessor类型别名
    Callable[[list[int], torch.Tensor], torch.Tensor]  # 接受已生成token列表和logits张量，返回修改后的logits张量
    | Callable[[list[int], list[int], torch.Tensor], torch.Tensor]  # 或者额外接受prompt token列表作为第一个参数
)
"""LogitsProcessor is a function that takes a list
of previously generated tokens, the logits tensor
for the next token and, optionally, prompt tokens as a
first argument, and returns a modified tensor of logits
to sample from."""


def get_bad_words_logits_processors(
    bad_words: list[str], tokenizer: TokenizerLike
) -> list[LogitsProcessor]:
    """获取禁用词logits处理器列表。

    根据给定的禁用词列表和分词器，创建并返回用于屏蔽禁用词的logits处理器。

    Args:
        bad_words: 需要禁用的词语字符串列表。
        tokenizer: 用于将词语编码为token ID的分词器。

    Returns:
        包含NoBadWordsLogitsProcessor的logits处理器列表。
    """
    bad_words_ids: list[list[int]] = list()  # 初始化禁用词的token ID列表

    for bad_word in bad_words:  # 遍历每个禁用词
        # To prohibit words both at the beginning
        # and in the middle of text
        # (related to add_prefix_space tokenizer parameter)
        for add_prefix_space in [False, True]:  # 分别尝试不加前缀空格和加前缀空格
            prefix = " " if add_prefix_space else ""  # 根据标志设置前缀空格
            prompt = prefix + bad_word.lstrip()  # 拼接前缀空格和去除左侧空格后的禁用词

            prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)  # 将禁用词编码为token ID，不添加特殊token

            # If no space at the beginning
            # or if prefix space produces a new word token
            if (not add_prefix_space) or (  # 如果不加前缀空格
                add_prefix_space  # 或者加了前缀空格
                and prompt_token_ids[0] != bad_words_ids[-1][0]  # 且首个token ID与上一个不同
                and len(prompt_token_ids) == len(bad_words_ids[-1])  # 且token ID长度相同
            ):
                bad_words_ids.append(prompt_token_ids)  # 将该禁用词的token ID列表添加到结果中

    return [NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids)]  # 返回包含禁用词处理器的列表


class NoBadWordsLogitsProcessor:
    """禁用词Logits处理器。

    通过将禁用词对应的logits设置为负无穷来阻止模型生成这些词语。
    支持单token和多token的禁用词。对于单token禁用词，直接在词偏置中设置；
    对于多token禁用词，在运行时检查已生成的token前缀是否匹配。
    """

    _SMALLEST_LOGIT = float("-inf")  # 最小logit值，设为负无穷
    _NEUTRAL_LOGIT = 0.0  # 中性logit值，不产生影响

    def __init__(self, bad_words_ids: list[list[int]]):
        """初始化禁用词Logits处理器。

        Args:
            bad_words_ids: 禁用词的token ID列表的列表，每个内部列表表示一个禁用词的token序列。
        """
        self.bad_words_ids = bad_words_ids  # 保存禁用词的token ID列表
        self.word_bias: torch.FloatTensor = None  # 初始化词偏置张量为None，稍后延迟创建

    def __call__(
        self,
        past_tokens_ids: Sequence[int],
        logits: torch.FloatTensor,
    ) -> torch.Tensor:
        """调用处理器，修改logits以屏蔽禁用词。

        Args:
            past_tokens_ids: 已生成的token ID序列。
            logits: 下一个token的logits张量。

        Returns:
            修改后的logits张量，禁用词对应位置的logit被设为负无穷。
        """
        if self.word_bias is None:  # 如果词偏置尚未初始化
            self._init_word_bias(logits=logits)  # 初始化词偏置

        last_token_bias = torch.zeros_like(logits)  # 创建与logits形状相同的零张量，用于多token禁用词的偏置

        for bad_word_ids in self.bad_words_ids:  # 遍历每个禁用词的token ID序列
            if len(bad_word_ids) == 1:  # 1-token words already processed  # 单token禁用词已在word_bias中处理，跳过
                continue

            if len(bad_word_ids) > len(past_tokens_ids) + 1:  # 如果禁用词长度超过已生成token数+1，无法匹配，跳过
                continue

            prefix_length = len(bad_word_ids) - 1  # 计算禁用词前缀长度（不含最后一个token）
            last_token_id = bad_word_ids[-1]  # 获取禁用词的最后一个token ID
            actual_prefix = past_tokens_ids[-prefix_length:]  # 从已生成token中取出与前缀等长的尾部
            expected_prefix = bad_word_ids[:prefix_length]  # 获取禁用词的前缀部分

            assert len(actual_prefix) == len(expected_prefix)  # 断言实际前缀和期望前缀长度一致

            is_match = tuple(actual_prefix) == tuple(expected_prefix)  # 检查实际前缀是否与期望前缀匹配
            last_token_bias[last_token_id] += (  # 根据匹配结果设置最后一个token的偏置
                self._SMALLEST_LOGIT if is_match else self._NEUTRAL_LOGIT  # 匹配则设为负无穷，不匹配则为0
            )

        logits = logits + self.word_bias + last_token_bias  # 将词偏置和最后token偏置加到logits上

        return logits  # 返回修改后的logits

    def _init_word_bias(self, logits: torch.FloatTensor) -> None:
        """初始化词偏置张量，为单token禁用词设置负无穷偏置。

        Args:
            logits: 当前的logits张量，用于获取词表大小和设备信息。
        """
        # Code based on NoBadWordsLogitsProcessor and SequenceBiasLogitsProcessor  # noqa: E501
        # from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

        vocab_size = logits.shape[-1]  # 获取词表大小

        self._check_token_ids_bounds(vocab_size=vocab_size)  # 检查所有token ID是否在词表范围内

        self.word_bias = torch.zeros(  # 创建词偏置张量，初始值全为0
            (vocab_size,), dtype=torch.float, device=logits.device  # 形状为词表大小，浮点类型，与logits相同设备
        )

        for bad_word_ids in self.bad_words_ids:  # 遍历每个禁用词
            if len(bad_word_ids) == 1:  # 如果是单token禁用词
                bad_word_id = bad_word_ids[-1]  # 获取该token ID
                self.word_bias[bad_word_id] = self._SMALLEST_LOGIT  # 将对应位置的偏置设为负无穷

    def _check_token_ids_bounds(self, vocab_size: int) -> None:
        """检查禁用词中的token ID是否在词表范围内。

        Args:
            vocab_size: 词表大小。

        Raises:
            ValueError: 如果存在超出词表范围的token ID。
        """
        invalid_token_ids = []  # 初始化无效token ID列表

        for bad_word_ids in self.bad_words_ids:  # 遍历每个禁用词
            for token_id in bad_word_ids:  # 遍历禁用词中的每个token ID
                if token_id < 0 or token_id >= vocab_size:  # 如果token ID小于0或大于等于词表大小
                    invalid_token_ids.append(token_id)  # 添加到无效列表

        if len(invalid_token_ids) > 0:  # 如果存在无效的token ID
            raise ValueError(  # 抛出值错误异常
                f"The model vocabulary size is {vocab_size},"  # 提示模型词表大小
                f" but the following tokens"  # 提示以下token被指定为禁用词
                f" were specified as bad: {invalid_token_ids}."  # 列出无效的token ID
                f" All token id values should be integers satisfying:"  # 提示token ID应满足的条件
                f" 0 <= token_id < {vocab_size}."  # 提示有效范围
            )
