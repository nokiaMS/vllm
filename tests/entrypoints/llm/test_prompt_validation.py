# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试提示验证：空提示拒绝、超出词表 token 拒绝、缺少 mm_embeds 标志时的错误提示

import pytest
import torch

from vllm import LLM


# 测试空字符串提示被正确拒绝
def test_empty_prompt():
    llm = LLM(model="openai-community/gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match="decoder prompt cannot be empty"):
        llm.generate([""])


# 测试超出词表范围的 token ID 被正确拒绝
def test_out_of_vocab_token():
    llm = LLM(model="openai-community/gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match="out of vocabulary"):
        llm.generate({"prompt_token_ids": [999999]})


# 测试未启用 enable_mm_embeds 时传入嵌入数据的错误提示
def test_require_mm_embeds():
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        enforce_eager=True,
        enable_mm_embeds=False,
    )
    with pytest.raises(ValueError, match="--enable-mm-embeds"):
        llm.generate(
            {
                "prompt": "<image>",
                "multi_modal_data": {"image": torch.empty(1, 1, 1)},
            }
        )
