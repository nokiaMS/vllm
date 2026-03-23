# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser as Holo2ReasoningParser,
)
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

REASONING_MODEL_NAME = "HCompany/Holo2-4B"


# [中文注释] 加载Holo2模型分词器夹具
@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# [中文注释] 测试Holo2根据thinking参数选择DeepSeekR1或Identity解析器
@pytest.mark.parametrize(
    "thinking,expected_parser_type",
    [
        (True, DeepSeekR1ReasoningParser),
        (False, IdentityReasoningParser),
    ],
)
def test_parser_selection(tokenizer, thinking, expected_parser_type):
    parser = Holo2ReasoningParser(
        tokenizer,
        chat_template_kwargs={
            "thinking": thinking,
        },
    )

    assert isinstance(parser._parser, expected_parser_type)


# [中文注释] 测试Holo2默认使用DeepSeekR1解析器
def test_holo2_default_parser_is_deepseekr1(tokenizer):
    parser = Holo2ReasoningParser(tokenizer)

    assert isinstance(parser._parser, DeepSeekR1ReasoningParser)


# [中文注释] 测试Holo2支持结构化输出：验证推理结束判断在StructuredOutputManager中的正确性
def test_holo2_supports_structured_output(tokenizer):
    # Structured output manager uses the reasoning parser to check if the
    # reasoning content is ended before applying the grammar. The main function
    # used is is_reasoning_end. This test checks if the parser is able to
    # correctly identify the end of the reasoning content.

    # important to not pass chat_template_kwargs here as it is done in the
    # StructuredOutputManager
    parser = Holo2ReasoningParser(tokenizer)

    end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    assert parser.is_reasoning_end([1, 2, 4, end_token_id])
    assert not parser.is_reasoning_end([1, 2, 4])
    assert parser.is_reasoning_end([1, 2, 4, end_token_id, 5])


# thinking is True, non-streaming
WITH_THINK = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
# thinking is True, streaming
WITH_THINK_STREAM = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
# thinking is False, non-streaming
THINKING_DISABLED = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}
# thinking is False, streaming
THINKING_DISABLED_STREAM = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}
# thinking is False but the model output </think>, non-streaming
THINKING_DISABLED_WITH_CLOSE_TAG = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "</think>This is the rest",
}
# thinking is False but the model output </think>, streaming
THINKING_DISABLED_WITH_CLOSE_TAG_STREAM = {
    "output": "some text</think>This is the rest",
    "reasoning": None,
    "content": "some text</think>This is the rest",
}
COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}

TEST_CASES = [
    pytest.param(
        False,
        WITH_THINK,
        None,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        None,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        WITH_THINK,
        {"thinking": True},
        id="with_think_enabled",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        {"thinking": True},
        id="with_think_stream_enabled",
    ),
    pytest.param(
        False,
        THINKING_DISABLED,
        {"thinking": False},
        id="thinking_disabled",
    ),
    pytest.param(
        True,
        THINKING_DISABLED_STREAM,
        {"thinking": False},
        id="thinking_disabled_stream",
    ),
    pytest.param(
        False,
        THINKING_DISABLED_WITH_CLOSE_TAG,
        {"thinking": False},
        id="thinking_disabled_with_close_tag",
    ),
    pytest.param(
        True,
        THINKING_DISABLED_WITH_CLOSE_TAG_STREAM,
        {"thinking": False},
        id="thinking_disabled_with_close_tag_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        None,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        None,
        id="complete_reasoning_stream",
    ),
]


# [中文注释] 参数化测试Holo2推理提取：支持chat_template_kwargs控制thinking行为
@pytest.mark.parametrize("streaming, param_dict, chat_template_kwargs", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    chat_template_kwargs: dict | None,
    tokenizer,
):
    output = tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser("holo2")(
        tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]
