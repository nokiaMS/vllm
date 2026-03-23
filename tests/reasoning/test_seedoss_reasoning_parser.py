# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "seed_oss"
start_token = "<seed:think>"
end_token = "</seed:think>"

# Use a test model that contains our custom tokens
REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# [中文注释] 加载SeedOSS分词器夹具：添加自定义<seed:think>/<\/seed:think>特殊token
@pytest.fixture(scope="module")
def seedoss_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)
    # Add custom SeedOSS tokens if they don't exist
    if start_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([start_token, end_token])
    return tokenizer


SIMPLE_REASONING: dict[str, Any] = {
    "output": "This is a reasoning section</seed:think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING: dict[str, Any] = {
    "output": "This is a reasoning section</seed:think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
NO_CONTENT: dict[str, Any] = {
    "output": "This is content",
    "reasoning": "This is content",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING_STREAMING: dict[str, Any] = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES: dict[str, Any] = {
    "output": "This\nThat</seed:think>This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
WITH_START_TOKEN: dict[str, Any] = {
    "output": ("<seed:think>This is a reasoning section</seed:think>This is the rest"),
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
ONLY_END_TOKEN: dict[str, Any] = {
    "output": "Some reasoning</seed:think>This is the rest",
    "reasoning": "Some reasoning",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
NO_TOKENS: dict[str, Any] = {
    "output": "This is just content without any reasoning tokens",
    "reasoning": "This is just content without any reasoning tokens",
    "content": None,
    "is_reasoning_end": False,
}


# [中文注释] 测试SeedOSS推理解析器的创建和注册
def test_seedoss_reasoning_parser_creation(seedoss_tokenizer):
    """Test that the SeedOSS reasoning parser can be created and registered."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)
    assert isinstance(parser, ReasoningParser)
    assert parser.start_token == start_token
    assert parser.end_token == end_token


# [中文注释] 测试SeedOSS基本推理提取：双标记场景
@pytest.mark.parametrize("streaming", [True, False])
def test_simple_reasoning(seedoss_tokenizer, streaming):
    """Test basic reasoning extraction with both tokens."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, SIMPLE_REASONING["output"])], streaming=streaming
    )

    assert reasoning == SIMPLE_REASONING["reasoning"]
    assert content == SIMPLE_REASONING["content"]


# [中文注释] 测试推理后无内容的场景
@pytest.mark.parametrize("streaming", [True, False])
def test_complete_reasoning(seedoss_tokenizer, streaming):
    """Test reasoning extraction when there's no content after reasoning."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, COMPLETE_REASONING["output"])], streaming=streaming
    )

    assert reasoning == COMPLETE_REASONING["reasoning"]
    assert content == COMPLETE_REASONING["content"]


# [中文注释] 测试无结束标记时所有内容视为推理
@pytest.mark.parametrize("streaming", [True, False])
def test_no_content(seedoss_tokenizer, streaming):
    """Test when there's no end token - everything is reasoning content."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, NO_CONTENT["output"])], streaming=streaming
    )

    assert reasoning == NO_CONTENT["reasoning"]
    assert content == NO_CONTENT["content"]


# [中文注释] 测试多行推理内容的提取
@pytest.mark.parametrize("streaming", [True, False])
def test_multiple_lines(seedoss_tokenizer, streaming):
    """Test reasoning extraction with multiline content."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, MULTIPLE_LINES["output"])], streaming=streaming
    )

    assert reasoning == MULTIPLE_LINES["reasoning"]
    assert content == MULTIPLE_LINES["content"]


# [中文注释] 测试同时包含开始和结束标记的推理提取
@pytest.mark.parametrize("streaming", [True, False])
def test_with_start_token(seedoss_tokenizer, streaming):
    """Test reasoning extraction with both start and end tokens."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, WITH_START_TOKEN["output"])], streaming=streaming
    )

    assert reasoning == WITH_START_TOKEN["reasoning"]
    assert content == WITH_START_TOKEN["content"]


# [中文注释] 测试仅有结束标记的典型SeedOSS行为
@pytest.mark.parametrize("streaming", [True, False])
def test_only_end_token(seedoss_tokenizer, streaming):
    """
    Test reasoning extraction with only end token
    (SeedOSS typical behavior).
    """
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, ONLY_END_TOKEN["output"])], streaming=streaming
    )

    assert reasoning == ONLY_END_TOKEN["reasoning"]
    assert content == ONLY_END_TOKEN["content"]


# [中文注释] 测试完全没有推理标记的场景
@pytest.mark.parametrize("streaming", [True, False])
def test_no_tokens(seedoss_tokenizer, streaming):
    """Test when there are no reasoning tokens at all."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, [cast(str, NO_TOKENS["output"])], streaming=streaming
    )

    assert reasoning == NO_TOKENS["reasoning"]
    assert content == NO_TOKENS["content"]


# [中文注释] 测试is_reasoning_end方法的推理结束判断
def test_is_reasoning_end(seedoss_tokenizer):
    """Test the is_reasoning_end method."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    # Test with end token present
    end_token_id = parser.end_token_id
    assert parser.is_reasoning_end([1, 2, end_token_id, 4]) is True

    # Test without end token
    assert parser.is_reasoning_end([1, 2, 3, 4]) is False


# [中文注释] 测试extract_content_ids方法：从token ID列表中提取推理结束后的内容ID
def test_extract_content_ids(seedoss_tokenizer):
    """Test the extract_content_ids method."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    end_token_id = parser.end_token_id

    # Test with end token in the middle
    input_ids = [1, 2, end_token_id, 4, 5]
    content_ids = parser.extract_content_ids(input_ids)
    assert content_ids == [4, 5]

    # Test with end token at the end
    input_ids = [1, 2, 3, end_token_id]
    content_ids = parser.extract_content_ids(input_ids)
    assert content_ids == []

    # Test without end token
    input_ids = [1, 2, 3, 4]
    content_ids = parser.extract_content_ids(input_ids)
    assert content_ids == []


# [中文注释] 测试SeedOSS流式增量处理
def test_streaming_delta_processing(seedoss_tokenizer):
    """Test streaming processing with small deltas."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(seedoss_tokenizer)

    # Test streaming with incremental tokens
    deltas = ["Some ", "reasoning ", "content", "</seed:think>", "Final ", "answer"]

    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)

    assert reasoning == "Some reasoning content"
    assert content == "Final answer"
