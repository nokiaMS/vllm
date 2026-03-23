# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试 Anthropic 消息格式到 OpenAI 格式的转换：图像源处理、tool_result 内容解析、
# 以及计费头部（x-anthropic-billing-header）的剥离

"""Unit tests for Anthropic-to-OpenAI request conversion.

Tests the image source handling and tool_result content parsing in
AnthropicServingMessages._convert_anthropic_to_openai_request().
"""

from vllm.entrypoints.anthropic.protocol import (
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

_convert = AnthropicServingMessages._convert_anthropic_to_openai_request
_img_url = AnthropicServingMessages._convert_image_source_to_url


# 辅助函数：构建 Anthropic 消息请求对象
def _make_request(
    messages: list[dict],
    **kwargs,
) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=128,
        messages=messages,
        **kwargs,
    )


# ======================================================================
# _convert_image_source_to_url
# ======================================================================


# 测试图像源到 URL 的转换：base64、URL、缺省类型等场景
class TestConvertImageSourceToUrl:
    def test_base64_source(self):
        source = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "iVBORw0KGgo=",
        }
        assert _img_url(source) == "data:image/jpeg;base64,iVBORw0KGgo="

    def test_base64_png(self):
        source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "AAAA",
        }
        assert _img_url(source) == "data:image/png;base64,AAAA"

    def test_url_source(self):
        source = {
            "type": "url",
            "url": "https://example.com/image.jpg",
        }
        assert _img_url(source) == "https://example.com/image.jpg"

    def test_missing_type_defaults_to_base64(self):
        """When 'type' is absent, treat as base64."""
        source = {
            "media_type": "image/webp",
            "data": "UklGR",
        }
        assert _img_url(source) == "data:image/webp;base64,UklGR"

    def test_missing_media_type_defaults_to_jpeg(self):
        source = {"type": "base64", "data": "abc123"}
        assert _img_url(source) == "data:image/jpeg;base64,abc123"

    def test_url_source_missing_url_returns_empty(self):
        source = {"type": "url"}
        assert _img_url(source) == ""

    def test_empty_source_returns_data_uri_shell(self):
        source: dict = {}
        assert _img_url(source) == "data:image/jpeg;base64,"


# ======================================================================
# Image blocks inside user messages
# ======================================================================


# 测试用户消息中图像内容块的转换（base64 和 URL 图像源）
class TestImageContentBlocks:
    def test_base64_image_in_user_message(self):
        request = _make_request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                    ],
                }
            ]
        )

        result = _convert(request)
        user_msg = result.messages[0]
        assert user_msg["role"] == "user"

        parts = user_msg["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "Describe this image"}
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,iVBORw0KGgo="},
        }

    def test_url_image_in_user_message(self):
        request = _make_request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/cat.png",
                            },
                        },
                    ],
                }
            ]
        )

        result = _convert(request)
        parts = result.messages[0]["content"]
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        }


# ======================================================================
# tool_result content handling
# ======================================================================


# 测试 tool_result 内容处理：纯文本、图像、混合内容及 None 值
class TestToolResultContent:
    def _make_tool_result_request(
        self, tool_result_content
    ) -> AnthropicMessagesRequest:
        """Build a request with assistant tool_use followed by user
        tool_result."""
        return _make_request(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_001",
                            "name": "read_file",
                            "input": {"path": "/tmp/img.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_001",
                            "content": tool_result_content,
                        }
                    ],
                },
            ]
        )

    def test_tool_result_string_content(self):
        request = self._make_tool_result_request("file contents here")
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "file contents here"
        assert tool_msg[0]["tool_call_id"] == "call_001"

    def test_tool_result_text_blocks(self):
        request = self._make_tool_result_request(
            [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ]
        )
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "line 1\nline 2"

    def test_tool_result_with_image(self):
        """Image in tool_result should produce a follow-up user message."""
        request = self._make_tool_result_request(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    },
                }
            ]
        )
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == ""

        # The image should be injected as a follow-up user message
        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        img_parts = follow_up[0]["content"]
        assert len(img_parts) == 1
        assert img_parts[0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }

    def test_tool_result_with_text_and_image(self):
        """Mixed text+image tool_result: text in tool msg, image in user
        msg."""
        request = self._make_tool_result_request(
            [
                {"type": "text", "text": "Here is the screenshot"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "QUFB",
                    },
                },
            ]
        )
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "Here is the screenshot"

        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        assert follow_up[0]["content"][0]["image_url"]["url"] == (
            "data:image/jpeg;base64,QUFB"
        )

    def test_tool_result_with_multiple_images(self):
        request = self._make_tool_result_request(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "IMG1",
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/img2.jpg",
                    },
                },
            ]
        )
        result = _convert(request)

        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        urls = [p["image_url"]["url"] for p in follow_up[0]["content"]]
        assert urls == [
            "data:image/png;base64,IMG1",
            "https://example.com/img2.jpg",
        ]

    def test_tool_result_none_content(self):
        request = self._make_tool_result_request(None)
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == ""

    def test_tool_result_no_follow_up_when_no_images(self):
        """Ensure no extra user message is added when there are no images."""
        request = self._make_tool_result_request(
            [
                {"type": "text", "text": "just text"},
            ]
        )
        result = _convert(request)

        user_follow_ups = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(user_follow_ups) == 0


# ======================================================================
# Attribution header stripping
# ======================================================================


# 测试系统消息中 x-anthropic-billing-header 计费头部的剥离
class TestAttributionHeaderStripping:
    def test_billing_header_stripped_from_system(self):
        """Claude Code's x-anthropic-billing-header block should be
        stripped to preserve prefix caching."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: "
                    "cc_version=2.1.37.abc; cc_entrypoint=cli;",
                },
            ],
        )
        result = _convert(request)
        system_msg = result.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "You are a helpful assistant."

    def test_system_without_billing_header_unchanged(self):
        """Normal system blocks should pass through unchanged."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": " Be concise."},
            ],
        )
        result = _convert(request)
        system_msg = result.messages[0]
        assert system_msg["content"] == "You are a helpful assistant. Be concise."

    def test_system_string_unchanged(self):
        """String system prompts should pass through unchanged."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )
        result = _convert(request)
        system_msg = result.messages[0]
        assert system_msg["content"] == "You are a helpful assistant."
