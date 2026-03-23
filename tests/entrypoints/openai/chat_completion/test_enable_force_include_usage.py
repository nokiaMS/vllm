# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试 enable_force_include_usage 参数强制在流式响应中包含 usage 信息

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="module")
# 启动带 enable_force_include_usage 参数的聊天服务器
def chat_server_with_force_include_usage(request):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--enforce-eager",
        "--max-num-seqs",
        "4",
        "--enable-force-include-usage",
        "--port",
        "55857",
        "--gpu-memory-utilization",
        "0.2",
    ]

    with RemoteOpenAIServer("Qwen/Qwen3-0.6B", args, auto_port=False) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def chat_client_with_force_include_usage(chat_server_with_force_include_usage):
    async with chat_server_with_force_include_usage.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
# 测试流式聊天中每个 chunk 都包含 usage 信息
async def test_chat_with_enable_force_include_usage(
    chat_client_with_force_include_usage: openai.AsyncOpenAI,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    stream = await chat_client_with_force_include_usage.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=10,
        extra_body=dict(min_tokens=10),
        temperature=0.0,
        stream=True,
    )
    last_completion_tokens = 0
    async for chunk in stream:
        if not len(chunk.choices):
            assert chunk.usage.prompt_tokens >= 0
            assert (
                last_completion_tokens == 0
                or chunk.usage.completion_tokens > last_completion_tokens
                or (
                    not chunk.choices
                    and chunk.usage.completion_tokens == last_completion_tokens
                )
            )
            assert chunk.usage.total_tokens == (
                chunk.usage.prompt_tokens + chunk.usage.completion_tokens
            )
        else:
            assert chunk.usage is None


@pytest.fixture(scope="module")
# 启动带 enable_force_include_usage 的转录服务器
def transcription_server_with_force_include_usage():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--enable-force-include-usage",
        "--gpu-memory-utilization",
        "0.2",
    ]

    with RemoteOpenAIServer("openai/whisper-large-v3-turbo", args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def transcription_client_with_force_include_usage(
    transcription_server_with_force_include_usage,
):
    async with (
        transcription_server_with_force_include_usage.get_async_client() as async_client
    ):
        yield async_client


@pytest.mark.asyncio
# 测试转录 API 流式响应中强制包含 usage 信息
async def test_transcription_with_enable_force_include_usage(
    transcription_client_with_force_include_usage, winning_call
):
    res = (
        await transcription_client_with_force_include_usage.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=winning_call,
            language="en",
            temperature=0.0,
            stream=True,
            timeout=30,
        )
    )

    async for chunk in res:
        if not len(chunk.choices):
            # final usage sent
            usage = chunk.usage
            assert isinstance(usage, dict)
            assert usage["prompt_tokens"] > 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] > 0
        else:
            assert not hasattr(chunk, "usage")
