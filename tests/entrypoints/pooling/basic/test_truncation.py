# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [在线截断测试模块：通过 OpenAI 兼容 API 验证 embeddings 端点的 truncate_prompt_tokens 参数行为]

from typing import Any

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
max_model_len = 128

input = """Immerse yourself in the enchanting chronicle of calculus, a 
    mathematical domain that has radically transformed our comprehension of 
    change and motion. Despite its roots in ancient civilizations, the 
    formal birth of calculus predominantly occurred in the 17th century, 
    primarily under the influential guidance of Sir Isaac Newton and Gottfried 
    Wilhelm Leibniz. The earliest traces of calculus concepts are found in 
    ancient Greek mathematics,most notably in the works of Eudoxus and 
    Archimedes, around 300 BCE. They utilized the 'method of exhaustion'—a 
    technique for computing areas and volumes through the use of finite sums. 
    This methodology laid crucial foundational work for integral calculus. 
    In the 17th century, both Newton and Leibniz independently pioneered 
    calculus, each contributing unique perspectives that would shape this new 
    field."""


# [测试夹具：启动远程 OpenAI 兼容服务器，使用 all-MiniLM-L12-v2 模型]
@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        str(max_model_len),
    ]

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


# [异步客户端夹具：从远程服务器获取异步 OpenAI 客户端]
@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


# [测试较小截断尺寸：验证截断后 prompt_tokens 等于指定的截断值]
@pytest.mark.asyncio
async def test_smaller_truncation_size(client: openai.AsyncOpenAI):
    truncation_size = 10
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "input": input,
        "truncate_prompt_tokens": truncation_size,
    }

    response = await client.post(path="embeddings", cast_to=object, body={**kwargs})

    assert response["usage"]["prompt_tokens"] == truncation_size


# [测试过大截断尺寸：验证截断值超过 max_model_len 时返回 400 错误]
@pytest.mark.asyncio
async def test_bigger_truncation_size(client: openai.AsyncOpenAI):
    truncation_size = max_model_len + 1
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "input": input,
        "truncate_prompt_tokens": truncation_size,
    }

    with pytest.raises(openai.BadRequestError) as err:
        await client.post(path="embeddings", cast_to=object, body={**kwargs})

    assert err.value.status_code == 400
    error_details = err.value.response.json()["error"]
    assert error_details["type"] == "BadRequestError"
    expected_message = (
        "truncate_prompt_tokens value is "
        "greater than max_model_len."
        " Please, select a smaller truncation size."
    )
    assert error_details["message"] == expected_message


# [测试最大截断尺寸：验证 truncation_size=-1 时截断到 max_model_len]
@pytest.mark.asyncio
async def test_max_truncation_size(client: openai.AsyncOpenAI):
    truncation_size = -1
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "input": input,
        "truncate_prompt_tokens": truncation_size,
    }

    response = await client.post(path="embeddings", cast_to=object, body={**kwargs})

    assert response["usage"]["prompt_tokens"] == max_model_len
