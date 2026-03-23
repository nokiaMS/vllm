# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run `pytest tests/entrypoints/openai/test_embedding_dimensions.py`.
"""

# [嵌入维度测试模块：验证 Matryoshka 嵌入维度参数对嵌入输出的控制，包括有效和无效维度值]

import openai
import pytest

from tests.conftest import HfRunner
from tests.models.language.pooling.embed_utils import run_embedding_correctness_test
from tests.models.utils import EmbedModelInfo
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse
from vllm.platforms import current_platform

MODELS = [
    EmbedModelInfo("intfloat/multilingual-e5-small", is_matryoshka=False),
    EmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        is_matryoshka=True,
        matryoshka_dimensions=[256],
    ),
]

input_texts = [
    "The chef prepared a delicious meal.",
]


# [模型信息夹具：参数化两个模型（普通嵌入模型和 Matryoshka 嵌入模型）]
@pytest.fixture(scope="module", params=MODELS)
def model_info(request):
    return request.param


# [数据类型夹具：使用 bfloat16 精度]
@pytest.fixture(scope="module", params=["bfloat16"])
def dtype(request):
    return request.param


# [服务器夹具：根据模型类型启动远程服务器，对 Matryoshka 模型启用维度限制]
@pytest.fixture(scope="module")
def server(model_info, dtype: str):
    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        dtype,
        "--enforce-eager",
        "--max-model-len",
        "512",
    ]

    if model_info.name == "Snowflake/snowflake-arctic-embed-m-v1.5":
        # Manually enable Matryoshka Embeddings
        args.extend(
            ["--trust_remote_code", "--hf_overrides", '{"matryoshka_dimensions":[256]}']
        )

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(model_info.name, args) as remote_server:
        yield remote_server


# [HuggingFace 参考模型夹具：用于嵌入正确性对比]
@pytest.fixture(scope="module")
def hf_model(hf_runner, model_info, dtype: str):
    with hf_runner(
        model_info.name, dtype=dtype, is_sentence_transformer=True
    ) as hf_model:
        yield hf_model


# [测试 Matryoshka 维度控制：验证有效和无效维度参数的嵌入输出，并与 HF 模型对比正确性]
@pytest.mark.asyncio
async def test_matryoshka(
    model_info: EmbedModelInfo, server: RemoteOpenAIServer, hf_model: HfRunner
):
    client = server.get_async_client()

    async def make_request_and_correctness_test(dimensions):
        prompts = input_texts * 3

        embedding_response = await client.embeddings.create(
            model=model_info.name,
            input=prompts,
            dimensions=dimensions,
            encoding_format="float",
        )
        embeddings = EmbeddingResponse.model_validate(
            embedding_response.model_dump(mode="json")
        )

        assert embeddings.id is not None
        assert len(embeddings.data) == 3
        assert len(embeddings.data[0].embedding) > 0
        assert embeddings.usage.completion_tokens == 0
        assert embeddings.usage.prompt_tokens > 0
        assert embeddings.usage.total_tokens > 0

        if dimensions is not None:
            assert len(embeddings.data[0].embedding) == dimensions

        vllm_outputs = [d.embedding for d in embeddings.data]
        run_embedding_correctness_test(hf_model, prompts, vllm_outputs, dimensions)

    if model_info.is_matryoshka:
        valid_dimensions: list[int | None] = [None]
        if model_info.matryoshka_dimensions is not None:
            valid_dimensions += model_info.matryoshka_dimensions[:2]

        for dimensions in valid_dimensions:
            await make_request_and_correctness_test(dimensions)

        invalid_dimensions: list[int | None] = [-1]
        if model_info.matryoshka_dimensions is not None:
            assert 5 not in model_info.matryoshka_dimensions
            invalid_dimensions.append(5)

        for dimensions in invalid_dimensions:
            with pytest.raises(openai.BadRequestError):
                await make_request_and_correctness_test(dimensions)

    else:
        for dimensions in [None]:
            await make_request_and_correctness_test(dimensions)

        for dimensions in [-1, 16]:
            with pytest.raises(openai.BadRequestError):
                await make_request_and_correctness_test(dimensions)
