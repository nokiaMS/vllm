# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import requests

# Test configuration for BGE-M3 sparse plugin
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse

model_config = {
    "model_name": "BAAI/bge-m3",
    "plugin": "bge_m3_sparse_plugin",
    "test_input": "What is the capital of France?",
    "hf_overrides": json.dumps(
        {"architectures": ["BgeM3EmbeddingModel"], "head_dtype": "float16"}
    ),
}


# [中文注释] 辅助函数：比较两个浮点数是否在误差范围内相近
def _float_close(expected: object, result: object):
    assert isinstance(expected, float) and isinstance(result, float), (
        f"{expected=}  or {result=} is not float"
    )
    return (expected - result) < 1e-3 or abs(expected / result - 1) < 1e-3


# [中文注释] 辅助函数：从字典或对象中获取指定键/属性值
def _get_attr_or_val(obj: object | dict, key: str):
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return getattr(obj, key, None)


# [中文注释] 验证稀疏嵌入输出的token_id、weight和token与预期值一致
def _check_sparse_embedding(data, check_tokens=False):
    expected_weights = [
        {"token_id": 32, "weight": 0.0552978515625, "token": "?"},
        {"token_id": 70, "weight": 0.09808349609375, "token": "the"},
        {"token_id": 83, "weight": 0.08154296875, "token": "is"},
        {"token_id": 111, "weight": 0.11810302734375, "token": "of"},
        {"token_id": 4865, "weight": 0.1171875, "token": "What"},
        {"token_id": 9942, "weight": 0.292236328125, "token": "France"},
        {"token_id": 10323, "weight": 0.2802734375, "token": "capital"},
    ]
    expected_embed = {x["token_id"]: x for x in expected_weights}

    assert len(data) == len(expected_embed)
    for entry in data:
        expected_val = expected_embed[_get_attr_or_val(entry, "token_id")]
        assert _float_close(
            expected_val["weight"], _get_attr_or_val(entry, "weight")
        ), f"actual embed {entry} not equal to {expected_val}"
        if check_tokens:
            assert expected_val["token"] == _get_attr_or_val(entry, "token"), (
                f"actual embed {entry} not equal to {expected_val}"
            )
        else:
            assert _get_attr_or_val(entry, "token") is None, (
                f"{entry} should not return token"
            )


# [中文注释] fixture：启动带BGE-M3稀疏插件的远程OpenAI服务器
@pytest.fixture(scope="function")
def server():
    args = [
        "--runner",
        "pooling",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--hf_overrides",
        model_config["hf_overrides"],
        "--io-processor-plugin",
        model_config["plugin"],
    ]

    with RemoteOpenAIServer(model_config["model_name"], args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "return_tokens",
    [True, False],
)
# [中文注释] 在线测试BGE-M3稀疏插件：通过API验证稀疏嵌入输出格式
async def test_bge_m3_sparse_plugin_online(
    server: RemoteOpenAIServer, return_tokens: bool
):
    """Test BGE-M3 sparse plugin in online mode via API."""
    request_payload = {
        "model": model_config["model_name"],
        "task": "token_classify",
        "data": {"input": model_config["test_input"], "return_tokens": return_tokens},
    }

    ret = requests.post(
        server.url_for("pooling"),
        json=request_payload,
    )

    response = ret.json()

    # Verify the request response is in the correct format
    assert (parsed_response := IOProcessorResponse(**response).data)

    # Verify the output is formatted as expected for this plugin
    assert _get_attr_or_val(parsed_response, "data")
    assert len(_get_attr_or_val(parsed_response, "data")) > 0

    data_entry = _get_attr_or_val(parsed_response, "data")[0]
    assert _get_attr_or_val(data_entry, "object") == "sparse-embedding"
    assert _get_attr_or_val(data_entry, "sparse_embedding")

    # Verify sparse embedding format
    sparse_embedding = _get_attr_or_val(data_entry, "sparse_embedding")
    assert isinstance(sparse_embedding, list)
    _check_sparse_embedding(sparse_embedding, return_tokens)

    # Verify usage information
    usage = _get_attr_or_val(parsed_response, "usage")
    assert usage, f"usage not found for {parsed_response}"
    assert _get_attr_or_val(usage, "prompt_tokens") > 0
    assert _get_attr_or_val(usage, "total_tokens") == _get_attr_or_val(
        usage, "prompt_tokens"
    )


@pytest.mark.parametrize(
    "return_tokens",
    [True, False],
)
# [中文注释] 离线测试BGE-M3稀疏插件：验证单条输入的稀疏嵌入正确性
def test_bge_m3_sparse_plugin_offline(vllm_runner, return_tokens: bool):
    """Test BGE-M3 sparse plugin in offline mode."""
    prompt = {
        "data": {
            "input": model_config["test_input"],
            "return_tokens": return_tokens,
        }
    }

    with vllm_runner(
        model_config["model_name"],
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=32,
        io_processor_plugin=model_config["plugin"],
        hf_overrides=json.loads(model_config["hf_overrides"]),
        default_torch_num_threads=1,
    ) as llm_runner:
        llm = llm_runner.get_llm()
        pooler_output = llm.encode(prompt, pooling_task="token_classify")

    outputs = pooler_output[0]

    # Verify output structure
    assert hasattr(outputs, "outputs")
    response = outputs.outputs
    assert hasattr(response, "data")
    assert len(response.data) == 1
    # Verify response data
    for i, output in enumerate(response.data):
        # Each output should have sparse embeddings
        sparse_embedding = output.sparse_embedding
        assert isinstance(sparse_embedding, list)
        _check_sparse_embedding(sparse_embedding, return_tokens)

    # Verify usage
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens


# [中文注释] 离线测试BGE-M3稀疏插件：验证多条输入批量处理
def test_bge_m3_sparse_plugin_offline_multiple_inputs(vllm_runner):
    """Test BGE-M3 sparse plugin with multiple inputs in offline mode."""
    prompts = {
        "data": {
            "input": [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of Spain?",
            ],
            "return_tokens": True,
        }
    }

    with vllm_runner(
        model_config["model_name"],
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=32,
        io_processor_plugin=model_config["plugin"],
        hf_overrides=json.loads(model_config["hf_overrides"]),
        default_torch_num_threads=1,
    ) as llm_runner:
        llm = llm_runner.get_llm()
        pooler_output = llm.encode(prompts, pooling_task="token_classify")

    outputs = pooler_output[0]

    # Verify output structure
    assert hasattr(outputs, "outputs")
    response = outputs.outputs
    assert hasattr(response, "data")
    assert len(response.data) == 3
    for i, output in enumerate(response.data):
        # Each output should have sparse embeddings
        sparse_embedding = output.sparse_embedding
        assert isinstance(sparse_embedding, list)

    # Verify usage
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens
