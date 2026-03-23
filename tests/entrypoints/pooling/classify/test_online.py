# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [在线分类测试模块：通过 HTTP API 验证分类端点的各种请求格式、批量处理、聊天格式、截断及 invocations 端点]

import pytest
import requests
import torch
import torch.nn.functional as F

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.classify.protocol import ClassificationResponse
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"
DTYPE = "float32"  # Use float32 to avoid NaN issue
input_text = "This product was excellent and exceeded my expectations"
input_tokens = [1986, 1985, 572, 9073, 323, 33808, 847, 16665]


# [测试夹具：启动使用 Qwen2.5-1.5B-apeach 分类模型的远程服务器]
@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--dtype",
        DTYPE,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


# [测试基础功能：验证 /v1/models 和 /tokenize 端点的正确性]
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_basic(server: RemoteOpenAIServer, model_name: str):
    # test /v1/models
    response = requests.get(server.url_for("/v1/models"))
    served_model = response.json()["data"][0]["id"]
    assert served_model == MODEL_NAME

    # test /tokenize
    response = requests.post(
        server.url_for("/tokenize"),
        json={"model": model_name, "prompt": input_text},
    )
    assert response.json()["tokens"] == input_tokens


# [测试分类请求：验证字符串输入和 token 列表输入的分类结果格式]
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_completion_request(server: RemoteOpenAIServer, model_name: str):
    # test input: str
    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": input_text},
    )

    classification_response.raise_for_status()
    output = ClassificationResponse.model_validate(classification_response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")

    # test input: list[int]
    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": input_tokens},
    )

    classification_response.raise_for_status()
    output = ClassificationResponse.model_validate(classification_response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")


# [测试批量分类请求：验证多条字符串和 token 列表输入的批量处理]
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_completion_request_batched(server: RemoteOpenAIServer, model_name: str):
    N = 10

    # test input: list[str]
    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": [input_text] * N},
    )
    output = ClassificationResponse.model_validate(classification_response.json())

    assert len(output.data) == N
    for i, item in enumerate(output.data):
        assert item.index == i
        assert hasattr(item, "label")
        assert hasattr(item, "probs")
        assert len(item.probs) == item.num_classes
        assert item.label in ["Default", "Spoiled"]

    # test input: list[list[int]]
    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": [input_tokens] * N},
    )
    output = ClassificationResponse.model_validate(classification_response.json())

    assert len(output.data) == N
    for i, item in enumerate(output.data):
        assert item.index == i
        assert hasattr(item, "label")
        assert hasattr(item, "probs")
        assert len(item.probs) == item.num_classes
        assert item.label in ["Default", "Spoiled"]


# [测试空输入错误：验证空字符串和空列表输入返回 400 错误]
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_empty_input_error(server: RemoteOpenAIServer, model_name: str):
    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": ""},
    )

    error = classification_response.json()
    assert classification_response.status_code == 400
    assert "error" in error

    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": []},
    )

    error = classification_response.json()
    assert classification_response.status_code == 400
    assert "error" in error


# [测试提示截断参数：验证有效截断值和超出限制截断值的行为]
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_truncate_prompt_tokens(server: RemoteOpenAIServer, model_name: str):
    long_text = "hello " * 600

    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": long_text, "truncate_prompt_tokens": 5},
    )

    classification_response.raise_for_status()
    output = ClassificationResponse.model_validate(classification_response.json())

    assert len(output.data) == 1
    assert output.data[0].index == 0
    assert hasattr(output.data[0], "probs")
    assert output.usage.prompt_tokens == 5
    assert output.usage.total_tokens == 5

    # invalid_truncate_prompt_tokens
    classification_response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": "test", "truncate_prompt_tokens": 513},
    )

    error = classification_response.json()
    assert classification_response.status_code == 400
    assert "truncate_prompt_tokens" in error["error"]["message"]


# [测试特殊 token 参数：验证 add_special_tokens 为 True/False 时请求均能成功]
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_add_special_tokens(server: RemoteOpenAIServer, model_name: str):
    # The add_special_tokens parameter doesn't seem to be working with this model.
    # working with papluca/xlm-roberta-base-language-detection
    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": input_text, "add_special_tokens": False},
    )
    response.raise_for_status()
    ClassificationResponse.model_validate(response.json())

    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "input": input_text, "add_special_tokens": True},
    )
    response.raise_for_status()
    ClassificationResponse.model_validate(response.json())


# [测试聊天格式请求：验证多轮对话输入的分类、add_generation_prompt、continue_final_message 等参数]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chat_request(server: RemoteOpenAIServer, model_name: str):
    messages = [
        {
            "role": "user",
            "content": "The cat sat on the mat.",
        },
        {
            "role": "assistant",
            "content": "A feline was resting on a rug.",
        },
        {
            "role": "user",
            "content": "Stars twinkle brightly in the night sky.",
        },
    ]

    # test chat request basic usage
    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages},
    )

    response.raise_for_status()
    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")
    assert output.usage.prompt_tokens == 51

    # test add_generation_prompt
    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages, "add_generation_prompt": True},
    )

    response.raise_for_status()
    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")
    assert output.usage.prompt_tokens == 54

    # test continue_final_message
    response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "messages": messages,
            "continue_final_message": True,
        },
    )

    response.raise_for_status()
    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")
    assert output.usage.prompt_tokens == 49

    # test add_special_tokens
    # The add_special_tokens parameter doesn't seem to be working with this model.
    response = requests.post(
        server.url_for("classify"),
        json={"model": model_name, "messages": messages, "add_special_tokens": True},
    )

    response.raise_for_status()
    output = ClassificationResponse.model_validate(response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")
    assert output.usage.prompt_tokens == 51

    # test continue_final_message with add_generation_prompt
    response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "messages": messages,
            "continue_final_message": True,
            "add_generation_prompt": True,
        },
    )
    assert (
        "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
        in response.json()["error"]["message"]
    )


# [测试 invocations 端点（补全请求）：验证 invocations 与 classify 端点结果一致]
@pytest.mark.asyncio
async def test_invocations_completion_request(server: RemoteOpenAIServer):
    request_args = {
        "model": MODEL_NAME,
        "input": input_text,
    }

    classification_response = requests.post(
        server.url_for("classify"), json=request_args
    )
    classification_response.raise_for_status()

    invocation_response = requests.post(
        server.url_for("invocations"), json=request_args
    )
    invocation_response.raise_for_status()

    classification_output = classification_response.json()
    invocation_output = invocation_response.json()

    assert classification_output.keys() == invocation_output.keys()
    for classification_data, invocation_data in zip(
        classification_output["data"], invocation_output["data"]
    ):
        assert classification_data.keys() == invocation_data.keys()
        assert classification_data["probs"] == pytest.approx(
            invocation_data["probs"], rel=0.01
        )


# [测试 invocations 端点（聊天请求）：验证聊天格式下 invocations 与 classify 端点结果一致]
@pytest.mark.asyncio
async def test_invocations_chat_request(server: RemoteOpenAIServer):
    messages = [
        {
            "role": "user",
            "content": "The cat sat on the mat.",
        },
        {
            "role": "assistant",
            "content": "A feline was resting on a rug.",
        },
        {
            "role": "user",
            "content": "Stars twinkle brightly in the night sky.",
        },
    ]

    request_args = {"model": MODEL_NAME, "messages": messages}

    classification_response = requests.post(
        server.url_for("classify"), json=request_args
    )
    classification_response.raise_for_status()

    invocation_response = requests.post(
        server.url_for("invocations"), json=request_args
    )
    invocation_response.raise_for_status()

    classification_output = classification_response.json()
    invocation_output = invocation_response.json()

    assert classification_output.keys() == invocation_output.keys()
    for classification_data, invocation_data in zip(
        classification_output["data"], invocation_output["data"]
    ):
        assert classification_data.keys() == invocation_data.keys()
        assert classification_data["probs"] == pytest.approx(
            invocation_data["probs"], rel=0.01
        )


# [测试激活函数控制：验证在线 API 中 use_activation 参数对分类概率的影响]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_use_activation(server: RemoteOpenAIServer, model_name: str):
    async def get_outputs(use_activation):
        response = requests.post(
            server.url_for("classify"),
            json={
                "model": model_name,
                "input": input_text,
                "use_activation": use_activation,
            },
        )
        outputs = response.json()
        return torch.tensor([x["probs"] for x in outputs["data"]])

    default = await get_outputs(use_activation=None)
    w_activation = await get_outputs(use_activation=True)
    wo_activation = await get_outputs(use_activation=False)

    assert torch.allclose(default, w_activation, atol=1e-2), (
        "Default should use activation."
    )
    assert not torch.allclose(w_activation, wo_activation, atol=1e-2), (
        "wo_activation should not use activation."
    )
    assert torch.allclose(F.softmax(wo_activation, dim=-1), w_activation, atol=1e-2), (
        "w_activation should be close to activation(wo_activation)."
    )


# [测试 score 端点不可用：验证 num_labels != 1 的分类模型不注册 score 端点]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_score(server: RemoteOpenAIServer, model_name: str):
    # score api is only enabled for num_labels == 1.
    response = requests.post(
        server.url_for("score"),
        json={
            "model": model_name,
            "queries": "ping",
            "documents": "pong",
        },
    )
    assert response.json()["detail"] == "Not Found"


# [测试 rerank 端点不可用：验证 num_labels != 1 的分类模型不注册 rerank 端点]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_rerank(server: RemoteOpenAIServer, model_name: str):
    # rerank api is only enabled for num_labels == 1.
    response = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": "ping",
            "documents": ["pong"],
        },
    )
    assert response.json()["detail"] == "Not Found"


# [测试通用 pooling 端点的 classify 任务：验证通过 pooling 端点执行分类]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_classify(server: RemoteOpenAIServer, model_name: str):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": "classify",
        },
    )
    poolings = PoolingResponse.model_validate(response.json())
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 2


# [测试通用 pooling 端点的 token_classify 任务：验证 token 级别分类的输出维度]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_token_classify(server: RemoteOpenAIServer, model_name: str):
    task = "token_classify"
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
        },
    )
    poolings = PoolingResponse.model_validate(response.json())
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 8
    assert len(poolings.data[0].data[0]) == 2


# [测试 pooling 端点不支持的任务：验证 embed、token_embed、plugin 任务返回错误]
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("task", ["embed", "token_embed", "plugin"])
async def test_pooling_not_supported(
    server: RemoteOpenAIServer, model_name: str, task: str
):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["message"].startswith(f"Unsupported task: {task!r}")
