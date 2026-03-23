# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .utils import ARGS, CONFIGS, ServerConfig


# [中文注释] 包级fixture：遍历Mistral模型配置，下载模型并跳过不支持ROCm的配置
# for each server config, download the model and return the config
@pytest.fixture(scope="package", params=CONFIGS.keys())
def server_config(request):
    config = CONFIGS[request.param]

    if current_platform.is_rocm() and not config.get("supports_rocm", True):
        pytest.skip(
            "The {} model can't be tested on the ROCm platform".format(config["model"])
        )

    # download model and tokenizer using transformers
    snapshot_download(config["model"])
    yield CONFIGS[request.param]


# [中文注释] 包级fixture：启动RemoteOpenAIServer用于Mistral模型测试
# run this for each server config
@pytest.fixture(scope="package")
def server(request, server_config: ServerConfig):
    model = server_config["model"]
    args_for_model = server_config["arguments"]
    with RemoteOpenAIServer(
        model, ARGS + args_for_model, max_wait_seconds=480
    ) as server:
        yield server


# [中文注释] 异步fixture：创建OpenAI异步客户端用于Mistral测试请求
@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    async with server.get_async_client() as async_client:
        yield async_client
