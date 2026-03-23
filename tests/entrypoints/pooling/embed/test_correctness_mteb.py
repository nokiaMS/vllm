# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [MTEB 嵌入正确性测试模块：通过 MTEB 基准评估 vLLM 嵌入结果与 SentenceTransformers 的一致性]

import os

import pytest

from tests.models.language.pooling_mteb_test.mteb_embed_utils import (
    MTEB_EMBED_TASKS,
    MTEB_EMBED_TOL,
    OpenAIClientMtebEncoder,
    run_mteb_embed_task,
)
from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "intfloat/e5-small"
MAIN_SCORE = 0.7422994752439667


# [测试夹具：启动 e5-small 嵌入模型的远程服务器]
@pytest.fixture(scope="module")
def server():
    args = ["--runner", "pooling", "--enforce-eager", "--disable-uvicorn-access-log"]

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


# [测试 MTEB 嵌入评分：比较 vLLM 与 SentenceTransformers 的 MTEB 主分数差异]
def test_mteb_embed(server):
    client = server.get_client()
    encoder = OpenAIClientMtebEncoder(MODEL_NAME, client)
    vllm_main_score = run_mteb_embed_task(encoder, MTEB_EMBED_TASKS)
    st_main_score = MAIN_SCORE

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    # We are not concerned that the vllm mteb results are better
    # than SentenceTransformers, so we only perform one-sided testing.
    assert st_main_score - vllm_main_score < MTEB_EMBED_TOL
