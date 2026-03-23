# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [OpenAI 入口测试的共享 conftest，提供音频相关 fixture 和注意力后端辅助函数]
import pytest

from vllm.assets.audio import AudioAsset


# [如果指定了注意力后端配置，则追加到服务器参数中]
def add_attention_backend(server_args, attention_config):
    """Append attention backend CLI arg if specified.

    Args:
        server_args: List of server arguments to extend in-place.
        attention_config: Dict with 'backend' key, or None.
    """
    if attention_config and "backend" in attention_config:
        server_args.extend(["--attention-backend", attention_config["backend"]])


# [为 ROCm 平台上的语音测试返回所需的注意力后端配置]
@pytest.fixture(scope="module")
def rocm_aiter_fa_attention():
    """Return attention config for transcription/translation tests on ROCm.

    On ROCm, audio tests require ROCM_AITER_FA attention backend.
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        return {"backend": "ROCM_AITER_FA"}
    return None


# [提供 "Mary Had a Lamb" 音频文件的 fixture]
@pytest.fixture
def mary_had_lamb():
    path = AudioAsset("mary_had_lamb").get_local_path()
    with open(str(path), "rb") as f:
        yield f


# [提供 "winning call" 音频文件的 fixture]
@pytest.fixture
def winning_call():
    path = AudioAsset("winning_call").get_local_path()
    with open(str(path), "rb") as f:
        yield f


# [提供意大利语音频文件 fixture，用于翻译测试]
@pytest.fixture
def foscolo():
    # Test translation it->en
    path = AudioAsset("azacinto_foscolo").get_local_path()
    with open(str(path), "rb") as f:
        yield f
