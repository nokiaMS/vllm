# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 工具解析器测试的共享配置文件，提供测试所需的公共 fixture

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers import TokenizerLike


# 提供默认的 GPT-2 分词器 fixture，供各工具解析器测试复用
@pytest.fixture(scope="function")
def default_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("gpt2")
