# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试输入预处理：验证多模态模型的输入预处理器始终使用多模态代码路径]

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.cpu_test


# [测试对空字符串和空 token 列表输入，预处理器仍能正确添加 sep token（多模态路径）]
@pytest.mark.parametrize("model_id", ["facebook/chameleon-7b"])
@pytest.mark.parametrize("prompt", ["", {"prompt_token_ids": []}])
@pytest.mark.skip(
    reason=(
        "Applying huggingface processor on text inputs results in "
        "significant performance regression for multimodal models. "
        "See https://github.com/vllm-project/vllm/issues/26320"
    )
)
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    vllm_config = VllmConfig(model_config=model_config)
    input_preprocessor = InputPreprocessor(vllm_config)

    # HF processor adds sep token
    tokenizer = input_preprocessor.get_tokenizer()
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]
