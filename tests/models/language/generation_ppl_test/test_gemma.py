# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试Gemma系列模型（gemma-2b/gemma-2-2b/gemma-3-4b-it）的困惑度

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import wikitext_ppl_test

MODELS = [
    GenerateModelInfo("google/gemma-2b", hf_ppl=21.48524284362793),
    GenerateModelInfo("google/gemma-2-2b", hf_ppl=102.59290313720703),
    GenerateModelInfo("google/gemma-3-4b-it", hf_ppl=27.79648208618164),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    wikitext_ppl_test(hf_runner, vllm_runner, model_info)
