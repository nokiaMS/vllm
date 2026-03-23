# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试外部模型（Out-Of-Tree）注册机制，验证自定义模型的文本生成、嵌入和多模态功能

import pytest

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

from ..utils import create_new_process_for_each_test


@create_new_process_for_each_test()
# 测试未注册插件时模型加载应失败
def test_plugin(
    monkeypatch: pytest.MonkeyPatch,
    dummy_opt_path: str,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "")

        with pytest.raises(ValueError, match="are not supported for now"):
            LLM(model=dummy_opt_path, load_format="dummy")


@create_new_process_for_each_test()
# 测试注册自定义模型后的文本生成功能
def test_oot_registration_text_generation(
    monkeypatch: pytest.MonkeyPatch,
    dummy_opt_path: str,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "register_dummy_model")
        prompts = ["Hello, my name is", "The text does not matter"]
        sampling_params = SamplingParams(temperature=0)
        llm = LLM(model=dummy_opt_path, load_format="dummy")
        first_token = llm.get_tokenizer().decode(0)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            # make sure only the first token is generated
            rest = generated_text.replace(first_token, "")
            assert rest == ""


@create_new_process_for_each_test()
# 测试注册自定义模型后的嵌入功能
def test_oot_registration_embedding(
    monkeypatch: pytest.MonkeyPatch,
    dummy_gemma2_embedding_path: str,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "register_dummy_model")
        prompts = ["Hello, my name is", "The text does not matter"]
        llm = LLM(
            model=dummy_gemma2_embedding_path, load_format="dummy", max_model_len=2048
        )
        outputs = llm.embed(prompts)

        for output in outputs:
            assert all(v == 0 for v in output.outputs.embedding)


image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")


@create_new_process_for_each_test()
# 测试注册自定义模型后的多模态生成功能
def test_oot_registration_multimodal(
    monkeypatch: pytest.MonkeyPatch,
    dummy_llava_path: str,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "register_dummy_model")
        prompts = [
            {
                "prompt": "What's in the image?<image>",
                "multi_modal_data": {"image": image},
            },
            {
                "prompt": "Describe the image<image>",
                "multi_modal_data": {"image": image},
            },
        ]

        sampling_params = SamplingParams(temperature=0)
        llm = LLM(
            model=dummy_llava_path,
            load_format="dummy",
            max_num_seqs=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.98,
            max_model_len=4096,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )

        first_token = llm.get_tokenizer().decode(0)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            # make sure only the first token is generated
            rest = generated_text.replace(first_token, "")
            assert rest == ""
