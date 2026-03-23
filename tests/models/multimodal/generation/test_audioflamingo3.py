# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 测试 AudioFlamingo3 模型的音频理解生成能力，包括单条和批量音频推理

# Copyright 2025 The vLLM team.
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm import LLM, SamplingParams

MODEL_NAME = "nvidia/audio-flamingo-3-hf"


# 获取测试固定数据文件的路径
def get_fixture_path(filename):
    return os.path.join(
        os.path.dirname(__file__), "../../fixtures/audioflamingo3", filename
    )


@pytest.fixture(scope="module")
def llm():
    # Check if the model is supported by the current transformers version
    model_info = HF_EXAMPLE_MODELS.get_hf_info("AudioFlamingo3ForConditionalGeneration")
    model_info.check_transformers_version(on_fail="skip")

    try:
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1},
        )
        return llm
    except Exception as e:
        pytest.skip(f"Failed to load model {MODEL_NAME}: {e}")


# 测试单条音频输入的语音转文字生成结果
def test_single_generation(llm):
    fixture_path = get_fixture_path("expected_results_single.json")
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        expected = json.load(f)

    audio_url = "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/Why_do_we_ask_questions_converted.wav"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_url}},
                {"type": "text", "text": "Transcribe the input speech."},
            ],
        }
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
    )
    generated_text = outputs[0].outputs[0].text.strip()

    expected_text = expected["transcriptions"][0]

    assert expected_text in generated_text or generated_text in expected_text


# 测试批量音频输入的推理结果，验证多条音频同时处理的正确性
def test_batched_generation(llm):
    fixture_path = get_fixture_path("expected_results_batched.json")
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        expected = json.load(f)

    items = [
        {
            "audio_url": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
            "question": "What is surprising about the relationship "
            "between the barking and the music?",
            "expected_idx": 0,
        },
        {
            "audio_url": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/Ch6Ae9DT6Ko_00-04-03_00-04-31.wav",
            "question": (
                "Why is the philosopher's name mentioned in the lyrics? "
                "(A) To express a sense of nostalgia "
                "(B) To indicate that language cannot express clearly, "
                "satirizing the inversion of black and white in the world "
                "(C) To add depth and complexity to the lyrics "
                "(D) To showcase the wisdom and influence of the philosopher"
            ),
            "expected_idx": 1,
        },
    ]

    conversations = []
    for item in items:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": item["audio_url"]}},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        conversations.append(messages)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
    )

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        expected_text = expected["transcriptions"][i]

        assert expected_text in generated_text or generated_text in expected_text
