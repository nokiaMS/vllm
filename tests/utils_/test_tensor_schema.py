# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.glm4_1v import Glm4vImageEmbeddingInputs
from vllm.model_executor.models.granite_speech import GraniteSpeechAudioInputs
from vllm.model_executor.models.hyperclovax_vision import HCXVisionVideoPixelInputs
from vllm.model_executor.models.phi3v import Phi3VImagePixelInputs


# [中文注释] 测试有效张量的TensorSchema验证通过
def test_tensor_schema_valid_tensor():
    Phi3VImagePixelInputs(
        pixel_values=torch.randn(16, 64, 3, 32, 32),
        image_sizes=torch.randint(0, 256, (16, 2)),
    )


# [中文注释] 测试可选字段为None或缺失时验证通过
def test_tensor_schema_optional_fields():
    Phi3VImagePixelInputs(
        pixel_values=torch.randn(16, 64, 3, 32, 32),
        image_sizes=None,
    )

    Phi3VImagePixelInputs(pixel_values=torch.randn(16, 64, 3, 32, 32))


# [中文注释] 测试常量维度不匹配时的验证失败
def test_tensor_schema_constant_dim_failure():
    with pytest.raises(ValueError, match="dim\\[2\\] expected 3, got 4"):
        Phi3VImagePixelInputs(
            pixel_values=torch.randn(16, 64, 4, 32, 32),  # dim[2] = 4
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


# [中文注释] 测试列表中包含非张量类型时的类型错误
def test_tensor_schema_invalid_types_in_list():
    with pytest.raises(TypeError, match="is not one of the expected types"):
        Phi3VImagePixelInputs(
            pixel_values=[
                torch.randn(64, 3, 32, 32),
                "not_a_tensor",
                torch.randn(64, 3, 32, 32),
            ],
            image_sizes=torch.randint(0, 256, (3, 2)),
        )


# [中文注释] 测试张量秩不匹配时的验证失败
def test_tensor_schema_rank_mismatch():
    with pytest.raises(ValueError, match="has rank 3 but expected 5"):
        Phi3VImagePixelInputs(
            pixel_values=torch.randn(16, 64, 3),
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


# [中文注释] 测试缺少必需字段时的验证失败
def test_tensor_schema_missing_required_field():
    with pytest.raises(ValueError, match="Required field 'pixel_values' is missing"):
        Phi3VImagePixelInputs(
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


# [中文注释] 测试符号维度跨字段不一致时的验证失败
def test_tensor_schema_symbolic_dim_mismatch():
    with pytest.raises(ValueError, match="expected 'bn'=12, got 16"):
        Phi3VImagePixelInputs(
            pixel_values=torch.randn(12, 64, 3, 32, 32),
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


# [中文注释] 测试张量列表输入的有效验证
def test_tensor_schema_list_tensor_valid():
    Phi3VImagePixelInputs(
        pixel_values=[torch.randn(64, 3, 32, 32) for _ in range(16)],
        image_sizes=torch.randint(0, 256, (16, 2)),
    )


# [中文注释] 测试不同patch数量的张量列表验证通过
def test_tensor_schema_variable_patch_counts_valid():
    # Each image has a different number of patches (p)
    # Each tensor has shape (p, 3, 32, 32)
    Phi3VImagePixelInputs(
        pixel_values=[
            torch.randn(16, 3, 32, 32),  # p = 16
            torch.randn(32, 3, 32, 32),  # p = 32
            torch.randn(64, 3, 32, 32),  # p = 64
        ],
        image_sizes=torch.randint(0, 256, (3, 2)),  # bn = 3
    )


# [中文注释] 测试元组形式的张量集合验证通过
def test_tensor_schema_tuple_tensor_valid():
    Phi3VImagePixelInputs(
        pixel_values=tuple(torch.randn(64, 3, 32, 32) for _ in range(16)),
        image_sizes=torch.randint(0, 256, (16, 2)),
    )


# [中文注释] 测试双重嵌套张量列表（视频帧）的验证
def test_tensor_schema_double_nested_tensors():
    x = torch.rand(4, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)

    HCXVisionVideoPixelInputs(pixel_values_videos=([x, y, x], [y], [x, y]))


# [中文注释] 测试列表中张量形状不一致时的验证失败
def test_tensor_schema_inconsistent_shapes_in_list():
    with pytest.raises(ValueError, match="contains inconsistent shapes"):
        Phi3VImagePixelInputs(
            pixel_values=[
                torch.randn(64, 3, 32, 32),
                torch.randn(64, 3, 16, 16),
                *(torch.randn(64, 3, 32, 32) for _ in range(14)),
            ],
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


# [中文注释] 测试空列表输入时的验证失败
def test_tensor_schema_empty_list():
    with pytest.raises(ValueError, match="is an empty sequence"):
        Phi3VImagePixelInputs(
            pixel_values=[],
            image_sizes=torch.randint(0, 256, (0, 2)),
        )


# [中文注释] 测试禁用验证时跳过形状检查
def test_tensor_schema_validation_disabled_skips_shape_check():
    # This should NOT raise, because validation is turned off
    # This would normally fail (dim[2] should be 3, not 4)
    Phi3VImagePixelInputs(
        pixel_values=torch.randn(16, 64, 4, 32, 32),
        image_sizes=torch.randint(0, 256, (16, 2)),
        validate=False,
    )


# [中文注释] 测试resolve_bindings参数正确绑定符号维度
def test_tensor_schema_with_valid_resolve_binding_dims():
    pixel_values = torch.randn(16, 64, 3, 336, 336)  # h=336, w=336
    image_sizes = torch.randint(0, 256, (16, 2))

    Phi3VImagePixelInputs(
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        resolve_bindings={"h": 336, "w": 336},
    )


# [中文注释] 测试resolve_bindings维度不匹配时的验证失败
def test_tensor_schema_with_invalid_resolve_binding_dims():
    pixel_values = torch.randn(16, 64, 3, 36, 36)  # h=36, w=36
    image_sizes = torch.randint(0, 256, (16, 2))

    # Should raise because 'h' and 'w' don't match resolve bindings
    with pytest.raises(ValueError, match="dim\\[3\\] expected 336, got 36"):
        Phi3VImagePixelInputs(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            resolve_bindings={"h": 336, "w": 336},
        )


# [中文注释] 测试列表长度作为符号维度绑定（GraniteSpeech音频输入）
def test_tensor_schema_with_list_of_symbolic_dim():
    input_features = torch.randn(3, 10, 160)  # (b=3, fi=10, 160)
    input_features_mask = torch.randn(3, 8)  # (b=3, fo=8)
    audio_embed_sizes = [8, 8, 8]  # len = b = 3

    GraniteSpeechAudioInputs(
        input_features=input_features,
        input_features_mask=input_features_mask,
        audio_embed_sizes=audio_embed_sizes,
    )


# [中文注释] 测试列表长度与符号维度不匹配时的验证失败
def test_tensor_schema_with_list_of_symbolic_dim_mismatch_in_length():
    input_features = torch.randn(4, 10, 160)  # (b=4, fi=10, 160)
    input_features_mask = torch.randn(4, 8)  # (b=4, fo=8)
    audio_embed_sizes = [8, 8, 8]  # len = 3 ≠ b

    with pytest.raises(ValueError, match="expected 'b'=4, got 3"):
        GraniteSpeechAudioInputs(
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_embed_sizes=audio_embed_sizes,
        )


# [中文注释] 测试静态最后维度的有效验证（Glm4v图像嵌入输入）
def test_valid_tensor_schema_with_static_last_dim():
    image_embeds = torch.randn(256, 1024)
    image_grid_thw = torch.randint(0, 4, (2, 3))

    Glm4vImageEmbeddingInputs(
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
    )


# [中文注释] 测试静态最后维度不匹配时的验证失败
def test_invalid_tensor_schema_with_static_last_dim():
    image_embeds = torch.randn(256, 1024)
    image_grid_thw = torch.randint(0, 4, (2, 4))  # Wrong last dim

    with pytest.raises(ValueError, match="dim\\[1\\] expected 3, got 4"):
        Glm4vImageEmbeddingInputs(
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )
