# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
from typing import TYPE_CHECKING, Any, Literal, TypeAlias  # 导入类型检查相关工具

import torch  # 导入PyTorch深度学习框架
from typing_extensions import NotRequired, TypedDict, assert_never  # 导入类型扩展工具

if TYPE_CHECKING:  # 仅在类型检查时导入以避免循环依赖
    from vllm.multimodal.inputs import (  # 从多模态输入模块导入类型
        MultiModalDataDict,  # 多模态数据字典类型
        MultiModalEncDecInputs,  # 多模态编码器-解码器输入类型
        MultiModalInputs,  # 多模态输入类型
        MultiModalUUIDDict,  # 多模态UUID字典类型
    )
else:  # 非类型检查时使用占位对象
    MultiModalDataDict = object  # 多模态数据字典占位
    MultiModalEncDecInputs = object  # 多模态编码器-解码器输入占位
    MultiModalInputs = object  # 多模态输入占位
    MultiModalUUIDDict = object  # 多模态UUID字典占位


# Inputs to LLM API  # LLM API 的输入定义
class _PromptOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt].
    """
    # 所有单例提示可用的附加选项

    multi_modal_data: NotRequired[MultiModalDataDict | None]  # 可选的多模态数据
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]  # 可选的多模态处理器参数
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

    multi_modal_uuids: NotRequired[MultiModalUUIDDict]  # 可选的多模态项UUID
    """
    Optional user-specified UUIDs for multimodal items, mapped by modality.
    Lists must match the number of items per modality and may contain `None`.
    For `None` entries, the hasher will compute IDs automatically; non-None
    entries override the default hashes for caching, and MUST be unique per
    multimodal item.
    """

    cache_salt: NotRequired[str]  # 可选的缓存盐值
    """
    Optional cache salt to be used for prefix caching.
    """


class TextPrompt(_PromptOptions):
    """Schema for a text prompt."""
    # 文本提示的数据模式

    prompt: str  # 在传递给模型之前要进行分词的输入文本
    """The input text to be tokenized before passing to the model."""


class TokensPrompt(_PromptOptions):
    """Schema for a tokenized prompt."""
    # 已分词提示的数据模式

    prompt_token_ids: list[int]  # 传递给模型的令牌ID列表
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]  # 对应令牌ID的提示文本（如果可用）
    """The prompt text corresponding to the token IDs, if available."""

    token_type_ids: NotRequired[list[int]]  # 传递给交叉编码器模型的令牌类型ID列表
    """A list of token type IDs to pass to the cross encoder model."""


class EmbedsPrompt(_PromptOptions):
    """Schema for a prompt provided via token embeddings."""
    # 通过令牌嵌入提供的提示数据模式

    prompt_embeds: torch.Tensor  # 提示的嵌入向量
    """The embeddings of the prompt."""

    prompt: NotRequired[str]  # 对应令牌嵌入的提示文本（如果可用）
    """The prompt text corresponding to the token embeddings, if available."""


DecoderOnlyPrompt: TypeAlias = (  # 仅解码器模型的提示类型别名
    str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt
)
"""
Schema of a prompt for a decoder-only model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.data.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.data.EmbedsPrompt])

For encoder-decoder models, passing a singleton prompt is shorthand for passing
`ExplicitEncoderDecoderPrompt(encoder_prompt=prompt, decoder_prompt=None)`.
"""


EncoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt  # 编码器提示类型别名
"""
Schema of a prompt for the encoder part of a encoder-decoder model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.data.TokensPrompt])
"""


DecoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt  # 解码器提示类型别名
"""
Schema of a prompt for the decoder part of an encoder-decoder model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.data.TokensPrompt])

Note:
    Multi-modal inputs are not supported for decoder prompts.
"""


class ExplicitEncoderDecoderPrompt(TypedDict):
    """
    Schema for a pair of encoder and decoder singleton prompts.

    Note:
        This schema is not valid for decoder-only models.
    """
    # 编码器和解码器单例提示对的数据模式，不适用于仅解码器模型

    encoder_prompt: EncoderPrompt  # 模型编码器部分的提示
    """The prompt for the encoder part of the model."""

    decoder_prompt: DecoderPrompt | None  # 模型解码器部分的提示，None表示自动推断
    """
    The prompt for the decoder part of the model.

    Passing `None` will cause the prompt to be inferred automatically.
    """


EncoderDecoderPrompt: TypeAlias = EncoderPrompt | ExplicitEncoderDecoderPrompt  # 编码器-解码器提示类型别名
"""
Schema for a prompt for an encoder-decoder model.

You can pass a singleton encoder prompt, in which case the decoder prompt is
considered to be `None` (i.e., infer automatically).
"""


SingletonPrompt: TypeAlias = DecoderOnlyPrompt | EncoderPrompt | DecoderPrompt  # 单个提示类型别名
"""
Schema for a single prompt. This is as opposed to a data structure
which encapsulates multiple prompts, such as
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt].
"""


PromptType: TypeAlias = DecoderOnlyPrompt | EncoderDecoderPrompt  # 任意提示类型别名
"""
Schema for any prompt, regardless of model type.

This is the input format accepted by most [`LLM`][vllm.entrypoints.llm.LLM] APIs.
"""


class DataPrompt(_PromptOptions):
    """
    Represents generic inputs that are converted to
    [`PromptType`][vllm.inputs.data.PromptType] by IO processor plugins.
    """
    # 通用输入数据，由IO处理器插件转换为PromptType

    data: Any  # 输入数据
    """The input data."""

    data_format: str  # 输入数据格式
    """The input data format."""


# Outputs of processor  # 处理器的输出定义
class _InputOptions(TypedDict):
    """
    Additional options available to all input types.
    """
    # 所有输入类型可用的附加选项

    arrival_time: NotRequired[float]  # 输入接收时间（渲染之前）
    """The time when the input was received (before rendering)."""

    cache_salt: NotRequired[str]  # 可选的前缀缓存盐值
    """Optional cache salt to be used for prefix caching."""


class TokenInputs(_InputOptions):
    """Represents token-based inputs."""
    # 基于令牌的输入表示

    type: Literal["token"]  # 输入类型标识
    """The type of inputs."""

    prompt_token_ids: list[int]  # 提示的令牌ID列表
    """The token IDs of the prompt."""

    prompt: NotRequired[str]  # 对应令牌ID的提示文本（如果可用）
    """The prompt text corresponding to the token IDs, if available."""


def token_inputs(  # 构造令牌输入的工厂函数
    prompt_token_ids: list[int],  # 提示令牌ID列表
    *,
    prompt: str | None = None,  # 可选的提示文本
    cache_salt: str | None = None,  # 可选的缓存盐值
) -> TokenInputs:  # 返回TokenInputs实例
    """Construct [`TokenInputs`][vllm.inputs.data.TokenInputs] from optional
    values."""
    # 从可选值构造TokenInputs实例
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)  # 创建基本令牌输入

    if prompt is not None:  # 如果提供了提示文本
        inputs["prompt"] = prompt  # 设置提示文本
    if cache_salt is not None:  # 如果提供了缓存盐值
        inputs["cache_salt"] = cache_salt  # 设置缓存盐值

    return inputs  # 返回构造的输入


class EmbedsInputs(_InputOptions):
    """Represents embeddings-based inputs."""
    # 基于嵌入向量的输入表示

    type: Literal["embeds"]  # 输入类型标识
    """The type of inputs."""

    prompt_embeds: torch.Tensor  # 提示的嵌入向量
    """The embeddings of the prompt."""

    prompt: NotRequired[str]  # 对应令牌ID的提示文本（如果可用）
    """The prompt text corresponding to the token IDs, if available."""


def embeds_inputs(  # 构造嵌入输入的工厂函数
    prompt_embeds: torch.Tensor,  # 提示嵌入张量
    *,
    prompt: str | None = None,  # 可选的提示文本
    cache_salt: str | None = None,  # 可选的缓存盐值
) -> EmbedsInputs:  # 返回EmbedsInputs实例
    """Construct [`EmbedsInputs`][vllm.inputs.data.EmbedsInputs] from optional
    values."""
    # 从可选值构造EmbedsInputs实例
    inputs = EmbedsInputs(type="embeds", prompt_embeds=prompt_embeds)  # 创建基本嵌入输入

    if prompt is not None:  # 如果提供了提示文本
        inputs["prompt"] = prompt  # 设置提示文本
    if cache_salt is not None:  # 如果提供了缓存盐值
        inputs["cache_salt"] = cache_salt  # 设置缓存盐值

    return inputs  # 返回构造的输入


DecoderOnlyInputs: TypeAlias = TokenInputs | EmbedsInputs | MultiModalInputs  # 仅解码器处理后输入类型
"""
A processed prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
for decoder-only models.
"""


EncoderInputs: TypeAlias = TokenInputs | MultiModalEncDecInputs  # 编码器处理后输入类型
"""
A processed encoder prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
for encoder-decoder models.
"""


DecoderInputs: TypeAlias = TokenInputs | MultiModalInputs  # 解码器处理后输入类型
"""
A processed decoder prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
for encoder-decoder models.
"""


class EncoderDecoderInputs(TypedDict):
    """
    A processed pair of encoder and decoder singleton prompts.
    [`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
    which can be passed to
    [`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
    for encoder-decoder models.
    """
    # 编码器-解码器模型处理后的输入对

    type: Literal["enc_dec"]  # 输入类型标识为编码器-解码器

    encoder_prompt: EncoderInputs  # 编码器部分的输入
    """The inputs for the encoder portion."""

    decoder_prompt: DecoderInputs  # 解码器部分的输入
    """The inputs for the decoder portion."""

    arrival_time: NotRequired[float]  # 输入接收时间（渲染之前）
    """The time when the input was received (before rendering)."""


ProcessorInputs: TypeAlias = DecoderOnlyInputs | EncoderDecoderInputs  # 处理器输入类型别名
"""
A processed prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor].
"""


SingletonInputs: TypeAlias = DecoderOnlyInputs | MultiModalEncDecInputs  # 单例输入类型别名
"""The inputs for a single encoder/decoder prompt."""


def _validate_enc_inputs(inputs: SingletonInputs) -> EncoderInputs:  # 验证编码器输入的内部函数
    """验证单例输入是否适用于编码器，不支持嵌入输入类型。"""
    if inputs["type"] == "embeds":  # 嵌入输入不支持编码器-解码器模型
        raise ValueError(  # 抛出值错误
            "Embedding inputs are not supported for encoder-decoder models"
        )

    if inputs["type"] == "multimodal" and "encoder_prompt_token_ids" not in inputs:  # 多模态输入需要编码器令牌ID
        raise RuntimeError(  # 抛出运行时错误
            "You should register an encoder-decoder multi-modal processor "
            "for encoder-decoder models."
        )

    return inputs  # type: ignore[return-value]  # 返回验证后的输入


def _validate_dec_inputs(inputs: SingletonInputs) -> DecoderInputs:  # 验证解码器输入的内部函数
    """验证单例输入是否适用于解码器，不支持嵌入输入类型。"""
    if inputs["type"] == "embeds":  # 嵌入输入不支持编码器-解码器模型
        raise ValueError(  # 抛出值错误
            "Embedding inputs are not supported for encoder-decoder models"
        )

    return inputs  # 返回验证后的输入


def _prepare_decoder_input_ids_for_generation(  # 为生成准备解码器输入ID
    decoder_input_ids: list[int],  # 解码器输入令牌ID列表
    decoder_start_token_id: int,  # 解码器起始令牌ID
) -> list[int]:  # 返回处理后的令牌ID列表
    """
    Prepare `decoder_input_ids` for generation with encoder-decoder models,
    according to `GenerationMixin._prepare_decoder_input_ids_for_generation()`.

    Source:
    https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/generation/utils.py
    """
    # 为编码器-解码器模型的生成准备解码器输入ID
    if len(decoder_input_ids) == 0 or decoder_input_ids[0] != decoder_start_token_id:  # 如果为空或首个令牌不是起始令牌
        decoder_input_ids = [decoder_start_token_id] + decoder_input_ids  # 在开头插入起始令牌

    return decoder_input_ids  # 返回处理后的ID列表


def build_enc_dec_inputs(  # 构建编码器-解码器输入
    encoder_inputs: SingletonInputs,  # 编码器输入
    decoder_inputs: SingletonInputs | None,  # 解码器输入（可为None）
    decoder_start_token_id: int,  # 解码器起始令牌ID
) -> EncoderDecoderInputs:  # 返回编码器-解码器输入
    """构建编码器-解码器模型的完整输入，处理多模态和令牌类型。"""
    enc_inputs = _validate_enc_inputs(encoder_inputs)  # 验证编码器输入

    if decoder_inputs is None:  # 如果没有提供解码器输入
        dec_inputs: DecoderInputs = enc_inputs  # 使用编码器输入作为解码器输入
    else:  # 否则验证解码器输入
        dec_inputs = _validate_dec_inputs(decoder_inputs)  # 验证解码器输入

    enc_inputs_new: EncoderInputs  # 声明新的编码器输入变量
    dec_inputs_new: DecoderInputs  # 声明新的解码器输入变量

    if enc_inputs["type"] == "multimodal":  # 如果编码器输入是多模态类型
        from vllm.multimodal.inputs import mm_inputs  # 导入多模态输入构造函数

        enc_inputs_new = token_inputs(  # 从多模态输入中提取编码器令牌输入
            enc_inputs["encoder_prompt_token_ids"],  # 编码器提示令牌ID
            prompt=enc_inputs.get("encoder_prompt"),  # 编码器提示文本
        )
        dec_inputs_new = mm_inputs(  # 构建多模态解码器输入
            prompt_token_ids=dec_inputs["prompt_token_ids"],  # 解码器令牌ID
            prompt=dec_inputs.get("prompt"),  # 解码器提示文本
            mm_kwargs=enc_inputs["mm_kwargs"],  # 多模态关键字参数
            mm_hashes=enc_inputs["mm_hashes"],  # 多模态哈希值
            mm_placeholders=enc_inputs["mm_placeholders"],  # 多模态占位符
        )
    elif enc_inputs["type"] == "token":  # 如果编码器输入是令牌类型
        enc_inputs_new = token_inputs(prompt_token_ids=[])  # 创建空的编码器令牌输入
        dec_inputs_new = dec_inputs  # 直接使用解码器输入
    else:  # 不应到达此分支
        assert_never(enc_inputs)  # 类型穷尽检查

    dec_inputs_new["prompt_token_ids"] = _prepare_decoder_input_ids_for_generation(  # 为生成准备解码器输入ID
        dec_inputs_new["prompt_token_ids"],  # 当前解码器令牌ID
        decoder_start_token_id,  # 起始令牌ID
    )

    if cache_salt := enc_inputs.get("cache_salt"):  # 如果编码器输入中有缓存盐值
        dec_inputs_new["cache_salt"] = cache_salt  # 传递给解码器输入

    return EncoderDecoderInputs(  # 返回编码器-解码器输入
        type="enc_dec",  # 类型标识
        encoder_prompt=enc_inputs_new,  # 编码器提示
        decoder_prompt=dec_inputs_new,  # 解码器提示
    )
