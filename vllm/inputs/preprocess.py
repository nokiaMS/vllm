# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from collections.abc import Mapping  # 导入映射抽象基类
from typing import Any, overload  # 导入类型注解工具

from typing_extensions import assert_never  # 导入穷尽类型检查工具

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.inputs.data import build_enc_dec_inputs  # 导入编码器-解码器输入构建函数
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 导入多模态注册表
from vllm.multimodal.inputs import (  # 导入多模态输入类型
    MultiModalDataDict,  # 多模态数据字典
    MultiModalInputs,  # 多模态输入
    MultiModalUUIDDict,  # 多模态UUID字典
)
from vllm.renderers import BaseRenderer, renderer_from_config  # 导入渲染器基类和配置创建函数
from vllm.renderers.inputs import (  # 导入渲染器输入类型
    DecoderDictPrompt,  # 解码器字典提示
    DecoderOnlyDictPrompt,  # 仅解码器字典提示
    EncoderDecoderDictPrompt,  # 编码器-解码器字典提示
    EncoderDictPrompt,  # 编码器字典提示
    SingletonDictPrompt,  # 单例字典提示
)
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt, parse_enc_dec_prompt  # 导入提示解析函数
from vllm.tokenizers import TokenizerLike  # 导入分词器协议类型

from .data import (  # 从data模块导入数据类型
    DecoderInputs,  # 解码器输入
    DecoderOnlyInputs,  # 仅解码器输入
    EmbedsInputs,  # 嵌入输入
    EmbedsPrompt,  # 嵌入提示
    EncoderDecoderInputs,  # 编码器-解码器输入
    EncoderInputs,  # 编码器输入
    ProcessorInputs,  # 处理器输入
    PromptType,  # 提示类型
    SingletonInputs,  # 单例输入
    TextPrompt,  # 文本提示
    TokenInputs,  # 令牌输入
    TokensPrompt,  # 令牌提示
    token_inputs,  # 令牌输入构造函数
)

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class InputPreprocessor:
    """输入预处理器，负责将原始提示转换为模型可处理的输入格式。"""

    def __init__(  # 初始化方法
        self,
        vllm_config: VllmConfig,  # vLLM配置对象
        renderer: BaseRenderer | None = None,  # 可选的渲染器实例
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.model_config = vllm_config.model_config  # 保存模型配置
        self.renderer = renderer or renderer_from_config(vllm_config)  # 创建或使用渲染器
        self.mm_registry = mm_registry  # 保存多模态注册表

    @property
    def tokenizer(self) -> TokenizerLike | None:  # 分词器属性
        """获取渲染器的分词器实例。"""
        return self.renderer.tokenizer  # 返回渲染器的分词器

    def get_tokenizer(self) -> TokenizerLike:  # 获取分词器方法
        """获取渲染器的分词器，确保不为None。"""
        return self.renderer.get_tokenizer()  # 返回渲染器的分词器

    def _tokenize_prompt(  # 分词提示的内部方法
        self,
        prompt: str,  # 文本提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 可选的分词参数
    ) -> list[int]:  # 返回令牌ID列表
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        # 对文本提示应用模型的分词器，返回对应的令牌ID
        renderer = self.renderer  # 获取渲染器

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(  # 获取默认分词参数并合并自定义参数
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(  # 对单例提示进行分词
            TextPrompt(prompt=prompt),  # 创建文本提示对象
            tok_params,  # 分词参数
        )

        return tok_prompt["prompt_token_ids"]  # 返回令牌ID列表

    def _process_multimodal(  # 处理多模态输入的内部方法
        self,
        prompt: str | list[int],  # 文本或令牌ID列表
        mm_data: MultiModalDataDict,  # 多模态数据字典
        mm_processor_kwargs: Mapping[str, object] | None = None,  # 多模态处理器参数
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
        *,
        mm_uuids: MultiModalUUIDDict | None = None,  # 多模态UUID
    ) -> MultiModalInputs:  # 返回多模态输入
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        # 对多模态提示应用模型的多模态处理器，返回令牌ID和元数据
        return self.renderer._process_multimodal(  # 调用渲染器的多模态处理方法
            prompt,  # 提示内容
            mm_data,  # 多模态数据
            mm_uuids=mm_uuids,  # UUID信息
            mm_processor_kwargs=mm_processor_kwargs,  # 处理器参数
            tokenization_kwargs=tokenization_kwargs,  # 分词参数
        )

    def _process_embeds(  # 处理嵌入输入的内部方法
        self,
        parsed_content: EmbedsPrompt,  # 嵌入提示对象
    ) -> EmbedsInputs:  # 返回嵌入输入
        """处理嵌入类型的提示内容。"""
        return self.renderer._process_embeds(parsed_content)  # 调用渲染器的嵌入处理方法

    def _truncate_inputs(  # 截断输入的内部方法
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None  # 令牌ID列表和分词参数
    ) -> list[int]:  # 返回截断后的令牌ID列表
        """根据分词参数截断输入令牌列表。"""
        renderer = self.renderer  # 获取渲染器

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(  # 合并分词参数
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(  # 对令牌提示进行处理（可能截断）
            TokensPrompt(prompt_token_ids=inputs),  # 创建令牌提示对象
            tok_params,  # 分词参数
        )

        return tok_prompt["prompt_token_ids"]  # 返回处理后的令牌ID

    def _process_tokens(  # 处理令牌输入的内部方法
        self,
        parsed_content: TokensPrompt,  # 令牌提示对象
        tokenization_kwargs: dict[str, Any] | None = None,  # 可选的分词参数
    ) -> TokenInputs | MultiModalInputs:  # 返回令牌输入或多模态输入
        """处理已分词的提示，可能包含多模态数据。"""
        prompt_token_ids = self._truncate_inputs(  # 截断令牌ID
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )

        inputs: TokenInputs | MultiModalInputs  # 声明输入变量类型
        if multi_modal_data := parsed_content.get("multi_modal_data"):  # 如果包含多模态数据
            inputs = self._process_multimodal(  # 处理多模态输入
                prompt_token_ids,  # 令牌ID
                multi_modal_data,  # 多模态数据
                parsed_content.get("mm_processor_kwargs"),  # 处理器参数
                tokenization_kwargs=tokenization_kwargs,  # 分词参数
                mm_uuids=parsed_content.get("multi_modal_uuids"),  # UUID
            )
        else:  # 不包含多模态数据
            inputs = token_inputs(prompt_token_ids)  # 创建纯令牌输入

        if prompt_text := parsed_content.get("prompt"):  # 如果有提示文本
            inputs["prompt"] = prompt_text  # 设置提示文本
        if cache_salt := parsed_content.get("cache_salt"):  # 如果有缓存盐值
            inputs["cache_salt"] = cache_salt  # 设置缓存盐值

        return inputs  # 返回处理后的输入

    def _process_text(  # 处理文本输入的内部方法
        self,
        parsed_content: TextPrompt,  # 文本提示对象
        tokenization_kwargs: dict[str, Any] | None = None,  # 可选的分词参数
    ) -> TokenInputs | MultiModalInputs:  # 返回令牌输入或多模态输入
        """处理文本提示，进行分词并可能处理多模态数据。"""
        prompt_text = parsed_content["prompt"]  # 获取提示文本

        inputs: TokenInputs | MultiModalInputs  # 声明输入变量类型
        if multi_modal_data := parsed_content.get("multi_modal_data"):  # 如果包含多模态数据
            inputs = self._process_multimodal(  # 处理多模态输入
                prompt_text,  # 文本提示
                multi_modal_data,  # 多模态数据
                parsed_content.get("mm_processor_kwargs") or {},  # 处理器参数
                tokenization_kwargs=tokenization_kwargs,  # 分词参数
            )
        else:  # 不包含多模态数据
            prompt_token_ids = self._tokenize_prompt(  # 对文本进行分词
                prompt_text,  # 文本内容
                tokenization_kwargs=tokenization_kwargs,  # 分词参数
            )
            inputs = token_inputs(prompt_token_ids)  # 创建令牌输入

        inputs["prompt"] = prompt_text  # 设置提示文本

        if cache_salt := parsed_content.get("cache_salt"):  # 如果有缓存盐值
            inputs["cache_salt"] = cache_salt  # 设置缓存盐值

        return inputs  # 返回处理后的输入

    @overload
    def _prompt_to_llm_inputs(  # 重载：编码器提示到LLM输入
        self,
        prompt: EncoderDictPrompt,  # 编码器字典提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> EncoderInputs: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]  # 重载：解码器提示到LLM输入
        self,
        prompt: DecoderDictPrompt,  # 解码器字典提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> DecoderInputs: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]  # 重载：仅解码器提示到LLM输入
        self,
        prompt: DecoderOnlyDictPrompt,  # 仅解码器字典提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> DecoderOnlyInputs: ...

    def _prompt_to_llm_inputs(  # 将提示转换为LLM输入的方法
        self,
        prompt: SingletonDictPrompt,  # 单例字典提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> SingletonInputs:  # 返回单例输入
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
        """
        # 从提示中提取单例输入
        if "prompt_embeds" in prompt:  # 如果是嵌入提示
            return self._process_embeds(prompt)  # type: ignore[arg-type]  # 处理嵌入

        if "prompt_token_ids" in prompt:  # 如果是令牌提示
            return self._process_tokens(prompt)  # type: ignore[arg-type]  # 处理令牌

        if "prompt" in prompt:  # 如果是文本提示
            return self._process_text(  # 处理文本
                prompt,  # type: ignore[arg-type]  # 文本提示
                tokenization_kwargs=tokenization_kwargs,  # 分词参数
            )

        assert_never(prompt)  # type: ignore[arg-type]  # 穷尽类型检查

    def _process_encoder_decoder_prompt(  # 处理编码器-解码器提示的方法
        self,
        prompt: EncoderDecoderDictPrompt,  # 编码器-解码器字典提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> EncoderDecoderInputs:  # 返回编码器-解码器输入
        """
        For encoder/decoder models only:
        Process an input prompt into an
        [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
        instance.

        Arguments:

        * prompt: an input prompt

        Returns:

        * [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
          instance
        """
        # 仅用于编码器/解码器模型：将输入提示处理为EncoderDecoderInputs实例
        encoder_prompt = prompt["encoder_prompt"]  # 获取编码器提示
        decoder_prompt = prompt["decoder_prompt"]  # 获取解码器提示

        return build_enc_dec_inputs(  # 构建编码器-解码器输入
            encoder_inputs=self._prompt_to_llm_inputs(  # 转换编码器提示
                encoder_prompt,  # 编码器提示
                tokenization_kwargs=tokenization_kwargs,  # 分词参数
            ),
            decoder_inputs=(  # 转换解码器提示
                None  # 如果解码器提示为None
                if decoder_prompt is None
                else self._prompt_to_llm_inputs(  # 否则转换解码器提示
                    decoder_prompt,  # 解码器提示
                    tokenization_kwargs=tokenization_kwargs,  # 分词参数
                )
            ),
            decoder_start_token_id=self.renderer.get_dec_start_token_id(),  # 获取解码器起始令牌ID
        )

    def _process_decoder_only_prompt(  # 处理仅解码器提示的方法
        self,
        prompt: DecoderOnlyDictPrompt,  # 仅解码器字典提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> DecoderOnlyInputs:  # 返回仅解码器输入
        """
        For decoder-only models:
        Process an input prompt into a
        [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance.

        Arguments:

        * prompt: input prompt

        Returns:

        * [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance
        """
        # 仅用于解码器模型：将输入提示处理为DecoderOnlyInputs实例
        return self._prompt_to_llm_inputs(  # 转换提示为LLM输入
            prompt,  # 输入提示
            tokenization_kwargs=tokenization_kwargs,  # 分词参数
        )

    def preprocess(  # 预处理入口方法
        self,
        prompt: PromptType,  # 输入提示
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词参数
    ) -> ProcessorInputs:  # 返回处理器输入
        """Preprocess the input prompt."""
        # 预处理输入提示
        if self.model_config.is_encoder_decoder:  # 如果是编码器-解码器模型
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            return self._process_encoder_decoder_prompt(  # 处理编码器-解码器提示
                parse_enc_dec_prompt(prompt),  # 解析编码器-解码器提示
                tokenization_kwargs,  # 分词参数
            )

        return self._process_decoder_only_prompt(  # 处理仅解码器提示
            parse_dec_only_prompt(prompt),  # 解析仅解码器提示
            tokenization_kwargs=tokenization_kwargs,  # 分词参数
        )
