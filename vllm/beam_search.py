# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass  # 导入数据类装饰器

from vllm.inputs import EncoderDecoderInputs, TokenInputs, token_inputs  # 导入编码器-解码器输入、令牌输入类型和令牌输入构造函数
from vllm.inputs.data import DecoderInputs  # 导入解码器输入类型
from vllm.logprobs import Logprob  # 导入对数概率类
from vllm.lora.request import LoRARequest  # 导入LoRA请求类
from vllm.multimodal.inputs import MultiModalInputs, mm_inputs  # 导入多模态输入类型和多模态输入构造函数


@dataclass  # 使用数据类装饰器
class BeamSearchSequence:
    """束搜索序列类。

    用于跟踪序列的令牌和对数概率。
    text字段是可选的，仅在序列即将返回给用户时才会填充。
    """

    orig_prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs  # 原始提示，可以是令牌输入、多模态输入或编码器-解码器输入

    # NOTE: Tokens represents decoder tokens in the encoder / decoder case
    tokens: list[int]  # 令牌列表（在编码器/解码器情况下表示解码器令牌）
    logprobs: list[dict[int, Logprob]]  # 每个位置的对数概率字典列表
    lora_request: LoRARequest | None = None  # 可选的LoRA请求
    cum_logprob: float = 0.0  # 累积对数概率，默认为0.0
    text: str | None = None  # 可选的文本输出
    finish_reason: str | None = None  # 可选的完成原因
    stop_reason: int | str | None = None  # 可选的停止原因

    def get_prompt(self):
        """获取当前束搜索序列的提示。

        根据提示类型（编码器-解码器、令牌或多模态）构建并返回相应格式的提示。
        """
        prompt = self.orig_prompt  # 获取原始提示

        if prompt["type"] == "enc_dec":  # 如果是编码器-解码器类型
            return self._build_encoder_decoder_inputs(prompt)  # 构建编码器-解码器输入

        # Handle decoder-only inputs
        prompt_text = prompt.get("prompt")  # 获取提示文本
        cache_salt = prompt.get("cache_salt")  # 获取缓存盐值

        if prompt["type"] == "token":  # 如果是令牌类型
            return token_inputs(  # 返回令牌输入
                self.tokens,  # 当前序列的令牌
                prompt=prompt_text,  # 提示文本
                cache_salt=cache_salt,  # 缓存盐值
            )

        return mm_inputs(  # 返回多模态输入
            prompt_token_ids=self.tokens,  # 当前序列的令牌ID
            mm_kwargs=prompt["mm_kwargs"],  # 多模态关键字参数
            mm_hashes=prompt["mm_hashes"],  # 多模态哈希值
            mm_placeholders=prompt["mm_placeholders"],  # 多模态占位符
            prompt=prompt_text,  # 提示文本
            cache_salt=cache_salt,  # 缓存盐值
        )

    def _build_encoder_decoder_inputs(
        self, prompt: EncoderDecoderInputs
    ) -> EncoderDecoderInputs:
        """使用当前束搜索序列的令牌重新构建编码器-解码器输入。

        注意：编码器多模态缓存尚未正确连接，这意味着目前我们在每个新束上
        都会运行编码器，因为每个新请求的num_computed_tokens为0。
        一旦缓存正确实现，这个问题将会被修复。
        """
        dec_prompt = prompt["decoder_prompt"]  # 获取解码器提示

        # Rebuild decoder prompt with updated tokens,
        # but keep everything else the same.
        new_dec_prompt: DecoderInputs  # 声明新的解码器提示变量
        if dec_prompt["type"] == "multimodal":  # 如果解码器提示是多模态类型
            new_dec_prompt = mm_inputs(  # 构建多模态输入
                self.tokens,  # 当前序列的令牌
                mm_kwargs=dec_prompt["mm_kwargs"],  # 多模态关键字参数
                mm_hashes=dec_prompt["mm_hashes"],  # 多模态哈希值
                mm_placeholders=dec_prompt["mm_placeholders"],  # 多模态占位符
                prompt=dec_prompt.get("prompt"),  # 提示文本
                cache_salt=dec_prompt.get("cache_salt"),  # 缓存盐值
            )
        else:  # 否则是普通令牌类型
            new_dec_prompt = token_inputs(  # 构建令牌输入
                self.tokens,  # 当前序列的令牌
                prompt=dec_prompt.get("prompt"),  # 提示文本
                cache_salt=dec_prompt.get("cache_salt"),  # 缓存盐值
            )

        return EncoderDecoderInputs(  # 返回编码器-解码器输入
            type="enc_dec",  # 类型为编码器-解码器
            encoder_prompt=prompt["encoder_prompt"],  # 保持原始编码器提示不变
            decoder_prompt=new_dec_prompt,  # 使用更新后的解码器提示
        )


@dataclass  # 使用数据类装饰器
class BeamSearchOutput:
    """束搜索输出类。

    包含最佳束搜索序列的列表。
    列表的长度等于束宽度。
    """

    sequences: list[BeamSearchSequence]  # 束搜索序列列表


class BeamSearchInstance:
    """束搜索实例类。

    管理一组束搜索序列，包括活跃的束和已完成的束。
    """

    def __init__(
        self,
        prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs,  # 输入提示
        lora_request: LoRARequest | None = None,  # 可选的LoRA请求
        logprobs: list[dict[int, Logprob]] | None = None,  # 可选的初始对数概率
        **kwargs,  # 其他关键字参数
    ):
        """初始化束搜索实例。

        根据提示类型提取初始令牌，并创建初始束搜索序列。

        Args:
            prompt: 输入提示，支持令牌输入、多模态输入或编码器-解码器输入。
            lora_request: 可选的LoRA请求。
            logprobs: 可选的初始对数概率列表。
            **kwargs: 传递给BeamSearchSequence的其他参数。
        """
        decoder_prompt = (  # 获取解码器提示
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]  # 如果是编码器-解码器类型则取解码器部分
        )
        initial_tokens = decoder_prompt["prompt_token_ids"]  # 提取初始令牌ID

        self.beams: list[BeamSearchSequence] = [  # 初始化活跃束列表
            BeamSearchSequence(  # 创建初始束搜索序列
                orig_prompt=prompt,  # 原始提示
                tokens=initial_tokens,  # 初始令牌
                logprobs=[] if logprobs is None else list(logprobs),  # 对数概率列表
                lora_request=lora_request,  # LoRA请求
                **kwargs,  # 其他参数
            )
        ]
        self.completed: list[BeamSearchSequence] = []  # 初始化已完成束列表为空


def get_beam_search_score(
    tokens: list[int],  # 令牌列表
    cumulative_logprob: float,  # 累积对数概率
    eos_token_id: int,  # 结束令牌ID
    length_penalty: float = 1.0,  # 长度惩罚系数，默认为1.0
) -> float:
    """计算带长度惩罚的束搜索分数。

    根据累积对数概率和序列长度计算束搜索分数，
    长度惩罚用于控制生成序列的长度偏好。

    改编自 https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938

    Args:
        tokens: 令牌ID列表。
        cumulative_logprob: 累积对数概率。
        eos_token_id: 结束符令牌ID。
        length_penalty: 长度惩罚系数。

    Returns:
        束搜索分数。
    """
    seq_len = len(tokens)  # 获取序列长度
    if tokens[-1] == eos_token_id:  # 如果最后一个令牌是结束符
        seq_len -= 1  # 序列长度减1（不计入结束符）

    return cumulative_logprob / (seq_len**length_penalty)  # 返回累积对数概率除以长度惩罚后的分数


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    """创建用于束排序的键函数。

    返回一个函数，该函数可用于按束搜索分数对束进行排序。

    Args:
        eos_token_id: 结束符令牌ID。
        length_penalty: 长度惩罚系数。

    Returns:
        一个接受BeamSearchSequence并返回其分数的函数。
    """
    def sort_beams_key(x: BeamSearchSequence) -> float:
        """束排序键函数，返回给定束搜索序列的分数。"""
        return get_beam_search_score(  # 计算束搜索分数
            x.tokens, x.cum_logprob, eos_token_id, length_penalty  # 传入令牌、累积概率、结束符ID和长度惩罚
        )

    return sort_beams_key  # 返回排序键函数
