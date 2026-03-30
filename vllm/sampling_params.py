# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling parameters for text generation.
文本生成的采样参数。
"""

import copy  # 导入深拷贝/浅拷贝模块
import json as json_mod  # 导入JSON模块并重命名为json_mod
from dataclasses import field  # 导入数据类字段定义工具
from enum import Enum, IntEnum  # 导入枚举类型和整数枚举类型
from functools import cached_property  # 导入缓存属性装饰器
from typing import Any  # 导入Any类型注解

import msgspec  # 导入msgspec序列化/反序列化库
from pydantic.dataclasses import dataclass  # 导入pydantic数据类装饰器

from vllm.config import ModelConfig, SpeculativeConfig, StructuredOutputsConfig  # 导入模型配置、推测解码配置和结构化输出配置
from vllm.exceptions import VLLMValidationError  # 导入vLLM验证错误类
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.tokenizers import TokenizerLike  # 导入分词器类型协议
from vllm.utils.mistral import is_mistral_tokenizer  # 导入Mistral分词器检测函数
from vllm.v1.serial_utils import PydanticMsgspecMixin  # 导入Pydantic与Msgspec混合序列化工具

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

_SAMPLING_EPS = 1e-5  # 采样精度阈值，低于此值视为零温度（贪心采样）
_MAX_TEMP = 1e-2  # 最低有效温度值，防止数值错误


class SamplingType(IntEnum):
    """采样类型枚举，定义不同的采样策略。"""
    GREEDY = 0  # 贪心采样，始终选择概率最高的token
    RANDOM = 1  # 随机采样
    RANDOM_SEED = 2  # 带固定种子的随机采样


# maybe make msgspec?
@dataclass
class StructuredOutputsParams:
    """结构化输出参数，用于约束模型生成符合特定格式的输出。"""
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None  # JSON Schema约束，可以是字符串或字典
    regex: str | None = None  # 正则表达式约束
    choice: list[str] | None = None  # 选项列表约束，生成结果必须是列表中的某一个
    grammar: str | None = None  # 语法约束（如BNF/EBNF语法）
    json_object: bool | None = None  # JSON对象约束，要求输出为有效JSON对象
    # These are other options that can be set.
    disable_any_whitespace: bool = False  # 是否禁用任意空白符匹配
    disable_additional_properties: bool = False  # 是否禁用JSON Schema中的额外属性
    whitespace_pattern: str | None = None  # 自定义空白符匹配模式
    structural_tag: str | None = None  # 结构化标签约束

    _backend: str | None = field(default=None, init=False)  # 后端名称（内部使用，不通过构造函数初始化）
    """CAUTION: Should only be set by Processor._validate_structured_output
    注意：此字段仅应由Processor._validate_structured_output设置。
    """
    _backend_was_auto: bool = field(default=False, init=False)  # 标记后端是否由auto模式自动选择
    """CAUTION: Should only be set by Processor._validate_structured_output
    注意：此字段仅应由Processor._validate_structured_output设置。
    """

    def __post_init__(self):
        """Validate that some fields are mutually exclusive.
        验证某些字段互斥，确保只指定了一种结构化输出约束。
        """
        count = sum(  # 计算已指定的约束类型数量
            [
                self.json is not None,  # 检查json约束是否已设置
                self.regex is not None,  # 检查regex约束是否已设置
                self.choice is not None,  # 检查choice约束是否已设置
                self.grammar is not None,  # 检查grammar约束是否已设置
                self.json_object is not None,  # 检查json_object约束是否已设置
                self.structural_tag is not None,  # 检查structural_tag约束是否已设置
            ]
        )
        if count > 1:  # 如果指定了多个约束类型，抛出错误
            raise ValueError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:  # 如果没有指定任何约束类型，抛出错误
            raise ValueError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        如果所有结构化输出约束字段均为None，则返回True。
        """
        return all(  # 检查所有约束字段是否都为None
            getattr(self, field) is None  # 获取指定字段的值并判断是否为None
            for field in (
                "json",  # JSON约束字段
                "regex",  # 正则表达式约束字段
                "choice",  # 选项约束字段
                "grammar",  # 语法约束字段
                "json_object",  # JSON对象约束字段
                "structural_tag",  # 结构化标签约束字段
            )
        )

    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        如果所有非structural_tag的结构化输出约束字段均为None，则返回True。
        """
        return all(  # 检查除structural_tag外的所有约束字段是否都为None
            getattr(self, field) is None  # 获取指定字段的值并判断是否为None
            for field in (
                "json",  # JSON约束字段
                "regex",  # 正则表达式约束字段
                "choice",  # 选项约束字段
                "grammar",  # 语法约束字段
                "json_object",  # JSON对象约束字段
            )
        )


@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens.
    用于检测输出token中重复N-gram模式的参数。
    """

    max_pattern_size: int = 0  # 要检测的最大N-gram模式大小，设为0表示禁用
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count.
    检测序列重复的最大N-gram模式大小。设为0禁用。必须与min_count一起使用。
    """

    min_pattern_size: int = 0  # 要检查的最小N-gram模式大小，设为0则默认为1
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size.
    检查序列重复的最小N-gram模式大小。如果设为0，默认为1。必须<=max_pattern_size。
    """

    min_count: int = 0  # N-gram模式必须重复的最少次数，必须>=2
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size.
    N-gram模式必须重复的最少次数才能触发检测。必须>=2。
    例如：设为3表示检测重复了3次的短语。必须与max_pattern_size一起使用。
    """

    def __post_init__(self):
        """验证重复检测参数的有效性。"""
        if (  # 验证参数范围和大小关系
            self.max_pattern_size < 0  # 最大模式大小不能为负
            or self.min_pattern_size < 0  # 最小模式大小不能为负
            or self.min_pattern_size > self.max_pattern_size  # 最小值不能超过最大值
        ):
            raise ValueError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        if self.max_pattern_size > 0 and self.min_count < 2:  # 启用检测时min_count必须>=2
            raise ValueError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )


class RequestOutputKind(Enum):
    """请求输出类型枚举，控制输出返回方式。"""
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0  # 累积模式，每次返回到目前为止的完整输出
    # Return only deltas in each RequestOutput
    DELTA = 1  # 增量模式，每次只返回新增的部分
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2  # 仅最终结果模式，不返回中间结果


class SamplingParams(
    PydanticMsgspecMixin,  # 混合Pydantic和Msgspec的序列化能力
    msgspec.Struct,  # 继承msgspec结构体
    omit_defaults=True,  # type: ignore[call-arg]  # 序列化时省略默认值
    # required for @cached_property.
    dict=True,  # 启用字典转换支持，cached_property所需
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    文本生成的采样参数。
    总体上，我们遵循OpenAI文本补全API的采样参数。
    此外，我们还支持OpenAI不支持的束搜索。
    """

    n: int = 1  # 每个提示请求返回的输出数量
    """Number of outputs to return for the given prompt request.

    NOTE:
        `AsyncLLM` streams outputs by default. When `n > 1`, all `n` outputs
        are generated and streamed cumulatively per request. To see all `n`
        outputs upon completion, use `output_kind=RequestOutputKind.FINAL_ONLY`
        in `SamplingParams`.
    每个提示请求返回的输出数量。
    注意：AsyncLLM默认流式输出。当n>1时，所有n个输出按请求累积生成和流式传输。
    """
    presence_penalty: float = 0.0  # 存在惩罚系数，基于token是否出现过进行惩罚
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens.
    基于新token是否出现在已生成文本中进行惩罚。值>0鼓励使用新token，值<0鼓励重复token。
    """
    frequency_penalty: float = 0.0  # 频率惩罚系数，基于token出现频率进行惩罚
    """Penalizes new tokens based on their frequency in the generated text so
    far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens.
    基于新token在已生成文本中的出现频率进行惩罚。值>0鼓励使用新token，值<0鼓励重复token。
    """
    repetition_penalty: float = 1.0  # 重复惩罚系数，基于token是否出现在提示和已生成文本中
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens.
    基于新token是否出现在提示和已生成文本中进行惩罚。值>1鼓励使用新token，值<1鼓励重复token。
    """
    temperature: float = 1.0  # 温度参数，控制采样随机性
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling.
    控制采样的随机性。较低值使模型更确定性，较高值使模型更随机。零表示贪心采样。
    """
    top_p: float = 1.0  # 核采样参数，控制累积概率阈值
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens.
    控制要考虑的顶部token的累积概率。必须在(0, 1]之间。设为1考虑所有token。
    """
    top_k: int = 0  # Top-K采样参数，控制考虑的顶部token数量
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens.
    控制要考虑的顶部token数量。设为0（或-1）考虑所有token。
    """
    min_p: float = 0.0  # 最小概率阈值，相对于最可能token的概率
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this.
    表示token被考虑的最小概率（相对于最可能token的概率）。必须在[0, 1]之间。设为0禁用。
    """
    seed: int | None = None  # 随机种子，用于可复现的生成
    """Random seed to use for the generation.
    用于生成的随机种子。
    """
    stop: str | list[str] | None = None  # 停止字符串，生成到这些字符串时停止
    """String(s) that stop the generation when they are generated. The returned
    output will not contain the stop strings.
    生成时遇到这些字符串则停止生成。返回的输出不包含停止字符串。
    """
    stop_token_ids: list[int] | None = None  # 停止token ID列表
    """Token IDs that stop the generation when they are generated. The returned
    output will contain the stop tokens unless the stop tokens are special
    tokens.
    生成时遇到这些token ID则停止生成。除非是特殊token，否则输出会包含停止token。
    """
    ignore_eos: bool = False  # 是否忽略EOS token继续生成
    """Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated.
    是否忽略EOS token并在生成EOS token后继续生成。
    """
    max_tokens: int | None = 16  # 每个输出序列的最大生成token数
    """Maximum number of tokens to generate per output sequence.
    每个输出序列生成的最大token数量。
    """
    min_tokens: int = 0  # 在允许EOS或stop_token_ids之前的最少生成token数
    """Minimum number of tokens to generate per output sequence before EOS or
    `stop_token_ids` can be generated
    每个输出序列在允许生成EOS或stop_token_ids之前必须生成的最少token数。
    """
    logprobs: int | None = None  # 每个输出token返回的对数概率数量
    """Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities.
    每个输出token返回的对数概率数量。设为None不返回概率。设为-1返回全部词汇表的对数概率。
    """
    prompt_logprobs: int | None = None  # 每个提示token返回的对数概率数量
    """Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities.
    每个提示token返回的对数概率数量。设为-1返回全部词汇表的对数概率。
    """
    flat_logprobs: bool = False  # 是否以扁平格式返回logprobs以提高性能
    """Whether to return logprobs in flatten format (i.e. FlatLogprob)
    for better performance.
    NOTE: GC costs of FlatLogprobs is significantly smaller than
    list[dict[int, Logprob]]. After enabled, PromptLogprobs and
    SampleLogprobs would populated as FlatLogprobs.
    是否以扁平格式返回logprobs以获得更好性能。FlatLogprobs的GC开销远小于list[dict]格式。
    """
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    detokenize: bool = True  # 是否对输出进行反分词（token转文本）
    """Whether to detokenize the output.
    是否对输出进行反分词处理。
    """
    skip_special_tokens: bool = True  # 是否在输出中跳过特殊token
    """Whether to skip special tokens in the output.
    是否在输出中跳过特殊token。
    """
    spaces_between_special_tokens: bool = True  # 是否在特殊token之间添加空格
    """Whether to add spaces between special tokens in the output.
    是否在输出的特殊token之间添加空格。
    """
    include_stop_str_in_output: bool = False  # 是否在输出文本中包含停止字符串
    """Whether to include the stop strings in output text.
    是否在输出文本中包含停止字符串。
    """
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE  # 输出模式，默认为累积模式
    skip_clone: bool = False  # 是否跳过深拷贝，使用浅拷贝代替
    """Internal flag indicating that this SamplingParams instance is safe to
    reuse without cloning. When True, clone() will return self without
    performing a deep copy. This should only be set when the params object
    is guaranteed to be dedicated to a single request and won't be modified
    in ways that would affect other uses.
    内部标志，表示此SamplingParams实例可安全复用而无需克隆。
    当为True时，clone()将返回浅拷贝而非深拷贝。
    """

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0  # 输出文本缓冲区长度，用于停止字符串评估
    _eos_token_id: int | None = None  # EOS token ID（内部使用）
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)  # 所有停止token ID的集合

    # Fields used to construct logits processors
    structured_outputs: StructuredOutputsParams | None = None  # 结构化输出参数
    """Parameters for configuring structured outputs.
    配置结构化输出的参数。
    """
    logit_bias: dict[int, float] | None = None  # logit偏置字典，用于调整特定token的概率
    """If provided, the engine will construct a logits processor that applies
    these logit biases.
    如果提供，引擎将构建一个应用这些logit偏置的logits处理器。
    """
    allowed_token_ids: list[int] | None = None  # 允许的token ID列表
    """If provided, the engine will construct a logits processor which only
    retains scores for the given token ids.
    如果提供，引擎将构建一个仅保留给定token ID分数的logits处理器。
    """
    extra_args: dict[str, Any] | None = None  # 额外参数，可用于自定义采样实现
    """Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc. Not used by any in-tree sampling
    implementations.
    任意额外参数，可用于自定义采样实现、插件等。不被任何内置采样实现使用。
    """

    # Fields used for bad words
    bad_words: list[str] | None = None  # 禁止生成的词语列表
    """Words that are not allowed to be generated. More precisely, only the
    last token of a corresponding token sequence is not allowed when the next
    generated token can complete the sequence.
    不允许生成的词语。更准确地说，当下一个生成的token能完成对应token序列时，
    该序列的最后一个token将被禁止。
    """
    _bad_words_token_ids: list[list[int]] | None = None  # 禁止词语对应的token ID序列列表（内部使用）

    skip_reading_prefix_cache: bool | None = None  # 是否跳过读取前缀缓存

    repetition_detection: RepetitionDetectionParams | None = None  # 重复检测参数
    """Parameters for detecting repetitive N-gram patterns in output tokens.
    If such repetition is detected, generation will be ended early. LLMs can
    sometimes generate repetitive, unhelpful token patterns, stopping only
    when they hit the maximum output length (e.g. 'abcdabcdabcd...' or
    '\\emoji \\emoji \\emoji ...'). This feature can detect such behavior
    and terminate early, saving time and tokens.
    用于检测输出token中重复N-gram模式的参数。如果检测到重复，生成将提前结束。
    LLM有时会生成重复无用的token模式，此功能可检测并提前终止，节省时间和token。
    """

    @staticmethod
    def from_optional(
        n: int | None = 1,  # 输出数量（可选）
        presence_penalty: float | None = 0.0,  # 存在惩罚（可选）
        frequency_penalty: float | None = 0.0,  # 频率惩罚（可选）
        repetition_penalty: float | None = 1.0,  # 重复惩罚（可选）
        temperature: float | None = 1.0,  # 温度（可选）
        top_p: float | None = 1.0,  # Top-P（可选）
        top_k: int = 0,  # Top-K
        min_p: float = 0.0,  # Min-P
        seed: int | None = None,  # 随机种子
        stop: str | list[str] | None = None,  # 停止字符串
        stop_token_ids: list[int] | None = None,  # 停止token ID列表
        bad_words: list[str] | None = None,  # 禁止词语列表
        include_stop_str_in_output: bool = False,  # 是否在输出中包含停止字符串
        ignore_eos: bool = False,  # 是否忽略EOS
        max_tokens: int | None = 16,  # 最大token数
        min_tokens: int = 0,  # 最小token数
        logprobs: int | None = None,  # 对数概率数量
        prompt_logprobs: int | None = None,  # 提示对数概率数量
        detokenize: bool = True,  # 是否反分词
        skip_special_tokens: bool = True,  # 是否跳过特殊token
        spaces_between_special_tokens: bool = True,  # 特殊token间是否加空格
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,  # 输出类型
        structured_outputs: StructuredOutputsParams | None = None,  # 结构化输出参数
        logit_bias: dict[int, float] | dict[str, float] | None = None,  # logit偏置
        allowed_token_ids: list[int] | None = None,  # 允许的token ID
        extra_args: dict[str, Any] | None = None,  # 额外参数
        skip_clone: bool = False,  # 是否跳过克隆
        repetition_detection: RepetitionDetectionParams | None = None,  # 重复检测参数
    ) -> "SamplingParams":
        """从可选参数创建SamplingParams实例，将None值替换为默认值。"""
        if logit_bias is not None:  # 如果提供了logit偏置
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            logit_bias = {  # 将token_id转为整数并将偏置值钳制在[-100, 100]范围内
                int(token): min(100.0, max(-100.0, bias))  # 转换并钳制每个偏置值
                for token, bias in logit_bias.items()  # 遍历所有logit偏置项
            }

        return SamplingParams(  # 创建并返回SamplingParams实例
            n=1 if n is None else n,  # n为None时使用默认值1
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,  # 存在惩罚为None时使用默认值0.0
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,  # 频率惩罚为None时使用默认值0.0
            repetition_penalty=1.0  # 重复惩罚为None时使用默认值1.0
            if repetition_penalty is None
            else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,  # 温度为None时使用默认值1.0
            top_p=1.0 if top_p is None else top_p,  # top_p为None时使用默认值1.0
            top_k=top_k,  # 传入top_k值
            min_p=min_p,  # 传入min_p值
            seed=seed,  # 传入随机种子
            stop=stop,  # 传入停止字符串
            stop_token_ids=stop_token_ids,  # 传入停止token ID列表
            bad_words=bad_words,  # 传入禁止词语列表
            include_stop_str_in_output=include_stop_str_in_output,  # 传入是否包含停止字符串
            ignore_eos=ignore_eos,  # 传入是否忽略EOS
            max_tokens=max_tokens,  # 传入最大token数
            min_tokens=min_tokens,  # 传入最小token数
            logprobs=logprobs,  # 传入对数概率数量
            prompt_logprobs=prompt_logprobs,  # 传入提示对数概率数量
            detokenize=detokenize,  # 传入是否反分词
            skip_special_tokens=skip_special_tokens,  # 传入是否跳过特殊token
            spaces_between_special_tokens=spaces_between_special_tokens,  # 传入特殊token间空格设置
            output_kind=output_kind,  # 传入输出类型
            structured_outputs=structured_outputs,  # 传入结构化输出参数
            logit_bias=logit_bias,  # 传入logit偏置
            allowed_token_ids=allowed_token_ids,  # 传入允许的token ID
            extra_args=extra_args,  # 传入额外参数
            skip_clone=skip_clone,  # 传入是否跳过克隆
            repetition_detection=repetition_detection,  # 传入重复检测参数
        )

    def __post_init__(self) -> None:
        """初始化后处理，验证参数并设置默认值。"""
        if 0 < self.temperature < _MAX_TEMP:  # 如果温度大于0但小于最低有效值
            logger.warning(  # 发出警告日志
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)  # 将温度提升到最低有效值

        if self.seed == -1:  # 如果种子为-1，视为未设置
            self.seed = None  # 将种子设为None

        if self.stop is None:  # 如果停止字符串为None
            self.stop = []  # 初始化为空列表
        elif isinstance(self.stop, str):  # 如果停止字符串是单个字符串
            self.stop = [self.stop]  # 将其包装为列表

        if self.stop_token_ids is None:  # 如果停止token ID列表为None
            self.stop_token_ids = []  # 初始化为空列表

        if self.bad_words is None:  # 如果禁止词语列表为None
            self.bad_words = []  # 初始化为空列表

        if self.logprobs is True:  # 如果logprobs设为True（布尔值）
            self.logprobs = 1  # 将其转换为整数1

        if self.prompt_logprobs is True:  # 如果prompt_logprobs设为True（布尔值）
            self.prompt_logprobs = 1  # 将其转换为整数1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:  # 如果有停止字符串且不包含在输出中
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1  # 设置文本缓冲区长度为最长停止字符串长度-1

        self._verify_args()  # 验证参数有效性

        if self.temperature < _SAMPLING_EPS:  # 如果温度低于采样精度阈值（视为零温度）
            # Zero temperature means greedy sampling.
            self.top_p = 1.0  # 贪心采样时重置top_p为1.0
            self.top_k = 0  # 贪心采样时重置top_k为0（禁用）
            self.min_p = 0.0  # 贪心采样时重置min_p为0.0（禁用）
            self._verify_greedy_sampling()  # 验证贪心采样的特殊约束

        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)  # 将停止token ID添加到全部停止token集合中

        if self.skip_reading_prefix_cache is None:  # 如果未显式设置跳过前缀缓存读取
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None  # 当需要prompt logprobs时跳过前缀缓存

    def _verify_args(self) -> None:
        """验证所有采样参数的有效性和范围。"""
        if not isinstance(self.n, int):  # 检查n是否为整数类型
            raise ValueError(f"n must be an int, but is of type {type(self.n)}")
        if self.n < 1:  # 检查n是否至少为1
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:  # 检查存在惩罚是否在[-2, 2]范围内
            raise ValueError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:  # 检查频率惩罚是否在[-2, 2]范围内
            raise ValueError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        if self.repetition_penalty <= 0.0:  # 检查重复惩罚是否大于0
            raise ValueError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        if self.temperature < 0.0:  # 检查温度是否非负
            raise VLLMValidationError(
                f"temperature must be non-negative, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if not 0.0 < self.top_p <= 1.0:  # 检查top_p是否在(0, 1]范围内
            raise VLLMValidationError(
                f"top_p must be in (0, 1], got {self.top_p}.",
                parameter="top_p",
                value=self.top_p,
            )
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:  # 检查top_k是否有效（允许-1表示禁用）
            raise ValueError(
                f"top_k must be 0 (disable), or at least 1, got {self.top_k}."
            )
        if not isinstance(self.top_k, int):  # 检查top_k是否为整数类型
            raise TypeError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )
        if not 0.0 <= self.min_p <= 1.0:  # 检查min_p是否在[0, 1]范围内
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:  # 检查max_tokens是否至少为1
            raise VLLMValidationError(
                f"max_tokens must be at least 1, got {self.max_tokens}.",
                parameter="max_tokens",
                value=self.max_tokens,
            )
        if self.min_tokens < 0:  # 检查min_tokens是否非负
            raise ValueError(
                f"min_tokens must be greater than or equal to 0, got {self.min_tokens}."
            )
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:  # 检查min_tokens不超过max_tokens
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:  # 检查logprobs是否有效
            raise VLLMValidationError(
                f"logprobs must be non-negative or -1, got {self.logprobs}.",
                parameter="logprobs",
                value=self.logprobs,
            )
        if (  # 检查prompt_logprobs是否有效
            self.prompt_logprobs is not None
            and self.prompt_logprobs != -1
            and self.prompt_logprobs < 0
        ):
            raise VLLMValidationError(
                f"prompt_logprobs must be non-negative or -1, got "
                f"{self.prompt_logprobs}.",
                parameter="prompt_logprobs",
                value=self.prompt_logprobs,
            )
        assert isinstance(self.stop_token_ids, list)  # 断言stop_token_ids是列表类型
        if not all(isinstance(st_id, int) for st_id in self.stop_token_ids):  # 检查所有停止token ID是否为整数
            raise ValueError(
                f"stop_token_ids must contain only integers, got {self.stop_token_ids}."
            )
        assert isinstance(self.stop, list)  # 断言stop是列表类型
        if any(not stop_str for stop_str in self.stop):  # 检查停止字符串列表中是否有空字符串
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:  # 检查使用停止字符串时是否启用了反分词
            raise ValueError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop."
            )

    def _verify_greedy_sampling(self) -> None:
        """验证贪心采样模式下的参数约束。"""
        if self.n > 1:  # 贪心采样时n必须为1
            raise ValueError(f"n must be 1 when using greedy sampling, got {self.n}.")

    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],  # 生成配置字典
        eos_token_id: int | None = None,  # EOS token ID
    ) -> None:
        """Update if there are non-default values from generation_config
        从生成配置中更新非默认值，处理EOS token和额外的停止token。
        """
        if not self.ignore_eos:  # 如果不忽略EOS token
            self._eos_token_id = eos_token_id  # 设置EOS token ID

        if eos_token_id is not None:  # 如果提供了EOS token ID
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(eos_token_id)  # 将EOS token添加到停止token集合（支持min_tokens处理）

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:  # 从生成配置获取EOS token ID
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)  # 统一转为集合格式
            if eos_token_id is not None:  # 如果有主EOS token ID
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(eos_token_id)  # 移除主EOS token，因为它已单独处理
            if eos_ids:  # 如果还有其他EOS token
                self._all_stop_token_ids.update(eos_ids)  # 更新停止token集合
                if not self.ignore_eos:  # 如果不忽略EOS
                    assert self.stop_token_ids is not None  # 断言stop_token_ids已初始化
                    eos_ids.update(self.stop_token_ids)  # 合并现有的停止token
                    self.stop_token_ids = list(eos_ids)  # 更新停止token列表

    def update_from_tokenizer(self, tokenizer: TokenizerLike) -> None:
        """从分词器更新禁止词语的token ID映射。"""
        if not self.bad_words:  # 如果没有禁止词语，直接返回
            return
        self._bad_words_token_ids = []  # 初始化禁止词语token ID列表
        for bad_word in self.bad_words:  # 遍历每个禁止词语
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:  # 分别处理有无前缀空格的情况
                prefix = " " if add_prefix_space else ""  # 根据标志决定是否添加前缀空格
                prompt = prefix + bad_word.lstrip()  # 构建带/不带前缀空格的提示
                prompt_token_ids = tokenizer.encode(  # 对提示进行分词编码
                    text=prompt, add_special_tokens=False  # 不添加特殊token
                )

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (  # 如果不加前缀空格，或加前缀空格产生了不同的token
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]  # 首token与前一个不同
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])  # 且长度相同
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)  # 添加到禁止词语token列表

        invalid_token_ids = [  # 检查是否有无效的token ID
            token_id
            for bad_words_token_ids in self._bad_words_token_ids  # 遍历所有禁止词语token序列
            for token_id in bad_words_token_ids  # 遍历每个token ID
            if token_id < 0 or token_id > tokenizer.max_token_id  # 检查是否超出词汇表范围
        ]
        if len(invalid_token_ids) > 0:  # 如果存在无效token ID
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )

    @cached_property
    def sampling_type(self) -> SamplingType:
        """根据当前参数确定采样类型（贪心/随机/带种子随机）。"""
        if self.temperature < _SAMPLING_EPS:  # 温度接近0时为贪心采样
            return SamplingType.GREEDY
        if self.seed is not None:  # 有种子时为带种子的随机采样
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM  # 否则为随机采样

    @property
    def eos_token_id(self) -> int | None:
        """获取EOS token ID。"""
        return self._eos_token_id  # 返回内部存储的EOS token ID

    @property
    def all_stop_token_ids(self) -> set[int]:
        """获取所有停止token ID的集合。"""
        return self._all_stop_token_ids  # 返回所有停止token ID的集合

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        """获取禁止词语的token ID序列列表（仅内部使用，不保证向后兼容）。"""
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids  # 返回禁止词语的token ID列表

    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy.
        克隆采样参数。如果skip_clone为True，使用浅拷贝代替深拷贝。
        """
        if self.skip_clone:  # 如果标记为可跳过克隆
            return copy.copy(self)  # 返回浅拷贝

        return copy.deepcopy(self)  # 否则返回深拷贝

    def verify(
        self,
        model_config: ModelConfig,  # 模型配置
        speculative_config: SpeculativeConfig | None,  # 推测解码配置
        structured_outputs_config: StructuredOutputsConfig | None,  # 结构化输出配置
        tokenizer: TokenizerLike | None,  # 分词器
    ) -> None:
        """验证采样参数与模型配置的兼容性。"""
        self._validate_logprobs(model_config)  # 验证对数概率参数
        self._validate_logit_bias(model_config)  # 验证logit偏置参数
        self._validate_logits_processors(model_config)  # 验证logits处理器参数
        self._validate_allowed_token_ids(tokenizer)  # 验证允许的token ID
        self._validate_spec_decode(speculative_config)  # 验证推测解码兼容性
        self._validate_structured_outputs(structured_outputs_config, tokenizer)  # 验证结构化输出参数

    def _validate_logprobs(self, model_config: ModelConfig) -> None:
        """验证对数概率参数不超过模型配置的最大值。"""
        max_logprobs = model_config.max_logprobs  # 获取最大允许的logprobs数量
        if max_logprobs == -1:  # 如果设为-1，表示允许全部词汇表
            max_logprobs = model_config.get_vocab_size()  # 使用词汇表大小作为最大值

        # Validate sample logprobs.
        if num_logprobs := self.logprobs:  # 如果设置了样本logprobs
            if num_logprobs == -1:  # 如果请求全部logprobs
                num_logprobs = model_config.get_vocab_size()  # 转换为词汇表大小
            if num_logprobs > max_logprobs:  # 如果超过最大允许值
                raise VLLMValidationError(
                    f"Requested sample logprobs of {num_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="logprobs",
                    value=num_logprobs,
                )

        # Validate prompt logprobs.
        if num_prompt_logprobs := self.prompt_logprobs:  # 如果设置了提示logprobs
            if num_prompt_logprobs == -1:  # 如果请求全部prompt logprobs
                num_prompt_logprobs = model_config.get_vocab_size()  # 转换为词汇表大小
            if num_prompt_logprobs > max_logprobs:  # 如果超过最大允许值
                raise VLLMValidationError(
                    f"Requested prompt logprobs of {num_prompt_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="prompt_logprobs",
                    value=num_prompt_logprobs,
                )

    def _validate_logit_bias(self, model_config: ModelConfig) -> None:
        """Validate logit_bias token IDs are within vocabulary range.
        验证logit偏置中的token ID是否在词汇表范围内。
        """
        if not self.logit_bias:  # 如果没有logit偏置，直接返回
            return

        vocab_size = model_config.get_vocab_size()  # 获取词汇表大小
        invalid_token_ids = [  # 找出超出词汇表范围的token ID
            token_id
            for token_id in self.logit_bias  # 遍历logit偏置中的所有token ID
            if token_id < 0 or token_id >= vocab_size  # 检查是否超出范围
        ]

        if invalid_token_ids:  # 如果存在无效的token ID
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in logit_bias contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="logit_bias",
                value=invalid_token_ids,
            )

    def _validate_logits_processors(self, model_config: ModelConfig) -> None:
        """验证自定义logits处理器参数的有效性。"""
        from vllm.v1.sample.logits_processor import (  # 延迟导入logits处理器验证函数
            validate_logits_processors_parameters,
        )

        validate_logits_processors_parameters(model_config.logits_processors, self)  # 执行验证

    def _validate_allowed_token_ids(self, tokenizer: TokenizerLike | None) -> None:
        """验证允许的token ID列表的有效性。"""
        allowed_token_ids = self.allowed_token_ids  # 获取允许的token ID列表
        if allowed_token_ids is None:  # 如果未设置，直接返回
            return

        if len(allowed_token_ids) == 0:  # 如果列表为空，抛出错误
            raise VLLMValidationError(
                "allowed_token_ids is not None and empty!",
                parameter="allowed_token_ids",
                value=allowed_token_ids,
            )

        if tokenizer is not None:  # 如果有分词器，验证token ID范围
            vocab_size = len(tokenizer)  # 获取词汇表大小
            invalid_token_ids = [  # 找出超出词汇表范围的token ID
                token_id
                for token_id in allowed_token_ids  # 遍历允许的token ID
                if token_id < 0 or token_id >= vocab_size  # 检查是否超出范围
            ]
            if invalid_token_ids:  # 如果存在无效的token ID
                raise VLLMValidationError(
                    "allowed_token_ids contains out-of-vocab token id!",
                    parameter="allowed_token_ids",
                    value=invalid_token_ids,
                )

    def _validate_spec_decode(
        self,
        speculative_config: SpeculativeConfig | None,  # 推测解码配置
    ) -> None:
        """验证采样参数与推测解码的兼容性。"""
        if speculative_config is None:  # 如果没有推测解码配置，直接返回
            return

        # Some sampling parameters are not yet compatible with spec decoding.
        if self.min_p > _SAMPLING_EPS or self.logit_bias:  # 检查不兼容的参数
            raise ValueError(
                "The min_p and logit_bias sampling parameters "
                "are not yet supported with speculative decoding."
            )

    def _validate_structured_outputs(
        self,
        structured_outputs_config: StructuredOutputsConfig | None,  # 结构化输出配置
        tokenizer: TokenizerLike | None,  # 分词器
    ) -> None:
        """验证结构化输出参数的有效性，并选择合适的后端。"""
        if structured_outputs_config is None or self.structured_outputs is None:  # 如果未配置结构化输出，直接返回
            return

        if tokenizer is None:  # 结构化输出需要分词器
            raise ValueError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        backend = structured_outputs_config.backend  # 获取配置的后端名称
        if _backend := self.structured_outputs._backend:  # 如果请求已指定后端
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (  # 检查后端不匹配的情况
                backend == "auto" and self.structured_outputs._backend_was_auto  # 允许auto模式之前自动选择的后端
            ):
                raise ValueError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend  # 设置后端为配置中的后端

        # Request content validation
        if (  # 验证choice约束不是空列表
            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise ValueError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (  # 验证grammar约束不是空字符串
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise ValueError("structured_outputs.grammar cannot be an empty string")

        from vllm.v1.structured_output.backend_guidance import (  # 延迟导入guidance后端验证函数
            has_guidance_unsupported_json_features,
            validate_guidance_grammar,
        )
        from vllm.v1.structured_output.backend_lm_format_enforcer import (  # 延迟导入lm-format-enforcer后端验证函数
            validate_structured_output_request_lm_format_enforcer,
        )
        from vllm.v1.structured_output.backend_outlines import (  # 延迟导入outlines后端验证函数
            validate_structured_output_request_outlines,
        )
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar  # 延迟导入xgrammar后端验证函数

        if backend.startswith("xgrammar"):  # 如果后端是xgrammar
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)  # 验证xgrammar语法
        elif backend.startswith("guidance"):  # 如果后端是guidance
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            if is_mistral_tokenizer(tokenizer):  # 检查是否为Mistral分词器（不支持guidance）
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'guidance' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_guidance_grammar(self, tokenizer=None)  # 验证guidance语法
        elif backend == "outlines":  # 如果后端是outlines
            # outlines backend
            validate_structured_output_request_outlines(self)  # 验证outlines请求
        elif backend == "lm-format-enforcer":  # 如果后端是lm-format-enforcer
            # lm format enforcer backend
            if is_mistral_tokenizer(tokenizer):  # 检查是否为Mistral分词器（不支持lm-format-enforcer）
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(self)  # 验证lm-format-enforcer请求
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:  # auto模式：首先尝试xgrammar
                validate_xgrammar_grammar(self)  # 验证xgrammar语法
                self.structured_outputs._backend = "xgrammar"  # 设置后端为xgrammar
            except ValueError:  # 如果xgrammar验证失败
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.

                # Check if schema has features unsupported by guidance
                so_params = self.structured_outputs  # 获取结构化输出参数
                skip_guidance = False  # 是否跳过guidance后端
                if so_params.json:  # 如果是JSON约束
                    if isinstance(so_params.json, str):  # 如果JSON是字符串格式
                        schema = json_mod.loads(so_params.json)  # 解析JSON字符串
                    else:
                        schema = so_params.json  # 直接使用字典格式
                    skip_guidance = has_guidance_unsupported_json_features(schema)  # 检查是否有guidance不支持的特性

                if is_mistral_tokenizer(tokenizer) or skip_guidance:  # 如果是Mistral分词器或需要跳过guidance
                    # Fall back to outlines if the tokenizer is Mistral
                    # or if schema contains features unsupported by guidance
                    validate_structured_output_request_outlines(self)  # 回退到outlines后端
                    self.structured_outputs._backend = "outlines"  # 设置后端为outlines
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(self, tokenizer=None)  # 默认回退到guidance后端
                    self.structured_outputs._backend = "guidance"  # 设置后端为guidance
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True  # 记录后端是自动选择的

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        self.structured_outputs.__post_init__()  # 运行结构化输出参数的后初始化验证

    def __repr__(self) -> str:
        """返回SamplingParams的字符串表示形式。"""
        return (
            f"SamplingParams(n={self.n}, "  # 输出数量
            f"presence_penalty={self.presence_penalty}, "  # 存在惩罚
            f"frequency_penalty={self.frequency_penalty}, "  # 频率惩罚
            f"repetition_penalty={self.repetition_penalty}, "  # 重复惩罚
            f"temperature={self.temperature}, "  # 温度
            f"top_p={self.top_p}, "  # Top-P值
            f"top_k={self.top_k}, "  # Top-K值
            f"min_p={self.min_p}, "  # Min-P值
            f"seed={self.seed}, "  # 随机种子
            f"stop={self.stop}, "  # 停止字符串
            f"stop_token_ids={self.stop_token_ids}, "  # 停止token ID
            f"bad_words={self.bad_words}, "  # 禁止词语
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "  # 是否包含停止字符串
            f"ignore_eos={self.ignore_eos}, "  # 是否忽略EOS
            f"max_tokens={self.max_tokens}, "  # 最大token数
            f"min_tokens={self.min_tokens}, "  # 最小token数
            f"logprobs={self.logprobs}, "  # 对数概率数量
            f"prompt_logprobs={self.prompt_logprobs}, "  # 提示对数概率
            f"skip_special_tokens={self.skip_special_tokens}, "  # 是否跳过特殊token
            "spaces_between_special_tokens="  # 特殊token间空格
            f"{self.spaces_between_special_tokens}, "
            f"structured_outputs={self.structured_outputs}, "  # 结构化输出参数
            f"extra_args={self.extra_args})"  # 额外参数
        )

    @staticmethod
    def for_sampler_warmup() -> "SamplingParams":
        """Set parameters to exercise all sampler logic.
        设置参数以测试所有采样器逻辑，用于采样器预热。
        """
        return SamplingParams(  # 创建包含各种采样参数的预热实例
            temperature=0.9,  # 设置温度为0.9
            top_p=0.9,  # 设置Top-P为0.9
            top_k=50,  # 设置Top-K为50
            min_p=0.1,  # 设置Min-P为0.1
            frequency_penalty=0.5,  # 设置频率惩罚为0.5
            presence_penalty=0.5,  # 设置存在惩罚为0.5
            repetition_penalty=1.2,  # 设置重复惩罚为1.2
            min_tokens=2,  # 设置最小token数为2
            logit_bias={0: -1.0, 1: 0.5},  # 设置logit偏置
            _bad_words_token_ids=[[0], [1, 2]],  # 设置禁止词语token ID
            logprobs=5,  # 设置返回5个对数概率
            prompt_logprobs=1,  # 设置返回1个提示对数概率
        )


class BeamSearchParams(
    msgspec.Struct,  # 继承msgspec结构体
    omit_defaults=True,  # type: ignore[call-arg]  # 序列化时省略默认值
    # required for @cached_property.
    dict=True,  # 启用字典转换支持
):  # type: ignore[call-arg]
    """Beam search parameters for text generation.
    文本生成的束搜索参数。
    """

    beam_width: int  # 束宽度，即同时保留的候选序列数量
    max_tokens: int  # 最大生成token数量
    ignore_eos: bool = False  # 是否忽略EOS token
    temperature: float = 0.0  # 温度参数，默认为0（贪心）
    length_penalty: float = 1.0  # 长度惩罚系数，用于调整不同长度序列的得分
    include_stop_str_in_output: bool = False  # 是否在输出中包含停止字符串
