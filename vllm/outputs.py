# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import MutableSequence  # 导入可变序列类型
from collections.abc import Sequence as GenericSequence  # 导入通用序列类型
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, Generic  # 导入类型提示工具

import numpy as np  # 导入numpy库
import torch  # 导入PyTorch库
from typing_extensions import TypeVar  # 导入扩展的TypeVar

from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.logprobs import PromptLogprobs, SampleLogprobs  # 导入日志概率类型
from vllm.lora.request import LoRARequest  # 导入LoRA请求类
from vllm.v1.metrics.stats import RequestStateStats  # 导入请求状态统计类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


@dataclass  # 数据类装饰器，自动生成__init__等方法
class CompletionOutput:
    """单个补全输出的数据类。

    存储一个请求中某个补全序列的输出结果，包括生成的文本、token ID、
    对数概率、完成原因等信息。

    Args:
        index: 该输出在请求中的索引位置。
        text: 生成的输出文本。
        token_ids: 生成输出文本对应的token ID序列。
        cumulative_logprob: 生成输出文本的累积对数概率。
        logprobs: 如果请求了logprobs，则返回每个位置上最高概率词的对数概率。
        finish_reason: 序列结束的原因。
        stop_reason: 导致补全停止的停止字符串或token id，
            如果因其他原因（包括遇到EOS token）完成则为None。
        lora_request: 用于生成输出的LoRA请求。
    """

    index: int  # 输出在请求中的索引
    text: str  # 生成的文本内容
    token_ids: GenericSequence[int]  # 生成文本对应的token ID列表
    cumulative_logprob: float | None  # 累积对数概率，可为None
    logprobs: SampleLogprobs | None  # 采样对数概率，可为None
    routed_experts: np.ndarray | None = None  # 路由专家信息 [seq_len,layer_num,topk]
    finish_reason: str | None = None  # 完成原因，如"stop"或"length"
    stop_reason: int | str | None = None  # 停止原因（停止字符串或token id）
    lora_request: LoRARequest | None = None  # 关联的LoRA请求

    def finished(self) -> bool:
        """判断该补全输出是否已完成。

        Returns:
            如果finish_reason不为None，则返回True，表示已完成。
        """
        return self.finish_reason is not None  # 通过finish_reason判断是否已完成

    def __repr__(self) -> str:
        """返回CompletionOutput的字符串表示形式。

        Returns:
            包含所有关键字段的格式化字符串。
        """
        return (
            f"CompletionOutput(index={self.index}, "  # 索引
            f"text={self.text!r}, "  # 文本（带引号）
            f"token_ids={self.token_ids}, "  # token ID序列
            f"routed_experts={self.routed_experts}, "  # 路由专家信息
            f"cumulative_logprob={self.cumulative_logprob}, "  # 累积对数概率
            f"logprobs={self.logprobs}, "  # 对数概率
            f"finish_reason={self.finish_reason}, "  # 完成原因
            f"stop_reason={self.stop_reason})"  # 停止原因
        )


@dataclass  # 数据类装饰器
class PoolingOutput:
    """单个池化输出的数据类。

    存储池化请求中提取的隐藏状态数据。

    Args:
        data: 提取的隐藏状态张量。
    """

    data: torch.Tensor  # 池化输出的张量数据

    def __repr__(self) -> str:
        """返回PoolingOutput的字符串表示形式。

        Returns:
            包含数据张量信息的格式化字符串。
        """
        return f"PoolingOutput(data={self.data})"  # 返回包含数据的字符串表示

    def __eq__(self, other: object) -> bool:
        """比较两个PoolingOutput对象是否相等。

        Args:
            other: 要比较的对象。

        Returns:
            如果other是同类实例且数据张量完全相同则返回True。
        """
        return isinstance(other, self.__class__) and bool(  # 检查类型并比较数据
            (self.data == other.data).all()  # 逐元素比较所有数据是否相同
        )


class RequestOutput:
    """LLM补全请求的输出数据类。

    封装一个完整请求的所有输出信息，包括提示信息、生成的补全输出、
    指标统计等。支持增量合并多个输出结果。

    Args:
        request_id: 请求的唯一ID。
        prompt: 请求的提示字符串。对于编码器/解码器模型，这是解码器输入的提示。
        prompt_token_ids: 提示的token ID列表。
            对于编码器/解码器模型，这是解码器输入的提示token id。
        prompt_logprobs: 每个提示token要返回的对数概率。
        outputs: 请求的输出序列列表。
        finished: 整个请求是否已完成。
        metrics: 与请求关联的指标。
        lora_request: 用于生成输出的LoRA请求。
        encoder_prompt: 编码器提示字符串，仅解码器模型为None。
        encoder_prompt_token_ids: 编码器提示的token ID列表，仅解码器模型为None。
        num_cached_tokens: 前缀缓存命中的token数量。
        kv_transfer_params: 远程K/V传输的参数。
    """

    def __init__(
        self,
        request_id: str,  # 请求的唯一标识符
        prompt: str | None,  # 提示文本，可为None
        prompt_token_ids: list[int] | None,  # 提示token ID列表，可为None
        prompt_logprobs: PromptLogprobs | None,  # 提示对数概率，可为None
        outputs: list[CompletionOutput],  # 补全输出列表
        finished: bool,  # 请求是否已完成
        metrics: RequestStateStats | None = None,  # 请求状态统计指标
        lora_request: LoRARequest | None = None,  # LoRA请求
        encoder_prompt: str | None = None,  # 编码器提示文本
        encoder_prompt_token_ids: list[int] | None = None,  # 编码器提示token ID
        num_cached_tokens: int | None = None,  # 缓存命中的token数量
        *,
        kv_transfer_params: dict[str, Any] | None = None,  # KV传输参数
        # Forward compatibility, code that uses args added in new release can
        # still run with older versions of vLLM without breaking.
        **kwargs: Any,  # 前向兼容的额外参数
    ) -> None:
        """初始化RequestOutput实例。

        Args:
            request_id: 请求的唯一标识符。
            prompt: 提示文本。
            prompt_token_ids: 提示的token ID列表。
            prompt_logprobs: 提示的对数概率。
            outputs: 补全输出列表。
            finished: 是否已完成。
            metrics: 请求状态统计。
            lora_request: LoRA请求。
            encoder_prompt: 编码器提示文本。
            encoder_prompt_token_ids: 编码器提示token ID列表。
            num_cached_tokens: 缓存命中的token数。
            kv_transfer_params: KV传输参数。
            **kwargs: 额外的前向兼容参数。
        """
        if kwargs:  # 如果有未知的额外参数
            logger.warning_once(  # 记录一次性警告
                "RequestOutput: Ignoring extra arguments: %s", str(kwargs)  # 忽略额外参数
            )
        self.request_id = request_id  # 设置请求ID
        self.prompt = prompt  # 设置提示文本
        self.prompt_token_ids = prompt_token_ids  # 设置提示token ID列表
        self.prompt_logprobs = prompt_logprobs  # 设置提示对数概率
        self.outputs = outputs  # 设置补全输出列表
        self.finished = finished  # 设置完成标志
        self.metrics = metrics  # 设置指标统计
        self.lora_request = lora_request  # 设置LoRA请求
        self.encoder_prompt = encoder_prompt  # 设置编码器提示
        self.encoder_prompt_token_ids = encoder_prompt_token_ids  # 设置编码器提示token ID
        self.num_cached_tokens = num_cached_tokens  # 设置缓存命中token数
        self.kv_transfer_params = kv_transfer_params  # 设置KV传输参数

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        """将后续的RequestOutput合并到当前实例中。

        支持两种模式：聚合模式（累积文本和token）和替换模式（直接替换输出）。

        Args:
            next_output: 要合并的下一个RequestOutput。
            aggregate: 如果为True，则累积合并；否则直接替换。
        """

        self.finished |= next_output.finished  # 更新完成状态（任一完成即完成）
        self.kv_transfer_params = next_output.kv_transfer_params  # 更新KV传输参数

        for next_completion in next_output.outputs:  # 遍历新输出的每个补全
            for i, completion in enumerate(self.outputs):  # 遍历当前输出的每个补全
                if completion.index == next_completion.index:  # 找到相同索引的补全
                    if aggregate:  # 聚合模式
                        # Merge outputs with same index
                        completion.text += next_completion.text  # 拼接文本
                        if not isinstance(completion.token_ids, MutableSequence):  # 如果token_ids不可变
                            completion.token_ids = list(completion.token_ids)  # 转换为可变列表
                        completion.token_ids.extend(next_completion.token_ids)  # 扩展token ID列表
                        if next_completion.logprobs:  # 如果新输出有对数概率
                            assert completion.logprobs is not None  # 断言当前也有对数概率
                            completion.logprobs.extend(next_completion.logprobs)  # type: ignore[arg-type]  # 扩展对数概率
                        completion.cumulative_logprob = (  # 更新累积对数概率
                            next_completion.cumulative_logprob
                        )
                        completion.finish_reason = next_completion.finish_reason  # 更新完成原因
                        completion.stop_reason = next_completion.stop_reason  # 更新停止原因
                    else:  # 替换模式
                        # Replace the output with the new one
                        self.outputs[i] = next_completion  # 直接替换该输出
                    break  # 找到匹配项后跳出内循环
            else:  # 如果没有找到匹配的索引
                self.outputs.append(next_completion)  # 添加为新输出

    def __repr__(self) -> str:
        """返回RequestOutput的字符串表示形式。

        Returns:
            包含所有关键字段的格式化字符串。
        """
        return (
            f"RequestOutput(request_id={self.request_id}, "  # 请求ID
            f"prompt={self.prompt!r}, "  # 提示文本
            f"prompt_token_ids={self.prompt_token_ids}, "  # 提示token ID
            f"encoder_prompt={self.encoder_prompt!r}, "  # 编码器提示
            f"encoder_prompt_token_ids={self.encoder_prompt_token_ids}, "  # 编码器提示token ID
            f"prompt_logprobs={self.prompt_logprobs}, "  # 提示对数概率
            f"outputs={self.outputs}, "  # 输出列表
            f"finished={self.finished}, "  # 完成状态
            f"metrics={self.metrics}, "  # 指标
            f"lora_request={self.lora_request}, "  # LoRA请求
            f"num_cached_tokens={self.num_cached_tokens})"  # 缓存token数
        )


# Sentinel to indicate request is finished, used with streaming inputs.
STREAM_FINISHED = RequestOutput(  # 流式输入的完成哨兵对象
    request_id="",  # 空的请求ID
    prompt=None,  # 无提示
    prompt_token_ids=None,  # 无提示token ID
    prompt_logprobs=None,  # 无提示对数概率
    outputs=[],  # 空输出列表
    finished=True,  # 标记为已完成
)

_O = TypeVar("_O", default=PoolingOutput)  # 定义泛型类型变量，默认为PoolingOutput


class PoolingRequestOutput(Generic[_O]):
    """池化请求的输出数据类。

    封装池化请求的完整输出结果，支持泛型以适配不同类型的池化输出
    （如嵌入、分类、评分等）。

    Args:
        request_id (str): 池化请求的唯一标识符。
        outputs (PoolingOutput): 给定输入的池化结果。
        prompt_token_ids (list[int]): 提示中使用的token ID列表。
        num_cached_tokens: 前缀缓存命中的token数量。
        finished (bool): 表示池化是否已完成的标志。
    """

    def __init__(
        self,
        request_id: str,  # 请求的唯一标识符
        outputs: _O,  # 池化输出结果
        prompt_token_ids: list[int],  # 提示token ID列表
        num_cached_tokens: int,  # 缓存命中的token数量
        finished: bool,  # 是否已完成
    ):
        """初始化PoolingRequestOutput实例。

        Args:
            request_id: 请求的唯一标识符。
            outputs: 池化输出结果。
            prompt_token_ids: 提示token ID列表。
            num_cached_tokens: 缓存命中的token数量。
            finished: 是否已完成。
        """
        self.request_id = request_id  # 设置请求ID
        self.prompt_token_ids = prompt_token_ids  # 设置提示token ID列表
        self.num_cached_tokens = num_cached_tokens  # 设置缓存命中token数
        self.finished = finished  # 设置完成标志
        self.outputs = outputs  # 设置池化输出

    def __repr__(self):
        """返回PoolingRequestOutput的字符串表示形式。

        Returns:
            包含所有关键字段的格式化字符串。
        """
        return (
            f"{type(self).__name__}(request_id={self.request_id!r}, "  # 类名和请求ID
            f"outputs={self.outputs!r}, "  # 输出
            f"prompt_token_ids={self.prompt_token_ids}, "  # 提示token ID
            f"num_cached_tokens={self.num_cached_tokens}, "  # 缓存token数
            f"finished={self.finished})"  # 完成状态
        )


@dataclass  # 数据类装饰器
class EmbeddingOutput:
    """单个嵌入输出的数据类。

    存储嵌入向量结果，向量长度取决于模型的隐藏维度。

    Args:
        embedding: 嵌入向量，为浮点数列表。其长度取决于模型的隐藏维度。
    """

    embedding: list[float]  # 嵌入向量（浮点数列表）

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        """从基础PoolingOutput创建EmbeddingOutput。

        Args:
            pooling_output: 基础的池化输出对象。

        Returns:
            由池化数据转换而来的EmbeddingOutput实例。

        Raises:
            ValueError: 如果pooled_data不是一维向量。
        """
        pooled_data = pooling_output.data  # 获取池化数据
        if pooled_data.ndim != 1:  # 检查是否为一维向量
            raise ValueError("pooled_data should be a 1-D embedding vector")  # 不是一维则报错

        return EmbeddingOutput(pooled_data.tolist())  # 将张量转换为列表并创建实例

    @property
    def hidden_size(self) -> int:
        """获取嵌入向量的隐藏维度大小。

        Returns:
            嵌入向量的长度（即隐藏维度大小）。
        """
        return len(self.embedding)  # 返回嵌入向量长度

    def __repr__(self) -> str:
        """返回EmbeddingOutput的字符串表示形式。

        Returns:
            包含隐藏维度大小的格式化字符串。
        """
        return f"EmbeddingOutput(hidden_size={self.hidden_size})"  # 显示隐藏维度大小


class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    """嵌入请求的输出类，继承自PoolingRequestOutput。

    将通用的池化请求输出特化为嵌入类型的输出。
    """

    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        """从基础PoolingRequestOutput创建EmbeddingRequestOutput。

        Args:
            request_output: 基础的池化请求输出对象。

        Returns:
            转换后的EmbeddingRequestOutput实例。
        """
        return EmbeddingRequestOutput(  # 创建嵌入请求输出
            request_id=request_output.request_id,  # 请求ID
            outputs=EmbeddingOutput.from_base(request_output.outputs),  # 转换嵌入输出
            prompt_token_ids=request_output.prompt_token_ids,  # 提示token ID
            num_cached_tokens=request_output.num_cached_tokens,  # 缓存token数
            finished=request_output.finished,  # 完成状态
        )


@dataclass  # 数据类装饰器
class ClassificationOutput:
    """单个分类输出的数据类。

    存储分类结果的概率向量，向量长度取决于分类类别数量。

    Args:
        probs: 概率向量，为浮点数列表。其长度取决于分类类别数量。
    """

    probs: list[float]  # 分类概率向量（浮点数列表）

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        """从基础PoolingOutput创建ClassificationOutput。

        Args:
            pooling_output: 基础的池化输出对象。

        Returns:
            由池化数据转换而来的ClassificationOutput实例。

        Raises:
            ValueError: 如果pooled_data不是一维概率向量。
        """
        # pooling_output shape: (num_classes)
        pooled_data = pooling_output.data  # 获取池化数据
        if pooled_data.ndim != 1:  # 检查是否为一维向量
            raise ValueError("pooled_data should be a 1-D probability vector")  # 不是一维则报错

        return ClassificationOutput(pooled_data.tolist())  # 将张量转换为列表并创建实例

    @property
    def num_classes(self) -> int:
        """获取分类类别数量。

        Returns:
            概率向量的长度（即分类类别数量）。
        """
        return len(self.probs)  # 返回概率向量长度即类别数

    def __repr__(self) -> str:
        """返回ClassificationOutput的字符串表示形式。

        Returns:
            包含类别数量的格式化字符串。
        """
        return f"ClassificationOutput(num_classes={self.num_classes})"  # 显示类别数


class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    """分类请求的输出类，继承自PoolingRequestOutput。

    将通用的池化请求输出特化为分类类型的输出。
    """

    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        """从基础PoolingRequestOutput创建ClassificationRequestOutput。

        Args:
            request_output: 基础的池化请求输出对象。

        Returns:
            转换后的ClassificationRequestOutput实例。
        """
        return ClassificationRequestOutput(  # 创建分类请求输出
            request_id=request_output.request_id,  # 请求ID
            outputs=ClassificationOutput.from_base(request_output.outputs),  # 转换分类输出
            prompt_token_ids=request_output.prompt_token_ids,  # 提示token ID
            num_cached_tokens=request_output.num_cached_tokens,  # 缓存token数
            finished=request_output.finished,  # 完成状态
        )


@dataclass  # 数据类装饰器
class ScoringOutput:
    """单个评分输出的数据类。

    存储相似度评分结果，为单个标量值。

    Args:
        score: 相似度评分，为标量值。
    """

    score: float  # 相似度评分（标量值）

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        """从基础PoolingOutput创建ScoringOutput。

        Args:
            pooling_output: 基础的池化输出对象。

        Returns:
            由池化数据转换而来的ScoringOutput实例。

        Raises:
            ValueError: 如果pooled_data不是标量值。
        """
        # pooling_output shape:
        #   classify task: (num_classes) num_classes == 1  # 分类任务：(num_classes) 其中num_classes == 1
        #   embed task: a scalar value  # 嵌入任务：标量值
        pooled_data = pooling_output.data.squeeze()  # 压缩维度，去除大小为1的维度
        if pooled_data.ndim != 0:  # 检查是否为标量（0维）
            raise ValueError("pooled_data should be a scalar score")  # 不是标量则报错

        return ScoringOutput(pooled_data.item())  # 将张量标量转换为Python数值并创建实例

    def __repr__(self) -> str:
        """返回ScoringOutput的字符串表示形式。

        Returns:
            包含评分值的格式化字符串。
        """
        return f"ScoringOutput(score={self.score})"  # 显示评分值


class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    """评分请求的输出类，继承自PoolingRequestOutput。

    将通用的池化请求输出特化为评分类型的输出。
    """

    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        """从基础PoolingRequestOutput创建ScoringRequestOutput。

        Args:
            request_output: 基础的池化请求输出对象。

        Returns:
            转换后的ScoringRequestOutput实例。
        """
        return ScoringRequestOutput(  # 创建评分请求输出
            request_id=request_output.request_id,  # 请求ID
            outputs=ScoringOutput.from_base(request_output.outputs),  # 转换评分输出
            prompt_token_ids=request_output.prompt_token_ids,  # 提示token ID
            num_cached_tokens=request_output.num_cached_tokens,  # 缓存token数
            finished=request_output.finished,  # 完成状态
        )
