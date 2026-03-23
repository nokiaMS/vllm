# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod  # 从abc模块导入抽象基类ABC和抽象方法装饰器abstractmethod

import tokenizers  # 导入tokenizers库，用于高性能分词和解码
from packaging import version  # 从packaging库导入version模块，用于版本号比较
from tokenizers import Tokenizer  # 从tokenizers库导入Tokenizer类，底层分词器实现
from tokenizers.decoders import DecodeStream  # 从tokenizers.decoders导入DecodeStream类，用于流式增量解码
from transformers import PreTrainedTokenizerFast  # 从transformers库导入PreTrainedTokenizerFast，HuggingFace快速分词器基类

from vllm.logger import init_logger  # 从vllm日志模块导入日志初始化函数
from vllm.tokenizers import TokenizerLike  # 从vllm分词器模块导入TokenizerLike类型别名，表示所有支持的分词器类型
from vllm.tokenizers.detokenizer_utils import (  # 从vllm解码工具模块导入辅助函数
    convert_prompt_ids_to_tokens,  # 将提示词token ID列表转换为token字符串列表
    detokenize_incrementally,  # 增量解码函数，逐token将ID转换回文本
)  # 导入语句结束
from vllm.utils import length_from_prompt_token_ids_or_embeds  # 从vllm工具模块导入函数，根据prompt token ID或嵌入向量计算长度
from vllm.v1.engine import EngineCoreRequest  # 从v1引擎模块导入EngineCoreRequest，表示引擎核心请求对象

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
# 仅tokenizers >= 0.22.0版本支持带有原生预填充（ids参数）的DecodeStream，用于FastIncrementalDetokenizer
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")  # 比较当前tokenizers版本是否>=0.22.0，决定是否使用快速解码器

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
# 来自tokenizers Rust源码的错误字符串，用于识别无效前缀解码错误
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"  # 无效前缀错误消息常量，用于异常匹配


# [中文注释] 增量 detokenizer 基类（无 tokenizer 时的空实现）。
#   子类 FastIncrementalDetokenizer 使用 tokenizers 库的 DecodeStream（高性能）
#   子类 SlowIncrementalDetokenizer 使用 Python 逐 token 解码（兼容旧版 tokenizer）
#   工厂方法 from_new_request 根据 tokenizer 类型自动选择实现
class IncrementalDetokenizer:  # 增量解码器基类定义
    """增量解码器基类。

    当没有分词器可用时，作为空操作实现使用。
    它维护一个token_ids列表，但不执行实际的解码操作。
    子类通过重写update()和get_next_output_text()方法来实现真正的解码逻辑。
    工厂方法from_new_request()根据分词器类型自动选择合适的子类实例。
    """

    def __init__(self):  # 构造函数定义
        """初始化增量解码器基类，创建空的token ID列表。"""
        self.token_ids: list[int] = []  # 存储所有输出token ID的列表

    @property  # 属性装饰器
    def output_token_ids(self) -> list[int]:  # 输出token ID属性方法定义
        """属性方法：返回输出的token ID列表。"""
        return self.token_ids  # 直接返回token_ids列表

    def num_output_tokens(self) -> int:  # 获取输出token数量的方法定义
        """返回输出token的数量。"""
        return len(self.token_ids)  # 返回token_ids列表的长度

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:  # 更新方法定义，接收新token并返回匹配的停止字符串
        """更新解码器状态，添加新的token ID。

        Args:
            new_token_ids: 新生成的token ID列表
            stop_terminated: 是否因停止条件而终止

        Returns:
            匹配到的停止字符串，如果没有匹配则返回None
        """
        self.token_ids.extend(new_token_ids)  # 将新的token ID添加到列表末尾
        return None  # 基类不执行停止字符串检查，直接返回None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:  # 获取下一段输出文本的方法定义
        """获取下一段输出文本。

        Args:
            finished: 序列是否已完成生成
            delta: 是否只返回增量文本（自上次调用以来的新文本）

        Returns:
            输出文本字符串，基类始终返回空字符串
        """
        return ""  # 基类不执行解码，返回空字符串

    @classmethod  # 类方法装饰器
    def from_new_request(  # 工厂方法定义
        cls,  # 类本身作为第一个参数
        tokenizer: TokenizerLike | None,  # 分词器对象，可以为None
        request: EngineCoreRequest,  # 引擎核心请求对象
    ) -> "IncrementalDetokenizer":  # 返回类型为IncrementalDetokenizer实例
        """工厂方法：根据分词器类型和版本，为新请求创建合适的增量解码器实例。

        Args:
            tokenizer: 分词器实例，如果为None则跳过解码
            request: 引擎核心请求对象，包含采样参数和提示词信息

        Returns:
            适当类型的IncrementalDetokenizer实例
        """
        assert request.sampling_params is not None  # 断言请求必须包含采样参数

        if tokenizer is None:  # 如果没有分词器
            # No tokenizer => skipping detokenization.
            # 没有分词器 => 跳过解码，返回基类空实现
            return IncrementalDetokenizer()  # 返回基类实例（空操作）

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):  # 如果支持快速解码且分词器是快速分词器类型
            # Fast tokenizer => use tokenizers library DecodeStream.
            # 快速分词器 => 使用tokenizers库的DecodeStream进行高性能解码
            return FastIncrementalDetokenizer(tokenizer, request)  # 返回快速增量解码器实例

        # Fall back to slow python-based incremental detokenization.
        # 回退到基于Python的慢速增量解码
        return SlowIncrementalDetokenizer(tokenizer, request)  # 返回慢速增量解码器实例


# [中文注释] 增量 detokenizer 的真正实现基类。核心流程：
#   update() — 逐 token 调用 decode_next() 解码，然后检查 stop strings
#   get_next_output_text() — 返回增量/全量文本，stop_buffer_length 防止部分匹配的停止词被提前发送
class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):  # 增量解码器抽象基类定义，继承IncrementalDetokenizer和ABC
    """增量解码器的抽象基类，提供完整的解码和停止字符串检查逻辑。

    继承自IncrementalDetokenizer和ABC（抽象基类）。
    实现了update()方法的核心逻辑：逐token调用decode_next()解码，然后检查停止字符串。
    实现了get_next_output_text()方法：支持增量和全量文本输出，使用stop_buffer_length
    防止部分匹配的停止词被提前发送给客户端。
    子类只需实现decode_next()抽象方法即可。
    """

    def __init__(self, request: EngineCoreRequest):  # 构造函数定义，接收引擎核心请求对象
        """初始化基础增量解码器。

        Args:
            request: 引擎核心请求对象，包含采样参数和停止条件
        """
        super().__init__()  # 调用父类IncrementalDetokenizer的初始化方法

        # Stop strings
        # 停止字符串相关设置
        params = request.sampling_params  # 获取采样参数
        assert params is not None  # 断言采样参数不为None
        if params.stop is None:  # 如果没有设置停止字符串
            self.stop = []  # 初始化为空列表
        elif isinstance(params.stop, str):  # 如果停止字符串是单个字符串
            self.stop = [params.stop]  # 将其包装为列表
        else:  # 如果停止字符串已经是列表
            self.stop = params.stop  # 直接使用
        self.min_tokens = params.min_tokens  # 最小生成token数，在此之前不检查停止条件
        self.include_stop_str_in_output = params.include_stop_str_in_output  # 是否在输出中包含停止字符串

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        # 当需要从流式输出中排除停止字符串时，需要缓冲的字符数
        if self.stop and not self.include_stop_str_in_output:  # 如果有停止字符串且不包含在输出中
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1  # 缓冲长度为最长停止字符串长度减1，防止部分匹配被提前发送
        else:  # 如果没有停止字符串或者需要包含在输出中
            self.stop_buffer_length = 0  # 不需要缓冲
        self._last_output_text_offset: int = 0  # 上次输出文本的偏移量，用于增量输出

        # Generation data
        # 生成数据
        self.output_text = ""  # 累积的输出文本字符串

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:  # 更新方法定义，处理新token并检查停止条件
        """更新请求状态：增量解码新token并检查停止条件。

        处理流程：
            1) 对新token ID进行增量解码
            2) 评估停止条件

        Args:
            new_token_ids: 新生成的token ID列表
            stop_terminated: 是否因停止条件而终止

        Returns:
            匹配到的停止字符串，如果没有匹配则返回None
        """
        if not new_token_ids:  # 如果没有新的token ID
            # Skip detokenization if no new token ids.
            # 如果没有新的token ID则跳过解码
            return None  # 直接返回None

        if stop_terminated and not self.include_stop_str_in_output:  # 如果因停止条件终止且不需要在输出中包含停止字符串
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            # 如果因停止条件终止，根据include_stop_str_in_output参数排除最后一个token的解码
            skipped_stop_token_id = new_token_ids[-1]  # 保存被跳过的停止token ID
            new_token_ids = new_token_ids[:-1]  # 从解码列表中移除最后一个token
        else:  # 如果不是停止终止或者需要包含停止字符串
            skipped_stop_token_id = None  # 没有被跳过的token

        # 1) Detokenize the new token ids incrementally.
        # 1) 对新token ID进行增量解码
        stop_check_offset = len(self.output_text)  # 记录停止检查的起始偏移量
        for new_token_id in new_token_ids:  # 遍历每个新的token ID
            self.token_ids.append(new_token_id)  # 将token ID添加到列表
            self.output_text += self.decode_next(new_token_id)  # 增量解码当前token并追加到输出文本
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            # 支持min_tokens参数，参见PR #22014
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:  # 如果设置了最小token数且当前输出token数未超过最小值
                stop_check_offset = len(self.output_text)  # 更新停止检查偏移量，跳过对这些token的停止检查

        if skipped_stop_token_id is not None:  # 如果有被跳过的停止token ID
            # Cleanup after skipping detokenization.
            # 跳过解码后的清理：将停止token ID添加到列表中（但不解码为文本）
            self.token_ids.append(skipped_stop_token_id)  # 将停止token ID添加到token列表

        # 2) Evaluate stop strings.
        # 2) 评估停止字符串匹配
        stop_string = None  # 初始化停止字符串匹配结果为None
        if self.stop and self.num_output_tokens() > self.min_tokens:  # 如果有停止字符串且已超过最小token数
            stop = check_stop_strings(  # 调用停止字符串检查函数
                output_text=self.output_text,  # 传入当前输出文本
                new_char_count=len(self.output_text) - stop_check_offset,  # 传入新增字符数
                stop=self.stop,  # 传入停止字符串列表
                include_in_output=self.include_stop_str_in_output,  # 传入是否在输出中包含停止字符串
            )  # check_stop_strings调用结束
            if stop is not None:  # 如果匹配到了停止字符串
                stop_string, truncate_to = stop  # 解包匹配结果：停止字符串和截断位置
                if truncate_to != -1:  # 如果需要截断（-1表示不需要截断）
                    self.output_text = self.output_text[:truncate_to]  # 截断输出文本到指定位置

        return stop_string  # 返回匹配到的停止字符串或None

    @abstractmethod  # 抽象方法装饰器，子类必须实现此方法
    def decode_next(self, next_token_id: int) -> str:  # 抽象方法定义：解码下一个token
        """抽象方法：解码下一个token ID为文本字符串。

        子类必须实现此方法以提供具体的解码逻辑。

        Args:
            next_token_id: 要解码的token ID

        Returns:
            解码后的文本字符串

        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError  # 抽象方法，子类必须实现

    def get_next_output_text(self, finished: bool, delta: bool) -> str:  # 获取下一段输出文本的方法定义
        """获取下一段输出文本，支持增量和全量模式。

        如果delta为True，只返回自上次调用以来的新文本。
        使用stop_buffer_length缓冲机制防止停止字符串的部分匹配被提前发送。

        Args:
            finished: 序列是否已完成生成
            delta: 是否只返回增量文本

        Returns:
            输出文本字符串
        """

        # We return the full output text if the sequence is finished.
        # 如果序列已完成则返回完整输出文本
        buffer_length = 0 if finished else self.stop_buffer_length  # 已完成时不需要缓冲，否则使用停止字符串缓冲长度
        if not delta:  # 如果不是增量模式（全量模式）
            if not buffer_length:  # 如果不需要缓冲
                return self.output_text  # 返回完整输出文本
            return self.output_text[:-buffer_length]  # 返回去掉缓冲区的输出文本

        length = len(self.output_text) - buffer_length  # 计算可输出的文本长度（减去缓冲区）
        last_offset = self._last_output_text_offset  # 获取上次输出的偏移量
        if last_offset < length:  # 如果有新的文本可以输出
            self._last_output_text_offset = length  # 更新偏移量到当前位置
            return self.output_text[last_offset:length]  # 返回增量文本（从上次偏移到当前位置）
        return ""  # 没有新文本，返回空字符串


# [中文注释] 快速 detokenizer：基于 tokenizers 库的 DecodeStream（需要 tokenizers>=0.22.0）
#   利用 native prefill 将 prompt token 预加载到 stream 中，后续只需 step() 逐 token 解码
#   处理 spaces_between_special_tokens 和特殊 token 的空格抑制
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):  # 快速增量解码器类定义，继承BaseIncrementalDetokenizer
    """快速增量解码器，基于tokenizers库的DecodeStream实现。

    需要tokenizers >= 0.22.0版本支持。
    利用DecodeStream的原生预填充功能将prompt token预加载到解码流中，
    后续只需调用step()方法逐token解码，性能远优于Python实现。
    同时处理特殊token之间的空格抑制逻辑。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, request: EngineCoreRequest):  # 构造函数定义，接收快速分词器和请求对象
        """初始化快速增量解码器。

        Args:
            tokenizer: HuggingFace快速分词器实例
            request: 引擎核心请求对象
        """
        super().__init__(request)  # 调用父类BaseIncrementalDetokenizer的初始化方法

        sampling_params = request.sampling_params  # 获取采样参数
        assert sampling_params is not None  # 断言采样参数不为None

        self.request_id = request.request_id  # 保存请求ID，用于日志记录
        self.skip_special_tokens = sampling_params.skip_special_tokens  # 是否跳过特殊token的解码

        self.tokenizer: Tokenizer = tokenizer._tokenizer  # 获取底层的tokenizers.Tokenizer对象（Rust实现）

        # Use native prefill to prime the decode stream with prompt tokens.
        # 使用原生预填充功能将prompt token预加载到解码流中
        self.stream = DecodeStream(  # 创建DecodeStream解码流实例
            ids=request.prompt_token_ids,  # 传入prompt token ID列表作为预填充
            skip_special_tokens=self.skip_special_tokens,  # 设置是否跳过特殊token
        )  # DecodeStream构造结束

        self.spaces_between_special_tokens = (  # 设置特殊token之间是否需要空格
            sampling_params.skip_special_tokens  # 如果跳过特殊token则不需要处理空格
            or sampling_params.spaces_between_special_tokens  # 或者明确指定需要空格
        )  # 空格设置赋值结束

        if not self.spaces_between_special_tokens:  # 如果不需要在特殊token之间添加空格（需要抑制空格）
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            # 存储已添加token的ID字典，以便我们可以抑制它们之间的空格
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)  # 尝试获取缓存的已添加token ID字典
            if added_token_ids is None:  # 如果还没有缓存
                self.tokenizer.added_token_ids = added_token_ids = {  # 创建并缓存token ID到token内容的映射字典
                    tid: tok.content  # 键为token ID，值为token的文本内容
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()  # 从分词器获取所有已添加token的解码映射
                }  # 字典推导式结束

            if added_token_ids:  # 如果存在已添加的token
                self.last_special = False  # 初始化标志：上一个token是否为特殊token
                self.added_token_ids = added_token_ids  # 保存已添加token ID字典的引用
            else:  # 如果没有已添加的token
                # No added tokens.
                # 没有已添加的token，无需抑制空格
                self.spaces_between_special_tokens = True  # 设置为True以跳过空格抑制逻辑

    def decode_next(self, next_token_id: int) -> str:  # 解码下一个token的方法定义
        """解码下一个token ID为文本字符串。

        使用DecodeStream的step方法进行解码，并处理特殊token之间的空格抑制。

        Args:
            next_token_id: 要解码的token ID

        Returns:
            解码后的文本字符串
        """
        token = self._protected_step(next_token_id)  # 调用受保护的step方法解码token（带异常处理）

        if not self.spaces_between_special_tokens:  # 如果需要抑制特殊token之间的空格
            special_token = self.added_token_ids.get(next_token_id)  # 查找当前token是否为已添加的特殊token
            is_special = special_token is not None  # 判断当前token是否为特殊token
            if is_special and self.last_special:  # 如果当前和上一个都是特殊token
                # Return raw token string without any prefixed spaces.
                # 返回原始token字符串，不带任何前缀空格
                token = special_token  # 使用原始token内容替换解码结果，避免多余空格
            self.last_special = is_special  # 更新上一个token是否为特殊token的标志

        return token or ""  # 返回解码结果，如果为None则返回空字符串

    def _protected_step(self, next_token_id: int) -> str | None:  # 受保护的step方法定义，带异常处理
        """受保护的step方法，带有异常处理的token解码。

        处理以下异常情况：
        - OverflowError/TypeError: 处理罕见的溢出错误
        - 无效前缀错误: 重置解码流后重试

        Args:
            next_token_id: 要解码的token ID

        Returns:
            解码后的文本字符串，异常时可能返回None
        """
        try:  # 尝试执行解码
            token = self.stream.step(self.tokenizer, next_token_id)  # 调用DecodeStream的step方法解码单个token
        except (OverflowError, TypeError):  # 捕获溢出或类型错误异常
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            # 处理罕见的溢出错误，仍在诊断中。参见issue #21951
            logger.exception("Encountered invalid token id: %r", next_token_id)  # 记录异常日志，包含无效的token ID
            token = None  # 返回None表示解码失败
        except Exception as e:  # 捕获其他所有异常
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):  # 如果不是无效前缀错误
                raise e  # 重新抛出异常
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            # 从边缘情况恢复：分词器可能产生非单调的无效UTF-8输出，
            # 破坏tokenizers DecodeStream的内部状态。参见issue #17448
            logger.warning(  # 记录警告日志
                "Encountered invalid prefix detokenization error"  # 遇到无效前缀解码错误
                " for request %s, resetting decode stream.",  # 正在为请求重置解码流
                self.request_id,  # 传入请求ID
            )  # logger.warning调用结束
            self.stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)  # 创建新的DecodeStream实例重置解码流（不带预填充）
            token = self.stream.step(self.tokenizer, next_token_id)  # 使用新的解码流重试解码
        return token  # 返回解码结果


# [中文注释] 慢速 detokenizer：兼容所有 tokenizer 类型，使用 Python 实现的逐 token 增量解码
#   维护 tokens、prefix_offset、read_offset 三个状态用于增量解码
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):  # 慢速增量解码器类定义，继承BaseIncrementalDetokenizer
    """慢速增量解码器，使用Python实现的逐token增量解码。

    兼容所有类型的分词器（不仅限于PreTrainedTokenizerFast）。
    维护tokens列表、prefix_offset和read_offset三个状态来实现增量解码。
    性能低于FastIncrementalDetokenizer，但兼容性更好。
    """

    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):  # 构造函数定义，接收分词器和请求对象
        """初始化慢速增量解码器。

        Args:
            tokenizer: 任意支持的分词器实例
            request: 引擎核心请求对象
        """
        super().__init__(request)  # 调用父类BaseIncrementalDetokenizer的初始化方法

        self.tokenizer = tokenizer  # 保存分词器引用
        params = request.sampling_params  # 获取采样参数
        assert params is not None  # 断言采样参数不为None

        self.prompt_len = length_from_prompt_token_ids_or_embeds(  # 计算提示词长度
            request.prompt_token_ids, request.prompt_embeds  # 从token ID列表或嵌入向量计算
        )

        # Metadata for incremental detokenization.
        # 增量解码所需的元数据
        if request.prompt_token_ids is not None:  # 如果有提示词token ID
            self.tokens, self.prefix_offset, self.read_offset = (  # 初始化解码状态
                convert_prompt_ids_to_tokens(  # 将提示词token ID转换为token字符串并获取初始偏移量
                    tokenizer=tokenizer,  # 传入分词器
                    prompt_ids=request.prompt_token_ids,  # 传入提示词token ID列表
                    skip_special_tokens=params.skip_special_tokens,  # 传入是否跳过特殊token
                )  # convert_prompt_ids_to_tokens调用结束
            )  # 解构赋值结束
        else:  # 如果没有提示词token ID（使用嵌入向量的请求）
            # Prompt embedding requests cannot be detokenized, in general.
            # 提示词嵌入请求通常无法被解码
            self.tokens = [""] * self.prompt_len  # 用空字符串占位，长度等于提示词长度
            self.prefix_offset = 0  # 前缀偏移量初始化为0
            self.read_offset = 0  # 读取偏移量初始化为0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)  # 将提示词token ID添加到token_ids列表，如果没有则用0占位

        self.skip_special_tokens = params.skip_special_tokens  # 保存是否跳过特殊token的设置
        self.spaces_between_special_tokens = params.spaces_between_special_tokens  # 保存特殊token之间是否添加空格的设置

    @property  # 属性装饰器
    def output_token_ids(self) -> list[int]:  # 输出token ID属性方法定义（覆盖父类）
        """属性方法：返回输出的token ID列表（不包含提示词部分）。"""
        if self.prompt_len:  # 如果有提示词
            return self.token_ids[self.prompt_len :]  # 返回去掉提示词前缀后的token ID列表
        return self.token_ids  # 没有提示词则返回全部token ID

    def num_output_tokens(self) -> int:  # 获取输出token数量的方法定义（覆盖父类）
        """返回输出token的数量（不包含提示词部分）。"""
        return len(self.token_ids) - self.prompt_len  # 总token数减去提示词长度

    def decode_next(self, next_token_id: int) -> str:  # 解码下一个token的方法定义（实现抽象方法）
        """解码下一个token ID为文本字符串。

        使用Python实现的增量解码函数，维护内部状态以实现正确的解码。

        Args:
            next_token_id: 要解码的token ID

        Returns:
            解码后的文本字符串
        """
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(  # 调用增量解码函数
            tokenizer=self.tokenizer,  # 传入分词器
            all_input_ids=self.token_ids,  # 传入所有token ID（包含提示词和已生成的）
            prev_tokens=self.tokens,  # 传入之前解码的token字符串列表
            prefix_offset=self.prefix_offset,  # 传入当前前缀偏移量
            read_offset=self.read_offset,  # 传入当前读取偏移量
            skip_special_tokens=self.skip_special_tokens,  # 传入是否跳过特殊token
            spaces_between_special_tokens=self.spaces_between_special_tokens,  # 传入特殊token之间是否添加空格
        )  # detokenize_incrementally调用结束

        self.tokens.extend(new_tokens)  # 将新解码的token字符串添加到列表
        self.prefix_offset = prefix_offset  # 更新前缀偏移量
        self.read_offset = read_offset  # 更新读取偏移量

        return decoded_text  # 返回解码后的文本


# [中文注释] 检查输出文本中是否匹配到 stop strings。
#   只在新增字符范围内搜索（避免重复搜索已检查过的文本）
#   返回 (匹配的 stop_string, 截断位置) 或 None
def check_stop_strings(  # 检查停止字符串的函数定义
    output_text: str,  # 当前完整的输出文本
    new_char_count: int,  # 自上次检查以来新增的字符数
    stop: list[str],  # 停止字符串列表
    include_in_output: bool,  # 是否在输出中包含匹配到的停止字符串
) -> tuple[str, int] | None:  # 返回类型为(停止字符串, 截断位置)元组或None
    """检查是否匹配到任何停止字符串，并相应地截断序列输出文本。

    只在新增字符范围内搜索以避免重复检查已检查过的文本。

    Args:
        output_text: 当前完整的输出文本
        new_char_count: 新增的字符数量
        stop: 停止字符串列表
        include_in_output: 是否在输出中包含停止字符串

    Returns:
        如果匹配到则返回元组(stop_string, offset)，否则返回None。
        其中stop_string是匹配到的停止字符串，offset是输出文本应截断到的长度，
        -1表示不需要截断。
    """
    if not new_char_count or not stop:  # 如果没有新字符或没有停止字符串
        return None  # 直接返回None

    for stop_str in stop:  # 遍历每个停止字符串
        stop_string_len = len(stop_str)  # 获取当前停止字符串的长度
        # Avoid searching already-searched text.
        # 避免搜索已经搜索过的文本，只在可能包含新匹配的范围内搜索
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)  # 在限定范围内搜索停止字符串，计算搜索起始位置以覆盖跨越新旧文本边界的匹配
        if stop_index == -1:  # 如果没有找到匹配
            continue  # 继续检查下一个停止字符串

        if include_in_output:  # 如果需要在输出中包含停止字符串
            # Truncate to end of stop string.
            # 截断到停止字符串的末尾
            stop_index += stop_string_len  # 将索引移动到停止字符串末尾
            if stop_index >= len(output_text):  # 如果停止字符串末尾已经是文本末尾
                # No truncation required.
                # 不需要截断
                return stop_str, -1  # 返回-1表示不需要截断

        # Truncate the output text to either the beginning
        # or end of the stop string.
        # 将输出文本截断到停止字符串的开头或末尾
        return stop_str, stop_index  # 返回匹配的停止字符串和截断位置
    return None  # 所有停止字符串都没有匹配到，返回None
