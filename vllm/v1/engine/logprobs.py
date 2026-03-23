# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools  # 导入itertools模块，用于创建高效的迭代器工具
from collections.abc import Iterable  # 从collections.abc导入Iterable抽象基类，用于类型注解可迭代对象
from dataclasses import dataclass  # 导入dataclass装饰器，用于简化数据类的定义

from vllm.logger import init_logger  # 导入日志初始化函数，用于创建模块级别的日志记录器
from vllm.logprobs import (  # 从vllm.logprobs模块导入logprobs相关的类型和工具函数
    PromptLogprobs,  # 导入提示词logprobs类型定义
    SampleLogprobs,  # 导入采样logprobs类型定义
    append_logprobs_for_next_position,  # 导入追加下一个位置logprobs的工具函数
    create_prompt_logprobs,  # 导入创建提示词logprobs的工厂函数
    create_sample_logprobs,  # 导入创建采样logprobs的工厂函数
)  # logprobs导入块结束
from vllm.tokenizers.detokenizer_utils import (  # 从分词器工具模块导入解码相关工具
    TokenizerLike,  # 导入分词器类型协议，用于类型注解
    convert_ids_list_to_tokens,  # 导入将token ID列表转换为token字符串的函数
)  # 分词器工具导入块结束
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest  # 导入引擎核心的输出和请求数据类
from vllm.v1.outputs import LogprobsLists, LogprobsTensors  # 导入logprobs的列表格式和张量格式类型定义

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

NONES = itertools.repeat(None)  # 创建一个无限重复None的迭代器，用于在禁用解码时替代token文本


# [中文注释] 每请求的 logprobs 处理器。负责：
#   1. _update_sample_logprobs — 处理生成阶段的 token logprobs（支持 speculative decoding 多 token）
#   2. _update_prompt_logprobs — 处理 prefill 阶段的 prompt logprobs（按 position 逐个构造）
#   3. cumulative_logprob — 累计 log 概率（用于 beam search 等场景）
#   4. _verify_tokens / _correct_decoded_token — 修复 UTF-8 字节回退导致的乱码 token
@dataclass  # 使用dataclass装饰器，自动生成__init__等方法
class LogprobsProcessor:  # logprobs处理器数据类定义
    """logprobs处理器类，负责管理和更新每个请求的logprobs（对数概率）数据。

    该处理器处理两种类型的logprobs：
    1. 采样logprobs（sample logprobs）：在生成阶段产生的token对数概率
    2. 提示词logprobs（prompt logprobs）：在预填充阶段产生的提示词token对数概率

    同时维护累积对数概率，并处理UTF-8字节回退导致的解码问题。
    """

    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: TokenizerLike | None  # 当前请求使用的分词器，如果禁用了去分词化则为None

    # Logprobs for this request
    logprobs: SampleLogprobs | None  # 采样阶段的logprobs列表，如果未请求logprobs则为None
    prompt_logprobs: PromptLogprobs | None  # 提示词阶段的logprobs列表，如果未请求prompt logprobs则为None
    cumulative_logprob: float | None  # 累积对数概率值，用于跟踪生成序列的总对数概率
    num_logprobs: int | None  # 每个位置返回的top logprobs数量（采样阶段）
    num_prompt_logprobs: int | None  # 每个位置返回的top logprobs数量（提示词阶段）

    @classmethod  # 类方法装饰器，允许通过类直接调用
    def from_new_request(  # 工厂方法：从新请求创建LogprobsProcessor实例
        cls,  # 类本身（类方法的隐式参数）
        tokenizer: TokenizerLike | None,  # 分词器实例，禁用去分词化时为None
        request: EngineCoreRequest,  # 引擎核心请求对象
    ) -> "LogprobsProcessor":  # 返回LogprobsProcessor实例
        """从新请求创建LogprobsProcessor实例的工厂方法。

        根据请求的采样参数初始化logprobs处理器，包括采样logprobs和提示词logprobs的设置。

        Args:
            cls: 类本身（类方法的隐式参数）
            tokenizer: 分词器实例，用于将token ID解码为文本；如果禁用去分词化则为None
            request: 引擎核心请求对象，包含采样参数等配置信息

        Returns:
            初始化好的LogprobsProcessor实例
        """
        sampling_params = request.sampling_params  # 从请求中获取采样参数
        assert sampling_params is not None  # 断言采样参数不为空，logprobs处理器必须有采样参数
        num_logprobs = sampling_params.logprobs  # 获取请求的采样logprobs数量
        num_prompt_logprobs = sampling_params.prompt_logprobs  # 获取请求的提示词logprobs数量
        return cls(  # 使用类构造器创建新实例
            tokenizer=tokenizer,  # 设置分词器
            cumulative_logprob=(None if num_logprobs is None else 0.0),  # 如果启用logprobs则初始化累积概率为0.0，否则为None
            logprobs=(  # 初始化采样logprobs容器
                None  # 如果未请求logprobs则为None
                if num_logprobs is None  # 条件判断：是否未请求logprobs
                else create_sample_logprobs(sampling_params.flat_logprobs)  # 否则创建采样logprobs容器（支持扁平化格式）
            ),  # logprobs参数三元表达式结束
            prompt_logprobs=(  # 初始化提示词logprobs容器
                None  # 如果未请求prompt logprobs则为None
                if num_prompt_logprobs is None  # 条件判断：是否未请求提示词logprobs
                else create_prompt_logprobs(sampling_params.flat_logprobs)  # 否则创建提示词logprobs容器（支持扁平化格式）
            ),  # prompt_logprobs参数三元表达式结束
            num_prompt_logprobs=num_prompt_logprobs,  # 设置每个位置的提示词logprobs数量
            num_logprobs=num_logprobs,  # 设置每个位置的采样logprobs数量
        )  # cls构造器调用结束

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:  # 更新采样阶段logprobs的方法
        """使用引擎核心返回的采样logprobs更新处理器状态。

        外层列表长度大于1的情况仅在引擎核心在上一步中生成了多个token时发生
        （例如在推测解码speculative decoding中）。

        Args:
            logprobs_lists: 包含logprob token ID列表、logprobs值列表和排名列表的元组。
        """

        assert self.num_logprobs is not None  # 断言采样logprobs数量已设置
        assert self.logprobs is not None  # 断言logprobs容器已初始化
        assert self.cumulative_logprob is not None  # 断言累积对数概率已初始化

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists  # 解包logprobs列表，获取token ID、logprobs值、排名，忽略第四个元素

        for rank_np, logprobs_np, token_ids_np in zip(  # 遍历每个生成位置的排名、logprobs和token ID（支持多token生成）
            ranks_lst, logprobs_lst, token_ids_lst  # 传入排名、logprobs和token ID的列表进行并行迭代
        ):  # zip迭代器的闭合括号
            rank = rank_np.tolist()  # 将排名numpy数组转换为Python列表
            logprobs = logprobs_np.tolist()  # 将logprobs numpy数组转换为Python列表
            token_ids = token_ids_np.tolist()  # 将token ID numpy数组转换为Python列表
            # Detokenize (non-incrementally).
            decoded_tokens: list[str] | Iterable[None]  # 声明解码后的token变量类型：字符串列表或None迭代器
            if self.tokenizer is None:  # 如果没有分词器（禁用了去分词化）
                decoded_tokens = NONES  # 使用无限None迭代器作为占位符
            else:  # 如果分词器可用
                decoded_tokens_list = convert_ids_list_to_tokens(  # 将token ID列表转换为token字符串列表
                    self.tokenizer, token_ids  # 传入分词器和token ID列表
                )  # convert_ids_list_to_tokens调用结束
                decoded_tokens = self._verify_tokens(  # 验证并修正可能的UTF-8字节回退解码错误
                    decoded_tokens_list=decoded_tokens_list, tokens=token_ids  # 传入解码token列表和原始token ID
                )  # _verify_tokens调用结束

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]  # 获取被采样token的logprob值（采样器将其放在列表首位）
            self.cumulative_logprob += sampled_token_logprob  # 累加被采样token的logprob到累积对数概率

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(  # 将当前位置的logprobs追加到logprobs容器中
                self.logprobs,  # logprobs容器
                token_ids,  # 当前位置的top token ID列表
                logprobs,  # 对应的logprob值列表
                decoded_tokens,  # 解码后的token文本列表
                rank,  # 排名列表
                self.num_logprobs,  # 需要保留的top logprobs数量
            )  # append_logprobs_for_next_position调用结束

    def _update_prompt_logprobs(  # 更新提示词logprobs的方法定义
        self,  # 实例自身引用
        prompt_logprobs_tensors: LogprobsTensors,  # 提示词logprobs张量参数
    ) -> None:  # 返回值为None
        """使用引擎核心返回的提示词logprobs张量更新处理器状态。

        处理预填充阶段的提示词logprobs，将张量格式的数据转换为Python对象，
        并逐位置构建logprobs容器。

        Args:
            prompt_logprobs_tensors: 包含提示词logprobs张量的元组，包括token ID、
                                     logprobs值、排名等信息。
        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None  # 断言提示词logprobs数量已设置
        assert self.prompt_logprobs is not None  # 断言提示词logprobs容器已初始化

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors  # 解包提示词logprobs张量，获取token ID、logprobs、排名，忽略第四个元素

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape  # 获取提示词token数量和每个位置的logprobs数量

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (  # 一次性解码所有token ID为文本（扁平化处理）
            None  # 如果分词器不可用则为None
            if self.tokenizer is None  # 条件判断：分词器是否为None
            else convert_ids_list_to_tokens(  # 将扁平化的token ID列表转换为token文本列表
                self.tokenizer, token_ids.flatten().tolist()  # 传入分词器和扁平化后的token ID列表
            )  # convert_ids_list_to_tokens调用结束
        )  # all_decoded_tokens三元表达式结束

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()  # 将排名张量转换为Python嵌套列表
        prompt_logprobs = logprobs.tolist()  # 将logprobs张量转换为Python嵌套列表
        token_ids_list = token_ids.tolist()  # 将token ID张量转换为Python嵌套列表

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):  # 遍历每个提示词token位置
            # Handle flattening and UTF-8 correction per position
            offset = pos * num_logprobs  # 计算当前位置在扁平化列表中的起始偏移量
            offset_end = offset + num_logprobs  # 计算当前位置在扁平化列表中的结束偏移量

            decoded_tokens_for_pos: list[str] | Iterable[None]  # 声明当前位置解码token的变量类型
            if all_decoded_tokens is None:  # 如果没有解码的token文本（分词器不可用）
                decoded_tokens_for_pos = NONES  # 使用无限None迭代器作为占位符
            else:  # 如果有解码的token文本
                # Extract decoded tokens for this position
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]  # 从扁平化列表中切片获取当前位置的解码token
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(  # 验证并修正当前位置可能的UTF-8字节回退解码错误
                    decoded_tokens_list=decoded_tokens_slice, tokens=token_ids_list[pos]  # 传入当前位置的解码token和token ID
                )  # _verify_tokens调用结束

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(  # 将当前位置的logprobs追加到提示词logprobs容器中
                self.prompt_logprobs,  # 提示词logprobs容器
                token_ids_list[pos],  # 当前位置的top token ID列表
                prompt_logprobs[pos],  # 当前位置的logprob值列表
                decoded_tokens_for_pos,  # 当前位置解码后的token文本
                prompt_token_ranks[pos],  # 当前位置的排名列表
                self.num_prompt_logprobs,  # 需要保留的top logprobs数量
            )  # append_logprobs_for_next_position调用结束

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:  # 弹出并返回提示词logprobs的方法
        """弹出并返回所有已聚合的提示词logprobs。

        logprobs处理器会在一个或多个预填充块中聚合提示词logprobs。
        此方法一次性返回所有提示词logprobs，然后清空内部存储。
        这确保了RequestOutputKind.DELTA语义的正确性，
        即所有提示词logprobs在预填充结束时一次性返回。

        Returns:
            如果此请求禁用了提示词logprobs，则返回None。
            否则返回包含所有提示词logprobs的列表。
        """
        plp = self.prompt_logprobs  # 保存当前提示词logprobs的引用
        if plp:  # 如果提示词logprobs非空（有数据）
            self.prompt_logprobs = []  # 重置提示词logprobs为空列表，以便下次收集新数据
        return plp  # 返回之前保存的提示词logprobs

    def _correct_decoded_token(self, idx: int, tokens: list[int]) -> str:  # 修正解码错误token的方法
        """修正因UTF-8字节回退分词导致的解码错误token。

        当token解码结果包含替换字符（U+FFFD，显示为"□"）时，
        尝试通过与相邻token组合解码来获取正确的文本。

        Args:
            idx: 需要修正的token在列表中的索引位置
            tokens: 当前位置的token ID列表

        Returns:
            修正后的token文本字符串，如果无法修正则返回空字符串
        """
        assert self.tokenizer is not None, "self.tokenizer should not be None"  # 断言分词器可用

        # try with prev token id in same list
        if idx > 0:  # 如果不是列表中的第一个token，尝试与同一列表中的前一个token组合解码
            possible_decoded_token = self.tokenizer.decode(tokens[idx - 1 : idx + 1])  # 将前一个token和当前token一起解码
            if not possible_decoded_token.endswith("�"):  # 如果组合解码结果不以替换字符结尾
                return possible_decoded_token  # 返回修正后的解码结果
        # try with previous logprob token id
        if self.logprobs:  # 如果存在之前的logprobs记录，尝试使用上一个位置的被采样token进行组合解码
            latest_token_id = next(iter(self.logprobs[-1]))  # 获取最近一个logprobs记录中的第一个token ID（即被采样的token）

            decode_ids = [latest_token_id]  # 初始化解码ID列表，以上一个被采样token开始
            if idx > 0:  # 如果当前token不是列表首位
                decode_ids.extend(tokens[idx - 1 : idx + 1])  # 添加前一个token和当前token
            else:  # 如果当前token是列表首位
                decode_ids.extend(tokens[idx : idx + 1])  # 只添加当前token

            possible_decoded_token = self.tokenizer.decode(decode_ids)  # 将组合的token ID列表一起解码
            if not possible_decoded_token.endswith("�"):  # 如果组合解码结果不以替换字符结尾
                return possible_decoded_token  # 返回修正后的解码结果

        # by default return empty string
        return ""  # 如果所有修正尝试都失败，返回空字符串作为默认值

    def _verify_tokens(  # 验证解码token的方法定义
        self, decoded_tokens_list: list[str], tokens: list[int]  # 接收解码token列表和token ID列表
    ) -> list[str]:  # 返回修正后的字符串列表
        """验证解码后的token列表，修正其中包含UTF-8替换字符的token。

        遍历所有解码后的token文本，检测以替换字符（U+FFFD）结尾的token，
        并尝试通过_correct_decoded_token方法进行修正。

        Args:
            decoded_tokens_list: 解码后的token文本列表
            tokens: 对应的token ID列表，用于辅助修正

        Returns:
            修正后的token文本列表
        """
        corrected_decoded_token_map = dict()  # 创建字典，用于存储需要修正的token索引及其修正后的文本
        for idx, text in enumerate(decoded_tokens_list):  # 遍历所有解码后的token文本
            if text.endswith("�"):  # 如果token文本以UTF-8替换字符结尾
                # utf-8 char at the end means it's a potential unfinished byte sequence
                # from byte fallback tokenization.
                corrected_decoded_token_map[idx] = self._correct_decoded_token(  # 尝试修正该token并记录到映射中
                    idx, tokens  # 传入token索引和token ID列表
                )  # _correct_decoded_token调用结束

        for idx, text in corrected_decoded_token_map.items():  # 遍历所有需要修正的token
            decoded_tokens_list[idx] = text  # 将原列表中的token文本替换为修正后的文本

        return decoded_tokens_list  # 返回修正后的token文本列表

    def update_from_output(self, output: EngineCoreOutput) -> None:  # 根据引擎核心输出更新logprobs的方法
        """根据引擎核心输出更新logprobs处理器状态。

        检查输出中是否包含新的采样logprobs和提示词logprobs，
        分别调用对应的更新方法进行处理。

        Args:
            output: 引擎核心输出对象，可能包含新的采样logprobs和/或提示词logprobs
        """
        if output.new_logprobs is not None:  # 如果输出中包含新的采样logprobs
            self._update_sample_logprobs(output.new_logprobs)  # 更新采样logprobs
        if output.new_prompt_logprobs_tensors is not None:  # 如果输出中包含新的提示词logprobs张量
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)  # 更新提示词logprobs
