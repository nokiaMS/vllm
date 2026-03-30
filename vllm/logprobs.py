# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools  # 导入itertools模块，用于高效迭代操作
from collections.abc import Iterable, Iterator, MutableSequence  # 导入集合抽象基类
from dataclasses import dataclass, field  # 导入数据类装饰器和字段定义工具
from typing import overload  # 导入overload用于函数重载类型注解


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """对数概率信息类，用于支持OpenAI兼容的logprobs和token排名。

    存储单个token的对数概率、在词表中的排名以及解码后的文本。

    Attributes:
        logprob: 所选token的对数概率值。
        rank: 所选token在词表中的排名（从1开始）。
        decoded_token: 解码后的token字符串。
    """

    logprob: float  # 所选token的对数概率值
    rank: int | None = None  # 所选token的词表排名（>=1），可选
    decoded_token: str | None = None  # 解码后的token字符串，可选


LogprobsOnePosition = dict[int, Logprob]  # 单个位置的logprobs类型别名，token_id到Logprob的映射


@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """扁平化的对数概率存储容器。

    将请求的logprobs扁平化为多个基本类型列表进行存储。
    相比 list[dict[int, Logprob]]，该数据结构显著减少了GC（垃圾回收）开销。
    它将所有位置和排名的logprob信息扁平化到多个基本类型列表中
    （即logprobs、token_ids、每个token_id的ranks、decoded_tokens）。
    因此无论序列长度和top_logprobs设置如何，FlatLogprobs只会引入固定数量的对象。

    由于每个位置可能包含不同数量的排名，
    start_indices_per_position用于访问不同位置的logprob范围。

    注意：为减少迁移开销并提高向后兼容性，
    我们支持list的关键Sequence API，使其可以作为list[LogprobsOnePosition]使用。
    """

    # Start / end indices to indicate the range of logprobs for each position.
    start_indices: list[int] = field(default_factory=list)  # 每个位置的logprob起始索引列表
    end_indices: list[int] = field(default_factory=list)  # 每个位置的logprob结束索引列表（不包含）

    # Flatten Logprob information for (each position, rank).
    # For position <i>, the logprobs are ranged
    # from self.start_indices[i] to self.end_indices[i] (exclusive).
    token_ids: list[int] = field(default_factory=list)  # 扁平化存储的token ID列表
    logprobs: list[float] = field(default_factory=list)  # 扁平化存储的对数概率值列表
    ranks: list[int | None] = field(default_factory=list)  # 扁平化存储的排名列表
    decoded_tokens: list[str | None] = field(default_factory=list)  # 扁平化存储的解码token字符串列表

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
        """向容器中追加下一个位置的logprobs。

        Args:
            logprobs_one_position: 单个位置的logprobs字典，或None表示该位置无logprobs。
        """
        self.start_indices.append(len(self.logprobs))  # 记录当前位置的起始索引
        if logprobs_one_position:  # 如果该位置有logprobs数据
            for token_id, logprob in logprobs_one_position.items():  # 遍历该位置的每个token及其logprob
                self.token_ids.append(token_id)  # 追加token ID
                self.logprobs.append(logprob.logprob)  # 追加对数概率值
                self.ranks.append(logprob.rank)  # 追加排名
                self.decoded_tokens.append(logprob.decoded_token)  # 追加解码后的token字符串
        self.end_indices.append(len(self.logprobs))  # 记录当前位置的结束索引

    def append_fast(
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: itertools.chain[int],
        decoded_tokens: Iterable[str | None],
    ) -> None:
        """快速追加下一个位置的logprobs，无需创建中间logprob字典。

        直接从原始数据列表追加，避免创建中间字典对象，性能更优。

        Args:
            token_ids: 该位置的token ID列表。
            logprobs: 该位置的对数概率值列表。
            ranks: 该位置的排名迭代器。
            decoded_tokens: 该位置解码后的token字符串可迭代对象。
        """
        self.start_indices.append(len(self.logprobs))  # 记录当前位置的起始索引
        for token_id, logprob, rank, decoded_token in zip(  # 并行遍历所有数据
            token_ids, logprobs, ranks, decoded_tokens  # 将token_id、logprob、rank和decoded_token配对
        ):
            self.token_ids.append(token_id)  # 追加token ID
            self.logprobs.append(logprob)  # 追加对数概率值
            self.ranks.append(rank)  # 追加排名
            self.decoded_tokens.append(decoded_token)  # 追加解码后的token字符串
        self.end_indices.append(len(self.logprobs))  # 记录当前位置的结束索引

    def extend(self, logprobs_multi_positions) -> None:
        """扩展容器，追加多个位置的logprobs。

        Args:
            logprobs_multi_positions: 多个位置的logprobs可迭代对象。
        """
        for logprobs_one_position in logprobs_multi_positions:  # 遍历每个位置的logprobs
            self.append(logprobs_one_position)  # 逐个追加

    def __len__(self) -> int:
        """获取容器中存储的位置数量。

        Returns:
            已存储的位置总数。
        """
        return len(self.start_indices)  # 返回起始索引列表的长度，即位置数量

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...  # 整数索引重载，返回单个位置的logprobs

    @overload
    def __getitem__(self, s: slice, /) -> "FlatLogprobs": ...  # 切片索引重载，返回FlatLogprobs子集

    def __getitem__(self, index: int | slice):
        """根据索引或切片提取logprobs。

        Args:
            index: 整数索引获取单个位置的logprobs，切片获取多个位置的FlatLogprobs子集。

        Returns:
            整数索引时返回LogprobsOnePosition字典，切片时返回新的FlatLogprobs对象。

        Raises:
            TypeError: 索引类型无效时抛出。
        """
        if isinstance(index, int):  # 如果索引为整数
            return {  # 构建并返回该位置的logprobs字典
                self.token_ids[i]: Logprob(  # token ID作为键，Logprob对象作为值
                    logprob=self.logprobs[i],  # 对数概率值
                    rank=self.ranks[i],  # 排名
                    decoded_token=self.decoded_tokens[i],  # 解码后的token字符串
                )
                for i in range(self.start_indices[index], self.end_indices[index])  # 遍历该位置的索引范围
            }
        elif isinstance(index, slice):  # 如果索引为切片
            min_index = self.start_indices[index][0]  # 获取切片范围内的最小起始索引
            max_index = self.end_indices[index][-1]  # 获取切片范围内的最大结束索引
            return FlatLogprobs(  # 返回新的FlatLogprobs对象
                # Shift updated start_indices and end_indices to
                # be 0-indexed
                start_indices=[i - min_index for i in self.start_indices[index]],  # 将起始索引重新归零
                end_indices=[i - min_index for i in self.end_indices[index]],  # 将结束索引重新归零
                token_ids=self.token_ids[min_index:max_index],  # 截取对应范围的token ID
                logprobs=self.logprobs[min_index:max_index],  # 截取对应范围的对数概率值
                ranks=self.ranks[min_index:max_index],  # 截取对应范围的排名
                decoded_tokens=self.decoded_tokens[min_index:max_index],  # 截取对应范围的解码token
            )
        else:  # 如果索引类型既不是整数也不是切片
            raise TypeError(f"Invalid index type: {type(index)}")  # 抛出类型错误

    def __setitem__(self, item, value) -> None:
        """禁止设置操作。

        Raises:
            TypeError: 始终抛出，因为FlatLogprobs不支持设置操作。
        """
        raise TypeError("Cannot set logprobs in FlatLogprobs")  # 抛出类型错误，不支持设置操作

    def __delitem__(self, item) -> None:
        """禁止删除操作。

        Raises:
            TypeError: 始终抛出，因为FlatLogprobs不支持删除操作。
        """
        raise TypeError("Cannot delete logprobs from FlatLogprobs")  # 抛出类型错误，不支持删除操作

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:
        """禁止插入操作。

        Raises:
            TypeError: 始终抛出，因为FlatLogprobs不支持插入操作。
        """
        raise TypeError("Cannot insert logprobs to FlatLogprobs")  # 抛出类型错误，不支持插入操作

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        """迭代容器，为每个位置生成LogprobsOnePosition。

        Yields:
            每个位置的LogprobsOnePosition字典。
        """
        for i in range(0, len(self.start_indices)):  # 遍历所有位置索引
            yield self.__getitem__(i)  # 逐个生成该位置的logprobs字典


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]  # Prompt logprobs类型：每个序列组的{token_id -> logprob}映射，如果不需要则为None
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]  # 采样logprobs类型：每个序列组的{token_id -> logprob}映射


def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    """创建用于存储prompt logprobs的容器。

    Args:
        flat_logprobs: 是否使用扁平化存储格式。True使用FlatLogprobs，False使用普通列表。

    Returns:
        初始化好的PromptLogprobs容器，首个prompt token的logprob已设为None。
    """
    logprobs: PromptLogprobs = FlatLogprobs() if flat_logprobs else []  # 根据参数选择创建FlatLogprobs或普通列表
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)  # 第一个prompt token的logprob为None
    return logprobs  # 返回初始化后的容器


def create_sample_logprobs(flat_logprobs: bool) -> SampleLogprobs:
    """创建用于存储解码logprobs的容器。

    Args:
        flat_logprobs: 是否使用扁平化存储格式。True使用FlatLogprobs，False使用普通列表。

    Returns:
        初始化好的SampleLogprobs空容器。
    """
    return FlatLogprobs() if flat_logprobs else []  # 根据参数选择创建FlatLogprobs或普通列表


def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
    """向logprobs容器追加下一个位置的logprobs信息。

    将采样token和top-k token的logprob信息追加到请求的logprobs容器中。

    Args:
        request_logprobs: 请求的logprobs容器，可以是FlatLogprobs或普通列表。
        token_ids: 该位置的token ID列表（包括采样token和top-k token）。
        logprobs: 对应token的对数概率值列表。
        decoded_tokens: 解码后的token字符串可迭代对象。
        rank: 采样token的排名。
        num_logprobs: 需要返回的top logprobs数量，-1表示返回全部。
    """
    if num_logprobs == -1:  # 如果num_logprobs为-1，表示返回所有logprobs
        num_logprobs = len(logprobs)  # 设为logprobs列表的总长度
    # We do not need a special case for the sampled token
    # being in the topk, since inserting duplicated data
    # into a dictionary twice is the same as doing it once.
    topk_ranks = range(1, num_logprobs + 1)  # 生成top-k排名范围，从1到num_logprobs
    ranks = itertools.chain((rank,), topk_ranks)  # 将采样token的排名与top-k排名链接起来

    if isinstance(request_logprobs, FlatLogprobs):  # 如果使用扁平化存储
        request_logprobs.append_fast(token_ids, logprobs, ranks, decoded_tokens)  # 调用快速追加方法
    else:  # 如果使用普通列表存储
        request_logprobs.append(  # 追加一个logprobs字典
            {
                token_id: Logprob(  # 为每个token创建Logprob对象
                    logprob=logprob,  # 对数概率值
                    rank=rank,  # 排名
                    decoded_token=token,  # 解码后的token字符串
                )
                for token_id, logprob, rank, token in zip(  # 并行遍历所有数据
                    token_ids, logprobs, ranks, decoded_tokens  # 将token_id、logprob、rank和token配对
                )
            }
        )
