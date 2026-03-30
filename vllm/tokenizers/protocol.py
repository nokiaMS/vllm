# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
from collections.abc import Sequence  # 导入Sequence抽象基类
from pathlib import Path  # 导入路径处理类
from typing import TYPE_CHECKING, Any, Protocol, overload  # 导入类型注解相关工具

if TYPE_CHECKING:  # 仅在类型检查时导入以下模块
    from transformers import BatchEncoding  # 导入BatchEncoding类型

    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam  # 导入聊天消息参数类型


class TokenizerLike(Protocol):
    """分词器协议类，定义了所有分词器必须实现的接口规范。"""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,  # 模型路径或HuggingFace仓库ID
        *args,
        trust_remote_code: bool = False,  # 是否信任远程代码
        revision: str | None = None,  # 模型版本号
        download_dir: str | None = None,  # 下载目录
        **kwargs,
    ) -> "TokenizerLike":
        """从预训练模型加载分词器的类方法。"""
        raise NotImplementedError  # 子类必须实现此方法

    def num_special_tokens_to_add(self) -> int:
        """返回需要添加的特殊token数量。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def all_special_tokens(self) -> list[str]:
        """获取所有特殊token的字符串列表。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def all_special_ids(self) -> list[int]:
        """获取所有特殊token的ID列表。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def bos_token_id(self) -> int:
        """获取序列开始(BOS)token的ID。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def eos_token_id(self) -> int:
        """获取序列结束(EOS)token的ID。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def pad_token_id(self) -> int:
        """获取填充(PAD)token的ID。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def is_fast(self) -> bool:
        """判断是否为快速分词器。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def vocab_size(self) -> int:
        """获取词汇表大小。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def max_token_id(self) -> int:
        """获取最大token ID。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def max_chars_per_token(self) -> int:
        """获取单个token的最大字符数。"""
        raise NotImplementedError  # 子类必须实现此方法

    @property
    def truncation_side(self) -> str:
        """获取截断方向（'left'或'right'）。"""
        raise NotImplementedError  # 子类必须实现此方法

    def __hash__(self) -> int:
        """返回对象的哈希值，基于对象ID。"""
        return hash(id(self))  # 使用对象ID生成哈希值

    def __len__(self) -> int:
        """返回词汇表大小。"""
        return self.vocab_size  # 返回词汇表大小

    def __call__(
        self,
        text: str | list[str],  # 输入文本或文本列表
        text_pair: str | None = None,  # 可选的文本对
        add_special_tokens: bool = True,  # 是否添加特殊token
        truncation: bool = False,  # 是否进行截断
        max_length: int | None = None,  # 最大长度限制
    ) -> "BatchEncoding":
        """调用分词器对文本进行编码，返回BatchEncoding结果。"""
        raise NotImplementedError  # 子类必须实现此方法

    def get_vocab(self) -> dict[str, int]:
        """获取完整的词汇表字典（token字符串 -> token ID）。"""
        raise NotImplementedError  # 子类必须实现此方法

    def get_added_vocab(self) -> dict[str, int]:
        """获取额外添加的词汇表字典。"""
        raise NotImplementedError  # 子类必须实现此方法

    def encode(
        self,
        text: str,  # 输入文本
        truncation: bool | None = None,  # 是否截断
        max_length: int | None = None,  # 最大长度
        add_special_tokens: bool = True,  # 是否添加特殊token
    ) -> list[int]:
        """将文本编码为token ID列表。"""
        raise NotImplementedError  # 子类必须实现此方法

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],  # 聊天消息列表
        tools: list[dict[str, Any]] | None = None,  # 可选的工具定义
        **kwargs,
    ) -> str | list[int]:
        """应用聊天模板，将消息列表转换为文本或token ID列表。"""
        raise NotImplementedError  # 子类必须实现此方法

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...  # 单个token转ID的重载声明

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...  # token列表转ID列表的重载声明

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """将token字符串转换为对应的ID。"""
        raise NotImplementedError  # 子类必须实现此方法

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """将token列表转换为连续的字符串。"""
        raise NotImplementedError  # 子类必须实现此方法

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False  # token ID序列和是否跳过特殊token
    ) -> str:
        """将token ID序列解码为文本字符串。"""
        raise NotImplementedError  # 子类必须实现此方法

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],  # token ID序列
        skip_special_tokens: bool = False,  # 是否跳过特殊token
    ) -> list[str]:
        """将token ID序列转换为token字符串列表。"""
        raise NotImplementedError  # 子类必须实现此方法
