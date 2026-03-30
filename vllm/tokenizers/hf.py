# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
import contextlib  # 导入上下文管理工具
import copy  # 导入对象复制工具
from pathlib import Path  # 导入路径处理类
from typing import TypeAlias  # 导入类型别名

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast  # 导入HuggingFace分词器相关类

from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config  # 导入句子transformer分词器配置获取函数

from .protocol import TokenizerLike  # 从协议模块导入TokenizerLike

HfTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast  # 定义HuggingFace分词器类型别名


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """创建一个缓存属性的分词器代理。

    默认情况下，transformers会在每次调用时重新计算多个分词器属性，
    导致显著的性能下降。此代理缓存这些属性以加速访问。
    """
    cached_tokenizer = copy.copy(tokenizer)  # 浅拷贝分词器对象

    tokenizer_all_special_ids = tokenizer.all_special_ids  # 缓存所有特殊token ID
    tokenizer_all_special_tokens = tokenizer.all_special_tokens  # 缓存所有特殊token字符串
    tokenizer_vocab = tokenizer.get_vocab()  # 缓存词汇表
    tokenizer_len = len(tokenizer)  # 缓存分词器长度

    max_token_id = max(tokenizer_vocab.values())  # 计算最大token ID
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)  # 计算单个token的最大字符数

    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value.
    if hasattr(tokenizer, "vocab_size"):  # 如果分词器有vocab_size属性
        with contextlib.suppress(NotImplementedError):  # 忽略NotImplementedError异常
            max_token_id = max(max_token_id, tokenizer.vocab_size)  # 取较大值作为最大token ID

    class CachedTokenizer(tokenizer.__class__):  # type: ignore  # 动态创建缓存分词器类，继承原分词器类
        """缓存了常用属性的分词器子类，用于提升访问性能。"""

        @property
        def all_special_ids(self) -> list[int]:
            """返回缓存的所有特殊token ID列表。"""
            return tokenizer_all_special_ids  # 返回缓存值

        @property
        def all_special_tokens(self) -> list[str]:
            """返回缓存的所有特殊token字符串列表。"""
            return tokenizer_all_special_tokens  # 返回缓存值

        @property
        def max_token_id(self) -> int:
            """返回缓存的最大token ID。"""
            return max_token_id  # 返回缓存值

        @property
        def max_chars_per_token(self) -> int:
            """返回缓存的单个token最大字符数。"""
            return max_chars_per_token  # 返回缓存值

        def get_vocab(self) -> dict[str, int]:
            """返回缓存的词汇表字典。"""
            return tokenizer_vocab  # 返回缓存值

        def __len__(self) -> int:
            """返回缓存的分词器长度。"""
            return tokenizer_len  # 返回缓存值

        def __reduce__(self):
            """支持pickle序列化，返回重建所需的函数和参数。"""
            return get_cached_tokenizer, (tokenizer,)  # 返回工厂函数和原始分词器

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"  # 设置缓存类的名称

    cached_tokenizer.__class__ = CachedTokenizer  # 将拷贝对象的类替换为缓存类
    return cached_tokenizer  # 返回缓存分词器


class CachedHfTokenizer(TokenizerLike):
    """带缓存的HuggingFace分词器加载类，支持从预训练模型加载并缓存常用属性。"""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,  # 模型路径或HuggingFace仓库ID
        *args,
        trust_remote_code: bool = False,  # 是否信任远程代码
        revision: str | None = None,  # 模型版本号
        download_dir: str | None = None,  # 下载目录
        **kwargs,
    ) -> HfTokenizer:
        """从预训练模型加载分词器并应用缓存优化。"""
        try:  # 尝试加载分词器
            tokenizer = AutoTokenizer.from_pretrained(  # 使用AutoTokenizer自动加载
                path_or_repo_id,  # 模型路径或仓库ID
                *args,  # 传递额外位置参数
                trust_remote_code=trust_remote_code,  # 是否信任远程代码
                revision=revision,  # 版本号
                cache_dir=download_dir,  # 缓存目录
                **kwargs,  # 传递额外关键字参数
            )
        except ValueError as e:  # 捕获ValueError
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (  # 如果未启用信任远程代码
                "does not exist or is not currently imported." in str(e)  # 且错误信息包含类不存在
                or "requires you to execute the tokenizer file" in str(e)  # 或需要执行分词器文件
            ):
                err_msg = (  # 构造错误提示信息
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e  # 抛出运行时错误
            else:  # 其他ValueError
                raise e  # 直接抛出

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(  # 获取句子transformer编码器配置
            path_or_repo_id, revision  # 传入模型路径和版本号
        )
        if isinstance(encoder_config, dict) and encoder_config.get(  # 如果配置是字典且启用了小写
            "do_lower_case", False  # 检查是否需要转小写
        ):
            special_tokens_map = {  # 构建小写的特殊token映射
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()  # 将所有特殊token转小写
            }
            tokenizer.add_special_tokens(special_tokens_map)  # 添加小写的特殊token

        return get_cached_tokenizer(tokenizer)  # 返回缓存优化后的分词器
