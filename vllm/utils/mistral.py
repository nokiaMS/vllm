# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Mistral 分词器工具模块
# 通过 LazyLoader 延迟导入 vllm.tokenizers.mistral，避免在不使用 Mistral 模型时
# 引入不必要的依赖开销；同时提供类型守卫函数用于安全地判断分词器类型
"""Provides lazy import of the vllm.tokenizers.mistral module."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from vllm.tokenizers import TokenizerLike
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    # if type checking, eagerly import the module
    import vllm.tokenizers.mistral as mt
else:
    mt = LazyLoader("mt", globals(), "vllm.tokenizers.mistral")


# 类型守卫函数：判断给定的分词器是否为 MistralTokenizer 实例
# 先检查类属性 IS_MISTRAL_TOKENIZER 以避免触发延迟导入，仅在属性为 True 时才执行 isinstance 确认
def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGuard[mt.MistralTokenizer]:
    """Return true if the tokenizer is a MistralTokenizer instance."""
    cls = type(obj)
    # Check for special class attribute, this avoids importing the class to
    # do an isinstance() check.  If the attribute is True, do an isinstance
    # check to be sure we have the correct type.
    return bool(
        getattr(cls, "IS_MISTRAL_TOKENIZER", False)
        and isinstance(obj, mt.MistralTokenizer)
    )
