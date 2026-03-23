# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar, overload

from tqdm.auto import tqdm

_T = TypeVar("_T", bound=Iterable)


# 条件性进度条包装函数：当 use_tqdm 为 True 或自定义 tqdm 函数时包装迭代器显示进度，否则直接返回原迭代器
# 设计思路：统一进度条使用方式，调用方通过一个参数即可控制是否展示进度条
@overload
def maybe_tqdm(
    it: Sequence[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    **tqdm_kwargs: Any,
) -> Sequence[_T]: ...


@overload
def maybe_tqdm(
    it: Iterable[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    **tqdm_kwargs: Any,
) -> Iterable[_T]: ...


def maybe_tqdm(
    it: Iterable[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    **tqdm_kwargs: Any,
) -> Iterable[_T]:
    if not use_tqdm:
        return it

    tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
    return tqdm_func(it, **tqdm_kwargs)
