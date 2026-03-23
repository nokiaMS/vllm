# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper functions to work with nested JSON structures."""

from collections.abc import Callable, Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, overload

if TYPE_CHECKING:
    import torch

    from vllm.multimodal.inputs import BatchedTensorInputs

_T = TypeVar("_T")
_U = TypeVar("_U")

# JSONTree 类型别名：表示嵌套的 JSON 树结构（dict/list/tuple 的递归组合）
# 叶子节点类型为泛型 _T，不要求是 JSON 可序列化的
JSONTree: TypeAlias = (
    dict[str, "JSONTree[_T]"] | list["JSONTree[_T]"] | tuple["JSONTree[_T]", ...] | _T
)
"""A nested JSON structure where the leaves need not be JSON-serializable."""

_JSONTree: TypeAlias = (
    dict[str, "JSONTree[_T]"]
    | list["JSONTree[_T]"]
    | tuple["JSONTree[_T]", ...]
    | dict[str, _T]
    | list[_T]
    | tuple[_T, ...]
    | _T
)
"""
Same as `JSONTree` but with additional `Union` members to satisfy overloads.
"""


# 递归遍历嵌套 JSON 树结构中的所有叶子节点
# 遇到 dict/list/tuple 时递归展开，其他类型视为叶子节点
def json_iter_leaves(value: JSONTree[_T]) -> Iterable[_T]:
    """Iterate through each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        for v in value.values():
            yield from json_iter_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from json_iter_leaves(v)
    else:
        yield value


# 对嵌套 JSON 树结构中的每个叶子节点应用函数，保持树的结构不变
# 提供多个重载签名以支持精确的类型推断（dict/list/tuple/通用 JSONTree）
@overload
def json_map_leaves(
    func: Callable[["torch.Tensor"], "torch.Tensor"],
    value: "BatchedTensorInputs",
) -> "BatchedTensorInputs": ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: _T | dict[str, _T],
) -> _U | dict[str, _U]: ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: _T | list[_T],
) -> _U | list[_U]: ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: _T | tuple[_T, ...],
) -> _U | tuple[_U, ...]: ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: JSONTree[_T],
) -> JSONTree[_U]: ...


def json_map_leaves(
    func: Callable[[_T], _U],
    value: Any,
) -> "BatchedTensorInputs" | _JSONTree[_U]:
    """Apply a function to each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        return {k: json_map_leaves(func, v) for k, v in value.items()}  # type: ignore
    elif isinstance(value, list):
        return [json_map_leaves(func, v) for v in value]  # type: ignore
    elif isinstance(value, tuple):
        return tuple(json_map_leaves(func, v) for v in value)
    else:
        return func(value)


# 对嵌套 JSON 树的所有叶子节点从左到右进行累积归约
# 类似 functools.reduce，支持带初始值和不带初始值两种形式
@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: _T | dict[str, _T],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: _T | list[_T],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: _T | tuple[_T, ...],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: JSONTree[_T],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_U, _T], _U],
    value: JSONTree[_T],
    initial: _U,
    /,
) -> _U: ...


def json_reduce_leaves(
    func: Callable[[_T, _T], _T] | Callable[[_U, _T], _U],
    value: _JSONTree[_T],
    initial: _U = ...,  # type: ignore[assignment]
    /,
) -> _T | _U:
    """
    Apply a function of two arguments cumulatively to each leaf in a
    nested JSON structure, from left to right, so as to reduce the
    sequence to a single value.
    """
    if initial is ...:
        return reduce(func, json_iter_leaves(value))  # type: ignore

    return reduce(func, json_iter_leaves(value), initial)  # type: ignore


# 计算嵌套 JSON 树结构中叶子节点的总数
def json_count_leaves(value: JSONTree[_T]) -> int:
    """Count the number of leaves in a nested JSON structure."""
    return sum(1 for _ in json_iter_leaves(value))
