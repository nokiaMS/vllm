# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# 张量形状描述类，用于声明张量的维度信息（支持符号化维度名称和动态维度标记）
# 设计思路：通过字符串维度名实现符号化形状约束，配合 resolve() 方法在运行时将符号绑定为具体数值
class TensorShape:
    def __init__(
        self,
        *dims: int | str,
        dynamic_dims: set[str] | None = None,
    ) -> None:
        super().__init__()

        self.dims = dims
        self.dynamic_dims = dynamic_dims if dynamic_dims else set()

    # 将符号化维度名替换为具体的整数值，未绑定的维度名保持原样
    def resolve(self, **bindings: int) -> tuple[int | str, ...]:
        resolved = list[int | str]()
        for dim in self.dims:
            if isinstance(dim, str) and dim in bindings:
                resolved.append(bindings[dim])
            else:
                resolved.append(dim)
        return tuple(resolved)

    def __str__(self) -> str:
        """Return a string representation of the tensor shape."""
        dim_strs = []
        for dim in self.dims:
            if isinstance(dim, str):
                if dim in self.dynamic_dims:
                    dim_strs.append(f"{dim}*")  # Mark dynamic dimensions with *
                else:
                    dim_strs.append(dim)
            else:
                dim_strs.append(str(dim))
        return f"({', '.join(dim_strs)})"


# 张量模式验证类，利用 Python 类型注解（Annotated + TensorShape）对字段进行形状校验
# 设计思路：通过 type hints 自动提取每个字段的期望形状，在构造时自动验证张量维度是否匹配
# 关键算法：validate() 遍历所有带 TensorShape 注解的字段，使用 shape_env 字典追踪符号维度的绑定关系，确保跨字段维度一致性
class TensorSchema:
    def __init__(
        self,
        *,
        validate: bool = True,
        resolve_bindings: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._resolve_bindings = resolve_bindings if resolve_bindings else {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        if validate:
            self.validate()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    # 比较两个形状是否匹配，跳过动态维度（dynamic_dims 中标记的维度名不参与比较）
    def _match_shape_with_dynamic(
        self,
        actual: tuple[int, ...],
        reference: tuple[int, ...],
        expected_shape: tuple[int | str, ...],
        dynamic_dims: set[str],
    ) -> bool:
        if len(actual) != len(reference) or len(actual) > len(expected_shape):
            return False

        for i, (a, r) in enumerate(zip(actual, reference)):
            # When validating list inputs, we match shape suffixes only
            # (e.g. "p", 3, "h", "w"), assuming the list length corresponds
            # to the leading symbolic dim (e.g. "bn"). This allows comparing
            # only the trailing dimensions of each element in the list.
            dim = expected_shape[-len(actual) + i]
            # Skip this dimension if it's marked dynamic
            if dim in dynamic_dims:
                continue
            if a != r:
                return False
        return True

    def _fmt_indexer(self, idxs: tuple[int, ...]) -> str:
        if not idxs:
            return ""

        return str(list(idxs))

    # 递归验证单个字段的值：标量直接返回空形状，张量返回其形状，列表/元组则递归验证每个元素并检查形状一致性
    def _validate_field(
        self,
        value: object,
        field_name: str,
        expected_shape: tuple[int | str, ...],
        dynamic_dims: set[str],
        leading_idxs: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        """Validate a field and return the actual shape."""
        if isinstance(value, (int, float)):
            return ()  # Scalar
        if isinstance(value, torch.Tensor):
            return value.shape

        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{field_name}{self._fmt_indexer(leading_idxs)} is not "
                f"one of the expected types: int, float, Tensor, list, tuple. "
                f"Got: {type(value)}"
            )

        if len(value) == 0:
            raise ValueError(
                f"{field_name}{self._fmt_indexer(leading_idxs)} is an empty sequence"
            )

        # Ensure all tensors in the list have the same
        # shape, besides dynamic dimensions
        for i, v in enumerate(value):
            shape = self._validate_field(
                v,
                field_name,
                expected_shape[1:],
                dynamic_dims,
                leading_idxs=leading_idxs + (i,),
            )

            if i == 0:
                first_shape = shape
            elif not self._match_shape_with_dynamic(
                shape,
                first_shape,
                expected_shape,
                dynamic_dims,
            ):
                raise ValueError(
                    f"{field_name}{self._fmt_indexer(leading_idxs)} "
                    f"contains inconsistent shapes: {first_shape} "
                    f"(index 0) vs {shape} (index {i})"
                )

        # Treat the list as a stacked tensor:
        # shape = (len(list), *tensor.shape)
        return (len(value),) + first_shape

    # 逐维度验证实际形状与期望形状的匹配：整数维度精确比较，字符串维度通过 shape_env 实现跨字段绑定一致性检查
    def _validate_tensor_shape_expected(
        self,
        actual_shape: tuple[int, ...],
        expected_shape: tuple[int | str, ...],
        field_name: str,
        shape_env: dict[str, int],
        dynamic_dims: set[str],
    ) -> None:
        """Validate that the actual tensor shape matches the expected shape."""

        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{field_name} has rank {len(actual_shape)} "
                f"but expected {len(expected_shape)}. "
                f"Expected shape: {expected_shape}, "
                f"but got {actual_shape}"
            )

        for i, dim in enumerate(expected_shape):
            if dim in dynamic_dims:
                continue
            elif isinstance(dim, int):
                if actual_shape[i] != dim:
                    raise ValueError(
                        f"{field_name} dim[{i}] expected "
                        f"{dim}, got {actual_shape[i]}. "
                        f"Expected shape: {expected_shape}, "
                        f"but got {actual_shape}"
                    )
            elif isinstance(dim, str):
                if dim in shape_env:
                    if actual_shape[i] != shape_env[dim]:
                        raise ValueError(
                            f"{field_name} dim[{i}] expected "
                            f"'{dim}'={shape_env[dim]}, got "
                            f"{actual_shape[i]}"
                        )
                else:
                    shape_env[dim] = actual_shape[i]
            else:
                raise TypeError(
                    f"{field_name} dim[{i}] has unsupported type: {type(dim)}"
                )

    # 主校验入口：遍历所有类型注解字段，处理 Optional 字段和 Annotated[TensorShape] 字段的形状验证
    def validate(self) -> None:
        type_hints = get_type_hints(self.__class__, include_extras=True)
        shape_env = dict[str, int]()

        for field_name, field_type in type_hints.items():
            # Check if field is missing
            if not hasattr(self, field_name) or getattr(self, field_name) is None:
                # Check if field is marked as optional
                actual_type = field_type
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    actual_type = args[0]

                # Check arg was provided as Union
                if get_origin(actual_type) in {Union, UnionType}:
                    # Union for Union[X, Y] and UnionType for X | Y
                    args = get_args(actual_type)
                    # Skip validation when Union contains None
                    if type(None) in args:
                        continue
                # Otherwise field is required, raise error
                raise ValueError(f"Required field '{field_name}' is missing")

            # Field exists, proceed with validation
            value = getattr(self, field_name)
            if get_origin(field_type) is not None:
                args = get_args(field_type)

                for arg in args:
                    if isinstance(arg, TensorShape):
                        expected_shape = arg.resolve(**self._resolve_bindings)
                        actual_shape = self._validate_field(
                            value,
                            field_name,
                            expected_shape,
                            arg.dynamic_dims,
                        )

                        self._validate_tensor_shape_expected(
                            actual_shape,
                            expected_shape,
                            field_name,
                            shape_env,
                            arg.dynamic_dims,
                        )

    def print_shapes(self) -> None:
        """Print TensorShape annotations for debugging."""
        logger.debug("Shapes in %s:", self.__class__.__name__)
        type_hints = get_type_hints(self.__class__, include_extras=True)

        for field_name, field_type in type_hints.items():
            if get_origin(field_type) is not None:
                args = get_args(field_type)
                for arg in args:
                    if isinstance(arg, TensorShape):
                        logger.debug("  %s: %s", field_name, str(arg))
