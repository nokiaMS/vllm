# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

import pytest

from vllm.utils.func_utils import deprecate_kwargs, supports_kw

from ..utils import error_on_warning


# [中文注释] 测试始终触发弃用警告：当is_deprecated=True时，使用旧参数名应发出DeprecationWarning
def test_deprecate_kwargs_always():
    @deprecate_kwargs("old_arg", is_deprecated=True)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


# [中文注释] 测试从不触发弃用警告：当is_deprecated=False时，使用旧参数名不应发出警告
def test_deprecate_kwargs_never():
    @deprecate_kwargs("old_arg", is_deprecated=False)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


# [中文注释] 测试动态弃用判断：通过lambda回调动态控制是否触发弃用警告
def test_deprecate_kwargs_dynamic():
    is_deprecated = True

    @deprecate_kwargs("old_arg", is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)

    is_deprecated = False

    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


# [中文注释] 测试附加弃用信息：验证弃用警告中是否包含自定义的附加提示消息
def test_deprecate_kwargs_additional_message():
    @deprecate_kwargs("old_arg", is_deprecated=True, additional_message="abcd")
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="abcd"):
        dummy(old_arg=1)


# [中文注释] 测试supports_kw函数：验证能否正确判断callable是否支持指定的关键字参数
@pytest.mark.parametrize(
    ("callable", "kw_name", "requires_kw_only", "allow_var_kwargs", "is_supported"),
    [
        # Tests for positional argument support
        (lambda foo: None, "foo", True, True, False),
        (lambda foo: None, "foo", False, True, True),
        # Tests for positional or keyword / keyword only
        (lambda foo=100: None, "foo", True, True, False),
        (lambda *, foo: None, "foo", False, True, True),
        # Tests to make sure the names of variadic params are NOT supported
        (lambda *args: None, "args", False, True, False),
        (lambda **kwargs: None, "kwargs", False, True, False),
        # Tests for if we allow var kwargs to add support
        (lambda foo: None, "something_else", False, True, False),
        (lambda foo, **kwargs: None, "something_else", False, True, True),
        (lambda foo, **kwargs: None, "kwargs", True, True, False),
        (lambda foo, **kwargs: None, "foo", True, True, False),
    ],
)
def test_supports_kw(
    callable, kw_name, requires_kw_only, allow_var_kwargs, is_supported
):
    assert (
        supports_kw(
            callable=callable,
            kw_name=kw_name,
            requires_kw_only=requires_kw_only,
            allow_var_kwargs=allow_var_kwargs,
        )
        == is_supported
    )
