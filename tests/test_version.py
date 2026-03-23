# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试版本号模块：验证版本字符串、版本元组及上一次次版本号判断逻辑]

from unittest.mock import patch

import pytest

from vllm import version


# [测试版本号已被正确定义，不为 None]
def test_version_is_defined():
    assert version.__version__ is not None


# [测试版本元组长度为 3 到 5 个元素]
def test_version_tuple():
    assert len(version.__version_tuple__) in (3, 4, 5)


@pytest.mark.parametrize(
    "version_tuple, version_str, expected",
    [
        ((0, 0, "dev"), "0.0", True),
        ((0, 0, "dev"), "foobar", True),
        ((0, 7, 4), "0.6", True),
        ((0, 7, 4), "0.5", False),
        ((0, 7, 4), "0.7", False),
        ((1, 2, 3), "1.1", True),
        ((1, 2, 3), "1.0", False),
        ((1, 2, 3), "1.2", False),
        # This won't work as expected
        ((1, 0, 0), "1.-1", True),
        ((1, 0, 0), "0.9", False),
        ((1, 0, 0), "0.17", False),
    ],
)
# [参数化测试 _prev_minor_version_was 函数在不同版本元组和版本字符串下的判断是否正确]
def test_prev_minor_version_was(version_tuple, version_str, expected):
    with patch("vllm.version.__version_tuple__", version_tuple):
        assert version._prev_minor_version_was(version_str) == expected
