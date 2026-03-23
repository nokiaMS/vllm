# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.renderers.inputs.preprocess import prompt_to_seq


# [中文注释] 测试prompt_to_seq对空输入的处理
def test_empty_input():
    assert prompt_to_seq([]) == []
    assert prompt_to_seq([[]]) == [[]]
    assert prompt_to_seq([[], []]) == [[], []]


# [中文注释] 测试prompt_to_seq对文本字符串输入的处理
def test_text_input():
    assert prompt_to_seq("foo") == ["foo"]
    assert prompt_to_seq(["foo"]) == ["foo"]
    assert prompt_to_seq(["foo", "bar"]) == ["foo", "bar"]


# [中文注释] 测试prompt_to_seq对token ID列表输入的处理
def test_token_input():
    assert prompt_to_seq([1, 2]) == [[1, 2]]
    assert prompt_to_seq([[1, 2]]) == [[1, 2]]
    assert prompt_to_seq([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


# [中文注释] 测试prompt_to_seq对文本和token混合输入的处理
def test_text_token_input():
    assert prompt_to_seq([[1, 2], "foo"]) == [[1, 2], "foo"]
    assert prompt_to_seq(["foo", [1, 2]]) == ["foo", [1, 2]]


# [中文注释] 测试prompt_to_seq对字节串输入的处理
def test_bytes_input():
    assert prompt_to_seq(b"foo") == [b"foo"]
    assert prompt_to_seq([b"foo"]) == [b"foo"]
    assert prompt_to_seq([b"foo", b"bar"]) == [b"foo", b"bar"]


# [中文注释] 测试prompt_to_seq对字典格式输入的处理
def test_dict_input():
    assert prompt_to_seq({"prompt": "foo"}) == [{"prompt": "foo"}]
    assert prompt_to_seq([{"prompt": "foo"}]) == [{"prompt": "foo"}]
    assert prompt_to_seq([{"prompt": "foo"}, {"prompt_token_ids": [1, 2]}]) == [
        {"prompt": "foo"},
        {"prompt_token_ids": [1, 2]},
    ]
