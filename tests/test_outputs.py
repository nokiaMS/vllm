# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试 RequestOutput 的前向兼容性：验证新增未知参数不会导致构造失败]

import pytest

from vllm.outputs import RequestOutput

pytestmark = pytest.mark.cpu_test


# [测试 RequestOutput 接受新版本中可能新增的未知关键字参数而不报错]
def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None
