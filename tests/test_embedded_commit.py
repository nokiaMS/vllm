# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试嵌入式版本信息：验证 vllm 模块包含有效的版本号和版本元组]

import vllm


# [测试 vllm 模块存在 __version__ 和 __version_tuple__ 属性，且不为开发版本默认值]
def test_embedded_commit_defined():
    assert hasattr(vllm, "__version__")
    assert hasattr(vllm, "__version_tuple__")
    assert vllm.__version__ != "dev"
    assert vllm.__version_tuple__ != (0, 0, "dev")
