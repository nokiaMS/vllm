# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.attention.backends.placeholder_attn import PlaceholderAttentionBackend


# 测试用的虚拟注意力后端，继承自占位符注意力后端
class DummyAttentionBackend(PlaceholderAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
