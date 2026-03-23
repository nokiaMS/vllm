# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# 虚拟平台插件入口：注册虚拟平台和自定义算子
def dummy_platform_plugin() -> str | None:
    return "vllm_add_dummy_platform.dummy_platform.DummyPlatform"


def register_ops():
    import vllm_add_dummy_platform.dummy_custom_ops  # noqa
