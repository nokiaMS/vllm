# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

# 虚拟模型插件的安装配置，用于测试 vLLM 模型注册机制
setup(
    name="vllm_add_dummy_model",
    version="0.1",
    packages=["vllm_add_dummy_model"],
    entry_points={
        "vllm.general_plugins": ["register_dummy_model = vllm_add_dummy_model:register"]
    },
)
