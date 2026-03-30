# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.parameter import BasevLLMParameter, PackedvLLMParameter  # 导入vLLM基础参数类和打包参数类

__all__ = [  # 定义模块的公开接口列表
    "BasevLLMParameter",  # 基础vLLM参数类
    "PackedvLLMParameter",  # 打包vLLM参数类
]
