# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 以上为 Apache-2.0 开源许可证声明和版权声明

# 从 vLLM V1 引擎模块导入异步 LLM 类
from vllm.v1.engine.async_llm import AsyncLLM

# 将 AsyncLLMEngine 设置为 AsyncLLM 的别名，保持向后兼容性
AsyncLLMEngine = AsyncLLM  # type: ignore
"""
`AsyncLLMEngine` 类是 [vllm.v1.engine.async_llm.AsyncLLM][] 的别名。

该别名用于保持 API 向后兼容，旧代码中使用 AsyncLLMEngine 的地方
可以无缝迁移到 V1 版本的 AsyncLLM 实现。
"""
