# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 以上为 Apache-2.0 开源许可证声明和版权声明

# 从 vLLM V1 引擎模块导入 LLMEngine 类，并重命名为 V1LLMEngine 以避免命名冲突
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

# 将 LLMEngine 设置为 V1LLMEngine 的别名，保持向后兼容性
LLMEngine = V1LLMEngine  # type: ignore
"""
`LLMEngine` 类是 [vllm.v1.engine.llm_engine.LLMEngine][] 的别名。

该别名用于保持 API 向后兼容，旧代码中使用 LLMEngine 的地方
可以无缝迁移到 V1 版本的 LLMEngine 实现。
"""
