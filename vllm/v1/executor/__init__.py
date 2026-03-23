# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================================
# executor 包的初始化模块
# 本模块是 vLLM v1 执行器子系统的公开入口，负责导出核心类：
#   - Executor: 所有执行器的抽象基类
#   - UniProcExecutor: 单进程执行器，适用于单设备推理
# 使用者通过 `from vllm.v1.executor import Executor` 即可获取执行器基类，
# 再通过 Executor.get_class() 工厂方法根据配置动态选择具体实现。
# ============================================================================

from .abstract import Executor
from .uniproc_executor import UniProcExecutor

__all__ = ["Executor", "UniProcExecutor"]
