# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者

# ============================================================================
# executor 包的初始化模块
# 本模块是 vLLM v1 执行器子系统的公开入口，负责导出核心类：
#   - Executor: 所有执行器的抽象基类
#   - UniProcExecutor: 单进程执行器，适用于单设备推理
# 使用者通过 `from vllm.v1.executor import Executor` 即可获取执行器基类，
# 再通过 Executor.get_class() 工厂方法根据配置动态选择具体实现。
# ============================================================================

from .abstract import Executor  # 从当前包的 abstract 模块导入 Executor 抽象基类
from .uniproc_executor import UniProcExecutor  # 从当前包的 uniproc_executor 模块导入单进程执行器

__all__ = ["Executor", "UniProcExecutor"]  # 定义模块的公开接口列表，限制 from xxx import * 的导出范围
