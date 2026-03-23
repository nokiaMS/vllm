# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================================
# Ray 分布式执行器的向后兼容模块
# RayDistributedExecutor 类已迁移至 ray_executor.py，
# 本文件仅做重新导出以保持旧版导入路径可用。
# ============================================================================

from vllm.v1.executor.ray_executor import (
    RayDistributedExecutor as _RayDistributedExecutor,
)

# For backwards compatibility.
RayDistributedExecutor = _RayDistributedExecutor
