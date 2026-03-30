# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small helper wrappers for external Oink Blackwell custom ops.
外部Oink Blackwell自定义算子的小型辅助封装模块。

vLLM does not depend on the external Oink repository/package. When an external
plugin registers torch.library.custom_op entrypoints under the `oink::`
namespace (e.g. via vLLM's general_plugins mechanism) and
`VLLM_USE_OINK_OPS=1` is set, vLLM can route eligible calls to those ops.
vLLM不依赖外部Oink仓库/包。当外部插件在`oink::`命名空间下注册
torch.library.custom_op入口点（例如通过vLLM的通用插件机制），并且设置了
`VLLM_USE_OINK_OPS=1`时，vLLM可以将符合条件的调用路由到这些算子。

This module provides:
- A single place to probe Oink op availability at module init time
  (outside torch.compile tracing), and
- Thin wrappers around the torch.ops entrypoints for use in CUDA fast paths,
  without introducing graph breaks.
本模块提供：
- 在模块初始化时（torch.compile追踪之外）探测Oink算子可用性的统一入口
- torch.ops入口点的轻量封装，用于CUDA快速路径，不引入图中断

Important:
  Do not call the availability helpers in a compiled region. They may call
  functions decorated with `torch._dynamo.disable` to safely check
  conditions that should not be traced.
重要提示：
  不要在编译区域内调用可用性检查辅助函数。它们可能调用带有
  `torch._dynamo.disable`装饰器的函数，以安全检查不应被追踪的条件。
"""

from __future__ import annotations  # 启用延迟注解求值，支持前向引用

from collections.abc import Callable  # 导入可调用对象抽象基类

import torch  # 导入PyTorch框架

try:  # 尝试导入dynamo禁用装饰器
    from torch._dynamo import disable as _dynamo_disable  # type: ignore[attr-defined]  # 从torch._dynamo导入disable装饰器
except Exception:  # pragma: no cover  # 如果导入失败（例如旧版PyTorch）

    def _dynamo_disable(fn: Callable):  # type: ignore[misc]  # 定义回退的空装饰器
        return fn  # 直接返回原函数，不做任何处理


def _has_oink_op(op_name: str) -> bool:
    """Check if a specific oink op is registered.
    检查指定的oink算子是否已注册。

    Args:
        op_name: 要检查的算子名称。

    Returns:
        如果算子已注册则返回True，否则返回False。
    """
    return hasattr(torch.ops, "oink") and hasattr(torch.ops.oink, op_name)  # 检查torch.ops是否有oink命名空间及指定算子


@_dynamo_disable  # 使用dynamo禁用装饰器，防止在编译追踪时执行
def is_oink_available_for_device(device_index: int) -> bool:
    """Return True if Oink ops are registered and device is SM100+.
    如果Oink算子已注册且设备计算能力为SM100+，则返回True。

    This function is intended to be called during module initialization
    (e.g., in RMSNorm.__init__), not in the forward path.
    此函数用于模块初始化阶段（例如RMSNorm.__init__），而非前向传播路径。

    External plugins are expected to gate registration on SM100+ and
    VLLM_USE_OINK_OPS=1, so if the ops are present they should be usable.
    外部插件应在SM100+和VLLM_USE_OINK_OPS=1条件下注册，因此算子存在即可使用。

    Args:
        device_index: CUDA设备索引。

    Returns:
        如果Oink算子可用于指定设备则返回True。
    """
    if not torch.cuda.is_available():  # 检查CUDA是否可用
        return False  # CUDA不可用时返回False

    try:  # 尝试获取设备计算能力
        major, minor = torch.cuda.get_device_capability(device_index)  # 获取设备的主版本号和次版本号
        sm = 10 * major + minor  # 计算SM版本号（如SM100 = 10*10+0）
        if sm < 100:  # 如果SM版本低于100
            return False  # 不支持，返回False
    except Exception:  # 获取设备能力失败时
        return False  # 返回False

    return _has_oink_op("rmsnorm")  # 检查rmsnorm算子是否可用作为Oink可用性的标志


def has_fused_add_rms_norm() -> bool:
    """Return True if the in-place fused op is registered.
    如果原地融合算子已注册则返回True。

    Returns:
        如果fused_add_rms_norm算子可用则返回True。
    """
    return _has_oink_op("fused_add_rms_norm")  # 检查融合加法RMS归一化算子是否已注册


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Call `torch.ops.oink.rmsnorm`.
    调用`torch.ops.oink.rmsnorm`算子。

    This wrapper is safe to call in torch.compile regions.
    此封装可安全地在torch.compile区域内调用。

    Args:
        x: 输入张量。
        weight: 权重张量。
        eps: 防止除零的小数值epsilon。

    Returns:
        RMS归一化后的张量。
    """
    return torch.ops.oink.rmsnorm(x, weight, eps)  # 调用Oink的RMS归一化算子


def fused_add_rms_norm_(
    x: torch.Tensor,  # 输入张量（原地修改）
    residual: torch.Tensor,  # 残差张量（原地修改）
    weight: torch.Tensor,  # 权重张量
    eps: float,  # epsilon值
) -> None:
    """Call `torch.ops.oink.fused_add_rms_norm` (mutates x and residual).
    调用`torch.ops.oink.fused_add_rms_norm`（原地修改x和residual）。

    Args:
        x: 输入张量，将被原地修改。
        residual: 残差张量，将被原地修改。
        weight: RMS归一化的权重张量。
        eps: 防止除零的小数值epsilon。
    """
    torch.ops.oink.fused_add_rms_norm(x, residual, weight, eps)  # 调用融合的加法RMS归一化算子


def fused_add_rms_norm(
    x: torch.Tensor,  # 输入张量
    residual: torch.Tensor,  # 残差张量
    weight: torch.Tensor,  # 权重张量
    eps: float,  # epsilon值
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper returning (x, residual) after in-place mutation.
    便捷封装，在原地修改后返回(x, residual)元组。

    Args:
        x: 输入张量。
        residual: 残差张量。
        weight: RMS归一化的权重张量。
        eps: 防止除零的小数值epsilon。

    Returns:
        原地修改后的(x, residual)元组。
    """
    fused_add_rms_norm_(x, residual, weight, eps)  # 调用原地融合算子
    return x, residual  # 返回修改后的张量元组
