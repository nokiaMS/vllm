# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
from typing import TYPE_CHECKING  # 导入类型检查标志，用于静态类型分析

from vllm.triton_utils.importing import (  # 从importing模块导入Triton相关组件
    HAS_TRITON,  # 布尔值：系统中是否安装了Triton
    TritonLanguagePlaceholder,  # Triton语言模块的占位符类
    TritonPlaceholder,  # Triton主模块的占位符类
)

if TYPE_CHECKING or HAS_TRITON:  # 如果处于类型检查模式或Triton已安装
    import triton  # 导入真实的triton模块
    import triton.language as tl  # 导入triton语言模块，简称tl
    import triton.language.extra.libdevice as tldevice  # 导入triton的libdevice扩展库
else:  # 否则（Triton未安装且非类型检查模式）
    triton = TritonPlaceholder()  # 使用Triton占位符代替真实模块
    tl = TritonLanguagePlaceholder()  # 使用Triton语言占位符代替真实模块
    tldevice = TritonLanguagePlaceholder()  # 使用Triton语言占位符代替libdevice

LOG2E = 1.4426950408889634  # log2(e) 的常量值，即以2为底e的对数
LOGE2 = 0.6931471805599453  # ln(2) 的常量值，即以e为底2的自然对数

__all__ = ["HAS_TRITON", "triton", "tl", "tldevice", "LOG2E", "LOGE2"]  # 定义模块的公开接口列表
