# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from collections.abc import Callable  # 导入Callable抽象基类
from typing import Any, Literal  # 导入Any和Literal类型

from pydantic import Field, field_validator  # 导入pydantic的Field和字段验证器

from vllm.config.utils import config  # 导入config装饰器
from vllm.utils.hashing import safe_hash  # 导入安全哈希函数

MoEBackend = Literal[  # MoE后端类型字面量定义
    "auto",
    "triton",
    "deep_gemm",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "aiter",
]


@config  # 使用config装饰器创建pydantic数据类
class KernelConfig:
    """内核选择和预热行为的配置类。"""

    enable_flashinfer_autotune: bool = Field(default=None)  # 如果为True，在内核预热时运行FlashInfer自动调优
    """If True, run FlashInfer autotuning during kernel warmup."""

    moe_backend: MoEBackend = "auto"  # MoE专家计算内核的后端，默认自动选择
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware\n
    - "triton": Use Triton-based fused MoE kernels\n
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)\n
    - "cutlass": Use vLLM CUTLASS kernels\n
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels\n
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels\n
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)\n
    - "marlin": Use Marlin kernels (weight-only quantization)\n
    - "aiter": Use AMD AITer kernels (ROCm only)"""

    @field_validator("moe_backend", mode="before")  # 字段验证器，在验证前处理
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        """标准化MoE后端名称：转小写并将连字符替换为下划线。"""
        if isinstance(value, str):  # 如果值是字符串
            return value.lower().replace("-", "_")  # 转小写并替换连字符
        return value  # 非字符串直接返回

    def compute_hash(self) -> str:
        """
        计算唯一标识此配置的哈希值。

        警告：每当向此配置添加新字段时，如果影响计算图，
        请确保将其包含在因子列表中。
        """
        # 无需考虑任何因子
        # 此配置不会影响计算图
        factors: list[Any] = []  # 空因子列表
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()  # 计算哈希值
        return hash_str  # 返回哈希字符串

    @field_validator("enable_flashinfer_autotune", mode="wrap")  # 包装模式的字段验证器
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """当初始化延迟时，如果值为None则跳过验证。"""
        if value is None:  # 如果值为None
            return value  # 直接返回None
        return handler(value)  # 否则执行正常验证
