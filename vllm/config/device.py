# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from dataclasses import field  # 导入dataclass的field函数
from typing import Any, Literal  # 导入Any和Literal类型

import torch  # 导入PyTorch框架
from pydantic import ConfigDict, SkipValidation  # 导入pydantic的配置字典和跳过验证类型

from vllm.config.utils import config  # 导入config装饰器
from vllm.utils.hashing import safe_hash  # 导入安全哈希函数

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]  # 设备类型字面量定义


@config(config=ConfigDict(arbitrary_types_allowed=True))  # 允许任意类型的pydantic数据类
class DeviceConfig:
    """vLLM执行所用设备的配置类。"""

    device: SkipValidation[Device | torch.device | None] = "auto"  # 设备类型，默认自动检测
    """Device type for vLLM execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    device_type: str = field(init=False)  # 从当前平台获取的设备类型字符串，初始化时自动设置
    """Device type from the current platform. This is set in
    `__post_init__`."""

    def compute_hash(self) -> str:
        """
        计算唯一标识此配置的哈希值。

        警告：每当向此配置添加新字段时，如果影响计算图，
        请确保将其包含在因子列表中。
        """
        # 无需考虑任何因子
        # 设备/平台信息将由torch/vllm自动汇总
        factors: list[Any] = []  # 空因子列表
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()  # 计算哈希值
        return hash_str  # 返回哈希字符串

    def __post_init__(self):
        """初始化后处理：检测或设置设备类型。"""
        if self.device == "auto":  # 如果设备类型为自动检测
            # 自动设备类型检测
            from vllm.platforms import current_platform  # 导入当前平台信息

            self.device_type = current_platform.device_type  # 从平台获取设备类型
            if not self.device_type:  # 如果检测失败
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue."
                )
        else:  # 设备类型由用户显式指定
            # 设备类型被显式赋值
            if isinstance(self.device, str):  # 如果是字符串类型
                self.device_type = self.device  # 直接使用字符串
            elif isinstance(self.device, torch.device):  # 如果是torch.device类型
                self.device_type = self.device.type  # 获取设备类型字符串

        # 某些设备类型需要在CPU上处理输入
        if self.device_type in ["tpu"]:  # TPU设备需要特殊处理
            self.device = None  # 设置为None
        else:
            # 使用设备类型设置设备
            self.device = torch.device(self.device_type)  # 创建torch设备对象
