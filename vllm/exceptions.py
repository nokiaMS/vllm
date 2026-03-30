# SPDX-License-Identifier: Apache-2.0  # 许可证标识符：Apache-2.0开源协议
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

"""vLLM的自定义异常类模块。

Custom exceptions for vLLM."""

from typing import Any  # 导入Any类型，用于类型注解


class VLLMValidationError(ValueError):
    """vLLM专用的验证错误类，用于请求验证失败时抛出。

    vLLM-specific validation error for request validation failures.

    Args:
        message: 描述验证失败的错误消息。The error message describing the validation failure.
        parameter: 可选，验证失败的参数名称。Optional parameter name that failed validation.
        value: 可选，验证被拒绝的值。Optional value that was rejected during validation.
    """

    def __init__(
        self,
        message: str,  # 错误消息字符串
        *,
        parameter: str | None = None,  # 可选的参数名称，默认为None
        value: Any = None,  # 可选的被拒绝的值，默认为None
    ) -> None:
        """初始化验证错误实例。"""
        super().__init__(message)  # 调用父类ValueError的构造函数
        self.parameter = parameter  # 保存验证失败的参数名
        self.value = value  # 保存被拒绝的值

    def __str__(self):
        """返回错误的字符串表示，包含参数名和值的附加信息。"""
        base = super().__str__()  # 获取父类的字符串表示（即错误消息）
        extras = []  # 初始化附加信息列表
        if self.parameter is not None:  # 如果存在参数名
            extras.append(f"parameter={self.parameter}")  # 添加参数名信息
        if self.value is not None:  # 如果存在被拒绝的值
            extras.append(f"value={self.value}")  # 添加值信息
        return f"{base} ({', '.join(extras)})" if extras else base  # 返回带附加信息的完整错误字符串


class VLLMNotFoundError(Exception):
    """vLLM专用的未找到错误类。

    vLLM-specific NotFoundError"""

    pass  # 空实现，仅用作基类标识


class LoRAAdapterNotFoundError(VLLMNotFoundError):
    """当LoRA适配器未找到时抛出的异常。

    Exception raised when a LoRA adapter is not found.

    当请求的LoRA适配器在系统中不存在时，会抛出此异常。
    This exception is thrown when a requested LoRA adapter does not exist
    in the system.

    Attributes:
        message: 描述异常的错误消息字符串。The error message string describing the exception
    """

    message: str  # 错误消息属性的类型注解

    def __init__(
        self,
        lora_name: str,  # LoRA适配器的名称
        lora_path: str,  # LoRA适配器的路径
    ) -> None:
        """初始化LoRA适配器未找到错误实例。"""
        message = f"Loading lora {lora_name} failed: No adapter found for {lora_path}"  # 构造错误消息
        self.message = message  # 保存错误消息

    def __str__(self):
        """返回错误消息的字符串表示。"""
        return self.message  # 返回错误消息
