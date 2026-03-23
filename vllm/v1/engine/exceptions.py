# SPDX-License-Identifier: Apache-2.0  # 许可证标识：Apache-2.0 开源协议
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者所有


# [中文注释] generate() 调用失败时抛出的异常，属于可恢复错误（不影响引擎继续运行）
class EngineGenerateError(Exception):  # 定义引擎生成错误类，继承自内置异常基类 Exception
    """Raised when a AsyncLLM.generate() fails. Recoverable."""
    """当 AsyncLLM.generate() 调用失败时抛出此异常。此错误是可恢复的，不会导致引擎终止。"""

    pass  # 不需要额外的属性或方法，直接继承父类即可


# [中文注释] Engine Core 进程崩溃时抛出的异常，属于不可恢复错误。
#   suppress_context=True 用于在 LLMEngine 场景下隐藏无关的 ZMQError 堆栈
class EngineDeadError(Exception):  # 定义引擎死亡错误类，继承自内置异常基类 Exception
    """Raised when the EngineCore dies. Unrecoverable."""
    """当 EngineCore 进程崩溃时抛出此异常。此错误是不可恢复的，引擎无法继续运行。"""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):  # 构造函数，接受可变位置参数、是否抑制异常上下文的标志和可变关键字参数
        """初始化引擎死亡异常。

        参数:
            *args: 传递给父类 Exception 的可变位置参数
            suppress_context: 是否抑制异常链上下文（隐藏前一个异常的堆栈信息），默认为 False
            **kwargs: 传递给父类 Exception 的可变关键字参数
        """
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501  # 定义引擎死亡时的错误提示消息，提示用户查看上方堆栈获取根本原因

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)  # 调用父类构造函数，将错误消息和其他参数传递给 Exception 基类
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        # 在使用 LLMEngine 时，通过抑制不相关的 ZMQError 使堆栈跟踪更清晰
        self.__suppress_context__ = suppress_context  # 设置是否抑制异常上下文链，True 时不显示 "During handling of the above exception" 信息
