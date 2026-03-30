# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable  # 导入可调用对象类型
from typing import Any, Protocol  # 导入Any类型和Protocol协议

from vllm.config import CUDAGraphMode, VllmConfig  # 导入CUDA图模式和vLLM配置


class AbstractStaticGraphWrapper(Protocol):
    """静态图包装器接口协议。

    允许平台将可调用对象包装为静态图进行捕获和执行。
    """

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        **kwargs: Any,
    ) -> None:
        """初始化StaticGraphWrapper类，包含图捕获和执行相关配置。

        Args:
            runnable (Callable): 要包装和捕获的可调用对象。
            vllm_config (VllmConfig): vLLM的全局配置。
            runtime_mode (CUDAGraphMode): 静态图运行时模式。
                参见vllm/config.py中的CUDAGraphMode。
                注意只有枚举子集 `NONE`、`PIECEWISE` 和 `FULL`
                被用作CUDA图调度的具体运行时模式。
        Keyword Args:
            kwargs: 平台特定配置的额外关键字参数。
        """
        raise NotImplementedError  # 抛出未实现异常

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """执行被包装的可调用对象。

        如果ForwardContext中的当前运行时模式与此实例的运行时模式匹配，
        它将回放CUDA图，或者如果尚未捕获则使用可调用对象进行捕获。
        否则，直接调用原始可调用对象。

        Args:
            *args: 要传递给可调用对象的可变长度输入参数。
            **kwargs: 要传递给可调用对象的关键字参数。

        Returns:
            Any: 执行可调用对象后的输出。
        """
        raise NotImplementedError  # 抛出未实现异常
