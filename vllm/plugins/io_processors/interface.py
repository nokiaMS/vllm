# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings  # 导入警告模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections.abc import AsyncGenerator, Sequence  # 导入异步生成器和序列抽象基类
from typing import Generic, TypeVar  # 导入泛型支持和类型变量

from vllm.config import VllmConfig  # 导入vLLM总配置类
from vllm.inputs.data import PromptType  # 导入提示类型
from vllm.outputs import PoolingRequestOutput  # 导入池化请求输出类
from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.renderers import BaseRenderer  # 导入基础渲染器类
from vllm.sampling_params import SamplingParams  # 导入采样参数类

IOProcessorInput = TypeVar("IOProcessorInput")  # 定义IO处理器输入的类型变量
IOProcessorOutput = TypeVar("IOProcessorOutput")  # 定义IO处理器输出的类型变量


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """Abstract interface for pre/post-processing of engine I/O.
    引擎输入/输出预处理和后处理的抽象接口。

    IO处理器用于在请求进入引擎之前进行预处理，
    以及在引擎输出返回客户端之前进行后处理。
    """

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):  # 初始化IO处理器
        """
        初始化IO处理器。
        Args:
            vllm_config: vLLM配置对象
            renderer: 基础渲染器实例
        """
        super().__init__()  # 调用父类初始化

        self.vllm_config = vllm_config  # 存储vLLM配置

    def parse_data(self, data: object) -> IOProcessorInput:  # 解析输入数据
        """解析输入数据为IO处理器可处理的格式。"""
        if callable(parse_request := getattr(self, "parse_request", None)):  # 如果存在旧版parse_request方法
            warnings.warn(  # 发出弃用警告
                "`parse_request` has been renamed to `parse_data`. "
                "Please update your IO Processor Plugin to use the new name. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return parse_request(data)  # type: ignore  # 调用旧方法

        raise NotImplementedError  # 子类必须实现此方法

    def merge_sampling_params(  # 合并采样参数
        self,
        params: SamplingParams | None = None,  # 可选的采样参数
    ) -> SamplingParams:
        """合并并验证采样参数。"""
        if callable(  # 如果存在旧版validate_or_generate_params方法
            validate_or_generate_params := getattr(
                self, "validate_or_generate_params", None
            )
        ):
            warnings.warn(  # 发出弃用警告
                "`validate_or_generate_params` has been split into "
                "`merge_sampling_params` and `merge_pooling_params`."
                "Please update your IO Processor Plugin to use the new methods. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return validate_or_generate_params(params)  # type: ignore  # 调用旧方法

        return params or SamplingParams()  # 返回参数或默认采样参数

    def merge_pooling_params(  # 合并池化参数
        self,
        params: PoolingParams | None = None,  # 可选的池化参数
    ) -> PoolingParams:
        """合并并验证池化参数。"""
        if callable(  # 如果存在旧版validate_or_generate_params方法
            validate_or_generate_params := getattr(
                self, "validate_or_generate_params", None
            )
        ):
            warnings.warn(  # 发出弃用警告
                "`validate_or_generate_params` has been split into "
                "`merge_sampling_params` and `merge_pooling_params`."
                "Please update your IO Processor Plugin to use the new methods. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return validate_or_generate_params(params)  # type: ignore  # 调用旧方法

        return params or PoolingParams(task="plugin")  # 返回参数或默认池化参数

    @abstractmethod  # 抽象方法，子类必须实现
    def pre_process(  # 预处理方法
        self,
        prompt: IOProcessorInput,  # 输入提示
        request_id: str | None = None,  # 请求ID
        **kwargs,  # 额外关键字参数
    ) -> PromptType | Sequence[PromptType]:
        """预处理输入提示，将其转换为引擎可接受的格式。"""
        raise NotImplementedError  # 子类必须实现

    async def pre_process_async(  # 异步预处理方法
        self,
        prompt: IOProcessorInput,  # 输入提示
        request_id: str | None = None,  # 请求ID
        **kwargs,  # 额外关键字参数
    ) -> PromptType | Sequence[PromptType]:
        """异步预处理输入提示，默认调用同步版本。"""
        return self.pre_process(prompt, request_id, **kwargs)  # 默认调用同步方法

    @abstractmethod  # 抽象方法，子类必须实现
    def post_process(  # 后处理方法
        self,
        model_output: Sequence[PoolingRequestOutput],  # 模型输出序列
        request_id: str | None = None,  # 请求ID
        **kwargs,  # 额外关键字参数
    ) -> IOProcessorOutput:
        """后处理模型输出，将其转换为最终返回格式。"""
        raise NotImplementedError  # 子类必须实现

    async def post_process_async(  # 异步后处理方法
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],  # 异步模型输出生成器
        request_id: str | None = None,  # 请求ID
        **kwargs,  # 额外关键字参数
    ) -> IOProcessorOutput:
        """异步后处理模型输出。"""
        # We cannot guarantee outputs are returned in the same order they were
        # fed to vLLM.
        # Let's sort them by id before post_processing
        sorted_output = sorted(  # 按ID排序输出结果，因为不能保证返回顺序与输入顺序一致
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]  # 提取排序后的输出
        return self.post_process(collected_output, request_id=request_id, **kwargs)  # 调用同步后处理方法
