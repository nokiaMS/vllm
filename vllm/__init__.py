# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: 一个用于大语言模型的高吞吐量、低内存占用的推理引擎"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
# 版本模块应作为独立库，始终优先导入版本库。此假设对某些自定义功能至关重要。
from .version import __version__, __version_tuple__  # isort:skip  # 从版本模块导入版本号和版本元组

import typing  # 导入类型检查模块

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
# 环境变量覆盖模块应在其他所有模块之前导入，以确保在导入其他模块前环境变量已被设置。
import vllm.env_override  # noqa: F401  # 导入环境变量覆盖模块（仅用于副作用，不直接使用）

MODULE_ATTRS = {  # 模块属性映射字典，用于延迟加载各模块中的类和函数
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",  # 异步引擎参数类
    "EngineArgs": ".engine.arg_utils:EngineArgs",  # 引擎参数类
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",  # 异步LLM引擎类
    "LLMEngine": ".engine.llm_engine:LLMEngine",  # LLM引擎类
    "LLM": ".entrypoints.llm:LLM",  # LLM入口类
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",  # 初始化Ray集群函数
    "PromptType": ".inputs:PromptType",  # 提示类型
    "TextPrompt": ".inputs:TextPrompt",  # 文本提示类型
    "TokensPrompt": ".inputs:TokensPrompt",  # Token提示类型
    "ModelRegistry": ".model_executor.models:ModelRegistry",  # 模型注册表类
    "SamplingParams": ".sampling_params:SamplingParams",  # 采样参数类
    "PoolingParams": ".pooling_params:PoolingParams",  # 池化参数类
    "ClassificationOutput": ".outputs:ClassificationOutput",  # 分类输出类
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",  # 分类请求输出类
    "CompletionOutput": ".outputs:CompletionOutput",  # 补全输出类
    "EmbeddingOutput": ".outputs:EmbeddingOutput",  # 嵌入输出类
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",  # 嵌入请求输出类
    "PoolingOutput": ".outputs:PoolingOutput",  # 池化输出类
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",  # 池化请求输出类
    "RequestOutput": ".outputs:RequestOutput",  # 请求输出类
    "ScoringOutput": ".outputs:ScoringOutput",  # 评分输出类
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",  # 评分请求输出类
}

if typing.TYPE_CHECKING:  # 仅在类型检查时执行以下导入（用于IDE智能提示和静态分析）
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs  # 导入异步引擎参数和引擎参数
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # 导入异步LLM引擎
    from vllm.engine.llm_engine import LLMEngine  # 导入LLM引擎
    from vllm.entrypoints.llm import LLM  # 导入LLM入口类
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt  # 导入提示相关类型
    from vllm.model_executor.models import ModelRegistry  # 导入模型注册表
    from vllm.outputs import (  # 导入各类输出类型
        ClassificationOutput,  # 分类输出
        ClassificationRequestOutput,  # 分类请求输出
        CompletionOutput,  # 补全输出
        EmbeddingOutput,  # 嵌入输出
        EmbeddingRequestOutput,  # 嵌入请求输出
        PoolingOutput,  # 池化输出
        PoolingRequestOutput,  # 池化请求输出
        RequestOutput,  # 请求输出
        ScoringOutput,  # 评分输出
        ScoringRequestOutput,  # 评分请求输出
    )
    from vllm.pooling_params import PoolingParams  # 导入池化参数
    from vllm.sampling_params import SamplingParams  # 导入采样参数
    from vllm.v1.executor.ray_utils import initialize_ray_cluster  # 导入Ray集群初始化函数
else:  # 非类型检查时，使用延迟加载机制

    def __getattr__(name: str) -> typing.Any:
        """模块级别的属性延迟加载函数。

        当访问本模块中未直接定义的属性时，此函数会被调用。
        它会根据 MODULE_ATTRS 字典中的映射关系，动态导入并返回对应的模块属性。
        这种延迟加载机制可以显著减少初始导入时间，提升启动性能。

        Args:
            name: 要访问的属性名称。

        Returns:
            对应模块中的属性对象。

        Raises:
            AttributeError: 当请求的属性名不在 MODULE_ATTRS 中时抛出。
        """
        from importlib import import_module  # 导入模块动态加载函数

        if name in MODULE_ATTRS:  # 如果属性名在映射字典中
            module_name, attr_name = MODULE_ATTRS[name].split(":")  # 分割模块路径和属性名
            module = import_module(module_name, __package__)  # 动态导入目标模块
            return getattr(module, attr_name)  # 返回模块中的目标属性
        else:  # 如果属性名不在映射字典中
            raise AttributeError(f"module {__package__} has no attribute {name}")  # 抛出属性错误


__all__ = [  # 定义模块的公开接口列表，控制 from vllm import * 的导出内容
    "__version__",  # 版本号字符串
    "__version_tuple__",  # 版本号元组
    "LLM",  # LLM入口类
    "ModelRegistry",  # 模型注册表
    "PromptType",  # 提示类型
    "TextPrompt",  # 文本提示
    "TokensPrompt",  # Token提示
    "SamplingParams",  # 采样参数
    "RequestOutput",  # 请求输出
    "CompletionOutput",  # 补全输出
    "PoolingOutput",  # 池化输出
    "PoolingRequestOutput",  # 池化请求输出
    "EmbeddingOutput",  # 嵌入输出
    "EmbeddingRequestOutput",  # 嵌入请求输出
    "ClassificationOutput",  # 分类输出
    "ClassificationRequestOutput",  # 分类请求输出
    "ScoringOutput",  # 评分输出
    "ScoringRequestOutput",  # 评分请求输出
    "LLMEngine",  # LLM引擎
    "EngineArgs",  # 引擎参数
    "AsyncLLMEngine",  # 异步LLM引擎
    "AsyncEngineArgs",  # 异步引擎参数
    "initialize_ray_cluster",  # 初始化Ray集群
    "PoolingParams",  # 池化参数
]
