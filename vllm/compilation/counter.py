# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy  # 导入深拷贝模块
import dataclasses  # 导入数据类装饰器模块
from collections.abc import Generator  # 导入生成器类型
from contextlib import contextmanager  # 导入上下文管理器装饰器
from typing import Any  # 导入Any类型注解


@dataclasses.dataclass
class CompilationCounter:
    """编译计数器，用于跟踪编译过程中各个阶段的统计数据。"""

    num_models_seen: int = 0  # 已看到的模型数量
    num_graphs_seen: int = 0  # 已看到的计算图数量
    # including the splitting ops
    num_piecewise_graphs_seen: int = 0  # 已看到的分段图数量（包括分割操作）
    # not including the splitting ops
    num_piecewise_capturable_graphs_seen: int = 0  # 可捕获的分段图数量（不含分割操作）
    num_backend_compilations: int = 0  # 后端编译次数
    # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    num_gpu_runner_capture_triggers: int = 0  # GPU运行器触发CUDA图捕获的次数
    # Number of CUDAGraphs captured
    num_cudagraph_captured: int = 0  # 已捕获的CUDA图数量
    # InductorAdapter.compile calls
    num_inductor_compiles: int = 0  # Inductor编译器的编译调用次数
    # EagerAdapter.compile calls
    num_eager_compiles: int = 0  # Eager编译器的编译调用次数
    # The number of time vLLM's compiler cache entry was updated
    num_cache_entries_updated: int = 0  # vLLM编译器缓存条目更新次数
    # The number of standalone_compile compiled artifacts saved
    num_compiled_artifacts_saved: int = 0  # 独立编译产物保存次数
    # The number of standalone_compile compiled artifacts loaded from cache
    num_compiled_artifacts_loaded: int = 0  # 从缓存加载的编译产物次数
    # The number of AOT compile invocations
    num_aot_compiles: int = 0  # AOT编译调用次数
    # The number of AOT compiled artifacts saved to disk
    num_aot_artifacts_saved: int = 0  # AOT编译产物保存到磁盘的次数
    # The number of AOT compiled artifacts loaded from disk
    num_aot_artifacts_loaded: int = 0  # 从磁盘加载AOT编译产物的次数
    # Number of times a model was loaded with CompilationMode.STOCK_TORCH_COMPILE
    stock_torch_compile_count: int = 0  # 使用标准torch.compile模式加载模型的次数

    def clone(self) -> "CompilationCounter":
        """克隆当前计数器实例，返回深拷贝。"""
        return copy.deepcopy(self)  # 返回当前对象的深拷贝

    @contextmanager
    def expect(self, **kwargs: Any) -> Generator[None, None, None]:
        """上下文管理器，用于验证编译计数器在代码块执行前后的差异是否符合预期。"""
        old = self.clone()  # 保存当前计数器快照
        yield  # 执行上下文块
        for k, v in kwargs.items():  # 遍历期望的差异值
            assert getattr(self, k) - getattr(old, k) == v, (  # 验证差异是否匹配
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


compilation_counter = CompilationCounter()  # 全局编译计数器单例
