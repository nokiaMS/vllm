# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable  # 导入 Callable 类型用于类型注解
from typing import TYPE_CHECKING, Any, TypeVar  # 导入类型检查工具

import torch  # 导入 PyTorch 框架
import torch.nn as nn  # 导入 PyTorch 神经网络模块

from vllm.config import VllmConfig, set_current_vllm_config  # 导入 vLLM 配置类和上下文设置函数
from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.lora.request import LoRARequest  # 导入 LoRA 请求数据结构
from vllm.multimodal import MULTIMODAL_REGISTRY  # 导入多模态注册表
from vllm.tracing import instrument  # 导入追踪装饰器
from vllm.utils.import_utils import resolve_obj_by_qualname  # 导入按限定名解析对象的工具函数
from vllm.utils.system_utils import update_environment_variables  # 导入环境变量更新工具
from vllm.v1.kv_cache_interface import KVCacheSpec  # 导入 KV 缓存规格接口

if TYPE_CHECKING:  # 仅在类型检查时导入以下模块，避免循环依赖
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 调度器输出类型
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput  # 模型运行器输出类型
else:
    SchedulerOutput = object  # 运行时用 object 占位
    GrammarOutput = object  # 运行时用 object 占位
    AsyncModelRunnerOutput = object  # 运行时用 object 占位
    ModelRunnerOutput = object  # 运行时用 object 占位

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

_R = TypeVar("_R")  # 定义泛型类型变量，用于 apply_model 方法的返回类型


# Worker 基类，为不同硬件平台（GPU、TPU、XPU 等）提供统一的工作进程接口
# 定义了模型加载、推理执行、KV 缓存管理、LoRA 管理等核心抽象方法
# 同时负责管理分布式通信的控制平面，如向其他 worker 传递请求元数据
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    # 初始化 Worker 基类，接收 vLLM 配置和分布式参数
    def __init__(
        self,
        vllm_config: VllmConfig,  # 完整的 vLLM 配置对象
        local_rank: int,  # 本地设备索引
        rank: int,  # 分布式全局 rank
        distributed_init_method: str,  # 分布式初始化方法（如 TCP 地址）
        is_driver_worker: bool = False,  # 是否为驱动 worker（负责协调）
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        self.vllm_config = vllm_config  # 保存完整配置
        self.model_config = vllm_config.model_config  # 模型配置
        self.cache_config = vllm_config.cache_config  # 缓存配置
        self.lora_config = vllm_config.lora_config  # LoRA 配置
        self.load_config = vllm_config.load_config  # 模型加载配置
        self.parallel_config = vllm_config.parallel_config  # 并行配置
        self.scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.device_config = vllm_config.device_config  # 设备配置
        self.speculative_config = vllm_config.speculative_config  # 推测解码配置
        self.observability_config = vllm_config.observability_config  # 可观测性配置
        self.kv_transfer_config = vllm_config.kv_transfer_config  # KV 缓存传输配置
        self.compilation_config = vllm_config.compilation_config  # 编译配置

        from vllm.platforms import current_platform  # 延迟导入当前平台信息

        self.current_platform = current_platform  # 保存当前运行平台

        self.parallel_config.rank = rank  # 设置并行配置中的 rank
        self.local_rank = local_rank  # 保存本地 rank
        self.rank = rank  # 保存全局 rank
        self.distributed_init_method = distributed_init_method  # 保存分布式初始化方法
        self.is_driver_worker = is_driver_worker  # 保存是否为驱动 worker

        # Device and model state
        self.device: torch.device | None = None  # 设备对象，初始化后赋值
        self.model_runner: nn.Module | None = None  # 模型运行器，初始化后赋值

    # 获取 KV 缓存的规格信息，由子类实现
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError  # 子类必须实现

    # 通过编译或预热准备模型执行，返回编译耗时（秒）
    def compile_or_warm_up_model(self) -> float:
        """Prepare model for execution through compilation/warmup.

        Returns:
            The accumulated compilation time in seconds.
        """
        raise NotImplementedError  # 子类必须实现

    # 基本健康检查，子类可覆写以添加设备特定检查
    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return  # 默认不做任何检查

    # 初始化设备状态，如加载模型或分配设备内存
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError  # 子类必须实现

    # 重置多模态处理缓存
    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)  # 获取重置函数
        if callable(reset_fn):  # 如果函数存在且可调用
            reset_fn()  # 执行重置

    # 获取模型实例，由子类实现
    def get_model(self) -> nn.Module:
        raise NotImplementedError  # 子类必须实现

    # 对 Worker 内部的模型应用给定函数并返回结果
    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())  # 获取模型并应用函数

    # 返回 transformers 风格的模型层次结构视图字符串
    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection  # 延迟导入模型检查工具

        return format_model_inspection(self.get_model())  # 格式化模型结构

    # 将模型加载到目标设备上
    def load_model(self) -> None:
        """Load model onto target device."""
        raise NotImplementedError  # 子类必须实现

    # 执行模型推理，返回模型运行输出或 None（若返回 None 需立即调用 sample_tokens）
    def execute_model(
        self, scheduler_output: SchedulerOutput  # 调度器输出
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError  # 子类必须实现

    # 对模型输出进行 token 采样，应在 execute_model 返回 None 时立即调用
    def sample_tokens(
        self, grammar_output: GrammarOutput  # 语法输出
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError  # 子类必须实现

    # 返回单个缓存块的字节大小，用于推测解码
    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError  # 子类必须实现

    # 添加 LoRA 适配器
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError  # 子类必须实现

    # 移除指定 ID 的 LoRA 适配器
    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError  # 子类必须实现

    # 将指定 LoRA 适配器固定在内存中
    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError  # 子类必须实现

    # 列出当前加载的所有 LoRA 适配器 ID
    def list_loras(self) -> set[int]:
        raise NotImplementedError  # 子类必须实现

    @property
    # 从模型配置中获取词汇表大小
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()  # 返回词汇表大小

    # 清理 Worker 持有的资源
    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return  # 默认不做任何清理


# Worker 包装器基类，代表执行器/引擎中的一个进程
# 负责延迟初始化实际 Worker 实例，管理 Worker 的生命周期
# 设计思路：先实例化 Wrapper（轻量级），待环境变量配置完成后再创建真正的 Worker
# 支持动态注入 worker_extension_cls 以扩展 Worker 功能
class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    # 初始化 Worker 包装器，接收 RPC rank 和可选的全局 rank
    def __init__(
        self,
        rpc_rank: int = 0,  # Worker 在执行器中的 rank
        global_rank: int | None = None,  # 全局 rank（多执行器场景下可能不同于 rpc_rank）
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank  # 保存 RPC rank
        self.global_rank = self.rpc_rank if global_rank is None else global_rank  # 计算全局 rank

        # Initialized after init_worker is called
        self.worker: WorkerBase  # Worker 实例，init_worker 后赋值
        self.vllm_config: VllmConfig  # vLLM 配置，init_worker 后赋值

    # 关闭 Worker 并释放资源
    def shutdown(self) -> None:
        if self.worker is not None:  # 如果 Worker 已初始化
            self.worker.shutdown()  # 调用 Worker 的关闭方法

    # 更新当前 rank 对应的环境变量
    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],  # 每个 rank 的环境变量字典列表
    ) -> None:
        envs = envs_list[self.rpc_rank]  # 获取当前 rank 的环境变量
        update_environment_variables(envs)  # 应用环境变量

    @instrument(span_name="Worker init")  # 添加追踪 span
    # 初始化真正的 Worker 实例，注入通用逻辑后调用 Worker 构造函数
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]  # 获取当前 rank 的初始化参数

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")  # 提取 vLLM 配置
        assert vllm_config is not None, (  # 断言配置必须存在
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config  # 保存配置

        vllm_config.enable_trace_function_call_for_thread()  # 启用函数调用追踪

        from vllm.plugins import load_general_plugins  # 延迟导入插件加载器

        load_general_plugins()  # 加载通用插件

        parallel_config = vllm_config.parallel_config  # 获取并行配置
        if isinstance(parallel_config.worker_cls, str):  # 如果 worker 类是字符串形式
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(  # 按限定名解析 Worker 类
                parallel_config.worker_cls
            )
        else:
            raise ValueError(  # 不再支持直接传递类对象
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:  # 如果配置了 Worker 扩展类
            worker_extension_cls = resolve_obj_by_qualname(  # 解析扩展类
                parallel_config.worker_extension_cls
            )
            extended_calls = []  # 记录扩展的方法名
            if worker_extension_cls not in worker_class.__bases__:  # 如果尚未继承扩展类
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):  # 遍历扩展类的所有属性
                    if attr.startswith("__"):  # 跳过双下划线方法
                        continue
                    assert not hasattr(worker_class, attr), (  # 检查属性冲突
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):  # 如果是可调用方法
                        extended_calls.append(attr)  # 记录扩展方法名
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (  # 动态添加扩展类到继承链
                    worker_extension_cls,
                )
                logger.info(  # 记录注入信息
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)  # 提取共享锁参数
        if shared_worker_lock is None:  # 如果没有提供共享锁
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config  # 获取多模态配置
            if mm_config and mm_config.mm_processor_cache_type == "shm":  # 如果需要共享内存缓存
                raise ValueError(msg)  # 抛出错误
            else:
                logger.warning_once(msg)  # 仅警告一次

            self.mm_receiver_cache = None  # 不使用多模态接收缓存
        else:
            self.mm_receiver_cache = (  # 创建多模态接收缓存
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):  # 设置当前 vLLM 配置上下文
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)  # 创建真正的 Worker 实例

    # 从 KV 缓存配置初始化 Worker
    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]  # 获取当前 rank 的 KV 缓存配置
        assert self.vllm_config is not None  # 断言配置已初始化
        with set_current_vllm_config(self.vllm_config):  # 设置配置上下文
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore  # 调用 Worker 的初始化方法

    # 初始化设备
    def init_device(self):
        assert self.vllm_config is not None  # 断言配置已初始化
        with set_current_vllm_config(self.vllm_config):  # 设置配置上下文
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore  # 调用 Worker 的设备初始化方法

    # 属性代理：将未定义的属性访问转发给内部 Worker 实例
    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)  # 委托给 Worker

    # 将多模态缓存应用到调度器输出中的新请求
    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache  # 获取多模态缓存
        if mm_cache is None:  # 如果没有缓存则跳过
            return

        for req_data in scheduler_output.scheduled_new_reqs:  # 遍历新调度的请求
            req_data.mm_features = mm_cache.get_and_update_features(  # 获取并更新多模态特征
                req_data.mm_features
            )

    # 执行模型推理，先应用多模态缓存再委托给 Worker
    def execute_model(
        self, scheduler_output: SchedulerOutput  # 调度器输出
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)  # 应用多模态缓存

        return self.worker.execute_model(scheduler_output)  # 委托给 Worker 执行

    # 重置多模态缓存（包括接收端缓存和 Worker 端缓存）
    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache  # 获取接收端缓存
        if mm_receiver_cache is not None:  # 如果存在
            mm_receiver_cache.clear_cache()  # 清除缓存

        self.worker.reset_mm_cache()  # 调用 Worker 的缓存重置
