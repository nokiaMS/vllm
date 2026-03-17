# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.

    中文：WorkerBase 定义了统一的工作进程接口，用于隔离不同硬件后端实现，
    同时抽象控制面通信（例如在 worker 间同步请求元数据）。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        中文：初始化 Worker 的通用配置与运行时状态。

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities

        中文参数说明：
            vllm_config：完整的 vLLM 配置。
            local_rank：当前节点内设备编号。
            rank：分布式全局 rank。
            distributed_init_method：分布式初始化方式。
            is_driver_worker：是否承担 driver 侧职责。
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation.

        中文：返回 KV Cache 的实现规格定义。
        """
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> float:
        """Prepare model for execution through compilation/warmup.

        中文：通过编译或预热让模型进入可执行状态。

        Returns:
            The accumulated compilation time in seconds.

        中文返回：累计编译耗时（秒）。
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks).

        中文：基础健康检查；设备相关检查可由子类覆盖。
        """
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.

        中文：初始化设备侧状态，例如加载模型与设备内存分配。
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        """Reset multimodal cache if the model runner provides the hook.

        中文：若 model_runner 实现了 reset_mm_cache，则调用其重置多模态缓存。
        """
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        """Return the underlying model instance.

        中文：返回当前 Worker 持有的底层模型对象。
        """
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker.

        中文：在 Worker 内部模型上执行传入函数，并返回函数结果。
        """
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model.

        中文：返回类似 transformers 的模型层级结构描述。
        """
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self) -> None:
        """Load model onto target device.

        中文：将模型加载到目标设备。
        """
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        中文：若该方法返回 None，调用方需要立刻调用 sample_tokens 获取
        ModelRunnerOutput。

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.

        中文：当结构化输出并行机制重构时，该接口约定未来可能调整。
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None.

        中文：仅在 execute_model 返回 None 时，需紧接着调用本方法采样 token。
        """
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.

        中文：返回单个缓存块的字节大小，用于投机解码等场景。
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Attach a LoRA adapter to the worker.

        中文：向当前 Worker 挂载一个 LoRA 适配器。
        """
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRA adapter from the worker.

        中文：从当前 Worker 卸载指定 LoRA 适配器。
        """
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        """Pin a LoRA adapter to avoid eviction.

        中文：固定指定 LoRA，避免其被回收。
        """
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        """List currently loaded LoRA adapter IDs.

        中文：列出当前已加载的 LoRA ID 集合。
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration.

        中文：从模型配置中读取词表大小。
        """
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker.

        中文：清理 Worker 持有的资源。
        """
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
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
        self.rpc_rank = rpc_rank
        self.global_rank = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
