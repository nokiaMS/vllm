# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc  # 垃圾回收模块，用于手动触发内存清理
import os  # 操作系统接口，用于环境变量和路径操作
from collections.abc import Callable  # 可调用对象类型提示
from contextlib import AbstractContextManager, nullcontext  # 上下文管理器工具
from datetime import timedelta  # 时间差类型，用于设置分布式超时
from types import NoneType  # NoneType 类型，用于 isinstance 检查
from typing import TYPE_CHECKING, Any  # 类型检查标志和通用类型

import numpy as np  # NumPy 数组库
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块

import vllm.envs as envs  # vLLM 环境变量配置
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config  # vLLM 配置类和 CUDA Graph 模式枚举
from vllm.config.compilation import CompilationMode  # 编译模式枚举（如 VLLM_COMPILE）
from vllm.distributed import (  # 分布式通信初始化工具
    ensure_model_parallel_initialized,  # 确保张量/流水线并行组已创建
    init_distributed_environment,  # 初始化 NCCL 分布式环境
    set_custom_all_reduce,  # 设置自定义 AllReduce 策略
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized  # 编码器-上下文传输初始化（EPD 解聚模式）
from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb  # 弹性专家并行的环境变量覆盖
from vllm.distributed.kv_transfer import (  # KV 缓存传输相关工具（解聚推理）
    ensure_kv_transfer_initialized,  # 确保 KV 传输组已初始化
    ensure_kv_transfer_shutdown,  # 确保 KV 传输组已关闭
    get_kv_transfer_group,  # 获取 KV 传输组实例
    has_kv_transfer_group,  # 检查 KV 传输组是否存在
)
from vllm.distributed.parallel_state import (  # 分布式并行状态管理
    Handle,  # 异步通信句柄类型
    get_pp_group,  # 获取流水线并行组
    get_tp_group,  # 获取张量并行组
)
from vllm.distributed.weight_transfer import WeightTransferEngineFactory  # 权重传输引擎工厂（在线训练权重更新）
from vllm.logger import init_logger  # 日志初始化工具
from vllm.lora.request import LoRARequest  # LoRA 适配器请求类
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup  # 内核预热函数
from vllm.platforms import current_platform  # 当前平台抽象（CUDA/ROCm 等）
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper  # 性能分析器封装
from vllm.sequence import IntermediateTensors  # 流水线并行中间张量容器
from vllm.tasks import SupportedTask  # 支持的任务类型枚举
from vllm.tracing import instrument  # OpenTelemetry 追踪装饰器
from vllm.utils.mem_constants import GiB_bytes  # GiB 字节常量
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling  # 内存快照和格式化工具
from vllm.utils.torch_utils import set_random_seed  # 设置全局随机种子
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 调度器输出类型
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec  # KV 缓存配置和规格接口
from vllm.v1.outputs import (  # 模型运行器输出类型
    AsyncModelRunnerOutput,  # 异步模型输出（DBO 双缓冲模式）
    DraftTokenIds,  # 投机解码的草稿 token ID
    ModelRunnerOutput,  # 同步模型输出
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats  # 迭代详情计算和使用统计上报
from vllm.v1.worker.utils import is_residual_scattered_for_sp  # 检查残差是否为序列并行分片
from vllm.v1.worker.worker_base import WorkerBase  # Worker 基类
from vllm.v1.worker.workspace import init_workspace_manager  # 工作空间管理器初始化

from ...model_executor.model_loader import TensorizerLoader  # Tensorizer 模型加载器（高效序列化格式）
from .gpu.warmup import warmup_kernels  # GPU 内核预热函数（V2 ModelRunner 使用）
from .utils import request_memory  # 计算请求的 GPU 内存量

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环依赖
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig  # Tensorizer 配置类型
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # GPU 模型运行器类型


# 支持惰性通信同步的中间张量类，用于流水线并行（PP）。
# 在访问 .tensors 属性时自动等待异步通信完成，
# 避免在非必要时阻塞，实现计算与通信的重叠。
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],  # 中间张量字典（键为张量名，值为张量数据）
        comm_handles: list[Handle] | None = None,  # 异步通信句柄列表（用于等待通信完成）
        comm_postprocess: list[Callable[[], None]] | None = None,  # 通信完成后的后处理回调列表
    ) -> None:
        super().__init__(tensors)  # 调用父类构造函数，存储张量字典
        self._comm_handles = comm_handles  # 保存异步通信句柄
        self._comm_postprocess = comm_postprocess  # 保存后处理回调
        self._comm_waited = False  # 标记是否已等待通信完成

    # 等待所有异步通信完成并执行后处理回调。
    # 使用 _comm_waited 标志确保只等待一次，避免重复阻塞。
    def wait_for_comm(self) -> None:
        if self._comm_waited:  # 如果已经等待过，直接返回
            return
        if self._comm_handles:  # 如果有异步通信句柄
            for handle in self._comm_handles:  # 遍历所有句柄
                handle.wait()  # 阻塞等待每个通信操作完成
        if self._comm_postprocess:  # 如果有后处理回调
            for fn in self._comm_postprocess:  # 遍历所有回调
                fn()  # 执行后处理函数
        self._comm_waited = True  # 标记为已等待

    # 拦截属性访问：当访问 .tensors 属性时，自动触发 wait_for_comm()，
    # 确保张量数据在使用前已通过通信接收完毕。
    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):  # 如果访问的是 tensors 且尚未等待
            object.__getattribute__(self, "wait_for_comm")()  # 先等待通信完成
        return object.__getattribute__(self, name)  # 返回请求的属性值


# GPU Worker 类，是 vLLM v1 引擎中单个 GPU 设备上的工作进程。
# 职责包括：设备初始化、模型加载、内存分析与 KV 缓存分配、
# CUDA Graph 捕获与预热、模型前向执行与采样、LoRA 管理、
# 权重热更新、性能分析（profiling）以及休眠/唤醒的内存管理。
# 通过 GPUModelRunner 代理实际的模型推理逻辑。
class Worker(WorkerBase):

    # 构造函数：初始化 Worker 的基础配置和组件。
    # 设置浮点精度、弹性专家并行执行器、休眠缓冲区、权重传输引擎和性能分析器。
    #
    # 参数:
    #   vllm_config: vLLM 全局配置对象，包含模型、缓存、并行等所有配置
    #   local_rank: 当前进程在本机的 GPU 设备编号
    #   rank: 当前进程在全局分布式环境中的排名
    #   distributed_init_method: 分布式初始化方法（如 TCP URL）
    #   is_driver_worker: 是否为驱动 Worker（负责收集结果）
    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置
        local_rank: int,  # 本地 GPU 设备编号
        rank: int,  # 全局分布式排名
        distributed_init_method: str,  # 分布式初始化方法 URL
        is_driver_worker: bool = False,  # 是否为驱动 Worker
    ):
        super().__init__(  # 调用基类 WorkerBase 的构造函数
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION  # 获取 float32 矩阵乘法精度配置
        torch.set_float32_matmul_precision(precision)  # 设置 PyTorch 全局 float32 精度（highest/high/medium）

        from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor  # 弹性专家并行缩放执行器

        self.elastic_ep_executor = ElasticEPScalingExecutor(self)  # 创建弹性 EP 执行器实例

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}  # 休眠前保存的模型 buffer（level 2 休眠用）

        # Weight transfer engine (initialized on-demand)
        self.weight_transfer_engine = (  # 权重传输引擎（用于在线训练场景接收权重更新）
            WeightTransferEngineFactory.create_engine(  # 通过工厂创建传输引擎
                self.vllm_config.weight_transfer_config,  # 权重传输配置
                self.vllm_config.parallel_config,  # 并行配置
            )
            if self.vllm_config.weight_transfer_config is not None  # 仅在配置了权重传输时创建
            else None
        )

        # Torch/CUDA profiler. Enabled and configured through profiler_config.
        # Profiler wrapper is created lazily in profile() when start is called,
        # so we have all the information needed for proper trace naming.
        self.profiler: Any | None = None  # 性能分析器实例（延迟创建）
        self.profiler_config = vllm_config.profiler_config  # 性能分析器配置

        # Only validate profiler config is valid, don't instantiate yet
        if self.profiler_config.profiler not in ("torch", "cuda", None):  # 验证分析器类型是否有效
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER  # 是否使用 V2 模型运行器
        # pending non-blocking PP send work from the previous iteration
        self._pp_send_work: list[Handle] = []  # 上一轮迭代中未完成的流水线并行异步发送句柄

    # 休眠模式：释放 GPU 内存以供其他进程使用。
    # level=1 时仅卸载权重，level=2 时还保存模型 buffer 到 CPU。
    # 利用 CuMemAllocator 的内存池管理实现按标签的选择性内存释放。
    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator  # 自定义 CUDA 内存分配器

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]  # 记录休眠前的空闲 GPU 内存

        # Save the buffers before level 2 sleep
        if level == 2:  # level 2 深度休眠
            model = self.model_runner.model  # 获取模型实例
            self._sleep_saved_buffers = {  # 将所有模型 buffer 拷贝到 CPU 并保存
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()  # 获取 CuMem 分配器单例
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())  # 释放内存（level 1 仅释放权重标签）
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()  # 获取休眠后的内存信息
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep  # 计算释放的内存量
        used_bytes = total - free_bytes_after_sleep  # 计算仍在使用的内存量
        assert freed_bytes >= 0, "Memory usage increased after sleeping."  # 断言休眠后内存不应增加
        logger.info(  # 记录休眠释放的内存信息
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    # 唤醒模式：恢复休眠时释放的 GPU 内存，并在需要时重置 FP8 KV 缓存的缩放因子。
    def wake_up(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator  # 自定义 CUDA 内存分配器

        allocator = CuMemAllocator.get_instance()  # 获取 CuMem 分配器单例
        allocator.wake_up(tags)  # 恢复指定标签的内存分配

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):  # 如果存在 level 2 休眠保存的 buffer
            model = self.model_runner.model  # 获取模型实例
            for name, buffer in model.named_buffers():  # 遍历模型所有 buffer
                if name in self._sleep_saved_buffers:  # 如果该 buffer 在保存列表中
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)  # 从 CPU 备份恢复数据到 GPU
            self._sleep_saved_buffers = {}  # 清空保存的 buffer 引用

        # If the KV cache has just been woken up,
        # the internal state of cache_engine must be reset,
        # especially the FP8 scaling factor.
        if (  # 如果 KV 缓存刚被唤醒且使用 FP8 量化
            (tags is None or "kv_cache" in tags)  # 标签包含 kv_cache 或唤醒全部
            and self.cache_config.cache_dtype.startswith("fp8")  # 缓存数据类型为 FP8
            and hasattr(self.model_runner, "init_fp8_kv_scales")  # 模型运行器支持 FP8 缩放因子初始化
        ):
            self.model_runner.init_fp8_kv_scales()  # 重新初始化 FP8 KV 缓存缩放因子

    # 获取内存池上下文管理器：在休眠模式启用时，使用 CuMemAllocator 标记内存分配的用途。
    # 这允许 sleep/wake_up 按标签选择性释放/恢复内存。
    #
    # 参数:
    #   tag: 内存标签（如 "weights" 或 "kv_cache"）
    # 返回:
    #   上下文管理器，在其作用域内的内存分配会被标记
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if not self.vllm_config.model_config.enable_sleep_mode:  # 如果未启用休眠模式
            return nullcontext()  # 返回空上下文（不做标记）

        from vllm.device_allocator.cumem import CuMemAllocator  # 自定义 CUDA 内存分配器

        allocator = CuMemAllocator.get_instance()  # 获取 CuMem 分配器单例
        if tag == "weights":  # 如果是权重标签
            assert allocator.get_current_usage() == 0, (  # 断言当前无内存使用（休眠模式仅支持单实例）
                "Sleep mode can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)  # 返回带标签的内存池上下文管理器

    # 初始化 GPU 设备：设置 CUDA 设备、初始化分布式环境（NCCL）、
    # 随机种子、内存快照采集，以及构造 GPUModelRunner 实例。
    # 在数据并行场景下会根据 DP rank 调整 local_rank。
    @instrument(span_name="Init device")  # OpenTelemetry 追踪装饰器
    def init_device(self):
        if self.device_config.device_type == "cuda":  # 仅支持 CUDA 设备
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)  # 移除 Ray 设置的 NCCL 异步错误处理变量（会干扰 CUDA Graph）
            parallel_config = self.parallel_config  # 获取并行配置引用
            if (  # 非 Ray/外部启动器且为单节点 DP 模式时
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")  # 不是 Ray 或外部启动器
                and parallel_config.data_parallel_backend != "ray"  # 数据并行后端不是 Ray
                and parallel_config.nnodes_within_dp == 1  # 单节点模式
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local  # 获取本地 DP 排名
                if dp_local_rank is None:  # 如果本地 DP 排名不可用
                    dp_local_rank = self.parallel_config.data_parallel_index  # 使用全局 DP 索引

                tp_pp_world_size = (  # 计算 TP * PP 的总世界大小
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size  # 根据 DP rank 偏移 local_rank 以选择正确 GPU
                assert self.local_rank < torch.accelerator.device_count(), (  # 断言调整后的 local_rank 不超出 GPU 数量
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = (  # 获取可见 GPU 设备数量
                    torch.accelerator.device_count() if torch.cuda.is_available() else 0
                )
                assert self.parallel_config.local_world_size <= visible_device_count, (  # 断言本地世界大小不超过可见设备数
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )

            self.device = torch.device(f"cuda:{self.local_rank}")  # 创建 CUDA 设备对象
            torch.accelerator.set_device_index(self.device)  # 设置当前线程的默认 CUDA 设备

            current_platform.check_if_supports_dtype(self.model_config.dtype)  # 检查平台是否支持指定数据类型

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(  # 初始化分布式环境（NCCL、并行组等）
                self.vllm_config,  # vLLM 配置
                self.rank,  # 全局排名
                self.distributed_init_method,  # 分布式初始化方法 URL
                self.local_rank,  # 本地设备编号
                current_platform.dist_backend,  # 分布式后端（通常为 nccl）
            )

            if self.use_v2_model_runner:  # 如果使用 V2 模型运行器
                logger.info_once("Using V2 Model Runner", scope="local")  # 打印一次日志

            # Set random seed.
            set_random_seed(self.model_config.seed)  # 设置全局随机种子（确保可复现）

            # Now take memory snapshot after NCCL is initialized
            gc.collect()  # 强制垃圾回收
            torch.accelerator.empty_cache()  # 清空 CUDA 缓存

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)  # 采集初始内存快照
            self.requested_memory = request_memory(init_snapshot, self.cache_config)  # 计算请求的 GPU 内存量
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)  # 调试日志：初始内存快照
            logger.debug(  # 调试日志：请求的内存量
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:  # 不支持 CUDA 以外的设备
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1  # DBO（双缓冲优化）启用时使用 2 个微批次
        init_workspace_manager(self.device, num_ubatches)  # 初始化工作空间管理器

        # Construct the model runner
        if self.use_v2_model_runner:  # 使用 V2 模型运行器
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,  # 导入 V2 版本的 GPUModelRunner
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore  # 创建 V2 模型运行器实例
                self.vllm_config, self.device
            )
        else:  # 使用 V1 模型运行器
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,  # 导入 V1 版本的 GPUModelRunner
            )

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)  # 创建 V1 模型运行器实例

        if self.rank == 0:  # 仅在 rank 0 上报使用统计
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)  # 上报 vLLM 使用统计信息

    # 加载模型权重：支持正常加载和弹性 EP 扩缩容场景下的 dummy 权重加载。
    # 在休眠模式下使用内存池上下文标记权重分配，以便后续按标签释放。
    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        dummy_weights = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"  # 检查是否为弹性 EP 扩缩容启动
        if dummy_weights:  # 如果是扩缩容启动（先加载 dummy 权重，后续通过 EPLB 填充真实权重）
            (
                expanded_physical_to_logical,  # 物理专家到逻辑专家的映射表
                num_logical_experts,  # 逻辑专家数量
                old_num_physical_experts,  # 旧的物理专家数量
            ) = self.elastic_ep_executor.receive_expert_mapping()  # 从协调器接收专家映射
            num_physical_experts = expanded_physical_to_logical.shape[1]  # 新的物理专家数量
            self.parallel_config.eplb_config.num_redundant_experts = (  # 计算冗余专家数量
                num_physical_experts - num_logical_experts
            )

        with (  # 使用上下文管理器标记权重内存分配
            self._maybe_get_memory_pool_context(tag="weights"),  # 内存池标记为 "weights"
            set_current_vllm_config(self.vllm_config),  # 设置当前 vLLM 配置到线程上下文
        ):
            self.model_runner.load_model(load_dummy_weights=dummy_weights)  # 加载模型权重

        if dummy_weights:  # 如果是 dummy 权重模式
            self.model_runner.setup_eplb_from_mapping(  # 根据映射设置 EPLB
                expanded_physical_to_logical, old_num_physical_experts
            )
            self.model_runner.eep_eplb_suppressed = True  # 抑制 EPLB 自动平衡（等待真实权重填充）

    # 更新运行时配置覆盖项（如温度、top_p 等采样参数）。
    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)  # 委托给模型运行器处理

    # 重新加载模型权重（如从新的检查点热加载）。
    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)  # 委托给模型运行器处理

    # 分析可用 GPU 内存：通过 profile run 测量模型峰值内存使用量，
    # 估算 CUDA Graph 内存开销，计算可分配给 KV 缓存的剩余内存。
    # 支持通过 kv_cache_memory_bytes 手动指定 KV 缓存大小。
    @torch.inference_mode()  # 禁用梯度计算（推理模式）
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:  # 如果手动指定了 KV 缓存大小
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()  # 仍需执行 profile run 以编译模型

            msg = (  # 构造日志消息
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)  # 记录信息日志
            return kv_cache_memory_bytes  # 直接返回手动指定的 KV 缓存大小

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(  # 使用内存分析上下文管理器
            self.init_snapshot,  # 初始内存快照
            weights_memory=int(self.model_runner.model_memory_usage),  # 模型权重内存使用量
        ) as profile_result:  # 分析结果
            self.model_runner.profile_run()  # 执行一次 dummy 前向传播以测量内存

            profile_torch_peak = current_platform.memory_stats(self.device).get(  # 获取 PyTorch 分配器的峰值内存
                "allocated_bytes.all.peak", 0
            )

            # Profile CUDA graph memory if graphs will be captured.
            cudagraph_memory_estimate = 0  # CUDA Graph 内存估算值
            if not self.model_config.enforce_eager:  # 如果不是强制 eager 模式（即允许 CUDA Graph）
                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()  # 估算 CUDA Graph 内存开销

        # Use the pre-cudagraph torch peak to avoid double-counting.
        profile_result.torch_peak_increase = (  # 计算 PyTorch 峰值内存增量
            profile_torch_peak - profile_result.before_profile.torch_peak
        )
        profile_result.non_kv_cache_memory = (  # 非 KV 缓存内存总量 = 非 torch 增量 + torch 峰值增量 + 权重内存
            profile_result.non_torch_increase
            + profile_result.torch_peak_increase
            + profile_result.weights_memory
        )

        cudagraph_memory_estimate_applied = (  # 是否将 CUDA Graph 内存估算纳入计算
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS  # 根据环境变量决定
            else 0
        )

        self.non_torch_memory = profile_result.non_torch_increase  # 保存非 torch 内存增量（NCCL buffer 等）
        self.peak_activation_memory = (  # 保存峰值激活内存（含 CUDA Graph 估算）
            profile_result.torch_peak_increase + cudagraph_memory_estimate_applied
        )
        self.cudagraph_memory_estimate = cudagraph_memory_estimate  # 保存 CUDA Graph 内存估算值

        free_gpu_memory = profile_result.after_profile.free_memory  # 分析后的空闲 GPU 内存
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (  # 断言分析后空闲内存不应超过初始值
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (  # 可用于 KV 缓存的内存 = 请求内存 - 非 KV 缓存内存 - CUDA Graph 内存
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory  # 未请求的空闲内存（保留给系统/其他进程）
        logger.debug(  # 调试日志：初始空闲内存和请求内存
            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.requested_memory),
        )
        logger.debug(  # 调试日志：分析后的空闲内存
            "Free memory after profiling: %s GiB (total), %s GiB (within requested)",
            format_gib(free_gpu_memory),
            format_gib(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)  # 调试日志：完整分析结果
        logger.info_once(  # 信息日志：可用 KV 缓存内存（每个进程仅打印一次）
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )

        if cudagraph_memory_estimate > 0:  # 如果有 CUDA Graph 内存估算
            total_mem = self.init_snapshot.total_memory  # GPU 总内存
            current_util = self.cache_config.gpu_memory_utilization  # 当前 GPU 内存利用率设置
            cg_util_delta = cudagraph_memory_estimate / total_mem  # CUDA Graph 占用的利用率比例
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:  # 如果启用了 CUDA Graph 内存估算
                equiv_util = round(current_util - cg_util_delta, 4)  # 等效利用率（减去 CUDA Graph 占用）
                suggested_util = min(  # 建议的利用率（加上 CUDA Graph 占用以保持 KV 缓存大小）
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(  # 打印 CUDA Graph 内存分析信息和建议
                    "CUDA graph memory profiling is enabled "
                    "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1). "
                    "This will become the default in v0.19. "
                    "The current --gpu-memory-utilization=%.4f is equivalent "
                    "to --gpu-memory-utilization=%.4f without CUDA graph "
                    "memory profiling. To maintain the same effective KV "
                    "cache size as before, increase "
                    "--gpu-memory-utilization to %.4f.",
                    current_util,
                    equiv_util,
                    suggested_util,
                )
            else:  # 未启用 CUDA Graph 内存估算时
                suggested_util = min(  # 建议用户启用并调整利用率
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(  # 提示用户即将在 v0.19 默认启用
                    "In v0.19, CUDA graph memory profiling will be enabled "
                    "by default (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1), "
                    "which more accurately accounts for CUDA graph memory "
                    "during KV cache allocation. To try it now, set "
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 and increase "
                    "--gpu-memory-utilization from %.4f to %.4f to maintain "
                    "the same effective KV cache size.",
                    current_util,
                    suggested_util,
                )

        return int(self.available_kv_cache_memory_bytes)  # 返回可用 KV 缓存内存字节数

    # 获取 KV 连接器握手元数据：在解聚推理场景中，
    # 返回当前 Worker 的 KV 传输连接器元数据（按 TP rank 索引）。
    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():  # 如果没有 KV 传输组
            return None  # 返回 None

        connector = get_kv_transfer_group()  # 获取 KV 传输组连接器
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:  # 获取握手元数据
            return None  # 不需要握手的连接器返回 None

        tp_rank = get_tp_group().rank_in_group  # 获取当前进程在 TP 组中的排名
        return {tp_rank: metadata}  # 返回以 TP rank 为键的元数据字典

    # 获取 KV 缓存规格：返回模型各层所需的 KV 缓存配置信息。
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()  # 委托给模型运行器

    # 更新最大模型长度：在自动适配 GPU 内存后，
    # 将引擎确定的最大上下文长度同步到 Worker 和 ModelRunner。
    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.

        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len  # 更新模型配置中的最大长度
        if self.model_runner is not None:  # 如果模型运行器已创建
            self.model_runner.update_max_model_len(max_model_len)  # 同步到模型运行器
        logger.debug("Updated max_model_len to %d", max_model_len)  # 调试日志

    # 根据 KV 缓存配置分配 GPU 内存，初始化 KV 连接器和路由专家捕获器，
    # 并在需要时构建 KV 缓存零化元数据。
    @instrument(span_name="Allocate KV cache")  # OpenTelemetry 追踪
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks  # 更新缓存配置中的 GPU block 数量

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)  # 初始化 KV 传输连接器

        if self.vllm_config.model_config.enable_sleep_mode:  # 如果启用休眠模式
            from vllm.device_allocator.cumem import CuMemAllocator  # 自定义 CUDA 内存分配器

            allocator = CuMemAllocator.get_instance()  # 获取分配器单例
            with allocator.use_memory_pool(tag="kv_cache"):  # 在 kv_cache 标签的内存池中分配
                self.model_runner.initialize_kv_cache(kv_cache_config)  # 初始化 KV 缓存
        else:  # 非休眠模式直接初始化
            self.model_runner.initialize_kv_cache(kv_cache_config)  # 初始化 KV 缓存

        if self.model_config.enable_return_routed_experts:  # 如果启用了返回路由专家信息
            self.model_runner.init_routed_experts_capturer()  # 初始化路由专家捕获器

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(  # 如果需要 KV 缓存零化
            self.model_runner, "_init_kv_zero_meta"  # 且模型运行器支持零化元数据初始化
        ):
            self.model_runner._init_kv_zero_meta()  # 初始化 KV 缓存零化元数据

    # 编译和预热模型：执行 torch.compile 预热、内核调优、CUDA Graph 捕获，
    # 以及采样器/池化器的内存预分配，确保推理时无冷启动延迟。
    # 返回编译耗时（秒）。
    @instrument(span_name="Warmup (GPU)")  # OpenTelemetry 追踪
    def compile_or_warm_up_model(self) -> float:
        warmup_sizes: list[int] = []  # 需要预热的批次大小列表

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:  # 如果使用 vLLM 编译模式
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes  # 获取编译大小列表
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]  # 复制编译大小
            cg_capture_sizes: list[int] = []  # CUDA Graph 捕获大小列表

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:  # 如果启用了 CUDA Graph
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes  # 获取 CUDA Graph 捕获大小
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes  # 处理 None 情况
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]  # 排除已在 CUDA Graph 中的大小

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()  # 获取编译范围
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)  # 合并所有已有大小
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])  # 添加整数类型的预热大小
            for compile_range in compile_ranges:  # 遍历每个编译范围
                if not any(x in compile_range for x in all_sizes):  # 如果该范围内没有任何已有大小
                    warmup_sizes.append(compile_range.end)  # 添加范围末端以确保编译覆盖

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):  # 从大到小遍历预热大小
            logger.info("Compile and warming up model for size %d", size)  # 记录预热信息
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)  # 执行 dummy 前向传播进行预热
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)  # 预热后移除所有 LoRA 适配器

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)  # 预热和调优内核

        cuda_graph_memory_bytes = 0  # CUDA Graph 实际占用内存
        if not self.model_config.enforce_eager:  # 如果不是强制 eager 模式
            cuda_graph_memory_bytes = self.model_runner.capture_model()  # 捕获 CUDA Graph

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (  # 如果之前做过 CUDA Graph 内存估算
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)  # 字节转 GiB 的辅助函数
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)  # 计算实际与估算的差值
            logger.info(  # 记录 CUDA Graph 内存对比信息
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(  # 如果未手动指定 KV 缓存大小且做过内存分析
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)  # 150MiB 冗余缓冲区，防止 OOM
            non_kv_cache_memory = (  # 非 KV 缓存的内存总量
                self.model_runner.model_memory_usage  # 模型权重内存
                + self.peak_activation_memory  # 峰值激活内存
                + self.non_torch_memory  # 非 PyTorch 内存（如 NCCL buffer）
                + cuda_graph_memory_bytes  # CUDA Graph 内存
            )
            kv_cache_memory_bytes_to_gpu_limit = (  # 充分利用 GPU 内存时的 KV 缓存大小
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (  # 在请求内存限制内的 KV 缓存大小
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (  # 构造详细的内存使用报告
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)  # 调试日志：内存使用报告

        if self.use_v2_model_runner:  # V2 模型运行器
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)  # 预热 Triton 内核
        elif get_pp_group().is_last_rank:  # V1 且为流水线最后一级
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(  # 计算最大请求数
                self.scheduler_config.max_num_seqs,  # 最大序列数
                self.scheduler_config.max_num_batched_tokens,  # 最大批次 token 数
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(  # 执行 dummy 前向传播
                num_tokens=max_num_reqs,  # 使用最大请求数
                skip_eplb=True,  # 跳过 EPLB 指标记录
                cudagraph_runtime_mode=CUDAGraphMode.NONE,  # 不使用 CUDA Graph
            )
            if self.model_runner.is_pooling_model:  # 如果是池化模型（嵌入模型）
                self.model_runner._dummy_pooler_run(hidden_states)  # 预热池化器
            else:  # 生成模型
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)  # 预热采样器

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)  # 重置随机种子（消除预热对随机状态的影响）

        return self.compilation_config.compilation_time  # 返回编译耗时

    # 重置多模态缓存（清除缓存的图像/音频等多模态数据）。
    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()  # 委托给模型运行器

    # 重置编码器缓存（清除缓存的编码器输出）。
    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()  # 委托给模型运行器

    # 获取底层模型的 nn.Module 实例。
    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()  # 委托给模型运行器

    # 获取模型支持的任务类型列表。
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()  # 委托给模型运行器

    # 获取编码器计时统计信息（各编码器的延迟和调用次数）。
    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        return self.model_runner.get_encoder_timing_stats()  # 委托给模型运行器

    # 为性能分析添加追踪注解：在 trace 中标记当前迭代的 context/generation 请求数量和 token 数量，
    # 便于在性能分析工具中区分不同类型的请求。
    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:  # 如果分析器未启用
            return nullcontext()  # 返回空上下文

        self.profiler.step()  # 通知分析器前进一步

        iteration_details = compute_iteration_details(scheduler_output)  # 计算当前迭代的详细信息

        annotation = "".join(  # 构造注解字符串
            [
                "execute_context_",  # 上下文请求前缀
                str(iteration_details.num_ctx_requests),  # 上下文请求数量
                "(",
                str(iteration_details.num_ctx_tokens),  # 上下文 token 数量
                ")_generation_",  # 生成请求前缀
                str(iteration_details.num_generation_requests),  # 生成请求数量
                "(",
                str(iteration_details.num_generation_tokens),  # 生成 token 数量
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)  # 返回带注解的上下文管理器

    # 执行采样：在模型前向传播完成后，对 logits 进行采样生成 token。
    # 可选择性地应用语法约束（GrammarOutput）。
    @torch.inference_mode()  # 禁用梯度计算
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"  # 语法输出约束（结构化输出用）
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)  # 委托给模型运行器

    # 执行一步模型推理。处理流水线并行的中间张量接收/发送，
    # 通过 model_runner 执行前向传播。非末级 PP rank 返回 None 并异步发送中间结果。
    @torch.inference_mode()  # 禁用梯度计算
    def execute_model(
        self, scheduler_output: "SchedulerOutput"  # 调度器输出（包含本次迭代要处理的请求）
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ensure any previous non-blocking PP sends are complete
        if self._pp_send_work:  # 如果有上一轮未完成的异步发送
            for handle in self._pp_send_work:  # 遍历所有发送句柄
                handle.wait()  # 等待发送完成
            self._pp_send_work = []  # 清空发送句柄列表

        intermediate_tensors = None  # 中间张量（从上一级 PP 接收）
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0  # 是否有 token 需要处理
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 本次调度的总 token 数
        all_gather_tensors = {}  # 需要 AllGather 的张量字典（序列并行用）
        compilation_config = self.vllm_config.compilation_config  # 编译配置
        parallel_config = self.vllm_config.parallel_config  # 并行配置

        if (  # 如果启用了流水线并行 + 序列并行
            parallel_config.pipeline_parallel_size > 1  # PP 大小 > 1
            and compilation_config.pass_config.enable_sp  # 启用了序列并行
            and forward_pass  # 有 token 需要处理
        ):
            # currently only supported by V1 GPUModelRunner
            assert not self.use_v2_model_runner  # 序列并行 + PP 目前仅 V1 支持
            num_scheduled_tokens_np = np.array(  # 将每个请求的 token 数转为 numpy 数组
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # TODO(lucas): This is pretty gross; ideally we should only ever call
            # `_determine_batch_execution_and_padding` once (will get called again
            # in `execute_model`) but this requires a larger refactor of PP.
            _, batch_desc, _, _, _ = (  # 确定批次执行和填充策略
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,  # 总 token 数
                    num_reqs=len(num_scheduled_tokens_np),  # 请求数量
                    num_scheduled_tokens_np=num_scheduled_tokens_np,  # 每个请求的 token 数
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),  # 单个请求最大 token 数
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                )
            )
            all_gather_tensors = {  # 确定哪些张量需要 AllGather
                "residual": not is_residual_scattered_for_sp(  # 残差是否需要 AllGather（取决于 SP 模式）
                    self.vllm_config, batch_desc.num_tokens
                )
            }

        if forward_pass and not get_pp_group().is_first_rank:  # 如果有前向传播且不是 PP 第一级
            tensor_dict, comm_handles, comm_postprocess = (  # 异步接收上一级的中间张量
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),  # AllGather 使用 TP 组
                    all_gather_tensors=all_gather_tensors,  # 需要 AllGather 的张量
                )
            )
            assert tensor_dict is not None  # 断言接收到张量
            intermediate_tensors = AsyncIntermediateTensors(  # 封装为异步中间张量
                tensor_dict,  # 张量字典
                comm_handles=comm_handles,  # 通信句柄
                comm_postprocess=comm_postprocess,  # 后处理回调
            )

        with self.annotate_profile(scheduler_output):  # 添加性能分析注解
            output = self.model_runner.execute_model(  # 执行模型前向传播
                scheduler_output, intermediate_tensors  # 传入调度输出和中间张量
            )
            if (  # V2 池化模型的特殊处理
                self.use_v2_model_runner  # 使用 V2 运行器
                and self.model_runner.is_pooling_model  # 是池化模型
                and output is None  # execute_model 返回 None
            ):
                output = self.model_runner.pool()  # type: ignore  # 执行池化操作
            if isinstance(  # 如果输出是最终结果类型
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output  # 直接返回

        assert isinstance(output, IntermediateTensors)  # 断言输出为中间张量（非最后一级 PP）
        parallel_config = self.vllm_config.parallel_config  # 获取并行配置
        assert (  # 断言不是外部启动器且不是最后一级
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # launch non-blocking send of intermediate tensors
        self._pp_send_work = get_pp_group().isend_tensor_dict(  # 异步发送中间张量到下一级 PP
            output.tensors,  # 张量字典
            all_gather_group=get_tp_group(),  # AllGather 使用 TP 组
            all_gather_tensors=all_gather_tensors,  # 需要 AllGather 的张量
        )

        return None  # 非最后一级返回 None

    # 获取投机解码的草稿 token ID。
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()  # 委托给模型运行器

    # 启动或停止性能分析器。
    # is_start=True 时启动分析，is_start=False 时停止并导出 trace。
    # 支持 torch 和 cuda 两种分析器类型。
    #
    # 参数:
    #   is_start: True 启动分析，False 停止分析
    #   profile_prefix: trace 文件名前缀（用于区分不同分析会话）
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:  # 如果未配置分析器
            raise RuntimeError(  # 抛出错误提示用户配置
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:  # 启动分析
            # Generate the trace name by combining prefix with comprehensive rank suffix
            from vllm.distributed.utils import get_worker_rank_suffix  # 获取 Worker rank 后缀

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)  # 生成包含 rank 信息的后缀

            # Build the full trace name
            if profile_prefix:  # 如果提供了前缀
                trace_name = f"{profile_prefix}_{rank_suffix}"  # 前缀 + rank 后缀
            else:
                trace_name = rank_suffix  # 仅使用 rank 后缀

            # Create the profiler wrapper only on the first start call
            if self.profiler is None:  # 如果分析器尚未创建（首次启动）
                profiler_type = self.profiler_config.profiler  # 获取分析器类型
                if profiler_type == "torch":  # PyTorch 分析器
                    self.profiler = TorchProfilerWrapper(  # 创建 Torch 分析器封装
                        self.profiler_config,  # 分析器配置
                        worker_name=trace_name,  # trace 名称
                        local_rank=self.local_rank,  # 本地 rank
                        activities=["CPU", "CUDA"],  # 分析 CPU 和 CUDA 活动
                    )
                    logger.debug(  # 调试日志
                        "Starting torch profiler with trace name: %s", trace_name
                    )
                elif profiler_type == "cuda":  # CUDA 分析器
                    self.profiler = CudaProfilerWrapper(self.profiler_config)  # 创建 CUDA 分析器封装
                    logger.debug("Starting CUDA profiler")  # 调试日志
                else:  # 无效的分析器类型
                    # Config validation should prevent this code being reached
                    raise ValueError(  # 抛出值错误
                        f"Invalid profiler value of {self.profiler_config.profiler}"
                    )

            # If profiler already initialized, restart profiling but keep
            # the original trace name from the first initialization.
            self.profiler.start()  # 启动分析
        else:  # 停止分析
            if self.profiler is None:  # 如果分析器未启动
                logger.warning("Profiler was not started, nothing to stop.")  # 警告日志
                return
            self.profiler.stop()  # 停止分析并导出 trace

    # 执行一个 dummy 批次（用于预热或测试）。
    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)  # 执行单 token 的 dummy 前向传播

    # 添加 LoRA 适配器到模型。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)  # 委托给模型运行器

    # 移除指定 ID 的 LoRA 适配器。
    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)  # 委托给模型运行器

    # 列出当前已加载的所有 LoRA 适配器 ID。
    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()  # 委托给模型运行器

    # 固定 LoRA 适配器到 GPU 内存（防止被驱逐）。
    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)  # 委托给模型运行器

    # 健康检查：Worker 只要在运行就是健康的。
    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return  # 直接返回，无需检查

    # 保存模型的分片状态到磁盘（用于模型检查点持久化）。
    #
    # 参数:
    #   path: 保存路径
    #   pattern: 分片文件名模式
    #   max_size: 每个分片的最大字节数
    def save_sharded_state(
        self,
        path: str,  # 保存目标路径
        pattern: str | None = None,  # 分片文件名模式（可选）
        max_size: int | None = None,  # 每个分片最大大小（可选）
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader  # 分片状态加载器

        ShardedStateLoader.save_model(  # 保存模型分片状态
            self.model_runner.model,  # 模型实例
            path,  # 保存路径
            pattern=pattern,  # 文件名模式
            max_size=max_size,  # 最大分片大小
        )

    # 将模型保存为 Tensorizer 格式（高效的序列化格式）。
    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        TensorizerLoader.save_model(  # 保存为 Tensorizer 格式
            self.get_model(),  # 获取模型实例
            tensorizer_config=tensorizer_config,  # Tensorizer 配置
            model_config=self.model_config,  # 模型配置
        )

    # 初始化权重传输引擎，用于在线训练场景下从 trainer 接收模型权重更新。
    #
    # 参数:
    #   init_info: 后端特定的初始化信息字典（如 NCCL 进程组信息）
    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """
        Initialize weight transfer mechanism.
        For NCCL backend, this creates a process group with the trainer.

        Args:
            init_info: Dictionary containing backend-specific initialization info
        """
        if self.weight_transfer_engine is None:  # 如果未配置权重传输引擎
            raise RuntimeError(  # 抛出运行时错误
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )
        # Parse dict into backend-specific typed dataclass
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)  # 解析为类型化数据类
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)  # 初始化传输引擎

    # 从 trainer 接收并应用批量权重更新。
    # 支持两种模式：检查点格式（通过 load_weights 逐层加载）和直接格式（直接拷贝张量）。
    #
    # 参数:
    #   update_info: 后端特定的更新信息字典
    def update_weights(self, update_info: dict) -> None:
        """
        Batched weight update from the trainer.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        if self.weight_transfer_engine is None:  # 如果未配置权重传输引擎
            raise RuntimeError(  # 抛出运行时错误
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        # Parse dict into backend-specific typed dataclass
        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)  # 解析为类型化数据类

        model = self.model_runner.model  # 获取模型实例

        if typed_update_info.is_checkpoint_format:  # 如果权重为检查点格式
            from vllm.model_executor.model_loader.reload import (  # 逐层重载工具
                finalize_layerwise_reload,  # 完成逐层重载
                initialize_layerwise_reload,  # 初始化逐层重载
            )

            # Use layerwise reload pattern for checkpoint format weights
            with torch.device(self.device):  # 在当前设备上下文中
                initialize_layerwise_reload(model)  # 初始化逐层重载（准备接收权重）
                self.weight_transfer_engine.receive_weights(  # 接收权重
                    typed_update_info,  # 更新信息
                    load_weights=model.load_weights,  # 使用模型的 load_weights 方法加载
                )
                finalize_layerwise_reload(model, self.model_config)  # 完成逐层重载（更新内部状态）
        else:  # 权重已为内核格式，直接拷贝
            # Weights are already in kernel format, copy directly
            def load_weights_direct(  # 直接拷贝权重的回调函数
                weights: list[tuple[str, torch.Tensor]],  # 权重名-张量对列表
            ) -> None:
                for name, weight in weights:  # 遍历权重
                    param = model.get_parameter(name)  # 获取模型中的参数
                    param.copy_(weight)  # 就地拷贝权重数据

            self.weight_transfer_engine.receive_weights(  # 接收权重
                typed_update_info,  # 更新信息
                load_weights=load_weights_direct,  # 使用直接拷贝方式
            )

        # NCCL broadcast/packed path are asynchronous.
        # Sync here so the next step uses the new weights.
        torch.accelerator.synchronize()  # 同步 GPU，确保权重更新完成

    # 关闭 Worker：清理 KV 传输组、停止分析器、关闭权重传输引擎。
    def shutdown(self) -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:  # 如果 KV 传输关闭函数可用
            ensure_kv_transfer_shutdown()  # 确保 KV 传输组关闭
        if self.profiler is not None:  # 如果分析器已创建
            self.profiler.shutdown()  # 关闭分析器

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):  # 如果有权重传输引擎
            weight_transfer_engine.shutdown()  # 关闭权重传输引擎

    # 弹性专家并行执行：通过 ElasticEPScalingExecutor 执行指定方法，
    # 支持专家并行的动态扩缩容。
    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)  # 委托给弹性 EP 执行器


# 初始化 Worker 的分布式环境：设置 NCCL 后端、张量/流水线并行组、
# 批次不变性优化以及弹性专家并行的环境变量。
#
# 参数:
#   vllm_config: vLLM 全局配置
#   rank: 全局分布式排名
#   distributed_init_method: 分布式初始化方法（默认 "env://"）
#   local_rank: 本地设备编号
#   backend: 分布式后端（默认 "nccl"）
def init_worker_distributed_environment(
    vllm_config: VllmConfig,  # vLLM 全局配置
    rank: int,  # 全局分布式排名
    distributed_init_method: str | None = None,  # 分布式初始化方法
    local_rank: int = -1,  # 本地设备编号
    backend: str = "nccl",  # 分布式后端
) -> None:
    """Initialize the distributed environment."""
    attention_config = vllm_config.attention_config  # 获取注意力配置
    parallel_config = vllm_config.parallel_config  # 获取并行配置
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance  # 批次不变性初始化

    init_batch_invariance(attention_config.backend)  # 初始化批次不变性优化（针对特定注意力后端）
    override_envs_for_eplb(parallel_config)  # 覆盖弹性专家并行负载均衡的环境变量
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)  # 设置自定义 AllReduce（除非被禁用）

    init_method = distributed_init_method or "env://"  # 使用提供的初始化方法或默认 "env://"

    timeout = None  # 分布式超时设置
    if parallel_config.distributed_timeout_seconds is not None:  # 如果配置了超时
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)  # 转换为 timedelta

    init_distributed_environment(  # 初始化分布式环境（创建进程组）
        parallel_config.world_size,  # 全局进程数
        rank,  # 当前进程排名
        init_method,  # 初始化方法 URL
        local_rank,  # 本地设备编号
        backend,  # 通信后端
        timeout,  # 超时设置
    )

    ensure_model_parallel_initialized(  # 确保模型并行组已创建
        parallel_config.tensor_parallel_size,  # 张量并行大小
        parallel_config.pipeline_parallel_size,  # 流水线并行大小
        parallel_config.prefill_context_parallel_size,  # 预填充上下文并行大小
        parallel_config.decode_context_parallel_size,  # 解码上下文并行大小
    )

    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)  # 初始化编码器-上下文传输连接器（EPD 解聚模式）
