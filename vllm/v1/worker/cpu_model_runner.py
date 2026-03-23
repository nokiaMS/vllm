# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


# CPUModelRunner：继承 GPUModelRunner 并适配 CPU 后端
# 设计思路：复用 GPU 模型运行器的大部分逻辑，但在初始化时：
#   1. 用 _torch_cuda_wrapper 模拟 CUDA Event/Stream 以绕过父类对 GPU 的依赖
#   2. 禁用 CUDA Graph 和级联注意力等 GPU 专属优化
#   3. 通过 _postprocess_tensors 将所有 GPU 张量替换为 CPU 张量
class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    # 后处理：将父类初始化时创建的 GPU 张量全部替换为对应的 CPU 张量
    # 包括 CpuGpuBuffer 中的 gpu 指针指向 cpu，以及 input_batch 中的设备张量
    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if isinstance(cpu_tensor, torch.Tensor) and isinstance(
                device_tensor, torch.Tensor
            ):
                setattr(obj, device_attr_name, cpu_tensor)

        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    # 在 CPU 上加载模型权重，不支持虚拟权重（弹性 EP 扩展场景）
    @instrument(span_name="Loading (CPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise ValueError(
                "Loading dummy weights (needed for elastic EP scale-up) "
                "Is not supported by the CPU Model Runner."
            )
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    # 模型预热：使用虚拟输入进行一次前向推理，触发 torch.compile 编译优化
    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        logger.info("Warming up done.")

    # CPU 无需初始化设备属性（GPU 版本中会读取显存等信息）
    def _init_device_properties(self) -> None:
        pass

    # CPU 无需设备同步（GPU 版本中调用 torch.cuda.synchronize）
    def _sync_device(self) -> None:
        pass

    # CPU 后端目前不需要数据并行填充
    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        # Note: For CPU backend, dp padding is not required for now.
        return 0, None


# _torch_cuda_wrapper：临时替换 torch.Event 和 torch.cuda.Stream 为空操作占位符
# 目的：父类 GPUModelRunner.__init__ 中会创建 CUDA 事件和流，
# 在 CPU 模式下通过猴子补丁避免 CUDA 初始化错误，退出后恢复原始实现
@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


# _set_global_compilation_settings：临时设置 torch inductor 编译参数
# 当启用 max_autotune 时，需要开启参数冻结（freezing）以支持 MKLDNN/CPPGEMM 后端优化
# 使用上下文管理器确保编译完成后恢复原始设置
@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
