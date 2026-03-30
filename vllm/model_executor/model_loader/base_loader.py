# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器

import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch神经网络模块

import vllm.envs as envs  # 导入vLLM环境变量配置模块
from vllm.config import ModelConfig, VllmConfig  # 导入模型配置和vLLM全局配置类
from vllm.config.load import LoadConfig  # 导入模型加载配置类
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.model_executor.model_loader.utils import (  # 从模型加载工具模块导入
    initialize_model,  # 导入模型初始化函数（根据配置创建模型实例）
    process_weights_after_loading,  # 导入权重加载后处理函数（量化、设备迁移等）
)
from vllm.platforms import current_platform  # 导入当前硬件平台检测模块
from vllm.tracing import instrument  # 导入性能追踪装饰器
from vllm.utils.mem_utils import format_gib  # 导入内存格式化工具（字节转GiB）
from vllm.utils.torch_utils import set_default_torch_dtype  # 导入设置默认张量数据类型的上下文管理器

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# 模型加载器的抽象基类，定义了模型下载、权重加载和完整模型加载的接口。
# 所有具体的模型加载器（如DummyModelLoader、DefaultModelLoader等）都继承此类。
class BaseModelLoader(ABC):
    """Base class for model loaders."""

    # 初始化模型加载器，保存加载配置。
    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config  # 保存模型加载配置（加载格式、设备、量化等）

    @abstractmethod  # 标记为抽象方法，子类必须实现
    # 下载模型文件，使其可以被立即加载。
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError  # 抛出未实现异常（子类必须覆写）

    @abstractmethod  # 标记为抽象方法，子类必须实现
    # 将权重加载到已初始化的模型中（支持原地加载）。
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError  # 抛出未实现异常（子类必须覆写）

    @instrument(span_name="Load model")  # 使用性能追踪装饰器，记录"Load model"追踪段
    # 完整的模型加载流程：初始化模型结构 -> 加载权重 -> 后处理 -> 返回eval模式的模型。
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:  # 返回加载完成的nn.Module模型
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config  # 获取设备配置
        load_config = vllm_config.load_config  # 获取加载配置
        load_device = (  # 确定模型加载的目标设备
            device_config.device if load_config.device is None else load_config.device  # 优先使用加载配置中的设备，否则使用设备配置
        )
        target_device = torch.device(load_device)  # 创建目标设备对象
        with set_default_torch_dtype(model_config.dtype):  # 设置默认张量数据类型为模型配置指定的类型
            with target_device:  # 在目标设备上下文中创建模型（张量直接分配在目标设备上）
                model = initialize_model(  # 根据配置初始化模型结构（创建模型实例但不加载权重）
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )

            log_model_inspection(model)  # 如果启用了模型检查日志，打印模型结构

            logger.debug("Loading weights on %s ...", load_device)  # 记录调试日志：正在加载权重
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)  # 调用子类实现的权重加载方法

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda():  # 如果当前平台是CUDA（GPU）
                peak_memory = torch.cuda.max_memory_allocated()  # 获取GPU峰值内存使用量
                logger.debug_once(  # 记录一次性调试日志
                    "Peak GPU memory after loading weights: %s GiB",  # 日志格式：权重加载后GPU峰值内存
                    format_gib(peak_memory),  # 将字节数格式化为GiB
                    scope="local",  # 本地作用域（仅当前进程）
                )

            process_weights_after_loading(model, model_config, target_device)  # 权重后处理：量化压缩、设备迁移、数据类型转换等

        return model.eval()  # 将模型设置为评估模式（禁用dropout等训练行为）并返回


# 打印模型结构检查日志。仅在环境变量 VLLM_LOG_MODEL_INSPECTION=1 时生效，
# 用于调试时查看模型的完整层级结构。
def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:  # 如果未启用模型检查环境变量
        return  # 直接返回，不打印

    from vllm.model_inspection import format_model_inspection  # 延迟导入模型检查格式化函数（避免循环依赖）

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))  # 打印格式化后的模型结构信息
