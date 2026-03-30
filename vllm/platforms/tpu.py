# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from vllm.logger import init_logger  # 导入vLLM日志初始化函数

logger = init_logger(__name__)  # 获取当前模块的日志记录器


try:
    from tpu_inference.platforms import (  # 尝试从tpu_inference包导入TPU平台类
        TpuPlatform as TpuInferencePlatform,
    )

    TpuPlatform = TpuInferencePlatform  # type: ignore  # 将导入的TPU平台类赋值给TpuPlatform
    USE_TPU_INFERENCE = True  # 标记tpu_inference库可用
except ImportError:  # 如果导入失败（未安装tpu_inference）
    logger.error(  # 记录错误日志
        "tpu_inference not found, please install tpu_inference to run vllm on TPU"
    )
    pass  # 静默处理导入错误
