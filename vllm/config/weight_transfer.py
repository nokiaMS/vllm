# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
from typing import Literal  # 导入Literal类型，用于限定字面量类型


from vllm.config.utils import config  # 导入config装饰器，用于创建pydantic数据类


@config  # 使用config装饰器创建pydantic数据类
class WeightTransferConfig:
    """RL训练期间权重传输的配置类。"""

    backend: Literal["nccl", "ipc"] = "nccl"  # 权重传输使用的后端，支持"nccl"和"ipc"，默认"nccl"
    """The backend to use for weight transfer."""  # 权重传输后端的文档字符串
