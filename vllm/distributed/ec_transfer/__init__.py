# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.ec_transfer.ec_transfer_state import (  # 从EC传输状态模块导入核心函数
    ensure_ec_transfer_initialized,  # 确保EC传输已初始化的函数
    get_ec_transfer,  # 获取EC传输连接器实例的函数
    has_ec_transfer,  # 检查EC传输是否已配置的函数
)

__all__ = [  # 定义模块的公共接口列表
    "get_ec_transfer",  # 导出获取EC传输连接器的函数
    "ensure_ec_transfer_initialized",  # 导出确保EC传输初始化的函数
    "has_ec_transfer",  # 导出检查EC传输是否存在的函数
]
