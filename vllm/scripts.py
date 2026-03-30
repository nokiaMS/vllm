# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.cli.main import main as vllm_main  # 从CLI入口模块导入主函数
from vllm.logger import init_logger  # 导入日志初始化工具

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# Backwards compatibility for the move from vllm.scripts to
# vllm.entrypoints.cli.main
def main():
    """已弃用的主入口函数，保留用于向后兼容。请使用 vllm.entrypoints.cli.main.main() 代替。"""
    logger.warning(  # 输出弃用警告日志
        "vllm.scripts.main() is deprecated. Please re-install "  # 提示用户该函数已弃用
        "vllm or use vllm.entrypoints.cli.main.main() instead."  # 建议用户使用新的入口函数
    )
    vllm_main()  # 调用新的主函数执行实际逻辑
