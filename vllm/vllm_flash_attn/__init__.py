# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

from vllm.vllm_flash_attn.flash_attn_interface import (  # 从flash_attn_interface模块导入核心组件
    FA2_AVAILABLE,  # Flash Attention 2 是否可用的标志
    FA3_AVAILABLE,  # Flash Attention 3 是否可用的标志
    fa_version_unsupported_reason,  # 获取FA版本不支持的原因
    flash_attn_varlen_func,  # 变长序列的Flash Attention函数
    get_scheduler_metadata,  # 获取调度器元数据（仅FA3使用）
    is_fa_version_supported,  # 检查指定FA版本是否受支持
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):  # 如果FA2和FA3都不可用
    raise ImportError(  # 抛出导入错误
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "  # 提示需要CUDA flash attention扩展
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."  # 提示ROCm平台应使用上游flash_attn
    )

__all__ = [  # 定义模块的公开接口列表
    "fa_version_unsupported_reason",  # 导出：获取不支持原因的函数
    "flash_attn_varlen_func",  # 导出：变长Flash Attention函数
    "get_scheduler_metadata",  # 导出：获取调度器元数据函数
    "is_fa_version_supported",  # 导出：检查版本支持函数
]
