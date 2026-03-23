# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# 注册 Prithvi 地理空间多模态数据处理器插件入口点
def register_prithvi():
    return "prithvi_io_processor.prithvi_processor.PrithviMultimodalDataProcessor"  # noqa: E501
