# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

from .data import ProcessorInputs, SingletonInputs  # 从data模块导入处理器输入和单例输入类型


def split_enc_dec_inputs(  # 拆分编码器-解码器输入的函数
    inputs: ProcessorInputs,  # 处理器输入参数
) -> tuple[SingletonInputs | None, SingletonInputs]:  # 返回编码器输入和解码器输入的元组
    """拆分编码器-解码器模型的输入为编码器部分和解码器部分。"""
    if inputs["type"] == "enc_dec":  # 如果输入类型是编码器-解码器
        return inputs["encoder_prompt"], inputs["decoder_prompt"]  # 返回编码器提示和解码器提示

    return None, inputs  # 非编码器-解码器输入则编码器部分为None，解码器部分为原始输入
