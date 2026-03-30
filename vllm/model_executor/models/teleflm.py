# SPDX-License-Identifier: Apache-2.0  # Apache-2.0许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch  # 导入PyTorch张量计算库
import torch.nn as nn  # 导入PyTorch神经网络模块

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from vllm.model_executor.models.llama import (  # 从Llama模型导入基础组件
    LlamaDecoderLayer,  # 导入Llama解码层
    LlamaForCausalLM,  # 导入Llama因果语言模型
    LlamaModel,  # 导入Llama基础模型
)


class TeleFLMModel(LlamaModel):
    """TeleFLM模型，继承自LlamaModel，支持µScaling（mup）缩放技术。"""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,  # vLLM配置对象
        prefix: str = "",  # 参数名前缀
        layer_type: type[nn.Module] = LlamaDecoderLayer,  # 解码层类型，默认为Llama解码层
    ):
        """初始化TeleFLM模型，设置µScaling相关参数。"""
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)  # 调用父类Llama模型初始化
        """
        This implementation is based on the µScaling paper presented at
        the ICLR 2025 Workshop:
        NanoLM: An Affordable LLM Study Benchmark \
        via Accurate Loss Prediction across Scales
        by Yiqun Yao et al.
        Available at: https://openreview.net/forum?id=IwaPYg1SCA
        arXiv preprint: https://arxiv.org/abs/2304.06875
        """
        self.use_mup = self.config.use_mup  # 是否使用µP（µ参数化）
        if self.use_mup:  # 如果启用µP
            self.input_mult = self.config.input_mult  # 输入缩放因子

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入token ID转换为嵌入向量，如果启用µP则进行缩放。"""
        embedding = self.embed_tokens(input_ids)  # 通过嵌入层获取token嵌入
        if self.use_mup:  # 如果启用µP缩放
            embedding = embedding * self.input_mult  # 乘以输入缩放因子
        return embedding  # 返回嵌入向量


class TeleFLMForCausalLM(LlamaForCausalLM):
    """TeleFLM因果语言模型，继承自LlamaForCausalLM，支持µScaling输出缩放。"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化TeleFLM因果语言模型，设置µP输出缩放参数。"""
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # 调用父类Llama因果语言模型初始化
        # mup
        self.use_mup = self.config.use_mup  # 是否使用µP参数化
        if self.use_mup:  # 如果启用µP
            self.mup_scale_factor = self.config.mup_scale_factor  # µP缩放因子
            self.output_mult = self.config.output_mult / self.mup_scale_factor  # 计算输出缩放系数
            logit_scale = self.output_mult  # logit缩放值
            self.logits_processor = LogitsProcessor(  # 创建logits处理器
                self.config.vocab_size, scale=logit_scale  # 传入词表大小和缩放值
            )
