# SPDX-License-Identifier: Apache-2.0  # Apache-2.0许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

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
from collections.abc import Iterable  # 导入可迭代类型抽象基类

import torch  # 导入PyTorch张量计算库
import torch.nn as nn  # 导入PyTorch神经网络模块

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.model_executor.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel  # 导入Llama因果语言模型和基础模型

from .llama import LlamaDecoderLayer  # 导入Llama解码层
from .utils import (  # 从工具模块导入辅助类和函数
    AutoWeightsLoader,  # 自动权重加载器
    PPMissingLayer,  # 流水线并行缺失层占位符
    WeightsMapper,  # 权重名称映射器
    is_pp_missing_parameter,  # 检查参数是否因流水线并行而缺失
)


class TeleChat2Model(LlamaModel):
    """TeleChat2模型，继承自LlamaModel，支持交错KV头的自定义权重加载。"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化TeleChat2模型，配置属性映射和偏置设置。"""
        hf_config = vllm_config.model_config.hf_config  # 获取HuggingFace模型配置

        vllm_config.model_config.hf_config.attribute_map = {  # 设置配置属性名映射
            "num_hidden_layers": "n_layer",  # 隐藏层数映射
            "num_attention_heads": "n_head",  # 注意力头数映射
            "intermediate_size": "ffn_hidden_size",  # FFN中间层大小映射
            "rms_norm_eps": "layer_norm_epsilon",  # RMS归一化epsilon映射
        }
        vllm_config.model_config.hf_config.hidden_act = "silu"  # 设置激活函数为SiLU

        # 1. Initialize the LlamaModel with bias
        hf_config.bias = True  # 启用注意力层偏置
        hf_config.mlp_bias = True  # 启用MLP层偏置

        super().__init__(vllm_config=vllm_config, prefix=prefix)  # 调用父类LlamaModel初始化
        # 2. Remove the bias from the qkv_proj and gate_up_proj based on config
        # Telechat2's gate_up_proj and qkv_proj don't have bias
        # see: https://github.com/vllm-project/vllm/pull/10311#issuecomment-2490297566
        for layer in self.layers:  # 遍历所有解码层
            if not isinstance(layer, PPMissingLayer):  # 跳过流水线并行缺失层
                layer.self_attn.qkv_proj.bias = None  # 移除QKV投影的偏置
                layer.self_attn.qkv_proj.skip_bias_add = True  # 跳过偏置加法
                layer.mlp.gate_up_proj.bias = None  # 移除门控上投影的偏置
                layer.mlp.gate_up_proj.skip_bias_add = True  # 跳过偏置加法

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载模型权重，处理TeleChat2特有的交错KV头权重格式。"""
        stacked_params_mapping = [  # 堆叠参数映射列表
            ("gate_up_proj", "gate_proj", 0),  # 门控投影映射
            ("gate_up_proj", "up_proj", 1),  # 上投影映射
        ]
        params_dict = dict(self.named_parameters())  # 获取模型所有参数字典
        loaded_params: set[str] = set()  # 已加载参数名集合
        total_num_heads = self.config.n_head  # 总注意力头数
        head_dim = self.config.hidden_size // total_num_heads  # 每个头的维度
        for name, loaded_weight in weights:  # 遍历待加载的权重
            if "self_attn.key_value" in name:  # 处理交错的KV权重
                k_weight = []  # K权重列表
                v_weight = []  # V权重列表
                for i in range(total_num_heads):  # 按头遍历，拆分交错的KV权重
                    start = i * head_dim * 2  # 计算当前头的起始位置
                    k_weight.append(loaded_weight[start : start + head_dim, :])  # 提取K权重
                    v_weight.append(  # 提取V权重
                        loaded_weight[start + head_dim : start + 2 * head_dim :]
                    )
                k_weight = torch.cat(k_weight, dim=0)  # 拼接所有头的K权重
                v_weight = torch.cat(v_weight, dim=0)  # 拼接所有头的V权重
                name = name.replace("key_value", "qkv_proj")  # 重命名为qkv_proj
                if is_pp_missing_parameter(name, self):  # 检查流水线并行缺失
                    continue
                param = params_dict[name]  # 获取目标参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, k_weight, "k")  # 加载K权重分片
                weight_loader(param, v_weight, "v")  # 加载V权重分片
            elif "query" in name:  # 处理查询权重
                name = name.replace("query", "qkv_proj")  # 重命名为qkv_proj
                if is_pp_missing_parameter(name, self):  # 检查流水线并行缺失
                    continue
                param = params_dict[name]  # 获取目标参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, "q")  # 加载Q权重分片
            else:  # 处理其他权重
                for param_name, weight_name, shard_id in stacked_params_mapping:  # 检查堆叠参数映射
                    if weight_name not in name:  # 跳过不匹配的映射
                        continue
                    name = name.replace(weight_name, param_name)  # 替换权重名称
                    if is_pp_missing_parameter(name, self):  # 检查流水线并行缺失
                        continue
                    param = params_dict[name]  # 获取目标参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(param, loaded_weight, shard_id)  # 按分片加载权重
                    break
                else:  # 普通权重处理
                    if is_pp_missing_parameter(name, self):  # 检查流水线并行缺失
                        continue
                    param = params_dict[name]  # 获取目标参数
                    weight_loader = getattr(  # 获取权重加载器，默认使用标准加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
            loaded_params.add(name)  # 记录已加载的参数名
        return loaded_params  # 返回已加载参数集合


class TeleChat2ForCausalLM(LlamaForCausalLM):
    """TeleChat2因果语言模型，继承自LlamaForCausalLM，提供权重名称映射。"""

    hf_to_vllm_mapper = WeightsMapper(  # HuggingFace到vLLM的权重名称映射器
        orig_to_new_prefix={  # 前缀映射
            "transformer.": "model.",  # transformer前缀映射为model
        },
        orig_to_new_substr={  # 子串映射
            ".h.": ".layers.",  # 层名称映射
            ".self_attention.": ".self_attn.",  # 自注意力名称映射
            ".word_embeddings.": ".embed_tokens.",  # 词嵌入名称映射
            ".dense.": ".o_proj.",  # 密集层映射为输出投影
            ".ln_f.": ".norm.",  # 最终层归一化名称映射
        },
    )

    def _init_model(
        self,
        vllm_config: VllmConfig,  # vLLM配置对象
        prefix: str = "",  # 参数名前缀
        layer_type: type[nn.Module] = LlamaDecoderLayer,  # 解码层类型
    ):
        """初始化TeleChat2内部模型实例。"""
        return TeleChat2Model(vllm_config=vllm_config, prefix=prefix)  # 创建TeleChat2模型

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """使用自动权重加载器和名称映射器加载模型权重。"""
        loader = AutoWeightsLoader(  # 创建自动权重加载器
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),  # 如果共享词嵌入则跳过lm_head
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)  # 使用映射器加载权重
