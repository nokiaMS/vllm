# SPDX-License-Identifier: Apache-2.0  # Apache-2.0许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明

# Copyright 2025 The vLLM team.
# Copyright 2025 IBM.
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
"""Wrapper around `Terratorch` models"""

from collections import OrderedDict  # 导入有序字典
from collections.abc import Iterable, Mapping, Sequence  # 导入可迭代、映射、序列抽象基类
from functools import cached_property  # 导入缓存属性装饰器
from typing import Any  # 导入Any类型注解

import torch  # 导入PyTorch张量计算库
import torch.nn as nn  # 导入PyTorch神经网络模块
from terratorch.vllm import (  # 从terratorch的vLLM集成模块导入
    DummyDataGenerator,  # 虚拟数据生成器
    InferenceRunner,  # 推理运行器
    InputDefinition,  # 输入定义
    InputTypeEnum,  # 输入类型枚举
)
from transformers import BatchFeature  # 导入批量特征类

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.config.multimodal import BaseDummyOptions  # 导入多模态虚拟选项基类
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.model_executor.layers.pooler import IdentityPooler  # 导入恒等池化层
from vllm.model_executor.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from vllm.model_executor.models.utils import AutoWeightsLoader  # 导入自动权重加载器
from vllm.multimodal import MULTIMODAL_REGISTRY  # 导入多模态注册表
from vllm.multimodal.inputs import (  # 导入多模态输入相关类
    ImageItem,  # 图像项
    ModalityData,  # 模态数据
    MultiModalDataDict,  # 多模态数据字典
    MultiModalFieldConfig,  # 多模态字段配置
    MultiModalInputs,  # 多模态输入
    MultiModalKwargsItems,  # 多模态关键字参数项
    PlaceholderRange,  # 占位符范围
    mm_inputs,  # 多模态输入构造函数
)
from vllm.multimodal.parse import (  # 导入多模态解析相关类
    DictEmbeddingItems,  # 字典嵌入项
    ModalityDataItems,  # 模态数据项
    MultiModalDataItems,  # 多模态数据项
    MultiModalDataParser,  # 多模态数据解析器
)
from vllm.multimodal.processing import (  # 导入多模态处理相关类
    BaseDummyInputsBuilder,  # 虚拟输入构建器基类
    BaseMultiModalProcessor,  # 多模态处理器基类
    BaseProcessingInfo,  # 处理信息基类
    ProcessorInputs,  # 处理器输入
    PromptUpdate,  # 提示更新
    TimingContext,  # 计时上下文
)
from vllm.sequence import IntermediateTensors  # 导入中间张量类

from .interfaces import IsAttentionFree, MultiModalEmbeddings, SupportsMultiModal  # 导入无注意力、多模态嵌入、多模态支持接口
from .interfaces_base import attn_type  # 导入注意力类型装饰器

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


def _terratorch_field_names(input_definition: InputDefinition):
    """从输入定义中提取所有字段名称。"""
    return set(input_definition.data.keys())  # 返回输入定义中的数据键集合


def _terratorch_field_factory(
    input_definition: InputDefinition,  # 输入定义对象
    *,
    is_shared: bool = True,  # True表示未处理的数据，False表示已处理的数据
):
    """创建Terratorch字段配置工厂函数。"""

    def _terratorch_field_config(
        hf_inputs: Mapping[str, torch.Tensor],  # HuggingFace输入张量映射
    ) -> Mapping[str, MultiModalFieldConfig]:
        """根据输入定义生成多模态字段配置。"""
        fields = dict[str, MultiModalFieldConfig]()  # 初始化字段配置字典
        for name, input in input_definition.data.items():  # 遍历输入定义的数据
            modality = "image"  # 设置模态为图像
            if input.type == InputTypeEnum.tensor:  # 如果输入类型是张量
                fields[name] = (  # 根据是否共享选择字段配置
                    MultiModalFieldConfig.shared(modality, batch_size=1)  # 共享字段配置
                    if is_shared
                    else MultiModalFieldConfig.batched(modality)  # 批量字段配置
                )

        return fields  # 返回字段配置字典

    return _terratorch_field_config  # 返回字段配置工厂函数


class TerratorchMultiModalDataParser(MultiModalDataParser):
    """Terratorch多模态数据解析器，处理图像数据的自定义解析逻辑。"""

    def __init__(self, input_definition: InputDefinition, *args, **kwargs):
        """初始化Terratorch多模态数据解析器。"""
        super().__init__(*args, **kwargs)  # 调用父类初始化

        self.input_definition = input_definition  # 保存输入定义

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],  # 图像数据，可以是字典或模态数据
    ) -> ModalityDataItems[Any, Any] | None:
        """解析图像数据，支持字典格式和标准格式。"""
        if isinstance(data, dict):  # 如果数据是字典格式
            return DictEmbeddingItems(  # 创建字典嵌入项
                data,
                modality="image",  # 模态为图像
                required_fields=_terratorch_field_names(self.input_definition),  # 必需字段
                fields_factory=_terratorch_field_factory(self.input_definition),  # 字段工厂
            )

        return super()._parse_image_data(data)  # 否则调用父类方法处理

    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        """解析多模态数据，自动将非image键的数据包装为image模态。"""
        if "image" not in mm_data:  # 如果数据中没有image键
            mm_data = {"image": mm_data}  # 将数据包装在image键下

        return super().parse_mm_data(mm_data)  # 调用父类方法解析


class TerratorchProcessingInfo(BaseProcessingInfo):
    """Terratorch处理信息类，提供输入定义和数据解析器。"""

    @cached_property
    def input_definition(self) -> InputDefinition:
        """从预训练配置中获取并缓存输入定义。"""
        pretrained_cfg = self.get_hf_config().to_dict()["pretrained_cfg"]  # 获取预训练配置
        return InputDefinition(**pretrained_cfg["input"])  # 创建输入定义

    def get_data_parser(self):
        """获取Terratorch多模态数据解析器。"""
        return TerratorchMultiModalDataParser(  # 创建Terratorch数据解析器
            self.input_definition,  # 传入输入定义
            expected_hidden_size=self._get_expected_hidden_size(),  # 传入期望的隐藏层大小
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """获取支持的多模态数据限制，图像无上限。"""
        return {"image": None}  # 图像模态无数量限制


class TerratorchInputBuilder(BaseDummyInputsBuilder[TerratorchProcessingInfo]):
    """Terratorch虚拟输入构建器，用于生成模型分析所需的虚拟输入。"""

    def __init__(self, info: TerratorchProcessingInfo):
        """初始化Terratorch输入构建器和虚拟数据生成器。"""
        super().__init__(info)  # 调用父类初始化
        self.dummy_data_generator = DummyDataGenerator(  # 创建虚拟数据生成器
            self.info.get_hf_config().to_dict()["pretrained_cfg"]  # 传入预训练配置
        )

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """返回空字符串作为虚拟文本（Terratorch不需要文本输入）。"""
        return ""  # 返回空字符串

    def get_dummy_mm_data(
        self,
        seq_len: int,  # 序列长度
        mm_counts: Mapping[str, int],  # 多模态数据计数
        mm_options: Mapping[str, BaseDummyOptions],  # 多模态虚拟选项
    ) -> MultiModalDataDict:
        """生成虚拟多模态数据用于模型分析。"""
        # Dummy data is generated based on the 'input' section
        # defined in the HF configuration file

        if mm_options:  # 如果有自定义选项
            logger.warning(  # 发出警告
                "Configurable multimodal profiling "
                "options are not supported for Terratorch. "
                "They are ignored for now."
            )

        return self.dummy_data_generator.get_dummy_mm_data()  # 返回虚拟多模态数据


class TerratorchMultiModalProcessor(BaseMultiModalProcessor[TerratorchProcessingInfo]):
    """Terratorch多模态处理器，处理多模态输入的预处理和后处理。"""

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,  # HuggingFace批量特征
        hf_processor_mm_kwargs: Mapping[str, object],  # HuggingFace处理器多模态参数
        *,
        is_shared: bool = True,  # 是否为共享数据
    ) -> Mapping[str, MultiModalFieldConfig]:
        """获取多模态字段配置。"""
        factory = _terratorch_field_factory(  # 创建字段工厂
            self.info.input_definition,  # 传入输入定义
            is_shared=is_shared,  # 传入共享标志
        )
        return factory(hf_inputs)  # 调用工厂函数生成字段配置

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,  # 多模态数据项
        hf_processor_mm_kwargs: Mapping[str, object],  # HuggingFace处理器参数
        out_mm_kwargs: MultiModalKwargsItems,  # 输出多模态参数
    ) -> Sequence[PromptUpdate]:
        """获取提示更新列表（Terratorch不需要提示更新）。"""
        return []  # 返回空列表

    def apply(
        self,
        inputs: ProcessorInputs,  # 处理器输入
        timing_ctx: TimingContext,  # 计时上下文
    ) -> MultiModalInputs:
        """应用多模态处理器，将输入数据转换为模型可接受的格式。"""
        mm_items = inputs.mm_data_items  # 获取多模态数据项
        hf_processor_mm_kwargs = inputs.hf_processor_mm_kwargs  # 获取HuggingFace处理器参数

        with timing_ctx.record("apply_hf_processor"):  # 记录处理时间
            _, passthrough_data = self._get_hf_mm_data(mm_items)  # 获取HuggingFace多模态数据
            mm_processed_data = BatchFeature(  # 创建批量特征
                {
                    k: torch.as_tensor(v).unsqueeze(0)  # 将数据转为张量并增加批次维度
                    for k, v in passthrough_data.items()
                },
                tensor_type="pt",  # 使用PyTorch张量类型
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(  # 从HuggingFace输入创建多模态参数
            mm_processed_data,  # 处理后的多模态数据
            self._get_mm_fields_config(  # 获取字段配置
                mm_processed_data,
                hf_processor_mm_kwargs,
                is_shared=False,  # 已处理数据使用非共享模式
            ),
        )

        with timing_ctx.record("get_mm_hashes"):  # 记录哈希计算时间
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)  # 计算多模态哈希

        mm_placeholders = {"image": [PlaceholderRange(offset=0, length=0)]}  # 图像占位符范围（长度为0）

        return mm_inputs(  # 构造并返回多模态输入
            prompt_token_ids=[1],  # 虚拟提示token ID
            mm_kwargs=mm_kwargs,  # 多模态参数
            mm_hashes=mm_hashes,  # 多模态哈希
            mm_placeholders=mm_placeholders,  # 多模态占位符
        )


@attn_type("attention_free")  # 标记为无注意力机制模型
@MULTIMODAL_REGISTRY.register_processor(  # 注册多模态处理器
    TerratorchMultiModalProcessor,  # 指定处理器类
    info=TerratorchProcessingInfo,  # 指定处理信息类
    dummy_inputs=TerratorchInputBuilder,  # 指定虚拟输入构建器
)
class Terratorch(nn.Module, IsAttentionFree, SupportsMultiModal):
    """Terratorch地理空间模型，无注意力机制的池化模型，用于遥感图像处理。"""

    supports_multimodal_raw_input_only = True  # 仅支持原始多模态输入
    is_pooling_model = True  # 标记为池化模型

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """获取模态占位符字符串，仅支持图像模态。"""
        if modality.startswith("image"):  # 如果是图像模态
            return None  # 图像不需要占位符字符串

        raise ValueError("Only image modality is supported")  # 不支持其他模态

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        """初始化Terratorch模型，创建推理运行器和池化层。"""
        super().__init__()  # 调用父类初始化

        config = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]  # 获取预训练配置

        self.inference_runner = InferenceRunner(config)  # 创建推理运行器
        self.model = self.inference_runner.model  # 获取底层模型

        self.pooler = IdentityPooler()  # 创建恒等池化层

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        multimodal_embeddings: MultiModalEmbeddings | None = None,  # 多模态嵌入
        *,
        is_multimodal: torch.Tensor | None = None,  # 多模态标记张量
    ) -> torch.Tensor:
        """生成输入嵌入（Terratorch不使用文本token，返回空嵌入）。"""
        # We do not really use any input tokens and therefore no embeddings
        # to be calculated. However, due to the mandatory token ids in
        # the input prompt we pass one token and the size of the dummy
        # embedding tensors must reflect that.
        return torch.empty((input_ids.shape[0], 0))  # 返回空嵌入张量

    def forward(
        self,
        input_ids: torch.Tensor | None,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        intermediate_tensors: IntermediateTensors | None = None,  # 中间张量
        inputs_embeds: torch.Tensor | None = None,  # 输入嵌入
        **kwargs: object,  # 额外关键字参数（包含多模态数据）
    ):
        """前向传播，调用推理运行器处理输入并返回输出。"""
        model_output = self.inference_runner.forward(**kwargs)  # 调用推理运行器前向传播
        return model_output.output  # 返回模型输出

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载模型权重，处理state_dict格式和缓冲区加载。"""
        params_list = []  # 参数列表
        model_buffers = dict(self.named_buffers())  # 获取模型缓冲区字典
        loaded_buffers = []  # 已加载缓冲区列表
        for key, value in weights:  # 遍历待加载的权重
            if isinstance(value, (dict, OrderedDict)):  # 如果值是字典类型
                if key == "state_dict":  # 如果是state_dict格式
                    weights_to_parse = value  # 获取待解析的权重
                    for name, weight in weights_to_parse.items():  # 遍历权重项
                        name = f"inference_runner.{name}"  # 添加推理运行器前缀

                        if "pos_embed" in name:  # 跳过位置嵌入
                            continue

                        if "_timm_module." in name:  # 移除timm模块前缀
                            name = name.replace("_timm_module.", "")

                        # this model requires a couple of buffers to be loaded
                        # that are not loadable with the AutoWeightsLoader
                        if name in model_buffers:  # 如果是缓冲区
                            if "_timm_module." in name:  # 再次检查并移除timm前缀
                                name = name.replace("_timm_module.", "")
                            buffer = model_buffers[name]  # 获取缓冲区
                            weight_loader = getattr(  # 获取权重加载器
                                buffer, "weight_loader", default_weight_loader
                            )
                            weight_loader(buffer, weight)  # 加载缓冲区权重
                            loaded_buffers.append(name)  # 记录已加载缓冲区
                        else:  # 如果是普通参数
                            params_list.append((name, weight))  # 添加到参数列表
                    break

            elif isinstance(value, torch.Tensor):  # 如果值是张量
                params_list.append((f"inference_runner.model.{key}", value))  # 添加模型前缀

        # Load the remaining model parameters
        loader = AutoWeightsLoader(self)  # 创建自动权重加载器
        autoloaded_weights = loader.load_weights(params_list)  # 加载剩余权重

        return autoloaded_weights.union(set(loaded_buffers))  # 返回所有已加载权重的并集
