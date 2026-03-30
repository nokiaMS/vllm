# SPDX-License-Identifier: Apache-2.0  # Apache-2.0许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明


from transformers import SmolVLMProcessor  # 导入SmolVLM处理器

from vllm.config import VllmConfig  # 导入vLLM配置类
from vllm.multimodal import MULTIMODAL_REGISTRY  # 导入多模态注册表

from .idefics3 import Idefics3DummyInputsBuilder as SmolVLMDummyInputsBuilder  # 导入Idefics3虚拟输入构建器并重命名为SmolVLM版本
from .idefics3 import Idefics3ForConditionalGeneration, Idefics3ProcessingInfo  # 导入Idefics3条件生成模型和处理信息类
from .idefics3 import Idefics3MultiModalProcessor as SmolVLMMultiModalProcessor  # 导入Idefics3多模态处理器并重命名为SmolVLM版本


class SmolVLMProcessingInfo(Idefics3ProcessingInfo):
    """SmolVLM处理信息类，继承自Idefics3Process理信息，用于处理SmolVLM特定的处理器和图像标记。"""

    def get_hf_processor(self, **kwargs: object) -> SmolVLMProcessor:
        """获取HuggingFace SmolVLM处理器实例。"""
        return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)  # 通过上下文获取SmolVLM处理器

    def _get_image_token(self, processor: SmolVLMProcessor) -> tuple[str, str, str]:
        """获取图像标记、伪图像标记和全局图像标记的元组。"""
        image_token = processor.image_token  # 获取图像标记
        fake_image_token = processor.fake_image_token  # 获取伪图像标记
        global_image_token = processor.global_image_token  # 获取全局图像标记
        return image_token, fake_image_token, global_image_token  # 返回三个图像标记的元组


@MULTIMODAL_REGISTRY.register_processor(  # 注册多模态处理器装饰器
    SmolVLMMultiModalProcessor,  # 指定SmolVLM多模态处理器
    info=SmolVLMProcessingInfo,  # 指定处理信息类
    dummy_inputs=SmolVLMDummyInputsBuilder,  # 指定虚拟输入构建器
)
class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    """SmolVLM条件生成模型，继承自Idefics3条件生成模型，是Idefics3的轻量级封装。"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化SmolVLM条件生成模型。"""
        super().__init__(  # 调用父类Idefics3的初始化方法
            vllm_config=vllm_config,  # 传入vLLM配置
            prefix=prefix,  # 传入参数名前缀
        )
