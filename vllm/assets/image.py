# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""图像资产管理模块。
提供图像测试资产的下载、加载和访问功能。
"""

from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入路径操作类
from typing import Literal  # 导入字面量类型

import torch  # 导入PyTorch框架
from PIL import Image  # 导入PIL图像处理库

from .base import get_vllm_public_assets  # 从基础模块导入资产获取函数

VLM_IMAGES_DIR = "vision_model_images"  # 视觉模型图像资产目录名

ImageAssetName = Literal[  # 图像资产名称的字面量类型
    "stop_sign",  # 停车标志
    "cherry_blossom",  # 樱花
    "hato",  # 鸽子
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk",  # 威斯康星麦迪逊自然步道
    "Grayscale_8bits_palette_sample_image",  # 8位灰度调色板样本图像
    "1280px-Venn_diagram_rgb",  # RGB维恩图
    "RGBA_comp",  # RGBA合成图
    "237-400x300",  # 400x300测试图像
    "231-200x300",  # 200x300测试图像
    "27-500x500",  # 500x500测试图像
    "17-150x600",  # 150x600测试图像
    "handelsblatt-preview",  # 商报预览图
    "paper-11",  # 论文图像11
]


@dataclass(frozen=True)  # 不可变数据类装饰器
class ImageAsset:
    """图像资产数据类。
    封装图像测试资产的访问方法，支持从S3下载和本地缓存。

    Attributes:
        name: 图像资产名称。
    """

    name: ImageAssetName  # 图像资产名称

    def get_path(self, ext: str) -> Path:
        """
        Return s3 path for given image.
        返回给定图像的S3路径。

        Args:
            ext: 文件扩展名（如jpg、png、pt等）。

        Returns:
            图像文件的本地缓存路径。
        """
        return get_vllm_public_assets(  # 获取公共资产的本地路径
            filename=f"{self.name}.{ext}", s3_prefix=VLM_IMAGES_DIR  # 构建带扩展名的文件名
        )

    @property
    def pil_image(self) -> Image.Image:
        """获取PIL图像对象（默认jpg格式）。

        Returns:
            PIL Image对象。
        """
        return self.pil_image_ext(ext="jpg")  # 默认加载jpg格式

    def pil_image_ext(self, ext: str) -> Image.Image:
        """获取指定格式的PIL图像对象。

        Args:
            ext: 图像文件扩展名。

        Returns:
            PIL Image对象。
        """
        image_path = self.get_path(ext=ext)  # 获取指定扩展名的文件路径
        return Image.open(image_path)  # 使用PIL打开图像

    @property
    def image_embeds(self) -> torch.Tensor:
        """
        Image embeddings, only used for testing purposes with llava 1.5.
        图像嵌入张量，仅用于llava 1.5的测试目的。

        Returns:
            图像嵌入的PyTorch张量。
        """
        image_path = self.get_path("pt")  # 获取.pt格式的嵌入文件路径
        return torch.load(image_path, map_location="cpu", weights_only=True)  # 加载嵌入张量到CPU

    def read_bytes(self, ext: str) -> bytes:
        """读取指定格式图像文件的原始字节数据。

        Args:
            ext: 图像文件扩展名。

        Returns:
            图像文件的原始字节内容。
        """
        p = Path(self.get_path(ext))  # 获取文件路径
        return p.read_bytes()  # 读取并返回文件的字节内容
