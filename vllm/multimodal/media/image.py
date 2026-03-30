# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO  # 导入字节流IO类
from pathlib import Path  # 导入路径处理类

import pybase64  # 导入高性能base64编解码库
import torch  # 导入PyTorch深度学习框架
from PIL import Image  # 导入PIL图像处理库

from vllm.utils.serial_utils import tensor2base64  # 导入张量转base64工具函数

from ..image import convert_image_mode, rgba_to_rgb  # 从上级image模块导入图像模式转换函数
from .base import MediaIO, MediaWithBytes  # 从基类模块导入MediaIO和MediaWithBytes


class ImageMediaIO(MediaIO[Image.Image]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    图像媒体IO类，负责图像的加载、转换和编码。
    """

    def __init__(self, image_mode: str = "RGB", **kwargs) -> None:
        """初始化图像媒体IO。"""
        super().__init__()  # 调用父类初始化

        self.image_mode = image_mode  # 设置目标图像模式
        # `kwargs` contains custom arguments from  # kwargs包含来自以下来源的自定义参数
        # --media-io-kwargs for this modality, merged with  # 此模态的--media-io-kwargs，与
        # per-request runtime media_io_kwargs via merge_kwargs().  # 每个请求的运行时参数合并
        # They can be passed to the underlying  # 它们可以传递给底层
        # media loaders (e.g. custom implementations)  # 媒体加载器
        # for flexible control.  # 以实现灵活控制
        self.kwargs = kwargs  # 保存额外的关键字参数

        # Extract RGBA background color from kwargs if provided  # 如果提供了RGBA背景颜色则提取
        # Default to white background for backward compatibility  # 默认为白色背景以保持向后兼容
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))  # 获取RGBA背景颜色配置
        # Convert list to tuple for consistency  # 将列表转为元组以保持一致性
        if isinstance(rgba_bg, list):  # 如果是列表类型
            rgba_bg = tuple(rgba_bg)  # 转为元组

        # Validate rgba_background_color format  # 验证RGBA背景颜色格式
        if not (  # 如果格式不正确
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(  # 抛出格式错误
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg  # 保存RGBA背景颜色

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color.
        使用自定义背景颜色转换图像模式。
        """
        if isinstance(image, MediaWithBytes):  # 如果是带字节的媒体包装
            image = image.media  # 解包获取底层图像
        if image.mode == self.image_mode:  # 如果已经是目标模式
            return image  # 直接返回
        elif image.mode == "RGBA" and self.image_mode == "RGB":  # 如果是RGBA转RGB
            return rgba_to_rgb(image, self.rgba_background_color)  # 使用自定义背景色转换
        else:  # 其他模式转换
            return convert_image_mode(image, self.image_mode)  # 使用通用转换函数

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]:
        """从字节数据加载图像，返回带原始字节的包装对象。"""
        image = Image.open(BytesIO(data))  # 从字节流打开图像
        return MediaWithBytes(self._convert_image_mode(image), data)  # 返回转换后的图像和原始字节

    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        """从base64编码的字符串加载图像。"""
        return self.load_bytes(pybase64.b64decode(data, validate=True))  # 解码base64后加载

    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]:
        """从文件路径加载图像。"""
        with open(filepath, "rb") as f:  # 以二进制模式打开文件
            data = f.read()  # 读取文件内容
        image = Image.open(BytesIO(data))  # 从字节流打开图像
        return MediaWithBytes(self._convert_image_mode(image), data)  # 返回转换后的图像和原始字节

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        """将图像编码为base64字符串。"""
        image = media  # 获取图像

        with BytesIO() as buffer:  # 创建字节缓冲区
            image = self._convert_image_mode(image)  # 转换图像模式
            image.save(buffer, image_format)  # 保存图像到缓冲区
            data = buffer.getvalue()  # 获取缓冲区内容

        return pybase64.b64encode(data).decode("utf-8")  # 编码为base64并返回字符串


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.
    图像嵌入媒体IO实现。

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    配置值可以通过--media-io-kwargs或运行时API字段提供。
    """

    def __init__(self) -> None:
        """初始化图像嵌入媒体IO。"""
        super().__init__()  # 调用父类初始化

    def load_bytes(self, data: bytes) -> torch.Tensor:
        """从字节数据加载图像嵌入张量。"""
        buffer = BytesIO(data)  # 创建字节缓冲区
        # Enable sparse tensor integrity checks to prevent out-of-bounds  # 启用稀疏张量完整性检查以防止越界
        # writes from maliciously crafted tensors  # 写入来自恶意制作的张量
        with torch.sparse.check_sparse_tensor_invariants():  # 开启稀疏张量不变量检查
            tensor = torch.load(buffer, weights_only=True)  # 仅加载权重
            return tensor.to_dense()  # 转为稠密张量

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        """从base64编码的字符串加载图像嵌入张量。"""
        return self.load_bytes(pybase64.b64decode(data, validate=True))  # 解码base64后加载

    def load_file(self, filepath: Path) -> torch.Tensor:
        """从文件路径加载图像嵌入张量。"""
        # Enable sparse tensor integrity checks to prevent out-of-bounds  # 启用稀疏张量完整性检查
        # writes from maliciously crafted tensors  # 防止恶意张量的越界写入
        with torch.sparse.check_sparse_tensor_invariants():  # 开启稀疏张量不变量检查
            tensor = torch.load(filepath, weights_only=True)  # 仅加载权重
            return tensor.to_dense()  # 转为稠密张量

    def encode_base64(self, media: torch.Tensor) -> str:
        """将图像嵌入张量编码为base64字符串。"""
        return tensor2base64(media)  # 使用工具函数转换
