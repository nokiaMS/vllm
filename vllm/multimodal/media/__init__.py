# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .audio import AudioEmbeddingMediaIO, AudioMediaIO  # 导入音频媒体IO和音频嵌入媒体IO
from .base import MediaIO, MediaWithBytes  # 导入媒体IO基类和带字节的媒体包装类
from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector  # 导入媒体连接器注册表和媒体连接器
from .image import ImageEmbeddingMediaIO, ImageMediaIO  # 导入图像媒体IO和图像嵌入媒体IO
from .video import VIDEO_LOADER_REGISTRY, VideoMediaIO  # 导入视频加载器注册表和视频媒体IO

__all__ = [  # 定义模块的公开接口列表
    "MediaIO",  # 媒体IO基类
    "MediaWithBytes",  # 带字节的媒体包装类
    "AudioEmbeddingMediaIO",  # 音频嵌入媒体IO
    "AudioMediaIO",  # 音频媒体IO
    "ImageEmbeddingMediaIO",  # 图像嵌入媒体IO
    "ImageMediaIO",  # 图像媒体IO
    "VIDEO_LOADER_REGISTRY",  # 视频加载器注册表
    "VideoMediaIO",  # 视频媒体IO
    "MEDIA_CONNECTOR_REGISTRY",  # 媒体连接器注册表
    "MediaConnector",  # 媒体连接器
]
