# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""音频资产管理模块。
提供音频测试资产的下载、加载和访问功能。
"""

from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入路径操作类
from typing import Literal  # 导入字面量类型
from urllib.parse import urljoin  # 导入URL拼接函数

import numpy.typing as npt  # 导入NumPy类型注解

from vllm.utils.import_utils import PlaceholderModule  # 导入占位模块工具

from .base import VLLM_S3_BUCKET_URL, get_vllm_public_assets  # 从基础模块导入S3 URL和资产获取函数

try:  # 尝试导入librosa音频处理库
    import librosa  # 导入librosa库
except ImportError:  # 如果导入失败
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]  # 使用占位模块替代

ASSET_DIR = "multimodal_asset"  # 多模态资产目录名

AudioAssetName = Literal["winning_call", "mary_had_lamb"]  # 音频资产名称的字面量类型（获胜电话、玛丽有只小羊羔）


@dataclass(frozen=True)  # 不可变数据类装饰器
class AudioAsset:
    """音频资产数据类。
    封装音频测试资产的访问方法，支持从S3下载和本地缓存。

    Attributes:
        name: 音频资产名称。
    """

    name: AudioAssetName  # 音频资产名称

    @property
    def filename(self) -> str:
        """获取音频文件名。
        返回带.ogg扩展名的文件名。
        """
        return f"{self.name}.ogg"  # 返回ogg格式的文件名

    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, float]:
        """加载音频数据和采样率。
        从S3下载音频文件（如果需要）并使用librosa加载。

        Returns:
            (音频数据数组, 采样率)元组。
        """
        audio_path = get_vllm_public_assets(filename=self.filename, s3_prefix=ASSET_DIR)  # 获取音频文件的本地路径
        return librosa.load(audio_path, sr=None)  # 使用librosa加载音频，保持原始采样率

    def get_local_path(self) -> Path:
        """获取音频文件的本地缓存路径。

        Returns:
            音频文件的本地Path对象。
        """
        return get_vllm_public_assets(filename=self.filename, s3_prefix=ASSET_DIR)  # 返回本地缓存路径

    @property
    def url(self) -> str:
        """获取音频资产的S3 URL。

        Returns:
            音频文件的完整S3 URL。
        """
        return urljoin(VLLM_S3_BUCKET_URL, f"{ASSET_DIR}/{self.name}.ogg")  # 拼接完整的S3 URL
