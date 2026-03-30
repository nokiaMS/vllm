# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""视频资产管理模块。
提供视频测试资产的下载、帧提取、元数据获取等功能。
"""

from dataclasses import dataclass  # 导入数据类装饰器
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Any, ClassVar, Literal  # 导入类型注解工具

import numpy as np  # 导入NumPy数组库
import numpy.typing as npt  # 导入NumPy类型注解
from huggingface_hub import hf_hub_download  # 导入HuggingFace Hub文件下载函数
from PIL import Image  # 导入PIL图像处理库

from vllm.utils.import_utils import PlaceholderModule  # 导入占位模块工具

from .base import get_cache_dir  # 从基础模块导入缓存目录获取函数

try:  # 尝试导入librosa音频处理库
    import librosa  # 导入librosa库
except ImportError:  # 如果导入失败
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]  # 使用占位模块替代


@lru_cache  # 使用LRU缓存，避免重复下载
def download_video_asset(filename: str) -> str:
    """
    Download and open an image from huggingface
    repo: raushan-testing-hf/videos-test
    从HuggingFace仓库raushan-testing-hf/videos-test下载视频文件。

    Args:
        filename: 要下载的视频文件名。

    Returns:
        视频文件的本地路径字符串。
    """
    video_directory = get_cache_dir() / "video-example-data"  # 构建视频缓存目录路径
    video_directory.mkdir(parents=True, exist_ok=True)  # 递归创建目录

    video_path = video_directory / filename  # 构建视频文件完整路径
    video_path_str = str(video_path)  # 转换为字符串路径
    if not video_path.exists():  # 如果文件不存在
        video_path_str = hf_hub_download(  # 从HuggingFace Hub下载
            repo_id="raushan-testing-hf/videos-test",  # 视频测试数据仓库
            filename=filename,  # 下载的文件名
            repo_type="dataset",  # 仓库类型为数据集
            cache_dir=video_directory,  # 缓存到指定目录
        )
    return video_path_str  # 返回视频文件路径


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    """将视频文件转换为NumPy数组。
    从视频中均匀采样指定数量的帧并返回RGB格式的数组。

    Args:
        path: 视频文件路径。
        num_frames: 要提取的帧数，-1表示提取所有帧。

    Returns:
        形状为(num_frames, height, width, 3)的RGB图像数组。

    Raises:
        ValueError: 当无法打开视频或提取的帧数不足时。
    """
    import cv2  # 导入OpenCV库

    cap = cv2.VideoCapture(path)  # 打开视频文件
    if not cap.isOpened():  # 如果无法打开视频
        raise ValueError(f"Could not open video file {path}")  # 抛出值错误

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    frames = []  # 帧列表

    num_frames = num_frames if num_frames > 0 else total_frames  # 如果num_frames为-1则使用总帧数
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)  # 均匀采样帧索引
    for idx in range(total_frames):  # 遍历所有帧
        ok = cap.grab()  # next img  # 抓取下一帧
        if not ok:  # 如果抓取失败
            break  # 退出循环
        if idx in frame_indices:  # only decompress needed  # 仅解压需要的帧
            ret, frame = cap.retrieve()  # 解码当前帧
            if ret:  # 如果解码成功
                # OpenCV uses BGR format, we need to convert it to RGB
                # for PIL and transformers compatibility
                # OpenCV使用BGR格式，需要转换为RGB以兼容PIL和transformers
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR转RGB并添加到列表

    frames = np.stack(frames)  # 将帧列表堆叠为NumPy数组
    if len(frames) < num_frames:  # 如果提取的帧数不足
        raise ValueError(  # 抛出值错误
            f"Could not read enough frames from video file {path}"
            f" (expected {num_frames} frames, got {len(frames)})"
        )
    return frames  # 返回帧数组


def video_to_pil_images_list(path: str, num_frames: int = -1) -> list[Image.Image]:
    """将视频文件转换为PIL图像列表。

    Args:
        path: 视频文件路径。
        num_frames: 要提取的帧数，-1表示所有帧。

    Returns:
        PIL Image对象列表。
    """
    frames = video_to_ndarrays(path, num_frames)  # 将视频转换为NumPy数组
    return [Image.fromarray(frame) for frame in frames]  # 将每帧转换为PIL图像


def video_get_metadata(path: str, num_frames: int = -1) -> dict[str, Any]:
    """获取视频文件的元数据信息。

    Args:
        path: 视频文件路径。
        num_frames: 要使用的帧数，-1表示所有帧。

    Returns:
        包含总帧数、FPS、时长等信息的字典。

    Raises:
        ValueError: 当无法打开视频文件时。
    """
    import cv2  # 导入OpenCV库

    cap = cv2.VideoCapture(path)  # 打开视频文件
    if not cap.isOpened():  # 如果无法打开
        raise ValueError(f"Could not open video file {path}")  # 抛出值错误

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    duration = total_frames / fps if fps > 0 else 0  # 计算视频时长（秒）

    if num_frames == -1 or num_frames > total_frames:  # 如果未指定帧数或超出总帧数
        num_frames = total_frames  # 使用总帧数

    metadata = {  # 构建元数据字典
        "total_num_frames": num_frames,  # 总帧数
        "fps": duration / num_frames,  # 每帧时长
        "duration": duration,  # 视频总时长
        "video_backend": "opencv",  # 使用的视频后端
        "frames_indices": list(range(num_frames)),  # 帧索引列表
        # extra field used to control hf processor's video
        # sampling behavior
        # 额外字段，用于控制HuggingFace处理器的视频采样行为
        "do_sample_frames": num_frames == total_frames,  # 是否需要采样帧
    }
    return metadata  # 返回元数据字典


VideoAssetName = Literal["baby_reading"]  # 视频资产名称的字面量类型（婴儿阅读）


@dataclass(frozen=True)  # 不可变数据类装饰器
class VideoAsset:
    """视频资产数据类。
    封装视频测试资产的访问方法，支持从HuggingFace下载和本地缓存。

    Attributes:
        name: 视频资产名称。
        num_frames: 要提取的帧数，-1表示所有帧。
    """

    name: VideoAssetName  # 视频资产名称
    num_frames: int = -1  # 要提取的帧数（-1表示全部）

    _NAME_TO_FILE: ClassVar[dict[VideoAssetName, str]] = {  # 名称到文件名的类级映射
        "baby_reading": "sample_demo_1.mp4",  # 婴儿阅读视频文件名
    }

    @property
    def filename(self) -> str:
        """获取视频文件名。"""
        return self._NAME_TO_FILE[self.name]  # 通过名称映射获取文件名

    @property
    def video_path(self) -> str:
        """获取视频文件的本地路径。"""
        return download_video_asset(self.filename)  # 下载并返回本地路径

    @property
    def pil_images(self) -> list[Image.Image]:
        """将视频转换为PIL图像列表。

        Returns:
            视频帧的PIL Image列表。
        """
        ret = video_to_pil_images_list(self.video_path, self.num_frames)  # 转换视频为PIL图像列表
        return ret  # 返回图像列表

    @property
    def np_ndarrays(self) -> npt.NDArray:
        """将视频转换为NumPy数组。

        Returns:
            视频帧的NumPy数组。
        """
        ret = video_to_ndarrays(self.video_path, self.num_frames)  # 转换视频为NumPy数组
        return ret  # 返回帧数组

    @property
    def metadata(self) -> dict[str, Any]:
        """获取视频的元数据信息。

        Returns:
            包含帧数、FPS、时长等信息的字典。
        """
        ret = video_get_metadata(self.video_path, self.num_frames)  # 获取视频元数据
        return ret  # 返回元数据

    def get_audio(self, sampling_rate: float | None = None) -> npt.NDArray:
        """
        Read audio data from the video asset, used in Qwen2.5-Omni examples.
        从视频资产中读取音频数据，用于Qwen2.5-Omni示例。

        See also: examples/offline_inference/qwen2_5_omni/only_thinker.py
        参见: examples/offline_inference/qwen2_5_omni/only_thinker.py

        Args:
            sampling_rate: 目标采样率，None表示使用原始采样率。

        Returns:
            音频数据的NumPy数组。
        """
        return librosa.load(self.video_path, sr=sampling_rate)[0]  # 使用librosa加载视频中的音频数据
