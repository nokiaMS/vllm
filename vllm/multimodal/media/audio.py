# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64  # 导入base64编解码模块
from io import BytesIO  # 导入字节流IO类
from pathlib import Path  # 导入路径处理类

import numpy as np  # 导入NumPy数值计算库
import numpy.typing as npt  # 导入NumPy类型标注
import pybase64  # 导入高性能base64编解码库
import torch  # 导入PyTorch深度学习框架

from vllm.utils.import_utils import PlaceholderModule  # 导入占位模块工具
from vllm.utils.serial_utils import tensor2base64  # 导入张量转base64工具函数

from .base import MediaIO  # 从基类模块导入MediaIO

try:
    import librosa  # 尝试导入音频处理库librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]  # 导入失败时使用占位模块

try:
    import soundfile  # 尝试导入音频文件读写库soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]  # 导入失败时使用占位模块

try:
    import av  # 尝试导入PyAV音视频处理库
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]  # 导入失败时使用占位模块


def extract_audio_from_video_bytes(
    data: bytes,
) -> tuple[npt.NDArray, float]:
    """Extract the audio track from raw video bytes using PyAV.
    使用PyAV从原始视频字节中提取音频轨道。

    PyAV wraps FFmpeg's C libraries in-process — no subprocess is
    spawned, which is critical to avoid crashing CUDA-active vLLM
    worker processes.
    PyAV在进程内包装了FFmpeg的C库——不会产生子进程，这对避免崩溃CUDA活动的vLLM工作进程至关重要。

    The returned waveform is at the native sample rate of the video's
    audio stream.  Resampling to a model-specific rate is left to the
    downstream :class:`AudioResampler` in the parsing pipeline.
    返回的波形使用视频音频流的原始采样率。重采样到模型特定的采样率留给解析管道中下游的AudioResampler。

    Args:
        data: Raw video file bytes (e.g. from an mp4 file).
             原始视频文件字节（例如来自mp4文件）。

    Returns:
        A tuple of ``(waveform, sample_rate)`` suitable for use as an
        :class:`AudioItem`.
        一个``(波形, 采样率)``的元组，适合用作AudioItem。
    """
    if data is None or len(data) == 0:  # 如果数据为空
        raise ValueError(  # 抛出值错误
            "Cannot extract audio: video bytes are missing or empty. "
            "Ensure video was loaded with keep_video_bytes=True for "
            "audio-in-video extraction."
        )
    try:
        with av.open(BytesIO(data)) as container:  # 打开视频容器
            if not container.streams.audio:  # 如果没有音频流
                raise ValueError("No audio stream found in the video.")  # 抛出未找到音频流错误
            stream = container.streams.audio[0]  # 获取第一个音频流
            native_sr = stream.rate  # 获取原始采样率

            chunks: list[npt.NDArray] = []  # 初始化音频块列表
            for frame in container.decode(audio=0):  # 解码音频帧
                arr = frame.to_ndarray()  # 将帧转为numpy数组
                chunks.append(arr.mean(axis=0) if arr.ndim > 1 else arr)  # 多声道取平均，单声道直接添加
    except ValueError:  # 捕获值错误
        raise  # 重新抛出
    except Exception as e:  # 捕获其他异常
        raise ValueError(  # 包装为值错误抛出
            "Invalid or corrupted video data when extracting audio. "
            "Ensure the input is valid video bytes (e.g. a complete MP4)."
        ) from e

    if not chunks:  # 如果没有提取到音频块
        raise ValueError("No audio found in the video.")  # 抛出未找到音频错误

    audio = np.concatenate(chunks).astype(np.float32)  # 拼接所有音频块并转为float32
    return audio, float(native_sr)  # 返回音频数据和采样率


def is_video(data: bytes) -> bool:
    """Check if the fetched bytes are video
    检查获取的字节数据是否为视频格式。
    """
    if len(data) < 12:  # 如果数据太短
        return False  # 不是视频

    box_type = data[4:8]  # 提取box类型字段
    major_brand = data[8:12]  # 提取主品牌字段

    MP4_BRANDS = {  # MP4格式的品牌标识集合
        b"mp41",
        b"mp42",  # MP4
        b"isom",  # ISO Base Media  # ISO基础媒体
        b"iso2",
        b"iso4",
        b"iso5",
        b"iso6",
        b"M4V ",
        b"M4A ",  # Apple  # 苹果格式
        b"avc1",  # H.264
        b"dash",  # DASH
        b"mmp4",
        b"MSNV",
    }

    is_avi = data[:4] == b"RIFF" and major_brand == b"AVI "  # 检查是否为AVI格式
    is_mp4 = box_type == b"ftyp" and major_brand in MP4_BRANDS  # 检查是否为MP4格式
    return is_mp4 or is_avi  # 返回是否为视频


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    音频媒体IO类。配置值可以通过--media-io-kwargs或运行时API字段提供。
    """

    def __init__(self, **kwargs) -> None:
        """初始化音频媒体IO。"""
        super().__init__()  # 调用父类初始化

        # `kwargs` contains custom arguments from  # kwargs包含来自以下来源的自定义参数
        # --media-io-kwargs for this modality, merged with  # 此模态的--media-io-kwargs，与
        # per-request runtime media_io_kwargs via merge_kwargs().  # 每个请求的运行时media_io_kwargs通过merge_kwargs()合并
        # They can be passed to the underlying  # 它们可以传递给底层
        # media loaders (e.g. custom implementations)  # 媒体加载器（如自定义实现）
        # for flexible control.  # 以实现灵活控制
        self.kwargs = kwargs  # 保存额外的关键字参数

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        """从字节数据加载音频，支持从视频中提取音频。"""
        if is_video(data):  # 如果是视频数据
            return extract_audio_from_video_bytes(data)  # 从视频中提取音频
        return librosa.load(BytesIO(data), sr=None)  # 使用librosa加载音频

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        """从base64编码的字符串加载音频。"""
        return self.load_bytes(base64.b64decode(data))  # 解码base64后加载

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        """从文件路径加载音频。"""
        return librosa.load(filepath, sr=None)  # 使用librosa从文件加载

    def encode_base64(
        self,
        media: tuple[npt.NDArray, int],
        *,
        audio_format: str = "WAV",
    ) -> str:
        """将音频编码为base64字符串。"""
        audio, sr = media  # 解包音频数据和采样率

        with BytesIO() as buffer:  # 创建字节缓冲区
            soundfile.write(buffer, audio, sr, format=audio_format)  # 写入音频数据
            data = buffer.getvalue()  # 获取缓冲区内容

        return base64.b64encode(data).decode("utf-8")  # 编码为base64并返回字符串


class AudioEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    音频嵌入媒体IO类，用于加载和处理音频嵌入张量。
    """

    def __init__(self) -> None:
        """初始化音频嵌入媒体IO。"""
        super().__init__()  # 调用父类初始化

    def load_bytes(self, data: bytes) -> torch.Tensor:
        """从字节数据加载音频嵌入张量。"""
        buffer = BytesIO(data)  # 创建字节缓冲区
        # Enable sparse tensor integrity checks to prevent out-of-bounds  # 启用稀疏张量完整性检查以防止越界
        # writes from maliciously crafted tensors  # 写入来自恶意制作的张量
        with torch.sparse.check_sparse_tensor_invariants():  # 开启稀疏张量不变量检查
            tensor = torch.load(buffer, weights_only=True)  # 仅加载权重
            return tensor.to_dense()  # 转为稠密张量

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        """从base64编码的字符串加载音频嵌入张量。"""
        return self.load_bytes(pybase64.b64decode(data, validate=True))  # 解码base64后加载

    def load_file(self, filepath: Path) -> torch.Tensor:
        """从文件路径加载音频嵌入张量。"""
        # Enable sparse tensor integrity checks to prevent out-of-bounds  # 启用稀疏张量完整性检查以防止越界
        # writes from maliciously crafted tensors  # 写入来自恶意制作的张量
        with torch.sparse.check_sparse_tensor_invariants():  # 开启稀疏张量不变量检查
            tensor = torch.load(filepath, weights_only=True)  # 仅加载权重
            return tensor.to_dense()  # 转为稠密张量

    def encode_base64(self, media: torch.Tensor) -> str:
        """将音频嵌入张量编码为base64字符串。"""
        return tensor2base64(media)  # 使用工具函数转换
