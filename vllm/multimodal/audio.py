# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math  # 导入数学运算模块
from dataclasses import dataclass  # 导入数据类装饰器
from enum import Enum  # 导入枚举类
from typing import Literal  # 导入字面量类型

import numpy as np  # 导入NumPy数值计算库
import numpy.typing as npt  # 导入NumPy类型标注
import torch  # 导入PyTorch深度学习框架

from vllm.utils.import_utils import PlaceholderModule  # 导入占位模块工具

try:
    import librosa  # 尝试导入音频处理库librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]  # 导入失败时使用占位模块替代

try:
    import scipy.signal as scipy_signal  # 尝试导入SciPy信号处理模块
except ImportError:
    scipy_signal = PlaceholderModule("scipy").placeholder_attr("signal")  # type: ignore[assignment]  # 导入失败时使用占位模块替代

# ============================================================


class ChannelReduction(str, Enum):
    """Method to reduce multi-channel audio to target channels.
    将多声道音频降至目标声道数的方法枚举。
    """

    MEAN = "mean"  # Average across channels (default, preserves energy balance)  # 跨声道取平均值（默认，保持能量平衡）
    FIRST = "first"  # Take first channel only  # 仅取第一个声道
    MAX = "max"  # Take max value across channels  # 跨声道取最大值
    SUM = "sum"  # Sum across channels  # 跨声道求和


@dataclass  # 数据类装饰器
class AudioSpec:
    """Specification for target audio format.
    目标音频格式的规格说明。

    This dataclass defines the expected audio format for a model's feature
    extractor. It is used to normalize audio data before processing.
    此数据类定义了模型特征提取器所需的音频格式，用于在处理前对音频数据进行标准化。

    Attributes:
        target_channels: Number of output channels. None means passthrough
            (no normalization). 1 = mono, 2 = stereo, etc.
            输出声道数。None表示直通（不做标准化）。1=单声道，2=立体声，等等。
        channel_reduction: Method to reduce channels when input has more
            channels than target. Only used when reducing channels.
            当输入声道数多于目标时的降声道方法。仅在降声道时使用。
    """

    target_channels: int | None = 1  # 目标声道数，默认为单声道
    channel_reduction: ChannelReduction = ChannelReduction.MEAN  # 声道降维方法，默认取平均

    @property
    def needs_normalization(self) -> bool:
        """Whether audio normalization is needed.
        是否需要音频标准化。
        """
        return self.target_channels is not None  # 如果目标声道数不为None则需要标准化

    def __repr__(self) -> str:
        """返回AudioSpec的字符串表示。"""
        if self.target_channels is None:  # 如果是直通模式
            return "AudioSpec(passthrough)"  # 返回直通表示
        return (  # 返回包含声道数和降维方法的表示
            f"AudioSpec(channels={self.target_channels}, "
            f"reduction={self.channel_reduction.value})"
        )


# Pre-defined specs for common use cases  # 常用场景的预定义规格
MONO_AUDIO_SPEC = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)  # 单声道音频规格
PASSTHROUGH_AUDIO_SPEC = AudioSpec(target_channels=None)  # 直通音频规格（不做处理）


def normalize_audio(
    audio: npt.NDArray[np.floating] | torch.Tensor,
    spec: AudioSpec,
) -> npt.NDArray[np.floating] | torch.Tensor:
    """Normalize audio to the specified format.
    将音频标准化为指定格式。

    This function handles channel reduction for multi-channel audio,
    supporting both numpy arrays and torch tensors.
    此函数处理多声道音频的声道降维，同时支持numpy数组和torch张量。

    Args:
        audio: Input audio data. Can be:
            - 1D array/tensor: (time,) - already mono
            - 2D array/tensor: (channels, time) - standard format from torchaudio
            - 2D array/tensor: (time, channels) - format from soundfile
              (will be auto-detected and transposed if time > channels)
            输入音频数据。可以是：
            - 一维数组/张量：(时间,) - 已经是单声道
            - 二维数组/张量：(声道, 时间) - torchaudio标准格式
            - 二维数组/张量：(时间, 声道) - soundfile格式（如果时间>声道会自动检测并转置）
        spec: AudioSpec defining the target format.
            定义目标格式的AudioSpec。

    Returns:
        Normalized audio in the same type as input (numpy or torch).
        For mono output (target_channels=1), returns 1D array/tensor.
        与输入相同类型的标准化音频。对于单声道输出返回一维数组/张量。

    Raises:
        ValueError: If audio has unsupported dimensions or channel expansion
            is requested (e.g., mono to stereo).
        如果音频维度不支持或请求了声道扩展则抛出ValueError。
    """
    if not spec.needs_normalization:  # 如果不需要标准化
        return audio  # 直接返回原始音频

    # Handle 1D audio (already mono)  # 处理一维音频（已经是单声道）
    if audio.ndim == 1:  # 如果是一维数组
        if spec.target_channels == 1:  # 如果目标是单声道
            return audio  # 直接返回
        raise ValueError(f"Cannot expand mono audio to {spec.target_channels} channels")  # 无法将单声道扩展为多声道

    # Handle 2D audio  # 处理二维音频
    if audio.ndim != 2:  # 如果不是二维
        raise ValueError(f"Unsupported audio shape: {audio.shape}. Expected 1D or 2D.")  # 不支持的音频维度

    # Auto-detect format: if shape[0] > shape[1], assume (time, channels)  # 自动检测格式：如果shape[0] > shape[1]，假设为(时间, 声道)格式
    # This handles soundfile format where time dimension is typically much larger  # 处理soundfile格式，其中时间维度通常远大于声道维度
    if audio.shape[0] > audio.shape[1]:  # 如果第一维大于第二维
        # Transpose from (time, channels) to (channels, time)  # 从(时间, 声道)转置为(声道, 时间)
        audio = audio.T if isinstance(audio, np.ndarray) else audio.T  # 执行转置

    num_channels = audio.shape[0]  # 获取声道数

    # No reduction needed if already at target  # 如果已经是目标声道数则无需降维
    if num_channels == spec.target_channels:  # 检查是否等于目标声道数
        return audio  # 直接返回

    # Cannot expand channels  # 无法扩展声道
    if num_channels < spec.target_channels:  # 如果当前声道数小于目标
        raise ValueError(  # 抛出无法扩展声道的错误
            f"Cannot expand {num_channels} channels to {spec.target_channels}"
        )

    # Reduce channels  # 降低声道数
    is_numpy = isinstance(audio, np.ndarray)  # 判断是否为numpy数组

    if spec.target_channels == 1:  # 如果目标是单声道
        # Reduce to mono  # 降为单声道
        if spec.channel_reduction == ChannelReduction.MEAN:  # 如果使用平均值降维
            result = np.mean(audio, axis=0) if is_numpy else audio.mean(dim=0)  # 计算声道平均值
        elif spec.channel_reduction == ChannelReduction.FIRST:  # 如果使用取第一声道
            result = audio[0]  # 取第一个声道
        elif spec.channel_reduction == ChannelReduction.MAX:  # 如果使用最大值降维
            result = np.max(audio, axis=0) if is_numpy else audio.max(dim=0).values  # 取声道最大值
        elif spec.channel_reduction == ChannelReduction.SUM:  # 如果使用求和降维
            result = np.sum(audio, axis=0) if is_numpy else audio.sum(dim=0)  # 计算声道求和
        else:  # 未知的降维方法
            raise ValueError(f"Unknown reduction method: {spec.channel_reduction}")  # 抛出未知降维方法错误
        return result  # 返回降维结果
    else:  # 如果目标不是单声道
        # Reduce to N channels (take first N and apply reduction if needed)  # 降至N个声道（取前N个声道）
        # For now, just take first N channels  # 目前仅取前N个声道
        return audio[: spec.target_channels]  # 返回前N个声道


# ============================================================
# Audio Resampling  # 音频重采样
# ============================================================


def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    """使用librosa库对音频进行重采样。"""
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)  # 调用librosa重采样函数


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    """使用SciPy库对音频进行重采样。"""
    if orig_sr > target_sr:  # 如果原始采样率高于目标采样率（下采样）
        return scipy_signal.resample_poly(audio, 1, orig_sr // target_sr)  # 执行多相滤波下采样
    elif orig_sr < target_sr:  # 如果原始采样率低于目标采样率（上采样）
        return scipy_signal.resample_poly(audio, target_sr // orig_sr, 1)  # 执行多相滤波上采样
    return audio  # 采样率相同则直接返回


class AudioResampler:
    """Resample audio data to a target sample rate.
    将音频数据重采样到目标采样率。
    """

    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["librosa", "scipy"] = "librosa",
    ):
        """初始化音频重采样器。"""
        self.target_sr = target_sr  # 设置目标采样率
        self.method = method  # 设置重采样方法

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        """执行音频重采样操作。"""
        if self.target_sr is None:  # 如果未设置目标采样率
            raise RuntimeError(  # 抛出运行时错误
                "Audio resampling is not supported when `target_sr` is not provided"
            )
        if math.isclose(  # 如果原始采样率与目标采样率足够接近
            float(orig_sr),
            float(self.target_sr),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            return audio  # 直接返回原始音频，无需重采样
        if self.method == "librosa":  # 如果使用librosa方法
            return resample_audio_librosa(  # 调用librosa重采样
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        elif self.method == "scipy":  # 如果使用scipy方法
            return resample_audio_scipy(  # 调用scipy重采样
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        else:  # 无效的重采样方法
            raise ValueError(  # 抛出值错误
                f"Invalid resampling method: {self.method}. "
                "Supported methods are 'librosa' and 'scipy'."
            )


# ============================================================
# Audio Chunking / Splitting  # 音频分块/分割
# ============================================================


def split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    max_clip_duration_s: float,
    overlap_duration_s: float,
    min_energy_window_size: int,
) -> list[np.ndarray]:
    """Split audio into chunks with intelligent split points.
    将音频智能分割为多个片段。

    Splits long audio into smaller chunks at low-energy regions to minimize
    cutting through speech. Uses overlapping windows to find quiet moments
    for splitting.
    在低能量区域将长音频分割为较小的片段，以最小化对语音的切割。
    使用重叠窗口寻找静音点进行分割。

    Args:
        audio_data: Audio array to split. Can be 1D (mono) or multi-dimensional.
                   Splits along the last dimension (time axis).
                   要分割的音频数组。可以是一维（单声道）或多维。沿最后一个维度（时间轴）分割。
        sample_rate: Sample rate of the audio in Hz.
                    音频的采样率（赫兹）。
        max_clip_duration_s: Maximum duration of each chunk in seconds.
                            每个片段的最大时长（秒）。
        overlap_duration_s: Overlap duration in seconds between consecutive chunks.
                           Used to search for optimal split points.
                           连续片段之间的重叠时长（秒），用于搜索最优分割点。
        min_energy_window_size: Window size in samples for finding low-energy regions.
                               用于查找低能量区域的窗口大小（采样点数）。

    Returns:
        List of audio chunks. Each chunk is a numpy array with the same shape
        as the input except for the last (time) dimension.
        音频片段列表。每个片段是与输入形状相同的numpy数组（最后的时间维度除外）。

    Example:
        >>> audio = np.random.randn(1040000)  # 65 seconds at 16kHz  # 16kHz下65秒的音频
        >>> chunks = split_audio(
        ...     audio_data=audio,
        ...     sample_rate=16000,
        ...     max_clip_duration_s=30.0,
        ...     overlap_duration_s=1.0,
        ...     min_energy_window_size=1600,
        ... )
        >>> len(chunks)
        3
    """
    chunk_size = int(sample_rate * max_clip_duration_s)  # 计算每个片段的采样点数
    overlap_size = int(sample_rate * overlap_duration_s)  # 计算重叠区域的采样点数
    chunks = []  # 初始化片段列表
    i = 0  # 初始化当前位置索引

    while i < audio_data.shape[-1]:  # 当当前位置未到达音频末尾
        if i + chunk_size >= audio_data.shape[-1]:  # 如果当前位置加片段大小超过音频长度
            # Handle last chunk - take everything remaining  # 处理最后一个片段 - 取所有剩余数据
            chunks.append(audio_data[..., i:])  # 添加剩余所有数据
            break  # 退出循环

        # Find the best split point in the overlap region  # 在重叠区域内寻找最佳分割点
        search_start = i + chunk_size - overlap_size  # 搜索起始位置
        search_end = min(i + chunk_size, audio_data.shape[-1])  # 搜索结束位置
        split_point = find_split_point(  # 查找最佳分割点
            audio_data, search_start, search_end, min_energy_window_size
        )

        # Extract chunk up to the split point  # 提取到分割点为止的片段
        chunks.append(audio_data[..., i:split_point])  # 添加当前片段
        i = split_point  # 更新当前位置

    return chunks  # 返回所有片段


def find_split_point(
    wav: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window: int,
) -> int:
    """Find the best point to split audio by looking for silence or low amplitude.
    通过查找静音或低振幅区域来找到最佳音频分割点。

    Searches for the quietest region within a specified range by calculating
    RMS energy in sliding windows.
    通过在滑动窗口中计算RMS能量，在指定范围内搜索最安静的区域。

    Args:
        wav: Audio array. Can be 1D or multi-dimensional.
            音频数组。可以是一维或多维。
        start_idx: Start index of search region (inclusive).
                  搜索区域的起始索引（包含）。
        end_idx: End index of search region (exclusive).
                搜索区域的结束索引（不包含）。
        min_energy_window: Window size in samples for energy calculation.
                          用于能量计算的窗口大小（采样点数）。

    Returns:
        Index of the quietest point within the search region. This is the
        recommended split point to minimize audio artifacts.
        搜索区域内最安静点的索引。这是为最小化音频伪影而推荐的分割点。

    Example:
        >>> audio = np.random.randn(32000)
        >>> # Insert quiet region  # 插入静音区域
        >>> audio[16000:17600] = 0.01
        >>> split_idx = find_split_point(
        ...     wav=audio,
        ...     start_idx=0,
        ...     end_idx=32000,
        ...     min_energy_window=1600,
        ... )
        >>> 16000 <= split_idx <= 17600
        True
    """
    segment = wav[start_idx:end_idx]  # 提取搜索区域的音频片段

    # Calculate RMS energy in small windows  # 在小窗口中计算RMS能量
    min_energy = math.inf  # 初始化最小能量为无穷大
    quietest_idx = 0  # 初始化最安静点索引

    for i in range(0, len(segment) - min_energy_window, min_energy_window):  # 遍历每个窗口
        window = segment[i : i + min_energy_window]  # 提取当前窗口
        energy = (window**2).mean() ** 0.5  # 计算RMS能量
        if energy < min_energy:  # 如果当前能量小于最小能量
            quietest_idx = i + start_idx  # 更新最安静点索引
            min_energy = energy  # 更新最小能量

    return quietest_idx  # 返回最安静点索引
