# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明


from vllm.config.utils import config  # 导入config装饰器，用于创建pydantic数据类


@config  # 使用config装饰器创建pydantic数据类
class SpeechToTextConfig:
    """语音转文本模型的配置类。"""

    sample_rate: float = 16_000  # 输入音频的采样率（Hz），大多数语音模型期望16kHz
    """Sample rate (Hz) to resample input audio to. Most speech models expect
    16kHz audio input. The input audio will be automatically resampled to this
    rate before processing."""

    max_audio_clip_s: int | None = 30  # 单个音频片段的最大时长（秒），超过则分块处理
    """Maximum duration in seconds for a single audio clip without chunking.
    Audio longer than this will be split into smaller chunks if
    `allow_audio_chunking` evaluates to True, otherwise it will be rejected.
    `None` means audio duration can be unlimited and won't be chunked."""

    overlap_chunk_second: int = 1  # 连续音频块之间的重叠时长（秒），用于保持上下文连续性
    """Overlap duration in seconds between consecutive audio chunks when
    splitting long audio. This helps maintain context across chunk boundaries
    and improves transcription quality at split points."""

    min_energy_split_window_size: int | None = 1600  # 用于寻找低能量（安静）区域的窗口大小（采样点数）
    """Window size in samples for finding low-energy (quiet) regions to split
    audio chunks. The algorithm looks for the quietest moment within this
    window to minimize cutting through speech. Default 1600 samples ≈ 100ms
    at 16kHz. If None, no chunking will be done."""

    @property  # 属性装饰器，定义只读属性
    def allow_audio_chunking(self) -> bool:
        """判断是否允许音频分块处理。"""
        return (  # 仅当分块窗口大小和最大音频时长都不为None时允许分块
            self.min_energy_split_window_size is not None
            and self.max_audio_clip_s is not None
        )
