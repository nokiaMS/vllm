# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


# [中文注释] 生成随机尺寸的RGB测试图像
def random_image(rng: np.random.RandomState, min_wh: int, max_wh: int):
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    arr = rng.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# [中文注释] 生成随机帧数和尺寸的测试视频数组
def random_video(
    rng: np.random.RandomState,
    min_frames: int,
    max_frames: int,
    min_wh: int,
    max_wh: int,
):
    num_frames = rng.randint(min_frames, max_frames)
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    return rng.randint(0, 255, size=(num_frames, w, h, 3), dtype=np.uint8)


# [中文注释] 生成随机长度的测试音频数据
def random_audio(
    rng: np.random.RandomState,
    min_len: int,
    max_len: int,
    sr: int,
):
    audio_len = rng.randint(min_len, max_len)
    return rng.rand(audio_len), sr


# [中文注释] 从静态图像创建测试视频文件：支持彩色/灰度和自定义编码格式
def create_video_from_image(
    image_path: str,
    video_path: str,
    num_frames: int = 10,
    fps: float = 1.0,
    is_color: bool = True,
    fourcc: str = "mp4v",
):
    image = cv2.imread(image_path)
    if not is_color:
        # Convert to grayscale if is_color is False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
    else:
        height, width, _ = image.shape

    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
        isColor=is_color,
    )

    for _ in range(num_frames):
        video_writer.write(image)

    video_writer.release()
    return video_path


# [中文注释] 计算两个向量的余弦相似度
def cosine_similarity(A: npt.NDArray, B: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """Compute cosine similarity between two vectors."""
    return np.sum(A * B, axis=axis) / (
        np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    )


# [中文注释] 将图像像素值归一化到[0, 1]范围
def normalize_image(image: npt.NDArray) -> npt.NDArray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0
