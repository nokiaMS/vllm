# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools  # 导入函数工具模块
import hashlib  # 导入哈希算法模块
import pickle  # 导入序列化模块
import uuid  # 导入UUID模块
from collections.abc import Callable, Iterable  # 导入可调用对象和可迭代对象抽象基类

import numpy as np  # 导入NumPy数值计算库
import torch  # 导入PyTorch深度学习框架
from PIL import Image  # 导入PIL图像处理库

import vllm.envs as envs  # 导入vLLM环境配置
from vllm.logger import init_logger  # 导入日志初始化函数

from .media import MediaWithBytes  # 从media模块导入带字节的媒体包装类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


@functools.lru_cache(maxsize=3)  # 使用LRU缓存装饰器，最多缓存3个结果
def _get_hasher_factory(algorithm: str) -> Callable[[], "hashlib._Hash"]:
    """
    Get the hasher factory based on the configured algorithm.
    根据配置的算法获取哈希工厂。

    Args:
        algorithm: Hash algorithm name (blake3, sha256, or sha512)
                  哈希算法名称（blake3、sha256或sha512）

    Returns a callable that creates a new hasher instance.
    Supports blake3 (default), sha256, and sha512 for FIPS compliance.
    返回一个创建新哈希实例的可调用对象。
    支持blake3（默认）、sha256和sha512以满足FIPS合规性。

    See: https://github.com/vllm-project/vllm/issues/18334
    """
    algorithm = algorithm.lower()  # 将算法名称转为小写

    if algorithm == "blake3":  # 如果使用blake3算法
        from blake3 import blake3  # 导入blake3哈希函数

        return blake3  # 返回blake3工厂函数
    elif algorithm == "sha256":  # 如果使用sha256算法
        return hashlib.sha256  # 返回sha256工厂函数
    elif algorithm == "sha512":  # 如果使用sha512算法
        return hashlib.sha512  # 返回sha512工厂函数
    else:  # 不支持的算法
        # This should never happen due to env_with_choices validation  # 由于环境变量选择验证，这不应该发生
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")  # 抛出不支持的哈希算法错误


class MultiModalHasher:
    """多模态数据哈希器，用于计算多模态数据的哈希值。"""

    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]:
        """将单个对象序列化为字节序列，用于哈希计算。"""
        # Simple cases  # 简单情况
        if isinstance(obj, (bytes, memoryview)):  # 如果已经是字节或内存视图
            return (obj,)  # 直接返回
        if isinstance(obj, str):  # 如果是字符串
            return (obj.encode("utf-8"),)  # 编码为UTF-8字节
        if isinstance(obj, (int, float)):  # 如果是整数或浮点数
            return (np.array(obj).tobytes(),)  # 转换为numpy数组再转字节

        if isinstance(obj, Image.Image):  # 如果是PIL图像
            exif = obj.getexif()  # 获取EXIF信息
            if Image.ExifTags.Base.ImageID in exif and isinstance(  # 如果EXIF中包含ImageID且为UUID
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return (exif[Image.ExifTags.Base.ImageID].bytes,)  # 使用UUID的字节作为哈希输入

            data = {"mode": obj.mode, "data": np.asarray(obj)}  # 提取图像模式和像素数据
            palette = obj.palette  # 获取调色板
            if palette is not None:  # 如果有调色板
                data["palette"] = palette.palette  # 添加调色板数据
                if palette.rawmode is not None:  # 如果有原始模式
                    data["palette_rawmode"] = palette.rawmode  # 添加调色板原始模式

            return cls.iter_item_to_bytes("image", data)  # 递归序列化图像数据

        if isinstance(obj, MediaWithBytes) and isinstance(obj.media, Image.Image):  # 如果是带字节的图像媒体
            exif = obj.media.getexif()  # 获取EXIF信息
            if Image.ExifTags.Base.ImageID in exif and isinstance(  # 如果EXIF中包含UUID类型的ImageID
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return (exif[Image.ExifTags.Base.ImageID].bytes,)  # 使用UUID字节

            return cls.iter_item_to_bytes("image", obj.original_bytes)  # 使用原始字节进行序列化

        if isinstance(obj, torch.Tensor):  # 如果是PyTorch张量
            tensor_obj: torch.Tensor = obj.cpu()  # 将张量移到CPU
            tensor_dtype = tensor_obj.dtype  # 获取张量数据类型
            tensor_shape = tensor_obj.shape  # 获取张量形状

            # NumPy does not support bfloat16.  # NumPy不支持bfloat16
            # Workaround: View the tensor as a contiguous 1D array of bytes  # 解决方案：将张量视为连续的一维字节数组
            if tensor_dtype == torch.bfloat16:  # 如果是bfloat16类型
                tensor_obj = tensor_obj.contiguous()  # 确保内存连续
                tensor_obj = tensor_obj.view((tensor_obj.numel(),)).view(torch.uint8)  # 转为uint8视图

                return cls.iter_item_to_bytes(  # 递归序列化bfloat16张量
                    "tensor",
                    {
                        "original_dtype": str(tensor_dtype),  # 保存原始数据类型
                        "original_shape": tuple(tensor_shape),  # 保存原始形状
                        "data": tensor_obj.numpy(),  # 转为numpy数组
                    },
                )

            return cls.iter_item_to_bytes("tensor", tensor_obj.numpy())  # 将张量转为numpy后序列化

        if isinstance(obj, np.ndarray):  # 如果是numpy数组
            if obj.ndim == 0:  # 如果是零维数组
                arr_data = obj.item()  # 提取标量值
            elif obj.flags.c_contiguous:  # 如果是C连续的
                # Not valid for 0-D arrays  # 对零维数组无效
                arr_data = obj.view(np.uint8).data  # 以uint8视图获取数据
            else:  # 如果是非连续数组
                # If the array is non-contiguous, we need to copy it first  # 如果非连续，需要先拷贝
                arr_data = obj.tobytes()  # 转为字节

            return cls.iter_item_to_bytes(  # 递归序列化numpy数组
                "ndarray",
                {
                    "dtype": obj.dtype.str,  # 数据类型字符串
                    "shape": obj.shape,  # 数组形状
                    "data": arr_data,  # 数组数据
                },
            )

        logger.warning(  # 记录警告日志
            "No serialization method found for %s. Falling back to pickle.", type(obj)
        )

        return (pickle.dumps(obj),)  # 回退到pickle序列化

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[bytes | memoryview]:
        """递归地将对象转换为键值对形式的字节序列。"""
        if obj is None:  # 如果对象为None
            yield key.encode("utf-8")  # 仅输出键
            return
        # Recursive cases  # 递归情况
        if isinstance(obj, (list, tuple)):  # 如果是列表或元组
            for i, elem in enumerate(obj):  # 遍历每个元素
                yield from cls.iter_item_to_bytes(f"{key}.{i}", elem)  # 递归处理每个元素
        elif isinstance(obj, dict):  # 如果是字典
            for k, v in obj.items():  # 遍历每个键值对
                yield from cls.iter_item_to_bytes(f"{key}.{k}", v)  # 递归处理每个值
        else:  # 基本类型
            yield key.encode("utf-8")  # 输出键
            yield from cls.serialize_item(obj)  # 序列化值

    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str:
        """计算关键字参数的哈希值，返回十六进制哈希字符串。"""
        hasher_factory = _get_hasher_factory(envs.VLLM_MM_HASHER_ALGORITHM)  # 获取哈希工厂
        hasher = hasher_factory()  # 创建哈希器实例

        for k, v in sorted(kwargs.items(), key=lambda kv: kv[0]):  # 按键排序遍历参数
            for bytes_ in cls.iter_item_to_bytes(k, v):  # 序列化每个参数
                hasher.update(bytes_)  # 更新哈希器

        return hasher.hexdigest()  # 返回十六进制哈希值
