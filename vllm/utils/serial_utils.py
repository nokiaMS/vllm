# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pybase64
import torch

sys_byteorder = sys.byteorder


# 数据类型信息的不可变数据类，封装了 PyTorch 和 NumPy 之间的类型映射关系，
# 用于张量序列化/反序列化时的类型转换
@dataclass(frozen=True)
class DTypeInfo:
    torch_dtype: torch.dtype

    torch_view_dtype: torch.dtype
    numpy_view_dtype: npt.DTypeLike

    @property
    def nbytes(self) -> int:
        return self.torch_dtype.itemsize


# 嵌入向量支持的数据类型、字节序和编码格式的类型定义
EmbedDType = Literal["float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"]
Endianness = Literal["native", "big", "little"]
EncodingFormat = Literal["float", "base64", "bytes", "bytes_only"]

# 嵌入数据类型到 DTypeInfo 的映射表，包含 PyTorch 和 NumPy 的类型对应关系；
# 注意：NumPy 不支持 bfloat16 和 fp8，因此使用 float16/uint8 作为视图类型
# I'm not sure if other platforms' CPUs support the fp8 data format.
# EMBED_DTYPE only uses the fp8 data representation,
# does not use fp8 computation, and only occurs on the CPU.
# Apologize for any possible break.
# NOTE: numpy does not support bfloat16 and fp8
EMBED_DTYPES: Mapping[EmbedDType, DTypeInfo] = {
    "float32": DTypeInfo(torch.float32, torch.float32, np.float32),
    "float16": DTypeInfo(torch.float16, torch.float16, np.float16),
    "bfloat16": DTypeInfo(torch.bfloat16, torch.float16, np.float16),
    "fp8_e4m3": DTypeInfo(torch.float8_e4m3fn, torch.uint8, np.uint8),
    "fp8_e5m2": DTypeInfo(torch.float8_e5m2, torch.uint8, np.uint8),
}
ENDIANNESS: tuple[Endianness, ...] = get_args(Endianness)


# 将 PyTorch 张量序列化为 base64 编码字符串，通过 BytesIO 缓冲区中转
def tensor2base64(x: torch.Tensor) -> str:
    with io.BytesIO() as buf:
        torch.save(x, buf)
        buf.seek(0)
        binary_data = buf.read()

    return pybase64.b64encode(binary_data).decode("utf-8")


# 将张量转换为指定数据类型和字节序的原始二进制字节，
# 核心流程：类型转换 -> 展平 -> 视图转换 -> 转 NumPy -> 字节序处理 -> 输出字节
def tensor2binary(
    tensor: torch.Tensor,
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> bytes:
    assert isinstance(tensor, torch.Tensor)
    assert embed_dtype in EMBED_DTYPES
    assert endianness in ENDIANNESS

    dtype_info = EMBED_DTYPES[embed_dtype]

    np_array = (
        tensor.to(dtype_info.torch_dtype)
        .flatten()
        .contiguous()
        .view(dtype_info.torch_view_dtype)
        .numpy()
    )

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return np_array.tobytes()


# 将原始二进制字节反序列化为 PyTorch 张量，是 tensor2binary 的逆操作；
# 核心流程：字节 -> NumPy 数组 -> 字节序处理 -> 转 PyTorch 张量 -> 视图还原
def binary2tensor(
    binary: bytes,
    shape: tuple[int, ...],
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> torch.Tensor:
    assert embed_dtype in EMBED_DTYPES
    assert endianness in ENDIANNESS

    dtype_info = EMBED_DTYPES[embed_dtype]

    np_array = np.frombuffer(binary, dtype=dtype_info.numpy_view_dtype).reshape(shape)

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return torch.from_numpy(np_array).view(dtype_info.torch_dtype)
