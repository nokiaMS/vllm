# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者
from abc import ABC  # 导入抽象基类模块，用于定义加载/存储规格的抽象接口

import numpy as np  # 导入 NumPy 库，用于高效的数组操作

from vllm.v1.kv_offload.abstract import LoadStoreSpec  # 导入加载/存储规格抽象基类


# 基于块 ID 的加载/存储规格基类
# 定义了通过块编号来加载/存储 KV 缓存块的通用接口
class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """基于块 ID 的加载/存储规格抽象基类。

    定义了通过给定的块编号来加载或存储 KV 缓存块的通用接口。
    """

    def __init__(self, block_ids: list[int]):  # 构造函数，接受块 ID 列表
        """初始化块 ID 加载/存储规格。

        Args:
            block_ids: 需要加载或存储的 KV 缓存块的 ID 列表
        """
        self.block_ids = np.array(block_ids, dtype=np.int64)  # 将块 ID 列表转换为 NumPy 的 int64 数组，提高访问效率

    def __repr__(self) -> str:  # 定义对象的字符串表示方法
        """返回块 ID 数组的字符串表示。"""
        return repr(self.block_ids)  # 返回底层 NumPy 数组的字符串表示


# GPU 加载/存储规格类，表示将 KV 缓存块加载到或存储自 GPU 显存
class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """GPU 加载/存储规格，用于将 KV 缓存块与 GPU 显存之间进行传输。"""

    @staticmethod  # 静态方法装饰器
    def medium() -> str:  # 获取存储介质名称的静态方法
        """返回存储介质标识。"""
        return "GPU"  # 返回 GPU 存储介质标识字符串


# CPU 加载/存储规格类，表示将 KV 缓存块加载到或存储自 CPU 内存
class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """CPU 加载/存储规格，用于将 KV 缓存块与 CPU 内存之间进行传输。"""

    @staticmethod  # 静态方法装饰器
    def medium() -> str:  # 获取存储介质名称的静态方法
        """返回存储介质标识。"""
        return "CPU"  # 返回 CPU 存储介质标识字符串
