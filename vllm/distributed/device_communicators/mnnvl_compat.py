# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any  # 导入Any类型注解

import torch.distributed as dist  # 导入PyTorch分布式通信模块
from flashinfer.comm.mnnvl import CommBackend as CommBackend  # 从flashinfer导入MNNVL通信后端基类

from vllm.utils.flashinfer import has_flashinfer_all2all  # 导入检测flashinfer all2all是否可用的函数

assert has_flashinfer_all2all(), "Flashinfer alltoallv module cannot be found"  # 断言flashinfer all2all模块可用


class CustomCommunicator(CommBackend):
    """自定义通信器，实现FlashInfer MNNVL兼容的通信后端接口。

    继承CommBackend抽象类，使用PyTorch分布式进程组实现allgather操作。"""

    def __init__(self, group):
        """初始化自定义通信器。

        Args:
            group: PyTorch分布式进程组
        """
        self._group = group  # 存储进程组引用

    def Get_rank(self) -> int:
        """获取当前进程在组中的排名。"""
        return self._group.rank()  # 返回当前进程的排名

    def Get_size(self) -> int:
        """获取进程组的大小。"""
        return self._group.size()  # 返回进程组中的进程数

    def allgather(self, data: int):
        """在组内收集所有进程的数据。

        Args:
            data: 当前进程的数据

        Returns:
            包含所有进程数据的列表
        """
        gathered = [None] * self.Get_size()  # 创建用于存储收集结果的列表
        dist.all_gather_object(gathered, data, group=self._group)  # 执行对象级别的全收集操作
        return gathered  # 返回收集结果

    # NOTE(rob): CommBackend is an abstract class, and bcast/barrier
    # are unimplemented on vLLM side. If we need to utilize these
    # methods in the future, can create a concrete implementation.
    def bcast(self, data: Any, root: int) -> Any:
        """广播操作（未实现）。

        Args:
            data: 要广播的数据
            root: 广播源进程的排名
        """
        raise NotImplementedError  # 抛出未实现异常

    def barrier(self) -> None:
        """屏障同步操作（未实现）。"""
        raise NotImplementedError  # 抛出未实现异常

    def Split(self, color: int, key: int) -> "CustomCommunicator":
        """分裂通信器（返回自身，未实际实现分裂逻辑）。

        Args:
            color: 分裂颜色标识
            key: 分裂排序键
        """
        return self  # 返回自身，不执行实际分裂
