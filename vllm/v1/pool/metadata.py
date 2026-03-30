# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass  # 导入dataclass装饰器，用于简化数据类定义

import numpy as np  # 导入NumPy库，用于数组操作
import torch  # 导入PyTorch库

from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.tasks import PoolingTask  # 导入池化任务类型
from vllm.utils.platform_utils import is_pin_memory_available  # 导入检查锁页内存是否可用的工具函数

pin_memory = is_pin_memory_available()  # 检测当前平台是否支持锁页内存


@dataclass
class PoolingCursor:  # 池化游标类，跟踪池化操作中每个序列的位置和状态信息
    index: list[int]  # 序列索引列表
    first_token_indices_gpu: torch.Tensor  # GPU上的首token索引张量
    last_token_indices_gpu: torch.Tensor  # GPU上的末token索引张量
    prompt_lens_cpu: torch.Tensor  # CPU上的提示长度张量
    seq_lens_cpu: torch.Tensor  # CPU上的序列长度张量
    num_scheduled_tokens_cpu: torch.Tensor  # CPU上的已调度token数量张量

    def __getitem__(self, indices: slice):  # 切片操作，返回指定索引范围的子PoolingCursor
        return PoolingCursor(  # 创建新的PoolingCursor实例
            index=self.index[indices],  # 切片序列索引
            first_token_indices_gpu=self.first_token_indices_gpu[indices],  # 切片首token索引
            last_token_indices_gpu=self.last_token_indices_gpu[indices],  # 切片末token索引
            prompt_lens_cpu=self.prompt_lens_cpu[indices],  # 切片提示长度
            seq_lens_cpu=self.seq_lens_cpu[indices],  # 切片序列长度
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],  # 切片已调度token数量
        )

    def is_partial_prefill(self):  # 判断是否存在部分预填充（即某些序列尚未完全调度所有提示token）
        return not torch.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)  # 如果提示长度不全等于已调度token数量，则为部分预填充

    def is_finished(self):  # 判断序列是否已完成（提示长度等于序列长度）
        return self.prompt_lens_cpu == self.seq_lens_cpu  # 返回布尔张量，标记哪些序列已完成


class PoolingStates:  # 池化状态类，用于在分块预填充时缓存隐藏状态
    def __init__(self):  # 初始化池化状态
        # for chunked prefill with ALL pooling
        self.hidden_states_cache: list[torch.Tensor] = []  # 隐藏状态缓存列表，存储中间结果

    def clean(self):  # 清理方法，清空隐藏状态缓存
        self.hidden_states_cache.clear()  # 清空缓存列表


@dataclass
class PoolingMetadata:  # 池化元数据类，包含池化操作所需的所有张量和参数
    """Tensors for pooling."""

    prompt_lens: torch.Tensor  # CPU Tensor  # 提示长度张量（存储在CPU上）
    prompt_token_ids: torch.Tensor | None  # 提示token ID张量，可为None
    pooling_params: list[PoolingParams]  # 池化参数列表
    pooling_states: list[PoolingStates]  # 池化状态列表
    pooling_cursor: PoolingCursor | None = None  # 池化游标，默认为None

    def __post_init__(self) -> None:  # 数据类初始化后的后处理，提取并验证池化任务
        pooling_params = self.pooling_params  # 获取池化参数列表

        tasks: list[PoolingTask] = [  # 从池化参数中提取所有非空的任务
            task  # 任务对象
            for pooling_param in pooling_params  # 遍历每个池化参数
            if (task := pooling_param.task) is not None  # 使用海象运算符提取非空任务
        ]
        assert len(pooling_params) == len(tasks)  # 断言所有池化参数都有对应的任务

        self.tasks = tasks  # 存储提取的任务列表

    def __getitem__(self, indices: slice):  # 切片操作，返回指定索引范围的子PoolingMetadata
        return PoolingMetadata(  # 创建新的PoolingMetadata实例
            prompt_lens=self.prompt_lens[indices],  # 切片提示长度
            prompt_token_ids=None  # 如果原始prompt_token_ids为None则保持None
            if self.prompt_token_ids is None  # 判断是否为None
            else self.prompt_token_ids[indices],  # 否则进行切片
            pooling_params=self.pooling_params[indices],  # 切片池化参数
            pooling_states=self.pooling_states[indices],  # 切片池化状态
            pooling_cursor=None  # 如果原始pooling_cursor为None则保持None
            if self.pooling_cursor is None  # 判断是否为None
            else self.pooling_cursor[indices],  # 否则进行切片
        )

    def get_prompt_token_ids(self) -> list[torch.Tensor]:  # 获取每个提示的token ID列表，按实际长度截取
        prompt_token_ids = self.prompt_token_ids  # 获取prompt_token_ids张量
        assert prompt_token_ids is not None, (  # 断言token ID不为空
            "Please set `requires_token_ids=True` in `get_pooling_updates`"  # 提示用户需要设置requires_token_ids参数
        )

        return [prompt_token_ids[i, :num] for i, num in enumerate(self.prompt_lens)]  # 返回按实际长度截取的token ID列表

    def get_pooling_cursor(self) -> PoolingCursor:  # 获取池化游标，必须先调用build_pooling_cursor构建
        pooling_cursor = self.pooling_cursor  # 获取池化游标
        assert pooling_cursor is not None, "Should call `build_pooling_cursor` first"  # 断言游标已构建

        return pooling_cursor  # 返回池化游标

    def build_pooling_cursor(  # 构建池化游标，计算每个序列在拼接隐藏状态中的位置索引
        self,
        num_scheduled_tokens_np: np.ndarray,  # 已调度token数量的NumPy数组
        seq_lens_cpu: torch.Tensor,  # CPU上的序列长度张量
        device: torch.device,  # 目标计算设备
    ):
        n_seq = len(num_scheduled_tokens_np)  # 获取序列数量
        prompt_lens = self.prompt_lens  # 获取提示长度张量

        assert len(prompt_lens) == n_seq  # 断言提示长度数量与序列数量一致

        index = list(range(n_seq))  # 生成序列索引列表
        num_scheduled_tokens_cpu = torch.from_numpy(num_scheduled_tokens_np)  # 将NumPy数组转换为PyTorch张量
        cumsum = torch.zeros(  # 创建累积和张量，长度为序列数+1
            n_seq + 1, dtype=torch.int64, pin_memory=pin_memory, device="cpu"  # 使用int64类型，在CPU上创建，可选锁页内存
        )
        torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])  # 计算已调度token数量的累积和
        cumsum = cumsum.to(device, non_blocking=True)  # 异步将累积和张量传输到目标设备
        self.pooling_cursor = PoolingCursor(  # 创建池化游标实例
            index=index,  # 序列索引
            first_token_indices_gpu=cumsum[:n_seq],  # 每个序列的首token在拼接张量中的索引
            last_token_indices_gpu=cumsum[1:] - 1,  # 每个序列的末token在拼接张量中的索引
            prompt_lens_cpu=prompt_lens,  # 提示长度
            seq_lens_cpu=seq_lens_cpu,  # 序列长度
            num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,  # 已调度token数量
        )
