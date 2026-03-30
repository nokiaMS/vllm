# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections.abc import Iterator  # 导入迭代器类型
from typing import TYPE_CHECKING  # 导入类型检查标志

import torch  # 导入 PyTorch 框架

from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.v1.attention.backend import AttentionBackend  # 导入注意力后端类
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager  # 导入加载存储规格和卸载管理器
from vllm.v1.kv_offload.worker.worker import OffloadingHandler  # 导入卸载处理器

if TYPE_CHECKING:  # 仅在类型检查时导入以下模块
    from vllm.config import VllmConfig  # vLLM 配置类
    from vllm.v1.kv_cache_interface import KVCacheConfig  # KV 缓存配置类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class OffloadingSpec(ABC):
    """Spec for an offloading connector
    卸载连接器的规格抽象基类，定义了卸载连接器需要实现的接口。"""

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):  # 初始化卸载规格
        """初始化卸载规格，设置基础配置参数。"""
        logger.warning(  # 记录警告信息
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.vllm_config = vllm_config  # 存储 vLLM 配置
        self.kv_cache_config = kv_cache_config  # 存储 KV 缓存配置

        kv_transfer_config = vllm_config.kv_transfer_config  # 获取 KV 传输配置
        assert kv_transfer_config is not None  # 断言 KV 传输配置不为空
        self.extra_config = kv_transfer_config.kv_connector_extra_config  # 获取连接器额外配置

        # block size used by vLLM for hashing request tokens for the sake
        # of enabling prefix caching
        self.hash_block_size = vllm_config.cache_config.block_size  # vLLM 用于哈希请求令牌的块大小，用于启用前缀缓存
        # gpu block size per group
        self.gpu_block_size: tuple[int, ...] = tuple(  # 每个 KV 缓存组的 GPU 块大小
            kv_cache_group.kv_cache_spec.block_size  # 获取每个缓存组的块大小
            for kv_cache_group in kv_cache_config.kv_cache_groups  # 遍历所有 KV 缓存组
        )

        for block_size in self.gpu_block_size:  # 遍历每个 GPU 块大小
            assert block_size % self.hash_block_size == 0  # 断言 GPU 块大小是哈希块大小的整数倍

        # offloaded_block_size / gpu_block_size
        self.block_size_factor: int = 1  # 卸载块大小与 GPU 块大小的比率，默认为 1

        offloaded_block_size = self.extra_config.get("block_size")  # 从额外配置中获取卸载块大小
        if offloaded_block_size is not None:  # 如果指定了卸载块大小
            offloaded_block_size_int = int(offloaded_block_size)  # 转换为整数
            gpu_block_sizes = set(self.gpu_block_size)  # 获取所有不同的 GPU 块大小
            assert len(gpu_block_sizes) == 1, (  # 断言所有 KV 缓存组的块大小必须相同
                "If 'block_size' is specified in kv_connector_extra_config, "
                "there must be at least one KV cache group, "
                "and all groups must have the same block size."
            )
            gpu_block_size = gpu_block_sizes.pop()  # 获取唯一的 GPU 块大小

            assert offloaded_block_size_int % gpu_block_size == 0  # 断言卸载块大小是 GPU 块大小的整数倍
            self.block_size_factor = offloaded_block_size_int // gpu_block_size  # 计算块大小比率

    @abstractmethod  # 抽象方法装饰器
    def get_manager(self) -> OffloadingManager:  # 获取卸载管理器
        """
        Get an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.
        获取一个 OffloadingManager，供调度器侧的卸载连接器用于
        跟踪已卸载的块并管理驱逐策略。
        """
        pass  # 子类必须实现此方法

    @abstractmethod  # 抽象方法装饰器
    def get_handlers(  # 获取卸载处理器及其对应的源和目标类型
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        Get offloading handlers along with their respective src and dst types.
        获取卸载处理器及其对应的源类型和目标类型。

        Args:
            kv_caches: A dictionary of layer_name -> gpu_kv_cache tensor.
            kv_caches: 层名称 -> GPU KV 缓存张量的字典。
            attn_backends: A dictionary of layer_name -> AttentionBackend.
            attn_backends: 层名称 -> 注意力后端的字典。

        Yields:
            Tuples of (src_type, dst_type, offloading_handler).
            生成 (源类型, 目标类型, 卸载处理器) 的元组。
        """
        pass  # 子类必须实现此方法
