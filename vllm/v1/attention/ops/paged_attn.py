# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者


import torch  # 导入 PyTorch 库，用于张量操作

from vllm.platforms import current_platform  # 导入当前平台检测工具

if current_platform.is_cuda_alike():  # 如果当前平台是 CUDA 或类 CUDA（如 ROCm）
    from vllm import _custom_ops as ops  # 导入 vLLM 自定义 CUDA 操作库
elif current_platform.is_xpu():  # 如果当前平台是 Intel XPU
    from vllm._xpu_ops import xpu_ops as ops  # type: ignore[no-redef]  # 导入 XPU 专用操作库


# 分页注意力工具类，提供 KV 缓存的分页管理操作
# 包括 KV 缓存的分割和写入操作，是分页注意力机制的底层基础设施
class PagedAttention:
    """分页注意力工具类。

    提供 KV 缓存的分页管理静态方法，包括将 KV 缓存张量分割为
    独立的 key 和 value 缓存，以及将新的 key/value 写入分页缓存中。
    """

    @staticmethod  # 静态方法装饰器
    def split_kv_cache(  # 分割 KV 缓存张量的静态方法
        kv_cache: torch.Tensor,  # 合并的 KV 缓存张量
        num_kv_heads: int,  # KV 注意力头的数量
        head_size: int,  # 每个注意力头的维度大小
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回分离的 key 缓存和 value 缓存元组
        """将合并的 KV 缓存张量分割为独立的 key 和 value 缓存视图。

        Args:
            kv_cache: 合并的 KV 缓存张量，shape 为 [2, num_blocks, ...]
            num_kv_heads: KV 注意力头数
            head_size: 每个注意力头的维度大小

        Returns:
            分离的 (key_cache, value_cache) 张量元组
        """
        x = 16 // kv_cache.element_size()  # 计算向量化因子：16 字节除以每个元素的字节数，用于内存对齐优化
        num_blocks = kv_cache.shape[1]  # 获取缓存块的数量

        key_cache = kv_cache[0]  # 提取第一个维度作为 key 缓存
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)  # 重塑 key 缓存形状以适配向量化访问模式
        value_cache = kv_cache[1]  # 提取第二个维度作为 value 缓存
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)  # 重塑 value 缓存形状
        return key_cache, value_cache  # 返回分离的 key 和 value 缓存

    @staticmethod  # 静态方法装饰器
    def write_to_paged_cache(  # 将 key/value 写入分页缓存的静态方法
        key: torch.Tensor,  # 待写入的 key 张量
        value: torch.Tensor,  # 待写入的 value 张量
        key_cache: torch.Tensor,  # key 缓存目标张量
        value_cache: torch.Tensor,  # value 缓存目标张量
        slot_mapping: torch.Tensor,  # 槽位映射张量，指定每个 token 写入的缓存位置
        kv_cache_dtype: str,  # KV 缓存的数据类型字符串（如 "auto"、"fp8"）
        k_scale: torch.Tensor,  # key 的量化缩放因子
        v_scale: torch.Tensor,  # value 的量化缩放因子
    ) -> None:  # 无返回值，直接写入缓存
        """将 key 和 value 张量写入分页 KV 缓存。

        Args:
            key: 待写入的 key 张量
            value: 待写入的 value 张量
            key_cache: key 的分页缓存
            value_cache: value 的分页缓存
            slot_mapping: 槽位映射，指定写入位置
            kv_cache_dtype: KV 缓存数据类型
            k_scale: key 量化缩放因子
            v_scale: value 量化缩放因子
        """
        ops.reshape_and_cache(  # 调用底层自定义操作将 key/value 重塑并写入缓存
            key,  # 传入 key 张量
            value,  # 传入 value 张量
            key_cache,  # 传入 key 缓存目标
            value_cache,  # 传入 value 缓存目标
            slot_mapping.flatten(),  # 将槽位映射展平为一维张量
            kv_cache_dtype,  # 传入缓存数据类型
            k_scale,  # 传入 key 缩放因子
            v_scale,  # 传入 value 缩放因子
        )
