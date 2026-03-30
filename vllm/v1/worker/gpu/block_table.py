# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable  # 导入可迭代对象抽象基类

import torch  # 导入 PyTorch 深度学习框架

from vllm.triton_utils import tl, triton  # 导入 Triton JIT 编译工具
from vllm.utils.math_utils import cdiv  # 导入向上取整除法工具函数
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # 导入填充槽位 ID 常量
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor  # 导入分阶段写入张量和 UVA 支撑张量


# 块表管理器：管理 KV 缓存的分页块表（block table），负责块 ID 的写入、
# 从请求索引到批次索引的 gather 操作、以及 slot mapping 的计算。
# 设计要点：
# - 使用分阶段写入（staged write）机制：先在 CPU 端暂存修改，再批量通过 Triton 内核写入 GPU。
# - 支持多个 KV 缓存组（不同注意力层可能有不同的块大小）。
# - 支持上下文并行（Context Parallelism），通过交错方式在多个 rank 间分片 KV 缓存。
# - input_block_tables 为前向传播使用的持久化张量，确保 CUDA graph 捕获时地址不变。
class BlockTables:
    # 初始化块表管理器，分配 GPU 张量和 UVA 缓冲区
    def __init__(
        self,
        block_sizes: list[int],  # 每个 KV 缓存组的块大小列表
        max_num_reqs: int,  # 最大请求数量
        max_num_batched_tokens: int,  # 最大批处理 token 数量
        max_model_len: int,  # 模型支持的最大序列长度
        device: torch.device,  # 计算设备
        cp_size: int = 1,  # 上下文并行的 rank 数量
        cp_rank: int = 0,  # 当前 rank 的编号
        cp_interleave: int = 1,  # 上下文并行的交错因子
    ):
        self.block_sizes = block_sizes  # 保存块大小列表
        self.max_num_reqs = max_num_reqs  # 保存最大请求数
        self.max_num_batched_tokens = max_num_batched_tokens  # 保存最大批处理 token 数
        self.max_model_len = max_model_len  # 保存最大模型长度
        self.device = device  # 保存计算设备

        self.cp_size = cp_size  # 保存上下文并行大小
        self.cp_rank = cp_rank  # 保存当前 rank 编号
        self.cp_interleave = cp_interleave  # 保存交错因子

        self.num_kv_cache_groups = len(self.block_sizes)  # 计算 KV 缓存组数量
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []  # 初始化块表列表
        for i in range(self.num_kv_cache_groups):  # 遍历每个 KV 缓存组
            block_size = self.block_sizes[i]  # 获取当前组的块大小
            # When using DCP, each request's KV cache is sharded among different ranks.
            # As a result, one block on the current rank covers `block_size * cp_size`
            # tokens in the full, global (unsharded) sequence.
            max_num_blocks = cdiv(self.max_model_len, block_size * self.cp_size)  # 计算每个请求最多需要的块数
            block_table = StagedWriteTensor(  # 创建分阶段写入张量作为块表
                (self.max_num_reqs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_tables.append(block_table)  # 将块表添加到列表
        self.block_table_ptrs = self._make_ptr_tensor(  # 将所有块表的 GPU 数据指针打包为张量
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(  # 将所有块表的行步幅打包为张量
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )

        self.block_sizes_tensor = torch.tensor(  # 将块大小列表转换为 GPU 张量
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.num_blocks = UvaBackedTensor(  # 创建 UVA 支撑张量记录每个请求在每个组中的块数量
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [  # 分配持久化的输入块表，用于前向传播和 CUDA graph
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)  # 将输入块表的数据指针打包为张量

        self.slot_mappings = torch.zeros(  # 分配 slot 映射张量，记录每个 token 对应的 KV 缓存槽位
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    # 将多个张量的数据指针打包为 uint64 张量，供 Triton 内核通过指针间接寻址访问。
    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(  # 将每个张量的数据指针转换为 uint64 并打包为 GPU 张量
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    # 为指定请求追加或覆写新的块 ID，暂存到 staged write 缓冲区中。
    def append_block_ids(
        self,
        req_index: int,  # 请求在状态数组中的索引
        new_block_ids: tuple[list[int], ...],  # 每个 KV 缓存组的新块 ID 列表
        overwrite: bool,  # 是否覆写（True 表示从头开始写，False 表示追加）
    ) -> None:
        for i in range(self.num_kv_cache_groups):  # 遍历每个 KV 缓存组
            start = self.num_blocks.np[i, req_index] if not overwrite else 0  # 确定写入起始位置：追加从当前末尾开始，覆写从 0 开始
            block_ids = new_block_ids[i]  # 获取该组的新块 ID 列表
            self.block_tables[i].stage_write(req_index, start, block_ids)  # 暂存写入操作
            self.num_blocks.np[i, req_index] = start + len(block_ids)  # 更新该请求在该组中的块数量

    # 将所有暂存的块 ID 写入操作批量应用到 GPU 张量上。
    def apply_staged_writes(self) -> None:
        # TODO(woosuk): This can be inefficient since it launches one kernel per
        # block table. Implement a kernel to handle all block tables at once.
        for block_table in self.block_tables:  # 遍历每个块表
            block_table.apply_write()  # 执行暂存的写入操作
        self.num_blocks.copy_to_uva()  # 将块数量数据拷贝到 UVA 缓冲区供 GPU 访问

    # 通过 Triton 内核按 idx_mapping 将源块表的行 gather 到输入块表中，
    # 同时将超出实际请求数的填充行清零（用于 CUDA graph 的固定形状要求）。
    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,  # 批次索引到请求索引的映射
        num_reqs_padded: int,  # 填充后的请求数量（满足 CUDA graph 要求）
    ) -> tuple[torch.Tensor, ...]:
        num_reqs = idx_mapping.shape[0]  # 获取实际请求数量
        # Launch kernel with num_reqs_padded to fuse zeroing of padded rows.
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](  # 启动 Triton 内核执行 gather 操作
            idx_mapping,
            self.block_table_ptrs,
            self.input_block_table_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            self.input_block_tables[0].shape[1],  # max_num_blocks
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)  # 返回填充后大小的输入块表切片

    # 返回用于 CUDA graph 捕获的虚拟块表（返回持久化张量的切片以保持地址不变）。
    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)  # 返回持久化输入块表的前 num_reqs 行

    # 通过 Triton 内核计算 slot mapping：将每个 token 的位置映射到对应的 KV 缓存槽位。
    # 支持上下文并行（CP）场景下的交错分片逻辑，非本 rank 拥有的 token 映射为 PAD_SLOT_ID。
    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,  # 批次索引到请求索引的映射
        query_start_loc: torch.Tensor,  # query 起始位置张量
        positions: torch.Tensor,  # token 位置张量
        num_tokens_padded: int,  # 填充后的 token 数量
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]  # 获取实际请求数量
        num_groups = self.num_kv_cache_groups  # 获取 KV 缓存组数量
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](  # 启动 Triton 内核计算 slot 映射，+1 是为了处理尾部填充
            self.max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return self.slot_mappings[:, :num_tokens_padded]  # 返回填充后大小的 slot 映射切片

    # 返回用于 CUDA graph 捕获的虚拟 slot mapping，所有槽位填充为 PAD_SLOT_ID。
    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)  # 将整个 slot 映射张量填充为填充槽位 ID
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]  # 返回持久化张量的前 num_tokens 列


# Triton 内核：将源块表的行按 batch_idx -> req_idx 的映射 gather 到目标块表中。
# 对于超出实际请求数的填充行，将其清零以确保 CUDA graph 回放时的正确性。
@triton.jit  # Triton JIT 编译装饰器
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size] 批次索引到请求索引的映射数组
    src_block_table_ptrs,  # [num_kv_cache_groups] 源块表数据指针数组
    dst_block_table_ptrs,  # [num_kv_cache_groups] 目标块表数据指针数组
    block_table_strides,  # [num_kv_cache_groups] 块表行步幅数组
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs] 每个请求的块数量
    num_blocks_stride,  # num_blocks 张量的行步幅
    num_reqs,  # actual number of requests (for padding) 实际请求数量
    max_num_blocks,  # stride for zeroing padded rows 最大块数，用于清零填充行
    BLOCK_SIZE: tl.constexpr,  # Triton 块大小常量
):
    # kv cache group id
    group_id = tl.program_id(0)  # 获取当前 KV 缓存组的 ID
    batch_idx = tl.program_id(1)  # 获取当前批次索引

    stride = tl.load(block_table_strides + group_id)  # 加载当前组的块表行步幅
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)  # 加载目标块表的数据指针
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride  # 计算目标行的起始地址

    if batch_idx >= num_reqs:  # 如果批次索引超出实际请求数（填充行）
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):  # 遍历填充行的所有块
            offset = i + tl.arange(0, BLOCK_SIZE)  # 计算块内偏移
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)  # 将填充行清零
        return  # 提前返回

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)  # 加载实际的请求索引
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride  # 计算当前组块数量数组的起始地址
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)  # 加载该请求的实际块数量

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)  # 加载源块表的数据指针
    src_row_ptr = src_block_table_ptr + req_idx * stride  # 计算源行的起始地址

    for i in tl.range(0, num_blocks, BLOCK_SIZE):  # 遍历该请求的所有块
        offset = i + tl.arange(0, BLOCK_SIZE)  # 计算块内偏移
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)  # 从源块表加载块 ID
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)  # 将块 ID 写入目标块表


# Triton 内核：根据 token 位置和块表计算每个 token 的 KV 缓存槽位 ID。
# 算法：position -> (block_index, block_offset) -> block_number * block_size + offset。
# 支持上下文并行（CP_SIZE > 1）时的交错分片：仅当前 rank 拥有的 token 写入有效槽位，
# 其余写入 PAD_ID。最后一个 program 负责将尾部填充为 PAD_ID（用于 CUDA graph）。
@triton.jit  # Triton JIT 编译装饰器
def _compute_slot_mappings_kernel(
    max_num_tokens,  # 最大 token 数量
    idx_mapping,  # [num_reqs] 批次索引到请求索引的映射
    query_start_loc,  # [num_reqs + 1] query 起始位置数组
    pos,  # [num_tokens] token 位置数组
    block_table_ptrs,  # [num_kv_cache_groups] 块表数据指针数组
    block_table_strides,  # [num_kv_cache_groups] 块表行步幅数组
    block_sizes,  # [num_kv_cache_groups] 块大小数组
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens] 输出 slot 映射数组
    slot_mappings_stride,  # slot 映射的行步幅
    cp_rank,  # 当前上下文并行 rank
    CP_SIZE: tl.constexpr,  # 上下文并行大小常量
    CP_INTERLEAVE: tl.constexpr,  # 上下文并行交错因子常量
    PAD_ID: tl.constexpr,  # 填充槽位 ID 常量
    TRITON_BLOCK_SIZE: tl.constexpr,  # Triton 块大小常量
):
    # kv cache group id
    group_id = tl.program_id(0)  # 获取当前 KV 缓存组的 ID
    batch_idx = tl.program_id(1)  # 获取当前批次索引
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride  # 计算当前组 slot 映射的起始地址

    if batch_idx == tl.num_programs(1) - 1:  # 如果是最后一个 program（负责尾部填充）
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)  # 加载实际 token 总数
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):  # 从实际 token 数到最大 token 数遍历
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)  # 计算块内偏移
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)  # 将尾部填充为 PAD_ID
        return  # 提前返回

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)  # 加载当前组块表的数据指针
    block_table_stride = tl.load(block_table_strides + group_id)  # 加载块表行步幅
    block_size = tl.load(block_sizes + group_id)  # 加载当前组的块大小

    req_state_idx = tl.load(idx_mapping + batch_idx)  # 加载该批次位置对应的请求状态索引
    start_idx = tl.load(query_start_loc + batch_idx)  # 加载该请求的 query 起始位置
    end_idx = tl.load(query_start_loc + batch_idx + 1)  # 加载下一个请求的起始位置（即当前请求的结束位置）
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):  # 遍历该请求的所有 token
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)  # 计算块内偏移
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)  # 加载 token 位置

        block_indices = positions // (block_size * CP_SIZE)  # 计算每个 token 所在的块索引
        block_offsets = positions % (block_size * CP_SIZE)  # 计算每个 token 在块组内的偏移
        block_numbers = tl.load(  # 从块表中加载实际的块编号
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:  # 如果不使用上下文并行（常见情况）
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets  # 直接计算槽位 ID = 块编号 * 块大小 + 偏移
        else:  # 如果使用上下文并行
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank  # 判断该 token 是否属于当前 rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)  # 计算完整轮次数
            remainder = block_offsets % CP_INTERLEAVE  # 计算交错组内的余数
            local_offsets = rounds * CP_INTERLEAVE + remainder  # 计算本地偏移
            slot_ids = block_numbers * block_size + local_offsets  # 计算槽位 ID
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)  # 非本 rank 的 token 使用 PAD_ID

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)  # 将槽位 ID 写入 slot 映射


# Triton 辅助函数：从指针的指针加载实际数据指针，并转换为指定元素类型的指针。
@triton.jit  # Triton JIT 编译装饰器
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)  # 从指针地址加载实际数据指针值
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))  # 将指针值转换为指定元素类型的指针
    return tl.multiple_of(ptr, 16)  # 标记指针为 16 字节对齐以启用向量化访问
