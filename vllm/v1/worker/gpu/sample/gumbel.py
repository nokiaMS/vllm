# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch  # 导入 PyTorch 张量库

from vllm.triton_utils import tl, triton  # 导入 Triton JIT 编译工具


# Triton 温度缩放内核
# 将每个 token 的 logits 除以对应请求的温度参数，控制采样的随机性
# 温度为 0 或 1 时提前返回以避免不必要的显存访问
@triton.jit
def _temperature_kernel(
    logits_ptr,  # logits 张量指针
    logits_stride,  # logits 行步长
    expanded_idx_mapping_ptr,  # 扩展索引映射指针
    temperature_ptr,  # 温度参数指针
    vocab_size,  # 词表大小
    BLOCK_SIZE: tl.constexpr,  # 每个块处理的元素数（编译时常量）
):
    token_idx = tl.program_id(0)  # 获取当前 token 索引（第一维网格 ID）
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)  # 加载该 token 对应的请求状态索引
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)  # 加载并转换温度为 FP32
    if temperature == 0.0 or temperature == 1.0:  # 如果温度为 0（贪心）或 1（无缩放）
        # Early return to avoid loading logits.
        return  # 提前返回，跳过 logits 加载

    block_idx = tl.program_id(1)  # 获取当前块索引（第二维网格 ID）
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算当前块内的元素索引
    mask = block < vocab_size  # 创建边界掩码，防止越界访问

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)  # 加载当前块的 logits
    logits = logits.to(tl.float32)  # 转换为 FP32 以保证精度
    logits = logits / temperature  # 除以温度进行缩放
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)  # 将缩放后的 logits 写回


# 温度缩放的入口函数，按词表大小分块启动 Triton 内核
def apply_temperature(
    logits: torch.Tensor,  # logits 张量 [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # 扩展的索引映射 [num_tokens]
    temperature: torch.Tensor,  # 温度参数 [max_num_reqs]
) -> None:  # 无返回值
    num_tokens, vocab_size = logits.shape  # 获取 token 数量和词表大小
    BLOCK_SIZE = 8192  # 每个块处理 8192 个元素
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)  # 计算需要的块数（向上取整除法）
    _temperature_kernel[(num_tokens, num_blocks)](  # 以二维网格启动内核
        logits,  # logits 张量
        logits.stride(0),  # logits 行步长
        expanded_idx_mapping,  # 扩展的索引映射
        temperature,  # 温度参数
        vocab_size,  # 词表大小
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小常量
    )


# Gumbel-Max 采样 Triton 内核
# 核心算法：利用 Gumbel-Max 技巧实现无需显式归一化的随机采样
# 1. 可选地对 logits 施加温度缩放
# 2. 基于请求种子和位置生成确定性的 Gumbel 噪声（支持可复现采样）
# 3. 将 Gumbel 噪声加到 logits 上，取 argmax 即等价于从 softmax 分布中采样
# 4. 分块计算局部最大值，后续由主机端汇总得到全局采样结果
@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,  # 局部 argmax 结果指针
    local_argmax_stride,  # 局部 argmax 行步长
    local_max_ptr,  # 局部最大值指针
    local_max_stride,  # 局部最大值行步长
    processed_logits_ptr,  # 处理后的 logits 输出指针（可为 None）
    processed_logits_stride,  # 处理后 logits 行步长
    logits_ptr,  # 原始 logits 指针
    logits_stride,  # 原始 logits 行步长
    expanded_idx_mapping_ptr,  # 扩展索引映射指针
    seeds_ptr,  # 随机种子指针
    pos_ptr,  # 位置指针
    temp_ptr,  # 温度指针
    vocab_size,  # 词表大小
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
    APPLY_TEMPERATURE: tl.constexpr,  # 是否在内核中应用温度（编译时常量）
):
    token_idx = tl.program_id(0)  # 获取当前 token 索引
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)  # 加载请求状态索引

    block_idx = tl.program_id(1)  # 获取当前块索引
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算块内元素索引
    mask = block < vocab_size  # 创建边界掩码
    logits = tl.load(  # 加载当前块的 logits
        logits_ptr + token_idx * logits_stride + block,  # 计算 logits 地址
        mask=mask,  # 应用边界掩码
        other=float("-inf"),  # 越界位置填充负无穷
    )
    logits = logits.to(tl.float32)  # 转换为 FP32

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)  # 加载温度并转换为 FP32
    if temp != 0.0 and APPLY_TEMPERATURE:  # 如果温度非零且需要应用温度
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp  # 对 logits 进行温度缩放

    # Store the temperature-applied logits.
    if processed_logits_ptr is not None:  # 如果需要输出处理后的 logits
        tl.store(  # 存储处理后的 logits
            processed_logits_ptr + req_state_idx * processed_logits_stride + block,  # 计算输出地址
            logits,  # 处理后的 logits 值
            mask=mask,  # 应用边界掩码
        )

    if temp != 0.0:  # 如果温度非零（非贪心模式）
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)  # 加载该请求的随机种子
        pos = tl.load(pos_ptr + token_idx)  # 加载当前 token 位置
        gumbel_seed = tl.randint(seed, pos)  # 基于种子和位置生成确定性的 Gumbel 种子

        # Generate gumbel noise in FP32.
        u = tl.rand(gumbel_seed, block)  # 生成 [0, 1) 均匀分布随机数
        u = tl.maximum(u, 1e-7)  # 下限截断防止 log(0)
        gumbel_noise = -tl.log(-tl.log(u))  # 通过逆变换生成 Gumbel 分布噪声

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))  # 将 Gumbel 噪声加到 logits 上

    value, idx = tl.max(logits, axis=0, return_indices=True)  # 计算当前块内的最大值和对应索引
    token_id = block_idx * BLOCK_SIZE + idx  # 将块内索引转换为全局 token ID
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)  # 存储局部 argmax token ID
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)  # 存储局部最大值


# Gumbel-Max 采样的入口函数
# 先分块计算局部 argmax，再通过 gather 获取全局采样 token
# 温度为 0 时退化为贪心解码（不加 Gumbel 噪声，直接取 argmax）
def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size] logits 张量
    expanded_idx_mapping: torch.Tensor,  # [num_tokens] 扩展的索引映射
    temperature: torch.Tensor,  # [max_num_reqs] 温度参数
    seed: torch.Tensor,  # [max_num_reqs] 随机种子
    pos: torch.Tensor,  # [num_tokens] token 位置
    apply_temperature: bool,  # 是否在内核中应用温度
    processed_logits_out: torch.Tensor | None = None,  # [num_reqs, vocab_size] 可选的处理后 logits 输出
) -> torch.Tensor:  # 返回采样的 token ID 张量
    num_tokens, vocab_size = logits.shape  # 获取 token 数量和词表大小
    BLOCK_SIZE = 1024  # 每个块处理 1024 个元素
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)  # 计算块数（向上取整）
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)  # 分配局部 argmax 结果张量
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)  # 分配局部最大值张量
    _gumbel_sample_kernel[(num_tokens, num_blocks)](  # 以二维网格启动 Gumbel 采样内核
        local_argmax,  # 局部 argmax 结果
        local_argmax.stride(0),  # 局部 argmax 行步长
        local_max,  # 局部最大值
        local_max.stride(0),  # 局部最大值行步长
        processed_logits_out,  # 处理后的 logits 输出
        processed_logits_out.stride(0) if processed_logits_out is not None else 0,  # 处理后 logits 行步长（为 None 时用 0）
        logits,  # 原始 logits
        logits.stride(0),  # 原始 logits 行步长
        expanded_idx_mapping,  # 扩展的索引映射
        seed,  # 随机种子
        pos,  # token 位置
        temperature,  # 温度参数
        vocab_size,  # 词表大小
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小常量
        APPLY_TEMPERATURE=apply_temperature,  # 是否应用温度标志
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)  # 在所有块中找到全局最大值所在的块索引
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)  # 根据块索引取出对应的 token ID 并展平
    return sampled  # 返回采样的 token ID
