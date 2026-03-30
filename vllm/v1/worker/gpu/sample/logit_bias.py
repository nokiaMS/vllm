# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np  # 导入 numpy 用于数组操作
import torch  # 导入 PyTorch 张量库

from vllm.sampling_params import SamplingParams  # 导入采样参数类
from vllm.triton_utils import tl, triton  # 导入 Triton JIT 编译工具
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor  # 导入缓冲区工具类

MAX_NUM_ALLOWED_TOKEN_IDS = 1024  # 白名单 token ID 的最大数量
MAX_NUM_LOGIT_BIAS_TOKENS = 1024  # logit 偏置 token 的最大数量
MAX_NUM_STOP_TOKEN_IDS = 128  # 停止 token ID 的最大数量


# Logit 偏置状态管理类
# 统一管理三种 logit 修改机制：
# 1. allowed_token_ids（白名单）：仅允许指定 token，其余 token 的 logit 置为 -inf
# 2. logit_bias（偏置）：对指定 token 的 logit 加上偏置值
# 3. min_tokens（最小生成长度）：在达到最小长度前，将 stop token 的 logit 置为 -inf
class LogitBiasState:
    # 初始化 logit 偏置状态，分配白名单、偏置值和停止 token 的缓冲区
    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs  # 最大并发请求数

        # Allowed token IDs.
        self.num_allowed_token_ids = UvaBackedTensor(  # 每个请求的白名单 token 数量
            self.max_num_reqs, dtype=torch.int32  # 使用 int32 类型
        )
        self.allowed_token_ids = StagedWriteTensor(  # 白名单 token ID 张量
            (self.max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS),  # 形状为 [最大请求数, 最大白名单数]
            dtype=torch.int32,  # 使用 int32 类型
            device=device,  # 存储在 GPU 上
        )
        # Logit bias.
        self.num_logit_bias = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)  # 每个请求的 logit 偏置 token 数量
        self.logit_bias_token_ids = StagedWriteTensor(  # logit 偏置对应的 token ID 张量
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),  # 形状为 [最大请求数, 最大偏置数]
            dtype=torch.int32,  # 使用 int32 类型
            device=device,  # 存储在 GPU 上
        )
        self.logit_bias = StagedWriteTensor(  # logit 偏置值张量
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),  # 形状为 [最大请求数, 最大偏置数]
            dtype=torch.float32,  # 使用 float32 类型
            device=device,  # 存储在 GPU 上
        )
        # Min tokens.
        self.min_lens = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)  # 每个请求的最小生成长度
        self.num_stop_token_ids = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)  # 每个请求的停止 token 数量
        self.stop_token_ids = StagedWriteTensor(  # 停止 token ID 张量
            (self.max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),  # 形状为 [最大请求数, 最大停止 token 数]
            dtype=torch.int32,  # 使用 int32 类型
            device=device,  # 存储在 GPU 上
        )

        # Using any of the above.
        self.use_logit_bias = np.zeros(max_num_reqs, dtype=bool)  # 标记每个请求是否使用了 logit 偏置

    # 为新请求注册 logit 偏置相关参数，包括白名单 token、logit 偏置值和最小生成长度约束
    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams  # 请求索引、提示词长度、采样参数
    ) -> None:
        # Using any logit bias.
        use_logit_bias = False  # 初始化使用标志为 False

        # Allowed token IDs.
        allowed_token_ids = sampling_params.allowed_token_ids  # 获取白名单 token ID 列表
        if allowed_token_ids:  # 如果有白名单 token
            num_allowed_token_ids = len(allowed_token_ids)  # 获取白名单 token 数量
            if num_allowed_token_ids > MAX_NUM_ALLOWED_TOKEN_IDS:  # 检查是否超过最大限制
                raise ValueError(  # 抛出值错误异常
                    f"Too many allowed token IDs: {num_allowed_token_ids}. "  # 错误信息：白名单 token 过多
                    f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}."  # 提示最大限制值
                )
            self.num_allowed_token_ids.np[req_idx] = num_allowed_token_ids  # 记录白名单 token 数量
            self.allowed_token_ids.stage_write(req_idx, 0, allowed_token_ids)  # 暂存白名单 token ID
            use_logit_bias = True  # 标记使用了 logit 偏置
        else:  # 如果没有白名单 token
            self.num_allowed_token_ids.np[req_idx] = 0  # 设置数量为 0

        # Logit bias.
        logit_bias = sampling_params.logit_bias  # 获取 logit 偏置字典
        if logit_bias:  # 如果有 logit 偏置
            num_logit_bias = len(logit_bias)  # 获取偏置 token 数量
            if num_logit_bias > MAX_NUM_LOGIT_BIAS_TOKENS:  # 检查是否超过最大限制
                raise ValueError(  # 抛出值错误异常
                    f"Too many logit bias tokens: {num_logit_bias}. "  # 错误信息：偏置 token 过多
                    f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}."  # 提示最大限制值
                )
            self.num_logit_bias.np[req_idx] = num_logit_bias  # 记录偏置 token 数量
            self.logit_bias_token_ids.stage_write(req_idx, 0, logit_bias.keys())  # 暂存偏置 token ID
            self.logit_bias.stage_write(req_idx, 0, logit_bias.values())  # 暂存偏置值
            use_logit_bias = True  # 标记使用了 logit 偏置
        else:  # 如果没有 logit 偏置
            self.num_logit_bias.np[req_idx] = 0  # 设置数量为 0

        # Min tokens.
        min_tokens = sampling_params.min_tokens  # 获取最小生成 token 数
        min_len = prompt_len + min_tokens  # 计算最小总长度（提示词 + 最小生成数）
        self.min_lens.np[req_idx] = min_len  # 记录最小长度
        stop_token_ids = sampling_params.all_stop_token_ids  # 获取所有停止 token ID
        if min_tokens > 0 and stop_token_ids:  # 如果设置了最小 token 数且有停止 token
            num_stop_token_ids = len(stop_token_ids)  # 获取停止 token 数量
            if num_stop_token_ids > MAX_NUM_STOP_TOKEN_IDS:  # 检查是否超过最大限制
                raise ValueError(  # 抛出值错误异常
                    f"Too many stop tokens: {num_stop_token_ids}. "  # 错误信息：停止 token 过多
                    f"The max size is {MAX_NUM_STOP_TOKEN_IDS}."  # 提示最大限制值
                )
            self.num_stop_token_ids.np[req_idx] = num_stop_token_ids  # 记录停止 token 数量
            self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)  # 暂存停止 token ID
            use_logit_bias = True  # 标记使用了 logit 偏置
        else:  # 如果不需要最小长度约束
            self.num_stop_token_ids.np[req_idx] = 0  # 设置停止 token 数量为 0

        self.use_logit_bias[req_idx] = use_logit_bias  # 更新该请求的使用标志

    # 将暂存的偏置数据批量刷写到 GPU 显存
    def apply_staged_writes(self) -> None:
        self.num_allowed_token_ids.copy_to_uva()  # 同步白名单 token 数量到 UVA
        self.allowed_token_ids.apply_write()  # 刷写白名单 token ID 到 GPU

        self.num_logit_bias.copy_to_uva()  # 同步 logit 偏置数量到 UVA
        self.logit_bias_token_ids.apply_write()  # 刷写偏置 token ID 到 GPU
        self.logit_bias.apply_write()  # 刷写偏置值到 GPU

        self.min_lens.copy_to_uva()  # 同步最小长度到 UVA
        self.num_stop_token_ids.copy_to_uva()  # 同步停止 token 数量到 UVA
        self.stop_token_ids.apply_write()  # 刷写停止 token ID 到 GPU

    # 对 logits 原地应用所有偏置操作（白名单、偏置值、最小长度约束）
    def apply_logit_bias(
        self,
        logits: torch.Tensor,  # logits 张量 [num_tokens, vocab_size]
        expanded_idx_mapping: torch.Tensor,  # 扩展的索引映射
        idx_mapping_np: np.ndarray,  # numpy 格式的索引映射
        pos: torch.Tensor,  # 位置张量
    ) -> None:
        if not np.any(self.use_logit_bias[idx_mapping_np]):  # 如果当前批次没有请求使用 logit 偏置
            # No request uses logit bias. Skip the kernel launch.
            return  # 跳过内核启动

        apply_logit_bias(  # 调用偏置内核入口函数
            logits,  # logits 张量
            expanded_idx_mapping,  # 扩展的索引映射
            pos,  # 位置张量
            self.num_allowed_token_ids.gpu,  # GPU 上的白名单 token 数量
            self.allowed_token_ids.gpu,  # GPU 上的白名单 token ID
            self.num_logit_bias.gpu,  # GPU 上的偏置数量
            self.logit_bias_token_ids.gpu,  # GPU 上的偏置 token ID
            self.logit_bias.gpu,  # GPU 上的偏置值
            self.min_lens.gpu,  # GPU 上的最小长度
            self.num_stop_token_ids.gpu,  # GPU 上的停止 token 数量
            self.stop_token_ids.gpu,  # GPU 上的停止 token ID
        )


# Triton logit 偏置内核
# 在一个内核中依次执行三种操作：
# 1. 白名单：先保存白名单 token 的 logit，将所有 logit 设为 -inf，再恢复白名单 token 的 logit
# 2. 偏置：将指定 token 的 logit 加上对应的偏置值
# 3. 最小长度：若当前位置未达到最小生成长度，将 stop token 的 logit 设为 -inf
@triton.jit
def _bias_kernel(
    logits_ptr,  # logits 张量指针
    logits_stride,  # logits 行步长
    vocab_size,  # 词表大小
    expanded_idx_mapping_ptr,  # 扩展索引映射指针
    # Allowed token IDs.
    num_allowed_token_ids_ptr,  # 白名单 token 数量指针
    allowed_token_ids_ptr,  # 白名单 token ID 指针
    allowed_token_ids_stride,  # 白名单 token ID 行步长
    # Logit bias.
    num_logit_bias_ptr,  # logit 偏置数量指针
    bias_token_ids_ptr,  # 偏置 token ID 指针
    bias_token_ids_stride,  # 偏置 token ID 行步长
    bias_ptr,  # 偏置值指针
    bias_stride,  # 偏置值行步长
    # Min tokens.
    pos_ptr,  # 位置指针
    min_lens_ptr,  # 最小长度指针
    num_stop_token_ids_ptr,  # 停止 token 数量指针
    stop_token_ids_ptr,  # 停止 token ID 指针
    stop_token_ids_stride,  # 停止 token ID 行步长
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
    LOGITS_BLOCK_SIZE: tl.constexpr,  # logits 块大小（编译时常量）
):
    token_idx = tl.program_id(0)  # 获取当前 token 索引
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)  # 加载请求状态索引

    block = tl.arange(0, BLOCK_SIZE)  # 生成块内索引范围

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)  # 加载白名单 token 数量
    if num_allowed_token_ids > 0:  # 如果有白名单 token
        block = tl.arange(0, BLOCK_SIZE)  # 重新生成块内索引
        mask = block < num_allowed_token_ids  # 创建白名单掩码

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(  # 加载白名单 token ID
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,  # 计算地址
            mask=mask,  # 应用掩码
        )
        logits = tl.load(  # 保存白名单 token 对应的 logits 值
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask  # 通过 token ID 间接索引
        )

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):  # 遍历整个词表，分块处理
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)  # 计算当前块的偏移
            tl.store(  # 将所有 logits 设为负无穷
                logits_ptr + token_idx * logits_stride + offset,  # 计算存储地址
                -float("inf"),  # 设为负无穷
                mask=offset < vocab_size,  # 边界掩码
            )

        # Restore logits for allowed token IDs.
        tl.store(  # 恢复白名单 token 的 logits 值
            logits_ptr + token_idx * logits_stride + allowed_token_ids,  # 通过 token ID 间接索引
            logits,  # 之前保存的 logits 值
            mask=mask,  # 应用掩码
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)  # 加载偏置 token 数量
    if num_logit_bias > 0:  # 如果有 logit 偏置
        mask = block < num_logit_bias  # 创建偏置掩码
        token_ids = tl.load(  # 加载需要偏置的 token ID
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,  # 计算地址
            mask=mask,  # 应用掩码
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)  # 加载偏置值
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)  # 加载对应 token 的 logits
        logits += bias  # 将偏置值加到 logits 上
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)  # 将修改后的 logits 写回

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)  # 加载停止 token 数量
    pos = tl.load(pos_ptr + token_idx)  # 加载当前位置
    min_len = tl.load(min_lens_ptr + req_state_idx)  # 加载最小长度
    if num_stop_token_ids > 0 and pos < min_len:  # 如果有停止 token 且未达到最小长度
        mask = block < num_stop_token_ids  # 创建停止 token 掩码
        stop_token_ids = tl.load(  # 加载停止 token ID
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,  # 计算地址
            mask=mask,  # 应用掩码
        )
        tl.store(  # 将停止 token 的 logit 设为负无穷
            logits_ptr + token_idx * logits_stride + stop_token_ids,  # 通过 token ID 间接索引
            -float("inf"),  # 设为负无穷
            mask=mask,  # 应用掩码
        )


# 偏置内核的入口函数，计算合适的 BLOCK_SIZE 并启动 Triton 内核
def apply_logit_bias(
    logits: torch.Tensor,  # logits 张量 [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # 扩展的索引映射
    pos: torch.Tensor,  # 位置张量
    num_allowed_token_ids: torch.Tensor,  # 白名单 token 数量
    allowed_token_ids: torch.Tensor,  # 白名单 token ID
    num_logit_bias: torch.Tensor,  # logit 偏置数量
    logit_bias_token_ids: torch.Tensor,  # 偏置 token ID
    logit_bias: torch.Tensor,  # 偏置值
    min_lens: torch.Tensor,  # 最小长度
    num_stop_token_ids: torch.Tensor,  # 停止 token 数量
    stop_token_ids: torch.Tensor,  # 停止 token ID
) -> None:
    num_tokens, vocab_size = logits.shape  # 获取 token 数量和词表大小
    BLOCK_SIZE = triton.next_power_of_2(  # 计算块大小为 2 的幂次
        max(  # 取三者中的最大值
            allowed_token_ids.shape[-1],  # 白名单 token 的最大维度
            logit_bias_token_ids.shape[-1],  # 偏置 token 的最大维度
            stop_token_ids.shape[-1],  # 停止 token 的最大维度
        )
    )
    LOGITS_BLOCK_SIZE = 8192  # logits 块大小设为 8192
    _bias_kernel[(num_tokens,)](  # 以一维网格启动 Triton 内核
        logits,  # logits 张量
        logits.stride(0),  # logits 行步长
        vocab_size,  # 词表大小
        expanded_idx_mapping,  # 扩展的索引映射
        num_allowed_token_ids,  # 白名单 token 数量
        allowed_token_ids,  # 白名单 token ID
        allowed_token_ids.stride(0),  # 白名单 token ID 行步长
        num_logit_bias,  # logit 偏置数量
        logit_bias_token_ids,  # 偏置 token ID
        logit_bias_token_ids.stride(0),  # 偏置 token ID 行步长
        logit_bias,  # 偏置值
        logit_bias.stride(0),  # 偏置值行步长
        pos,  # 位置张量
        min_lens,  # 最小长度
        num_stop_token_ids,  # 停止 token 数量
        stop_token_ids,  # 停止 token ID
        stop_token_ids.stride(0),  # 停止 token ID 行步长
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小常量
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,  # logits 块大小常量
    )
