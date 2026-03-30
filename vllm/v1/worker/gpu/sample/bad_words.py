# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np  # 导入 numpy 用于数组操作
import torch  # 导入 PyTorch 张量库

from vllm.sampling_params import SamplingParams  # 导入采样参数类
from vllm.triton_utils import tl, triton  # 导入 Triton JIT 编译工具
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor  # 导入缓冲区工具类
from vllm.v1.worker.gpu.states import RequestState  # 导入请求状态类

MAX_BAD_WORDS_TOTAL_TOKENS = 1024  # Max total tokens for all bad words per request # 每个请求所有禁用词的最大 token 总数
MAX_NUM_BAD_WORDS = 128  # Max number of bad words per request # 每个请求最大禁用词数量


# 禁用词（bad words）状态管理类
# 负责存储每个请求的禁用词 token 序列，并在采样时将匹配的禁用词的最后一个 token 的 logit 置为 -inf
# 数据结构设计：将每个请求的多个禁用词展平存储，并用偏移量数组记录每个禁用词的边界
# 使用 StagedWriteTensor 实现 CPU 端暂存、批量写入 GPU 的高效传输模式
class BadWordsState:
    # 初始化禁用词状态，分配存储禁用词 token、偏移量和计数的缓冲区
    def __init__(self, req_states: RequestState):  # 构造函数，接收请求状态对象
        self.req_states = req_states  # 保存请求状态的引用
        self.max_num_reqs = req_states.max_num_reqs  # 最大并发请求数
        self.device = req_states.device  # GPU 设备

        # flattened bad word tokens: [max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS]
        self.bad_word_token_ids = StagedWriteTensor(  # 展平的禁用词 token ID 张量
            (self.max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS),  # 形状为 [最大请求数, 最大 token 总数]
            dtype=torch.int32,  # 使用 int32 类型
            device=self.device,  # 存储在 GPU 上
        )
        # cumulative offsets of bad words: [max_num_reqs, MAX_NUM_BAD_WORDS + 1]
        self.bad_word_offsets = StagedWriteTensor(  # 禁用词的累计偏移量张量
            (self.max_num_reqs, MAX_NUM_BAD_WORDS + 1),  # 形状为 [最大请求数, 最大禁用词数+1]
            dtype=torch.int32,  # 使用 int32 类型
            device=self.device,  # 存储在 GPU 上
        )
        # number of bad words per request
        self.num_bad_words = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)  # 每个请求的禁用词数量

    # 为新请求注册禁用词列表，将禁用词展平并计算累计偏移量，暂存到 StagedWriteTensor 中
    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:  # 添加请求方法
        bad_words_token_ids = sampling_params.bad_words_token_ids  # 获取请求的禁用词 token 列表
        if not bad_words_token_ids:  # 如果没有禁用词
            self.num_bad_words.np[req_idx] = 0  # 将该请求的禁用词数量设为 0
            return  # 直接返回

        num_bad_words = len(bad_words_token_ids)  # 计算禁用词数量
        if num_bad_words > MAX_NUM_BAD_WORDS:  # 检查是否超过最大限制
            raise ValueError(  # 抛出值错误异常
                f"Too many bad words: {num_bad_words}. "  # 错误信息：禁用词过多
                f"The max number is {MAX_NUM_BAD_WORDS}."  # 提示最大限制值
            )

        # Flatten bad words and compute offsets
        flattened_tokens: list[int] = []  # 初始化展平的 token 列表
        offsets: list[int] = [0]  # 初始化偏移量列表，起始偏移为 0
        for bad_word in bad_words_token_ids:  # 遍历每个禁用词
            flattened_tokens.extend(bad_word)  # 将禁用词的 token 追加到展平列表
            offsets.append(len(flattened_tokens))  # 记录当前累计偏移量

        if len(flattened_tokens) > MAX_BAD_WORDS_TOTAL_TOKENS:  # 检查展平后的总 token 数是否超限
            raise ValueError(  # 抛出值错误异常
                f"Too many total bad word tokens: {len(flattened_tokens)}. "  # 错误信息：总 token 数过多
                f"The max is {MAX_BAD_WORDS_TOTAL_TOKENS}."  # 提示最大限制值
            )

        # Stage writes
        self.bad_word_token_ids.stage_write(req_idx, 0, flattened_tokens)  # 将展平的 token 暂存到写入缓冲区
        self.bad_word_offsets.stage_write(req_idx, 0, offsets)  # 将偏移量暂存到写入缓冲区
        self.num_bad_words.np[req_idx] = num_bad_words  # 记录该请求的禁用词数量

    # 将暂存的禁用词数据批量刷写到 GPU 显存
    def apply_staged_writes(self) -> None:  # 应用暂存写入方法
        self.num_bad_words.copy_to_uva()  # 将禁用词数量同步到 UVA 张量
        self.bad_word_token_ids.apply_write()  # 将禁用词 token 刷写到 GPU
        self.bad_word_offsets.apply_write()  # 将禁用词偏移量刷写到 GPU

    # 对 logits 应用禁用词掩码：若当前已生成的 token 序列与某禁用词的前缀匹配，则将该禁用词最后一个 token 的 logit 置为 -inf
    def apply_bad_words(
        self,
        logits: torch.Tensor,  # logits 张量 [num_tokens, vocab_size]
        expanded_idx_mapping: torch.Tensor,  # 扩展的索引映射
        idx_mapping_np: np.ndarray,  # numpy 格式的索引映射
        input_ids: torch.Tensor,  # 输入 token ID
        expanded_local_pos: torch.Tensor,  # 扩展的本地位置
    ) -> None:  # 无返回值
        max_num_bad_words = int(self.num_bad_words.np[idx_mapping_np].max())  # 获取当前批次中最大禁用词数量
        if max_num_bad_words == 0:  # 如果没有任何请求使用禁用词
            # No request uses bad words. Skip the kernel launch.
            return  # 跳过内核启动

        apply_bad_words(  # 调用禁用词内核入口函数
            logits,  # logits 张量
            expanded_idx_mapping,  # 扩展的索引映射
            self.bad_word_token_ids.gpu,  # GPU 上的禁用词 token
            self.bad_word_offsets.gpu,  # GPU 上的禁用词偏移量
            self.num_bad_words.gpu,  # GPU 上的禁用词数量
            self.req_states.all_token_ids.gpu,  # 所有已生成的 token ID
            self.req_states.prompt_len.gpu,  # 提示词长度
            self.req_states.total_len.gpu,  # 总长度
            input_ids,  # 输入 token ID
            expanded_local_pos,  # 扩展的本地位置
            max_num_bad_words,  # 最大禁用词数量
        )


# Triton 禁用词匹配内核
# 算法：对每个 token 位置和每个禁用词，检查已生成序列的后缀是否与禁用词的前缀完全匹配
# 若匹配，则将禁用词最后一个 token 对应的 logit 设为 -inf，阻止模型生成该 token
# 支持投机解码场景：可同时检查来自输出历史和投机输入的 token
@triton.jit
def _bad_words_kernel(
    logits_ptr,  # logits 张量指针
    logits_stride,  # logits 行步长
    expanded_idx_mapping_ptr,  # 扩展索引映射指针
    bad_word_token_ids_ptr,  # 禁用词 token ID 指针
    bad_word_token_ids_stride,  # 禁用词 token ID 行步长
    bad_word_offsets_ptr,  # 禁用词偏移量指针
    bad_word_offsets_stride,  # 禁用词偏移量行步长
    num_bad_words_ptr,  # 禁用词数量指针
    all_token_ids_ptr,  # 所有已生成 token ID 指针
    all_token_ids_stride,  # 所有已生成 token ID 行步长
    prompt_len_ptr,  # 提示词长度指针
    total_len_ptr,  # 总长度指针
    input_ids_ptr,  # 输入 token ID 指针
    expanded_local_pos_ptr,  # 扩展本地位置指针
):
    token_idx = tl.program_id(0)  # 获取当前 token 的索引（第一维网格 ID）
    bw_idx = tl.program_id(1)  # 获取当前禁用词的索引（第二维网格 ID）

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)  # 加载该 token 对应的请求状态索引
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)  # 加载该请求的禁用词数量

    if bw_idx >= num_bad_words:  # 如果当前禁用词索引超出该请求的禁用词数量
        return  # 直接返回

    pos = tl.load(expanded_local_pos_ptr + token_idx)  # 加载当前 token 的本地位置
    cur_req_first_pos = token_idx - pos  # 计算当前请求在展开序列中的起始位置

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)  # 加载提示词长度
    total_len = tl.load(total_len_ptr + req_state_idx)  # 加载总长度
    output_len = total_len - prompt_len  # 计算已输出的 token 数
    effective_len = output_len + pos  # 计算有效长度（已输出 + 当前投机位置）

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride  # 计算该请求的禁用词偏移量基地址
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride  # 计算该请求的禁用词 token 基地址
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len  # 计算输出 token 的基地址

    start = tl.load(bd_offsets_base + bw_idx)  # 加载当前禁用词在展平数组中的起始偏移
    end = tl.load(bd_offsets_base + bw_idx + 1)  # 加载当前禁用词在展平数组中的结束偏移
    bad_word_len = end - start  # 计算当前禁用词的长度
    prefix_len = bad_word_len - 1  # 前缀长度（不含最后一个 token）

    if prefix_len > effective_len:  # 如果前缀长度大于有效长度，无法匹配
        return  # 直接返回

    last_token = tl.load(bd_tokens_base + end - 1)  # 加载禁用词的最后一个 token
    match = 1  # 初始化匹配标志为 True
    for i in range(prefix_len):  # 遍历禁用词前缀的每个 token
        expected = tl.load(bd_tokens_base + start + i)  # 加载禁用词前缀中第 i 个期望的 token
        actual_pos = effective_len - prefix_len + i  # 计算实际序列中对应位置

        from_spec_input = actual_pos >= output_len  # 判断该位置是否来自投机输入
        if from_spec_input:  # 如果来自投机输入
            spec_offset = actual_pos - output_len  # 计算在投机输入中的偏移
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset)  # 从投机输入中加载实际 token
        else:  # 如果来自已输出的 token
            actual = tl.load(output_base + actual_pos)  # 从输出历史中加载实际 token

        match = match & (expected == actual)  # 更新匹配标志

    if match:  # 如果禁用词前缀完全匹配
        tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))  # 将禁用词最后一个 token 的 logit 设为负无穷


# 禁用词内核的入口函数，以 (num_tokens, max_num_bad_words) 的二维网格启动 Triton 内核
def apply_bad_words(
    logits: torch.Tensor,  # logits 张量
    expanded_idx_mapping: torch.Tensor,  # 扩展的索引映射
    bad_word_token_ids: torch.Tensor,  # 禁用词 token ID 张量
    bad_word_offsets: torch.Tensor,  # 禁用词偏移量张量
    num_bad_words: torch.Tensor,  # 禁用词数量张量
    all_token_ids: torch.Tensor,  # 所有已生成的 token ID
    prompt_len: torch.Tensor,  # 提示词长度张量
    total_len: torch.Tensor,  # 总长度张量
    input_ids: torch.Tensor,  # 输入 token ID 张量
    expanded_local_pos: torch.Tensor,  # 扩展本地位置张量
    max_num_bad_words: int,  # 当前批次最大禁用词数量
) -> None:  # 无返回值
    num_tokens = logits.shape[0]  # 获取 token 数量
    _bad_words_kernel[(num_tokens, max_num_bad_words)](  # 以二维网格启动 Triton 内核
        logits,  # logits 张量
        logits.stride(0),  # logits 行步长
        expanded_idx_mapping,  # 扩展的索引映射
        bad_word_token_ids,  # 禁用词 token ID
        bad_word_token_ids.stride(0),  # 禁用词 token ID 行步长
        bad_word_offsets,  # 禁用词偏移量
        bad_word_offsets.stride(0),  # 禁用词偏移量行步长
        num_bad_words,  # 禁用词数量
        all_token_ids,  # 所有已生成的 token ID
        all_token_ids.stride(0),  # 所有已生成 token ID 行步长
        prompt_len,  # 提示词长度
        total_len,  # 总长度
        input_ids,  # 输入 token ID
        expanded_local_pos,  # 扩展本地位置
    )
