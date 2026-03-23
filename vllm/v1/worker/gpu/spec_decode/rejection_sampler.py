# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler


# 严格拒绝采样的 Triton 内核：逐步比较目标模型采样结果与草稿模型采样结果，
# 遇到第一个不匹配的 token 即停止接受，将目标采样结果写入输出缓冲区。
# 每个请求由一个 Triton program 处理，避免了 CPU-GPU 同步。
@triton.jit
def _strict_rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            if target_sampled != draft_sampled:
                rejected = True
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, target_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


# 严格拒绝采样的入口函数：分配输出张量并启动 Triton 内核，
# 返回每个请求被接受的 token 序列及其数量
def strict_rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    num_speculative_steps,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = target_sampled.new_empty(num_reqs, num_speculative_steps + 1)
    num_sampled = target_sampled.new_empty(num_reqs, dtype=torch.int32)
    _strict_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        draft_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


# 概率拒绝采样的 Triton 内核：对每个草稿 token，以 min(1, target_prob / draft_prob) 的概率接受，
# 使用基于位置的随机数生成器保证与目标采样的 Gumbel 噪声一致性。
# 记录每个请求中第一个被拒绝的步骤索引，用于后续残差分布重采样。
@triton.jit
def _probabilistic_rejection_sample_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_reqs, num_speculative_steps, V]
    draft_probs_ptr,
    draft_probs_stride_0,
    draft_probs_stride_1,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    pos_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [num_reqs]
    seeds_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_tokens = tl.load(cu_num_logits_ptr + req_idx + 1) - start_idx
    seed = tl.load(seeds_ptr + tl.load(idx_mapping_ptr + req_idx))

    rejected_step = 0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            draft_sampled = tl.load(draft_sampled_ptr + start_idx + i + 1)
            target_prob = tl.load(
                target_probs_ptr + (start_idx + i) * target_probs_stride + draft_sampled
            )
            draft_prob = tl.load(
                draft_probs_ptr
                + req_idx * draft_probs_stride_0
                + i * draft_probs_stride_1
                + draft_sampled
            )
            pos = tl.load(pos_ptr + start_idx + i)
            u = tl.sum(tl.rand(seed, pos + tl.arange(0, 1)))
            accepted &= target_prob > u * draft_prob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)


# 计算残差 logits 的 Triton 内核：对于被拒绝的 token 位置，计算 max(target_prob - draft_prob, 0) 并取对数，
# 作为重采样的分布；对于奖励 token（所有草稿 token 都被接受的情况），直接使用目标模型的 logits。
# 同时记录残差 logits 对应的位置信息，用于 Gumbel 采样。
@triton.jit
def _compute_residual_logits_kernel(
    # [num_reqs, V]
    residual_logits_ptr,
    residual_logits_stride,
    # [num_reqs]
    residual_pos_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_reqs, num_speculative_steps, V]
    draft_probs_ptr,
    draft_probs_stride_0,
    draft_probs_stride_1,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    rejected_draft_step = tl.load(rejected_step_ptr + req_idx)
    rejected_logit_idx = start_idx + rejected_draft_step

    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if rejected_logit_idx < end_idx - 1:
        target_probs = tl.load(
            target_probs_ptr + rejected_logit_idx * target_probs_stride + block_offsets,
            mask=mask,
            other=0.0,
        )
        draft_probs = tl.load(
            draft_probs_ptr
            + req_idx * draft_probs_stride_0
            + rejected_draft_step * draft_probs_stride_1
            + block_offsets,
            mask=mask,
            other=0.0,
        )
        residual_probs = tl.maximum(target_probs - draft_probs, 0.0)
        residual_logits = tl.log(residual_probs)
    else:
        # This is a bonus token. Directly return the target logits.
        residual_logits = tl.load(
            target_logits_ptr
            + rejected_logit_idx * target_logits_stride
            + block_offsets,
            mask=mask,
            other=0.0,
        )

    tl.store(
        residual_logits_ptr + req_idx * residual_logits_stride + block_offsets,
        residual_logits,
        mask=mask,
    )

    # First block computes the residual logit positions.
    if block_idx == 0:
        pos_val = tl.load(pos_ptr + rejected_logit_idx)
        tl.store(residual_pos_ptr + req_idx, pos_val)


# 概率拒绝采样的完整流程：
# 1. 计算目标和草稿模型的 softmax 概率分布
# 2. 运行概率拒绝采样内核，按概率比决定接受或拒绝每个草稿 token
# 3. 对被拒绝/奖励位置计算残差 logits 分布
# 4. 使用 Gumbel 采样从残差分布中重采样替代 token
# 与严格拒绝采样不同，概率方法在目标概率接近草稿概率时仍有较高接受率
def probabilistic_rejection_sample(
    # [num_draft_tokens + num_reqs, V]
    target_logits: torch.Tensor,
    # [num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = target_logits.shape[-1]

    # Compute target and draft probs.
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    # Rejection sample.
    # [num_reqs, num_speculative_steps + 1]
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    # [num_reqs]
    rejected_steps = sampled.new_empty(num_reqs)
    _probabilistic_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        rejected_steps,
        draft_sampled,
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        cu_num_logits,
        pos,
        idx_mapping,
        seed,
        num_warps=1,
    )

    # Compute the logits and positions to resample the rejected/bonus
    # tokens from.
    # [num_reqs, vocab_size]
    residual_logits = target_logits.new_empty(num_reqs, vocab_size)
    # [num_reqs]
    residual_pos = pos.new_empty(num_reqs)
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _compute_residual_logits_kernel[(num_reqs, num_blocks)](
        residual_logits,
        residual_logits.stride(0),
        residual_pos,
        target_logits,
        target_logits.stride(0),
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        rejected_steps,
        cu_num_logits,
        pos,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Gumbel sample tokens from the residual distribution.
    resampled = gumbel_sample(
        residual_logits,
        idx_mapping,
        temperature,
        seed,
        residual_pos,
        apply_temperature=False,
    )
    sampled.scatter_(1, rejected_steps.unsqueeze(1), resampled.unsqueeze(1))

    return sampled, rejected_steps + 1


# 拒绝采样器：封装了严格拒绝采样和概率拒绝采样两种策略。
# 严格模式直接比较目标与草稿采样结果（贪婪匹配），适用于确定性采样场景；
# 概率模式根据概率比接受草稿 token，接受率更高但需要额外的概率计算开销。
# 该类作为推测解码验证阶段的核心组件，决定哪些草稿 token 被最终接受。
class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        num_speculative_steps,
        use_strict_rejection_sampling: bool = True,
    ):
        self.sampler = sampler
        self.num_speculative_steps = num_speculative_steps
        self.use_strict_rejection_sampling = use_strict_rejection_sampling

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        if self.use_strict_rejection_sampling:
            sampler_output = self.sampler(logits, input_batch)
            logprobs_tensors = sampler_output.logprobs_tensors
            sampled, num_sampled = strict_rejection_sample(
                sampler_output.sampled_token_ids.view(-1),
                draft_sampled,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
        else:
            assert draft_logits is not None
            pos = input_batch.positions[input_batch.logits_indices]
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                input_batch.expanded_idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                draft_sampled,
                input_batch.expanded_local_pos,
            )
            # TODO (TheEpicDolphin): Return logprobs for sampled token ids.
            logprobs_tensors = None
            sampled, num_sampled = probabilistic_rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                input_batch.cu_num_logits,
                pos,
                input_batch.idx_mapping,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
