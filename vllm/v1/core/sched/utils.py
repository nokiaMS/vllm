# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib  # 导入上下文管理工具
from collections.abc import Sequence  # 导入序列抽象基类

from vllm.sampling_params import RepetitionDetectionParams  # 导入重复检测参数
from vllm.v1.request import Request, RequestStatus  # 导入请求对象和请求状态枚举


# [中文注释] 检测 token 序列尾部是否存在重复模式。
#   从末尾向前检查长度为 pattern_len 的模式是否连续重复了 repetition_min_count 次。
def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):  # 从末尾向前遍历模式中的每个位置
        target_token = token_ids[-n]  # 获取目标token（最后一个模式中的token）
        for m in range(1, repetition_min_count):  # 检查前面的重复次数
            if token_ids[-(pattern_len * m + n)] != target_token:  # 如果对应位置的token不匹配
                return False  # 不是重复模式
    return True  # 所有位置都匹配，确认存在重复模式


# [中文注释] 检测 token 序列是否存在重复模式（用于 repetition_detection 停止条件）。
#   遍历 [min_pattern_size, max_pattern_size] 范围内的所有模式长度，
#   对每个长度调用 _has_repeating_pattern 检查是否有 min_count 次重复。
def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size  # 最大模式长度
    min_pattern_size = params.min_pattern_size  # 最小模式长度
    min_count = params.min_count  # 最小重复次数

    if min_pattern_size <= 0:  # 最小模式长度不合法时默认为1
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:  # 参数不合法时直接返回
        return False

    for pattern_len in range(  # 遍历所有可能的模式长度
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):  # token序列长度不足以包含该模式的重复
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):  # 检测到重复模式
            return True

    return False  # 未检测到任何重复模式


# [中文注释] 从列表中移除指定集合中的所有元素。
#   优化：单元素移除时直接用 list.remove()（最常见情况），多元素时用列表推导式。
def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:  # 空集合无需移除
        return lst

    if len(items_to_remove) == 1:  # 单元素快速路径（最常见情况）
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))  # 获取唯一元素
        with contextlib.suppress(ValueError):  # 忽略元素不存在的异常
            lst.remove(item)  # 原地移除
        return lst  # 返回修改后的原列表
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]  # 多元素时用列表推导式创建新列表


# [中文注释] 检查请求是否满足停止条件（每生成一个 token 后调用）。
#   按优先级依次检查：
#     1. min_tokens 未满足 → 不停止
#     2. EOS token → FINISHED_STOPPED
#     3. stop_token_ids 命中 → FINISHED_STOPPED
#     4. 达到 max_model_len 或 max_tokens → FINISHED_LENGTH_CAPPED
#     5. 重复模式检测 → FINISHED_REPETITION
def check_stop(request: Request, max_model_len: int) -> bool:
    """检查请求是否满足停止条件，返回True表示应停止生成。"""
    assert not request.pooling_params  # 池化请求不应调用此函数

    sampling_params = request.sampling_params  # 获取采样参数
    assert sampling_params is not None  # 采样参数不能为空

    if request.num_output_tokens < sampling_params.min_tokens:  # 未达到最小token数要求
        return False  # 不停止

    last_token_id = request.output_token_ids[-1]  # 获取最后生成的token ID
    if last_token_id == sampling_params.eos_token_id:  # 命中EOS（结束符）token
        request.status = RequestStatus.FINISHED_STOPPED  # 设置为正常停止
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):  # 命中自定义停止token
        request.status = RequestStatus.FINISHED_STOPPED  # 设置为正常停止
        request.stop_reason = last_token_id  # 记录停止原因为该token ID
        return True
    if (
        request.num_tokens >= max_model_len  # 达到模型最大长度
        or request.num_output_tokens >= request.max_tokens  # 达到请求最大输出token数
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED  # 设置为长度上限停止
        return True

    repetition_detection = sampling_params.repetition_detection  # 获取重复检测参数
    if repetition_detection is not None and (  # 如果启用了重复检测
        check_sequence_repetition(
            request.output_token_ids,  # 检查输出token序列
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION  # 设置为重复停止
        request.stop_reason = "repetition_detected"  # 记录停止原因
        return True

    return False  # 未满足任何停止条件
