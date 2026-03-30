# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib  # 导入上下文管理工具模块
from collections.abc import Generator  # 导入生成器类型

import torch  # 导入PyTorch库

from vllm.logger import init_logger  # 导入日志初始化函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool:
    """检查一个节点是否应该在dynamo图分区中被分割。

    它在dynamo图上操作，因此node.target可以是任何值。
    我们只需要检查并分割OpOverload和OpOverloadPacket类型的节点。
    """

    if node.op != "call_function":  # 如果节点操作不是函数调用
        return False  # 返回False

    target = node.target  # 获取节点目标

    if isinstance(target, torch._ops.OpOverloadPacket):  # 如果目标是OpOverloadPacket类型
        # Example: "aten::add"
        return target._qualified_op_name in splitting_ops  # 检查操作名是否在分割操作列表中

    if isinstance(target, torch._ops.OpOverload):  # 如果目标是OpOverload类型
        # Example: "aten::add"
        packet_name = target.name()  # 获取操作包名

        # Example: "aten::add.default"
        op_overload_name = f"{packet_name}.{target._overloadname}"  # 构造完整的重载操作名
        return op_overload_name in splitting_ops or packet_name in splitting_ops  # 检查是否在分割列表中

    return False  # 其他类型返回False


@contextlib.contextmanager
def inductor_partition_rule_context(
    splitting_ops: list[str] | None,
) -> Generator[None, None, None]:
    """上下文管理器，用于临时注册Inductor分区规则。

    为指定的操作符注册自定义分区规则，强制
    Inductor调度器在这些操作符处对图进行分区。退出时
    规则会自动恢复到之前的状态。

    Args:
        splitting_ops: 要在其处分区的操作符名称列表。
    """
    if not splitting_ops:  # 如果没有提供分割操作
        logger.debug("No partition ops provided; skipping rule registration.")  # 记录跳过日志
        yield  # 直接让出控制权
        return  # 返回

    # Save current state before registering

    saved_splitting_ops: list[str] = list(  # 保存当前的分割操作列表
        torch._inductor.config.custom_should_partition_ops
    )
    torch._inductor.config.custom_should_partition_ops = splitting_ops  # 设置新的分割操作

    logger.debug(  # 记录注册的分区规则数量
        "Registered inductor partition rules for %d operators", len(splitting_ops)
    )

    try:  # 尝试执行
        yield  # 让出控制权给调用者
    finally:  # 最终清理
        # Clear and restore previous state
        torch._inductor.config.custom_should_partition_ops = saved_splitting_ops  # 恢复之前的分割操作
        logger.debug("Restored previous partition rules state.")  # 记录恢复日志
