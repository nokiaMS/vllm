# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型检查工具模块，用于格式化和展示vLLM模型的结构信息。"""

import torch.nn as nn  # 导入PyTorch神经网络模块


def _get_module_info(module: nn.Module) -> str:
    """获取模块的信息字符串。

    提取模块的类名、量化方法和额外表示信息，
    组合成可读的描述字符串。

    Args:
        module: PyTorch神经网络模块。

    Returns:
        模块信息的格式化字符串。
    """
    class_name = type(module).__name__  # 获取模块的类名
    parts = []  # 初始化信息部分列表

    # Add quant_method if present
    quant_method = getattr(module, "quant_method", None)  # 获取量化方法属性（如果存在）
    if quant_method is not None:  # 如果存在量化方法
        quant_name = type(quant_method).__name__  # 获取量化方法的类名
        # For CompressedTensors, show the underlying scheme instead
        scheme = getattr(module, "scheme", None)  # 获取压缩张量的底层方案
        if scheme is not None:  # 如果存在方案
            quant_name = type(scheme).__name__  # 使用方案的类名替代
        # Skip unquantized methods
        if "Unquantized" not in quant_name:  # 如果不是未量化的方法
            parts.append(f"quant={quant_name}")  # 添加量化方法信息

    # If module has extra_repr, use it
    if hasattr(module, "extra_repr"):  # 如果模块有extra_repr方法
        parts.append(module.extra_repr().replace("\n", ""))  # 添加额外表示信息（移除换行）

    if parts:  # 如果有信息部分
        return f"{class_name}({', '.join(parts)})"  # 返回带信息的格式化字符串

    # For unknown modules, use the default PyTorch repr
    return str(module)  # 对于未知模块使用默认的PyTorch表示


def _get_child_signature(child: nn.Module) -> str:
    """获取子模块的签名字符串，用于检测重复模块。

    遍历子模块的所有子级，生成唯一的签名字符串，
    用于判断两个模块结构是否相同。

    Args:
        child: 要生成签名的PyTorch子模块。

    Returns:
        子模块的签名字符串。
    """
    lines = []  # 初始化签名行列表
    for name, submodule in child.named_modules():  # 遍历所有子模块
        lines.append(f"{name}:{_get_module_info(submodule)}")  # 添加模块名和信息
    return "\n".join(lines)  # 将所有行连接成签名字符串


def _format_index_ranges(indices: list[int]) -> str:
    """将索引列表格式化为范围表示法。

    例如 [0,1,2,4,5,6] 格式化为 '0-2, 4-6'。

    Args:
        indices: 整数索引列表。

    Returns:
        格式化的索引范围字符串。
    """
    indices = sorted(indices)  # 对索引列表排序
    ranges = []  # 初始化范围列表
    start = end = indices[0]  # 设置起始和结束索引为第一个元素

    for idx in indices[1:]:  # 遍历剩余索引
        if idx == end + 1:  # 如果是连续的
            end = idx  # 扩展当前范围的结束位置
        else:  # 如果不连续
            ranges.append(str(start) if start == end else f"{start}-{end}")  # 保存当前范围
            start = end = idx  # 开始新的范围

    ranges.append(str(start) if start == end else f"{start}-{end}")  # 保存最后一个范围
    return ", ".join(ranges)  # 用逗号连接所有范围


def _format_module_tree(
    module: nn.Module,  # 要格式化的模块
    name: str = "",  # 模块名称
    indent: int = 0,  # 缩进级别
) -> list[str]:
    """将模块树格式化为带缩进的字符串列表，并对相同层进行分组。

    生成类似以下格式的输出：
        (layers): ModuleList(
          (0-27, 29-47): 47 x LlamaDecoderLayer(
            ...
          )
          (28, 48): 2 x DifferentDecoderLayer(
            ...
          )
        )

    Args:
        module: 要格式化的PyTorch模块。
        name: 模块的名称。
        indent: 当前缩进级别。

    Returns:
        格式化后的字符串行列表。
    """
    lines = []  # 初始化输出行列表
    prefix = "  " * indent  # 计算当前缩进前缀
    children = list(module.named_children())  # 获取所有命名子模块

    # Leaf node - just output the module info
    if not children:  # 如果是叶子节点（没有子模块）
        info = _get_module_info(module)  # 获取模块信息
        lines.append(f"{prefix}({name}): {info}" if name else f"{prefix}{info}")  # 添加带名称或不带名称的信息
        return lines  # 返回行列表

    # Non-leaf node - output opening line and recurse into children
    info = _get_module_info(module)  # 获取模块信息
    lines.append(f"{prefix}({name}): {info}(" if name else f"{prefix}{info}(")  # 添加开始行（带左括号）

    # Separate numbered children (e.g., "0", "1") from named ones (e.g., "norm")
    numbered: list[tuple[int, nn.Module]] = []  # 编号子模块列表
    non_numbered: list[tuple[str, nn.Module]] = []  # 非编号子模块列表
    for child_name, child_module in children:  # 遍历子模块
        try:  # 尝试将名称转为整数
            numbered.append((int(child_name), child_module))  # 添加到编号列表
        except ValueError:  # 名称不是数字
            non_numbered.append((child_name, child_module))  # 添加到非编号列表

    # Group numbered children by structure signature to collapse identical layers
    # e.g., layers 0-27 and 29-47 with same structure become "(0-27, 29-47): 47 x"
    if numbered:  # 如果有编号子模块
        sig_to_group: dict[str, list[tuple[int, nn.Module]]] = {}  # 签名到分组的映射字典
        for idx, child_module in numbered:  # 遍历编号子模块
            sig = _get_child_signature(child_module)  # 获取子模块签名
            sig_to_group.setdefault(sig, []).append((idx, child_module))  # 按签名分组

        # Output groups sorted by first index
        for group in sorted(sig_to_group.values(), key=lambda g: g[0][0]):  # 按组内最小索引排序
            indices = [idx for idx, _ in group]  # 提取所有索引
            representative = group[0][1]  # 取组中第一个模块作为代表
            child_lines = _format_module_tree(representative, "", indent + 1)  # 递归格式化代表模块
            first_line = child_lines[0].lstrip()  # 获取第一行并去除左侧空白
            child_prefix = "  " * (indent + 1)  # 子级缩进前缀

            if len(indices) > 1:  # 如果组中有多个模块
                range_str = _format_index_ranges(indices)  # 格式化索引范围
                child_lines[0] = (  # 更新第一行为分组格式
                    f"{child_prefix}({range_str}): {len(indices)} x {first_line}"  # 如"(0-27): 28 x ..."
                )
            else:  # 如果组中只有一个模块
                child_lines[0] = f"{child_prefix}({indices[0]}): {first_line}"  # 使用单个索引
            lines.extend(child_lines)  # 将子模块行添加到输出

    # Output non-numbered children (e.g., "embed_tokens", "norm")
    for child_name, child_module in non_numbered:  # 遍历非编号子模块
        lines.extend(_format_module_tree(child_module, child_name, indent + 1))  # 递归格式化并添加到输出

    lines.append(f"{prefix})")  # 添加闭合括号
    return lines  # 返回格式化后的行列表


def format_model_inspection(model: nn.Module) -> str:
    """将模型格式化为类似transformers风格的层级结构字符串。

    这是模型检查的主入口函数，将整个模型树转换为
    可读的层级结构表示。

    Args:
        model: 要检查的PyTorch模型。

    Returns:
        模型结构的格式化字符串。
    """
    return "\n".join(_format_module_tree(model))  # 将模块树格式化后用换行连接
