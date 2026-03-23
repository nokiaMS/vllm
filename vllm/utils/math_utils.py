# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 数学工具函数模块，提供整数取整、幂次计算等常用数学操作，广泛用于内存对齐和分块计算
"""Math utility functions for vLLM."""

# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216


# 向上取整除法：利用 Python 负数地板除的特性实现，避免浮点运算
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# 求不小于 n 的最小 2 的幂，利用 bit_length() 位运算高效计算
def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


# 求不大于 n 的最大 2 的幂，利用 bit_length() 位运算高效计算
def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    return 0 if n <= 0 else 1 << (n.bit_length() - 1)


# 将 x 向上对齐到 y 的整数倍，常用于内存地址对齐
def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


# 将 x 向下对齐到 y 的整数倍
def round_down(x: int, y: int) -> int:
    """Round down x to the nearest multiple of y."""
    return (x // y) * y


# 返回能整除 n 的最大 2 的幂，通过 n & (-n) 隔离最低有效位实现
def largest_power_of_2_divisor(n: int) -> int:
    """Return the largest power-of-2 that divides *n* (isolate lowest set bit)."""
    return n & (-n)
