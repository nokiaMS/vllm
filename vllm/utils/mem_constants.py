# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 内存单位常量定义模块
# 提供 SI 制（MB/GB，以 1000 为基数）和二进制制（MiB/GiB，以 1024 为基数）的字节换算常量
MB_bytes = 1_000_000
"""The number of bytes in one megabyte (MB)."""

MiB_bytes = 1 << 20
"""The number of bytes in one mebibyte (MiB)."""

GB_bytes = 1_000_000_000
"""The number of bytes in one gigabyte (GB)."""

GiB_bytes = 1 << 30
"""The number of bytes in one gibibyte (GiB)."""
