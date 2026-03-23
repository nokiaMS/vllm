# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess

import pytest


# [中文注释] 测试vllm bench startup CLI命令是否能正常执行并返回成功状态码
@pytest.mark.benchmark
def test_bench_startup():
    command = [
        "vllm",
        "bench",
        "startup",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
