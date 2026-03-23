# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that platform imports do not prematurely initialize CUDA.

This is critical for Ray-based multi-GPU setups where workers need to
set CUDA_VISIBLE_DEVICES after importing vLLM but before CUDA is initialized.
If CUDA is initialized during import, device_count() gets locked and ignores
subsequent env var changes.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent / "scripts"


# [中文注释] 在子进程中运行测试脚本，确保干净的CUDA状态
def run_script(script_name: str) -> subprocess.CompletedProcess:
    """Run a test script in a subprocess with clean CUDA state."""
    script_path = SCRIPTS_DIR / script_name
    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )


# [中文注释] 测试导入vllm.platforms不会初始化CUDA，这对Ray多GPU设置至关重要
def test_platform_import_does_not_init_cuda():
    """Test that importing vllm.platforms does not initialize CUDA."""
    result = run_script("check_platform_no_cuda_init.py")
    if result.returncode != 0:
        pytest.fail(f"Platform import initialized CUDA:\n{result.stderr}")


# [中文注释] 测试平台导入后device_count仍然尊重CUDA_VISIBLE_DEVICES环境变量
def test_device_count_respects_env_after_platform_import():
    """Test that device_count respects CUDA_VISIBLE_DEVICES after import."""
    result = run_script("check_device_count_respects_env.py")
    if result.returncode != 0:
        pytest.fail(
            f"device_count does not respect env var after import:\n{result.stderr}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
