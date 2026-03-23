# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.platforms import current_platform


# [中文注释] 通过CUDA驱动API检查当前线程的CUDA上下文状态
def check_cuda_context():
    """Check CUDA driver context status"""
    try:
        cuda = ctypes.CDLL("libcuda.so")
        device = ctypes.c_int()
        result = cuda.cuCtxGetDevice(ctypes.byref(device))
        return (True, device.value) if result == 0 else (False, None)
    except Exception:
        return False, None


# [中文注释] 在独立线程中运行CUDA上下文测试，验证set_device对不同输入类型的处理
def run_cuda_test_in_thread(device_input, expected_device_id):
    """Run CUDA context test in separate thread for isolation"""
    try:
        # New thread should have no CUDA context initially
        valid_before, device_before = check_cuda_context()
        if valid_before:
            return (
                False,
                "CUDA context should not exist in new thread, "
                f"got device {device_before}",
            )

        # Test setting CUDA context
        current_platform.set_device(device_input)

        # Verify context is created correctly
        valid_after, device_id = check_cuda_context()
        if not valid_after:
            return False, "CUDA context should be valid after set_cuda_context"
        if device_id != expected_device_id:
            return False, f"Expected device {expected_device_id}, got {device_id}"

        return True, "Success"
    except Exception as e:
        return False, f"Exception in thread: {str(e)}"


# [中文注释] 测试set_cuda_context函数，验证整数、torch.device和字符串三种设备输入类型
class TestSetCudaContext:
    """Test suite for the set_cuda_context function."""

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
    @pytest.mark.parametrize(
        argnames="device_input,expected_device_id",
        argvalues=[
            (0, 0),
            (torch.device("cuda:0"), 0),
            ("cuda:0", 0),
        ],
        ids=["int", "torch_device", "string"],
    )
    def test_set_cuda_context_parametrized(self, device_input, expected_device_id):
        """Test setting CUDA context in isolated threads."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_cuda_test_in_thread, device_input, expected_device_id
            )
            success, message = future.result(timeout=30)
        assert success, message

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
    def test_set_cuda_context_invalid_device_type(self):
        """Test error handling for invalid device type."""
        with pytest.raises(ValueError, match="Expected a cuda device"):
            current_platform.set_device(torch.device("cpu"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
