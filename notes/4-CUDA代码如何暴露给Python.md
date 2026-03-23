# vLLM 中 CUDA C++/CU 代码暴露给 Python 的原理与实例

## 整体架构（4 层）

```
┌─────────────────────────────────────────────┐
│  Python 层:  torch.ops._C.silu_and_mul()    │  vllm/_custom_ops.py
├─────────────────────────────────────────────┤
│  注册层:  TORCH_LIBRARY(ops.def/ops.impl)   │  csrc/torch_bindings.cpp
├─────────────────────────────────────────────┤
│  C++ Wrapper: void silu_and_mul(Tensor&...) │  csrc/activation_kernels.cu
├─────────────────────────────────────────────┤
│  CUDA Kernel: __global__ act_and_mul_kernel │  csrc/activation_kernels.cu
└─────────────────────────────────────────────┘
```

## 原理说明

vLLM 使用的是 **PyTorch 的 `torch.library` 自定义算子注册机制**（非传统 pybind11），流程如下：

1. **CMake 编译** — `setup.py` 定义 `CMakeExtension(name="vllm._C")`，调用 CMake 将所有 `.cu`/`.cpp` 编译为一个共享库 `vllm/_C.abi3.so`
2. **torch.library 注册** — 在 `csrc/torch_bindings.cpp` 中用 `TORCH_LIBRARY` 宏将 C++ 函数注册为 PyTorch 算子
3. **Python 导入** — `import vllm._C` 加载 .so，算子自动注册到 `torch.ops._C` 命名空间
4. **Python 调用** — 通过 `torch.ops._C.xxx()` 直接调用

---

## 具体例子：`silu_and_mul` 完整链路

### 第 1 层：CUDA Kernel

文件：`csrc/activation_kernels.cu`

```cuda
template <typename scalar_t, typename packed_t, ...>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d) {
  // 每个线程处理一个元素，计算 silu(x) * y
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&x_ptr[idx]);
    const scalar_t y = VLLM_LDG(&y_ptr[idx]);
    out_ptr[idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}
```

### 第 2 层：C++ Wrapper

文件：`csrc/activation_kernels.cu`

```cpp
void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // 接收 torch::Tensor，启动 CUDA kernel
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel,
                                vllm::packed_silu_kernel, true);
}
```

### 第 3 层：torch.library 注册

文件：`csrc/torch_bindings.cpp`

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // 定义算子签名（schema）
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  // 绑定 CUDA 实现
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
}
```

关键点：`TORCH_EXTENSION_NAME` 会被 CMake 展开为 `_C`，所以算子注册在 `torch.ops._C` 命名空间下。

### 第 4 层：Python 调用

文件：`vllm/_custom_ops.py`

```python
def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul(out, input)
```

---

## 构建系统

### setup.py 定义 CMake 扩展

```python
ext_modules = []
if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._C"))
    ext_modules.append(CMakeExtension(name="vllm._moe_C"))
```

### CMakeLists.txt 列出所有源文件并编译

```cmake
set(VLLM_EXT_SRC
  "csrc/activation_kernels.cu"
  "csrc/cache_kernels.cu"
  "csrc/attention/paged_attention_v1.cu"
  "csrc/attention/paged_attention_v2.cu"
  "csrc/layernorm_kernels.cu"
  ...
  "csrc/torch_bindings.cpp")   # ← 注册入口

# 编译为 Python 扩展模块
define_gpu_extension_target(_C ...)
```

### csrc/core/registration.h 提供模块初始化宏

```cpp
#define REGISTER_EXTENSION(NAME)
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() { ... }
```

---

## 模块加载入口

`vllm/platforms/interface.py`:

```python
@classmethod
def import_kernels(cls) -> None:
    try:
        import vllm._C        # 加载编译好的 .so，触发 TORCH_LIBRARY 注册
    except ImportError as e:
        logger.warning("Failed to import from vllm._C: %r", e)
```

在 `_custom_ops.py` 开头调用：

```python
current_platform.import_kernels()  # 之后 torch.ops._C.* 即可使用
```

---

## 支持 torch.compile 的 fake 实现

为了让 `torch.compile` 能做形状推导，还需注册 "fake" 实现：

```python
@register_fake("_C::awq_dequantize")
def _awq_dequantize_fake(qweight, scales, zeros, split_k_iters, thx, thy):
    # 只描述输出 shape/dtype，不做实际计算
    return torch.empty((in_c, out_c), dtype=scales.dtype, device=scales.device)
```

---

## 关键文件索引

| 层次 | 文件 | 职责 |
|------|------|------|
| CUDA Kernel | `csrc/*.cu` | GPU 并行计算逻辑 |
| C++ Wrapper | `csrc/*.cu` / `csrc/ops.h` | 将 `torch::Tensor` 映射为 kernel 参数并 launch |
| 算子注册 | `csrc/torch_bindings.cpp` | 用 `torch.library` 注册到 `torch.ops._C` |
| 编译构建 | `CMakeLists.txt` + `setup.py` | CMake 编译所有源码为 `_C.so` |
| Python 封装 | `vllm/_custom_ops.py` | 提供 Python API，调用 `torch.ops._C.*` |

---

## 总结

核心机制是 **PyTorch 的 `torch.library` 自定义算子系统**，它替代了传统的 pybind11 方式，天然支持 `torch.compile`、autograd、以及多设备分发（CUDA/CPU/ROCm）。

编译产物：

- `vllm/_C.abi3.so` — 主 CUDA 算子库
- `vllm/_moe_C.abi3.so` — MoE 专用算子库
- `vllm/_rocm_C.abi3.so` — ROCm 专用算子库
