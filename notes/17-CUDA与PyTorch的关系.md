# CUDA 与 PyTorch 的关系

## 概述

CUDA 和 PyTorch 是**不同层级**的技术，它们之间是**底层基础设施与上层框架**的关系：

- **CUDA** = GPU 的"操作系统 + 编译器 + 标准库"
- **PyTorch** = 基于 CUDA 构建的深度学习框架

---

## 1. 层级架构

```
┌─────────────────────────────────────────────┐
│           用户代码 / vLLM                     │  ← 你写的 Python 代码
├─────────────────────────────────────────────┤
│           PyTorch (torch)                    │  ← 自动微分、张量抽象、设备管理
├─────────────────────────────────────────────┤
│           ATen / C++ Tensor Library          │  ← PyTorch 的 C++ 后端
├──────────────┬──────────────────────────────┤
│   cuDNN      │   cuBLAS    │   其他 CUDA 库  │  ← CUDA 加速库（卷积、矩阵乘法等）
├──────────────┴──────────────────────────────┤
│           CUDA Runtime API                   │  ← 内存管理、kernel 启动、流同步
├─────────────────────────────────────────────┤
│           CUDA Driver API                    │  ← 底层驱动接口
├─────────────────────────────────────────────┤
│           GPU 硬件 (NVIDIA)                   │  ← SM、显存、Tensor Core
└─────────────────────────────────────────────┘
```

---

## 2. CUDA 提供了什么

| 组件 | 作用 | 对应 PyTorch 中的使用 |
|---|---|---|
| **CUDA Toolkit** | 编译器（nvcc）、运行时库 | PyTorch 编译 C++/CUDA 扩展时使用 |
| **cuDNN** | 高度优化的卷积/RNN/注意力 kernel | `torch.nn.Conv2d` 底层调用 cuDNN |
| **cuBLAS** | 矩阵乘法（GEMM） | `torch.matmul` / `torch.mm` 底层调用 cuBLAS |
| **CUDA Runtime** | `cudaMalloc`、`cudaMemcpy`、流管理 | `torch.cuda.memory_allocated()`、`.to("cuda")` |
| **CUDA Graphs** | 捕获并重放一系列 GPU 操作 | `torch.cuda.CUDAGraph` |
| **NCCL** | 多 GPU 集合通信（AllReduce 等） | `torch.distributed` 底层通信 |

### 关键点

CUDA 本身**不关心**你在做深度学习还是科学计算。它只提供：
- 在 GPU 上分配/释放内存
- 启动并行计算 kernel
- GPU 间数据传输
- 多 GPU 通信

---

## 3. PyTorch 在 CUDA 之上提供了什么

| 能力 | 说明 |
|---|---|
| **Tensor 抽象** | 统一的多维数组 API，屏蔽 CPU/GPU 差异 |
| **自动微分 (Autograd)** | 自动计算梯度，构建计算图 |
| **设备透明** | 同一份代码 `.to("cuda")` 即可切换到 GPU |
| **内存管理** | 基于 CUDA 的缓存分配器（caching allocator），减少 `cudaMalloc` 开销 |
| **算子分发** | 根据张量设备/数据类型自动选择最优 kernel（cuBLAS、cuDNN 或自定义） |
| **torch.compile** | JIT 编译优化，融合算子，减少 kernel 启动开销 |
| **分布式训练** | 封装 NCCL，提供 DDP、FSDP 等高级 API |

---

## 4. 在 vLLM 中的体现

### 4.1 PyTorch 层面

```python
# 张量操作 — PyTorch 自动调用 cuBLAS
hidden_states = model(input_ids)  # 内部大量 torch.matmul → cuBLAS GEMM

# 设备管理
model.to("cuda:0")
input_ids = input_ids.cuda()

# 内存查询 — PyTorch 封装了 CUDA Runtime 的内存 API
free_mem = torch.cuda.mem_get_info()[0]
```

### 4.2 CUDA 层面（PyTorch 背后）

```python
# CUDA Graphs — vLLM 用它来减少 kernel 启动开销
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(captured_input)
# 之后只需 graph.replay()，跳过 Python 开销

# 自定义 CUDA kernel — vLLM 的 PagedAttention
# 直接用 CUDA C++ 编写，通过 PyTorch 的 C++ 扩展机制加载
from vllm._C import ops
ops.paged_attention_v1(...)  # 调用自定义 CUDA kernel

# NCCL 通信 — 张量并行
torch.distributed.all_reduce(tensor)  # 底层调用 NCCL
```

### 4.3 vLLM 中的典型调用链

```
vLLM Python 代码
    → PyTorch API (torch.matmul, F.scaled_dot_product_attention)
        → ATen C++ 分发
            → cuBLAS (矩阵乘法) / FlashAttention CUDA kernel (注意力)
                → CUDA Runtime (启动 kernel, 管理显存)
                    → GPU 硬件执行
```

---

## 5. 类比理解

| 类比 | CUDA | PyTorch |
|---|---|---|
| 操作系统 vs 应用 | CUDA ≈ GPU 的操作系统 | PyTorch ≈ 运行在其上的应用框架 |
| C 语言 vs Python | CUDA C++ ≈ 底层语言 | PyTorch ≈ 高级语言 |
| 汇编 vs 编译器 | CUDA kernel ≈ 手写汇编 | PyTorch 算子 ≈ 编译器自动生成的代码 |

---

## 6. 一句话总结

> **CUDA 是 NVIDIA GPU 的编程平台（底层），PyTorch 是基于 CUDA 构建的深度学习框架（上层）。PyTorch 让你用 Python 写 `torch.matmul(A, B)`，背后自动调用 CUDA 的 cuBLAS 在 GPU 上执行矩阵乘法。**

---

# vLLM、PyTorch、CUDA 三者的关系

## 7. 完整层级架构

```
┌───────────────────────────────────────────────────────┐
│                    用户请求 (HTTP/gRPC)                 │
├───────────────────────────────────────────────────────┤
│                      vLLM                              │
│  ┌─────────────┬──────────────┬──────────────────┐    │
│  │ API Server  │  Scheduler   │  KV Cache Manager │    │
│  │ (FastAPI)   │  (请求调度)   │  (显存页表管理)    │    │
│  ├─────────────┴──────────────┴──────────────────┤    │
│  │              GPUModelRunner                     │    │
│  │  (输入准备 → 模型前向 → 采样 → 输出)             │    │
│  ├───────────────────────────────────────────────┤    │
│  │  自定义 CUDA Kernels (C++/CUDA 扩展)            │    │
│  │  PagedAttention, RotaryEmbedding, Quantize...  │    │
│  └───────────────────────────────────────────────┘    │
├───────────────────────────────────────────────────────┤
│                     PyTorch                             │
│  Tensor 抽象 | Autograd | 算子分发 | 内存管理            │
│  torch.compile | CUDAGraph | torch.distributed         │
├──────────────┬────────────────┬───────────────────────┤
│   cuBLAS     │    cuDNN       │    NCCL / CUDA Runtime │
├──────────────┴────────────────┴───────────────────────┤
│                   CUDA Driver                          │
├───────────────────────────────────────────────────────┤
│                GPU 硬件 (NVIDIA)                        │
└───────────────────────────────────────────────────────┘
```

---

## 8. vLLM 依赖 PyTorch 做什么

| 用途 | vLLM 怎么用 | PyTorch 提供什么 |
|---|---|---|
| **模型加载** | `initialize_model()` 创建模型实例 | `nn.Module`、`state_dict`、权重反序列化 |
| **张量运算** | 前向推理中的矩阵乘法、激活函数 | `torch.matmul` → cuBLAS，`F.silu` 等 |
| **显存管理** | KV Cache 分配、模型权重存放 | `torch.empty()`、caching allocator |
| **设备管理** | 多 GPU 分配 | `torch.cuda.set_device()`、`tensor.to(device)` |
| **CUDA Graphs** | 捕获推理计算图，消除 Python 开销 | `torch.cuda.CUDAGraph`、`graph.replay()` |
| **分布式通信** | 张量并行、流水线并行 | `torch.distributed`（底层 NCCL） |
| **自定义算子注册** | 注册 PagedAttention 等自定义 kernel | `torch.library`、C++ 扩展机制 |

**一句话：vLLM 把 PyTorch 当作 GPU 编程的"标准库"使用。**

---

## 9. vLLM 直接使用 CUDA 做什么（绕过 PyTorch）

PyTorch 的通用算子无法满足 LLM 推理的所有性能需求，vLLM 在关键路径上直接编写 CUDA kernel：

| 自定义 CUDA Kernel | 为什么不用 PyTorch 自带的 |
|---|---|
| **PagedAttention** | PyTorch 没有分页式 KV Cache 注意力 |
| **FlashAttention / FlashMLA** | 需要 IO-aware 的融合注意力 kernel |
| **量化 kernel** (AWQ, GPTQ, FP8) | PyTorch 不原生支持这些量化格式的高效计算 |
| **Rotary Embedding** | 融合版本比 PyTorch 拼接实现快数倍 |
| **Fused MoE** | 专家混合路由+计算需要深度融合 |
| **采样 kernel** | Top-k/Top-p 采样需要 GPU 端高效实现 |

这些 kernel 的调用链：

```
vLLM Python → torch.ops.vllm.paged_attention(...)
                  ↓
              PyTorch 算子分发（只做桥接，不做计算）
                  ↓
              vLLM 的 C++/CUDA 扩展 (csrc/ 目录)
                  ↓
              CUDA Runtime → GPU 硬件
```

---

## 10. 三者的依赖关系

```
vLLM ──依赖──→ PyTorch ──依赖──→ CUDA
  │                                  ↑
  └────────── 直接调用 ──────────────┘
             (自定义 CUDA kernel)
```

- **CUDA** 是最底层，提供 GPU 编程能力
- **PyTorch** 在 CUDA 之上，提供张量计算框架
- **vLLM** 在 PyTorch 之上，但同时也直接调用 CUDA

这是一个**"双层依赖"**模式：vLLM 既通过 PyTorch 间接使用 CUDA，也直接编写 CUDA kernel。

---

## 11. 在 vLLM 代码中的具体体现

### 11.1 通过 PyTorch（间接 CUDA）

```python
# GPUModelRunner.load_model — 通过 PyTorch 加载模型
with set_default_torch_dtype(model_config.dtype):
    with target_device:  # torch.device("cuda")
        model = initialize_model(...)  # nn.Module 子类

# 前向推理 — PyTorch 自动调度到 cuBLAS
hidden_states = model(input_ids, positions, kv_caches, ...)

# CUDA Graphs 捕获 — PyTorch 封装
with torch.cuda.graph(self.graph):
    output = model(captured_inputs)
self.graph.replay()  # 重放，零 Python 开销

# 多 GPU 通信 — PyTorch distributed
torch.distributed.all_reduce(tensor, group=tp_group)
```

### 11.2 直接 CUDA（绕过 PyTorch 通用算子）

```python
# PagedAttention — vLLM 自定义 CUDA kernel
from vllm._C import ops
ops.paged_attention_v1(
    output, query, key_cache, value_cache,
    head_mapping, scale, block_tables, context_lens, ...)

# FlashAttention
from vllm.vllm_flash_attn import flash_attn_varlen_func
output = flash_attn_varlen_func(q, k, v, ...)

# 量化计算
ops.awq_gemm(input, qweight, qzeros, scales, ...)
```

---

## 12. 为什么 vLLM 不能只用 PyTorch

| 原因 | 说明 |
|---|---|
| **PagedAttention** | LLM 推理的核心创新，PyTorch 没有 |
| **连续批处理 (Continuous Batching)** | 需要自定义内存管理和调度，PyTorch 不提供 |
| **极致性能** | 通用算子有额外开销，自定义 kernel 可针对 LLM 推理深度优化 |
| **量化推理** | INT4/INT8/FP8 量化格式需要专用 kernel |
| **Speculative Decoding** | 投机解码需要特殊的调度和验证逻辑 |

---

## 13. 为什么 vLLM 不能抛弃 PyTorch

| 原因 | 说明 |
|---|---|
| **模型生态** | HuggingFace 上所有模型权重都是 PyTorch 格式 |
| **nn.Module 体系** | 模型定义、权重管理、序列化都依赖 PyTorch |
| **CUDA Graphs** | PyTorch 提供了最方便的 CUDA Graph 捕获 API |
| **分布式通信** | `torch.distributed` 封装了 NCCL，开箱即用 |
| **内存管理** | PyTorch 的 caching allocator 比直接 `cudaMalloc` 高效得多 |
| **开发效率** | 用 Python + PyTorch 写上层逻辑，只在瓶颈处写 CUDA |

---

## 14. 一句话总结

> **vLLM 是一个 LLM 推理引擎，它以 PyTorch 为骨架（模型定义、张量管理、分布式通信），以自定义 CUDA kernel 为利刃（PagedAttention、量化计算、融合算子），两者结合实现了远超 PyTorch 原生推理的性能。**

---

# CUDA Graphs 详解

## 15. 问题背景：为什么需要 CUDA Graphs

正常的 GPU 执行流程中，每个 CUDA 操作都要经过 CPU 端的"启动开销"：

```
普通模式（每次推理）：
CPU: [准备kernel0] → [启动kernel0] → [准备kernel1] → [启动kernel1] → [准备kernel2] → ...
GPU:                  [==执行kernel0==]                [==执行kernel1==]                [==执行kernel2==]
                      ↑                                ↑
                      等CPU准备好才能开始                  又在等CPU
```

每次 `kernel launch` 的 CPU 开销大约 **5~10 微秒**。一次 Transformer forward pass 可能包含 **几百个 kernel**（矩阵乘法、LayerNorm、激活函数、Attention...），累计的 CPU 端开销可达 **数毫秒**。

对于 LLM 推理的 decode 阶段，每个 token 只需 GPU 计算 **几毫秒**，这时 CPU 启动开销可能占到总时间的 **30%~50%**。

---

## 16. CUDA Graphs 的核心思想

**一次录制，反复重放。**

```
录制阶段（只做一次）：
CPU: [录制 kernel0] → [录制 kernel1] → [录制 kernel2] → ... → [生成Graph]

重放阶段（每次推理）：
CPU: [graph.replay()]  ← 一次调用
GPU: [==kernel0==][==kernel1==][==kernel2==]...  ← 所有kernel连续执行，无间断
```

类比理解：
- **普通模式** = 乐队指挥逐个提示每个乐手何时演奏
- **CUDA Graphs** = 指挥提前排练好，演出时只需按下"播放"，所有乐手自动按编排演奏

---

## 17. 工作原理

### 17.1 录制（Capture）

```python
# 1. 准备固定大小的输入 buffer（地址不能变！）
static_input = torch.empty(batch_size, seq_len, device="cuda")

# 2. 预热：先跑一遍让 CUDA 完成各种初始化
model(static_input)

# 3. 录制
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model(static_input)  # 不真正计算，只记录操作序列
```

录制期间，CUDA 运行时拦截所有 GPU 操作，记录为一个**有向无环图（DAG）**：

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ MatMul_0 │────→│ LayerNorm│────→│  SiLU    │
  └──────────┘     └──────────┘     └──────────┘
       │                                  │
       │           ┌──────────┐           │
       └──────────→│ MatMul_1 │←──────────┘
                   └──────────┘
                        │
                   ┌──────────┐
                   │ Softmax  │
                   └──────────┘
```

图中记录了：
- 每个 kernel 的**函数指针**
- 每个 kernel 的**参数**（包括张量的**内存地址**）
- kernel 之间的**依赖关系**
- 每个 kernel 的**线程块配置**（grid/block size）

### 17.2 重放（Replay）

```python
# 每次推理：
static_input.copy_(real_input)   # 把真实数据拷贝到录制时的 buffer
graph.replay()                    # 一次 CPU 调用，GPU 连续执行所有 kernel
result = static_output.clone()    # 从录制时的输出 buffer 取结果
```

重放时 CPU 只需一次调用（~微秒级），GPU 端所有 kernel 按预设顺序连续执行，没有 CPU 介入的间隔。

---

## 18. 关键约束

| 约束 | 原因 | 影响 |
|---|---|---|
| **张量地址固定** | Graph 中记录的是内存地址，不是变量名 | 必须用固定的 static buffer，通过 `copy_` 填充数据 |
| **计算图固定** | 录制后不能改变 kernel 序列 | 不能有 if/else 分支、不能改变张量形状 |
| **不能分配/释放内存** | `cudaMalloc`/`cudaFree` 不可录制 | 所有张量必须预分配 |
| **不能做 CPU-GPU 同步** | `torch.cuda.synchronize()` 等会打断流 | 录制区间内不能有同步操作 |
| **不能做主机端操作** | `print()`、`tensor.item()` 等会触发同步 | 录制区间内只能有纯 GPU 操作 |

---

## 19. 在 vLLM 中的使用

### 19.1 为什么 vLLM 需要 CUDA Graphs

LLM 推理分两个阶段：

| 阶段 | 特点 | CUDA Graphs |
|---|---|---|
| **Prefill（首次编码）** | 处理长序列，GPU 计算密集，kernel 启动开销占比小 | 通常不用 |
| **Decode（逐 token 生成）** | 每次只处理 1 个 token，GPU 计算量小，kernel 启动开销占比大 | **非常适合** |

### 19.2 vLLM 的 CUDA Graph 捕获策略

```python
# vllm/v1/worker/gpu_model_runner.py 中的实现

# 1. 预先为不同 batch size 各捕获一个 Graph
#    例如 batch_size = 1, 2, 4, 8, 16, 32, ...
for batch_size in cudagraph_batch_sizes:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(static_input[:batch_size], ...)
    self.cudagraph_graphs[batch_size] = graph

# 2. 推理时选择匹配的 Graph
def execute_model(self, ...):
    # 向上取整到最近的预捕获 batch size
    padded_batch_size = self._get_padded_batch_size(actual_batch_size)

    # 填充输入数据到 static buffer
    self.input_ids[:padded_batch_size].copy_(real_input_ids)

    # 重放
    self.cudagraph_graphs[padded_batch_size].replay()
```

### 19.3 三种编译模式

vLLM 支持三种 CUDA Graph 模式：

```
CompilationLevel.NO_COMPILATION     → 不使用 CUDA Graphs（纯eager模式）
CompilationLevel.DYNAMO_ONCE        → 使用 torch.compile + CUDA Graphs
CompilationLevel.PIECEWISE          → 分段捕获（处理不可捕获的操作）
```

**PIECEWISE 模式**（分段捕获）解决了"计算图必须固定"的约束：

```
完整 forward pass：
[可捕获region0] → [不可捕获op] → [可捕获region1] → [不可捕获op] → [可捕获region2]

拆分为多个小 Graph：
Graph_0: [region0 的所有 kernel]
Graph_1: [region1 的所有 kernel]
Graph_2: [region2 的所有 kernel]

执行时：
graph_0.replay() → 执行不可捕获op → graph_1.replay() → 执行不可捕获op → graph_2.replay()
```

---

## 20. 性能收益

```
以 Llama-7B decode 阶段为例（batch_size=1, 单个 token）：

无 CUDA Graphs:
  GPU 计算: ~3ms
  CPU 开销: ~2ms (约400个kernel × 5μs)
  总延迟:   ~5ms

有 CUDA Graphs:
  GPU 计算: ~3ms
  CPU 开销: ~0.01ms (1次 replay 调用)
  总延迟:   ~3ms

加速比: ~1.67x
```

batch size 越小，加速越明显（因为 GPU 计算量小时 CPU 开销占比更高）。

---

## 21. CUDA Graphs 的内存代价

```
每个 Graph 需要额外内存：
  - 录制时所有中间张量都不会被释放（Graph 持有引用）
  - 每个 batch_size 各一份 Graph = 各一份中间张量

例如 Llama-7B:
  - 单个 Graph 中间张量: ~200MB
  - 预捕获 10 种 batch_size: ~2GB 额外显存

vLLM 的优化：
  - 使用 CUDA Graph Pool 让多个 Graph 共享内存区域
  - graph = torch.cuda.CUDAGraph()
  - with torch.cuda.graph(graph, pool=self.cudagraph_pool):  # 共享 pool
```

---

## 22. 完整生命周期图

```
                    vLLM 启动
                       │
              ┌────────▼────────┐
              │  加载模型权重     │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  分配 KV Cache   │
              └────────┬────────┘
                       │
              ┌────────▼────────────────┐
              │  CUDA Graph 预捕获       │
              │  for bs in [1,2,4,8...]: │
              │    warmup(bs)            │  ← 预热
              │    capture(bs)           │  ← 录制
              └────────┬────────────────┘
                       │
          ═════════════╪═══════ 服务就绪 ═══════
                       │
              ┌────────▼────────┐
              │  收到推理请求     │
              └────────┬────────┘
                       │
              ┌────────▼──────────────────┐
              │  判断能否使用 CUDA Graph    │
              │  ① batch_size ≤ 最大捕获值  │
              │  ② 是 decode 阶段          │
              │  ③ 无动态形状操作           │
              └────┬──────────┬───────────┘
                   │          │
               能用 │          │ 不能用
                   │          │
          ┌────────▼───┐  ┌───▼──────────┐
          │ pad → copy_ │  │ 普通 eager    │
          │ → replay()  │  │ forward pass  │
          │ → 取输出     │  │              │
          └────────────┘  └──────────────┘
```

---

## 23. CUDA Graphs 一句话总结

> **CUDA Graphs 通过"一次录制、反复重放"消除了 GPU kernel 的 CPU 端启动开销，对 LLM 推理的 decode 阶段（每次只生成一个 token、计算量小但 kernel 数量多）效果尤为显著，是 vLLM 实现低延迟的关键技术之一。**
