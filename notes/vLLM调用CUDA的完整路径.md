# vLLM 调用 CUDA 的完整路径

本文以 **Paged Attention** 为例，追踪 vLLM 从 Python 到 GPU 执行的完整调用链，共经过 **6 层**。

---

## 调用链总览

```
① Python 模型层 (Attention forward)
    │
    ▼
② Python 自定义算子包装 (vllm/_custom_ops.py)
    │  torch.ops._C.paged_attention_v1(...)
    ▼
③ PyTorch Dispatcher (torch.library 机制)
    │  查找 torch::kCUDA 注册的实现
    ▼
④ C++ 算子注册层 (csrc/torch_bindings.cpp)
    │  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1)
    ▼
⑤ CUDA Launcher (csrc/attention/paged_attention_v1.cu)
    │  模板分发 → kernel<<<grid, block, shared_mem, stream>>>(...)
    ▼
⑥ CUDA Kernel 执行 (csrc/attention/attention_kernels.cuh)
       __device__ void paged_attention_kernel(...)  在 GPU 上并行计算
```

---

## 第 ① 层：Python 模型层

模型的 Attention 层在 forward 时调用自定义算子：

`vllm/v1/attention/backends/tree_attn.py` 等后端文件中调用 `ops.paged_attention_v1()`。

---

## 第 ② 层：Python 自定义算子包装

**文件**：`vllm/_custom_ops.py`

```python
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    # ... 更多参数
) -> None:
    torch.ops._C.paged_attention_v1(    # ← 关键：调用注册的 torch op
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, ...
    )
```

`torch.ops._C` 中的 `_C` 对应编译产物 `vllm/_C.cpython-*.so`。

---

## 第 ③ 层：PyTorch Dispatcher 机制

当 Python 调用 `torch.ops._C.paged_attention_v1()` 时，PyTorch 内部的 Dispatcher 根据输入张量所在的设备（CUDA），查找通过 `TORCH_LIBRARY` 注册的对应实现。

**扩展库加载入口**：`vllm/platforms/cuda.py`

```python
import vllm._C  # noqa   ← 加载编译好的 .so，触发 op 注册
```

---

## 第 ④ 层：C++ 算子注册

**文件**：`csrc/torch_bindings.cpp`

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // 定义算子签名
    ops.def(
        "paged_attention_v1("
        "    Tensor! out, Tensor query, Tensor key_cache,"
        "    Tensor value_cache, int num_kv_heads, float scale,"
        "    Tensor block_tables, Tensor seq_lens, int block_size,"
        "    int max_seq_len, Tensor? alibi_slopes,"
        "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
        "    ...) -> ()");

    // 绑定 CUDA 实现
    ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);
}
```

函数声明在 `csrc/ops.h` 中，实现在各 `.cu` 文件中。

---

## 第 ⑤ 层：CUDA Launcher（模板分发 + 内核启动）

**文件**：`csrc/attention/paged_attention_v1.cu`

```cpp
// 宏：展开为实际的 CUDA kernel 调用
#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                              \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,     \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>\
      <<<grid, block, shared_mem_size, stream>>>(                        \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, ...)

template <typename T, typename CACHE_T, int BLOCK_SIZE, ...>
void paged_attention_v1_launcher(...) {
    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(NUM_THREADS);           // 通常 128

    switch (head_size) {               // 按 head_size 分发到不同模板特化
        case 64:  LAUNCH_PAGED_ATTENTION_V1(64);  break;
        case 128: LAUNCH_PAGED_ATTENTION_V1(128); break;
        // ...更多 head_size
    }
}

// 顶层入口：按 KV cache 数据类型分发
void paged_attention_v1(...) {
    DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                               CALL_V1_LAUNCHER_BLOCK_SIZE)
}
```

**分发链**：数据类型 → block_size → head_size → 启动对应的模板特化内核。

---

## 第 ⑥ 层：CUDA Kernel（GPU 上实际执行）

**文件**：`csrc/attention/attention_kernels.cuh`

```cuda
template <typename scalar_t, typename cache_t, int HEAD_SIZE,
          int BLOCK_SIZE, int NUM_THREADS, ...>
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache, ...)
{
    // 1. 线程索引映射
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    // 2. 将 Query 加载到寄存器/共享内存
    __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

    // 3. 遍历 KV Cache 的 block，计算 Q·K
    for (int block_idx = ...; block_idx < end_block_idx; ...) {
        float qk = scale * Qk_dot<scalar_t, ...>::dot(q_vecs[...], k_vecs);
    }

    // 4. Softmax 归一化
    float exp_sum = 0.f;
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

    // 5. 加权求和 V，写出结果
    for (int block_idx = ...) {
        accs[i] += dot(logits_vec, v_vec);
    }
}
```

---

## 编译构建

**文件**：`CMakeLists.txt`

```cmake
set(VLLM_EXT_SRC
  "csrc/torch_bindings.cpp"              # 算子注册
  "csrc/attention/paged_attention_v1.cu"  # Launcher
  "csrc/attention/paged_attention_v2.cu"
  "csrc/attention/attention_kernels.cuh"  # 实际内核
  "csrc/cache_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  # ... 更多 .cu 文件
)

define_extension_target(
  _C                       # 编译产物名 → vllm/_C.cpython-*.so
  DESTINATION vllm
  LANGUAGE CUDA
  SOURCES ${VLLM_EXT_SRC}
)
```

`nvcc` 编译 `.cu` → `g++` 编译 `.cpp` → 链接为 `vllm/_C.so` → Python `import vllm._C` 加载。

---

## 关键目录结构

```
vllm/
├── _custom_ops.py              ← ② Python 包装层
├── platforms/
│   └── cuda.py                 ← ③ import vllm._C 触发注册
│
csrc/
├── ops.h                       ← ④ C++ 函数声明
├── torch_bindings.cpp          ← ④ TORCH_LIBRARY 算子注册
├── core/registration.h         ← TORCH_LIBRARY_EXPAND 宏定义
├── attention/
│   ├── paged_attention_v1.cu   ← ⑤ Launcher (模板分发+内核启动)
│   ├── paged_attention_v2.cu   ← ⑤ Partitioned attention
│   └── attention_kernels.cuh   ← ⑥ 实际 CUDA kernel
├── activation_kernels.cu       ← 激活函数 kernel
├── layernorm_kernels.cu        ← LayerNorm kernel
├── cache_kernels.cu            ← KV Cache 管理 kernel
└── ...
```

---

## 核心设计要点

### 1. 零拷贝

Python tensor 直接通过 `.data_ptr()` 传递给 C++，无中间拷贝。

### 2. 编译期模板特化

所有 (数据类型 × head_size × block_size) 组合在编译时生成，运行时 switch 分发，避免运行期开销。

### 3. PyTorch Dispatcher 统一路由

同一个 op 名可注册 CUDA / CPU / ROCm 等不同实现，运行时按设备自动选择。

### 4. Kernel 启动参数

- **Grid**：`(num_heads, num_seqs, max_partitions)` — 每个 head × 每个序列 = 一个 thread block
- **Block**：`(NUM_THREADS,)` — 通常 128 或 256 个线程
- **Shared Memory**：根据注意力上下文窗口大小动态分配

### 5. Warp 级优化

- **Shuffle 操作**：线程间高速数据交换（`VLLM_SHFL_*`），避免走共享内存
- **共享内存归约**：并行 softmax 计算
- **合并访存**：精心设计的内存访问模式，最大化显存带宽利用率

### 6. 量化支持

内核通过模板参数支持多种量化格式，推理时即时反量化：

```cpp
if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
    // 无量化：直接读取
    k_vecs[j] = *reinterpret_cast<const K_vec*>(...);
} else {
    // FP8 量化：读取后即时反量化
    Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(...);
    k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
        k_vec_quant, *k_scale);
}
```

---

## 其他 CUDA Kernel 列表

vLLM 中除了 Attention 之外，还有大量 CUDA 内核，调用路径结构相同：

| Kernel 文件 | 功能 |
|-------------|------|
| `csrc/attention/paged_attention_v1.cu` | PagedAttention V1（单阶段） |
| `csrc/attention/paged_attention_v2.cu` | PagedAttention V2（分区并行） |
| `csrc/attention/merge_attn_states.cu` | 合并多个注意力分区的结果 |
| `csrc/activation_kernels.cu` | SiLU、GELU 等激活函数 |
| `csrc/layernorm_kernels.cu` | RMSNorm、LayerNorm |
| `csrc/cache_kernels.cu` | KV Cache 的 reshape/copy/swap |
| `csrc/pos_encoding_kernels.cu` | RoPE 旋转位置编码 |
| `csrc/quantization/` | GPTQ、AWQ、FP8 等量化算子 |
| `csrc/moe/` | MoE（混合专家）路由和计算 |

所有这些内核均遵循相同的调用路径：

```
Python (_custom_ops.py)
  → torch.ops._C.xxx()
    → torch_bindings.cpp 注册
      → .cu Launcher 模板分发
        → .cuh CUDA Kernel 执行
```

---

## 附录：什么是 PagedAttention

PagedAttention 是 vLLM 的核心创新，它借鉴了操作系统**虚拟内存分页**的思想来管理 KV Cache，解决了 LLM 推理中显存浪费严重的问题。

### 1. 问题背景：传统 KV Cache 的显存浪费

LLM 在生成文本时，需要缓存每个 token 的 Key 和 Value 向量（即 KV Cache），以避免重复计算。传统做法是为每个请求**预分配一块连续的显存**来存放 KV Cache。

```
传统方式：为每个请求预分配连续显存

请求 A（实际生成了 3 个 token，预分配了 8 个位置）：
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ K1 │ K2 │ K3 │空闲│空闲│空闲│空闲│空闲│  ← 62.5% 显存浪费
└────┴────┴────┴────┴────┴────┴────┴────┘

请求 B（实际生成了 6 个 token，预分配了 8 个位置）：
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ K1 │ K2 │ K3 │ K4 │ K5 │ K6 │空闲│空闲│  ← 25% 浪费
└────┴────┴────┴────┴────┴────┴────┴────┘
```

这种方式有三大问题：

| 问题 | 说明 |
|------|------|
| **内部碎片** | 预分配空间未用满，大量显存空闲浪费 |
| **外部碎片** | 请求结束释放后，显存出现不连续的"空洞"，无法被新请求利用 |
| **过度预留** | 因为不知道序列最终多长，必须按最大长度预分配 |

实测显示，传统方式下 **60%~80% 的 KV Cache 显存被浪费**。

### 2. PagedAttention 的解决方案

PagedAttention 的核心思想：**不再预分配连续显存，而是将 KV Cache 拆分成固定大小的"页"（Block），按需动态分配**。

#### 类比操作系统虚拟内存

| 操作系统概念 | PagedAttention 对应 |
|-------------|-------------------|
| 虚拟地址空间 | 序列的逻辑 KV Cache |
| 物理内存页 | GPU 显存中的 KV Block |
| 页表 | Block Table（映射表） |
| 按需分页 | 每生成一个 token，才分配新的 Block |
| 页大小 | Block Size（通常 16 个 token） |

#### 工作方式

```
PagedAttention：按需分配非连续的 Block

GPU 显存（物理层）：
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│Blk 0 │Blk 1 │Blk 2 │Blk 3 │Blk 4 │Blk 5 │Blk 6 │Blk 7 │
│ A-p0 │ B-p0 │ A-p1 │ 空闲  │ B-p1 │ A-p2 │ 空闲  │ B-p2 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  ↑              ↑              ↑       ↑
  └──────────────┴──────────────┘       │
  请求 A 的 Block 分散存放               │
                                        │
        请求 B 的 Block 也分散存放 ───────┘

Block Table（页表）：
┌─────────┬────────────────────────┐
│ 请求 A  │ [Blk0, Blk2, Blk5]    │  ← 逻辑连续，物理不连续
│ 请求 B  │ [Blk1, Blk4, Blk7]    │
└─────────┴────────────────────────┘
```

每个 Block 存放固定数量（如 16 个）token 的 Key 和 Value：

```
一个 KV Block 的结构（Block Size = 16）：

Key Block:
┌──────────────────────────────────────────────┐
│ K_token0  K_token1  K_token2 ... K_token15   │  shape: [num_heads, block_size, head_dim]
└──────────────────────────────────────────────┘

Value Block:
┌──────────────────────────────────────────────┐
│ V_token0  V_token1  V_token2 ... V_token15   │  shape: [num_heads, head_dim, block_size]
└──────────────────────────────────────────────┘
```

### 3. PagedAttention 的计算过程

传统 Attention 在一块连续内存上计算 Q·K 和 softmax(Q·K)·V。PagedAttention 需要"跨 Block"计算：

```
                         Block Table
Query (当前 token)        查找映射
    │                      │
    ▼                      ▼
┌───────┐          ┌──────────────┐
│   Q   │          │ [B0, B2, B5] │  ← 请求 A 的所有 KV Block 地址
└───┬───┘          └──────┬───────┘
    │                     │
    │    ┌────────────────┬┴───────────────┐
    │    ▼                ▼                ▼
    │  ┌──────┐      ┌──────┐        ┌──────┐
    │  │Blk 0 │      │Blk 2 │        │Blk 5 │
    │  │K0..K15│      │K16..K31│      │K32..K47│
    │  └──┬───┘      └──┬───┘        └──┬───┘
    │     │              │               │
    ▼     ▼              ▼               ▼
  ┌─────────────────────────────────────────┐
  │  对每个 Block 分别计算 Q·K              │
  │  score_blk0 = Q · [K0..K15]             │
  │  score_blk2 = Q · [K16..K31]            │
  │  score_blk5 = Q · [K32..K47]            │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  跨 Block 全局 Softmax                  │
  │  weights = softmax(所有 score 拼接)      │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  加权求和 V                              │
  │  out = Σ weights[i] · V[i]（跨 Block）  │
  └─────────────────────────────────────────┘
```

### 4. V1 与 V2 的区别

vLLM 实现了两个版本的 PagedAttention：

#### PagedAttention V1（单阶段）

```
一个 CUDA thread block 处理一个 (head, sequence) 对
  → 遍历该序列的所有 KV Block
  → 在同一个 thread block 内完成 softmax + 加权求和
  → 直接输出结果

适用于：序列较短的场景
```

#### PagedAttention V2（分区并行）

```
将序列的 KV Block 切分成多个 Partition（分区）
  → 每个 Partition 由一个独立的 thread block 并行处理
  → 每个 Partition 各自计算局部 softmax 和局部加权和
  → 最后通过 reduce kernel 合并所有 Partition 的结果

适用于：序列很长的场景（长序列需要更多并行度）
```

```
V2 分区并行示意：

序列有 6 个 KV Block，分成 2 个 Partition：

Partition 0 (thread block 0):       Partition 1 (thread block 1):
┌──────┬──────┬──────┐              ┌──────┬──────┬──────┐
│Blk 0 │Blk 1 │Blk 2 │              │Blk 3 │Blk 4 │Blk 5 │
└──────┴──────┴──────┘              └──────┴──────┴──────┘
    │ 局部 softmax + 加权和               │ 局部 softmax + 加权和
    ▼                                    ▼
┌────────────┐                      ┌────────────┐
│partial_out0│                      │partial_out1│
│partial_max0│                      │partial_max1│
│partial_sum0│                      │partial_sum1│
└──────┬─────┘                      └──────┬─────┘
       │                                   │
       └──────────────┬────────────────────┘
                      ▼
              ┌──────────────┐
              │ Reduce Kernel │  ← merge_attn_states.cu
              │ 合并各分区结果 │
              └──────┬───────┘
                     ▼
               最终 Attention 输出
```

### 5. 显存收益

| 指标 | 传统连续分配 | PagedAttention |
|------|------------|----------------|
| 内部碎片 | 大量（预分配未用满） | 极少（仅最后一个 Block 可能有） |
| 外部碎片 | 严重（释放后空洞） | 无（Block 级别复用） |
| 显存利用率 | ~20%-40% | **>95%** |
| 最大批处理量 | 受限于浪费 | 提升 **2-4 倍** |
| 序列长度预知 | 必须预估最大长度 | 不需要，按需分配 |

### 6. 额外能力：高效的内存共享

PagedAttention 的分页机制天然支持**跨请求共享 KV Block**，类似操作系统的共享内存/Copy-on-Write：

```
场景：Beam Search（束搜索），多个候选共享前缀

Beam 1:  "The cat sat on the" → [Blk0, Blk1, Blk2, Blk3_beam1]
Beam 2:  "The cat sat on the" → [Blk0, Blk1, Blk2, Blk3_beam2]
Beam 3:  "The cat sat on the" → [Blk0, Blk1, Blk2, Blk3_beam3]
                                   ↑     ↑     ↑
                                   共享前缀 Block（引用计数管理）
                                   只有最后的 Block 不同
```

同理适用于：
- **Prefix Caching**：相同 system prompt 的请求共享前缀 KV Cache
- **Parallel Sampling**：同一 prompt 的多次采样共享输入部分

### 7. 在 vLLM 中的代码位置

| 组件 | 文件 | 职责 |
|------|------|------|
| Block 管理器 | `vllm/core/block_manager.py` | 管理 Block 分配、释放、引用计数 |
| Block Table | `vllm/core/block/` | 维护逻辑 Block → 物理 Block 的映射表 |
| Cache 引擎 | `vllm/worker/cache_engine.py` | 在 GPU 显存中分配物理 Block 池 |
| Attention 后端 | `vllm/v1/attention/backends/` | 选择 V1/V2/FlashAttention 等实现 |
| CUDA Kernel | `csrc/attention/paged_attention_v1.cu` | V1 内核启动 |
| CUDA Kernel | `csrc/attention/paged_attention_v2.cu` | V2 内核启动（分区并行） |
| CUDA Kernel | `csrc/attention/attention_kernels.cuh` | 实际 GPU 计算逻辑 |
| CUDA Kernel | `csrc/attention/merge_attn_states.cu` | V2 分区结果合并 |
| Cache 操作 | `csrc/cache_kernels.cu` | Block 级别的 KV Cache 读写/交换 |

### 8. 一句话总结

> **PagedAttention = 操作系统的虚拟内存分页思想 + GPU 上的 Attention 计算**。通过将 KV Cache 拆分为固定大小的非连续 Block 并按需分配，将显存利用率从 ~20% 提升到 >95%，使同等硬件可服务 2-4 倍的并发请求。

---

## 附录：什么是 Attention（注意力机制）

Attention 是 Transformer 架构的核心组件，也是所有现代大语言模型（GPT、Claude、Gemini、Llama 等）的基础。它解决的根本问题是：**让模型在处理一个词时，能"看到"并"关注"序列中所有其他词，而不只是相邻的几个词**。

### 1. 直觉理解

阅读句子"**小明**把作业交给了**老师**，**他**非常满意"时，人类能瞬间判断"他"指的是"老师"而不是"小明"。这是因为大脑自动将注意力分配到了上下文中最相关的词上。

Attention 机制让模型做同样的事：

```
输入：小明 把 作业 交给了 老师 ， 他 非常 满意

处理"他"这个词时，模型的注意力分配：

  小明   把   作业   交给了   老师    ，    他   非常   满意
  0.15  0.02  0.05   0.08   0.55   0.01  0.04  0.03   0.07
                              ↑
                        注意力最高 → "他"最关注"老师"
```

### 2. 三个核心向量：Q、K、V

Attention 的计算基于三个向量，每个 token 都会生成这三个向量：

| 向量 | 全称 | 直觉含义 |
|------|------|---------|
| **Q**（Query） | 查询 | "我在找什么？" — 当前词要查找的信息 |
| **K**（Key） | 键 | "我能提供什么？" — 每个词的标签/索引 |
| **V**（Value） | 值 | "我的实际内容" — 每个词要传递的信息 |

类比图书馆检索：

```
你想找关于"机器学习"的书（这是你的 Query）

图书馆每本书都有标签（Key）：
  📕 "深度学习"        → 和你的 Query 很匹配（相似度高）
  📗 "线性代数"        → 有一定相关性（相似度中）
  📘 "法国大革命"      → 不相关（相似度低）
  📙 "神经网络原理"    → 很匹配（相似度高）

匹配后，你获取对应的内容（Value）：
  最终你得到的 = 0.4×📕内容 + 0.1×📗内容 + 0.01×📘内容 + 0.49×📙内容
                 ↑ 按相关程度加权混合
```

### 3. 数学公式

Attention 的完整计算可以用一个公式表达：

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

分步拆解：

```
步骤 1：计算相似度（Q · Kᵀ）
─────────────────────────────────────────
  Q (当前 token 的查询向量)
  ×
  Kᵀ (所有 token 的键向量的转置)
  =
  Score 矩阵（每对 token 之间的原始相似度分数）

步骤 2：缩放（÷ √d_k）
─────────────────────────────────────────
  Score / √d_k

  为什么要缩放？
  → d_k 是向量维度（如 128）。维度越高，点积值越大，
    softmax 会趋于极端（接近 one-hot），梯度消失。
  → 除以 √128 ≈ 11.3，让数值回到合理范围。

步骤 3：Softmax 归一化
─────────────────────────────────────────
  softmax(缩放后的 Score)

  将分数转化为概率分布（所有权重之和 = 1）。
  相似度高的 token 获得更大的权重。

步骤 4：加权求和（× V）
─────────────────────────────────────────
  Attention 权重 × V (所有 token 的值向量)
  =
  输出向量 = 各 token 值向量的加权组合
```

### 4. 数值计算示例

假设有 4 个 token，向量维度 d_k = 4：

```
Q（"他"的查询向量）= [1.0, 0.5, 0.3, 0.8]

K（所有 token 的键向量）：
  K_小明 = [0.7, 0.2, 0.1, 0.9]
  K_老师 = [0.9, 0.6, 0.4, 0.7]
  K_作业 = [0.1, 0.8, 0.5, 0.2]
  K_交给 = [0.3, 0.1, 0.7, 0.4]

步骤 1：Q · K（点积）
  Q·K_小明 = 1.0×0.7 + 0.5×0.2 + 0.3×0.1 + 0.8×0.9 = 1.55
  Q·K_老师 = 1.0×0.9 + 0.5×0.6 + 0.3×0.4 + 0.8×0.7 = 1.88
  Q·K_作业 = 1.0×0.1 + 0.5×0.8 + 0.3×0.5 + 0.8×0.2 = 0.81
  Q·K_交给 = 1.0×0.3 + 0.5×0.1 + 0.3×0.7 + 0.8×0.4 = 0.88

步骤 2：缩放（÷ √4 = 2）
  [1.55, 1.88, 0.81, 0.88] / 2 = [0.775, 0.940, 0.405, 0.440]

步骤 3：Softmax
  e^0.775 = 2.17,  e^0.940 = 2.56,  e^0.405 = 1.50,  e^0.440 = 1.55
  总和 = 7.78
  权重 = [0.279, 0.329, 0.193, 0.199]
            ↑       ↑
          小明     老师 ← 权重最高，"他"最关注"老师"

步骤 4：加权求和 V
  输出 = 0.279 × V_小明 + 0.329 × V_老师 + 0.193 × V_作业 + 0.199 × V_交给
```

### 5. Self-Attention（自注意力）

在 Transformer 中最常见的是 **Self-Attention**（自注意力）：Q、K、V 全部来自同一个序列。

```
输入序列：[x1, x2, x3, x4]（每个 xi 是一个 token 的嵌入向量）
     │
     ▼
  ┌──────────────────────────┐
  │ 三个线性变换（可学习参数） │
  │  Q = X · W_Q              │
  │  K = X · W_K              │
  │  V = X · W_V              │
  └──────────────────────────┘
     │
     ▼
  Q, K, V 都来自同一个输入 X → 所以叫"自"注意力

  每个 token 都同时和序列中所有 token（包括自己）计算注意力
```

Self-Attention 让序列中**任意两个位置**可以直接交互，无论它们距离多远。这是相比 RNN（只能逐步传递）和 CNN（只能看局部窗口）的核心优势。

### 6. Multi-Head Attention（多头注意力）

实际的 Transformer 不只用一个 Attention，而是并行使用多个"注意力头"（Head）：

```
               输入 X
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Head 1 │ │ Head 2 │ │ Head 3 │ ...  共 h 个头
│Q1,K1,V1│ │Q2,K2,V2│ │Q3,K3,V3│
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    │ out1     │ out2     │ out3
    │          │          │
    └──────────┼──────────┘
               │ Concat（拼接所有头的输出）
               ▼
        ┌─────────────┐
        │ Linear (W_O) │  ← 最终线性变换
        └──────┬──────┘
               ▼
            输出
```

**为什么要多头？**

```
Head 1 可能学会关注"语法关系"：
  "他" → 关注 "老师"（主语-代词 关系）

Head 2 可能学会关注"位置关系"：
  "他" → 关注 "满意"（相邻词组合）

Head 3 可能学会关注"语义关系"：
  "他" → 关注 "交给"（动作-主体 关系）

多个头同时捕捉不同类型的关联，然后组合起来 → 更全面的理解
```

参数关系：

```
假设模型维度 d_model = 768，头数 h = 12

每个头的维度：d_k = d_v = d_model / h = 768 / 12 = 64

总计算量不变：12 个头 × 64 维 = 1 个头 × 768 维
但信息更丰富（多种视角）
```

### 7. Masked Attention（掩码注意力）

在 LLM 的文本生成中，使用 **Causal Mask**（因果掩码）确保每个 token 只能看到它之前的 token，不能"偷看"未来：

```
序列：[A, B, C, D]

Attention Score 矩阵（掩码前）：
        A     B     C     D
  A  [ 0.5   0.3   0.1   0.1 ]
  B  [ 0.2   0.4   0.2   0.2 ]
  C  [ 0.1   0.3   0.4   0.2 ]
  D  [ 0.1   0.2   0.3   0.4 ]

应用因果掩码后（上三角设为 -∞）：
        A     B     C     D
  A  [ 0.5   -∞    -∞    -∞  ]  ← A 只能看到 A
  B  [ 0.2   0.4   -∞    -∞  ]  ← B 只能看到 A, B
  C  [ 0.1   0.3   0.4   -∞  ]  ← C 只能看到 A, B, C
  D  [ 0.1   0.2   0.3   0.4 ]  ← D 能看到所有

Softmax 后 -∞ 变成 0 → 未来 token 的权重为 0
```

这就是为什么 LLM 是"自回归"的——每次只能基于前面的内容预测下一个词。

### 8. KV Cache 与 Attention 的关系

在 LLM 推理（文本生成）时，每一步只新增一个 token，但 Attention 需要和之前所有 token 计算。如果每次都重新计算所有 token 的 K 和 V，会有大量重复计算：

```
不用 KV Cache（每步重新计算所有 K、V）：

步骤 1：生成 token_1
  Q=[q1], K=[k1], V=[v1]               计算 1 次

步骤 2：生成 token_2
  Q=[q2], K=[k1,k2], V=[v1,v2]         重新计算 k1,v1 ← 浪费！

步骤 3：生成 token_3
  Q=[q3], K=[k1,k2,k3], V=[v1,v2,v3]   重新计算 k1,v1,k2,v2 ← 更多浪费！

步骤 n：
  总计算量 ∝ n²（灾难性增长）
```

```
使用 KV Cache（缓存之前的 K、V）：

步骤 1：生成 token_1
  计算 k1,v1 → 存入 Cache
  Q=[q1], K=[k1], V=[v1]

步骤 2：生成 token_2
  计算 k2,v2 → 存入 Cache
  Q=[q2], K=[k1,k2]（k1 从 Cache 读取）, V=[v1,v2]

步骤 3：生成 token_3
  计算 k3,v3 → 存入 Cache
  Q=[q3], K=[k1,k2,k3]（k1,k2 从 Cache 读取）, V=[v1,v2,v3]

步骤 n：
  每步只需计算 1 个新 token 的 K,V → 总计算量 ∝ n（线性增长）
```

KV Cache 用**显存换计算时间**，但也带来了显存管理的挑战——这正是 PagedAttention 要解决的问题。

### 9. Attention 的演化路线

```
原始 Attention (2014, Bahdanau)
  │  用于 Seq2Seq 的编解码器之间
  ▼
Self-Attention (2017, Transformer "Attention Is All You Need")
  │  序列内部的自注意力，替代 RNN
  ▼
Multi-Head Attention (2017, Transformer)
  │  多个注意力头并行，捕捉不同关系
  ▼
Masked/Causal Attention (GPT 系列)
  │  因果掩码，用于自回归语言模型
  ▼
KV Cache (推理优化)
  │  缓存历史 K/V，避免重复计算
  ▼
Multi-Query Attention / MQA (2019)
  │  多个 Q 头共享一组 K/V，减少显存
  ▼
Grouped-Query Attention / GQA (2023, Llama 2)
  │  Q 头分组共享 K/V，MQA 与 MHA 的折中
  ▼
PagedAttention (2023, vLLM)
  │  分页管理 KV Cache 显存
  ▼
FlashAttention (2022/2023)
  │  IO 感知的精确注意力，利用 GPU SRAM 减少显存读写
  ▼
Sparse Attention / DSA (2024/2025, DeepSeek)
  │  稀疏注意力，只计算最相关的 token 对
  ▼
更多创新持续涌现...
```

### 10. 常见 Attention 变体对比

| 变体 | 核心思想 | 优势 | 代表模型 |
|------|---------|------|---------|
| **MHA**（Multi-Head） | 每个头独立的 Q/K/V | 表达能力最强 | 原始 Transformer |
| **MQA**（Multi-Query） | 所有 Q 头共享 1 组 K/V | KV Cache 最小 | PaLM, Falcon |
| **GQA**（Grouped-Query） | Q 头分组，每组共享 K/V | 平衡质量与效率 | Llama 2/3/4, Qwen |
| **FlashAttention** | IO 感知，分块计算 | 速度快，省显存 | 几乎所有现代模型 |
| **PagedAttention** | KV Cache 分页管理 | 显存利用率极高 | vLLM |
| **Sparse Attention** | 只计算部分 token 对 | 支持超长序列 | DeepSeek, Longformer |

### 11. 一句话总结

> **Attention 是让模型在处理每个词时，动态地"关注"输入序列中所有相关词的机制**。通过 Q·K 计算相关性权重，再用权重对 V 加权求和，得到融合了全局上下文信息的输出。它是 Transformer 的核心，也是所有现代大模型的基石。

---

## 附录：什么是 FlashAttention

FlashAttention 是由斯坦福大学 Tri Dao 等人于 2022 年提出的一种**IO 感知的精确注意力算法**。它不改变 Attention 的数学结果（不是近似），而是通过重新组织计算顺序，极大减少 GPU 显存（HBM）的读写次数，从而实现 **2-4 倍加速** 和 **5-20 倍显存节省**。

### 1. 问题背景：GPU 的内存层级瓶颈

GPU 有两级存储，速度差异巨大：

```
┌─────────────────────────────────────────────────┐
│                   GPU 芯片                       │
│                                                 │
│   ┌───────────────────────────┐                 │
│   │    SRAM（片上缓存）        │                 │
│   │    容量：~20 MB            │                 │
│   │    带宽：~19 TB/s          │  ← 极快但极小   │
│   │    （每个 SM 的共享内存）    │                 │
│   └─────────────┬─────────────┘                 │
│                 │ 数据搬运                       │
│                 │ （这是瓶颈！）                  │
│   ┌─────────────▼─────────────┐                 │
│   │    HBM（高带宽显存）        │                 │
│   │    容量：40-80 GB          │                 │
│   │    带宽：~2 TB/s           │  ← 大但慢 10x   │
│   │    （全局显存）             │                 │
│   └───────────────────────────┘                 │
└─────────────────────────────────────────────────┘

SRAM 比 HBM 快约 10 倍，但容量小约 1000 倍。
Attention 的瓶颈不在计算，而在 HBM 读写！
```

### 2. 标准 Attention 为什么慢

标准实现的计算流程需要**多次读写 HBM**：

```
标准 Attention 的 HBM 读写流程：

步骤 1：计算 S = Q × Kᵀ
  从 HBM 读取 Q, K        ──→  HBM 读 ①②
  计算矩阵乘法
  将 S 写回 HBM            ──→  HBM 写 ③
                                S 是 N×N 矩阵，N=序列长度
                                当 N=4096 时，S 占 64MB（FP32）

步骤 2：计算 P = softmax(S)
  从 HBM 读取 S            ──→  HBM 读 ④
  计算 softmax
  将 P 写回 HBM            ──→  HBM 写 ⑤

步骤 3：计算 O = P × V
  从 HBM 读取 P, V         ──→  HBM 读 ⑥⑦
  计算矩阵乘法
  将 O 写回 HBM            ──→  HBM 写 ⑧

总计：8 次 HBM 读写，且中间矩阵 S 和 P 占 O(N²) 显存
```

核心问题：

| 问题 | 说明 |
|------|------|
| **HBM 读写次数多** | Q, K, V, S, P, O 反复在 HBM 和 SRAM 之间搬运 |
| **中间矩阵 S 巨大** | N×N 的 Attention Score 矩阵，N=4096 时占 64MB |
| **显存占用 O(N²)** | 随序列长度平方增长，限制了最大序列长度 |

### 3. FlashAttention 的核心思想：分块计算（Tiling）

FlashAttention 的关键洞察：**不需要把整个 N×N 的 S 矩阵存下来**。通过分块处理，每次只在 SRAM 中计算一小块，然后用在线算法（Online Softmax）逐步累加结果。

```
标准 Attention：一次性计算整个 N×N 矩阵

Q  ┌───────────┐     K^T ┌───────────┐      S  ┌───────────┐
   │           │    ×    │           │    =    │ N×N 矩阵   │
   │  N × d    │         │  d × N    │         │ 全部存在   │
   │           │         │           │         │ HBM 中！   │
   └───────────┘         └───────────┘         └───────────┘


FlashAttention：分块计算，每次只处理一小块

Q 按行分块     K^T 按列分块
┌───┐          ┌──┬──┬──┐
│Q_1│    ×     │K₁│K₂│K₃│     每次只计算一个小块 S_ij
├───┤          │  │  │  │     在 SRAM 中完成
│Q_2│          │  │  │  │     结果直接累加到 O
├───┤          │  │  │  │     不需要存完整的 S 矩阵
│Q_3│          └──┴──┴──┘
└───┘

每次：Q_i × K_j^T → 小块 S_ij → 在 SRAM 中 softmax → 累加到 O_i
```

### 4. 在线 Softmax（Online Softmax）— 核心算法

标准 Softmax 需要看到**所有**分数才能计算（因为需要全局 max 和全局 sum）。FlashAttention 使用 **Online Softmax** 技巧，可以分块计算并逐步更新：

```
标准 Softmax（需要两遍扫描）：
─────────────────────────────────
  第 1 遍：找全局最大值 m = max(s_1, s_2, ..., s_N)
  第 2 遍：计算 softmax(s_i) = exp(s_i - m) / Σ exp(s_j - m)
  → 必须把所有 s_i 存在 HBM 中


Online Softmax（单遍扫描，逐块更新）：
─────────────────────────────────
  处理第 1 块：
    m₁ = max(块1的分数)
    l₁ = Σ exp(块1的分数 - m₁)
    o₁ = softmax(块1) × V₁

  处理第 2 块：
    m₂ = max(m₁, max(块2的分数))        ← 更新全局 max
    l₂ = l₁ × exp(m₁ - m₂) + Σ exp(块2的分数 - m₂)  ← 修正累加和
    o₂ = o₁ × (l₁ × exp(m₁ - m₂) / l₂) + softmax(块2) × V₂  ← 修正输出

  处理第 k 块：
    mₖ = max(mₖ₋₁, max(块k的分数))
    lₖ = lₖ₋₁ × exp(mₖ₋₁ - mₖ) + Σ exp(块k的分数 - mₖ)
    oₖ = oₖ₋₁ × (修正系数) + 当前块贡献

  最终 O = oₖ  → 和标准 Attention 数学上完全一致！
```

关键：每次更新时，之前的结果通过一个**修正系数** `exp(m_old - m_new)` 来调整，保证数学精确性。

### 5. FlashAttention 的完整流程

```
┌──────────────────────────────────────────────────────────────┐
│                    FlashAttention 执行流程                     │
└──────────────────────────────────────────────────────────────┘

HBM（全局显存）：存放 Q, K, V, O
SRAM（片上缓存）：临时存放当前处理的小块

for 每个 Q 的分块 Q_i:                       ← 外层循环
│
│   从 HBM 加载 Q_i 到 SRAM                   ← HBM 读 1 次
│   初始化：m_i = -∞,  l_i = 0,  o_i = 0
│
│   for 每个 K,V 的分块 K_j, V_j:             ← 内层循环
│   │
│   │   从 HBM 加载 K_j, V_j 到 SRAM          ← HBM 读
│   │
│   │   ┌──── 以下全部在 SRAM 中完成 ────┐
│   │   │                                │
│   │   │  S_ij = Q_i × K_j^T / √d      │  计算小块注意力分数
│   │   │                                │
│   │   │  m_new = max(m_i, max(S_ij))   │  更新全局最大值
│   │   │                                │
│   │   │  l_i = l_i × exp(m_i - m_new)  │  修正累加和
│   │   │       + Σ exp(S_ij - m_new)    │
│   │   │                                │
│   │   │  o_i = o_i × exp(m_i - m_new)  │  修正之前的输出
│   │   │       + exp(S_ij - m_new) × V_j│  加上当前块的贡献
│   │   │                                │
│   │   │  m_i = m_new                   │  保存最新的 max
│   │   │                                │
│   │   └────────────────────────────────┘
│   │
│   end for
│
│   O_i = o_i / l_i                           ← 最终归一化
│   将 O_i 写回 HBM                           ← HBM 写 1 次
│
end for
```

### 6. 标准 Attention vs FlashAttention 对比

```
标准 Attention（以 N=4096, d=128 为例）：

  HBM 读写：
    Q: 4096×128 = 2 MB    ×2（读+写）
    K: 4096×128 = 2 MB    ×1（读）
    V: 4096×128 = 2 MB    ×1（读）
    S: 4096×4096 = 64 MB  ×2（写+读）  ← 巨大！
    P: 4096×4096 = 64 MB  ×2（写+读）  ← 巨大！
    O: 4096×128 = 2 MB    ×1（写）
  ─────────────────────
  总 HBM 流量：~264 MB
  额外显存：~128 MB（S 和 P 矩阵）


FlashAttention：

  HBM 读写：
    Q: 每块读 1 次                     合计 2 MB
    K: 每块读 T_q 次（T_q = Q 的块数） 合计 2×T_q MB
    V: 同 K                            合计 2×T_q MB
    O: 每块写 1 次                     合计 2 MB
    S, P: 不需要存到 HBM！             0 MB  ← 核心优势
  ─────────────────────
  总 HBM 流量：~20 MB（大幅减少）
  额外显存：~0 MB（只用 SRAM，O(N) 复杂度）
```

| 维度 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| HBM 读写量 | O(N² + Nd) | O(N²d / M)，M = SRAM 大小 |
| 额外显存 | O(N²) | O(N)（不存 S/P 矩阵） |
| 计算量 | O(N²d) | O(N²d)（相同，不是近似！） |
| 实际速度 | 基准 | **2-4 倍加速** |
| 最大序列长度 | 受 N² 显存限制 | 可处理更长序列 |
| 数学精确性 | 精确 | **同样精确**（不是近似） |

### 7. FlashAttention 的版本演化

```
FlashAttention-1 (2022, Tri Dao)
  │  核心贡献：Tiling + Online Softmax
  │  首次实现 IO 感知的精确注意力
  ▼
FlashAttention-2 (2023, Tri Dao)
  │  优化：更好的并行策略
  │  - 在序列长度维度上并行（而非 batch/head 维度）
  │  - 减少非矩阵乘法的 FLOPs
  │  - 更好的 warp 间工作分配
  │  速度提升：比 v1 快 2 倍，达到理论峰值的 50-73%
  ▼
FlashAttention-3 (2024, Tri Dao)
  │  针对 Hopper 架构（H100）优化
  │  - 利用 WGMMA（Warp Group Matrix Multiply-Accumulate）
  │  - 异步 TMA（Tensor Memory Accelerator）数据加载
  │  - FP8 量化支持
  │  - 计算与数据加载的流水线重叠
  │  速度：接近 H100 理论峰值的 75%
  ▼
FlashAttention 已成为所有现代 LLM 的标配
```

### 8. 为什么 FlashAttention 不是近似？

很多人误以为 FlashAttention 是一种"近似注意力"（像 Sparse Attention 那样跳过一些计算）。**这是错误的**。

```
FlashAttention 的数学等价性证明：

标准方式：
  O = softmax(QK^T / √d) × V

FlashAttention 方式：
  将 QK^T 分块为 S_11, S_12, S_21, S_22, ...
  对每块计算局部 softmax，用 Online Softmax 修正合并
  最终得到的 O 和标准方式完全相同（位精度可能有微小差异，来自浮点运算顺序）

关键：
  - 没有丢弃任何 token 对（不是 Sparse）
  - 没有降低精度（不是低秩近似）
  - 只是改变了计算顺序和内存访问模式
  - 结果在数学上是精确的
```

### 9. FlashAttention 在 vLLM 中的应用

vLLM 同时使用 FlashAttention 和 PagedAttention，它们解决不同的问题：

```
FlashAttention                    PagedAttention
──────────                        ──────────────
解决：Attention 计算本身的         解决：KV Cache 的显存管理问题
     显存读写效率问题

优化目标：减少 HBM 读写次数        优化目标：减少显存碎片和浪费

作用阶段：单次 Attention 计算      作用阶段：跨请求的 KV Cache 分配

核心技术：Tiling + Online Softmax  核心技术：分页 + Block Table

两者互补：
  ┌─────────────────────────────────────────┐
  │           vLLM 推理流程                   │
  │                                         │
  │  KV Cache 存储：PagedAttention 管理       │
  │    ↓ 按需分配 Block，无碎片               │
  │                                         │
  │  Attention 计算：FlashAttention 执行      │
  │    ↓ 分块计算，最小化 HBM 读写            │
  │                                         │
  │  结果：显存利用率高 + 计算速度快           │
  └─────────────────────────────────────────┘
```

vLLM 中的 Attention 后端选择：

| 后端 | 文件位置 | 使用场景 |
|------|---------|---------|
| FlashAttention | `vllm/v1/attention/backends/flash_attn.py` | Prefill 阶段（首 token，长序列） |
| PagedAttention V1/V2 | `vllm/v1/attention/backends/tree_attn.py` | Decode 阶段（逐 token 生成） |
| FlashInfer | `vllm/v1/attention/backends/flashinfer.py` | 替代后端，某些场景更优 |

实际推理中，vLLM 通常在 **Prefill 阶段用 FlashAttention**（处理长 prompt，Q 的长度大），在 **Decode 阶段用 PagedAttention**（逐 token 生成，Q 长度为 1，重点是高效访问分页的 KV Cache）。

### 10. 直觉总结

```
类比：你要在一个巨大的图书馆里查资料

标准 Attention：
  1. 把所有书（K）搬到你的桌子（SRAM）上     ← 桌子放不下！
  2. 和你的问题（Q）逐一比对
  3. 比对结果（S）写成一张巨大的表格           ← 表格也放不下！
  4. 对表格做 softmax
  5. 根据权重取出对应内容（V）
  → 大量搬运，桌子不够大只能反复跑仓库

FlashAttention：
  1. 每次只搬一个书架的书（K 的一个 block）到桌子上
  2. 和你的问题（Q）比对，记下临时结果
  3. 搬下一个书架，继续比对，用修正系数更新之前的结果
  4. 所有书架过完后，最终结果和标准方式一模一样
  → 跑仓库的次数大幅减少，桌子从头到尾够用
```

### 11. 一句话总结

> **FlashAttention 是一种 IO 感知的精确注意力算法，通过分块计算（Tiling）和在线 Softmax（Online Softmax），将 Attention 的中间结果保持在 GPU 快速片上缓存（SRAM）中，避免反复读写慢速显存（HBM），在不改变数学结果的前提下实现 2-4 倍加速和显著的显存节省。**
