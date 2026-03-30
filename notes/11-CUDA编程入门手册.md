# CUDA 编程入门手册（由浅入深）

> 本手册面向有 C/C++ 基础但无 GPU 编程经验的开发者，由浅入深讲解 CUDA 编程核心知识。
> 每个知识点均附带可编译运行的完整代码示例。

---

## 目录

- [第一部分：基础概念](#第一部分基础概念)
  - [1. CUDA 简介](#1-cuda-简介)
  - [2. 环境配置与验证](#2-环境配置与验证)
  - [3. Hello World — 第一个 CUDA 程序](#3-hello-world--第一个-cuda-程序)
  - [4. 线程模型：Thread / Block / Grid](#4-线程模型thread--block--grid)
  - [5. 内存模型基础：Host 与 Device](#5-内存模型基础host-与-device)
- [第二部分：核心编程](#第二部分核心编程)
  - [6. 向量加法 — 完整 Kernel 编写流程](#6-向量加法--完整-kernel-编写流程)
  - [7. 错误处理](#7-错误处理)
  - [8. 多维线程索引计算](#8-多维线程索引计算)
  - [9. 内存类型详解](#9-内存类型详解)
  - [10. 共享内存实战：矩阵转置](#10-共享内存实战矩阵转置)
- [第三部分：性能优化](#第三部分性能优化)
  - [11. 合并访存（Coalesced Memory Access）](#11-合并访存coalesced-memory-access)
  - [12. Bank Conflict](#12-bank-conflict)
  - [13. CUDA Stream 与异步并发](#13-cuda-stream-与异步并发)
  - [14. 原子操作](#14-原子操作)
  - [15. 并行归约（Parallel Reduction）](#15-并行归约parallel-reduction)
- [第四部分：进阶主题](#第四部分进阶主题)
  - [16. Unified Memory（统一内存）](#16-unified-memory统一内存)
  - [17. cuBLAS 基础](#17-cublas-基础)
  - [18. Thrust 库](#18-thrust-库)
  - [19. 性能分析工具](#19-性能分析工具)
  - [20. CUDA 在 vLLM 中的应用](#20-cuda-在-vllm-中的应用)

---

# 第一部分：基础概念

## 1. CUDA 简介

### 什么是 CUDA

CUDA（Compute Unified Device Architecture）是 NVIDIA 推出的并行计算平台和编程模型，让开发者可以使用 C/C++ 编写运行在 GPU 上的程序。

### CPU vs GPU

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 几个到几十个（强核心） | 数千个（轻量核心） |
| 适合任务 | 串行、复杂逻辑 | 大规模并行、数据密集型 |
| 时钟频率 | 高（~5 GHz） | 较低（~2 GHz） |
| 缓存 | 大 | 小 |

**核心思想**：CPU 擅长"跑得快"，GPU 擅长"一起跑"。

### CUDA 编程模型概览

```
Host (CPU)                    Device (GPU)
┌──────────┐                  ┌──────────────────┐
│ 主程序   │ ──cudaMemcpy──> │ 显存（全局内存）  │
│          │                  │                    │
│ 启动     │ ──kernel<<<>>>──>│ ┌──┐┌──┐┌──┐┌──┐ │
│ kernel   │                  │ │T0││T1││T2││T3│ │ ← 成千上万线程并行
│          │ <──cudaMemcpy── │ └──┘└──┘└──┘└──┘ │
│ 获取结果 │                  │                    │
└──────────┘                  └──────────────────┘
```

基本流程：
1. 在 CPU 端准备数据
2. 将数据从主机内存拷贝到显存
3. 启动 GPU kernel（核函数）执行并行计算
4. 将结果从显存拷贝回主机内存

---

## 2. 环境配置与验证

### 前置条件

- NVIDIA GPU（计算能力 ≥ 3.0）
- NVIDIA 驱动
- CUDA Toolkit（推荐 11.x 或 12.x）

### 安装验证

```bash
# 查看 GPU 信息
nvidia-smi

# 查看 CUDA 编译器版本
nvcc --version

# 查看 GPU 计算能力
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### 代码示例：查询 GPU 属性

```cuda
// file: query_device.cu
// 编译: nvcc -o query_device query_device.cu && ./query_device

#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("检测到 %d 个 CUDA 设备\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力:           %d.%d\n", prop.major, prop.minor);
        printf("  全局内存:           %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  SM 数量:            %d\n", prop.multiProcessorCount);
        printf("  每个 Block 最大线程: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp 大小:          %d\n", prop.warpSize);
        printf("  共享内存/Block:     %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  时钟频率:           %.2f GHz\n", prop.clockRate / 1e6);
        printf("\n");
    }
    return 0;
}
```

---

## 3. Hello World — 第一个 CUDA 程序

### 关键概念

- `__global__`：修饰 kernel 函数，由 CPU 调用、在 GPU 上执行
- `__device__`：修饰设备函数，由 GPU 调用、在 GPU 上执行
- `__host__`：修饰主机函数（默认），由 CPU 调用、在 CPU 上执行
- `<<<blocks, threads>>>`：kernel 启动配置，表示几个组，每组几个线程。这也是一个grid(即kernel启动时的任务总大小。)
- `block`：GPU上执行的最小单位。GPU不会一个线程一个线程的去调度，它一次调度一个block。一个组里一般有32，64，128等多个线程。

### 代码示例

```cuda
// file: hello_cuda.cu
// 编译: nvcc -o hello_cuda hello_cuda.cu && ./hello_cuda

#include <stdio.h>

// __global__ 标记这是一个 kernel 函数
__global__ void helloKernel() {
    // threadIdx.x 是当前线程在 Block 内的索引
    // blockIdx.x 是当前 Block 在 Grid 内的索引
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("从 CPU 启动 kernel...\n");

    // <<<2, 4>>> 表示启动 2 个 Block，每个 Block 有 4 个线程
    // 总共 2 × 4 = 8 个线程并行执行
    helloKernel<<<2, 4>>>();

    // 等待 GPU 完成所有工作
    cudaDeviceSynchronize();

    printf("GPU 执行完毕!\n");
    return 0;
}
```

**预期输出**（顺序可能不同）：
```
从 CPU 启动 kernel...
Hello from GPU! Block 0, Thread 0
Hello from GPU! Block 0, Thread 1
Hello from GPU! Block 0, Thread 2
Hello from GPU! Block 0, Thread 3
Hello from GPU! Block 1, Thread 0
Hello from GPU! Block 1, Thread 1
Hello from GPU! Block 1, Thread 2
Hello from GPU! Block 1, Thread 3
GPU 执行完毕!
```

> **注意**：线程输出顺序不确定，因为 GPU 线程是并行执行的。

---

## 4. 线程模型：Thread / Block / Grid

### 层次结构

```
Grid（网格）
├── Block (0,0)          Block (1,0)          Block (2,0)
│   ├── Thread (0,0)     ├── Thread (0,0)     ├── ...
│   ├── Thread (1,0)     ├── Thread (1,0)
│   ├── Thread (0,1)     ├── Thread (0,1)
│   └── Thread (1,1)     └── Thread (1,1)
```

- **Thread**（线程）：最小执行单元
- **Block**（线程块）：一组线程，同一 Block 内的线程可共享内存、可同步
- **Grid**（网格）：一组 Block，构成一次 kernel 调用
- **Warp**：硬件调度单位，32 个线程为一组（同步执行相同指令）

### 内置变量

| 变量 | 含义 |
|------|------|
| `threadIdx.x/y/z` | 线程在 Block 内的索引 |
| `blockIdx.x/y/z` | Block 在 Grid 内的索引 |
| `blockDim.x/y/z` | 每个 Block 的线程数 |
| `gridDim.x/y/z` | Grid 中 Block 的数量 |

### 代码示例：打印线程层次信息

```cuda
// file: thread_hierarchy.cu
// 编译: nvcc -o thread_hierarchy thread_hierarchy.cu && ./thread_hierarchy

#include <stdio.h>

__global__ void printInfo() {
    // 计算全局唯一线程 ID
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("globalId=%d  blockIdx=%d  threadIdx=%d  blockDim=%d  gridDim=%d\n",
           globalId, blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
}

int main() {
    // 3 个 Block，每个 Block 4 个线程，共 12 个线程
    printInfo<<<3, 4>>>();
    cudaDeviceSynchronize();  //等待gpu把所有活干完cpu再继续跑。

    printf("\n--- 全局索引计算公式 ---\n");
    printf("globalId = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("例: Block 2, Thread 3 → globalId = 2 * 4 + 3 = 11\n");
    return 0;
}
```

### 线程索引计算总结

对于一维情况：
```
全局线程 ID = blockIdx.x * blockDim.x + threadIdx.x （第几个块*每个块中的线程数量 + 线程在小组中的编号）
总线程数    = gridDim.x * blockDim.x （多少个块 * 多个少个线程）
```

---

## 5. 内存模型基础：Host 与 Device

### 核心 API

| 函数 | 作用                     |
|------|------------------------|
| `cudaMalloc(&ptr, size)` | 在 GPU 显存上分配内存          |
| `cudaFree(ptr)` | 释放 GPU 显存              |
| `cudaMemcpy(dst, src, size, kind)` | 主机与设备之间拷贝数据（所谓设备就是gpu） |

`cudaMemcpy` 的 `kind` 参数：
- `cudaMemcpyHostToDevice`：CPU → GPU
- `cudaMemcpyDeviceToHost`：GPU → CPU
- `cudaMemcpyDeviceToDevice`：GPU → GPU

### 代码示例：数组拷贝往返

```cuda
// file: memcpy_demo.cu
// 编译: nvcc -o memcpy_demo memcpy_demo.cu && ./memcpy_demo

#include <stdio.h>
#include <stdlib.h>

// 简单的 kernel：每个元素乘以 2
__global__ void doubleArray(float *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] *= 2.0f;
    }
}

int main() {
    const int N = 8;
    size_t bytes = N * sizeof(float);

    // 步骤 1: 在 Host（CPU）分配并初始化
    float *h_arr = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_arr[i] = (float)i;  // [0, 1, 2, ..., 7]
    }

    printf("原始数据: ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_arr[i]);
    printf("\n");

    // 步骤 2: 在 Device（GPU）分配内存
    float *d_arr;
    cudaMalloc(&d_arr, bytes);

    // 步骤 3: 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);

    // 步骤 4: 启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // 步骤 5: 将结果从 Device 拷贝回 Host
    cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);

    printf("×2 结果:  ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_arr[i]);
    printf("\n");

    // 步骤 6: 释放内存
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
```

**输出**：
```
原始数据: 0 1 2 3 4 5 6 7
×2 结果:  0 2 4 6 8 10 12 14
```

---

# 第二部分：核心编程

## 6. 向量加法 — 完整 Kernel 编写流程

这是 CUDA 编程的经典入门示例，展示完整的工作流。

```cuda
// file: vector_add.cu
// 编译: nvcc -o vector_add vector_add.cu && ./vector_add

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Kernel: C[i] = A[i] + B[i]
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  //线程全局id。
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1000000;  // 100 万元素
    size_t bytes = N * sizeof(float);

    // --- Host 端分配与初始化， 100万float数组 ---
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i) * sinf(i);
        h_B[i] = cosf(i) * cosf(i);
    }

    // --- Device 端分配 ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // --- Host → Device ---
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // --- 启动 Kernel ---
    int blockSize = 256;  //每个块里多少个线程。
    int gridSize = (N + blockSize - 1) / blockSize;  // 向上取整，多少个块。
    printf("Grid: %d blocks, Block: %d threads\n", gridSize, blockSize);

    // vectorAdd<<<块数，线程数>>>，此函数由cpu调用，在gpu上执行。
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // --- Device → Host ---
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- 验证结果 ---
    // sin²(x) + cos²(x) = 1.0
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(h_C[i] - 1.0f));
    }
    printf("最大误差: %e\n", maxError);
    printf("验证: %s\n", maxError < 1e-5 ? "通过 ✓" : "失败 ✗");

    // --- 清理 ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

**输出**：
```
Grid: 3907 blocks, Block: 256 threads
最大误差: 1.192093e-07
验证: 通过 ✓
```

---

## 7. 错误处理

CUDA API 调用不会抛出异常，必须手动检查返回值。推荐使用宏封装。

```cuda
// file: error_handling.cu
// 编译: nvcc -o error_handling error_handling.cu && ./error_handling

#include <stdio.h>

// ===== 错误检查宏（推荐在所有项目中使用） =====
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// 检查最近一次 kernel 启动的错误
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "Kernel Launch Error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void dummyKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    const int N = 100;
    float *d_data;

    // 正常使用 - 每个 CUDA 调用都用宏包裹
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    dummyKernel<<<1, N>>>(d_data, N);
    CUDA_CHECK_KERNEL();                     // 检查 kernel 启动
    CUDA_CHECK(cudaDeviceSynchronize());     // 检查 kernel 执行

    printf("正常执行完成\n");

    // 演示错误捕获：故意释放两次
    CUDA_CHECK(cudaFree(d_data));

    // 下面这行会触发错误
    cudaError_t err = cudaFree(d_data);
    if (err != cudaSuccess) {
        printf("捕获到预期错误: %s\n", cudaGetErrorString(err));
    }

    printf("错误处理演示完成\n");
    return 0;
}
```

---

## 8. 多维线程索引计算

CUDA 支持最多 3 维的 Block 和 Grid，适合处理矩阵、图像等多维数据。

```cuda
// file: index_2d.cu
// 编译: nvcc -o index_2d index_2d.cu && ./index_2d

#include <stdio.h>

// 2D 矩阵赋值 kernel
__global__ void fillMatrix(int *matrix, int rows, int cols) {
    // 2D 线程索引 → 行列映射
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // 行优先存储：线性索引 = row * cols + col
        int linearIdx = row * cols + col;
        matrix[linearIdx] = row * 10 + col;  // 编码行列信息
    }
}

int main() {
    const int ROWS = 4, COLS = 6;
    size_t bytes = ROWS * COLS * sizeof(int);

    int *h_matrix = (int *)malloc(bytes);
    int *d_matrix;
    cudaMalloc(&d_matrix, bytes);

    // 2D Block 和 Grid 配置
    dim3 blockDim(4, 4);  // 每个 Block 4×4 = 16 个线程
    dim3 gridDim(
        (COLS + blockDim.x - 1) / blockDim.x,  // x 方向覆盖列
        (ROWS + blockDim.y - 1) / blockDim.y    // y 方向覆盖行
    );
    printf("Grid: (%d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    fillMatrix<<<gridDim, blockDim>>>(d_matrix, ROWS, COLS);
    cudaMemcpy(h_matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);

    // 打印矩阵
    printf("\n矩阵内容 (值=行*10+列):\n");
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            printf("%3d ", h_matrix[r * COLS + c]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    free(h_matrix);
    return 0;
}
```

**输出**：
```
Grid: (2, 1), Block: (4, 4)

矩阵内容 (值=行*10+列):
  0   1   2   3   4   5
 10  11  12  13  14  15
 20  21  22  23  24  25
 30  31  32  33  34  35
```

### 索引公式速查（这个比较好，固定公式）

```
1D:  idx = blockIdx.x * blockDim.x + threadIdx.x

2D:  col = blockIdx.x * blockDim.x + threadIdx.x
     row = blockIdx.y * blockDim.y + threadIdx.y
     linear = row * width + col

3D:  x = blockIdx.x * blockDim.x + threadIdx.x
     y = blockIdx.y * blockDim.y + threadIdx.y
     z = blockIdx.z * blockDim.z + threadIdx.z
     linear = z * (width * height) + y * width + x
```

---

## 9. 内存类型详解

### 内存层次

```
┌─────────────────────────────────────┐
│           全局内存 (Global)          │  ← 最大、最慢，所有线程可访问
│  ┌───────────────────────────────┐  │
│  │      常量内存 (Constant)      │  │  ← 只读，有缓存，64KB
│  └───────────────────────────────┘  │
│  ┌──────────┐  ┌──────────┐        │
│  │ Block 0  │  │ Block 1  │        │
│  │ ┌──────┐ │  │ ┌──────┐ │        │
│  │ │共享   │ │  │ │共享   │ │        │  ← 每个 Block 独有，速度快
│  │ │内存   │ │  │ │内存   │ │        │
│  │ └──────┘ │  │ └──────┘ │        │
│  │ T0 T1 T2 │  │ T0 T1 T2 │        │  ← 每个线程有自己的寄存器
│  │ (寄存器) │  │ (寄存器) │        │     速度最快
│  └──────────┘  └──────────┘        │
└─────────────────────────────────────┘
```

| 内存类型 | 作用域 | 生命周期 | 速度 | 大小 |
|---------|--------|---------|------|------|
| 寄存器 | 线程 | 线程 | 最快 | 有限 |
| 共享内存 | Block | Block | 很快（~5ns） | 48-164 KB/Block |
| 全局内存 | 所有线程 | 应用 | 慢（~500ns） | 数 GB |
| 常量内存 | 所有线程（只读） | 应用 | 有缓存时快 | 64 KB |
| 纹理内存 | 所有线程（只读） | 应用 | 有空间局部性缓存 | 取决于显存 |

### 代码示例：各种内存类型

```cuda
// file: memory_types.cu
// 编译: nvcc -o memory_types memory_types.cu && ./memory_types

#include <stdio.h>

// ===== 常量内存 =====
// 在编译时声明，所有线程共享，只读
__constant__ float c_coeff[3];

// ===== Kernel 演示各种内存 =====
__global__ void memoryDemo(float *d_input, float *d_output, int n) {
    // 寄存器变量（默认，每个线程独有）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 共享内存（同一 Block 内线程共享）
    __shared__ float s_data[256];

    if (idx < n) {
        // 从全局内存读取到共享内存
        s_data[threadIdx.x] = d_input[idx];
        __syncthreads();  // 确保所有线程都写入完毕

        // 使用常量内存中的系数
        float val = s_data[threadIdx.x];
        float result = c_coeff[0] * val * val + c_coeff[1] * val + c_coeff[2];

        // 写回全局内存
        d_output[idx] = result;
    }
}

int main() {
    const int N = 8;
    size_t bytes = N * sizeof(float);

    // Host 数据
    float h_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_output[8];

    // 设置常量内存：f(x) = 1*x² + (-2)*x + 1 = (x-1)²
    float coeffs[3] = {1.0f, -2.0f, 1.0f};
    cudaMemcpyToSymbol(c_coeff, coeffs, 3 * sizeof(float));

    // Device 内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    memoryDemo<<<1, N>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    printf("f(x) = x² - 2x + 1 = (x-1)²\n\n");
    for (int i = 0; i < N; i++) {
        printf("f(%.0f) = %.0f\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

**输出**：
```
f(x) = x² - 2x + 1 = (x-1)²

f(1) = 0
f(2) = 1
f(3) = 4
f(4) = 9
f(5) = 16
f(6) = 25
f(7) = 36
f(8) = 49
```

---

## 10. 共享内存实战：矩阵转置

矩阵转置是展示共享内存优势的经典案例。朴素转置存在非合并访存问题，共享内存可以解决。

```cuda
// file: transpose.cu
// 编译: nvcc -o transpose transpose.cu && ./transpose

#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 16

// ===== 朴素转置（全局内存直接读写） =====
__global__ void transposeNaive(const float *input, float *output, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    if (col < width && row < height) {
        // 读: 行优先合并; 写: 列优先非合并 → 慢
        output[col * height + row] = input[row * width + col];
    }
}

// ===== 共享内存优化转置 =====
__global__ void transposeShared(const float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 避免 bank conflict

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // 合并读取到共享内存
    if (col < width && row < height) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    __syncthreads();

    // 转置后的坐标
    int newCol = blockIdx.y * TILE_DIM + threadIdx.x;
    int newRow = blockIdx.x * TILE_DIM + threadIdx.y;

    // 合并写入（从共享内存的转置位置读取）
    if (newCol < height && newRow < width) {
        output[newRow * height + newCol] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int WIDTH = 1024, HEIGHT = 1024;
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    float *h_output1 = (float *)malloc(bytes);
    float *h_output2 = (float *)malloc(bytes);

    // 初始化
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = (float)i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM,
              (HEIGHT + TILE_DIM - 1) / TILE_DIM);

    // --- 测试朴素版 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        transposeNaive<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msNaive;
    cudaEventElapsedTime(&msNaive, start, stop);
    cudaMemcpy(h_output1, d_output, bytes, cudaMemcpyDeviceToHost);

    // --- 测试共享内存版 ---
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        transposeShared<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msShared;
    cudaEventElapsedTime(&msShared, start, stop);
    cudaMemcpy(h_output2, d_output, bytes, cudaMemcpyDeviceToHost);

    printf("矩阵大小: %d × %d\n", WIDTH, HEIGHT);
    printf("朴素转置:     %.2f ms (100次)\n", msNaive);
    printf("共享内存转置: %.2f ms (100次)\n", msShared);
    printf("加速比: %.2fx\n", msNaive / msShared);

    // 验证
    int correct = 1;
    for (int i = 0; i < WIDTH * HEIGHT && correct; i++) {
        if (h_output1[i] != h_output2[i]) correct = 0;
    }
    printf("结果一致性: %s\n", correct ? "通过" : "失败");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output1);
    free(h_output2);
    return 0;
}
```

---

# 第三部分：性能优化

## 11. 合并访存（Coalesced Memory Access）

### 概念

GPU 的全局内存以 **128 字节为一个事务** 读取。如果一个 Warp（32 个线程）访问的地址是连续的，只需一次事务；如果是分散的，需要多次事务，严重降低性能。

```
合并访存（好）：                    非合并访存（坏）：
Thread 0 → addr[0]                Thread 0 → addr[0]
Thread 1 → addr[1]                Thread 1 → addr[128]
Thread 2 → addr[2]                Thread 2 → addr[256]
...                               ...
→ 1 次内存事务                     → 32 次内存事务！
```

### 代码示例：合并 vs 非合并访存对比

```cuda
// file: coalesced.cu
// 编译: nvcc -o coalesced coalesced.cu && ./coalesced

#include <stdio.h>

#define N (1024 * 1024)

// 合并访存：相邻线程访问相邻地址
__global__ void coalesced(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;  // 连续访问
    }
}

// 非合并访存：线程以大步长访问
__global__ void strided(float *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int target = (idx * stride) % n;  // 跳跃访问
    if (target < n) {
        data[target] *= 2.0f;
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 测试合并访存
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        coalesced<<<gridSize, blockSize>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msCoalesced;
    cudaEventElapsedTime(&msCoalesced, start, stop);

    // 测试步长访存（stride = 32，每个线程跳 32 个元素）
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        strided<<<gridSize, blockSize>>>(d_data, N, 32);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msStrided;
    cudaEventElapsedTime(&msStrided, start, stop);

    printf("合并访存:   %.2f ms (100次)\n", msCoalesced);
    printf("步长访存:   %.2f ms (100次)\n", msStrided);
    printf("性能差距: %.2fx\n", msStrided / msCoalesced);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    return 0;
}
```

**要点**：始终让相邻线程访问相邻内存地址。

---

## 12. Bank Conflict

### 概念

共享内存被分成 32 个 **bank**（每个 bank 宽 4 字节）。同一 Warp 中的不同线程如果访问同一 bank 的不同地址，会产生 **bank conflict**，访问被串行化。

```
无冲突：每线程访问不同 bank         有冲突：多线程访问同一 bank
Bank  0  1  2  3 ... 31            Bank  0  1  2  3 ... 31
      ↑  ↑  ↑  ↑     ↑                  ↑↑↑ ↑
      T0 T1 T2 T3    T31                T0T1T2 T3
                                         串行化！
```

### 代码示例

```cuda
// file: bank_conflict.cu
// 编译: nvcc -o bank_conflict bank_conflict.cu && ./bank_conflict

#include <stdio.h>

#define BLOCK_SIZE 256
#define ITERATIONS 10000

// 无 bank conflict：每线程访问不同 bank
__global__ void noBankConflict(float *output) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;

    // 线程 i 访问 smem[i]，步长为 1 → 无冲突
    smem[tid] = (float)tid;
    __syncthreads();

    float val = smem[tid];
    for (int i = 0; i < ITERATIONS; i++) {
        val = smem[tid] + val * 0.001f;
    }
    output[blockIdx.x * BLOCK_SIZE + tid] = val;
}

// 有 bank conflict：步长为 32，所有线程访问同一 bank
__global__ void withBankConflict(float *output) {
    // 32 × 33 的布局，步长 32 导致所有线程命中同一 bank
    __shared__ float smem[BLOCK_SIZE * 32];
    int tid = threadIdx.x;

    smem[tid * 32] = (float)tid;  // 步长 32 → 全部命中 bank 0
    __syncthreads();

    float val = smem[tid * 32];
    for (int i = 0; i < ITERATIONS; i++) {
        val = smem[tid * 32] + val * 0.001f;
    }
    output[blockIdx.x * BLOCK_SIZE + tid] = val;
}

int main() {
    float *d_output;
    cudaMalloc(&d_output, BLOCK_SIZE * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        noBankConflict<<<1, BLOCK_SIZE>>>(d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msNone;
    cudaEventElapsedTime(&msNone, start, stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        withBankConflict<<<1, BLOCK_SIZE>>>(d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msConflict;
    cudaEventElapsedTime(&msConflict, start, stop);

    printf("无 Bank Conflict:  %.2f ms\n", msNone);
    printf("有 Bank Conflict:  %.2f ms\n", msConflict);
    printf("性能差距: %.2fx\n", msConflict / msNone);

    // 解决方法：padding
    printf("\n解决方法: 声明 __shared__ float smem[N][32+1]\n");
    printf("加 1 列 padding 使步长错开不同 bank\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_output);
    return 0;
}
```

---

## 13. CUDA Stream 与异步并发

### 概念

CUDA Stream 是一个有序的 GPU 操作队列。不同 Stream 中的操作可以并发执行，实现**计算与数据传输的重叠**。

```
默认 (1个Stream):
  [H2D 拷贝] → [Kernel 执行] → [D2H 拷贝]
  ──────────────────────────────────────────→ 时间

多 Stream 并发:
  Stream 1: [H2D_1] → [Kernel_1] → [D2H_1]
  Stream 2:    [H2D_2] → [Kernel_2] → [D2H_2]
  ────────────────────────────────────────→ 时间（更短）
```

### 代码示例

```cuda
// file: streams.cu
// 编译: nvcc -o streams streams.cu && ./streams

#include <stdio.h>

#define N (1024 * 1024)
#define NSTREAMS 4

__global__ void kernel(float *data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 做一些计算，增加 kernel 执行时间
        float x = data[idx];
        for (int i = 0; i < 100; i++) {
            x = x * value + 0.1f;
        }
        data[idx] = x;
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    int chunkSize = N / NSTREAMS;
    size_t chunkBytes = chunkSize * sizeof(float);

    // 使用 pinned memory 以支持异步拷贝
    float *h_data;
    cudaMallocHost(&h_data, bytes);  // pinned memory
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float *d_data;
    cudaMalloc(&d_data, bytes);

    // 创建多个 Stream
    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (chunkSize + blockSize - 1) / blockSize;

    // ===== 单 Stream（串行） =====
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    kernel<<<(N + 255) / 256, 256>>>(d_data, N, 1.001f);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msSingle;
    cudaEventElapsedTime(&msSingle, start, stop);

    // 重置
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    // ===== 多 Stream（并发） =====
    //三个函数cudaMemcpyAsync，kernel，cudaMemcpyAsync都指定了Stream，且在同一Stream内的操作是串行的，不同Stream之间的操作可以并行执行。
    cudaEventRecord(start);
    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * chunkSize;
        // 异步拷贝 Host → Device
        cudaMemcpyAsync(d_data + offset, h_data + offset,
                        chunkBytes, cudaMemcpyHostToDevice, streams[i]);
        // 在同一 Stream 中启动 kernel
        kernel<<<gridSize, blockSize, 0, streams[i]>>>(
            d_data + offset, chunkSize, 1.001f);
        // 异步拷贝 Device → Host
        cudaMemcpyAsync(h_data + offset, d_data + offset,
                        chunkBytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msMulti;
    cudaEventElapsedTime(&msMulti, start, stop);

    printf("数据量: %d 个 float (%.1f MB)\n", N, bytes / (1024.0 * 1024));
    printf("单 Stream: %.2f ms\n", msSingle);
    printf("多 Stream (%d): %.2f ms\n", NSTREAMS, msMulti);
    printf("加速比: %.2fx\n", msSingle / msMulti);

    // 清理
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFreeHost(h_data);  // 释放 pinned memory

    return 0;
}
```

**关键点**：
- `cudaMallocHost` 分配 pinned memory（页锁定内存），才能使用 `cudaMemcpyAsync`
- 同一 Stream 内操作顺序执行，不同 Stream 间可并发

---

## 14. 原子操作

当多个线程需要更新同一内存位置时，必须使用原子操作避免竞态条件。

```cuda
// file: atomic_ops.cu
// 编译: nvcc -o atomic_ops atomic_ops.cu && ./atomic_ops

#include <stdio.h>

#define N (1024 * 1024)

// 错误做法：直接累加会丢失更新
__global__ void sumWrong(const float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        *result += data[idx];  // 竞态条件！
    }
}

// 正确做法：使用 atomicAdd
__global__ void sumAtomic(const float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, data[idx]);  // 原子操作，线程安全
    }
}

// 更好的做法：先 Block 内归约，再原子更新（减少原子操作次数）
__global__ void sumOptimized(const float *data, float *result, int n) {
    __shared__ float blockSum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程加载一个元素
    blockSum[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    // Block 内并行归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            blockSum[tid] += blockSum[tid + stride];
        }
        __syncthreads();
    }

    // 只有线程 0 执行一次原子操作
    if (tid == 0) {
        atomicAdd(result, blockSum[0]);
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_data = (float *)malloc(bytes);
    float h_result;

    // 每个元素 = 1.0，期望总和 = N
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float *d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    float expected = (float)N;

    // 测试错误版本
    h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    sumWrong<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("非原子操作结果:  %f (期望 %f) → 不正确!\n", h_result, expected);

    // 测试原子版本
    h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    sumAtomic<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("原子操作结果:    %f (期望 %f)\n", h_result, expected);

    // 测试优化版本
    h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    sumOptimized<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("优化归约结果:    %f (期望 %f)\n", h_result, expected);

    // 常用原子操作列表
    printf("\n--- 常用原子操作 ---\n");
    printf("atomicAdd(addr, val)    原子加\n");
    printf("atomicSub(addr, val)    原子减\n");
    printf("atomicMin(addr, val)    原子取最小\n");
    printf("atomicMax(addr, val)    原子取最大\n");
    printf("atomicExch(addr, val)   原子交换\n");
    printf("atomicCAS(addr, cmp, val) 比较并交换\n");

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
```

---

## 15. 并行归约（Parallel Reduction）

归约是 GPU 编程中最核心的算法模式之一（求和、求最大值等）。

```cuda
// file: reduction.cu
// 编译: nvcc -o reduction reduction.cu && ./reduction

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// ===== 版本 1：交错寻址（有 Warp 分歧） =====
__global__ void reduce_v1(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 交错归约：步长从 1 开始翻倍
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // 问题：导致 warp 分歧
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ===== 版本 2：连续线程工作（减少 Warp 分歧） =====
__global__ void reduce_v2(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 步长从大到小，让前面连续的线程工作
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ===== 版本 3：首次加载时做一次归约（减少空闲线程） =====
__global__ void reduce_v3(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 加载时就合并两个元素
    float val = 0.0f;
    if (idx < n) val += input[idx];
    if (idx + blockDim.x < n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

float reduceOnGPU(const float *d_input, int n,
                  void (*kernel)(const float*, float*, int),
                  int elementsPerBlock) {
    int gridSize = (n + elementsPerBlock - 1) / elementsPerBlock;
    float *d_output;
    cudaMalloc(&d_output, gridSize * sizeof(float));

    kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);

    // 将 Block 结果拷回 CPU 做最终求和
    float *h_output = (float *)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < gridSize; i++) sum += h_output[i];

    free(h_output);
    cudaFree(d_output);
    return sum;
}

int main() {
    const int N = 1 << 20;  // 1M 元素
    size_t bytes = N * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input;
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // CPU 参考结果
    float cpuSum = 0.0f;
    for (int i = 0; i < N; i++) cpuSum += h_input[i];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // V1: 交错寻址
    cudaEventRecord(start);
    float sum1 = reduceOnGPU(d_input, N, reduce_v1, BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);

    // V2: 连续线程
    cudaEventRecord(start);
    float sum2 = reduceOnGPU(d_input, N, reduce_v2, BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);

    // V3: 首次加载归约
    cudaEventRecord(start);
    float sum3 = reduceOnGPU(d_input, N, reduce_v3, BLOCK_SIZE * 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);

    printf("N = %d\n", N);
    printf("CPU 结果:            %.0f\n", cpuSum);
    printf("V1 (交错寻址):       %.0f  (%.3f ms)\n", sum1, ms1);
    printf("V2 (连续线程):       %.0f  (%.3f ms)\n", sum2, ms2);
    printf("V3 (首次加载归约):   %.0f  (%.3f ms)\n", sum3, ms3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    free(h_input);
    return 0;
}
```

**归约优化要点**：
1. 让连续线程执行工作，减少 Warp 分歧
2. 加载时就做一次归约，减少空闲线程
3. 最后 Warp 内归约可以用 `__shfl_down_sync` 避免共享内存
4. Warp是一个执行单元，包含32个线程。每个线程执行相同的指令，但访问不同的数据。当线程访问共享内存时，如果多个线程访问同一 bank 的不同地址，就会产生 bank conflict，导致访问被串行化，严重影响性能。通过在共享内存数组中添加 padding，可以避免多个线程访问同一 bank，从而提高性能。

---

# 第四部分：进阶主题

## 16. Unified Memory（统一内存）

CUDA 6.0+ 提供统一内存，让 CPU 和 GPU 共享同一地址空间，无需手动 `cudaMemcpy`。

```cuda
// file: unified_memory.cu
// 编译: nvcc -o unified_memory unified_memory.cu && ./unified_memory

#include <stdio.h>
#include <math.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int N = 1 << 20;

    // cudaMallocManaged: CPU 和 GPU 都能直接访问这块内存
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // 直接在 CPU 端初始化，无需 cudaMemcpy
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 启动 kernel，也无需 cudaMemcpy
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    saxpy<<<gridSize, blockSize>>>(N, 3.0f, x, y);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 直接在 CPU 端读取结果
    // y[i] = 3.0 * 1.0 + 2.0 = 5.0
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 5.0f));
    }
    printf("SAXPY: y = 3*x + y, where x=1, y=2\n");
    printf("期望结果: 5.0\n");
    printf("最大误差: %f\n", maxError);
    printf("结果: %s\n", maxError < 1e-5 ? "正确" : "错误");

    // 统一内存的优缺点
    printf("\n--- Unified Memory 特点 ---\n");
    printf("优点: 编程简单，无需手动管理数据搬运\n");
    printf("缺点: 页面迁移有开销，性能可能不如手动管理\n");
    printf("建议: 快速原型开发用 Unified Memory，\n");
    printf("      性能关键路径用手动 cudaMemcpy\n");

    cudaFree(x);
    cudaFree(y);
    return 0;
}
```

### Unified Memory 预取优化

```cuda
// 提示运行时提前迁移数据到 GPU
cudaMemPrefetchAsync(x, N * sizeof(float), deviceId, stream);

// 提示运行时提前迁移数据到 CPU
cudaMemPrefetchAsync(x, N * sizeof(float), cudaCpuDeviceId, stream);
```

---

## 17. cuBLAS 基础

cuBLAS 是 NVIDIA 提供的线性代数库，提供高度优化的矩阵运算。

```cuda
// file: cublas_demo.cu
// 编译: nvcc -o cublas_demo cublas_demo.cu -lcublas && ./cublas_demo

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

// 打印矩阵（小矩阵调试用）
void printMatrix(const float *m, int rows, int cols, const char *name) {
    printf("%s:\n", name);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // cuBLAS 使用列优先存储
            printf("%6.1f ", m[c * rows + r]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // C = alpha * A * B + beta * C
    // A: M×K, B: K×N, C: M×N
    const int M = 3, K = 4, N = 2;
    float alpha = 1.0f, beta = 0.0f;

    // 列优先存储初始化
    // A (3×4):
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    float h_A[] = {1, 5, 9,   2, 6, 10,   3, 7, 11,   4, 8, 12};  // 列优先

    // B (4×2):
    //  1  2
    //  3  4
    //  5  6
    //  7  8
    float h_B[] = {1, 3, 5, 7,   2, 4, 6, 8};  // 列优先

    float h_C[M * N] = {0};

    printMatrix(h_A, M, K, "A");
    printMatrix(h_B, K, N, "B");

    // Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 矩阵乘法: C = A * B
    // cublasSgemm: Single precision GEneral Matrix Multiply
    cublasSgemm(handle,
                CUBLAS_OP_N,  // A 不转置
                CUBLAS_OP_N,  // B 不转置
                M, N, K,      // 维度
                &alpha,
                d_A, M,       // A 及其 leading dimension
                d_B, K,       // B 及其 leading dimension
                &beta,
                d_C, M);      // C 及其 leading dimension

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(h_C, M, N, "C = A × B");

    // 手动验证: C[0][0] = 1*1 + 2*3 + 3*5 + 4*7 = 1+6+15+28 = 50
    printf("验证 C[0][0]: 1*1 + 2*3 + 3*5 + 4*7 = %d (got %.0f)\n",
           1+6+15+28, h_C[0]);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
```

**注意**：cuBLAS 使用**列优先（Column-major）**存储，与 C 语言的行优先不同。

---

## 18. Thrust 库

Thrust 是 CUDA 的高级 C++ 模板库，提供类似 STL 的接口，让 GPU 编程像写 CPU 代码一样简单。

```cuda
// file: thrust_demo.cu
// 编译: nvcc -o thrust_demo thrust_demo.cu && ./thrust_demo

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>

// 自定义函数对象
struct square {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

int main() {
    const int N = 10;

    // ===== 1. 基本容器 =====
    printf("=== Thrust 容器 ===\n");

    // host_vector 在 CPU 内存，device_vector 在 GPU 显存
    thrust::host_vector<float> h_vec(N);
    for (int i = 0; i < N; i++) h_vec[i] = N - i;  // [10, 9, 8, ..., 1]

    // 自动拷贝到 GPU
    thrust::device_vector<float> d_vec = h_vec;
    printf("大小: %zu\n", d_vec.size());

    // ===== 2. 排序 =====
    printf("\n=== 排序 ===\n");
    printf("排序前: ");
    thrust::copy(d_vec.begin(), d_vec.end(),
                 std::ostream_iterator<float>(std::cout, " "));
    printf("\n");

    thrust::sort(d_vec.begin(), d_vec.end());

    printf("排序后: ");
    thrust::copy(d_vec.begin(), d_vec.end(),
                 std::ostream_iterator<float>(std::cout, " "));
    printf("\n");

    // ===== 3. 归约 =====
    printf("\n=== 归约 ===\n");
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    printf("总和: %.0f\n", sum);

    float maxVal = thrust::reduce(d_vec.begin(), d_vec.end(), -1e30f, thrust::maximum<float>());
    printf("最大值: %.0f\n", maxVal);

    // ===== 4. 变换 =====
    printf("\n=== 变换 (平方) ===\n");
    thrust::device_vector<float> d_result(N);
    thrust::transform(d_vec.begin(), d_vec.end(), d_result.begin(), square());

    printf("原始: ");
    thrust::copy(d_vec.begin(), d_vec.end(),
                 std::ostream_iterator<float>(std::cout, " "));
    printf("\n");
    printf("平方: ");
    thrust::copy(d_result.begin(), d_result.end(),
                 std::ostream_iterator<float>(std::cout, " "));
    printf("\n");

    // ===== 5. 序列生成 + 计数 =====
    printf("\n=== 序列 & 计数 ===\n");
    thrust::device_vector<int> d_ints(20);
    thrust::sequence(d_ints.begin(), d_ints.end(), 0);  // [0, 1, 2, ..., 19]

    struct is_even {
        __host__ __device__
        bool operator()(int x) const { return x % 2 == 0; }
    };
    int evenCount = thrust::count_if(d_ints.begin(), d_ints.end(), is_even());
    printf("0-19 中偶数个数: %d\n", evenCount);

    printf("\n--- Thrust 优势 ---\n");
    printf("1. 无需手动管理 cudaMalloc / cudaMemcpy\n");
    printf("2. 类 STL 接口，学习成本低\n");
    printf("3. 自动选择最优算法实现\n");
    printf("4. 支持自定义函数对象\n");

    return 0;
}
```

---

## 19. 性能分析工具

### nvprof（CUDA 11 前）/ Nsight Compute（推荐）

```bash
# --- nvprof（旧版但简单） ---
nvprof ./your_program

# 详细 kernel 信息
nvprof --print-gpu-trace ./your_program

# 内存传输统计
nvprof --print-gpu-summary ./your_program

# --- Nsight Compute（新版推荐） ---
ncu ./your_program

# 详细分析特定 kernel
ncu --set full -k "kernelName" ./your_program

# 生成报告文件
ncu -o report ./your_program
# 用 Nsight Compute GUI 打开 report.ncu-rep

# --- Nsight Systems（系统级时间线） ---
nsys profile ./your_program
# 用 Nsight Systems GUI 打开 report.nsys-rep
```

### 代码内计时

```cuda
// file: profiling.cu
// 编译: nvcc -o profiling profiling.cu && ./profiling

#include <stdio.h>

__global__ void work(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        for (int i = 0; i < 1000; i++) x = sinf(x) + 1.0f;
        data[idx] = x;
    }
}

int main() {
    const int N = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    // ===== 方法 1: cudaEvent 计时（推荐） =====
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    work<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cudaEvent 计时: %.3f ms\n", ms);

    // ===== 方法 2: CUDA NVTX 标记（配合 Nsight 使用） =====
    // 需要链接 -lnvToolsExt
    // #include <nvToolsExt.h>
    // nvtxRangePush("My Kernel");
    // work<<<...>>>(...);
    // nvtxRangePop();

    printf("\n--- 性能分析建议 ---\n");
    printf("1. 用 cudaEvent 做基本计时\n");
    printf("2. 用 ncu 分析单个 kernel 的瓶颈\n");
    printf("3. 用 nsys 看整体时间线（含数据传输）\n");
    printf("4. 关注指标: 占用率、内存带宽利用率、计算吞吐量\n");
    printf("5. 常见瓶颈: 内存带宽 > 计算 > 延迟\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    return 0;
}
```

### 关键性能指标

| 指标 | 含义 | 目标 |
|------|------|------|
| Occupancy | SM 上活跃 Warp 占最大 Warp 的比例 | > 50% |
| Memory Throughput | 全局内存带宽利用率 | 接近硬件峰值 |
| Compute Throughput | 计算单元利用率 | 接近硬件峰值 |
| Warp Execution Efficiency | Warp 中活跃线程比例 | 100%（无分歧） |

---

## 20. CUDA 在 vLLM 中的应用

vLLM 是一个高性能 LLM 推理引擎，大量使用 CUDA 来加速推理过程。

### vLLM 中 CUDA 的使用方式

```
vLLM 架构中 CUDA 的角色：

Python 层 (vllm/)
    │
    ├── PyTorch 操作 → 自动使用 CUDA kernel
    │
    ├── 自定义 CUDA kernel (csrc/)
    │   ├── attention/         ← PagedAttention 核心
    │   ├── quantization/      ← 量化相关 kernel
    │   └── ops.h              ← 自定义操作注册
    │
    └── 第三方库
        ├── FlashAttention     ← 高效注意力
        ├── cuBLAS             ← 矩阵乘法
        └── NCCL               ← 多 GPU 通信
```

### 示例：理解 vLLM 中 CUDA 与 Python 的桥接

```python
# vLLM 通过 PyTorch 的 C++ 扩展机制注册自定义 CUDA 操作
# 位置: csrc/ops.h

# Python 端调用方式:
import torch

# 1. 直接使用 PyTorch（底层自动调度到 CUDA）
output = torch.matmul(query, key.transpose(-2, -1))

# 2. 调用 vLLM 自定义 CUDA kernel
from vllm._C import ops
ops.paged_attention_v1(...)  # 调用 csrc/ 中的 CUDA 实现
```

### vLLM 中的关键 CUDA 优化技术

```
1. PagedAttention
   - 核心创新：将 KV Cache 分页管理（类似操作系统虚拟内存）
   - CUDA 实现：自定义 attention kernel，支持不连续内存的高效读取
   - 位置: csrc/attention/

2. 量化 Kernel
   - 支持 GPTQ、AWQ、SqueezeLLM 等量化格式
   - 自定义 CUDA kernel 实现高效的反量化 + GEMM 融合
   - 位置: csrc/quantization/

3. 融合操作 (Fused Operations)
   - 将多个小操作融合为一个 kernel，减少 kernel 启动开销和内存访问
   - 例: fused_add_rms_norm（RMS 归一化 + 残差加法融合）

4. FlashAttention 集成
   - 利用 tiling 技术减少 HBM 访问
   - IO-aware 的注意力算法
```

### 学习路径建议

```
CUDA 基础（本手册 1-9 章）
    ↓
性能优化（本手册 10-15 章）
    ↓
阅读 vLLM 自定义 kernel（csrc/ 目录）
    ↓
理解 PagedAttention CUDA 实现
    ↓
尝试编写/修改 vLLM kernel
```

---

## 附录 A：常用编译选项

```bash
# 基本编译
nvcc -o program program.cu

# 指定 GPU 架构（以 A100 为例）
nvcc -arch=sm_80 -o program program.cu

# 开启优化
nvcc -O3 -o program program.cu

# 调试模式
nvcc -g -G -o program program.cu

# 链接 cuBLAS
nvcc -o program program.cu -lcublas

# 链接 cuRAND
nvcc -o program program.cu -lcurand

# 多架构编译（兼容多种 GPU）
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_89,code=sm_89 \
     -o program program.cu
```

## 附录 B：常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `illegal memory access` | 越界访问显存 | 检查索引计算，用 `cuda-memcheck` |
| `too many resources requested for launch` | Block 线程数超限或寄存器不足 | 减小 Block 大小 |
| `misaligned address` | 内存未对齐 | 确保数据类型对齐 |
| 结果全零 | 忘记 `cudaMemcpy` 或 kernel 未执行 | 检查错误码 |
| 结果随机 | 竞态条件 | 使用原子操作或 `__syncthreads()` |
| kernel 很慢 | 非合并访存/分支分歧 | 用 `ncu` 分析 |

```bash
# 内存检查工具
compute-sanitizer ./your_program

# 竞态条件检查
compute-sanitizer --tool racecheck ./your_program
```

## 附录 C：推荐学习资源

1. **NVIDIA 官方文档**: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. **CUDA by Example**: 入门经典书籍
3. **Professional CUDA C Programming**: 进阶书籍
4. **NVIDIA Developer Blog**: 性能优化案例
5. **vLLM 源码**: `csrc/` 目录下的真实 CUDA 工程代码
