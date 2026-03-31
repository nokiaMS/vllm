# NVIDIA Nsight Systems 使用手册与最佳实践

NVIDIA Nsight Systems 是一个**系统级性能分析工具**，用于生成 CPU/GPU 的全局时间线视图，帮助开发者定位性能瓶颈和优化机会。它是 CUDA 应用性能调优的第一站工具。

---

## 核心概念

| 概念 | 说明 |
|------|------|
| **nsys** | Nsight Systems 的命令行工具 |
| **nsys-rep** | 分析报告文件，可在 GUI 中打开 |
| **Timeline（时间线）** | 横轴为运行时间，纵轴为 CPU 线程 / GPU 流 / API 调用的分层视图 |
| **Trace（追踪）** | 记录特定类型的事件（CUDA API、内核、内存拷贝等） |
| **NVTX** | NVIDIA Tools Extension，用于在代码中插入自定义标记 |
| **Nsight Compute (ncu)** | 另一个工具，用于单个 CUDA Kernel 的深度分析（与 nsys 互补） |

### Nsight Systems vs Nsight Compute

| 维度 | Nsight Systems (nsys) | Nsight Compute (ncu) |
|------|----------------------|---------------------|
| 分析粒度 | 系统级 / 应用级 | 单个 Kernel 级 |
| 输出 | 全局时间线 | Kernel 性能指标 |
| 用途 | 找瓶颈在哪里 | 分析瓶颈为什么慢 |
| 开销 | 低 | 较高（会重放 Kernel） |
| 工作流 | **先用 nsys** 定位问题 | **再用 ncu** 深入分析 |

---

## 一、安装

### 1.1 独立安装

```bash
# 从 NVIDIA 官网下载
# https://developer.nvidia.com/nsight-systems

# Linux（.deb）
sudo dpkg -i NsightSystems-linux-public-*.deb

# Linux（.run）
chmod +x NsightSystems-linux-public-*.run
./NsightSystems-linux-public-*.run
```

### 1.2 随 CUDA Toolkit 安装

Nsight Systems 通常随 CUDA Toolkit 一起安装：

```bash
# 检查是否已安装
nsys --version
# 输出示例：NVIDIA Nsight Systems version 2025.2.1.130-...

# 常见安装路径
# /usr/local/cuda/bin/nsys
# /opt/nvidia/nsight-systems/*/target-linux-x64/nsys
```

### 1.3 Python NVTX 库

```bash
# 用于在 Python 代码中添加 NVTX 标记
pip install nvtx
```

---

## 二、命令行基础（nsys）

### 2.1 基本用法

```bash
# 最基础的 profile 命令
nsys profile ./my_cuda_app

# 指定输出文件名
nsys profile -o my_report ./my_cuda_app

# Profile Python 脚本
nsys profile python train.py

# 输出报告文件：my_report.nsys-rep
```

### 2.2 常用命令一览

| 命令 | 功能 | 示例 |
|------|------|------|
| `nsys profile` | 采集性能数据 | `nsys profile -o report ./app` |
| `nsys stats` | 从报告生成统计摘要 | `nsys stats report.nsys-rep` |
| `nsys export` | 导出为其他格式 | `nsys export --type=sqlite report.nsys-rep` |
| `nsys launch` | 启动应用但不立即采集 | `nsys launch --trace=cuda ./app` |
| `nsys start` | 开始采集（配合 launch） | `nsys start` |
| `nsys stop` | 停止采集 | `nsys stop` |
| `nsys shutdown` | 终止会话 | `nsys shutdown` |
| `nsys status` | 查看当前采集状态 | `nsys status` |
| `nsys --version` | 查看版本 | `nsys --version` |

---

## 三、nsys profile 详解

### 3.1 --trace（追踪选项）

`--trace` 指定要追踪的事件类型，是最重要的参数：

```bash
# 追踪 CUDA API 和 Kernel
nsys profile --trace=cuda -o report ./app

# 追踪 CUDA + NVTX 标注 + OS 运行时
nsys profile --trace=cuda,nvtx,osrt -o report ./app

# 追踪所有支持的事件
nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn -o report ./app
```

可用的 trace 选项：

| 选项 | 追踪内容 |
|------|---------|
| `cuda` | CUDA API 调用、Kernel 执行、内存拷贝 |
| `nvtx` | NVTX 标注（用户自定义标记） |
| `osrt` | OS 运行时库调用（pthread、文件 IO 等） |
| `cublas` | cuBLAS 库调用 |
| `cudnn` | cuDNN 库调用 |
| `opengl` | OpenGL API |
| `vulkan` | Vulkan API |
| `mpi` | MPI 通信 |
| `ucx` | UCX 通信（NCCL 底层） |
| `none` | 不追踪任何事件 |

```bash
# 推荐的深度学习 profile 命令
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --gpu-metrics-device=all \
  -o dl_report \
  python train.py
```

### 3.2 时间控制

```bash
# 延迟 10 秒后开始采集，采集 30 秒
nsys profile --delay=10 --duration=30 -o report ./app

# 用 CUDA Profiler API 控制（代码中 cudaProfilerStart/Stop）
nsys profile -c cudaProfilerApi -o report ./app

# 使用 NVTX capture range
nsys profile -c nvtx -p "my_range" -o report ./app
```

### 3.3 GPU 指标

```bash
# 采集所有 GPU 的硬件指标（SM 利用率、Tensor Core 使用率等）
nsys profile --gpu-metrics-device=all -o report ./app

# 追踪 CUDA 内存使用（注意：会显著降低性能）
nsys profile --cuda-memory-usage=true --trace=cuda -o report ./app
```

### 3.4 进程与线程

```bash
# 追踪 fork 出的子进程（多进程应用必须）
nsys profile --trace-fork-before-exec=true -o report ./app

# 追踪 CUDA Graph 节点
nsys profile --cuda-graph-trace=node -o report ./app

# 采样 CPU 调用栈（用于分析 CPU 瓶颈）
nsys profile --sample=cpu --backtrace=dwarf -o report ./app

# 采样频率（Hz）
nsys profile --sampling-frequency=1000 -o report ./app
```

### 3.5 输出与统计

```bash
# 采集后自动生成统计摘要
nsys profile --stats=true -o report ./app

# 强制覆盖已有报告
nsys profile --force-overwrite=true -o report ./app

# 限制报告文件大小（MB）
nsys profile --output-size-limit=500 -o report ./app
```

---

## 四、nsys stats（报告分析）

`nsys stats` 可以在命令行中直接输出统计分析结果，无需 GUI：

### 4.1 默认统计

```bash
# 生成所有默认统计报告
nsys stats report.nsys-rep

# 输出示例：
# CUDA API Statistics:
#  Time(%)  Total Time (ns)  Num Calls  Avg (ns)  Name
#   45.2    1,234,567,890     1,024     1,205,437  cudaLaunchKernel
#   30.1      823,456,789       512     1,608,314  cudaMemcpyAsync
#   ...
#
# CUDA Kernel Statistics:
#  Time(%)  Total Time (ns)  Instances  Avg (ns)   Name
#   25.3      692,345,678       256     2,704,475  paged_attention_v1_kernel<...>
#   18.7      512,234,567       512     1,000,458  layernorm_kernel<...>
```

### 4.2 指定报告类型

```bash
# 只看 CUDA Kernel 统计
nsys stats --report cuda_gpu_kern_sum report.nsys-rep

# 只看 CUDA API 统计
nsys stats --report cuda_api_sum report.nsys-rep

# 只看 GPU 内存操作
nsys stats --report cuda_gpu_mem_size_sum report.nsys-rep

# 只看 NVTX 范围统计
nsys stats --report nvtx_sum report.nsys-rep

# 组合多个报告
nsys stats \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  --report nvtx_sum \
  report.nsys-rep
```

### 4.3 输出格式

```bash
# 输出为 CSV（可用 Excel/Pandas 分析）
nsys stats --format csv --output report_stats report.nsys-rep

# 同时输出 CSV 和终端列表
nsys stats --format csv,column --output report_stats,- report.nsys-rep

# 导出为 SQLite 数据库（可自定义 SQL 查询）
nsys export --type=sqlite --output=report.sqlite report.nsys-rep
```

### 4.4 常用报告类型速查

| 报告名 | 内容 |
|--------|------|
| `cuda_api_sum` | CUDA API 调用汇总（时间、次数） |
| `cuda_api_trace` | CUDA API 调用逐条记录 |
| `cuda_gpu_kern_sum` | CUDA Kernel 执行汇总 |
| `cuda_gpu_kern_trace` | CUDA Kernel 逐条记录 |
| `cuda_gpu_mem_size_sum` | GPU 内存操作汇总 |
| `cuda_gpu_mem_time_sum` | GPU 内存操作时间汇总 |
| `nvtx_sum` | NVTX 标注范围汇总 |
| `osrt_sum` | OS 运行时调用汇总 |

---

## 五、GUI 时间线分析

### 5.1 打开报告

```bash
# 在本地机器上打开 GUI
nsys-ui report.nsys-rep

# 或者从 GUI 菜单 File > Open 选择 .nsys-rep 文件
```

如果在远程服务器上采集，需要把 `.nsys-rep` 文件传回本地：

```bash
scp user@server:/path/to/report.nsys-rep ./
# 然后在本地打开 nsys-ui
```

### 5.2 时间线布局

```
┌──────────────────────────────────────────────────────────┐
│                    时间线视图 (Timeline)                    │
│                                                          │
│  ┌─ Processes ──────────────────────────────────────────┐ │
│  │  ┌─ python (PID 12345) ────────────────────────────┐ │ │
│  │  │  ├─ CPU Threads                                  │ │ │
│  │  │  │  ├─ Thread 0  ████░░████░░████░░              │ │ │
│  │  │  │  ├─ Thread 1  ░░████░░████░░████              │ │ │
│  │  │  │  └─ ...                                       │ │ │
│  │  │  ├─ CUDA API     ▓▓░▓▓░▓▓░▓▓░▓▓░                │ │ │
│  │  │  ├─ NVTX         [forward][backward][optimizer]  │ │ │
│  │  │  └─ CUDA HW                                      │ │ │
│  │  │     ├─ GPU 0                                     │ │ │
│  │  │     │  ├─ Kernels  ▓▓▓░▓▓▓░▓▓▓░▓▓▓░             │ │ │
│  │  │     │  ├─ MemCpy   ▒░░░▒░░░▒░░░▒░░░             │ │ │
│  │  │     │  └─ Metrics  ━━━━━━━━━━━━━━━━              │ │ │
│  │  │     └─ GPU 1                                     │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Events View (底部面板) ─────────────────────────────┐ │
│  │  统计表格 / 事件详情 / 关联分析                        │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 5.3 关键行（Row）说明

| 行名 | 内容 | 关注点 |
|------|------|--------|
| **CPU Threads** | 每个线程的 API 调用和执行状态 | CPU 是否成为瓶颈 |
| **CUDA API** | cudaLaunchKernel、cudaMemcpy 等调用 | API 调用开销 |
| **NVTX** | 用户自定义的代码区域标记 | 关联代码逻辑与 GPU 活动 |
| **CUDA HW - Kernels** | GPU 上实际执行的 Kernel | Kernel 执行时间和空隙 |
| **CUDA HW - MemCpy** | GPU 内存拷贝操作 | H2D/D2H 数据传输瓶颈 |
| **GPU Metrics** | SM 利用率、Tensor Core 使用率等 | 硬件资源利用效率 |

### 5.4 GUI 操作技巧

| 操作 | 方法 |
|------|------|
| 缩放 | 鼠标滚轮 / 框选区域后右键 > Zoom to Selection |
| 平移 | 按住中键拖动 |
| 选择事件 | 左键点击时间线上的色块 |
| 查看统计 | 右键行名 > Show in Events View |
| 测量时间差 | 按住 Shift 拖动，显示时间间隔 |
| 搜索 Kernel | Ctrl+F 搜索 Kernel 名称 |
| 关联 CPU-GPU | 点击 CUDA API 调用，自动高亮对应的 GPU Kernel |
| 跳转到 ncu | 右键 Kernel > Profile Kernel（启动 Nsight Compute） |

### 5.5 精简视图（推荐起始视图）

打开报告后，默认信息量很大。推荐的精简步骤：

1. 折叠底部 Events View 面板
2. 折叠 CPU、GPU 和 Processes 节点
3. 只展开 **Processes > python > CUDA HW**
4. 重点查看 Kernel 执行的时间线和间隙（gap）

---

## 六、NVTX 代码标注

NVTX（NVIDIA Tools Extension）让你在代码中插入自定义标记，在时间线上显示为有意义的区域。

### 6.1 Python 中使用 NVTX

```python
import torch
import nvtx

# 方式一：上下文管理器
with nvtx.annotate("forward_pass", color="green"):
    output = model(input_data)

with nvtx.annotate("backward_pass", color="red"):
    loss.backward()

with nvtx.annotate("optimizer_step", color="blue"):
    optimizer.step()

# 方式二：使用 torch 内置 NVTX
torch.cuda.nvtx.range_push("forward")
output = model(input_data)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("backward")
loss.backward()
torch.cuda.nvtx.range_pop()

# 方式三：装饰器
@nvtx.annotate(message="data_preprocessing", color="yellow")
def preprocess(data):
    # ... 数据预处理逻辑
    return processed_data

# 方式四：条件标注（只在 profiling 时生效）
for epoch in range(num_epochs):
    with nvtx.annotate(f"epoch_{epoch}"):
        for batch_idx, (data, target) in enumerate(loader):
            with nvtx.annotate(f"batch_{batch_idx}"):
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
```

### 6.2 C++/CUDA 中使用 NVTX

```cpp
#include <nvtx3/nvToolsExt.h>

void forward_pass() {
    nvtxRangePushA("forward_pass");

    // ... 前向传播代码

    nvtxRangePop();
}

// 带颜色标注
void backward_pass() {
    nvtxEventAttributes_t attrs = {};
    attrs.version = NVTX_VERSION;
    attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrs.colorType = NVTX_COLOR_ARGB;
    attrs.color = 0xFFFF0000;  // 红色
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.message.ascii = "backward_pass";
    nvtxRangePushEx(&attrs);

    // ... 反向传播代码

    nvtxRangePop();
}
```

### 6.3 PyTorch 自动 NVTX 标注

```bash
# nsys 可以自动为 PyTorch 操作添加 NVTX 标注
nsys profile \
  --trace=cuda,nvtx \
  --pytorch=autograd-shapes-nvtx \
  -o report \
  python train.py

# 这会自动在时间线上标注每个 PyTorch op（linear、relu、conv2d 等）
# 包括 autograd 图的形状信息
```

---

## 七、vLLM 专项 Profiling

### 7.1 离线推理 Profiling

```bash
# 基本用法
nsys profile \
  -o vllm_report \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf

# 推荐设置环境变量（避免 fork 问题）
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  -o vllm_offline \
  --trace=cuda,nvtx \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  python my_vllm_script.py
```

### 7.2 服务端 Profiling

```bash
# 服务端需要指定延迟和持续时间
# 先让服务启动稳定，再开始采集
nsys profile \
  -o vllm_server \
  --trace=cuda,nvtx \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay=30 \
  --duration=60 \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000

# 在另一个终端发送请求进行压测
# 确保在 delay 之后、duration 之内发送请求
```

### 7.3 vLLM 关键参数说明

| 参数 | 说明 |
|------|------|
| `--trace-fork-before-exec=true` | vLLM 使用多进程，必须追踪 fork |
| `--cuda-graph-trace=node` | vLLM 使用 CUDA Graph，需要追踪 Graph 节点 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 使用 spawn 代替 fork，避免 profiling 冲突 |
| `--delay=N` | 服务端场景，等 N 秒后再开始采集 |
| `--duration=N` | 采集 N 秒后自动停止 |

### 7.4 vLLM Profiling 关注点

```
时间线中需要重点关注：

1. Prefill 阶段（首 token 延迟）
   ├─ Attention Kernel 耗时
   ├─ Linear/GEMM 耗时
   └─ CPU 调度开销

2. Decode 阶段（逐 token 生成）
   ├─ PagedAttention Kernel 耗时
   ├─ Kernel 之间的间隙（GPU idle）
   └─ CUDA Graph replay 耗时

3. KV Cache 操作
   ├─ cache_kernels 耗时
   └─ 内存拷贝（swap in/out）

4. 调度开销
   ├─ CPU 调度时间 vs GPU 执行时间
   └─ Batch 组装延迟
```

---

## 八、PyTorch 训练 Profiling

### 8.1 典型 Profile 命令

```bash
# 基础 PyTorch 训练 profile
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --gpu-metrics-device=all \
  --stats=true \
  -o pytorch_train \
  python train.py

# 配合 PyTorch 自动标注
nsys profile \
  --trace=cuda,nvtx \
  --pytorch=autograd-shapes-nvtx \
  --gpu-metrics-device=all \
  -o pytorch_train \
  python train.py
```

### 8.2 在代码中控制采集范围

```python
import torch

# 只采集第 5-7 个 iteration（避免采集预热阶段）
for i in range(num_iterations):
    if i == 5:
        torch.cuda.cudart().cudaProfilerStart()
    if i == 8:
        torch.cuda.cudart().cudaProfilerStop()
        break

    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

```bash
# 配合上述代码，使用 cudaProfilerApi 控制
nsys profile \
  -c cudaProfilerApi \
  --trace=cuda,nvtx \
  -o controlled_report \
  python train.py
```

### 8.3 torch.profiler 与 nsys 配合

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# torch.profiler 内置对 nsys 的支持
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (data, target) in enumerate(loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()

# torch.profiler 适合快速查看 → nsys 适合深入分析
# 两者可以互补使用
```

---

## 九、多 GPU / 分布式 Profiling

### 9.1 多 GPU 采集

```bash
# 自动追踪所有 GPU
nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-device=all \
  -o multi_gpu_report \
  python -m torch.distributed.launch --nproc_per_node=4 train.py

# 使用 torchrun
nsys profile \
  --trace=cuda,nvtx,ucx \
  --gpu-metrics-device=all \
  -o distributed_report \
  torchrun --nproc_per_node=4 train.py
```

### 9.2 MPI 应用

```bash
# 每个 rank 生成独立的报告
mpirun -np 4 nsys profile \
  --trace=cuda,nvtx,mpi \
  -o report_rank%q{OMPI_COMM_WORLD_RANK} \
  ./my_mpi_app
```

### 9.3 NCCL 通信分析

```bash
# 追踪 NCCL 通信（通过 UCX）
nsys profile \
  --trace=cuda,nvtx,ucx \
  --gpu-metrics-device=all \
  -o nccl_report \
  torchrun --nproc_per_node=8 train.py

# 在时间线中查看：
# - AllReduce / AllGather 等集合通信操作
# - 通信与计算的重叠程度
# - GPU 间同步等待时间
```

---

## 十、性能分析方法论

### 10.1 系统化分析流程

```
步骤 1：全局概览（nsys）
─────────────────────────
  nsys profile --trace=cuda,nvtx --stats=true -o overview ./app
  │
  ├─ 查看 nsys stats 输出，找到耗时最长的 Kernel
  ├─ 查看 GPU 利用率是否充分
  └─ 识别 GPU idle 间隙
        │
        ▼
步骤 2：定位瓶颈类型
─────────────────────────
  打开 GUI 时间线
  │
  ├─ CPU 密集？→ CPU 线程在忙，GPU 在等
  │   └─ 优化数据加载 / 减少 Python 开销
  │
  ├─ 内存传输密集？→ 大量 H2D/D2H 拷贝
  │   └─ 使用 pinned memory / 减少传输
  │
  ├─ Kernel 启动开销？→ 大量小 Kernel，间隙多
  │   └─ 使用 CUDA Graph / Kernel 融合
  │
  └─ Kernel 本身慢？→ 单个 Kernel 耗时长
      └─ 进入步骤 3
        │
        ▼
步骤 3：Kernel 深度分析（ncu）
─────────────────────────
  ncu --set full -o kernel_report ./app
  │
  ├─ 查看 Roofline Plot
  ├─ 分析 Memory Bound vs Compute Bound
  ├─ 检查 Occupancy
  └─ 查看 Warp 状态和指令吞吐
        │
        ▼
步骤 4：优化并验证
─────────────────────────
  应用优化 → 重新 nsys profile → 对比时间线
```

### 10.2 常见性能问题及诊断

| 症状 | 时间线表现 | 可能原因 | 解决方案 |
|------|-----------|---------|---------|
| GPU 空闲间隙多 | Kernel 之间有大段空白 | CPU 调度慢 / Python GIL | CUDA Graph / C++ 扩展 |
| 大量小 Kernel | 时间线上密密麻麻的窄条 | Kernel launch 开销 | Kernel 融合 / CUDA Graph |
| H2D 拷贝频繁 | 大量绿色 MemCpy 条 | 数据未常驻 GPU | Pinned memory / 预加载 |
| 单个 Kernel 很长 | 一个色块占据大片时间 | Kernel 效率低 | 用 ncu 分析优化 |
| GPU 利用率低 | Metrics 行显示 SM 使用率低 | 并行度不足 | 增大 batch size / 更多并发 |
| 通信等待长 | AllReduce 等操作耗时大 | 通信-计算未重叠 | 梯度分桶 / 通信-计算重叠 |

### 10.3 解读 GPU 利用率指标

```
GPU Metrics 行中的关键指标：

SM Active (%)       GPU 流多处理器活跃比例
                    ├─ > 80%  → GPU 利用充分
                    ├─ 50-80% → 有优化空间
                    └─ < 50%  → 严重不足，优先解决

Tensor Core Active  Tensor Core 使用比例
                    ├─ 深度学习训练应该很高
                    └─ 如果很低，检查是否用了合适的数据类型（FP16/BF16）

DRAM Bandwidth      显存带宽利用率
                    ├─ Memory bound kernel 应接近峰值
                    └─ 远低于峰值说明访存模式有问题
```

---

## 十一、最佳实践清单

### 采集阶段

1. **限制采集范围** — 只采集性能关键区域，避免采集启动/预热阶段
   ```bash
   nsys profile --delay=10 --duration=30 ...
   ```

2. **保持运行时间短** — nsys 不建议超过 5 分钟的采集，长时间采集会生成巨大文件
   ```bash
   nsys profile --duration=60 ...  # 1 分钟通常足够
   ```

3. **按需开启 trace** — 不要追踪所有事件，只启用需要的
   ```bash
   # 初次：只看 CUDA
   nsys profile --trace=cuda ...
   # 深入：加上 NVTX 和库调用
   nsys profile --trace=cuda,nvtx,cublas ...
   ```

4. **多次采集分离关注点** — 不同的 trace 选项在不同的 run 中采集，减少干扰
   ```bash
   # Run 1: CUDA + Kernel
   nsys profile --trace=cuda -o run1 ...
   # Run 2: CPU sampling
   nsys profile --sample=cpu --backtrace=dwarf -o run2 ...
   ```

5. **使用 NVTX 标注关键代码段** — 极大提升时间线的可读性

### 分析阶段

6. **先看统计摘要，再看时间线** — `nsys stats` 快速定位 Top N 耗时 Kernel

7. **关注 GPU idle gap** — 间隙代表 GPU 在等待，是最大的优化机会

8. **对比优化前后** — 每次优化后重新 profile，量化效果
   ```bash
   # 优化前
   nsys profile -o before ...
   # 优化后
   nsys profile -o after ...
   # 对比统计
   nsys stats before.nsys-rep > before_stats.txt
   nsys stats after.nsys-rep > after_stats.txt
   diff before_stats.txt after_stats.txt
   ```

9. **利用 nsys → ncu 工作流** — nsys 找到瓶颈 Kernel，ncu 分析优化方向

10. **远程采集，本地分析** — 在服务器上用 `nsys profile`，传回本地用 `nsys-ui` 查看

### 深度学习专项

11. **跳过前几个 iteration** — 避免采集 torch.compile / JIT 编译阶段

12. **使用 CUDA Graph trace** — 对使用 CUDA Graph 的框架（如 vLLM）必须加 `--cuda-graph-trace=node`

13. **关注 Tensor Core 利用率** — 如果 Tensor Core Active 很低，检查数据类型是否正确（应使用 FP16/BF16）

14. **检查数据加载** — 如果 GPU 在 iteration 之间空闲，很可能是 DataLoader 瓶颈
    ```bash
    # 追踪 OS 运行时调用以分析 IO
    nsys profile --trace=cuda,osrt ...
    ```

15. **分布式训练检查通信重叠** — AllReduce 应与下一层的前向计算重叠

---

## 十二、完整命令示例集

### 快速开始

```bash
# 最简单的 profile
nsys profile -o quick python train.py

# 带统计摘要
nsys profile --stats=true -o quick python train.py
```

### PyTorch 训练

```bash
nsys profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --gpu-metrics-device=all \
  --pytorch=autograd-shapes-nvtx \
  --stats=true \
  --force-overwrite=true \
  -o pytorch_train \
  python train.py
```

### vLLM 推理

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --trace=cuda,nvtx \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --gpu-metrics-device=all \
  --stats=true \
  --force-overwrite=true \
  -o vllm_inference \
  python benchmark_serving.py
```

### vLLM 服务端

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --trace=cuda,nvtx \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay=30 \
  --duration=60 \
  --gpu-metrics-device=all \
  --force-overwrite=true \
  -o vllm_server \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf
```

### 分布式训练

```bash
nsys profile \
  --trace=cuda,nvtx,ucx \
  --gpu-metrics-device=all \
  --stats=true \
  -o distributed \
  torchrun --nproc_per_node=4 train.py
```

### 仅 CPU 采样（分析 Python 开销）

```bash
nsys profile \
  --sample=cpu \
  --backtrace=dwarf \
  --sampling-frequency=2000 \
  --trace=none \
  -o cpu_only \
  python train.py
```

### 后处理分析

```bash
# 生成统计报告
nsys stats report.nsys-rep

# 导出为 SQLite 自定义分析
nsys export --type=sqlite --output=report.sqlite report.nsys-rep

# 用 SQL 查询分析（示例）
sqlite3 report.sqlite "
  SELECT shortName, sum(duration)/1e6 as total_ms, count(*) as count
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  GROUP BY shortName
  ORDER BY total_ms DESC
  LIMIT 20;
"
```

---

## 十三、常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `nsys: command not found` | 检查 CUDA Toolkit 是否安装，将 `/usr/local/cuda/bin` 加入 PATH |
| 报告文件巨大（>10GB） | 缩短 `--duration`，减少 `--trace` 选项，使用 `--output-size-limit` |
| 多进程应用只看到主进程 | 添加 `--trace-fork-before-exec=true` |
| CUDA Graph 看不到细节 | 添加 `--cuda-graph-trace=node` |
| `osrt` trace 导致 SIGILL | 改用 `--trace=cuda,nvtx`（去掉 osrt） |
| 远程服务器无法打开 GUI | 传回 `.nsys-rep` 到本地，用本地 nsys-ui 打开 |
| Docker 容器中运行报错 | 添加 `--cap-add=SYS_ADMIN` 或 `--privileged` |
| vLLM profiling fork 报错 | 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| 时间线上 Kernel 名称看不清 | C++ 编译时加 `-lineinfo`，或在 GUI 中搜索 |

---

> **参考来源**：
> - [Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
> - [Nsight Systems 2025.2 User Guide](https://docs.nvidia.com/nsight-systems/2025.2/UserGuide/index.html)
> - [vLLM Profiling 官方文档](https://docs.vllm.ai/en/stable/contributing/profiling/)
> - [NERSC NVIDIA Profiling Tools Guide](https://docs.nersc.gov/tools/performance/nvidiaproftools/)
> - [PyTorch Nsight Systems Profiling](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)
> - [Speed Up PyTorch Training by 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
> - [Navigating Nsight Systems for Efficient Profiling](https://henryhmko.github.io/posts/profiling/profiling.html)
> - [Red Hat vLLM Profiling Guide](https://developers.redhat.com/articles/2025/10/16/profiling-vllm-inference-server-gpu-acceleration-rhel)
