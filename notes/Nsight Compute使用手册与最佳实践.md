# NVIDIA Nsight Compute 使用手册与最佳实践

NVIDIA Nsight Compute（ncu）是一个**CUDA Kernel 级别的性能分析工具**，用于深入分析单个 GPU Kernel 的执行效率。它回答的核心问题是：**一个 Kernel 为什么慢？是计算瓶颈还是内存瓶颈？如何优化？**

---

## 核心概念

| 概念 | 说明 |
|------|------|
| **ncu** | Nsight Compute 的命令行工具 |
| **ncu-ui** | Nsight Compute 的 GUI 工具 |
| **ncu-rep** | 分析报告文件（`.ncu-rep`） |
| **Section（节）** | 一组逻辑关联的指标集合（如 SpeedOfLight、Occupancy） |
| **Section Set（节集）** | 多个 Section 的预设组合（basic / default / full） |
| **Replay（重放）** | ncu 通过多次重放同一个 Kernel 来收集不同组的指标 |
| **Roofline（屋顶线）** | 可视化 Kernel 是计算瓶颈还是内存瓶颈的模型 |
| **Baseline（基线）** | 用于对比优化前后 Kernel 性能的参考数据 |

### Nsight Systems vs Nsight Compute

| 维度 | Nsight Systems (nsys) | Nsight Compute (ncu) |
|------|----------------------|---------------------|
| 分析对象 | 整个应用（CPU + GPU + 通信） | 单个 CUDA Kernel |
| 分析深度 | 宏观时间线 | 微观硬件指标 |
| 核心问题 | 时间花在**哪里**？ | Kernel **为什么**慢？ |
| 开销 | 低 | 较高（需要重放 Kernel） |
| 工作流 | **第一步**：定位瓶颈 Kernel | **第二步**：深入分析该 Kernel |

---

## 一、安装与验证

```bash
# 随 CUDA Toolkit 安装（通常已包含）
# 检查是否可用
ncu --version
# 输出示例：NVIDIA (R) Nsight Compute Command Line Profiler
#           Copyright (c) 2018-2025 NVIDIA Corporation
#           Version 2025.2.1.0 (build ...)

# 常见路径
# /usr/local/cuda/bin/ncu
# /usr/local/cuda/nsight-compute-*/ncu

# 确保在 PATH 中
export PATH=/usr/local/cuda/bin:$PATH
```

---

## 二、命令行基础

### 2.1 基本用法

```bash
# 最基础：profile 所有 Kernel（默认指标集）
ncu ./my_cuda_app

# 指定输出报告文件
ncu -o report ./my_cuda_app

# Profile Python 脚本中的 CUDA Kernel
ncu python my_script.py

# 输出文件：report.ncu-rep（可在 GUI 打开）
```

### 2.2 常用命令一览

| 命令/选项 | 功能 | 示例 |
|----------|------|------|
| `ncu` | 启动 profiling | `ncu ./app` |
| `ncu -o <file>` | 保存报告 | `ncu -o report ./app` |
| `ncu --open-in-ui` | 采集后自动打开 GUI | `ncu --open-in-ui ./app` |
| `ncu --import <file>` | 在 CLI 中查看报告 | `ncu --import report.ncu-rep` |
| `ncu --query-metrics` | 列出可用指标 | `ncu --query-metrics` |
| `ncu --list-sections` | 列出可用 Section | `ncu --list-sections` |
| `ncu --list-sets` | 列出可用 Section Set | `ncu --list-sets` |
| `ncu --version` | 查看版本 | `ncu --version` |

---

## 三、指标收集控制

### 3.1 Section Set（指标集）

`--set` 控制收集的指标范围，从少到多：

```bash
# basic：最少指标（启动参数 + 利用率概览）
ncu --set basic -o report ./app

# default：默认指标集（大多数场景够用）
ncu --set default -o report ./app

# full：所有指标（包含 Roofline、所有内存层级、指令分析等）
ncu --set full -o report ./app

# roofline：只收集 Roofline 相关指标
ncu --set roofline -o report ./app
```

| Set | 指标量 | Replay 次数 | 开销 | 适用场景 |
|-----|-------|------------|------|---------|
| `basic` | 最少 | 1-2 | 低 | 快速概览 |
| `default` | 中等 | 3-5 | 中 | 日常分析 |
| `full` | 全部 | 8-12+ | 高 | 深度分析、Roofline |
| `roofline` | Roofline 相关 | 3-4 | 中 | 判断 Compute/Memory Bound |

### 3.2 指定 Section

```bash
# 只收集特定 Section
ncu --section SpeedOfLight -o report ./app

# 收集多个 Section
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section WarpStateStats \
  -o report ./app

# 在 default 基础上追加 Section
ncu --set default --section InstructionStats -o report ./app
```

### 3.3 指定具体指标

```bash
# 收集特定指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o report ./app

# 收集指标组
ncu --metrics group:memory__chart -o report ./app

# 查询所有可用指标（输出很长）
ncu --query-metrics > all_metrics.txt

# 搜索特定指标
ncu --query-metrics 2>&1 | grep occupancy
ncu --query-metrics 2>&1 | grep throughput
```

---

## 四、Kernel 过滤

实际应用会启动成千上万个 Kernel，必须过滤到关注的目标：

### 4.1 按 Kernel 名称过滤

```bash
# 只 profile 名称包含 "attention" 的 Kernel
ncu --kernel-name "attention" -o report ./app

# 正则匹配
ncu --kernel-name regex:"paged_attention.*" -o report ./app

# 排除特定 Kernel
ncu --kernel-name "!memset" -o report ./app
```

### 4.2 按启动序号过滤

```bash
# 只 profile 第 10 个启动的 Kernel
ncu --launch-skip 9 --launch-count 1 -o report ./app

# profile 第 100-104 个 Kernel（跳过 99 个，采集 5 个）
ncu --launch-skip 99 --launch-count 5 -o report ./app

# profile 所有 Kernel（默认）
ncu --launch-count 0 -o report ./app
```

### 4.3 组合过滤

```bash
# 名称包含 "gemm" 的前 3 个 Kernel
ncu --kernel-name "gemm" --launch-count 3 -o report ./app

# 跳过前 50 个 attention kernel，采集下一个
ncu --kernel-name "attention" --launch-skip 50 --launch-count 1 -o report ./app
```

---

## 五、重放与时钟控制

### 5.1 Replay 模式

ncu 通过多次**重放（Replay）**同一个 Kernel 来收集不同组的指标：

```bash
# 默认：Kernel 级别重放（最精确）
ncu --replay-mode kernel -o report ./app

# 应用级别重放（整个应用重跑多次，适合有副作用的 Kernel）
ncu --replay-mode application -o report ./app

# 禁用缓存清除（默认每次重放前清 L2 cache）
ncu --cache-control none -o report ./app

# 应用重放 + 不清缓存（最接近真实运行状态）
ncu --replay-mode application --cache-control none -o report ./app
```

| Replay 模式 | 行为 | 优势 | 劣势 |
|-------------|------|------|------|
| `kernel` | 只重放单个 Kernel | 快，精确隔离 | 缓存状态可能不真实 |
| `application` | 整个应用重跑 | 缓存状态真实 | 慢得多 |

### 5.2 时钟控制

```bash
# 默认：ncu 锁定 SM 时钟（确保结果可重复）
ncu -o report ./app

# 不锁定时钟（更接近真实运行频率，但结果可能波动）
ncu --clock-control none -o report ./app

# 与 nsys 对比时，建议手动锁定时钟
sudo nvidia-smi --lock-gpu-clocks=1200,1200
ncu --clock-control none -o report ./app
sudo nvidia-smi --reset-gpu-clocks
```

---

## 六、核心 Section 详解

### 6.1 GPU Speed Of Light（SOL）

**最重要的 Section**，一眼判断 Kernel 的瓶颈类型。

```
┌─────────────────────────────────────────────────────────┐
│              GPU Speed Of Light Throughput               │
├──────────────────────────┬──────────────────────────────┤
│ SM Throughput            │  45.2% of peak               │
│ Memory Throughput        │  78.6% of peak               │
│ Duration                 │  1.23 ms                     │
│ Elapsed Cycles           │  1,476,000                   │
│ SM Frequency             │  1.20 GHz                    │
│ DRAM Frequency           │  1.215 GHz                   │
└──────────────────────────┴──────────────────────────────┘

解读：
├─ Memory Throughput (78.6%) >> SM Throughput (45.2%)
│  → 这个 Kernel 是 Memory Bound（内存瓶颈）
│
├─ SM Throughput >> Memory Throughput
│  → Compute Bound（计算瓶颈）
│
└─ 两者都低
   → Latency Bound（可能是 Occupancy 不足或 Warp Stall）
```

**判断规则：**

| SM Throughput | Memory Throughput | 瓶颈类型 | 优化方向 |
|--------------|------------------|---------|---------|
| 高 (>60%) | 低 | Compute Bound | 算法优化、减少指令数 |
| 低 | 高 (>60%) | Memory Bound | 减少内存访问、提高缓存命中 |
| 低 | 低 | Latency Bound | 提高 Occupancy、减少 Warp Stall |
| 高 | 高 | 接近峰值 | 已经很好，优化空间有限 |

### 6.2 Roofline Chart（屋顶线图）

```
         性能 (FLOPS)
           ▲
           │         ┌──────────────── Peak Compute (平顶)
           │        ╱│
           │       ╱ │
           │      ╱  │       ★ Kernel A (Compute Bound)
           │     ╱   │
           │    ╱    │
           │   ╱ ★ Kernel B (Memory Bound)
           │  ╱      │
           │ ╱       │
           │╱        │
           └─────────┴──────────────► 算术强度 (FLOPS/Byte)
                Ridge Point
                (脊点)
```

- **斜线左下方** = Memory Bound（内存带宽是瓶颈）
- **水平线下方** = Compute Bound（计算能力是瓶颈）
- **脊点（Ridge Point）** = 两者的交界点
- **★ 越接近屋顶线** = 效率越高
- **★ 离屋顶线越远** = 优化空间越大

```bash
# 收集 Roofline 数据
ncu --set roofline -o report ./app

# 或在 full set 中包含
ncu --set full -o report ./app
```

### 6.3 Memory Workload Analysis（内存负载分析）

详细展示各级内存的吞吐和效率：

```
┌─────────────────────────────────────────────────────┐
│           Memory Workload Analysis                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Global Memory     ───►  L2 Cache  ───►  L1 Cache   │
│  (DRAM/HBM)              │                │         │
│                          │                │         │
│  Throughput: 1.2 TB/s    Hit: 65%         Hit: 45%  │
│  Utilization: 78%        │                │         │
│                          ▼                ▼         │
│                     Shared Memory    Registers      │
│                     Utilization: 30%                │
└─────────────────────────────────────────────────────┘

关注点：
├─ L2 Cache Hit Rate
│  └─ < 50% → 数据局部性差，考虑优化访存模式
│
├─ L1 Cache Hit Rate
│  └─ < 30% → 考虑使用 Shared Memory 手动缓存
│
├─ Shared Memory Utilization
│  └─ 很低但 Kernel memory bound → 可以用 shared memory 优化
│
└─ DRAM Throughput vs Peak
   └─ > 80% of peak → 已接近硬件极限，需要减少总访存量
```

### 6.4 Occupancy（占用率）

```
┌────────────────────────────────────────────────┐
│               Occupancy                         │
├────────────────────────────────────────────────┤
│ Theoretical Occupancy:    75.0%                │
│ Achieved Occupancy:       62.5%                │
│                                                │
│ Occupancy Limiters:                            │
│   Registers:              32 per thread        │
│   Shared Memory:          4096 bytes per block │
│   Block Size:             256 threads          │
│   Max Blocks per SM:      8                    │
└────────────────────────────────────────────────┘

解读：
├─ Theoretical（理论） vs Achieved（实际）差距大
│  → 负载不均衡或 Kernel 执行时间过短
│
├─ Limiter 是 Registers
│  → 可用 __launch_bounds__ 限制寄存器，或减少局部变量
│
├─ Limiter 是 Shared Memory
│  → 减少共享内存使用，或调整 block size
│
└─ Limiter 是 Block Size
   → 增大 block size（但不超过 1024）
```

**Occupancy 计算器**：ncu GUI 内置了 Occupancy Calculator，点击 Section 标题栏的计算器图标即可打开，可以交互式调整参数观察 Occupancy 变化。

### 6.5 Warp State Statistics（Warp 状态统计）

```
┌──────────────────────────────────────────────────┐
│          Warp State Statistics                     │
├──────────────────────────────────────────────────┤
│ Stall Reasons (% of cycles):                     │
│                                                  │
│   Long Scoreboard:     35.2%  ← 等待长延迟操作   │
│   Wait:                18.7%  ← 等待屏障同步     │
│   Not Selected:        15.3%  ← 调度器选了别的   │
│   Short Scoreboard:    12.1%  ← 等待短延迟操作   │
│   Memory Throttle:      8.4%  ← 内存子系统拥塞   │
│   Barrier:              5.6%  ← CTA 屏障等待     │
│   Dispatch Stall:       2.8%  ← 指令分发延迟     │
│   Other:                1.9%                     │
└──────────────────────────────────────────────────┘
```

**主要 Stall 原因及优化建议：**

| Stall 原因 | 含义 | 优化方向 |
|-----------|------|---------|
| **Long Scoreboard** | 等待全局内存 / L2 缓存加载完成 | 优化内存访问模式，预取，提高 Occupancy 隐藏延迟 |
| **Short Scoreboard** | 等待 Shared Memory / L1 操作 | 减少 bank conflict，优化 shared memory 访问 |
| **Wait** | 等待同步指令（__syncthreads 等） | 减少同步频率，平衡负载避免某些 warp 先到达 |
| **Barrier** | CTA 屏障等待 | 减少 barrier 前的代码分支，均匀分配工作 |
| **Not Selected** | 有足够 warp 但调度器选了其他 | 通常正常，说明 Occupancy 足够 |
| **Memory Throttle** | 内存子系统压力过大 | 减少并发内存请求，优化访存合并 |
| **Dispatch Stall** | 指令缓存/分发瓶颈 | 减少分支，简化控制流 |

### 6.6 Compute Workload Analysis（计算负载分析）

```
┌──────────────────────────────────────────────────┐
│         Compute Workload Analysis                 │
├──────────────────────────────────────────────────┤
│ Executed Instructions:     1,234,567              │
│ Issued Instructions:       1,456,789              │
│ Issue Efficiency:          84.7%                  │
│                                                  │
│ Instruction Mix:                                 │
│   FP32:         45.2%                            │
│   FP16/BF16:     0.0%   ← 未使用半精度！         │
│   Tensor Core:   0.0%   ← 未使用 Tensor Core！   │
│   Integer:      30.1%                            │
│   Memory:       20.5%                            │
│   Control:       4.2%                            │
└──────────────────────────────────────────────────┘

关注点（深度学习场景）：
├─ Tensor Core 使用率
│  └─ 为 0% → GEMM 未走 Tensor Core，检查数据类型
│
├─ FP16/BF16 比例
│  └─ 为 0% → 未使用混合精度，潜在 2x 加速
│
└─ Issue Efficiency
   └─ 远低于 100% → 存在指令发射瓶颈
```

### 6.7 Launch Statistics（启动参数统计）

```
┌──────────────────────────────────────────────────┐
│            Launch Statistics                       │
├──────────────────────────────────────────────────┤
│ Grid Size:             (128, 32, 1)              │
│ Block Size:            (256, 1, 1)               │
│ Threads:               1,048,576                 │
│ Registers Per Thread:  48                        │
│ Shared Memory:         8192 bytes                │
│ Waves Per SM:          4.27                      │
└──────────────────────────────────────────────────┘

关注点：
├─ Waves Per SM
│  ├─ < 1 → GPU 严重未充分利用（grid 太小）
│  ├─ 1-4 → 一般，"尾效应"可能明显
│  └─ > 4 → 良好，SM 可以持续有活跃 warp
│
├─ Registers Per Thread
│  └─ > 64 → 可能限制 Occupancy
│
└─ Block Size
   └─ < 128 → 通常太小，考虑增大
```

---

## 七、GUI 分析

### 7.1 打开报告

```bash
# 直接打开 GUI
ncu-ui report.ncu-rep

# 从 CLI 打开
ncu --import report.ncu-rep --page details

# 采集后自动打开
ncu --open-in-ui -o report ./app
```

远程服务器场景：
```bash
# 在服务器上采集
ncu --set full -o report ./app
# 传回本地
scp user@server:report.ncu-rep ./
# 本地打开
ncu-ui report.ncu-rep
```

### 7.2 GUI 布局

```
┌───────────────────────────────────────────────────────────┐
│  工具栏：导航、对比、过滤                                   │
├─────────────┬─────────────────────────────────────────────┤
│             │                                             │
│  Kernel     │  详情面板（Details）                          │
│  列表       │  ┌─ Summary ─────────────────────────────┐  │
│             │  │ GPU Speed Of Light                     │  │
│  kernel_1 ◄─┤  │ SM: 45%  Memory: 78%                  │  │
│  kernel_2   │  ├─ Roofline Chart ──────────────────────┤  │
│  kernel_3   │  │  [图形化屋顶线图]                      │  │
│  ...        │  ├─ Memory Workload Analysis ─────────────┤  │
│             │  │  L2 Hit: 65%  DRAM BW: 1.2 TB/s       │  │
│             │  ├─ Occupancy ───────────────────────────┤  │
│             │  │  Theoretical: 75%  Achieved: 62%       │  │
│             │  ├─ Warp State Stats ────────────────────┤  │
│             │  │  Long Scoreboard: 35%  Wait: 18%      │  │
│             │  └─ Source View ─────────────────────────┤  │
│             │     [CUDA 源代码 + 指标映射]               │  │
└─────────────┴─────────────────────────────────────────────┘
```

### 7.3 关键 GUI 操作

| 操作 | 方法 |
|------|------|
| 查看 Kernel 列表 | 左侧面板，按名称/时间/调用次数排序 |
| 切换 Section | 详情面板中上下滚动，或使用顶部 Section 标签 |
| 查看 Roofline | 展开 Speed Of Light 节，查看 Roofline Chart |
| 源码关联 | Source 页面，查看每行代码对应的指标 |
| 对比基线 | 菜单 > Add Baseline，选择另一个 .ncu-rep 文件 |
| Occupancy 计算器 | 点击 Occupancy Section 标题栏的计算器图标 |
| 指标详情 | 鼠标悬停在任何指标上，显示底层 metric 名和描述 |
| 导出数据 | File > Export，支持 CSV |

### 7.4 Baseline 对比（优化前后对比）

```bash
# 采集优化前
ncu --set full -o before ./app

# 修改代码...

# 采集优化后
ncu --set full -o after ./app

# 在 GUI 中对比
# 1. 打开 after.ncu-rep
# 2. 菜单 > Add Baseline > 选择 before.ncu-rep
# 3. 每个指标旁会显示变化量（↑↓ 和百分比）
```

---

## 八、Source View（源码级分析）

### 8.1 编译时启用行号信息

```bash
# 必须使用 --lineinfo 编译才能看到源码映射
nvcc --lineinfo -o my_app my_kernel.cu

# 或在 CMake 中
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
```

### 8.2 Source View 功能

```
┌──────────────────────────────────────────────────────────┐
│  Source: attention_kernels.cuh                            │
├──────┬────────────┬──────────────────────────────────────┤
│ Line │ Throughput │ Source Code                           │
├──────┼────────────┼──────────────────────────────────────┤
│  145 │   12.3%    │ float qk = scale * dot(q, k);        │
│  146 │    0.5%    │ qk += alibi_slope * offset;           │
│  147 │    8.7%    │ logits[idx] = qk;                     │
│  ...│            │                                       │
│  198 │   25.6%    │ float val = __expf(logits[i] - max);  │  ← 热点行
│  199 │   18.2%    │ accs[j] += val * v_vec;               │  ← 热点行
│  ...│            │                                       │
└──────┴────────────┴──────────────────────────────────────┘
```

可以精确看到每一行 CUDA 代码的：
- 指令吞吐量
- 内存吞吐量
- Stall 占比
- 执行的指令数

---

## 九、深度学习场景专项

### 9.1 Profile PyTorch 中的 Kernel

```bash
# 基础 profile（所有 Kernel）
ncu -o pytorch_report python train.py

# 只看 GEMM Kernel（矩阵乘法，通常是热点）
ncu --kernel-name regex:"gemm|cutlass|cublas" \
    --set full \
    --launch-count 5 \
    -o gemm_report \
    python train.py

# 只看 Attention Kernel
ncu --kernel-name regex:"attention|flash" \
    --set full \
    --launch-count 3 \
    -o attn_report \
    python train.py

# 跳过预热阶段（前 100 个 Kernel）
ncu --launch-skip 100 --launch-count 10 \
    --set full \
    -o steady_state \
    python train.py
```

### 9.2 vLLM Kernel 分析

```bash
# Profile vLLM 的 PagedAttention Kernel
VLLM_WORKER_MULTIPROC_METHOD=spawn \
ncu --kernel-name regex:"paged_attention" \
    --set full \
    --launch-skip 50 \
    --launch-count 3 \
    -o vllm_paged_attn \
    python benchmark.py

# Profile vLLM 的所有热点 Kernel
VLLM_WORKER_MULTIPROC_METHOD=spawn \
ncu --set default \
    --launch-skip 100 \
    --launch-count 20 \
    -o vllm_kernels \
    python benchmark.py
```

### 9.3 深度学习 Kernel 检查清单

```
对每个热点 Kernel，逐项检查：

□ 1. Compute vs Memory Bound？
     └─ 看 SOL 的 SM Throughput vs Memory Throughput

□ 2. 是否使用了 Tensor Core？
     └─ Compute Workload > Instruction Mix > Tensor Core %
     └─ 如果为 0%，检查数据类型（应为 FP16/BF16/INT8/FP8）

□ 3. Occupancy 是否足够？
     └─ Achieved Occupancy > 50% 通常 OK
     └─ 如果很低，查看 Limiter（寄存器/共享内存/block size）

□ 4. 主要 Warp Stall 原因？
     └─ Long Scoreboard 高 → 内存延迟大，提高 Occupancy 或优化访存
     └─ Barrier 高 → 同步开销大，检查 __syncthreads 位置

□ 5. 缓存效率？
     └─ L2 Hit Rate < 50% → 数据复用差
     └─ Shared Memory bank conflict → 检查 Memory Chart

□ 6. Grid/Block 配置合理？
     └─ Waves Per SM > 1（否则 SM 未充分利用）
     └─ Block Size >= 128（通常 256 最佳）
```

---

## 十、性能优化指南

### 10.1 Memory Bound Kernel 优化

```
问题：Memory Throughput 高，SM Throughput 低

策略 1：减少全局内存访问量
├─ 使用 Shared Memory 缓存重复访问的数据
├─ Kernel 融合（减少中间结果的写入和读取）
└─ 使用更小的数据类型（FP32 → FP16/BF16）

策略 2：提高访存效率
├─ 合并访存（Coalesced Access）：同一 warp 访问连续地址
├─ 减少 Shared Memory bank conflict
├─ 使用向量化加载（float4 代替 float）
└─ 对齐内存地址（128 byte 对齐）

策略 3：提高缓存命中率
├─ 数据分块（Tiling）以适配 L2/L1 缓存大小
├─ 调整数据布局（AoS → SoA）
└─ 使用 __ldg() 提示 L2 缓存
```

### 10.2 Compute Bound Kernel 优化

```
问题：SM Throughput 高，Memory Throughput 低

策略 1：减少指令数
├─ 使用快速数学函数（__expf, __sinf 等）
├─ 循环展开（#pragma unroll）
├─ 避免整数除法/取模（用位运算或乘法替代）
└─ 减少条件分支

策略 2：使用硬件加速单元
├─ Tensor Core（矩阵乘法，需要 FP16/BF16/INT8/FP8 数据类型）
├─ SFU（特殊函数单元，sin/cos/exp/rsqrt）
└─ 确保数据类型匹配（FP16 的吞吐通常是 FP32 的 2x）

策略 3：提高指令级并行（ILP）
├─ 每线程处理多个元素
├─ 交错计算和访存指令
└─ 减少依赖链长度
```

### 10.3 Latency Bound Kernel 优化

```
问题：SM Throughput 和 Memory Throughput 都低

策略 1：提高 Occupancy
├─ 减少每线程使用的寄存器（--maxrregcount 或 __launch_bounds__）
├─ 减少每 block 使用的共享内存
├─ 增大 block size（在合理范围内）
└─ 检查硬件限制（max blocks per SM）

策略 2：减少同步
├─ 减少 __syncthreads() 频率
├─ 使用 warp-level 原语替代 block-level 同步
├─ cooperative_groups 细粒度同步
└─ 平衡 warp 间的工作量

策略 3：减少分支发散
├─ warp 内的线程应走相同分支
├─ 将条件判断提升到 warp/block 级别
└─ 使用谓词（predication）替代分支
```

---

## 十一、完整命令示例集

### 快速概览

```bash
# 所有 Kernel 的基本信息
ncu --set basic -o quick ./app

# 默认指标集
ncu -o default_report ./app

# 带摘要的快速查看（不保存文件）
ncu --set basic --target-processes all python script.py
```

### 深度分析

```bash
# 全量指标（包含 Roofline）
ncu --set full -o full_report ./app

# 特定 Kernel 全量分析
ncu --kernel-name "my_kernel" \
    --set full \
    --launch-count 1 \
    -o deep_analysis \
    ./app
```

### 深度学习专项

```bash
# PyTorch 训练中的 GEMM Kernel
ncu --kernel-name regex:"gemm|cutlass" \
    --set full \
    --launch-skip 200 \
    --launch-count 5 \
    -o gemm_analysis \
    python train.py

# vLLM Attention Kernel
VLLM_WORKER_MULTIPROC_METHOD=spawn \
ncu --kernel-name regex:"paged_attention" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    --launch-skip 50 \
    --launch-count 3 \
    -o vllm_attn \
    python benchmark.py

# 检查 Tensor Core 使用情况
ncu --kernel-name regex:"gemm" \
    --section ComputeWorkloadAnalysis \
    --launch-count 3 \
    -o tensor_core_check \
    python train.py
```

### 对比分析

```bash
# 优化前
ncu --set full --kernel-name "my_kernel" --launch-count 1 -o before ./app_v1

# 优化后
ncu --set full --kernel-name "my_kernel" --launch-count 1 -o after ./app_v2

# CLI 对比
ncu --import before.ncu-rep --page raw
ncu --import after.ncu-rep --page raw
# 或在 GUI 中 Add Baseline 对比
```

### 高级选项

```bash
# 应用级重放 + 不清缓存（最真实的结果）
ncu --replay-mode application \
    --cache-control none \
    --clock-control none \
    --set full \
    -o realistic \
    ./app

# 采集源码级指标（需要 --lineinfo 编译）
ncu --set full \
    --import-source yes \
    --source-folders /path/to/source \
    -o with_source \
    ./app

# 导出为 CSV
ncu --import report.ncu-rep --csv > metrics.csv

# 报告序号递增（多次运行不覆盖）
ncu -o report_%i ./app
# 生成 report_0.ncu-rep, report_1.ncu-rep, ...
```

---

## 十二、最佳实践清单

### 采集阶段

1. **先用 nsys 找热点，再用 ncu 深入分析** — 不要一上来就用 ncu profile 全部 Kernel

2. **限制 Kernel 数量** — 使用 `--kernel-name` 和 `--launch-count` 过滤
   ```bash
   ncu --kernel-name "my_kernel" --launch-count 3 ...
   ```

3. **跳过预热阶段** — 特别是 PyTorch `torch.compile`、JIT 编译阶段
   ```bash
   ncu --launch-skip 100 ...
   ```

4. **按需选择指标集** — 不总是需要 `--set full`
   ```
   快速判断瓶颈 → --set default
   深度分析 → --set full
   只看 Roofline → --set roofline
   ```

5. **编译时加 `--lineinfo`** — 启用源码级分析
   ```bash
   nvcc --lineinfo my_kernel.cu
   ```

### 分析阶段

6. **先看 SOL（Speed Of Light）** — 一眼判断 Compute/Memory/Latency Bound

7. **看 Roofline 位置** — 离屋顶线越远，优化空间越大

8. **检查 Tensor Core** — 深度学习 GEMM 如果没用 Tensor Core，是最大的优化漏洞
   ```
   Compute Workload > Tensor Core Active = 0% → 数据类型不对
   ```

9. **Warp Stall 指导优化方向** — Long Scoreboard → 内存优化；Barrier → 同步优化

10. **用 Baseline 量化优化效果** — 每次改动后对比，确保改进可度量

### 深度学习专项

11. **优先检查数据类型** — FP32 → FP16/BF16 通常可获得 2x+ 加速（同时用上 Tensor Core）

12. **检查 Kernel Launch 配置** — Waves Per SM < 1 说明 Grid 太小

13. **关注 Kernel 融合机会** — 多个小 Kernel 可能融合为一个大 Kernel，减少启动开销和中间内存

14. **不要只看单个 Kernel** — 有时优化一个 Kernel 会影响其他 Kernel 的缓存行为

15. **对比不同实现** — 例如 FlashAttention vs PagedAttention，用 ncu 对比它们在相同输入下的表现

---

## 十三、端到端优化工作流

```
┌─────────────────────────────────────────────────────────┐
│                  完整优化工作流                            │
└─────────────────────────────────────────────────────────┘

步骤 1：宏观 Profiling（nsys）
──────────────────────────────
  nsys profile --trace=cuda,nvtx --stats=true -o overview python train.py
  │
  ├─ 查看 nsys stats：哪些 Kernel 占比最高？
  ├─ 查看时间线：GPU 是否有空闲间隙？
  └─ 确定 Top 3 热点 Kernel
        │
        ▼
步骤 2：快速分类（ncu default）
──────────────────────────────
  ncu --kernel-name "hot_kernel" --set default --launch-count 3 -o classify python train.py
  │
  ├─ 看 SOL：Compute Bound / Memory Bound / Latency Bound？
  ├─ 看 Occupancy：是否充分利用 GPU？
  └─ 初步判断优化方向
        │
        ▼
步骤 3：深度分析（ncu full）
──────────────────────────────
  ncu --kernel-name "hot_kernel" --set full --launch-count 1 -o deep python train.py
  │
  ├─ Roofline：离峰值多远？
  ├─ Memory Analysis：缓存命中率？访存模式？
  ├─ Warp State：主要 stall 原因？
  ├─ Compute Analysis：Tensor Core 使用？指令 mix？
  └─ Source View：哪些行是热点？
        │
        ▼
步骤 4：实施优化
──────────────────────────────
  基于分析结果进行代码修改：
  │
  ├─ Memory Bound → 访存优化 / 数据类型 / Kernel 融合
  ├─ Compute Bound → 算法优化 / Tensor Core / 快速数学
  └─ Latency Bound → 提高 Occupancy / 减少同步
        │
        ▼
步骤 5：验证优化效果
──────────────────────────────
  ncu --kernel-name "hot_kernel" --set full --launch-count 1 -o after python train.py
  │
  ├─ GUI 中 Add Baseline（before.ncu-rep）对比
  ├─ 检查目标指标是否改善
  ├─ 检查是否引入新的瓶颈
  └─ 如果满意 → 返回步骤 1 分析下一个热点
     如果不满意 → 返回步骤 3 继续分析
```

---

## 十四、常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `ncu: command not found` | 将 `/usr/local/cuda/bin` 加入 PATH |
| 权限不足 | Linux 需要 root 或 `CAP_SYS_ADMIN`；或设置 `/proc/sys/kernel/perf_event_paranoid` 为 1 |
| 非常慢（多次 replay） | 减少 Section（用 `--set basic` 或指定少量 Section），减少 `--launch-count` |
| Kernel 太多看不过来 | 用 `--kernel-name` 和 `--launch-skip/count` 精确过滤 |
| 结果与 nsys 的 duration 不一致 | ncu 默认锁 SM 时钟，nsys 不锁；用 `--clock-control none` 对齐 |
| 缓存命中率与真实不符 | ncu 默认清缓存；用 `--replay-mode application --cache-control none` |
| Docker 中权限问题 | 使用 `--cap-add SYS_ADMIN` 或 `--privileged` |
| Source View 无源码 | 编译时需加 `--lineinfo`（nvcc）或 `-g`（调试信息） |
| 报告文件过大 | 减少 `--launch-count`，减少 Section，或用 `--kernel-name` 过滤 |
| 多 GPU 场景 | 使用 `--target-processes all` 或 CUDA_VISIBLE_DEVICES 限定 |

---

> **参考来源**：
> - [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
> - [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
> - [Nsight Compute CLI Reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
> - [NVIDIA Blog: Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
> - [NASA HECC: GPU Kernel Profiling](https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html)
> - [Speed Up PyTorch Training by 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
> - [Fix GPU Bottlenecks: PyTorch Profiler + Nsight](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
> - [Deep Learning GPU Profiling with Nsight Compute](https://hackmd.io/@daya-shankar/deep-learning-gpu-profiling-with-nsight-compute)
> - [Nsight Compute with PyTorch](https://www.codegenes.net/blog/nsight-compute-pytorch/)
> - [UW-Madison GPU Profiling Guide](https://www.hep.wisc.edu/cms/comp/gpuprofiling.html)
