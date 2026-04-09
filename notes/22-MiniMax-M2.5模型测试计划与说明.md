# MiniMax-M2.5 模型验证测试计划与说明

---

**测试模型**：MiniMax-M2.5

**文档版本**：v1.0

**编制日期**：2026-04-08

---

## 一、测试概述

### 1.1 测试目标

对 MiniMax-M2.5 模型在多种硬件平台上进行全面的适配验证、性能测试与稳定性测试，涵盖单节点和分布式多节点两种部署场景，为生产环境选型与部署提供数据支撑。

### 1.2 测试范围

| 类别 | 内容 |
|------|------|
| 基本功能测试 | 模型加载、推理正确性、API 兼容性、多模态能力 |
| 性能测试 | 吞吐量、延迟（TTFT/TPOT/ITL）、不同并发与上下文长度 |
| 稳定性测试 | 7×24h 长时运行、显存泄漏检测 |
| 精度测试 | MMLU-Pro 等基准评测 |

### 1.3 重点指标与通过标准

| 指标 | 通过标准 | 说明 |
|------|----------|------|
| 推理吞吐量 | ≥ A100（H100）同卡数的 60% | 以 H100 为基准线 |
| TTFT（并发=16） | < 500 ms | 首 Token 延迟 |
| P99 TPOT | < 100 ms | 单 Token 生成延迟 |
| 集群稳定性 | 7×24h 无故障 | 显存无泄漏、服务不中断 |

### 1.4 测试总体顺序

```
环境准备 → 适配验证 → 单机测试 → 分布式测试 → 稳定性测试
```

---

## 二、测试环境

### 2.1 硬件配置

| 配置项 | Hygon BW1000 | Kunlun P800 | NVIDIA H100 |
|--------|-------------|-------------|-------------|
| 加速卡型号 | BW1000 | P800 OAM | H100 |
| 加速卡数量 | 8 | 8 | 8 |
| 精度支持 | FP16/BF16 | FP16/BF16 | FP16/BF16/FP8 |
| 显存 | 65520 MiB (~64 GB) | 98304 MiB (96 GB) | 80 GB HBM3 |
| 显存带宽 | — | — | 3.35 TB/s |
| TDP | 200 W | 400 W | 700 W |
| 互联方式 | XPULink + PCIe Gen4 x16 | — | NVLink 4.0 |
| CPU | Hygon C86 (128核) | Intel Xeon Platinum 8563C (208核) | Intel Xeon Platinum 8468 (192核) |
| 内存 | 503 GiB | 2.0 TiB | 2.0 TiB |
| 存储 | 437G + 1.7TiB | 446.6G + NVMe 4×3.5T | 894GB + 7TB×4 + 7TB + 25TB |

### 2.2 软件环境

| 组件 | Hygon BW1000 | Kunlun P800 | NVIDIA H100 |
|------|-------------|-------------|-------------|
| OS | Ubuntu 22.04.5 LTS | Ubuntu 22.04.5 LTS | Ubuntu 22.04.5 LTS |
| 内核 | 6.3.22-V1.2.0 | Kunlun/XPU Driver 5.0.21.26 | 570.133.20 / 580.126.09 |
| 加速卡 Toolkit | DTK-26.04-beta | XPU Container Runtime Hook 1.0.5 | CUDA 12.x |
| Docker | 28.0.4 | 28.4.0 | — |
| containerd | 2.1.1 | 1.7.28 | — |
| K8S | — | v1.28.2 | — |

### 2.3 模型与框架信息

| 项目 | Hygon BW1000 | Kunlun P800 | NVIDIA H100 |
|------|-------------|-------------|-------------|
| 模型 | MiniMax-M2.5 | MiniMax-M2.5 | MiniMax-M2.5 |
| 推理框架 | vLLM | vLLM | vLLM |
| vLLM 版本 | 0.11.0+das.opt1.rc2 | 0.14.5 | 0.15.1 |
| 数据精度 | BF16 / INT8 | FP16 / INT8 | FP16 / FP8 |
| Python | 3.10.12 | 3.10.15 | 3.12.3 |
| 通信库 | DTKRCCL | NCCL | NCCL |

---

## 三、测试场景与用例

### 3.1 性能测试

#### 3.1.1 场景一：标准性能测试（10K 输入 / 0.25K 输出）

| 参数 | 值 |
|------|-----|
| 数据分布 | random |
| 请求数 | 320 |
| 输入长度 | 10240 (10K tokens) |
| 输出长度 | 256 (0.25K tokens) |
| 并发数 | 1, 2, 4, 8, 10, 16, 32, 64, 80, 128 |

**vLLM 启动参数**：

| 参数 | BW1000 | P800 | H100 |
|------|--------|------|------|
| max-model-len | 196608 | 196608 | 196608 |
| max-num-seqs | 10 | 10 | 10 |
| max-num-batched-tokens | 8192 | 8192 | 8192 |
| gpu-memory-utilization | 0.95 | 0.95 | 0.85 |
| dp | 1 | 1 | 1 |
| tp | 8 | 8 | 8 |
| pp | 1 | 1 | 1 |
| enable-expert-parallel | True | False | True |
| tool-call-parser | minimax_m2 | minimax_m2 | minimax_m2 |
| reasoning-parser | minimax_m2 | minimax_m2 | minimax_m2 |

**压测命令示例**：

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model MiniMax-M2.5 \
    --dataset-type random \
    --num-prompts 320 \
    --random-input-len 10240 \
    --random-output-len 256 \
    --concurrency <N>
```

**结果记录表**：

| 并发 | 耗时 (s) | 输入 tokens | 输出 tokens | 请求吞吐 (req/s) | Token 吞吐 (tok/s) | 输出吞吐 (tok/s) | 总吞吐 (tok/s) |
|------|---------|------------|------------|-----------------|-------------------|-----------------|---------------|
| 1 | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — | — |
| 10 | — | — | — | — | — | — | — |
| 16 | — | — | — | — | — | — | — |
| 32 | — | — | — | — | — | — | — |
| 64 | — | — | — | — | — | — | — |
| 80 | — | — | — | — | — | — | — |
| 128 | — | — | — | — | — | — | — |

**延迟指标记录表（每个并发度填写）**：

| 指标 | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|------|-----------|-------------|----------|----------|
| TTFT | — | — | — | — |
| TPOT | — | — | — | — |
| ITL | — | — | — | — |

#### 3.1.2 场景二：长上下文测试（190K 输入 / 1K 输出）

| 参数 | 值 |
|------|-----|
| 数据分布 | random |
| 请求数 | 100 |
| 输入长度 | 194560 (190K tokens) |
| 输出长度 | 1024 (1K tokens) |
| 并发数 | 1, 2, 4, 8, 10 |

> 此场景测试模型在接近 max-model-len 上限时的 prefill 性能与稳定性。

#### 3.1.3 场景三：大批量测试（90K 输入 / 2K 输出）

| 参数 | 值 |
|------|-----|
| 数据分布 | random |
| 请求数 | 1000 |
| 输入长度 | 90000 (90K tokens) |
| 输出长度 | 2000 (2K tokens) |
| 并发数 | 32, 64 |
| max-num-seqs | 64 (调整) |

> 此场景测试高并发 + 长上下文混合负载下的系统表现。

#### 3.1.4 场景四：多轮对话测试

| 参数 | 值 |
|------|-----|
| 客户端数 (num_clients) | 10 |
| 对话数 (num_conversations) | 1000 |
| 活跃对话数 (active_conversations) | 10 |
| 对话轮数 (input_num_turns) | 10 |
| 每轮输入 tokens (input_num_tokens) | 200 |
| 每轮输出 tokens (output_num_tokens) | 100 |
| 种子文本 | pg1184.txt (seed=0) |
| 推理框架 | vLLM |

**结果记录表**：

| 平台 | 总耗时 (s) | RPS (req/s) | TTFT Mean (ms) | TPOT Mean (ms) | Latency Mean (ms) |
|------|-----------|-------------|----------------|----------------|-------------------|
| H100 | — | — | — | — | — |
| P800 | — | — | — | — | — |
| BW1000 | — | — | — | — | — |

**详细统计量**：

| 指标 | count | mean | std | 99% | 99.9% | max |
|------|-------|------|-----|-----|-------|-----|
| ttft_ms | — | — | — | — | — | — |
| tpot_ms | — | — | — | — | — | — |
| latency_ms | — | — | — | — | — | — |
| input_num_turns | — | — | — | — | — | — |
| input_num_tokens | — | — | — | — | — | — |
| output_num_tokens | — | — | — | — | — | — |
| output_num_chunks | — | — | — | — | — | — |

### 3.2 功能测试

#### A. 基础功能测试（8 项）

| 编号 | 测试项 | 测试内容 | 验证方式 | 优先级 |
|------|--------|---------|---------|--------|
| A1 | 单轮对话 | 发送 prompt，验证返回结构正确 | 接口调用 | P0 |
| A2 | 多轮对话 | 3-5 轮连续对话，上下文关联正确 | 接口调用 | P0 |
| A3 | System Prompt | 设定系统角色，验证遵循行为 | 接口调用 | P0 |
| A4 | 流式输出 | stream=true，SSE 逐 token 输出 | 接口调用 | P0 |
| A5 | 非流式输出 | stream=false，一次性返回完整结果 | 接口调用 | P0 |
| A6 | Max Tokens 限制 | max_tokens=50/100/500，验证截断 | 接口调用 | P0 |
| A7 | 多语言 | 中/英/日/法 等语言输入输出 | 接口调用 | P1 |
| A8 | 特殊 Token | emoji、HTML 标签等特殊字符处理 | 接口调用 | P1 |

#### B. 高级功能测试（9 项）

| 编号 | 测试项 | 测试内容 | 验证方式 | 优先级 |
|------|--------|---------|---------|--------|
| B1 | Thinking 模式 | thinking mode + 深度推理 | 接口调用 | P0 |
| B2 | Instant 模式 | 非 thinking 快速响应 | 接口调用 | P0 |
| B3 | 模式切换 | thinking / non-thinking 连续切换 | 接口调用 | P1 |
| B4 | 工具调用 - 单次 | function calling 单次调用 | 接口调用 | P0 |
| B5 | 工具调用 - 并行 | 多工具并行调用 | 接口调用 | P1 |
| B6 | 工具调用 - 多轮 | 3+ 轮工具调用链 | 接口调用 | P1 |
| B8 | 结构化输出 | JSON Schema 约束生成 | 接口调用 | P0 |
| B9 | 混合能力 | 推理 + 工具调用组合 | 接口调用 | P0 |

**Thinking 模式说明**：
- `--reasoning-parser minimax_m2`：content 和 think 分离输出（vLLM 标准模式）
- `--reasoning-parser_appended_think minimax_m2`：think 追加到 content 中，think 包含 content + tool_call 信息

#### H. API 兼容性测试（4 项）

| 编号 | 测试项 | 端点 | 验证内容 | 优先级 |
|------|--------|------|---------|--------|
| H1 | OpenAI Chat Completions | /v1/chat/completions | 请求/响应格式兼容 | P0 |
| H2 | OpenAI Completions | /v1/completions | 原始 completion 格式 | P1 |
| H3 | 模型列表 | /v1/models | 返回 object=list，包含模型 ID/created/owned_by | P1 |
| H4 | Usage 统计 | — | prompt_tokens + completion_tokens 正确 | P0 |

### 3.3 精度测试

| 评测集 | 说明 | 方法 |
|--------|------|------|
| MMLU-Pro | 多学科知识评测 | num_fewshot=5, batch_size=1 |

**结果记录表**：

| 学科 | H100 (%) | P800 (%) | BW1000 (%) |
|------|----------|----------|------------|
| Overall | — | — | — |
| Philosophy | — | — | — |
| CS | — | — | — |
| Chemistry | — | — | — |
| Economics | — | — | — |
| Business | — | — | — |
| Math | — | — | — |
| Health | — | — | — |
| Physics | — | — | — |
| History | — | — | — |
| Biology | — | — | — |
| Psychology | — | — | — |
| Engineering | — | — | — |
| Law | — | — | — |

> 各平台精度差异应 < 3%，否则需排查量化或适配问题。

### 3.4 稳定性测试

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 7×24h 持续推理 | 以中等并发 (16) 持续发送请求 | 无 OOM、无 hang、无崩溃 |
| 显存泄漏检测 | 每分钟采样显存占用，持续 24h | 显存无持续增长趋势 |
| 错误率 | 统计请求成功率 | 成功率 ≥ 99.9% |
| 吞吐稳定性 | 对比首小时与末小时吞吐 | 衰减 < 5% |

### 3.5 GPU/DCU/XPU 资源监控

各平台使用对应监控命令：

```bash
# NVIDIA H100
nvidia-smi dmon -s pucvmet -d 1 -f gpu_metrics.csv

# Hygon BW1000 (DCU)
hy-smi / rocm-smi    # rocm-smi 仅用于 DCU

# Kunlun P800 (XPU)
xpu-smi
```

在以下阶段采集 GPU/DCU/XPU 利用率与显存占用：
- vLLM bench serve 压测期间
- benchmark 长稳测试期间
- 多轮对话测试期间

---

## 四、测试指标定义

| 指标 | 英文 | 说明 | 单位 |
|------|------|------|------|
| 首 Token 延迟 | TTFT (Time To First Token) | 从请求发送到首个 token 返回 | ms |
| 单 Token 生成延迟 | TPOT (Time Per Output Token) | 每个输出 token 的平均耗时 | ms/token |
| Token 间延迟 | ITL (Inter-Token Latency) | 连续 token 之间的间隔 | ms |
| 请求吞吐 | Request Throughput | 每秒完成的请求数 | req/s |
| Token 吞吐 | Token Throughput | 每秒生成的 token 总数（输入+输出） | tok/s |
| 输出吞吐 | Output Throughput | 每秒生成的输出 token 数 | tok/s |
| 端到端延迟 | E2E Latency | 完整请求的总响应时间 | ms |

延迟指标统计分位数：**Mean / Median(P50) / P95 / P99**

---

## 五、测试执行计划

### Phase 1：环境准备（第 1-2 天）

| 序号 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| 1 | 确认各平台硬件就绪，安装驱动/Toolkit | — | ☐ |
| 2 | 部署 Docker 环境与 K8S（如适用） | — | ☐ |
| 3 | 安装 vLLM 及依赖（各平台对应版本） | — | ☐ |
| 4 | 下载 MiniMax-M2.5 模型权重（各精度版本） | — | ☐ |
| 5 | 准备测试数据集 (pg1184.txt, random) | — | ☐ |
| 6 | 验证 vLLM 服务可启动、基本推理可用 | — | ☐ |

### Phase 2：适配验证（第 3 天）

| 序号 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| 7 | 各平台模型加载验证（BF16/FP16/INT8/FP8） | — | ☐ |
| 8 | TP=8 通信库验证（NCCL/DTKRCCL/BKCL） | — | ☐ |
| 9 | 基础功能测试 A1-A8 | — | ☐ |
| 10 | 高级功能测试 B1-B9 | — | ☐ |
| 11 | API 兼容性测试 H1-H4 | — | ☐ |
| 12 | reasoning-parser 配置验证 | — | ☐ |

### Phase 3：单机测试（第 4-6 天）

| 序号 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| 13 | 标准性能测试 — 各并发度 (1~128) | — | ☐ |
| 14 | 长上下文测试 — 190K 输入 (并发 1~10) | — | ☐ |
| 15 | 大批量测试 — 90K 输入 (并发 32/64) | — | ☐ |
| 16 | 多轮对话测试 | — | ☐ |
| 17 | 精度测试 — MMLU-Pro | — | ☐ |
| 18 | GPU/DCU/XPU 利用率与显存采集 | — | ☐ |
| 19 | 各场景结果对比与指标达标判定 | — | ☐ |

### Phase 4：分布式测试（第 7-8 天）

| 序号 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| 20 | 多节点环境搭建 (2节点/4节点) | — | ☐ |
| 21 | PP (Pipeline Parallel) 跨节点通信验证 | — | ☐ |
| 22 | 多节点标准性能测试 | — | ☐ |
| 23 | 多节点扩展效率测试 (1→2→4 节点吞吐比) | — | ☐ |
| 24 | 跨节点通信稳定性验证 (buffer size, rank hang 排查) | — | ☐ |

### Phase 5：稳定性测试（第 9-10 天）

| 序号 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| 25 | 7×24h 持续推理测试启动 | — | ☐ |
| 26 | 显存占用周期采样 (每分钟) | — | ☐ |
| 27 | 请求成功率统计 | — | ☐ |
| 28 | 吞吐衰减对比 (首小时 vs 末小时) | — | ☐ |
| 29 | 异常场景测试 (进程 kill 恢复、OOM 恢复) | — | ☐ |

### Phase 6：结果整理与报告（第 11 天）

| 序号 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| 30 | 汇总所有平台测试数据 | — | ☐ |
| 31 | 绘制性能对比图表（吞吐/延迟/并发曲线） | — | ☐ |
| 32 | 各平台 Best Chip 汇总表 | — | ☐ |
| 33 | 指标达标判定与差距分析 | — | ☐ |
| 34 | 撰写最终测试报告 | — | ☐ |

---

## 六、结果汇总模板

### 6.1 请求吞吐对比 (Request Throughput)

| 并发 | 1 | 2 | 4 | 8 | 10 | 16 | 32 | 64 | 80 | 128 |
|------|---|---|---|---|----|----|----|----|----|----|
| Best Chip | — | — | — | — | — | — | — | — | — | — |
| Performance (req/s) | — | — | — | — | — | — | — | — | — | — |

### 6.2 Token 吞吐对比 (Total Token Throughput)

| 并发 | 1 | 2 | 4 | 8 | 10 | 16 | 32 | 64 | 80 | 128 |
|------|---|---|---|---|----|----|----|----|----|----|
| Best Chip | — | — | — | — | — | — | — | — | — | — |
| Performance (tok/s) | — | — | — | — | — | — | — | — | — | — |

### 6.3 TTFT P99 对比

| 并发 | 1 | 2 | 4 | 8 | 10 | 16 | 32 | 64 | 80 | 128 |
|------|---|---|---|---|----|----|----|----|----|----|
| Best Chip | — | — | — | — | — | — | — | — | — | — |
| Latency (ms) | — | — | — | — | — | — | — | — | — | — |

### 6.4 TPOT P99 对比

| 并发 | 1 | 2 | 4 | 8 | 10 | 16 | 32 | 64 | 80 | 128 |
|------|---|---|---|---|----|----|----|----|----|----|
| Best Chip | — | — | — | — | — | — | — | — | — | — |
| Latency (ms) | — | — | — | — | — | — | — | — | — | — |

### 6.5 平台综合对比

| 维度 | H100 | P800 | BW1000 |
|------|------|------|--------|
| 吞吐（相对 H100） | 100% | —% | —% |
| TTFT（相对 H100） | 1.0x | —x | —x |
| TPOT（相对 H100） | 1.0x | —x | —x |
| 精度 (MMLU-Pro) | —% | —% | —% |
| 稳定性 | — | — | — |

---

## 七、已知问题与注意事项

1. **通信库差异**：BW1000 使用 DTKRCCL，P800 使用 BKCL/XCCL，H100 使用 NCCL。TP=8 时需关注 buffer size 与 rank 同步问题，排查潜在 hang 现象
2. **vLLM 版本差异**：各平台 vLLM 版本不同（0.11.0 / 0.14.5 / 0.15.1），需记录版本并在报告中注明版本影响
3. **gpu-memory-utilization 差异**：H100 设为 0.85，其他平台设为 0.95，会影响 KV Cache 容量与最大并发
4. **enable-expert-parallel**：BW1000 和 H100 启用，P800 未启用，可能影响 MoE 相关性能
5. **量化方式差异**：BW1000 (BF16/INT8)、P800 (FP16/INT8)、H100 (FP16/FP8)，对比时需注明精度差异
6. **DCU/XPU 监控**：hy-smi 仅用于 DCU，rocm-smi 仅用于 DCU，xpu-smi 仅用于 XPU，不可混用
7. **GPU 利用率基线**：测试报告中参考值 — BW1000 约 1%~6%，P800 约 2%~16%，H100 约 1%~3%（空闲态）

---

## 八、附录

### 附录 A：vLLM bench 完整命令参考

```bash
# 标准性能测试
vllm bench serve \
    --base-url http://localhost:8000 \
    --model MiniMax-M2.5 \
    --dataset-type random \
    --num-prompts 320 \
    --random-input-len 10240 \
    --random-output-len 256 \
    --concurrency 1

# 长上下文测试
vllm bench serve \
    --base-url http://localhost:8000 \
    --model MiniMax-M2.5 \
    --dataset-type random \
    --num-prompts 100 \
    --random-input-len 194560 \
    --random-output-len 1024 \
    --concurrency 1

# 大批量测试
vllm bench serve \
    --base-url http://localhost:8000 \
    --model MiniMax-M2.5 \
    --dataset-type random \
    --num-prompts 1000 \
    --random-input-len 90000 \
    --random-output-len 2000 \
    --concurrency 32
```

### 附录 B：vLLM 服务启动命令参考

```bash
python -m vllm.entrypoints.openai.api_server \
    --model MiniMax-M2.5 \
    --max-model-len 196608 \
    --max-num-seqs 10 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2 \
    --port 8000
```

### 附录 C：显存监控脚本

```bash
#!/bin/bash
# GPU 显存周期采样 (每 60 秒)
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> gpu_mem_log.csv
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits >> gpu_mem_log.csv
    sleep 60
done
```
