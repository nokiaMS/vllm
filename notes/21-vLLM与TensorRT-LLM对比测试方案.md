# vLLM 与 TensorRT-LLM 对比测试方案

---

## 一、测试目标

对 vLLM 和 TensorRT-LLM 两大推理框架进行全面的性能与功能对比，为生产环境技术选型提供数据支撑。

核心评估维度：
1. **推理性能**：吞吐量、延迟、首 token 延迟 (TTFT)
2. **资源效率**：显存占用、GPU 利用率
3. **功能完备性**：量化支持、多模型、流式输出、API 兼容性
4. **部署运维**：安装复杂度、模型转换成本、运维友好度

---

## 二、测试环境

### 2.1 硬件配置

| 配置项 | 规格 |
|--------|------|
| GPU | NVIDIA A100 80GB（或 A800/H100，需统一） |
| GPU 数量 | 单卡测试 + 多卡测试 (2/4/8 卡) |
| CPU | x86_64, 64 核+ |
| 内存 | 256GB+ |
| 网络 | 多卡场景使用 NVLink / NVSwitch |

### 2.2 软件环境

| 组件 | 版本要求 |
|------|----------|
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.4+ |
| Driver | 550+ |
| Python | 3.10 / 3.12 |
| vLLM | 最新稳定版 (pip install vllm) |
| TensorRT-LLM | 最新稳定版 (对应 CUDA 版本) |
| Docker | 推荐容器化部署，避免环境差异 |

### 2.3 Docker 镜像

```bash
# vLLM
docker pull vllm/vllm-openai:latest

# TensorRT-LLM
docker pull nvcr.io/nvidia/tritonserver:xx.xx-trtllm-python-py3
# 或使用官方 tensorrt_llm 镜像
```

---

## 三、测试模型

选择不同规模的主流模型覆盖典型场景：

| 模型 | 参数量 | 用途 | 备注 |
|------|--------|------|------|
| Llama-3.1-8B-Instruct | 8B | 小模型基准 | 单卡可跑 |
| Qwen2.5-72B-Instruct | 72B | 大模型基准 | 需多卡 TP |
| Llama-3.1-70B-Instruct | 70B | 大模型对比 | 业界常用 benchmark 模型 |
| Mixtral-8x7B-Instruct | 47B (MoE) | MoE 架构 | 测试 MoE 支持差异 |

### 3.1 量化版本

每个模型额外测试以下量化配置：

| 量化方式 | vLLM 支持 | TensorRT-LLM 支持 |
|----------|-----------|-------------------|
| FP16 | Yes | Yes |
| BF16 | Yes | Yes |
| INT8 (W8A8) | Yes | Yes |
| INT4 (W4A16, GPTQ/AWQ) | Yes | Yes |
| FP8 (W8A8) | Yes (H100/A100) | Yes (H100 原生) |

---

## 四、测试场景与用例

### 4.1 场景一：在线服务 (Online Serving)

模拟真实用户请求，使用 OpenAI 兼容 API 接口。

#### 4.1.1 服务启动

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --port 8000

# TensorRT-LLM (需先转换引擎)
# Step 1: 转换 checkpoint
python convert_checkpoint.py \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./trt_ckpt/llama-8b \
    --dtype float16

# Step 2: 构建 TRT engine
trtllm-build \
    --checkpoint_dir ./trt_ckpt/llama-8b \
    --output_dir ./trt_engine/llama-8b \
    --max_batch_size 256 \
    --max_input_len 2048 \
    --max_seq_len 4096

# Step 3: 启动服务
python ../tensorrtllm_backend/launch_triton_server.py \
    --model_repo ./triton_model_repo
```

#### 4.1.2 负载模式

| 测试项 | 并发数 | 请求速率 (QPS) | 说明 |
|--------|--------|----------------|------|
| 低负载 | 1-4 | 1-5 | 单用户体验 |
| 中负载 | 16-32 | 10-30 | 正常业务量 |
| 高负载 | 64-128 | 50-100 | 压力测试 |
| 极限负载 | 256+ | 尽可能高 | 找到吞吐上限 |

### 4.2 场景二：离线批量推理 (Offline Batch)

```bash
# vLLM (Python API)
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(prompts, SamplingParams(max_tokens=512))
```

### 4.3 场景三：长上下文

| 输入长度 | 输出长度 | 说明 |
|----------|----------|------|
| 1K | 512 | 短对话 |
| 4K | 512 | 中等上下文 |
| 8K | 1024 | 长文档问答 |
| 32K | 1024 | 长上下文能力 |
| 128K | 2048 | 超长上下文 (如果模型支持) |

### 4.4 场景四：多轮对话

- 模拟 3-5 轮对话
- 测试 KV Cache 复用效率
- 测量每轮的 TTFT 变化

---

## 五、测试指标

### 5.1 性能指标

| 指标 | 英文缩写 | 说明 | 单位 |
|------|----------|------|------|
| 首 Token 延迟 | TTFT (Time To First Token) | 从请求到第一个 token 返回 | ms |
| 单 Token 生成延迟 | TPOT (Time Per Output Token) | 每生成一个 token 的耗时 | ms |
| 端到端延迟 | E2E Latency | 完整请求的响应时间 | ms |
| 吞吐量 | Throughput | 每秒生成的 token 数 | tokens/s |
| 请求吞吐量 | Request Throughput | 每秒完成的请求数 | req/s |
| ITL | Inter-Token Latency | token 间延迟的 P50/P95/P99 | ms |

### 5.2 资源指标

| 指标 | 说明 | 采集方式 |
|------|------|----------|
| 显存占用 (峰值) | 模型加载 + KV Cache + 临时缓冲 | nvidia-smi |
| 显存占用 (稳态) | 持续服务时的显存 | nvidia-smi (周期采样) |
| GPU 利用率 | SM 占用率 | nvidia-smi / DCGM |
| CPU 使用率 | 主机 CPU 占用 | top / htop |
| 内存使用 | 主机内存占用 | free -h |

### 5.3 功能指标

| 功能项 | 对比维度 |
|--------|----------|
| 模型转换时间 | TRT-LLM 需要 build engine，vLLM 直接加载 |
| 冷启动时间 | 从启动到可服务的时间 |
| 流式输出 | SSE 流式支持质量 |
| 结构化输出 | JSON mode / guided decoding |
| 多 LoRA | 同时加载多个 LoRA adapter |
| 前缀缓存 | Prefix Caching / RadixAttention |
| 动态批处理 | Continuous Batching 效果 |

---

## 六、测试工具

### 6.1 性能压测工具

推荐使用以下工具（任选其一或组合）：

#### 方案 A：vLLM 自带 benchmark（推荐用于对比）

```bash
# 在线服务压测
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --request-rate 10

# 离线吞吐压测
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 1000
```

#### 方案 B：GenAI-Perf (NVIDIA 官方)

```bash
genai-perf profile \
    -m meta-llama/Llama-3.1-8B-Instruct \
    --service-kind openai \
    --endpoint-type chat \
    --streaming \
    --concurrency 32 \
    --measurement-interval 60000 \
    --url localhost:8000
```

#### 方案 C：自定义脚本

```python
import asyncio
import aiohttp
import time

async def benchmark(url, prompts, concurrency):
    """自定义压测脚本，记录 TTFT/TPOT/E2E"""
    ...
```

### 6.2 监控工具

```bash
# GPU 监控 (每秒采样)
nvidia-smi dmon -s pucvmet -d 1 -f gpu_metrics.csv

# 或使用 DCGM
dcgmi dmon -e 155,156,203,204 -d 1000
```

---

## 七、测试数据集

| 数据集 | 说明 | 获取方式 |
|--------|------|----------|
| ShareGPT | 真实多轮对话，输入输出长度分布自然 | HuggingFace |
| MMLU | 标准评测，固定格式短输入 | HuggingFace |
| 合成数据 (固定长度) | 控制变量：固定 input_len/output_len | 脚本生成 |
| LongBench | 长上下文评测 | HuggingFace |

### 合成数据生成

```bash
# 使用 vLLM benchmark 工具的合成数据模式
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --num-prompts 2000 \
    --request-rate 20
```

---

## 八、测试执行计划

### Phase 1：环境搭建与验证（第 1-2 天）

- [ ] 准备硬件环境，安装驱动与 CUDA
- [ ] 部署 vLLM Docker 容器，验证服务可用
- [ ] 部署 TensorRT-LLM Docker 容器，验证服务可用
- [ ] 完成所有测试模型的下载
- [ ] TensorRT-LLM 引擎构建 (记录构建时间)
- [ ] 验证两个框架对同一 prompt 的输出正确性

### Phase 2：单卡性能测试（第 3-4 天）

- [ ] Llama-3.1-8B FP16 — 不同并发/QPS 的 TTFT/TPOT/吞吐
- [ ] Llama-3.1-8B INT4 (AWQ) — 同上
- [ ] Llama-3.1-8B FP8 — 同上 (如果 GPU 支持)
- [ ] 不同输入输出长度组合测试
- [ ] 离线批量推理吞吐对比
- [ ] 显存占用与 GPU 利用率采集

### Phase 3：多卡性能测试（第 5-6 天）

- [ ] Llama-3.1-70B TP=4 FP16 — 在线服务性能
- [ ] Qwen2.5-72B TP=4 FP16 — 在线服务性能
- [ ] Mixtral-8x7B TP=2 — MoE 模型性能
- [ ] 多卡扩展效率 (1→2→4→8 卡的吞吐变化)

### Phase 4：功能与易用性评估（第 7 天）

- [ ] 冷启动时间对比 (含 TRT-LLM engine build 时间)
- [ ] 模型更新流程对比 (vLLM 热加载 vs TRT-LLM 重新 build)
- [ ] 流式输出延迟抖动对比
- [ ] 结构化输出 / JSON mode 支持对比
- [ ] 多 LoRA 支持对比
- [ ] 长上下文场景测试
- [ ] API 兼容性 (OpenAI API 格式) 对比

### Phase 5：结果整理与报告（第 8 天）

- [ ] 汇总所有测试数据
- [ ] 绘制性能对比图表
- [ ] 撰写测试报告与选型建议

---

## 九、结果记录模板

### 9.1 性能测试记录表

| 框架 | 模型 | 量化 | TP | 并发 | QPS | TTFT P50 (ms) | TTFT P99 (ms) | TPOT P50 (ms) | 吞吐 (tok/s) | 显存 (GB) |
|------|------|------|----|------|-----|---------------|---------------|---------------|-------------|-----------|
| vLLM | Llama-8B | FP16 | 1 | 32 | 20 | — | — | — | — | — |
| TRT-LLM | Llama-8B | FP16 | 1 | 32 | 20 | — | — | — | — | — |

### 9.2 功能对比记录表

| 功能 | vLLM | TensorRT-LLM | 备注 |
|------|------|-------------|------|
| 安装复杂度 | pip install | 多步构建 | |
| 模型转换时间 | 无需转换 | 需 build engine | |
| 冷启动时间 | — | — | |
| OpenAI API 兼容 | 原生支持 | 需 Triton 后端 | |
| 流式输出 | SSE | SSE | |
| 结构化输出 | 支持 | 有限支持 | |
| 多 LoRA | 支持 | 有限支持 | |
| 前缀缓存 | 支持 (Automatic Prefix Caching) | 支持 | |
| Speculative Decoding | 支持 | 支持 | |

---

## 十、注意事项

1. **公平性**：两个框架使用相同的模型权重、相同的采样参数 (temperature, top_p 等)、相同的 max_tokens
2. **预热**：正式测试前发送 100+ 请求预热，排除冷启动影响
3. **多次采样**：每个测试点至少运行 3 次，取中位数
4. **隔离性**：测试期间不运行其他 GPU 任务
5. **TRT-LLM 引擎配置**：build engine 时的 max_batch_size / max_input_len 等参数需与测试场景匹配，否则影响公平性
6. **版本记录**：记录所有软件的精确版本号，便于复现

---

## 十一、预期产出

1. **性能对比报告**：包含各场景的详细数据与图表
2. **选型建议**：基于测试数据给出不同场景下的推荐方案
3. **部署指南**：两个框架的最佳实践配置参数
4. **成本分析**：结合性能数据估算单位推理成本 ($/1M tokens)
