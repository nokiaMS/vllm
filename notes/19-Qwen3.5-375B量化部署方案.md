# Qwen3.5-375B 量化方案（适配 2xA100-80GB 或更少资源）

## Context

Qwen3.5-375B 是超大规模 MoE 模型，375B 参数，256 个专家，每次推理激活 8 个（A22B）。BF16 需要约 **750GB** 显存。用户硬件为 **2xA100-80GB（共 160GB）**，必须使用激进的 4-bit 量化 + 多卡并行。

---

## 模型架构（来自 `vllm/transformers_utils/configs/qwen3_5_moe.py`）

| 参数 | 值 |
|------|-----|
| num_experts | 256 |
| num_experts_per_tok | 8 |
| num_hidden_layers | 40 |
| hidden_size | 2048 |
| head_dim | 256 |

---

## 推荐方案：GPTQ-Int4 + TP2

**显存估算：约 95GB 模型权重 + KV Cache -> 2xA100-80GB 刚好可行**

### 步骤 1：获取预量化模型

在 HuggingFace 上搜索 Qwen3.5-375B 的 GPTQ-Int4 或 AWQ 预量化版本：

```bash
# 常见命名模式（以实际发布名称为准）
# Qwen/Qwen3.5-375B-A22B-GPTQ-Int4
# Qwen/Qwen3.5-375B-A22B-AWQ
```

如果官方没有预量化版本，可以使用 AutoGPTQ 或 AutoAWQ 自行量化（需要一台大内存机器）：

```bash
# 使用 AutoGPTQ 量化（需要在有足够 CPU 内存的机器上运行）
pip install auto-gptq
python -c "
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-375B-A22B')
model = AutoGPTQForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-375B-A22B',
    quantize_config,
    device_map='cpu',  # 在 CPU 上量化
)
# 准备校准数据并量化
model.quantize(examples)
model.save_quantized('/path/to/Qwen3.5-375B-GPTQ-Int4')
"
```

### 步骤 2：使用 vLLM 启动推理服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-375B-A22B-GPTQ-Int4 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e5m2 \
    --enforce-eager
```

**参数说明**：
- `--tensor-parallel-size 2`：2 张 A100 并行
- `--max-model-len 8192`：限制最大序列长度以节省 KV Cache 显存（根据需要调整）
- `--gpu-memory-utilization 0.92`：使用 92% 的显存
- `--kv-cache-dtype fp8_e5m2`：KV Cache 用 FP8 存储，额外节省 ~50% Cache 显存
- `--enforce-eager`：如果显存紧张，禁用 CUDA Graph 以节省额外显存

---

## 备选方案：BitsAndBytes 4-bit（无需预量化模型）

如果找不到预量化模型，可直接从 BF16 动态量化：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-375B-A22B \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager
```

**注意**：BitsAndBytes 推理速度比 GPTQ 慢，但部署最简单。

---

## 备选方案：GGUF Q4_K_M

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-375B-A22B-Q4_K_M.gguf \
    --tokenizer Qwen/Qwen3.5-375B-A22B \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92
```

---

## 显存不足时的额外优化

如果 2xA100-80GB 仍然不够，可以叠加以下措施：

1. **缩短 max-model-len**：从 32768 -> 8192 -> 4096，大幅减少 KV Cache 占用
2. **KV Cache FP8**：`--kv-cache-dtype fp8_e5m2`
3. **禁用 CUDA Graph**：`--enforce-eager` 省出 CUDA Graph 预留的显存
4. **限制并发**：`--max-num-seqs 4` 减少同时处理的请求数

---

## 方案对比

| 方案 | 模型显存 | 精度损失 | 推理速度 | 部署难度 |
|------|---------|---------|---------|---------|
| **GPTQ-Int4** | ~95GB | 较小 | 快 | 需预量化模型 |
| **AWQ-Int4** | ~95GB | 较小 | 快 | 需预量化模型 |
| **BitsAndBytes 4bit** | ~95GB | 中等 | 较慢 | 简单（直接加载） |
| **GGUF Q4_K_M** | ~100GB | 较小 | 中等 | 需 GGUF 格式模型 |

---

## 关键代码文件

- 量化注册: `vllm/model_executor/layers/quantization/__init__.py`
- GPTQ: `vllm/model_executor/layers/quantization/gptq.py`
- AWQ: `vllm/model_executor/layers/quantization/awq.py`
- BitsAndBytes: `vllm/model_executor/layers/quantization/bitsandbytes.py`
- MoE 专用量化: `vllm/model_executor/layers/quantization/moe_wna16.py`
- Qwen3.5 模型: `vllm/model_executor/models/qwen3_5.py`
- 并行配置: `vllm/config/parallel.py`
