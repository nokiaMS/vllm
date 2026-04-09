# OPT-125M 模型介绍与 vLLM 量化指南

---

## 一、Facebook OPT-125M 模型介绍

### 1.1 概述

**OPT (Open Pre-trained Transformer)** 是 Meta (Facebook) AI 于 2022 年发布的一系列开源大语言模型，OPT-125M 是其中最小的版本。

### 1.2 基本信息

| 属性 | 值 |
|------|-----|
| 参数量 | 1.25 亿 (125M) |
| 架构 | Decoder-only Transformer (类似 GPT) |
| 训练数据 | RoBERTa 的数据集、The Pile、PushShift.io Reddit 等 |
| 上下文长度 | 2048 tokens |
| 词表大小 | 50,272 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| 层数 | 12 |
| 许可证 | OPT-175B License (研究用途) |

### 1.3 主要特点

- **轻量级**：参数量小，适合在消费级 GPU 甚至 CPU 上运行，常用于开发调试和教学
- **GPT-3 对标**：OPT 系列旨在复现 GPT-3 的性能，125M 对应 GPT-3 最小规模
- **开源开放**：Meta 公开了模型权重、训练日志和代码
- **纯文本生成**：属于因果语言模型 (Causal LM)，用于文本续写、问答等生成任务

### 1.4 在 vLLM 中的使用

OPT-125M 因为体积小、加载快，是 vLLM 测试和示例中最常用的模型之一：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0.8, max_tokens=128),
)
```

### 1.5 OPT 系列全家族

```
125M → 350M → 1.3B → 2.7B → 6.7B → 13B → 30B → 66B → 175B
```

OPT-125M 适合快速原型验证和测试流程，但因参数量太小，生成质量有限，不适合生产场景。

---

## 二、通过 vLLM 对 OPT-125M 进行量化

vLLM 支持多种量化方法，下面按易用程度依次介绍。

### 2.1 方案一：FP8 动态量化（最简单，无需校准数据）

>fp8(float point 8-bit) 量化是 vLLM 最新支持的高效量化方案，具有接近 FP16 的精度和更高的压缩比。
FP8（8位浮点）将模型权重从默认的 FP16 压缩到 8 位，降低显存占用、提升吞吐，精度损失很小。

>此代码中配置了quantization="fp8"，vLLM 会自动在加载模型时进行在线量化，无需预先处理模型或提供校准数据。

>直接在推理时指定 `quantization="fp8"`，vLLM 自动完成在线量化：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", quantization="fp8")
output = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0.8, max_tokens=128),
)
print(output[0].outputs[0].text)
```

> **要求**：需要 Ada/Hopper 架构 GPU（如 RTX 4090、H100）。

### 2.2 方案二：BitsAndBytes 4-bit 量化（消费级 GPU 友好）

无需校准数据，实时量化：

```python
from vllm import LLM, SamplingParams

# 在线量化（加载时自动转为 4-bit）
llm = LLM(model="facebook/opt-125m", quantization="bitsandbytes")
output = llm.generate(
    ["Hello, my name is"],
    SamplingParams(max_tokens=128),
)
print(output[0].outputs[0].text)
```

也可以直接使用预量化的模型：

```python
llm = LLM(model="poedator/opt-125m-bnb-4bit")
```

### 2.3 方案三：使用 llm-compressor 离线量化（推荐用于生产）

这是 vLLM 官方推荐的离线量化方式，支持 FP8/INT8/INT4。

#### 安装 llm-compressor

```bash
pip install llmcompressor
```

#### FP8 静态量化（无需校准数据）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(model=model, recipe=recipe, output_dir="opt-125m-fp8")
```

#### INT8 W8A8 量化（需要校准数据，精度更高）

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model="facebook/opt-125m",
    dataset="open_platypus",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    output_dir="opt-125m-int8",
)
```

#### INT4 W4A16 量化（压缩比最高）

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"],
)

oneshot(
    model="facebook/opt-125m",
    dataset="open_platypus",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    output_dir="opt-125m-int4",
)
```

#### 加载离线量化模型

```python
from vllm import LLM

llm = LLM(model="./opt-125m-fp8")  # 自动检测量化格式
```

### 2.4 方案四：TorchAO 量化

```python
from torchao.quantization import Int8WeightOnlyConfig
from torchao.utils import config_to_dict
import json
from vllm import LLM

config = Int8WeightOnlyConfig()
json_str = json.dumps(config_to_dict(config))

llm = LLM(
    model="facebook/opt-125m",
    hf_overrides={"quantization_config_dict_json": json_str},
)
```

---

## 三、量化效果评估方法

### 3.1 使用 lm-eval-harness 对比精度

```bash
pip install lm_eval

# 评估原始模型
lm_eval --model vllm \
  --model_args pretrained=facebook/opt-125m,add_bos_token=True \
  --tasks gsm8k,hellaswag,arc_easy \
  --num_fewshot 5 --batch_size auto

# 评估量化模型
lm_eval --model vllm \
  --model_args pretrained=./opt-125m-int8,add_bos_token=True \
  --tasks gsm8k,hellaswag,arc_easy \
  --num_fewshot 5 --batch_size auto
```

对比两者的准确率差距即可衡量量化带来的精度损失。

### 3.2 使用 vLLM 自带 benchmark 测吞吐量

```bash
# 原始模型
python benchmarks/benchmark_throughput.py \
  --model facebook/opt-125m --num-prompts 100

# 量化模型
python benchmarks/benchmark_throughput.py \
  --model ./opt-125m-fp8 --num-prompts 100
```

### 3.3 简单对比脚本

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Once upon a time",
    "In a galaxy far away",
]
params = SamplingParams(temperature=0.0, max_tokens=50)

# 原始模型
llm_base = LLM("facebook/opt-125m")
base_outputs = llm_base.generate(prompts, params)

# 量化模型
llm_quant = LLM("facebook/opt-125m", quantization="fp8")
quant_outputs = llm_quant.generate(prompts, params)

for p, b, q in zip(prompts, base_outputs, quant_outputs):
    print(f"Prompt: {p}")
    print(f"  Base:  {b.outputs[0].text}")
    print(f"  Quant: {q.outputs[0].text}")
    print()
```

---

## 四、各方案对比总结

| 方案 | 压缩比 | 需要校准 | GPU 要求 | 适用场景 |
|------|--------|---------|---------|---------|
| FP8 动态 | ~2x | 否 | Ada/Hopper | 快速部署 |
| BitsAndBytes 4-bit | ~4x | 否 | 大多数 GPU | 显存受限 |
| INT8 W8A8 (llm-compressor) | ~2x | 是 | Ada/Hopper/CPU | 生产环境 |
| INT4 W4A16 (llm-compressor) | ~4x | 是 | 大多数 GPU | 极致压缩 |
| TorchAO | ~2x | 否 | 大多数 GPU | 灵活集成 |

> **注意**：OPT-125M 本身很小（~250MB），量化的实际收益有限。量化更适用于 7B+ 的大模型。这里用 125M 主要是方便学习和验证流程。

---

## 五、vLLM 支持的全部量化方法一览

| 量化方法 | 说明 |
|---------|------|
| AWQ | 激活感知权重量化（已弃用，推荐用 llm-compressor） |
| BitsAndBytes | INT4/INT8，无需校准数据 |
| Compressed Tensors | 通过 llm-compressor 生成，支持多种量化方案 |
| GGUF | 实验性支持，单文件模型格式 |
| GPTQModel | INT4/INT8，支持 Marlin 加速 |
| Intel AutoRound | INT2-8, MXFP 格式 |
| FP8 W8A8 | 浮点 8 位量化，需 Ada/Hopper GPU |
| INT8 W8A8 | 8 位整数量化（权重 + 激活值） |
| INT4 W4A16 | 4 位整数权重量化 |
| NVIDIA ModelOpt | 支持 FP8, NVFP4, MXFP8 |
| AMD Quark | 支持 FP8, MXFP4/6, 混合精度 |
| TorchAO | Int8WeightOnly 等多种方案 |
| Quantized KV Cache | FP8 KV 缓存量化 |
