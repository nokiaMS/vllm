# 7 天学习大语言模型 (LLM) 指南

> 由浅入深，循序渐进，从基础原理到前沿技术。

---

## 目录

- [Day 1: 基础概念与语言模型入门](#day-1-基础概念与语言模型入门)
- [Day 2: 词嵌入与注意力机制](#day-2-词嵌入与注意力机制)
- [Day 3: Transformer 架构详解](#day-3-transformer-架构详解)
- [Day 4: 预训练与大语言模型](#day-4-预训练与大语言模型)
- [Day 5: 微调与对齐技术](#day-5-微调与对齐技术)
- [Day 6: 推理优化与部署](#day-6-推理优化与部署)
- [Day 7: 前沿技术与实践](#day-7-前沿技术与实践)

---

## Day 1: 基础概念与语言模型入门

### 1.1 什么是语言模型

语言模型 (Language Model) 是一种对自然语言文本的概率分布进行建模的模型。其核心任务是：**给定一段文本序列，预测下一个词（或 token）出现的概率。**

数学定义：

```
P(w₁, w₂, ..., wₙ) = ∏ P(wᵢ | w₁, w₂, ..., wᵢ₋₁)
```

即一段文本的联合概率等于每个词在其前文条件下的条件概率之积。

### 1.2 语言模型的发展脉络

```
统计语言模型 → 神经网络语言模型 → RNN/LSTM → Transformer → 大语言模型 (LLM)
  (N-gram)      (Word2Vec等)      (Seq2Seq)   (Attention)   (GPT/LLaMA等)
```

### 1.3 N-gram 模型

最简单的语言模型。假设当前词只依赖前 N-1 个词。

**Bigram (N=2) 示例：**

```
训练语料：
  "我 喜欢 学习"
  "我 喜欢 编程"
  "他 喜欢 学习"

统计 Bigram 概率：
  P(喜欢 | 我)   = 2/2 = 1.0    （"我"后面出现"喜欢"2次，"我"共出现2次）
  P(喜欢 | 他)   = 1/1 = 1.0
  P(学习 | 喜欢) = 2/3 = 0.67
  P(编程 | 喜欢) = 1/3 = 0.33

预测："我 喜欢 ?" → 最可能是 "学习"(0.67) > "编程"(0.33)
```

**N-gram 的局限性：**
- 无法捕捉长距离依赖（"这本书……非常好看" 中间隔了很多词）
- 数据稀疏问题（N 越大，组合越多，很多组合未出现过）
- 没有语义理解能力

### 1.4 神经网络语言模型 (NNLM)

Bengio 等人 (2003) 提出的开创性工作：用神经网络替代 N-gram 的查表方式。

```
输入层: [wₜ₋₃, wₜ₋₂, wₜ₋₁]  ← 前3个词的 one-hot 编码
    ↓
嵌入层: [e₃, e₂, e₁]          ← 查找嵌入矩阵，得到稠密向量
    ↓
拼接: [e₃ ; e₂ ; e₁]          ← 将三个向量拼接
    ↓
隐藏层: h = tanh(W·x + b)     ← 非线性变换
    ↓
输出层: y = softmax(V·h + c)   ← 输出词汇表上的概率分布
```

**关键突破：** 词被映射到连续的向量空间，语义相近的词距离也近。

### 1.5 关键概念速查

| 概念 | 解释 |
|------|------|
| Token | 文本被切分后的最小单位，可以是词、子词或字符 |
| Vocabulary | 模型能识别的所有 token 的集合 |
| Embedding | 将离散 token 映射到连续向量空间的表示 |
| Softmax | 将向量转化为概率分布的函数 |
| Perplexity (困惑度) | 衡量语言模型好坏的指标，越低越好 |

### 1.6 练习

1. 手动用 Bigram 计算句子 "我 喜欢 学习 编程" 的概率
2. 思考：为什么 one-hot 编码不适合表示词？（提示：维度灾难、无法表达语义相似性）

---

## Day 2: 词嵌入与注意力机制

### 2.1 Word2Vec

Word2Vec (Mikolov et al., 2013) 是学习词向量的里程碑模型，包含两种架构：

**CBOW (连续词袋模型)：** 用上下文预测中心词

```
上下文: [我, 学习]  →  预测中心词: 喜欢

           我(embedding)
              ↘
    平均池化 → hidden → softmax → P(喜欢)
              ↗
         学习(embedding)
```

**Skip-gram：** 用中心词预测上下文（实际更常用）

```
中心词: 喜欢  →  预测上下文: 我, 学习

    喜欢(embedding) → softmax → P(我), P(学习)
```

**训练技巧 - 负采样 (Negative Sampling)：**

Softmax 对整个词汇表计算代价太大。负采样将问题转化为二分类：

```
正样本: (喜欢, 学习) → 标签 1  （确实是上下文关系）
负样本: (喜欢, 桌子) → 标签 0  （随机采样的非上下文词）
负样本: (喜欢, 飞机) → 标签 0

损失函数:
L = -log σ(v'学习 · v喜欢) - Σ log σ(-v'neg · v喜欢)

其中 σ 是 sigmoid 函数，v 和 v' 分别是输入和输出嵌入向量
```

**词向量的神奇性质：**

```
v(国王) - v(男人) + v(女人) ≈ v(女王)
v(巴黎) - v(法国) + v(中国) ≈ v(北京)
```

### 2.2 从 RNN 到注意力机制

**RNN 的问题：**

```
h₁ → h₂ → h₃ → ... → hₙ
 ↑    ↑    ↑           ↑
 w₁   w₂   w₃          wₙ

信息从 w₁ 传到 wₙ 要经过 n-1 步，长距离依赖信息会衰减（梯度消失）
```

**注意力机制的核心思想：** 让模型在处理每个位置时，直接"关注"输入序列中所有位置的信息，而不是只依赖压缩后的隐藏状态。

### 2.3 注意力机制 (Attention) 详解

**基本公式：**

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √dₖ) · V
```

其中：
- **Q (Query):** 查询向量 — "我想找什么？"
- **K (Key):** 键向量 — "我这里有什么？"
- **V (Value):** 值向量 — "我能提供什么信息？"
- **√dₖ:** 缩放因子，防止点积值过大导致 softmax 梯度消失

**直觉理解：**

```
想象你在图书馆找书（Query），
书架上每本书的标题是 Key，书的内容是 Value。
你用你的需求和每本书的标题比较（Q·Kᵀ），
得到匹配度分数，再 softmax 归一化为权重，
最后按权重加权获取书的内容（·V）。
```

**具体计算示例：**

```
假设句子 "猫 坐在 垫子 上"，维度 dₖ = 4

对于 "坐在" 这个词，计算它对其他词的注意力：

Q_坐在 = [1.0, 0.5, 0.3, 0.8]

K_猫    = [0.9, 0.6, 0.2, 0.7]
K_坐在  = [1.0, 0.5, 0.3, 0.8]
K_垫子  = [0.3, 0.8, 0.9, 0.2]
K_上    = [0.2, 0.4, 0.7, 0.3]

Step 1: 计算点积 Q·Kᵀ
  score(猫)   = 1.0×0.9 + 0.5×0.6 + 0.3×0.2 + 0.8×0.7 = 1.82
  score(坐在) = 1.0×1.0 + 0.5×0.5 + 0.3×0.3 + 0.8×0.8 = 1.98
  score(垫子) = 1.0×0.3 + 0.5×0.8 + 0.3×0.9 + 0.8×0.2 = 1.13
  score(上)   = 1.0×0.2 + 0.5×0.4 + 0.3×0.7 + 0.8×0.3 = 0.85

Step 2: 缩放 (÷√4 = ÷2)
  [0.91, 0.99, 0.565, 0.425]

Step 3: Softmax
  [0.28, 0.30, 0.20, 0.17]  ← "坐在"最关注自身和"猫"

Step 4: 加权求和 Value 向量得到输出
  output = 0.28×V_猫 + 0.30×V_坐在 + 0.20×V_垫子 + 0.17×V_上
```

### 2.4 多头注意力 (Multi-Head Attention)

单一注意力可能只捕捉一种关系模式。多头注意力让模型同时关注不同类型的关系：

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) · Wᴼ

其中每个头:
  headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᴷ, V·Wᵢⱽ)
```

```
例如 8 头注意力，输入维度 512：
  每个头的维度: 512 / 8 = 64
  head₁ 可能关注: 语法关系（主谓宾）
  head₂ 可能关注: 指代关系（代词指向）
  head₃ 可能关注: 相邻词关系
  ...
  最后拼接 8×64 = 512，再线性变换
```

### 2.5 练习

1. 给定 Q=[1,0], K=[[1,0],[0,1],[1,1]], V=[[1,0],[0,1],[0.5,0.5]]，手算 Attention 输出
2. 思考：为什么需要除以 √dₖ？如果不除会怎样？

---

## Day 3: Transformer 架构详解

### 3.1 Transformer 总体架构

Transformer (Vaswani et al., 2017) — "Attention Is All You Need"

```
┌─────────────────────────────────────────────┐
│                Transformer                   │
│                                             │
│   Encoder (编码器)        Decoder (解码器)    │
│   ┌──────────────┐     ┌──────────────────┐ │
│   │ Input Embed  │     │ Output Embed     │ │
│   │ + Pos Enc    │     │ + Pos Enc        │ │
│   ├──────────────┤     ├──────────────────┤ │
│   │              │     │                  │ │
│   │ Multi-Head   │     │ Masked Multi-    │ │
│   │ Self-Attn    │     │ Head Self-Attn   │ │
│   │    ↓         │     │      ↓           │ │
│   │ Add & Norm   │     │ Add & Norm       │ │
│   │    ↓         │ ──→ │      ↓           │ │
│   │ Feed Forward │     │ Cross-Attention  │ │
│   │    ↓         │     │ (Q from decoder, │ │
│   │ Add & Norm   │     │  K,V from encoder)│ │
│   │              │     │      ↓           │ │
│   │   ×N 层      │     │ Add & Norm       │ │
│   │              │     │      ↓           │ │
│   └──────────────┘     │ Feed Forward     │ │
│                        │      ↓           │ │
│                        │ Add & Norm       │ │
│                        │                  │ │
│                        │   ×N 层          │ │
│                        └──────────────────┘ │
│                              ↓              │
│                        Linear + Softmax     │
│                        → 输出概率分布        │
└─────────────────────────────────────────────┘
```

### 3.2 位置编码 (Positional Encoding)

注意力机制本身不区分位置顺序。位置编码注入位置信息。

**正弦余弦位置编码（原始 Transformer）：**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中 pos 是位置，i 是维度索引，d_model 是嵌入维度

示例 (d_model=4):
位置0: [sin(0/1), cos(0/1), sin(0/100), cos(0/100)]
      = [0.000, 1.000, 0.000, 1.000]

位置1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100)]
      = [0.841, 0.540, 0.010, 1.000]

位置2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100)]
      = [0.909, -0.416, 0.020, 1.000]
```

**RoPE (旋转位置编码，现代 LLM 主流)：**

```
将位置信息编码为向量的旋转角度：
  对嵌入向量的每两个相邻维度 (x₂ᵢ, x₂ᵢ₊₁) 施加旋转：

  ┌ x'₂ᵢ   ┐   ┌ cos(mθᵢ)  -sin(mθᵢ) ┐ ┌ x₂ᵢ   ┐
  │         │ = │                        │ │        │
  └ x'₂ᵢ₊₁ ┘   └ sin(mθᵢ)   cos(mθᵢ) ┘ └ x₂ᵢ₊₁ ┘

  其中 m 是位置，θᵢ = 10000^(-2i/d)

优势:
  - 相对位置信息自然编码在内积中
  - 可以外推到训练时未见过的更长序列
```

### 3.3 残差连接与层归一化

```
残差连接 (Residual Connection):
  output = LayerNorm(x + SubLayer(x))

作用: 缓解深层网络的梯度消失问题，让梯度可以"直通"

层归一化 (Layer Normalization):
  对单个样本的所有特征维度进行归一化

  LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

  其中 μ = mean(x), σ² = var(x), γ 和 β 是可学习参数

Pre-Norm vs Post-Norm:
  Post-Norm (原始): x + SubLayer(LayerNorm(x))  ← 有时训练不稳定
  Pre-Norm (现代):  x + SubLayer(LayerNorm(x))  ← 训练更稳定，LLM 常用

  注: 现代 LLM 多使用 RMSNorm（只做缩放，不减均值）:
  RMSNorm(x) = γ · x / √(mean(x²) + ε)
```

### 3.4 前馈网络 (FFN)

```
标准 FFN:
  FFN(x) = W₂ · ReLU(W₁·x + b₁) + b₂

  输入维度: d_model (如 4096)
  中间维度: d_ff (通常 4×d_model = 16384)
  输出维度: d_model (如 4096)

  参数量: 2 × d_model × d_ff ≈ 模型总参数的 2/3

现代 LLM 常用 SwiGLU 激活:
  SwiGLU(x) = (W₁·x ⊙ σ(W_gate·x)) · W₂

  其中 ⊙ 是逐元素乘法，σ 是 sigmoid
  相比 ReLU，SwiGLU 通常能获得更好的性能
```

### 3.5 完整前向传播流程

```
输入: "Transformer 很 强大"

Step 1: Tokenize
  tokens = [7291, 203, 1542]

Step 2: Embedding + Positional Encoding
  x = Embedding(tokens) + PE(positions)
  x.shape = [3, d_model]

Step 3: 经过 N 个 Transformer Block
  for each block:
    # Self-Attention
    Q = x @ Wq;  K = x @ Wk;  V = x @ Wv
    attn_output = MultiHeadAttention(Q, K, V)
    x = LayerNorm(x + attn_output)

    # Feed Forward
    ff_output = FFN(x)
    x = LayerNorm(x + ff_output)

Step 4: 输出层
  logits = x @ W_output  → shape [3, vocab_size]
  probs = softmax(logits)
  next_token = argmax(probs[-1])  ← 取最后一个位置的预测
```

### 3.6 三种 Transformer 变体

| 类型 | 代表模型 | 注意力模式 | 典型任务 |
|------|----------|-----------|---------|
| Encoder-only | BERT | 双向注意力（看到所有位置） | 文本分类、NER |
| Decoder-only | GPT、LLaMA | 因果注意力（只看过去） | 文本生成 |
| Encoder-Decoder | T5、BART | 编码器双向 + 解码器因果 | 翻译、摘要 |

```
因果注意力掩码 (Causal Mask):
     猫  坐在  垫子  上
猫  [ 1   0    0    0 ]   ← "猫"只能看到自己
坐在[ 1   1    0    0 ]   ← "坐在"能看到"猫"和自己
垫子[ 1   1    1    0 ]   ← "垫子"能看到前三个
上  [ 1   1    1    1 ]   ← "上"能看到所有
```

### 3.7 练习

1. 画出 Decoder-only Transformer 一层的数据流图
2. 思考：为什么 GPT 类模型使用因果掩码？如果不用会怎样？

---

## Day 4: 预训练与大语言模型

### 4.1 预训练目标

**因果语言模型 (Causal LM / Autoregressive LM)：**

```
目标: 给定前缀，预测下一个 token

输入:  "大语言模型可以"
目标:  "语言模型可以理"

损失函数: 交叉熵损失
  L = -1/T Σ log P(xₜ | x₁, ..., xₜ₋₁)

示例:
  输入 tokens:     [大, 语言, 模型, 可以]
  目标 tokens:     [语言, 模型, 可以, 理解]
  模型预测概率:
    P(语言|大) = 0.02       → -log(0.02) = 3.91
    P(模型|大,语言) = 0.15  → -log(0.15) = 1.90
    P(可以|...) = 0.08      → -log(0.08) = 2.53
    P(理解|...) = 0.05      → -log(0.05) = 3.00
  Loss = (3.91+1.90+2.53+3.00)/4 = 2.84
```

**掩码语言模型 (Masked LM，BERT 使用)：**

```
随机遮挡 15% 的 token，让模型预测：
  输入: "大语言 [MASK] 可以理解自然 [MASK]"
  预测: [MASK₁] = "模型", [MASK₂] = "语言"
```

### 4.2 Tokenization (分词)

现代 LLM 使用子词分词 (Subword Tokenization)：

**BPE (Byte Pair Encoding) 算法：**

```
Step 1: 初始化 — 将所有词拆成字符
  "lower"  → ['l', 'o', 'w', 'e', 'r']
  "lowest" → ['l', 'o', 'w', 'e', 's', 't']
  "newer"  → ['n', 'e', 'w', 'e', 'r']

Step 2: 统计相邻字符对出现频率
  ('l','o'): 2, ('o','w'): 2, ('w','e'): 3, ('e','r'): 2, ...

Step 3: 合并频率最高的字符对
  合并 ('w','e') → 'we'
  "lower"  → ['l', 'o', 'we', 'r']
  "lowest" → ['l', 'o', 'we', 's', 't']
  "newer"  → ['n', 'e', 'we', 'r']

Step 4: 重复 Step 2-3，直到达到目标词汇量
  下一步合并 ('l','o') → 'lo'
  再合并 ('lo','we') → 'lowe'
  ...

最终词汇表: {'l','o','w','e','r','s','t','n', 'we','lo','lowe','lower',...}
```

### 4.3 Scaling Laws (缩放定律)

Kaplan et al. (2020) 和 Chinchilla (Hoffmann et al., 2022) 发现的关键规律：

```
模型性能 (Loss) 与三个因素呈幂律关系:

  L(N) ∝ N^(-α)     N = 模型参数量
  L(D) ∝ D^(-β)     D = 训练数据量 (tokens)
  L(C) ∝ C^(-γ)     C = 计算量 (FLOPs)

Chinchilla 最优配比:
  参数量增加 k 倍 → 数据量也应增加 k 倍

  例如:
  ┌──────────────┬──────────┬──────────────┐
  │ 模型         │ 参数量    │ 最优训练tokens│
  ├──────────────┼──────────┼──────────────┤
  │ 1B 模型      │ 1B       │ ~20B tokens  │
  │ 7B 模型      │ 7B       │ ~140B tokens │
  │ 70B 模型     │ 70B      │ ~1.4T tokens │
  │ 175B (GPT-3) │ 175B     │ ~3.5T tokens │
  └──────────────┴──────────┴──────────────┘
```

### 4.4 主流大模型架构对比

```
┌──────────────┬────────┬────────┬──────────┬──────────────┐
│ 模型         │ 参数量  │ 架构    │ 位置编码  │ 激活函数      │
├──────────────┼────────┼────────┼──────────┼──────────────┤
│ GPT-3        │ 175B   │ Dec    │ 绝对PE   │ GELU         │
│ LLaMA 2      │ 7-70B  │ Dec    │ RoPE     │ SwiGLU       │
│ Mistral 7B   │ 7B     │ Dec    │ RoPE     │ SwiGLU       │
│ Qwen 2       │ 0.5-72B│ Dec    │ RoPE     │ SwiGLU       │
│ DeepSeek V3  │ 671B   │ Dec+MoE│ RoPE     │ SwiGLU       │
└──────────────┴────────┴────────┴──────────┴──────────────┘

现代 LLM 的共同设计选择:
  ✓ Decoder-only 架构
  ✓ RoPE 旋转位置编码
  ✓ SwiGLU 激活函数
  ✓ RMSNorm (替代 LayerNorm)
  ✓ Pre-Norm (而非 Post-Norm)
  ✓ GQA (分组查询注意力，见 Day 6)
```

### 4.5 预训练数据与处理

```
典型的预训练数据管线:

  原始数据 (Common Crawl 等, ~PB级)
      ↓
  语言过滤 (去掉非目标语言)
      ↓
  质量过滤 (规则 + 分类器, 去低质量文本)
      ↓
  去重 (MinHash/SimHash 近似去重)
      ↓
  敏感内容过滤 (PII, 有毒内容)
      ↓
  Tokenization
      ↓
  打包成固定长度序列 (如 4096 tokens)
      ↓
  训练数据 (~T tokens 级别)
```

### 4.6 练习

1. 用 BPE 算法对 ["ab ab cb", "ab cb ab"] 进行 3 轮合并
2. 根据 Chinchilla 缩放定律，一个 13B 模型最优应训练多少 tokens？

---

## Day 5: 微调与对齐技术

### 5.1 从预训练到对齐的流程

```
预训练 (Pre-training)
  ↓  在海量文本上学习语言知识
SFT (Supervised Fine-Tuning / 有监督微调)
  ↓  在高质量指令-回复对上微调
RLHF / DPO (对齐)
  ↓  根据人类偏好进一步优化
部署模型
```

### 5.2 有监督微调 (SFT)

```
训练数据格式:

{
  "instruction": "解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}

训练方式:
  将 instruction + input 拼接为 prompt
  将 output 作为目标
  只在 output 部分计算损失 (不在 prompt 部分计算)

  tokens:   [<s>] [解释] [什么] [是] [机器] [学习] [回答:] [机器] [学习] [是] [...]
  loss mask: [ 0 ] [ 0 ] [ 0 ] [0] [ 0 ] [ 0 ]  [ 0 ]  [ 1 ] [ 1 ] [1] [...]
```

### 5.3 LoRA (Low-Rank Adaptation)

全量微调代价太大。LoRA 冻结原始权重，只训练低秩增量矩阵：

```
核心思想:
  原始: y = W·x           (W 是 d×d 矩阵，参数量 d²)
  LoRA: y = W·x + B·A·x   (A: d×r, B: r×d, r << d)

  冻结 W，只训练 A 和 B
  参数量: 2×d×r << d²

具体例子:
  d = 4096, r = 16
  原始参数量: 4096 × 4096 = 16,777,216 (16M)
  LoRA 参数量: 4096 × 16 + 16 × 4096 = 131,072 (131K)
  压缩比: ~128x

                    ┌─────────────┐
           x ─────→│ W (frozen)  │─────→ + ─→ y
           │        └─────────────┘       ↑
           │        ┌──────┐ ┌──────┐     │
           └──────→ │ A    │→│ B    │─────┘
                    │(d×r) │ │(r×d) │
                    └──────┘ └──────┘
                    ↑ 只训练这部分 ↑

初始化:
  A ~ N(0, σ²)  (随机初始化)
  B = 0          (零初始化，确保训练开始时 ΔW = BA = 0)
```

**QLoRA：** 结合量化进一步降低内存

```
1. 将预训练权重量化到 4-bit (NF4 格式)
2. 用 LoRA 在 4-bit 权重上训练
3. 反向传播时在 BFloat16 精度计算梯度

内存节省:
  7B 模型全量微调:  ~60 GB GPU 内存
  7B 模型 QLoRA:    ~6 GB GPU 内存 (单张消费级 GPU!)
```

### 5.4 RLHF (基于人类反馈的强化学习)

```
Step 1: 训练奖励模型 (Reward Model)
  收集人类偏好数据:
    Prompt: "写一首关于春天的诗"
    Response A: "春风拂面暖阳照..."  ← 人类选择: 更好 ✓
    Response B: "春天就是天气变暖..."  ← 较差

  训练奖励模型:
    R(prompt, response) → 标量分数
    损失: L = -log σ(R(x, y_win) - R(x, y_lose))   (Bradley-Terry 模型)

Step 2: PPO 强化学习优化
  目标:
    max_π  E[R(x, y)] - β · KL(π || π_ref)

  其中:
    π: 当前策略 (正在优化的模型)
    π_ref: 参考策略 (SFT 模型，防止偏离太远)
    β: KL 惩罚系数
    KL: KL散度，约束新策略不要偏离参考策略太远

  流程:
    ┌─────────────────────────────────────────────┐
    │ 1. 当前模型 π 生成回复 y ~ π(·|x)          │
    │ 2. 奖励模型打分 R(x, y)                     │
    │ 3. 计算 KL 惩罚 KL(π||π_ref)               │
    │ 4. 用 PPO 算法更新 π 的参数                  │
    │ 5. 重复                                      │
    └─────────────────────────────────────────────┘
```

### 5.5 DPO (Direct Preference Optimization)

DPO 跳过奖励模型，直接从偏好数据优化策略：

```
核心洞察:
  最优策略的 reward 可以用策略本身的对数概率表示:
  r(x,y) = β · log(π(y|x)/π_ref(y|x)) + const

DPO 损失函数:
  L_DPO = -E[ log σ( β · log(π(y_w|x)/π_ref(y_w|x))
                    - β · log(π(y_l|x)/π_ref(y_l|x)) ) ]

  y_w: 偏好的 (winning) 回复
  y_l: 不好的 (losing) 回复

优势:
  ✓ 不需要单独训练奖励模型
  ✓ 不需要在线采样 (RL 环境)
  ✓ 训练稳定，容易实现
  ✓ 效果与 RLHF 相当甚至更好
```

### 5.6 练习

1. 假设 d=8, r=2，手动计算 LoRA 的参数量相比全量微调节省了多少
2. 解释为什么 RLHF 中需要 KL 散度约束

---

## Day 6: 推理优化与部署

### 6.1 KV Cache

自回归生成中，每生成一个 token 都要重新计算之前所有 token 的 K 和 V，非常浪费。KV Cache 缓存已计算的 K、V 值。

```
无 KV Cache (naive):
  生成第 1 个token: 计算 K,V for [prompt]              → O(n)
  生成第 2 个token: 计算 K,V for [prompt, tok1]         → O(n+1)
  生成第 3 个token: 计算 K,V for [prompt, tok1, tok2]   → O(n+2)
  ...
  总计算量: O(n²)  ← 大量重复计算!

有 KV Cache:
  生成第 1 个token: 计算 K,V for [prompt], 缓存        → O(n)
  生成第 2 个token: 只计算 tok1 的 K,V, 追加到缓存      → O(1)
  生成第 3 个token: 只计算 tok2 的 K,V, 追加到缓存      → O(1)
  ...
  总新增计算量: O(n)  ← 大幅减少!

KV Cache 内存占用:
  每层每个 token: 2 × n_heads × head_dim × 2bytes (fp16)

  LLaMA-2 7B (32层, 32头, head_dim=128):
  每 token: 2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB
  4096 tokens: ~2 GB

  这就是为什么长上下文需要大量内存!
```

### 6.2 GQA (Grouped-Query Attention)

减少 KV Cache 的大小：

```
MHA (Multi-Head Attention):
  Q: 32 heads, K: 32 heads, V: 32 heads
  KV Cache: 32 × 2 × d_head per token

MQA (Multi-Query Attention):
  Q: 32 heads, K: 1 head, V: 1 head
  KV Cache: 1 × 2 × d_head per token
  ↓ KV Cache 减少 32 倍，但质量可能下降

GQA (Grouped-Query Attention) — 折中方案:
  Q: 32 heads, K: 8 groups, V: 8 groups
  每 4 个 Q head 共享 1 组 K,V
  KV Cache: 8 × 2 × d_head per token
  ↓ KV Cache 减少 4 倍，质量接近 MHA

  ┌─────────────────────────────────────┐
  │  Q heads:  q1 q2 q3 q4  q5 q6 q7 q8│
  │              ↓↓↓↓        ↓↓↓↓      │
  │  KV groups:  kv1          kv2       │
  │              (共享)       (共享)     │
  └─────────────────────────────────────┘
```

### 6.3 量化 (Quantization)

```
将高精度浮点数转为低精度整数，减少内存和加速推理:

FP16 → INT8 量化:
  原始值: [0.15, -0.82, 0.43, 1.00, -0.21]  (16-bit each)
  量化:   [19,   -105,  55,   127,  -27]     (8-bit each)

  scale = max(|values|) / 127 = 1.00/127 ≈ 0.00787
  quantized = round(values / scale)
  反量化: dequantized = quantized × scale

  内存节省: 16-bit → 8-bit = 50% 减少

量化方法对比:
  ┌───────────────┬──────────┬──────────┬────────────┐
  │ 方法          │ 精度损失  │ 推理速度  │ 应用场景    │
  ├───────────────┼──────────┼──────────┼────────────┤
  │ FP16/BF16     │ 极小     │ 基准     │ 训练+推理   │
  │ INT8 (W8A8)   │ 小       │ ~1.5x   │ 推理       │
  │ INT4 (W4A16)  │ 中等     │ ~2x     │ 推理       │
  │ GPTQ          │ 小       │ ~2x     │ 推理       │
  │ AWQ           │ 小       │ ~2x     │ 推理       │
  │ GGUF (llama.cpp)│ 可控   │ ~2-3x   │ CPU/边缘   │
  └───────────────┴──────────┴──────────┴────────────┘

7B 模型不同量化下的内存:
  FP32:  ~28 GB
  FP16:  ~14 GB
  INT8:  ~7 GB
  INT4:  ~3.5 GB  ← 可以在消费级 GPU 上运行!
```

### 6.4 PagedAttention (vLLM 核心技术)

```
传统 KV Cache 的问题:
  - 每个请求预分配最大序列长度的连续内存
  - 实际使用远小于预分配 → 大量内存浪费 (60-80%)

  ┌──────────────────────────────────────┐
  │ Request 1: [used][used][    waste    ]│
  │ Request 2: [used][         waste     ]│
  │ Request 3: [used][used][used][waste  ]│
  └──────────────────────────────────────┘

PagedAttention (借鉴 OS 虚拟内存分页):
  将 KV Cache 分成固定大小的"页" (block)
  按需分配，不需要连续内存

  物理块池: [B0][B1][B2][B3][B4][B5][B6]...

  Request 1 的页表: slot0→B0, slot1→B3, slot2→B6
  Request 2 的页表: slot0→B1, slot1→B4
  Request 3 的页表: slot0→B2, slot1→B5

  ✓ 内存浪费从 60-80% 降到 <4%
  ✓ 支持更大的 batch size → 更高吞吐量
  ✓ 天然支持 beam search (页共享)
```

### 6.5 推理加速技术总览

```
Continuous Batching:
  传统: 一批请求全部完成后才处理下一批
  连续: 有请求完成就立即填入新请求

  传统:  [R1,R2,R3] ──完成──→ [R4,R5,R6]
          ↑ R1先完成但要等R3

  连续:  [R1,R2,R3] → R1完成 → [R4,R2,R3] → R3完成 → [R4,R2,R5]
          ↑ 立即填入新请求

Speculative Decoding (投机解码):
  用小模型快速草拟多个 token，大模型并行验证

  小模型 (draft): 快速生成 [t1, t2, t3, t4]
  大模型 (verify): 并行检查 → 接受 [t1, t2, t3], 拒绝 t4
  结果: 一次大模型前向传播生成了 3 个 token!

FlashAttention:
  优化 GPU 内存访问模式，减少 HBM 读写
  通过 tiling (分块) 将注意力计算保持在 SRAM 中
  ↓ 2-4x 注意力计算加速，无精度损失
```

### 6.6 练习

1. 计算 LLaMA-2 13B 模型在 4096 上下文长度下 KV Cache 的内存占用
2. 解释为什么 PagedAttention 类似操作系统的虚拟内存管理

---

## Day 7: 前沿技术与实践

### 7.1 MoE (Mixture of Experts)

```
核心思想: 不是每个 token 都需要所有参数
  将 FFN 层替换为多个"专家"，每次只激活一部分

结构:
  输入 x
    ↓
  Gate(x) = softmax(W_gate · x)  ← 路由器，决定用哪些专家
    ↓
  选择 Top-K 个专家 (通常 K=2)
    ↓
  y = Σ gate_i × Expert_i(x)     ← 加权组合选中专家的输出

                    ┌──────────┐
            ┌──────→│ Expert 1 │──┐
            │       └──────────┘  │
  x → Gate ─┤       ┌──────────┐  ├──→ 加权求和 → y
            │  ┌───→│ Expert 2 │──┤
            │  │    └──────────┘  │
            └──┤    ┌──────────┐  │
               │    │ Expert 3 │  │ (未选中，不计算)
               │    └──────────┘  │
               │    ┌──────────┐  │
               └───→│ Expert 4 │──┘
                    └──────────┘

DeepSeek V3 示例:
  总参数: 671B
  每个 token 激活参数: ~37B
  专家数: 256 个路由专家 + 1 个共享专家
  每次激活: Top-8 个路由专家

负载均衡损失:
  防止所有 token 都选同一个专家
  L_balance = α · Σ fᵢ · pᵢ
  fᵢ = 分配给专家i的token比例
  pᵢ = 路由到专家i的平均概率
```

### 7.2 长上下文技术

```
挑战: 注意力复杂度 O(n²)，n=序列长度

技术演进:
  标准注意力: 4K context (GPT-3)
      ↓
  ALiBi/RoPE 外推: 8K-32K
      ↓
  位置编码插值: 100K+
      ↓
  Ring Attention: 1M+
      ↓
  稀疏注意力: 10M+

RoPE 位置插值 (Position Interpolation):
  训练时最大位置: L = 4096
  想要支持: L' = 32768

  直接外推: 用 pos=32768 计算 RoPE → 效果差，超出训练分布
  线性插值: pos' = pos × (L/L') → 将长位置"压缩"到训练范围内

  ┌──训练范围──┐
  0           4096
  |||||||||||||               ← 原始位置，密集

  0                    32768
  |  |  |  |  |  |  |  |     ← 插值后位置，稀疏但在范围内

NTK-Aware Scaling (YaRN):
  不同频率的维度用不同的缩放因子
  高频维度 (局部关系): 少缩放
  低频维度 (全局关系): 多缩放
  → 更好地保留局部和全局位置信息
```

### 7.3 RAG (Retrieval-Augmented Generation)

```
将外部知识注入 LLM:

  用户问题: "vLLM 的最新版本是什么？"
      ↓
  1. Retrieval (检索):
     将问题编码为向量 → 在知识库中检索相关文档
     ┌────────────────────────┐
     │ 知识库 (向量数据库)      │
     │ doc1: vLLM v0.6 发布... │ ← 相似度 0.92 ✓
     │ doc2: PyTorch 更新...    │ ← 相似度 0.31 ✗
     │ doc3: vLLM 安装指南...   │ ← 相似度 0.85 ✓
     └────────────────────────┘

  2. Augmentation (增强):
     将检索到的文档拼接到 prompt 中:
     "参考以下信息: {doc1} {doc3}\n请回答: vLLM 的最新版本是什么？"

  3. Generation (生成):
     LLM 基于增强后的 prompt 生成回答

优势:
  ✓ 知识可更新 (更新知识库而非重训模型)
  ✓ 减少幻觉 (有据可查)
  ✓ 可追溯来源
```

### 7.4 Agent (智能体)

```
LLM 作为推理引擎，配合工具使用:

ReAct 框架 (Reasoning + Acting):

用户: "北京今天的气温是多少摄氏度？换算成华氏度。"

思考 (Thought): 我需要先查询北京的天气，然后做温度换算
行动 (Action):  调用 weather_api("北京")
观察 (Observation): 返回 {"temp": 22, "unit": "celsius"}

思考: 北京气温22°C，需要换算: F = C × 9/5 + 32
行动: 调用 calculator("22 * 9/5 + 32")
观察: 71.6

思考: 计算完成，可以回答用户
回答: 北京今天气温22°C，换算成华氏度约为71.6°F。

工具定义示例:
  tools = [
    {
      "name": "weather_api",
      "description": "查询指定城市的天气",
      "parameters": {"city": "string"}
    },
    {
      "name": "calculator",
      "description": "计算数学表达式",
      "parameters": {"expression": "string"}
    }
  ]
```

### 7.5 实践路线图

```
入门实践:
  1. 用 Hugging Face transformers 加载并运行一个小模型 (如 Qwen2-0.5B)
  2. 用 vLLM 部署一个模型并测试 API
  3. 用 LoRA 微调一个 7B 模型 (可用 LLaMA-Factory)

进阶实践:
  4. 实现一个简单的 RAG 系统 (LangChain + ChromaDB)
  5. 用 DPO 对齐一个模型
  6. 尝试量化部署 (GPTQ/AWQ)

深入实践:
  7. 阅读 vLLM 源码，理解 PagedAttention 实现
  8. 实现 Speculative Decoding
  9. 从头预训练一个小型 LLM (如 100M 参数)
```

### 7.6 推荐资源

| 类别 | 资源 | 说明 |
|------|------|------|
| 论文 | "Attention Is All You Need" | Transformer 原始论文 |
| 论文 | "Language Models are Few-Shot Learners" | GPT-3 论文 |
| 论文 | "Training Language Models to Follow Instructions with Human Feedback" | InstructGPT/RLHF 论文 |
| 论文 | "LoRA: Low-Rank Adaptation of Large Language Models" | LoRA 论文 |
| 论文 | "Direct Preference Optimization" | DPO 论文 |
| 论文 | "Efficient Memory Management for Large Language Model Serving with PagedAttention" | vLLM 论文 |
| 课程 | Stanford CS224N | 自然语言处理经典课程 |
| 课程 | Andrej Karpathy "Let's build GPT" | 从零实现 GPT 的视频教程 |
| 工具 | Hugging Face Transformers | 模型库和工具集 |
| 工具 | vLLM | 高性能推理引擎 |
| 工具 | LLaMA-Factory | 一站式微调框架 |

---

## 学习计划总结

```
Day 1: 地基      → 语言模型基础、N-gram、NNLM
Day 2: 砖块      → 词嵌入、注意力机制
Day 3: 框架      → Transformer 完整架构
Day 4: 建筑      → 预训练技术、Scaling Laws、主流模型
Day 5: 装修      → SFT、LoRA、RLHF、DPO
Day 6: 优化      → KV Cache、量化、PagedAttention、推理加速
Day 7: 前沿+实践 → MoE、长上下文、RAG、Agent、动手实践

每天建议学习时间: 3-5 小时
  - 1-2 小时阅读理论
  - 1-2 小时动手实践/计算
  - 0.5-1 小时阅读相关论文或视频
```

> **提示：** 学习 LLM 最好的方式是理论 + 实践结合。每个概念学完后，尝试用代码实现或在已有框架中找到对应实现。
