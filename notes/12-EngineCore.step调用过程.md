# EngineCore.step() 调用过程详解

## 1. 概述

`EngineCore.step()` 是 vLLM v1 推理引擎的核心方法，负责完成一次完整的 **调度 → 模型执行 → 输出处理** 循环。每次调用产生一批推理结果。

**文件位置：** `vllm/v1/engine/core.py:374`

---

## 2. 调用链总览

```
用户代码 (离线批量推理)
  │
  ▼
LLM._run_engine()                    # vllm/entrypoints/llm.py:2138
  │
  ▼
LLMEngine.step()                     # vllm/v1/engine/llm_engine.py:318
  │
  ▼
EngineCoreClient.get_output()        # vllm/v1/engine/core_client.py
  ├─ InprocClient  → engine_core.step_fn()     [同进程直接调用]
  └─ SyncMPClient  → ZMQ 轮询后台进程输出      [多进程通信]

后台进程 (或同进程模式):
  │
  ▼
EngineCore.run_busy_loop()           # vllm/v1/engine/core.py:1102
  │
  ▼
_process_engine_step()               # vllm/v1/engine/core.py:1144
  │
  ▼
step_fn()
  ├─ step()                          # 基本模式
  └─ step_with_batch_queue()         # 批次队列模式 (max_concurrent_batches > 1)
```

---

## 3. step() 方法详解

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
```

**返回值：** `(输出字典, 模型是否执行)` —— 字典的 key 是 client index，value 是该 client 的所有请求输出。

### 3.1 执行流程图

```
┌─────────────────────────────────────┐
│     scheduler.has_requests() ?      │
│  检查是否有未完成/已完成但未返回的请求 │
└──────────┬──────────────────────────┘
           │
     ┌─────┴─────┐
     │ No        │ Yes
     ▼           ▼
  return      ┌──────────────────────┐
  ({}, False)  │ scheduler.schedule() │
              │ 执行调度，分配token预算 │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │ model_executor.execute_model() │
              │ 非阻塞提交模型前向传播         │
              │ 返回 Future                    │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────┐
              │ scheduler.get_grammar_bitmask()   │
              │ 获取结构化输出的语法约束位掩码     │
              │ (与模型执行并行计算)               │
              └──────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │ future.result()               │
              │ 阻塞等待模型前向传播完成       │
              └──────────┬───────────────────┘
                         │
                    ┌────┴────┐
                    │ None?   │
                    ▼         ▼
              ┌──────────┐  直接使用
              │ sample_  │  model_output
              │ tokens() │
              └────┬─────┘
                   │
                   ▼
              ┌──────────────────────────────┐
              │ _process_aborts_queue()       │
              │ 处理执行期间积累的中止请求     │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────┐
              │ scheduler.update_from_output()    │
              │ 用模型输出更新调度器状态           │
              │ 生成 EngineCoreOutputs            │
              └──────────┬───────────────────────┘
                         │
                         ▼
              return (engine_core_outputs,
                      total_num_scheduled_tokens > 0)
```

---

## 4. 各阶段详解

### 4.1 调度阶段：scheduler.schedule()

**文件位置：** `vllm/v1/core/sched/scheduler.py:385`

调度器是整个推理流水线的"大脑"，决定本轮哪些请求参与推理、每个请求处理多少个 token。

**两阶段调度：**

| 阶段 | 说明 |
|------|------|
| **Phase 1：调度 RUNNING 请求** | 遍历正在运行的请求，计算每个请求需要的新 token 数，分配 KV cache 槽位。如果 KV cache 不足，抢占低优先级请求 |
| **Phase 2：调度 WAITING 请求** | 仅在无抢占时执行。检查 LoRA 约束、查询前缀缓存、分配 KV cache、处理异步 KV 传输、调度多模态编码器输入 |

**关键约束：**
- `token_budget`：本轮最大处理 token 数
- `max_num_running_reqs`：最大并发请求数
- `long_prefill_token_threshold`：长 prefill 分块阈值
- `max_loras`：最大 LoRA 适配器数量

**输出：** `SchedulerOutput`，包含：
- 新/缓存请求的调度数据
- 每个请求的 token 数
- 被抢占的请求 ID
- 已完成的请求 ID
- KV cache 元数据

### 4.2 模型执行阶段：model_executor.execute_model()

**文件位置：** `vllm/v1/executor/abstract.py:246`

```python
exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

**关键设计：**
- 使用 `collective_rpc` 模式向所有 Worker 广播执行命令
- 支持同步（阻塞）和异步（非阻塞）两种模式
- 在 tensor parallel 场景下，只返回 rank-0 Worker 的输出
- 实际执行 GPU 上的模型前向传播（attention、FFN 等）

**执行器实现：**
| 类型 | 类名 | 说明 |
|------|------|------|
| 单进程 | `UniProcExecutor` | Worker 在同一进程内，直接调用 |
| 多进程 | `MultiprocExecutor` | Worker 在独立进程，通过 IPC 通信 |
| Ray 分布式 | `RayDistributedExecutor` | Worker 分布在多节点 |

### 4.3 语法约束：scheduler.get_grammar_bitmask()

**文件位置：** `vllm/v1/core/sched/scheduler.py:1361`

在模型前向传播的同时，并行计算结构化输出（如 JSON schema）的语法约束位掩码：

1. 筛选使用结构化输出且不在 prefill 阶段的请求
2. 调用 `structured_output_manager.grammar_bitmask()` 生成位掩码
3. 位掩码用于在采样时屏蔽不符合语法的 token

### 4.4 Token 采样：model_executor.sample_tokens()

**文件位置：** `vllm/v1/executor/abstract.py:268`

当 `execute_model()` 返回 `None`（表示前向传播和采样被分离执行时），需要单独调用采样：

```python
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

采样过程应用：
- 温度缩放、top-k/top-p 过滤
- 语法约束位掩码
- 频率/存在惩罚
- logprobs 提取

### 4.5 输出处理：scheduler.update_from_output()

**文件位置：** `vllm/v1/core/sched/scheduler.py:1394`

这是 step 的最后阶段，将模型输出转化为用户可消费的结果：

1. **提取模型输出** — 采样 token、logprobs、pooler 输出等
2. **处理 KV cache 失败** — 识别需要重新计算的请求
3. **逐请求处理：**
   - 获取生成的 token
   - 处理推测解码：计算接受/拒绝的 token
   - 检查停止条件（max_tokens、stop_token_ids、stop_strings）
   - 更新结构化输出语法状态
   - 构建 `EngineCoreOutput`
4. **清理已完成请求** — 从 running/waiting 队列移除
5. **更新 KV connector** — 处理完成的 KV 传输
6. **收集事件并发布** — KV cache 事件批次
7. **按 client 聚合输出** — 将所有请求输出按 client index 分组
8. **生成统计信息** — 调度统计、推测解码统计等

---

## 5. step_with_batch_queue() 变体

**文件位置：** `vllm/v1/engine/core.py:412`

当 `max_concurrent_batches > 1` 时启用，支持流水线化执行：

```
时间线:  ──────────────────────────────────────────►
批次1:   [调度] [前向传播............] [采样] [输出处理]
批次2:          [调度] [前向传播............] [采样] [输出处理]
批次3:                 [调度] [前向传播............] ...
```

**与 step() 的区别：**
- 维护一个 `batch_queue`（双端队列），存储 `(future, scheduler_output, exec_future)` 三元组
- 调度新批次的优先级高于等待输出（填满队列优先）
- 支持延迟采样：当有结构化输出 + 推测解码时，需要等前一步输出来计算语法位掩码
- 只在队列已满或没有更多请求时才阻塞等待

---

## 6. 主循环：run_busy_loop()

**文件位置：** `vllm/v1/engine/core.py:1102`

```python
def run_busy_loop(self):
    while True:
        # 1. 处理输入队列（新请求、中止请求等）
        self._process_input_queue()

        # 2. 执行一步推理
        has_output = self._process_engine_step()

        # 3. 检查关闭信号
        if self._should_shutdown():
            break
```

`_process_engine_step()` (line 1144) 调用 `step_fn()` 并通过输出队列将结果发送给前端：

```python
def _process_engine_step(self) -> bool:
    outputs, model_executed = self.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    if outputs:
        self._send_outputs(outputs)
        return True
    return False
```

---

## 7. post_step() 后处理

**文件位置：** `vllm/v1/engine/core.py:402`

在 step 完成后执行，主要用于推测解码场景：

```python
def post_step(self, model_executed: bool) -> None:
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
```

从模型执行器获取草稿 token ID，更新到调度器中供下一轮调度使用。

---

## 8. 完整时序图

```
    EngineCore          Scheduler           Executor           Worker/GPU
        │                   │                   │                   │
        │  has_requests()   │                   │                   │
        │──────────────────►│                   │                   │
        │◄──────────────────│                   │                   │
        │                   │                   │                   │
        │  schedule()       │                   │                   │
        │──────────────────►│                   │                   │
        │  ┌────────────────┤                   │                   │
        │  │Phase1: RUNNING │                   │                   │
        │  │ - 计算token数   │                   │                   │
        │  │ - 分配KV cache  │                   │                   │
        │  │ - 抢占低优先级  │                   │                   │
        │  ├────────────────┤                   │                   │
        │  │Phase2: WAITING │                   │                   │
        │  │ - 查前缀缓存   │                   │                   │
        │  │ - 分配KV cache  │                   │                   │
        │  │ - 调度编码器    │                   │                   │
        │  └────────────────┤                   │                   │
        │◄──SchedulerOutput─│                   │                   │
        │                   │                   │                   │
        │  execute_model(non_block=True)        │                   │
        │──────────────────────────────────────►│  collective_rpc   │
        │◄─────────Future───────────────────────│──────────────────►│
        │                   │                   │    前向传播 (GPU)   │
        │  get_grammar_     │                   │         ⋮          │
        │  bitmask()        │                   │         ⋮          │
        │──────────────────►│ (并行计算)         │         ⋮          │
        │◄─GrammarOutput────│                   │         ⋮          │
        │                   │                   │                   │
        │  future.result()  │                   │                   │
        │  ═══阻塞等待═══════════════════════════│◄──────────────────│
        │                   │                   │                   │
        │  [若output=None]  │                   │                   │
        │  sample_tokens()  │                   │                   │
        │──────────────────────────────────────►│  采样 (GPU)        │
        │◄─ModelRunnerOutput────────────────────│◄──────────────────│
        │                   │                   │                   │
        │  _process_aborts  │                   │                   │
        │  _queue()         │                   │                   │
        │──────┐            │                   │                   │
        │◄─────┘            │                   │                   │
        │                   │                   │                   │
        │  update_from_     │                   │                   │
        │  output()         │                   │                   │
        │──────────────────►│                   │                   │
        │  ┌────────────────┤                   │                   │
        │  │- 处理推测解码   │                   │                   │
        │  │- 检查停止条件   │                   │                   │
        │  │- 更新语法状态   │                   │                   │
        │  │- 清理完成请求   │                   │                   │
        │  │- 聚合输出       │                   │                   │
        │  └────────────────┤                   │                   │
        │◄─EngineCoreOutputs│                   │                   │
        │                   │                   │                   │
        │  post_step()      │                   │                   │
        │──────┐            │                   │                   │
        │◄─────┘            │                   │                   │
        │                   │                   │                   │
```

---

## 9. 关键数据结构

| 结构 | 定义位置 | 说明 |
|------|---------|------|
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | 调度结果，包含请求分配、token 数、KV cache 元数据 |
| `ModelRunnerOutput` | `vllm/v1/worker/gpu/` | 模型前向传播 + 采样输出 |
| `GrammarOutput` | `vllm/v1/core/sched/scheduler.py` | 语法约束位掩码 |
| `EngineCoreOutput` | `vllm/v1/engine/__init__.py:230` | 单个请求的输出（token、finish_reason、logprobs） |
| `EngineCoreOutputs` | `vllm/v1/engine/__init__.py:307` | 一批请求的聚合输出 |

---

## 10. 设计亮点

1. **异步流水线化**：`execute_model` 非阻塞提交，语法位掩码与前向传播并行计算
2. **批次队列**：`step_with_batch_queue()` 支持多批次重叠执行，提高 GPU 利用率
3. **推测解码集成**：在调度和输出处理中无缝集成推测解码的 draft/verify 机制
4. **优雅降级**：KV cache 不足时通过抢占机制保证系统稳定运行
5. **结构化输出**：通过语法位掩码在采样层面约束输出格式
