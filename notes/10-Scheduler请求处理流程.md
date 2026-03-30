# Scheduler 请求处理流程

本文档介绍 `vllm/v1/core/sched/scheduler.py` 中 `Scheduler` 类的请求消息处理流程。

---

## 一、整体架构概览

Scheduler 是 vLLM V1 的核心调度器，实现了 `SchedulerInterface` 接口。它负责管理请求的生命周期，决定每一步哪些请求被调度执行、分配多少 token。

**核心设计思想：没有 prefill/decode 阶段划分。** 每个请求维护 `num_computed_tokens` 和 `num_tokens_with_spec`，每步让 computed 追赶 total，统一处理 chunked prefill、prefix caching 和 speculative decoding。

### 关键数据结构

| 数据结构 | 类型 | 说明 |
|---------|------|------|
| `requests` | `dict[str, Request]` | 请求ID → 请求对象的全局映射 |
| `waiting` | `RequestQueue` | 等待调度的请求队列 |
| `skipped_waiting` | `RequestQueue` | 因异步依赖/约束暂时跳过的等待请求 |
| `running` | `list[Request]` | 正在运行的请求列表 |
| `finished_req_ids` | `set[str]` | 两步之间已完成的请求ID |

### 关键组件

- **kv_cache_manager** — KV cache block 分配与管理
- **encoder_cache_manager** — 多模态编码器缓存
- **connector** — KV 传输连接器（P/D 分离架构）
- **structured_output_manager** — 结构化输出语法约束

---

## 二、请求生命周期状态机

```
                  add_request()
                      │
                      ▼
    ┌──────────── WAITING ◄──────────────┐
    │                 │                  │
    │    schedule()   │   _preempt_      │
    │    分配KV块     │   request()      │
    │                 ▼                  │
    │            RUNNING ────────────────┘
    │                 │            (KV块不足时抢占)
    │                 │
    │   update_from_output()
    │                 │
    │        ┌────────┴────────┐
    │        ▼                 ▼
    │  FINISHED_STOPPED  FINISHED_LENGTH_CAPPED
    │  FINISHED_ABORTED  FINISHED_ERROR
    │
    │  (可恢复的流式请求)
    │        │
    │        ▼
    │  WAITING_FOR_STREAMING_REQ ──(收到新输入)──► WAITING
    │
    │  (P/D分离场景)
    └──► WAITING_FOR_REMOTE_KVS ──(KV接收完成)──► WAITING
```

---

## 三、请求入队：`add_request()`

**入口**：`scheduler.py:1869`

`add_request()` 是所有请求进入调度器的唯一入口。

### 3.1 新请求

```
add_request(request)
  │
  ├─ request.resumable? → 初始化 streaming_queue（双端队列）
  │
  ├─ _enqueue_waiting_request(request) → 加入 waiting 或 skipped_waiting 队列
  │
  ├─ self.requests[request_id] = request  → 注册到全局映射
  │
  └─ 记录 QUEUED 事件（统计用）
```

### 3.2 流式输入（同一 request_id 多次调用）

当同一 `request_id` 再次调用 `add_request()` 时：

- **请求正在处理中**：将新的输入块追加到 `streaming_queue`
- **请求在等待流式输入（WAITING_FOR_STREAMING_REQ）**：立即用 `_update_request_as_session()` 更新请求并重新入队
- **流式结束标记**：中止请求

---

## 四、核心调度循环：`schedule()`

**入口**：`scheduler.py:385`

EngineCore 主循环每步调用一次 `schedule()`，返回 `SchedulerOutput` 告诉 model runner 本步处理哪些请求的多少 token。

### 4.1 调度约束

- `max_num_running_reqs` — 最大并发运行请求数
- `max_num_scheduled_tokens` — 每步 token 预算上限
- `encoder_compute_budget` — 编码器计算预算（多模态场景）
- `max_model_len` — 模型最大序列长度

### 4.2 第一阶段：调度 RUNNING 请求

```
遍历 running 队列:
  │
  ├─ 计算 num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
  │
  ├─ 应用约束: min(num_new_tokens, token_budget, max_model_len - 1 - computed)
  │
  ├─ 长预填充阈值截断 (long_prefill_token_threshold)
  │
  ├─ 调度编码器输入 (_try_schedule_encoder_inputs)  [多模态]
  │
  ├─ Mamba块对齐分割 (_mamba_block_aligned_split)   [混合模型]
  │
  ├─ 分配KV缓存块 (kv_cache_manager.allocate_slots)
  │   │
  │   ├─ 成功 → 记录到 req_to_new_blocks, num_scheduled_tokens
  │   │
  │   └─ 失败 → 抢占最低优先级请求，循环重试
  │       │
  │       ├─ PRIORITY 策略: 抢占 priority 和 arrival_time 最大的请求
  │       └─ FCFS 策略: 抢占队列末尾的请求
  │
  ├─ 处理推测解码 token (spec_token_ids)
  │
  └─ 分配编码器缓存
```

**抢占机制详解**：当 KV 块不足时，调度器会抢占低优先级请求来释放内存：
1. 从 running 队列移除被抢占请求
2. 调用 `_preempt_request()`：释放 KV 缓存 → 重置 num_computed_tokens → 放回 waiting 队列头部
3. 被抢占请求已使用的 token 预算、块分配、编码器预算全部回退

### 4.3 第二阶段：调度 WAITING 请求

**前提条件**：第一阶段没有发生抢占，且调度器未暂停。

```
遍历 waiting + skipped_waiting 队列:
  │
  ├─ 检查 max_num_running_reqs 是否已满
  │
  ├─ 检查阻塞状态 (_is_blocked_waiting_status)
  │   ├─ WAITING_FOR_FSM — 等待有限状态机（结构化输出）
  │   ├─ WAITING_FOR_REMOTE_KVS — 等待远程KV传输
  │   └─ WAITING_FOR_STREAMING_REQ — 等待流式输入
  │   无法提升 → 跳过，加入 step_skipped_waiting
  │
  ├─ 检查 LoRA 约束 (max_loras 上限)
  │
  ├─ 获取前缀缓存命中 (kv_cache_manager.get_computed_blocks)
  │
  ├─ [P/D分离] 获取外部缓存命中 (connector.get_num_new_matched_tokens)
  │   └─ 异步加载 → 设为 WAITING_FOR_REMOTE_KVS，跳过本步调度
  │
  ├─ 计算 num_new_tokens = request.num_tokens - num_computed_tokens
  │
  ├─ 分块预填充检查 (chunked_prefill)
  │
  ├─ 调度编码器输入 + Mamba块对齐
  │
  ├─ 分配KV缓存块 (allocate_slots)
  │   └─ 失败 → 停止调度更多等待请求（不抢占）
  │
  ├─ 从 waiting 弹出，加入 running
  │
  └─ 设置状态为 RUNNING, 记录 num_computed_tokens
```

### 4.4 构建调度输出

调度完成后，构建 `SchedulerOutput` 包含：

| 字段 | 说明 |
|------|------|
| `scheduled_new_reqs` | 本步新调度的请求数据 |
| `scheduled_cached_reqs` | 已在运行的请求数据（增量更新） |
| `num_scheduled_tokens` | 每个请求的调度 token 数 |
| `scheduled_spec_decode_tokens` | 推测解码 token |
| `scheduled_encoder_inputs` | 编码器输入 |
| `preempted_req_ids` | 被抢占的请求ID |
| `finished_req_ids` | 已完成的请求ID |

最后调用 `_update_after_schedule()`，将所有已调度请求的 `num_computed_tokens` 推进。

---

## 五、输出处理：`update_from_output()`

**入口**：`scheduler.py:1394`

模型推理完成后，EngineCore 调用此方法更新调度器状态。

```
update_from_output(scheduler_output, model_runner_output)
  │
  ├─ 获取模型输出: sampled_token_ids, logprobs, pooler_output 等
  │
  ├─ 处理KV加载失败 (_handle_invalid_blocks)
  │
  ├─ 遍历所有调度的请求:
  │   │
  │   ├─ 推测解码处理:
  │   │   ├─ 计算被接受/拒绝的 token 数
  │   │   └─ 回退 num_computed_tokens（减去被拒绝数）
  │   │
  │   ├─ _update_request_with_output():
  │   │   ├─ 逐个追加 output_token_id 到请求
  │   │   └─ 调用 check_stop() 检查停止条件:
  │   │       ├─ EOS token
  │   │       ├─ stop_token_ids
  │   │       ├─ max_tokens 达到上限
  │   │       └─ 模型最大长度
  │   │
  │   ├─ 池化请求: 有输出即完成
  │   │
  │   ├─ 如果停止:
  │   │   ├─ _handle_stopped_request()
  │   │   │   ├─ 不可恢复 → 返回 True（真正完成）
  │   │   │   ├─ 流式队列有数据 → 更新会话，重新入队
  │   │   │   └─ 流式队列空 → WAITING_FOR_STREAMING_REQ
  │   │   │
  │   │   └─ _free_request() → 释放 KV 缓存 + 编码器缓存
  │   │
  │   ├─ 提取 logprobs（如请求）
  │   │
  │   ├─ 更新结构化输出语法状态 (grammar.accept_tokens)
  │   │
  │   └─ 构建 EngineCoreOutput 加入输出
  │
  ├─ 从 running/waiting 队列移除已停止请求
  │
  ├─ 更新 KV 传输状态 (_update_from_kv_xfer_finished)
  │
  ├─ 收集并发布 KV 缓存事件
  │
  └─ 返回 dict[client_index → EngineCoreOutputs]
```

---

## 六、请求终止：`finish_requests()`

**入口**：`scheduler.py:1894`

外部触发的请求终止（如客户端断开、前端检测到 stop string）。

```
finish_requests(request_ids, finished_status)
  │
  ├─ 第一遍: 收集需要移除的请求
  │   ├─ RUNNING → running_requests_to_remove
  │   └─ 其他 → waiting_requests_to_remove
  │
  ├─ 批量从队列移除 (remove_all / remove_requests)
  │
  └─ 第二遍: 设置完成状态 + _free_request()
      ├─ 通知 KV 连接器 (_connector_finished)
      ├─ 释放编码器缓存
      ├─ 记录到 finished_req_ids
      └─ 释放 KV 缓存块 (_free_blocks)
```

---

## 七、完整请求流程示意图

```
用户请求 (API Server)
    │
    ▼
EngineCore.add_request()
    │
    ▼
Scheduler.add_request()  ──────────────────────────────────►  waiting 队列
                                                                  │
                                                                  │
EngineCore 主循环 ─────► Scheduler.schedule()                     │
                              │                                   │
                    ┌─────────┴──────────┐                        │
                    ▼                    ▼                         │
              Phase 1:              Phase 2:                      │
            调度 RUNNING          调度 WAITING  ◄─────────────────┘
                    │                    │
                    └─────────┬──────────┘
                              │
                              ▼
                      SchedulerOutput
                    (发送给 Model Runner)
                              │
                              ▼
                    GPU 执行模型推理
                              │
                              ▼
                    ModelRunnerOutput
                              │
                              ▼
              Scheduler.update_from_output()
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
              未完成请求              已完成请求
            (继续下一步)          (_free_request)
                                       │
                                       ▼
                              EngineCoreOutputs
                              (返回给用户)
```

---

## 八、高级特性

### 8.1 推测解码 (Speculative Decoding)

- 草稿模型生成 `spec_token_ids`，附加到请求
- 调度时额外调度这些推测 token（`num_tokens_with_spec`）
- 输出处理时计算接受/拒绝：`num_accepted = len(generated) - 1`
- 被拒绝的 token 回退 `num_computed_tokens`

### 8.2 P/D 分离 (KV Transfer)

- 通过 `KVConnector` 查询外部缓存命中（`get_num_new_matched_tokens`）
- 异步加载时请求进入 `WAITING_FOR_REMOTE_KVS` 状态
- 传输完成后通过 `_update_waiting_for_remote_kv()` 恢复为 `WAITING`
- 传输失败可选重计算 (`recompute`) 或报错 (`error`)

### 8.3 流式输入 (Streaming Input)

- 请求标记为 `resumable`，初始化 `streaming_queue`
- 完成当前块后进入 `WAITING_FOR_STREAMING_REQ`
- 新输入块到达 → `_update_request_as_session()` 更新并重新入队
- 支持增量追加 prompt token 和多模态特征

### 8.4 前缀缓存 (Prefix Caching)

- 新请求入队时通过 `kv_cache_manager.get_computed_blocks()` 查找本地前缀缓存
- 命中的 token 跳过计算，直接从 `num_computed_tokens` 开始调度
- 记录在 `request.num_cached_tokens` 中用于统计

### 8.5 结构化输出 (Structured Output)

- `structured_output_manager` 管理语法约束
- 调度时通过 `get_grammar_bitmask()` 生成位掩码约束采样
- 输出处理时通过 `grammar.accept_tokens()` 推进语法状态
- 推测解码的 token 也需通过 `grammar.validate_tokens()` 验证

---

## 九、关键方法索引

| 方法 | 行号 | 说明 |
|------|------|------|
| `__init__` | 90 | 初始化调度器 |
| `schedule` | 385 | 单步调度入口 |
| `_preempt_request` | 1025 | 抢占请求 |
| `_update_after_schedule` | 1052 | 调度后更新状态 |
| `_try_schedule_encoder_inputs` | 1197 | 调度编码器输入 |
| `get_grammar_bitmask` | 1361 | 获取结构化输出位掩码 |
| `update_from_output` | 1394 | 模型输出后更新状态 |
| `_handle_stopped_request` | 1699 | 处理停止的请求 |
| `_update_request_with_output` | 1749 | 追加 token 并检查停止条件 |
| `update_draft_token_ids` | 1796 | 更新推测解码 token |
| `add_request` | 1869 | 添加新请求 |
| `finish_requests` | 1894 | 外部终止请求 |
| `_free_request` | 1962 | 释放请求资源 |
