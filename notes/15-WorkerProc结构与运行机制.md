# WorkerProc 结构与运行机制

## 1. 概述

`WorkerProc` 是 vLLM 多进程执行器（`MultiprocExecutor`）中的核心组件，位于 `vllm/v1/executor/multiproc_executor.py`。每个 `WorkerProc` 实例运行在独立的子进程中，对应一块 GPU，负责接收主进程的 RPC 调度指令、执行模型推理并返回结果。

**核心定位**：WorkerProc 是主进程（MultiprocExecutor）与实际计算单元（Worker）之间的**进程级桥梁**。

```
┌─────────────────────────────────────────────────────────┐
│                    主进程 (MultiprocExecutor)             │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ rpc_broadcast_mq │    │ response_mqs[0..N-1]     │   │
│  │   (广播队列)      │    │   (响应队列列表)          │   │
│  └────────┬─────────┘    └──────────▲───────────────┘   │
│           │                         │                    │
└───────────┼─────────────────────────┼────────────────────┘
            │ 共享内存广播             │ 共享内存回传
            ▼                         │
┌───────────────────┐  ┌──────────────────────┐  ┌───────────────────┐
│  WorkerProc 0     │  │  WorkerProc 1        │  │  WorkerProc N-1   │
│  (GPU 0)          │  │  (GPU 1)             │  │  (GPU N-1)        │
│                   │  │                      │  │                   │
│ ┌───────────────┐ │  │ ┌──────────────────┐ │  │ ┌───────────────┐ │
│ │WorkerWrapper  │ │  │ │ WorkerWrapper    │ │  │ │WorkerWrapper  │ │
│ │  └─Worker     │ │  │ │  └─Worker        │ │  │ │  └─Worker     │ │
│ │   (实际推理)   │ │  │ │   (实际推理)      │ │  │ │   (实际推理)   │ │
│ └───────────────┘ │  │ └──────────────────┘ │  │ └───────────────┘ │
└───────────────────┘  └──────────────────────┘  └───────────────────┘
```

---

## 2. 类层次结构

```
WorkerProc                   # 进程封装，运行在子进程中
  └── self.worker: WorkerWrapperBase   # Worker 包装器（属性代理）
        └── self.worker: WorkerBase    # 真正的 Worker（如 GPUWorker）
              └── self.model_runner    # 模型运行器（执行推理计算）
```

| 类 | 文件 | 职责 |
|---|---|---|
| `WorkerProc` | `vllm/v1/executor/multiproc_executor.py:586` | 子进程生命周期管理、RPC 消息循环、死亡监控 |
| `WorkerWrapperBase` | `vllm/v1/worker/worker_base.py:200` | 延迟初始化 Worker、属性代理、多模态缓存管理 |
| `WorkerBase` | `vllm/v1/worker/worker_base.py:36` | Worker 抽象接口（init_device / load_model / execute_model） |
| `MultiprocExecutor` | `vllm/v1/executor/multiproc_executor.py:124` | 主进程端执行器，管理所有 WorkerProc |

---

## 3. 数据类：进程句柄

WorkerProc 使用两阶段句柄来管理子进程的生命周期：

### 3.1 UnreadyWorkerProcHandle（未就绪句柄）

```python
@dataclass
class UnreadyWorkerProcHandle:        # 行 541
    proc: BaseProcess                  # Worker 子进程对象
    rank: int                          # 全局 rank
    ready_pipe: Connection             # 就绪通知管道（子→父）
    death_writer: Connection | None    # 死亡检测管道写端（父持有，关闭时通知子退出）
```

创建子进程后立即返回此句柄，此时 Worker 尚在初始化中。

### 3.2 WorkerProcHandle（就绪句柄）

```python
@dataclass
class WorkerProcHandle:               # 行 553
    proc: BaseProcess                  # Worker 子进程对象
    rank: int                          # 全局 rank
    worker_response_mq: MessageQueue | None    # 本地响应消息队列
    peer_worker_response_mqs: list[MessageQueue | None]  # 远程响应队列（多节点）
    death_writer: Connection | None    # 死亡检测管道
```

Worker 初始化完成并发送 READY 信号后，由 `from_unready_handle()` 升级创建。

---

## 4. WorkerProc 完整生命周期

### 4.1 创建阶段

```
MultiprocExecutor._init_executor()
  │
  ├── set_multiprocessing_worker_envs()     # 设置 spawn 模式、限制 OMP 线程
  ├── MessageQueue(world_size, ...)         # 创建 RPC 广播消息队列
  │
  └── for each local_rank:
        WorkerProc.make_worker_process()    # 工厂方法（行 696）
          │
          ├── 创建 ready_pipe (单向管道)     # 子进程→父进程的就绪通知
          ├── 创建 death_pipe (单向管道)     # 父进程→子进程的死亡检测
          ├── context.Process(target=worker_main, daemon=True)  # 创建守护进程
          ├── proc.start()                  # 启动子进程
          ├── 父进程关闭管道的子进程端        # ready_writer.close(), death_reader.close()
          └── 返回 UnreadyWorkerProcHandle
```

### 4.2 初始化阶段（子进程内）

```
WorkerProc.worker_main() (静态方法，行 846)  ← 子进程入口
  │
  ├── 注册信号处理器 (SIGTERM / SIGINT → SystemExit)
  ├── 关闭从父进程继承的多余文件描述符
  ├── 初始化 Worker 追踪器
  │
  ├── WorkerProc.__init__()                 # 行 630
  │     ├── WorkerWrapperBase(rpc_rank, global_rank)  # 创建 Wrapper
  │     ├── wrapper.init_worker(all_kwargs)           # 延迟初始化真正的 Worker
  │     │     ├── resolve_obj_by_qualname(worker_cls) # 按配置解析 Worker 类
  │     │     ├── 可选：动态注入 worker_extension_cls  # 扩展 Worker 功能
  │     │     └── worker_class(**kwargs)               # 实例化 Worker（如 GPUWorker）
  │     ├── worker.init_device()                       # 初始化 GPU 设备
  │     ├── worker.load_model()                        # 加载模型权重
  │     └── _init_message_queues()                     # 建立共享内存通信队列
  │           ├── 单节点：MessageQueue.create_from_handle() + MessageQueue(1,1)
  │           └── 多节点：通过分布式组创建跨节点队列
  │
  ├── monitor_death_pipe(death_pipe)        # 启动死亡管道监控线程
  │
  ├── ready_writer.send({"status":"READY", "handle":..., ...})  # 通知父进程就绪
  │
  ├── rpc_broadcast_mq.wait_until_ready()   # 等待广播队列握手完成
  ├── worker_response_mq.wait_until_ready() # 等待响应队列握手完成
  │
  └── worker_busy_loop()                    # 进入主循环 ← 核心运行逻辑
```

### 4.3 就绪等待（父进程端）

```
WorkerProc.wait_for_ready(unready_handles)   # 行 767
  │
  ├── multiprocessing.connection.wait(pipes) # 同时监听所有 Worker 的 ready_pipe
  ├── pipe.recv()                            # 接收 {"status":"READY", "handle":...}
  ├── 创建 MessageQueue.create_from_handle() # 建立响应队列连接
  └── WorkerProcHandle.from_unready_handle() # 升级为完整句柄
```

---

## 5. 通信机制

### 5.1 共享内存消息队列（MessageQueue）

vLLM 使用基于共享内存 + ZMQ 的 `MessageQueue`（定义在 `vllm/distributed/device_communicators/shm_broadcast.py`）实现高效的零拷贝进程间通信。

| 队列 | 方向 | 用途 |
|---|---|---|
| `rpc_broadcast_mq` | 主进程 → 所有 Worker | 广播 RPC 调用（方法名、参数） |
| `worker_response_mq` | Worker → 主进程 | 返回执行结果或异常 |

**消息格式**：

```python
# 广播消息（主进程发出）
(method: str|bytes, args: tuple, kwargs: dict, output_rank: int|None)

# 响应消息（Worker 返回）
(ResponseStatus.SUCCESS|FAILURE, result: Any|str)
```

### 5.2 管道通信

| 管道 | 方向 | 用途 |
|---|---|---|
| `ready_pipe` | 子进程 → 父进程 | 一次性：Worker 初始化完成后发送 READY 信号 |
| `death_pipe` | 父进程 → 子进程 | 持续监控：父进程退出时管道自动关闭，子进程检测到 EOF 后自行清理 |

### 5.3 output_rank 优化

在流水线并行（PP）场景下，只有**最后一个 PP 阶段的 TP rank=0 的 Worker**需要返回 `ModelRunnerOutput`。其他 Worker 的结果被丢弃（`output_rank` 机制），避免不必要的通信开销。

```python
# 计算公式（行 531）
output_rank = world_size - tp_size * pcp_size

# 例：TP=8, PP=4 → world_size=32 → output_rank = 32 - 8 = 24
# rank 24 即为最后 PP 阶段的首个 TP Worker
```

---

## 6. 主循环（worker_busy_loop）

```python
def worker_busy_loop(self):                              # 行 987
    while True:
        method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(indefinite=True)

        # 1. 解析方法
        if isinstance(method, str):
            func = getattr(self.worker, method)          # 反射调用 Worker 方法
        elif isinstance(method, bytes):
            func = partial(cloudpickle.loads(method), self.worker)  # 反序列化

        # 2. 执行
        output = func(*args, **kwargs)

        # 3. 返回（仅 output_rank 匹配时）
        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)
```

### 6.1 异步输出处理

当 `async_scheduling=True` 时，WorkerProc 启动一个额外的 `WorkerAsyncOutputCopy` 线程：

```
主循环线程                          异步输出线程
    │                                  │
    ├── execute_model(...)             │
    ├── handle_output(output) ────────►│ async_output_queue.put(output)
    │                                  │
    ├── (继续处理下一个 RPC)             ├── async_output_queue.get()
    │                                  ├── enqueue_output(output)
    │                                  │     └── worker_response_mq.enqueue(result)
```

这样主循环不必等待输出序列化/入队完成，可以立即处理下一个 RPC 请求，实现**计算与通信的流水线化**。

---

## 7. 异常处理与容错

### 7.1 Worker 异常退出

```
worker_busy_loop 中 func() 抛出异常
  └── 将异常转为字符串 → ResponseStatus.FAILURE → 通过 response_mq 回传主进程
      └── 主进程 collective_rpc 中检测到 FAILURE → 抛出 RuntimeError
```

### 7.2 父进程退出检测（death_pipe）

```
父进程退出（或调用 shutdown 关闭 death_writer）
  └── death_pipe 的读端（子进程）收到 EOF
      └── DeathPipeMonitor 线程触发：
          ├── 设置 shutdown_requested 事件
          └── 关闭 rpc_broadcast_mq + worker_response_mq
              └── worker_busy_loop 中 dequeue 抛出异常 → 循环退出
```

### 7.3 Worker 进程死亡（主进程端）

```
MultiprocWorkerMonitor 线程：
  multiprocessing.connection.wait(sentinels)  # 等待任一 Worker 进程退出
    └── 检测到 Worker 死亡：
        ├── 标记 is_failed = True
        ├── 调用 shutdown()             # 关闭所有 Worker
        └── 调用 failure_callback()     # 通知 EngineCore
```

### 7.4 优雅终止流程

```
MultiprocExecutor.shutdown()                    # 行 480
  ├── 关闭所有 death_writer → Worker 检测到 EOF 后自行清理
  ├── _ensure_worker_termination()              # 三阶段渐进终止
  │     ├── 等待 4 秒让进程自行退出
  │     ├── 发送 SIGTERM，再等 4 秒
  │     └── 发送 SIGKILL 强制终止
  ├── 关闭所有 worker_response_mq
  └── 关闭 rpc_broadcast_mq
```

---

## 8. 非阻塞 RPC（FutureWrapper）

`MultiprocExecutor.collective_rpc()` 支持 `non_block=True` 模式，此时返回 `FutureWrapper` 而非立即等待结果：

```python
class FutureWrapper(Future):                    # 行 88
    def __init__(self, futures_queue, aggregate):
        self.futures_queue = futures_queue       # 共享的 FIFO 队列

    def result(self):
        # 先排空队列中排在自己前面的所有 Future
        while not self.done():
            future, get_response = self.futures_queue.pop()
            future.wait_for_response(get_response)
        return super().result()
```

**设计目的**：在流水线并行场景下，多个 RPC 请求可以连续发出而不等待返回。当需要结果时，`FutureWrapper.result()` 按 FIFO 顺序依次等待前面的请求完成，保证结果顺序与发送顺序一致。

```
时间线：
  主进程:  send(RPC_1) → send(RPC_2) → send(RPC_3) → future_3.result()
                                                         │
                                                    先等 RPC_1 完成
                                                    再等 RPC_2 完成
                                                    最后等 RPC_3 完成
```

---

## 9. 多节点支持

当 `nnodes_within_dp > 1`（多机部署）时：

| 组件 | 单节点 | 多节点 |
|---|---|---|
| `rpc_broadcast_mq` | 直接从 Handle 创建 | 通过 `get_inner_dp_world_group().create_mq_broadcaster()` 创建跨节点广播 |
| `worker_response_mq` | `MessageQueue(1, 1)` 本地队列 | 通过 `create_single_reader_mq_broadcasters()` 创建，reader_rank=0 |
| `peer_response_handles` | 空列表 | 包含所有远程 Worker 的响应句柄 |

Leader 节点（`node_rank_within_dp == 0`）负责创建广播队列和收集所有节点的响应。

---

## 10. 关键配置与环境变量

| 配置/环境变量 | 说明 |
|---|---|
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | 消息队列单次传输最大块大小 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | execute_model RPC 超时时间 |
| `OMP_NUM_THREADS` | 多进程场景下自动设为 1 以避免 CPU 争抢 |
| `scheduler_config.async_scheduling` | 是否启用异步调度（影响输出处理方式） |
| `parallel_config.worker_cls` | Worker 类的限定名（如 `"vllm.v1.worker.gpu.worker.Worker"`） |
| `parallel_config.worker_extension_cls` | Worker 扩展类，动态注入到 Worker 继承链 |

---

## 11. 时序图：一次完整的 execute_model 调用

```
MultiprocExecutor                    WorkerProc_0              WorkerProc_N (output_rank)
      │                                  │                           │
      │── collective_rpc("execute_model", scheduler_output) ──────────────────────│
      │     │                            │                           │
      │     ├─ rpc_broadcast_mq.enqueue( │                           │
      │     │    ("execute_model",       │                           │
      │     │     (sched_out,), {},      │                           │
      │     │     output_rank) )         │                           │
      │     │                            │                           │
      │     │          ┌─── dequeue() ───┤                           │
      │     │          │                 │                           │
      │     │          │   execute_model │(sched_out)                │
      │     │          │                 │                           │
      │     │          │   rank ≠ output_rank                       │
      │     │          │   → 丢弃输出     │                           │
      │     │          │                 │                           │
      │     │          │                 │     ┌─── dequeue() ───────┤
      │     │          │                 │     │                     │
      │     │          │                 │     │  execute_model(sched_out)
      │     │          │                 │     │                     │
      │     │          │                 │     │  rank == output_rank│
      │     │          │                 │     │  → enqueue_output() │
      │     │          │                 │     │                     │
      │     ├─ response_mqs[output_rank].dequeue() ◄────────────────┤
      │     │                            │                           │
      │     └─ return (SUCCESS, ModelRunnerOutput)                   │
      │                                  │                           │
```

---

## 12. Rank 体系详解

本文件中 rank 有多个层次的含义：

### 12.1 local_rank — 本地设备索引

指当前节点（机器）内的 GPU 编号，从 0 开始。

```
节点 A:  GPU0(local_rank=0)  GPU1(local_rank=1)  GPU2(local_rank=2)  GPU3(local_rank=3)
```

用途：
- 决定子进程绑定到哪块 GPU（`CUDA_VISIBLE_DEVICES`）
- 作为 `WorkerWrapperBase` 的 `rpc_rank`，用于从 `all_kwargs[local_rank]` 中取出当前 Worker 的初始化参数

### 12.2 rank（global rank）— 全局唯一编号

跨所有节点的全局 Worker 编号，计算方式（行 192-193）：

```python
global_rank = local_world_size * node_rank_within_dp + local_rank
```

```
节点 A (node_rank=0, local_world_size=4):
  GPU0 → rank=0   GPU1 → rank=1   GPU2 → rank=2   GPU3 → rank=3

节点 B (node_rank=1, local_world_size=4):
  GPU0 → rank=4   GPU1 → rank=5   GPU2 → rank=6   GPU3 → rank=7
```

用途：
- 分布式通信中唯一标识每个 Worker（`torch.distributed` 的 rank）
- 确定 Worker 在并行组中的角色（TP rank、PP rank 等）
- `output_rank` 判断：只有 `self.rank == output_rank` 的 Worker 才回传结果

### 12.3 各种派生 rank

在 `setup_proc_title_and_log_prefix`（行 1016）中可以看到，global rank 会被映射到多个并行维度：

```
global rank → TP rank   (张量并行组内的位置)
            → PP rank   (流水线并行组内的位置)
            → DP rank   (数据并行组内的位置)
            → EP rank   (专家并行组内的位置)
            → PCP rank  (预填充上下文并行组内的位置)
```

例如 TP=4, PP=2, 共 8 个 Worker：

```
rank 0-3 → PP rank=0, TP rank=0,1,2,3
rank 4-7 → PP rank=1, TP rank=0,1,2,3
```

### 12.4 output_rank — 特殊角色

```python
output_rank = world_size - tp_size * pcp_size   # 行 531
```

指**最后一个 PP 阶段中 TP rank=0 的 Worker**。只有这个 Worker 拥有完整的推理结果（logits），所以只从它收集 `ModelRunnerOutput`，其余 Worker 的输出被丢弃。

### 12.5 driver_worker 判定

```python
def _is_driver_worker(self, rank):
    return rank % tensor_parallel_size == 0   # 行 287
```

每个 TP 组中 TP rank=0 的 Worker 是 driver，负责协调该组的操作。

### 12.6 一句话总结

**`local_rank` 是"我在这台机器上用第几块 GPU"，`rank` 是"我在整个分布式集群中是第几号 Worker"**，后者是全局唯一的身份标识，决定了该 Worker 在张量并行、流水线并行等维度中的角色。

---

## 13. 总结

WorkerProc 的设计体现了以下工程理念：

1. **进程隔离**：每个 GPU 一个独立进程，避免 GIL 争用，故障隔离
2. **零拷贝通信**：通过共享内存 MessageQueue 传递调度数据，避免序列化/反序列化开销
3. **渐进式容错**：death_pipe 检测 + 信号处理 + 三阶段终止，确保资源正确释放
4. **按需响应**：output_rank 机制减少不必要的通信，只收集有价值的输出
5. **流水线化**：FutureWrapper + 异步输出线程，实现计算与通信的重叠
6. **延迟初始化**：WorkerWrapperBase 先创建轻量壳，环境配置完成后再实例化真正的 Worker
