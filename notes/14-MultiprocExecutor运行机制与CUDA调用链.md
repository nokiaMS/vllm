# MultiprocExecutor 运行机制与 CUDA 调用链

## 一、总体定位

`MultiprocExecutor` 是 vLLM v1 引擎在**单机多卡**场景下的默认执行器后端。它通过 Python `multiprocessing` 创建多个 GPU Worker 子进程，使用共享内存消息队列（MessageQueue）进行零拷贝 IPC 通信，将调度器的推理请求分发到各 Worker 上执行。

```
EngineCore
  └─> MultiprocExecutor  (主进程，负责调度与分发)
        ├─> WorkerProc 0  (子进程，持有 GPU 0)
        │     └─> Worker → GPUModelRunner → Model → CUDA Kernels
        ├─> WorkerProc 1  (子进程，持有 GPU 1)
        │     └─> Worker → GPUModelRunner → Model → CUDA Kernels
        └─> ...
```

---

## 二、类层次结构

```
Executor (抽象基类, abstract.py)
  ├─ MultiprocExecutor   (多进程实现, multiproc_executor.py)
  ├─ RayDistributedExecutor  (Ray 分布式实现)
  └─ UniProcExecutor     (单进程实现)

WorkerProc (多进程 Worker 容器, multiproc_executor.py)
  └─ 持有 WorkerWrapperBase
       └─ 持有 Worker (GPU Worker, gpu_worker.py)
            └─ 持有 GPUModelRunner (gpu/model_runner.py)
                 └─ 持有 model (nn.Module, 如 LlamaForCausalLM)
```

---

## 三、执行器选择机制

**入口**: `EngineCore.__init__()` → `Executor.get_class(vllm_config)`

`vllm/v1/executor/abstract.py` 第 66-106 行的工厂方法根据配置选择执行器：

| `distributed_executor_backend` | 执行器类 | 适用场景 |
|---|---|---|
| `"mp"` | `MultiprocExecutor` | 单机多卡（默认） |
| `"ray"` | `RayDistributedExecutor` | 多机分布式 |
| `"uni"` | `UniProcExecutor` | 单进程单卡 |
| `"external_launcher"` | `UniProcExecutor` | 外部启动器 |

---

## 四、MultiprocExecutor 初始化流程

### 4.1 `_init_executor()` 主流程

```
MultiprocExecutor._init_executor()
  │
  ├─ ① set_multiprocessing_worker_envs()
  │     强制 spawn 方式创建子进程，限制 OMP_NUM_THREADS=1 避免 CPU 争抢
  │
  ├─ ② 获取分布式初始化 URL
  │     基于回环地址 + 可用端口，用于 torch.distributed 初始化
  │
  ├─ ③ 创建 RPC 广播消息队列 (仅 DP 组 leader 节点)
  │     rpc_broadcast_mq = MessageQueue(world_size, local_world_size, ...)
  │     使用共享内存环形缓冲 + ZMQ PUB/SUB 实现零拷贝广播
  │
  ├─ ④ 创建所有 Worker 子进程
  │     for local_rank in range(local_world_size):
  │       WorkerProc.make_worker_process(...)
  │       └─ 创建 multiprocessing.Process(daemon=True) 并 start()
  │       └─ 返回 UnreadyWorkerProcHandle (含就绪管道 + 死亡检测管道)
  │
  ├─ ⑤ 等待所有 Worker 就绪
  │     WorkerProc.wait_for_ready(unready_workers)
  │     └─ 通过管道接收 READY 信号和响应队列句柄
  │     └─ 返回 WorkerProcHandle 列表
  │
  ├─ ⑥ 启动 Worker 健康监控线程（可选）
  │     monitor_workers(): 监听所有 Worker sentinel
  │     任何 Worker 异常退出 → 触发 failure_callback
  │
  └─ ⑦ 等待所有消息队列就绪
        rpc_broadcast_mq.wait_until_ready()
        response_mqs[*].wait_until_ready()
```

### 4.2 Worker 子进程初始化

每个 Worker 子进程的入口是 `WorkerProc.worker_main()` 静态方法：

```
WorkerProc.worker_main()
  │
  ├─ 设置信号处理 (SIGTERM/SIGINT → SystemExit)
  │
  ├─ 创建 WorkerProc 实例
  │     WorkerProc.__init__()
  │       ├─ 创建 WorkerWrapperBase → 内部创建 Worker 实例
  │       ├─ worker.init_device()   ← 初始化 GPU 设备、torch.distributed
  │       ├─ worker.load_model()    ← 加载模型权重到 GPU
  │       └─ _init_message_queues() ← 从句柄恢复共享内存消息队列
  │
  ├─ 启动死亡管道监控线程
  │     monitor_death_pipe(): 父进程退出 → 关闭消息队列 → 优雅终止
  │
  ├─ 发送 READY 信号给主进程
  │     ready_writer.send({status, handle, peer_response_handles})
  │
  ├─ 等待消息队列就绪
  │
  └─ 进入主循环 worker_busy_loop()
```

---

## 五、消息队列通信机制 (MessageQueue)

**实现文件**: `vllm/distributed/device_communicators/shm_broadcast.py`

### 5.1 架构

```
主进程 (MultiprocExecutor)
  ├─ rpc_broadcast_mq ──广播──▶ 所有 Worker (1 → N)
  └─ response_mqs[i]  ◀──响应── 各 Worker   (N → 1)

Worker 子进程
  ├─ 从 rpc_broadcast_mq.dequeue() 取出 RPC 请求
  └─ 向 worker_response_mq.enqueue() 写入执行结果
```

### 5.2 核心组件

| 组件 | 作用 |
|---|---|
| **ShmRingBuffer** | 共享内存环形缓冲，多读者/单写者，零拷贝 |
| **SpinCondition** | 自适应忙轮询：最近活跃时自旋，空闲时通过 ZMQ 等待 |
| **Handle** | 序列化句柄，通过管道传递给子进程以恢复消息队列 |

### 5.3 消息格式

RPC 广播消息的结构：

```python
(method, args, kwargs, output_rank)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `method` | `str` 或 `bytes` | 方法名（反射调用）或 cloudpickle 序列化的 callable |
| `args` | `tuple` | 位置参数 |
| `kwargs` | `dict` | 关键字参数 |
| `output_rank` | `int \| None` | 指定哪个 rank 返回结果，None 表示所有 rank 都返回 |

---

## 六、RPC 调度与执行流程

### 6.1 主进程端: `collective_rpc()`

这是 MultiprocExecutor 的核心通信原语：

```python
def collective_rpc(self, method, args, kwargs, non_block, unique_reply_rank, ...):
    # ① 序列化并广播 RPC 请求
    rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))

    # ② 收集响应
    def get_response():
        for mq in target_response_mqs:
            status, result = mq.dequeue(timeout=...)
            if status != SUCCESS:
                raise RuntimeError(result)
        return responses

    # ③ 非阻塞模式：返回 FutureWrapper
    if non_block:
        return FutureWrapper(futures_queue, aggregate)

    # ④ 同步模式：排空队列 + 等待结果
    drain_pending_futures()
    return aggregate(get_response())
```

### 6.2 Worker 端: `worker_busy_loop()`

```python
def worker_busy_loop(self):
    while True:
        # ① 阻塞等待 RPC 请求
        method, args, kwargs, output_rank = rpc_broadcast_mq.dequeue(indefinite=True)

        try:
            # ② 解析并调用方法
            if isinstance(method, str):
                func = getattr(self.worker, method)        # 字符串反射
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)  # 反序列化

            output = func(*args, **kwargs)  # 执行
        except Exception as e:
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)       # 异常返回给主进程
            continue

        # ③ 仅 output_rank 匹配的 Worker 返回结果
        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)
```

### 6.3 FutureWrapper 与流水线调度

`FutureWrapper` 支持多批次流水线式调度，按 FIFO 顺序消费结果：

```
时间线:
  t0: dispatch Batch A → FutureA 入队
  t1: dispatch Batch B → FutureB 入队
  t2: FutureA.result() → 先排空队列前面的 Future → 返回 Batch A 结果
  t3: FutureB.result() → 返回 Batch B 结果
```

---

## 七、execute_model 完整调用链

### 7.1 主进程发起

```python
# multiproc_executor.py
def execute_model(self, scheduler_output, non_block=False):
    return self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,  # 仅从 PP 最后阶段收集
        non_block=non_block,
    )
```

**`output_rank` 优化**：只从流水线最后阶段的首个 TP Worker 收集输出，减少通信开销。

### 7.2 Worker 层: `Worker.execute_model()`

```
Worker.execute_model(scheduler_output)
  │
  ├─ 等待前一次 PP 异步发送完成
  │     for handle in self._pp_send_work: handle.wait()
  │
  ├─ PP 非首 rank: 接收前序 rank 的中间张量
  │     intermediate_tensors = get_pp_group().irecv_tensor_dict(...)
  │     └─ AsyncIntermediateTensors: 支持延迟同步，计算/通信重叠
  │
  ├─ 调用 ModelRunner 执行推理
  │     output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
  │
  └─ PP 非末 rank: 异步发送中间张量给后序 rank
        self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
        return None  # 非末 rank 不返回最终结果
```

### 7.3 GPUModelRunner 层: `execute_model()`

这是推理执行的核心，位于 `vllm/v1/worker/gpu/model_runner.py`：

```
GPUModelRunner.execute_model(scheduler_output, intermediate_tensors)
  │
  ├─ ① 更新请求状态
  │     finish_requests()   — 移除已完成请求
  │     free_states()       — 释放编码器缓存
  │     add_requests()      — 添加新请求
  │     update_requests()   — 追加新 KV 缓存块
  │
  ├─ ② 获取批次描述符
  │     batch_desc = cudagraph_manager.dispatch(num_reqs, num_toks, ...)
  │     决定使用 CUDA 图模式 (FULL/PIECEWISE/NONE)
  │
  ├─ ③ 准备模型输入
  │     input_batch = prepare_inputs(scheduler_output, batch_desc)
  │     block_tables, slot_mappings = prepare_attn(input_batch)
  │     构建: input_ids, positions, block_table, slot_mapping, logits_indices
  │
  ├─ ④ 准备注意力元数据
  │     attn_metadata = model_state.prepare_attn(input_batch, cg_mode, ...)
  │
  ├─ ⑤ 准备多模态嵌入 (如有图片/音频输入)
  │     inputs_embeds = model_state.get_mm_embeddings(...)
  │
  ├─ ⑥ 执行模型前向推理 ← 这里进入 CUDA 执行！
  │     if cg_mode == CUDAGraphMode.FULL:
  │       model_output = cudagraph_manager.run_fullgraph(batch_desc)  # CUDA 图回放
  │     else:
  │       with set_forward_context(attn_metadata, slot_mapping, ...):
  │         model_output = self.model(**model_inputs)  # PyTorch 前向传播
  │
  └─ ⑦ 返回
        PP 非末 rank → 返回 IntermediateTensors
        PP 末 rank   → 保存状态供 sample_tokens 使用
```

---

## 八、CUDA 调用链：从 Python 到 GPU Kernel

### 8.1 模型前向传播

`self.model(**model_inputs)` 调用 PyTorch 模型的 `forward()` 方法。以 LLaMA 为例：

```
LlamaForCausalLM.forward()
  └─ LlamaModel.forward()
       ├─ embed_tokens(input_ids)          ← CUDA: Embedding lookup
       │
       └─ for layer in layers:             ← 逐层执行 Transformer 块
            LlamaDecoderLayer.forward()
              ├─ input_layernorm(x)         ← CUDA: RMSNorm kernel
              ├─ self_attn.forward(x)       ← CUDA: Attention (详见 8.2)
              │   ├─ qkv_proj(x)            ← CUDA: Linear (矩阵乘法)
              │   ├─ rotary_emb(q, k)       ← CUDA: RoPE kernel
              │   ├─ attn.forward(q, k, v)  ← CUDA: FlashAttention kernel
              │   └─ o_proj(attn_out)       ← CUDA: Linear
              ├─ post_attention_layernorm(x) ← CUDA: RMSNorm kernel
              └─ mlp.forward(x)             ← CUDA: FFN
                  ├─ gate_up_proj(x)        ← CUDA: Linear
                  ├─ act_fn(gate)           ← CUDA: SiLU activation
                  └─ down_proj(x)           ← CUDA: Linear
```

### 8.2 注意力层的 CUDA 调用

注意力层是最关键的 CUDA 密集操作，位于 `vllm/v1/attention/backends/flash_attn.py`：

```
FlashAttentionImpl.forward(query, key, value, kv_cache, attn_metadata)
  │
  ├─ ① 写入 KV 缓存 (CUDA 自定义 kernel)
  │     reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)
  │     └─ torch.ops._C.reshape_and_cache_flash(...)
  │        └─ csrc/cache_kernels.cu  ← 实际 CUDA kernel
  │        功能: 将当前 token 的 K/V 写入分页式缓存的对应物理块
  │
  └─ ② 计算注意力 (FlashAttention CUDA kernel)
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=...,        # Query 序列长度前缀和
            max_seqlen_q=...,
            seqused_k=...,           # KV 序列长度
            block_table=block_table, # 逻辑块 → 物理块映射
            softmax_scale=self.scale,
            causal=True,             # 因果掩码
            fa_version=2/3/4,        # FlashAttention 版本
        )
        └─ vllm/vllm_flash_attn/flash_attn_interface.py
           └─ 调用 FlashAttention CUDA kernel
              功能: 高效计算 softmax(QK^T/√d)V，支持分页 KV 缓存
```

### 8.3 自定义 CUDA 算子注册

vLLM 通过 `torch.ops._C` 注册自定义 CUDA 算子：

```python
# vllm/_custom_ops.py — Python 绑定层
def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...):
    torch.ops._C.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)

def paged_attention_v1(out, query, key_cache, value_cache, ...):
    torch.ops._C.paged_attention_v1(out, query, key_cache, value_cache, ...)
```

```
注册路径:
  csrc/torch_bindings.cpp       ← torch::Library 注册 C++ 函数
    └─ csrc/cache_kernels.cu    ← CUDA kernel 实现
    └─ csrc/attention/*.cu      ← Attention CUDA kernel 实现
```

### 8.4 主要 CUDA Kernel 分类

| 类别 | 功能 | 实现位置 |
|---|---|---|
| **Paged Attention** | 分页式注意力计算 | `csrc/attention/paged_attention_v1.cu`, `v2.cu` |
| **FlashAttention** | 高效变长注意力 | `vllm/vllm_flash_attn/` |
| **KV Cache** | 缓存写入/拷贝/重塑 | `csrc/cache_kernels.cu` |
| **Activation** | GELU, SiLU, SwiGLU | `csrc/activation_kernels.cu` |
| **LayerNorm** | RMSNorm, LayerNorm | `csrc/layernorm_kernels.cu` |
| **Quantization** | FP8/INT8 量化反量化 | `csrc/quantization/` |
| **Merge States** | 合并分片注意力结果 | `csrc/attention/merge_attn_states.cu` |
| **Rotary Embedding** | RoPE 位置编码 | Triton kernel 或 CUDA kernel |
| **All-Reduce** | 张量并行通信 | NCCL (PyTorch 原生) |

---

## 九、CUDA 图优化

当批次大小匹配预录制的 CUDA 图时，走 **CUDA Graph 回放** 路径：

```
常规执行 (Eager Mode):
  Python → PyTorch → CUDA kernel launch × N → GPU 执行
  每次都有 Python/CPU 开销

CUDA 图回放 (Graph Replay):
  Python → cudagraph_manager.run_fullgraph() → 一次性回放预录制的所有 kernel
  消除了逐 kernel 的 Python/CPU launch 开销
```

关键代码：

```python
# gpu/model_runner.py
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
else:
    with set_forward_context(attn_metadata, ...):
        model_output = self.model(**model_inputs)  # Eager 模式
```

---

## 十、采样流程 (sample_tokens)

### 10.1 主进程发起

```python
def sample_tokens(self, grammar_output, non_block=False):
    return self.collective_rpc("sample_tokens", args=(grammar_output,),
                               unique_reply_rank=self.output_rank, ...)
```

### 10.2 GPU 端采样

```
GPUModelRunner.sample_tokens(grammar_output)
  │
  ├─ 获取 execute_model 保存的状态 (hidden_states, logits_indices, ...)
  │
  ├─ PP 非末 rank: 通过 pp_broadcast 接收采样结果
  │
  └─ PP 末 rank:
       ├─ compute_logits(hidden_states)     ← CUDA: 线性层投影到词表
       ├─ apply_logits_processors(logits)   ← CUDA: 温度/top-k/top-p 处理
       ├─ sampler.sample(logits)            ← CUDA: 采样 kernel
       ├─ 广播采样结果给其他 PP rank
       └─ 返回 ModelRunnerOutput
            包含: sampled_token_ids, logprob_token_ids, ...
```

---

## 十一、CUDA 流与内存管理

### 11.1 CUDA 流

```python
# GPUModelRunner 初始化时创建
self.main_stream = torch.cuda.current_stream()     # 主计算流
self.output_copy_stream = torch.cuda.Stream(device) # 异步输出拷贝流
self.output_copy_event = torch.cuda.Event()         # 同步事件
```

**双流流水线**：模型在 `main_stream` 上计算的同时，`output_copy_stream` 异步地将上一批结果从 GPU 拷贝到 CPU。

### 11.2 KV 缓存内存

```
分页式 KV 缓存:
  ├─ 物理布局: [2, num_blocks, block_size, num_kv_heads, head_size]
  │            2 = K 和 V 两个缓存
  ├─ BlockTable: 逻辑块序号 → 物理块 ID 的映射表
  ├─ SlotMapping: token 位置 → 物理 GPU 内存偏移
  └─ 由 Scheduler 动态分配/回收块，Worker 通过 slot_mapping 写入
```

---

## 十二、流水线并行 (PP) 执行

```
Stage 0 (Rank 0)        Stage 1 (Rank 1)        Stage 2 (Rank 2, 末)
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Layers 0-10  │        │ Layers 11-21 │        │ Layers 22-31 │
│              │        │              │        │ + LM Head    │
│ execute_model│        │ execute_model│        │ execute_model│
│   ↓          │        │   ↓          │        │   ↓          │
│ 产生中间张量 │─isend─▶│ irecv+计算   │─isend─▶│ irecv+计算   │
│ return None  │        │ return None  │        │ return output│
└──────────────┘        └──────────────┘        └──────────────┘
                                                       │
                              只有 output_rank 返回结果 ◀─┘
```

**AsyncIntermediateTensors**: 通信与计算重叠——`irecv` 返回惰性张量，在首次使用时才同步等待接收完成。

---

## 十三、优雅关闭流程

### 13.1 主进程关闭

```
MultiprocExecutor.shutdown()
  ├─ 关闭 death_writer → EOF 信号通知所有 Worker
  ├─ _ensure_worker_termination()
  │     等待 4s 自行退出 → SIGTERM → 等待 4s → SIGKILL
  └─ 关闭所有消息队列
```

### 13.2 Worker 关闭

```
WorkerProc.shutdown()
  ├─ rpc_broadcast_mq.shutdown()    — Reader 停止等待
  ├─ worker_response_mq.shutdown()  — 同上
  ├─ self.worker.shutdown()          — Worker 清理
  ├─ destroy_model_parallel()       — 销毁分布式组
  └─ destroy_distributed_environment()
```

### 13.3 死亡管道机制

```python
def monitor_death_pipe(death_pipe, shutdown_requested):
    """后台线程，阻塞读取死亡管道"""
    try:
        death_pipe.recv()  # 正常时阻塞
    except EOFError:       # 父进程退出 → 管道 EOF
        shutdown_requested.set()
        for mq in queues: mq.shutdown()  # 触发优雅终止
```

---

## 十四、完整执行流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        主进程 (MultiprocExecutor)                    │
│                                                                     │
│  EngineCore.step()                                                  │
│    └─ scheduler.schedule()  → scheduler_output                      │
│    └─ executor.execute_model(scheduler_output)                      │
│         └─ collective_rpc("execute_model", ...)                     │
│              └─ rpc_broadcast_mq.enqueue(...)  ─────────────┐       │
│                                                             │       │
│         └─ response_mq.dequeue()  ◀─────────────────┐      │       │
│              └─ 返回 ModelRunnerOutput               │      │       │
└──────────────────────────────────────────────────────┼──────┼───────┘
                                                       │      │
                          共享内存 MessageQueue          │      │
                                                       │      │
┌──────────────────────────────────────────────────────┼──────┼───────┐
│                    子进程 (WorkerProc, 每 GPU 一个)    │      │       │
│                                                       │      │       │
│  worker_busy_loop()                                   │      │       │
│    └─ rpc_broadcast_mq.dequeue()  ◀──────────────────┼──────┘       │
│    └─ Worker.execute_model(scheduler_output)          │              │
│         └─ model_runner.execute_model(...)             │              │
│              └─ prepare_inputs()                       │              │
│              └─ prepare_attn()     — 构建块表/槽映射   │              │
│              └─ model(**inputs)     — PyTorch 前向     │              │
│                   │                                    │              │
│                   ▼                                    │              │
│              ┌─────────────────────────┐              │              │
│              │     GPU (CUDA)          │              │              │
│              │ ┌─────────────────────┐ │              │              │
│              │ │ Embedding           │ │              │              │
│              │ │ RMSNorm             │ │              │              │
│              │ │ QKV Projection      │ │              │              │
│              │ │ RoPE                │ │              │              │
│              │ │ reshape_and_cache ──┼─┼── KV Cache   │              │
│              │ │ FlashAttention      │ │              │              │
│              │ │ Output Projection   │ │              │              │
│              │ │ FFN (Gate+Up+Down)  │ │              │              │
│              │ │ Logits + Sampling   │ │              │              │
│              │ └─────────────────────┘ │              │              │
│              └─────────────────────────┘              │              │
│                                                       │              │
│    └─ handle_output(output)                           │              │
│         └─ worker_response_mq.enqueue(result) ────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 十五、关键设计总结

| 设计 | 机制 | 收益 |
|---|---|---|
| **共享内存 MessageQueue** | ShmRingBuffer + ZMQ 信号 | 零拷贝 IPC，低延迟广播 |
| **output_rank 优化** | 仅 PP 末阶段首个 TP Worker 返回 | 减少 N-1 个 Worker 的响应通信 |
| **FutureWrapper FIFO 排空** | 队列化 Future，按序消费 | 支持流水线批次调度 |
| **CUDA 图回放** | 预录制 kernel 序列，一次性回放 | 消除逐 kernel 的 CPU launch 开销 |
| **双 CUDA 流** | main_stream 计算 + copy_stream 拷贝 | 计算与 Host 传输重叠 |
| **AsyncIntermediateTensors** | 惰性同步的 PP 通信 | 通信与计算重叠 |
| **死亡管道** | 管道 EOF 检测父进程退出 | 子进程孤儿防护 |
| **spawn 强制** | 避免 fork 后 CUDA 状态不一致 | 多进程安全 |
| **分页式 KV 缓存** | BlockTable 逻辑→物理映射 | 内存碎片最小化，动态分配 |
