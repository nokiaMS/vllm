# EngineCore 创建的函数调用过程

EngineCore 有两条创建路径：**InprocClient（同进程）** 和 **MPClient（多进程）**。

---

## 一、路径选择逻辑

入口在 `EngineCoreClient.make_client()`（`vllm/v1/engine/core_client.py:88`）：

```
multiprocess_mode=True  + asyncio_mode=True  → AsyncMPClient  (在线服务)
multiprocess_mode=True  + asyncio_mode=False → SyncMPClient   (离线同步推理)
multiprocess_mode=False + asyncio_mode=False → InprocClient    (V0风格同步推理)
```

---

## 二、InprocClient 路径（同进程直接创建）

```
LLMEngine.__init__()                           # vllm/v1/engine/llm_engine.py:58
  │
  └─ EngineCoreClient.make_client(             # vllm/v1/engine/core_client.py:88
  │    multiprocess_mode=False,
  │    asyncio_mode=False,
  │    vllm_config, executor_class, log_stats
  │  )
  │
  └─ InprocClient(vllm_config, executor_class, log_stats)   # core_client.py:291
       │
       └─ InprocClient.__init__()              # core_client.py:301
            │
            └─ EngineCore(vllm_config, executor_class, log_stats)  # core_client.py:302
                 │
                 └─ EngineCore.__init__()       # vllm/v1/engine/core.py:88
                      ├─ load_general_plugins()                        # :97
                      ├─ executor_class(vllm_config)   → 创建模型执行器  # :112
                      ├─ _initialize_kv_caches(vllm_config)            # :122
                      │    ├─ model_executor.get_kv_cache_specs()
                      │    ├─ model_executor.determine_available_memory()
                      │    ├─ get_kv_cache_configs()
                      │    └─ model_executor.initialize_from_config()
                      ├─ StructuredOutputManager(vllm_config)          # :123
                      ├─ Scheduler(vllm_config, kv_cache_config, ...)  # :141-148
                      └─ mm_registry.engine_receiver_cache_from_config()  # :154-156
```

**特点**：EngineCore 在当前进程内直接创建，无 IPC，无 busy loop。

---

## 三、MPClient 路径（多进程，子进程中创建）

### 3.1 AsyncMPClient 路径（在线服务，最常用）

```
AsyncLLM.__init__()                            # vllm/v1/engine/async_llm.py
  │
  └─ EngineCoreClient.make_client(             # core_client.py:88
  │    multiprocess_mode=True,
  │    asyncio_mode=True,
  │    vllm_config, executor_class, log_stats
  │  )
  │
  └─ EngineCoreClient.make_async_mp_client()   # core_client.py:122
  │    │  根据数据并行配置选择子类:
  │    │  dp_size > 1 + external_lb  → DPAsyncMPClient
  │    │  dp_size > 1 + internal_lb  → DPLBAsyncMPClient
  │    │  dp_size == 1               → AsyncMPClient
  │    │
  │    └─ AsyncMPClient(vllm_config, executor_class, log_stats)
  │
  └─ AsyncMPClient.__init__()                  # core_client.py:975
       │
       └─ MPClient.__init__(asyncio_mode=True) # core_client.py:550
            │
            ├─ 创建 ZMQ context + encoder/decoder
            ├─ get_engine_zmq_addresses()       # vllm/v1/engine/utils.py:1024
            ├─ 创建 input/output ZMQ sockets
            │
            └─ launch_core_engines()            # utils.py:1081 (context manager)
                 │
                 └─ CoreEngineProcManager()     # utils.py:162
                      │
                      │  对每个本地引擎:
                      ├─ context.Process(
                      │    target=EngineCoreProc.run_engine_core,
                      │    kwargs={vllm_config, executor_class, log_stats,
                      │            dp_rank, local_dp_rank, ...}
                      │  )
                      └─ proc.start()           # 启动子进程
```

### 3.2 子进程中的创建过程

```
═══════════════════════ 子进程边界 ═══════════════════════

EngineCoreProc.run_engine_core()               # core.py:1004 (子进程入口)
  │
  ├─ set_process_title("EngineCore" 或 "EngineCore_DP{rank}")
  ├─ maybe_init_worker_tracer()
  ├─ 更新 vllm_config 的 DP rank 信息
  │
  ├─ 选择创建哪种 EngineCoreProc:
  │    data_parallel + MOE模型  → DPEngineCoreProc(...)
  │    否则                     → EngineCoreProc(...)
  │
  └─ EngineCoreProc.__init__()                 # core.py:769
       │
       ├─ 创建 input/output queues
       ├─ _perform_handshakes()     → 与客户端进程交换 ZMQ 地址
       │
       ├─ EngineCore.__init__()     → 【同 InprocClient 路径中的初始化】
       │    ├─ load_general_plugins()
       │    ├─ executor_class(vllm_config)
       │    ├─ _initialize_kv_caches()
       │    ├─ StructuredOutputManager()
       │    ├─ Scheduler()
       │    └─ mm_registry.engine_receiver_cache_from_config()
       │
       ├─ 启动 I/O 后台线程:
       │    ├─ process_input_sockets()   (接收客户端请求)
       │    └─ process_output_sockets()  (发送推理结果)
       │
       └─ run_busy_loop()               → 进入主事件循环
```

### 3.3 SyncMPClient 路径（离线同步推理）

与 AsyncMPClient 类似，区别在于：
- 由 `LLMEngine.__init__()` 以 `multiprocess_mode=True, asyncio_mode=False` 调用
- `MPClient.__init__()` 中 `asyncio_mode=False`，使用同步 ZMQ context
- 启动后台线程（而非 asyncio task）接收 EngineCore 输出，通过 `queue.Queue` 传递给主线程

---

## 四、关键文件索引

| 组件 | 文件 | 行号 |
|------|------|------|
| LLMEngine (同步入口) | `vllm/v1/engine/llm_engine.py` | 58-127 |
| AsyncLLM (异步入口) | `vllm/v1/engine/async_llm.py` | - |
| EngineCoreClient.make_client() | `vllm/v1/engine/core_client.py` | 88-114 |
| make_async_mp_client() | `vllm/v1/engine/core_client.py` | 122-145 |
| InprocClient | `vllm/v1/engine/core_client.py` | 291-382 |
| MPClient | `vllm/v1/engine/core_client.py` | 537-686 |
| AsyncMPClient | `vllm/v1/engine/core_client.py` | 971-1003 |
| SyncMPClient | `vllm/v1/engine/core_client.py` | 782-910 |
| EngineCore | `vllm/v1/engine/core.py` | 85-282 |
| EngineCoreProc | `vllm/v1/engine/core.py` | 762-864 |
| run_engine_core() | `vllm/v1/engine/core.py` | 1004-1083 |
| CoreEngineProcManager | `vllm/v1/engine/utils.py` | 150-279 |
| launch_core_engines() | `vllm/v1/engine/utils.py` | 1081-1237 |
| get_engine_zmq_addresses() | `vllm/v1/engine/utils.py` | 1024-1071 |

---

## 五、两条路径对比

| 特性 | InprocClient | MPClient (Sync/Async) |
|------|-------------|----------------------|
| EngineCore 运行位置 | 当前进程 | 后台子进程 |
| 创建方式 | 直接 `EngineCore(...)` | `EngineCoreProc(...)` 继承自 EngineCore |
| IPC 通信 | 无，直接方法调用 | ZMQ (ROUTER/DEALER + PUSH/PULL) |
| 事件循环 | 无，由调用者驱动 step() | `run_busy_loop()` 持续运行 |
| 适用场景 | V0 风格同步推理 | 在线服务 / 离线批量推理 |
| 数据并行支持 | 不支持 | 支持（DP/DPLB 子类） |
