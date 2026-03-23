# Engine Client 与 Engine Core 通信机制

## 架构总览

vLLM v1 引擎采用**多进程架构**，Client（前端 API 服务进程）和 Core（后端推理进程）通过 **ZMQ (ZeroMQ)** 进行进程间通信，使用 **msgpack** 进行二进制序列化。

```
┌──────────────────────────────────────┐      ┌──────────────────────────────────────────┐
│         Client 进程 (API Server)      │      │         Core 进程 (EngineCore)            │
│                                      │      │                                          │
│  AsyncLLM / LLMEngine                │      │  ┌─────────────────────────────────────┐  │
│    │                                 │      │  │ Input Thread (process_input_sockets) │  │
│    ▼                                 │      │  │   DEALER socket ◄── ZMQ ──┐         │  │
│  EngineCoreClient                    │      │  │   反序列化请求             │         │  │
│    │                                 │      │  │   放入 input_queue         │         │  │
│    ├─ input_socket (ROUTER) ─── ZMQ ─┼──────┼──┼─────────────────────────────┘         │  │
│    │  发送请求                        │      │  └─────────────────────────────────────┘  │
│    │                                 │      │                                          │
│    │                                 │      │  ┌─────────────────────────────────────┐  │
│    │                                 │      │  │ Main Thread (run_busy_loop)          │  │
│    │                                 │      │  │   input_queue → _handle_client_req   │  │
│    │                                 │      │  │   step() → scheduler + executor      │  │
│    │                                 │      │  │   结果 → output_queue                │  │
│    │                                 │      │  └─────────────────────────────────────┘  │
│    │                                 │      │                                          │
│    │                                 │      │  ┌──────────────────────────────────────┐ │
│    └─ output_socket (PULL) ◄── ZMQ ──┼──────┼──┤ Output Thread (process_output_sockets)│ │
│       接收响应                        │      │  │   output_queue → 序列化              │ │
│       反序列化 → 返回给上层            │      │  │   PUSH socket ──► ZMQ               │ │
│                                      │      │  └──────────────────────────────────────┘ │
└──────────────────────────────────────┘      └──────────────────────────────────────────┘
```
### 引用的关键python模块
#### msgspec
- msgspec是一个转为python设计的高性能序列化库，支持多种数据格式（如JSON、MessagePack）和复杂数据结构的高效编码和解码。
- 利用python的类型注解系统，msgspec可以在反序列化的时候自动进行数据验证，且几乎不带来额外的性能开销。
### 关键设计

- **请求通道**：Client ROUTER socket → Core DEALER socket（支持多引擎路由）
- **响应通道**：Core PUSH socket → Client PULL socket（单向推送）
- **3 个线程**（Core 侧）：Input IO 线程、Main 业务线程、Output IO 线程，通过 `queue.Queue` 连接
- **零拷贝**：Tensor 数据通过 ZMQ 的 `copy=False` + `MessageTracker` 实现零拷贝传输

---

## Client 端实现

### Client 类型体系

文件：`vllm/v1/engine/core_client.py:67-101`

```python
class EngineCoreClient(ABC):
    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":
        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(...)  # AsyncMPClient
        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(...)                            # SyncMPClient
        return InprocClient(...)                                # InprocClient（无 IPC）
```

三种 Client：

| Client 类型 | 使用场景 | 通信方式 |
|-------------|---------|---------|
| `InprocClient` | 同进程直接调用（V0 兼容） | 直接方法调用 |
| `SyncMPClient` | 同步多进程（`LLMEngine`） | ZMQ 同步 |
| `AsyncMPClient` | 异步多进程（`AsyncLLM`） | ZMQ + asyncio |

### ZMQ Socket 初始化

文件：`vllm/v1/engine/core_client.py:570-633`

```python
# Client 创建 ROUTER socket（绑定端），等待 Core 连接
self.input_socket = make_zmq_socket(self.ctx, addresses.inputs[0], zmq.ROUTER)
self.resources.output_socket = make_zmq_socket(self.ctx, addresses.outputs[0], zmq.PULL)

# 每个 Engine 的 ZMQ identity 是 2 字节小端整数
self.core_engines: list[EngineIdentity] = [
    rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
]

# 握手：等待每个 Engine Core 发送就绪消息
identities = set(self.core_engines)
sync_input_socket = zmq.Socket.shadow(self.input_socket)
while identities:
    identity, _ = sync_input_socket.recv_multipart()
    identities.remove(identity)
```

### 发送请求

文件：`vllm/v1/engine/core_client.py:822-834`

```python
def _send_input(self, request_type: EngineCoreRequestType, request: Any):
    self.ensure_alive()
    self.free_pending_messages()
    # 多帧消息: (Engine Identity, 请求类型, 序列化数据...)
    msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

    if len(msg) <= 3:
        # 无 tensor 辅助缓冲区，直接发送
        self.input_socket.send_multipart(msg, copy=False)
        return

    # 有 tensor 数据时，需要 track 确保 ZMQ 发完前不释放内存
    tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
    self.add_pending_message(tracker, request)
```

### 接收响应

文件：`vllm/v1/engine/core_client.py:769-797`（SyncMPClient 的后台线程）

```python
def process_outputs_socket():
    poller = zmq.Poller()
    poller.register(out_socket, zmq.POLLIN)
    while True:
        socks = poller.poll()
        frames = out_socket.recv_multipart(copy=False)
        # 反序列化响应
        outputs: EngineCoreOutputs = decoder.decode(frames)
        if outputs.utility_output:
            # RPC 调用的响应，设置到对应的 Future 上
            _process_utility_output(outputs.utility_output, utility_results)
        else:
            # 推理输出，放入队列供上层消费
            outputs_queue.put_nowait(outputs)
```

### 添加推理请求

文件：`vllm/v1/engine/core_client.py:847-850`

```python
def add_request(self, request: EngineCoreRequest) -> None:
    if self.is_dp:
        self.engines_running = True
    self._send_input(EngineCoreRequestType.ADD, request)
```

### RPC 风格的 Utility 调用

文件：`vllm/v1/engine/core_client.py:836-842`

```python
def call_utility(self, method: str, *args) -> Any:
    call_id = uuid.uuid1().int >> 64          # 生成唯一调用 ID
    future: Future[Any] = Future()
    self.utility_results[call_id] = future     # 注册等待 Future
    self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))
    return future.result()                     # 阻塞等待结果
```

用法示例（同文件 856-888 行）：

```python
def profile(self, is_start=True, profile_prefix=None):
    self.call_utility("profile", is_start, profile_prefix)

def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
    return self.call_utility("reset_prefix_cache", reset_running_requests, reset_connector)

def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.call_utility("add_lora", lora_request)
```

---

## Core 端实现

### EngineCoreProc 初始化

文件：`vllm/v1/engine/core.py:776-874`

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    def __init__(self, vllm_config, local_client, handshake_address, ...):
        # 两个进程内队列，连接 IO 线程和主线程
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()

        # Engine 的 ZMQ identity
        identity = self.engine_index.to_bytes(length=2, byteorder="little")

        # 启动 Input IO 线程（DEALER socket 连接到 Client 的 ROUTER）
        input_thread = threading.Thread(
            target=self.process_input_sockets,
            args=(addresses.inputs, addresses.coordinator_input, identity, ready_event),
            daemon=True,
        )
        input_thread.start()

        # 启动 Output IO 线程（PUSH socket 推送到 Client 的 PULL）
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output, self.engine_index),
            daemon=True,
        )
        self.output_thread.start()
```

### Input IO 线程：接收请求

文件：`vllm/v1/engine/core.py:1335-1419`

```python
def process_input_sockets(self, input_addresses, coord_input_address, identity, ready_event):
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    generic_decoder = MsgpackDecoder()

    with ExitStack() as stack, zmq.Context() as ctx:
        # 创建 DEALER socket 并连接到 Client 的 ROUTER
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.DEALER, identity=identity, bind=False)
            )
            for addr in input_addresses
        ]
        # 向 Client 发送就绪消息（DEALER → ROUTER 握手）
        for input_socket in input_sockets:
            input_socket.send(b"")

        ready_event.set()
        while True:
            for input_socket, _ in poller.poll():
                # 接收多帧消息: (RequestType, Data...)
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                # 根据类型反序列化
                if request_type == EngineCoreRequestType.ADD:
                    req = add_request_decoder.decode(data_frames)
                    request = self.preprocess_add_request(req)
                else:
                    request = generic_decoder.decode(data_frames)

                # 推入 input_queue 供主线程消费
                self.input_queue.put_nowait((request_type, request))
```

### Main 线程：业务循环

文件：`vllm/v1/engine/core.py:1127-1186`

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) 从 input_queue 读取并处理请求
        self._process_input_queue()
        # 2) 执行推理并输出结果
        self._process_engine_step()

def _process_input_queue(self):
    while not self.has_work() and self.is_running():
        req = self.input_queue.get(block=True)   # 阻塞等待请求
        self._handle_client_request(*req)
    # 处理队列中剩余的请求
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)

def _process_engine_step(self):
    outputs, model_executed = self.step_fn()     # 调用 step() 执行推理
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)      # 结果放入 output_queue
```

### 请求分发处理

文件：`vllm/v1/engine/core.py:1230-1262`

```python
def _handle_client_request(self, request_type, request):
    if request_type == EngineCoreRequestType.ADD:
        req, request_wave = request
        self.add_request(req, request_wave)

    elif request_type == EngineCoreRequestType.ABORT:
        self.abort_requests(request)

    elif request_type == EngineCoreRequestType.UTILITY:
        client_idx, call_id, method_name, args = request
        output = UtilityOutput(call_id)
        # 反射调用 EngineCore 上的方法
        get_result = lambda: (method := getattr(self, method_name)) and method(
            *self._convert_msgspec_args(method, args)
        )
        enqueue_output = lambda out: self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=out))
        )
        self._invoke_utility_method(method_name, get_result, output, enqueue_output)
```

### step()：推理执行

文件：`vllm/v1/engine/core.py:378-407`

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output."""
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

### Output IO 线程：发送响应

文件：`vllm/v1/engine/core.py:1421-1489`

```python
def process_output_sockets(self, output_paths, coord_output_path, engine_index):
    encoder = MsgpackEncoder()
    reuse_buffers: list[bytearray] = []

    with ExitStack() as stack, zmq.Context() as ctx:
        # 创建 PUSH socket 绑定地址
        sockets = [
            stack.enter_context(make_zmq_socket(ctx, path, zmq.PUSH, linger=4000))
            for path in output_paths
        ]

        while True:
            output = self.output_queue.get()          # 阻塞等待主线程的输出
            if output == EngineCoreProc.ENGINE_CORE_DEAD:
                for socket in sockets:
                    socket.send(output)                # 发送死亡信号
                break

            client_index, outputs = output
            outputs.engine_index = engine_index

            # 序列化并零拷贝发送
            buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
            buffers = encoder.encode_into(outputs, buffer)
            tracker = sockets[client_index].send_multipart(
                buffers, copy=False, track=True
            )
            if not tracker.done:
                pending.appendleft((tracker, ref, buffer))  # 追踪直到发送完成
```

---

## 消息类型与数据结构

### 请求类型枚举

文件：`vllm/v1/engine/__init__.py:217-230`

```python
class EngineCoreRequestType(enum.Enum):
    ADD = b"\x00"              # 添加推理请求
    ABORT = b"\x01"            # 中止请求
    START_DP_WAVE = b"\x02"    # DP wave 同步
    UTILITY = b"\x03"          # RPC 方法调用
    EXECUTOR_FAILED = b"\x04"  # 执行器失败信号
    WAKEUP = b"\x05"           # 唤醒信号（用于关闭）
```

### 请求结构体

文件：`vllm/v1/engine/__init__.py:66-111`

```python
class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None
    client_index: int = 0
    priority: int = 0
    ...
```

### 响应结构体

文件：`vllm/v1/engine/__init__.py:140-214`

```python
class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    num_cached_tokens: int = 0
    ...

class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None  # RPC 调用的返回值
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    ...

class UtilityOutput(msgspec.Struct, array_like=True):
    call_id: int
    failure_message: str | None = None
    result: UtilityResult | None = None
```

---

## 序列化机制

文件：`vllm/v1/serial_utils.py`

### MsgpackEncoder

```python
class MsgpackEncoder:
    """支持 torch.Tensor 和 numpy array 的 msgpack 编码器"""

    def encode(self, obj: Any) -> Sequence[bytestr]:
        self.aux_buffers = bufs = [b""]
        bufs[0] = self.encoder.encode(obj)
        # bufs 列表包含：[msgpack主数据, tensor缓冲区1, tensor缓冲区2, ...]
        # tensor 数据通过零拷贝方式直接引用原始内存
        return bufs
```

### MsgpackDecoder

```python
class MsgpackDecoder:
    """支持 torch.Tensor 和 numpy array 的 msgpack 解码器"""

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        self.aux_buffers = bufs
        return self.decoder.decode(bufs[0])  # bufs[0] 是 msgpack 主数据
```

关键特性：
- Tensor 数据**不**被复制到 msgpack 缓冲区中，而是作为 ZMQ 多帧消息的独立帧发送
- 小于 `VLLM_MSGPACK_ZERO_COPY_THRESHOLD`（默认 256B）的 tensor 内联序列化
- 大 tensor 通过 `zmq.send_multipart(copy=False)` 零拷贝传输

---

## ZMQ 地址管理

文件：`vllm/v1/engine/utils.py:54-67`

```python
@dataclass
class EngineZmqAddresses:
    inputs: list[str]               # ROUTER socket 地址（Client → Core 请求）
    outputs: list[str]              # PUSH socket 地址（Core → Client 响应）
    coordinator_input: str | None   # DP 协调器输入地址
    coordinator_output: str | None  # DP 协调器输出地址
    frontend_stats_publish_address: str | None  # 统计信息发布地址
```

---

## ZMQ 消息帧格式

### 请求消息（Client → Core）

通过 ROUTER/DEALER 模式发送，Client 端发送：

```
帧 0: Engine Identity (2 bytes, 小端整数)   ← ROUTER 用来路由到正确的 Engine
帧 1: RequestType (1 byte, 如 b"\x00")     ← 请求类型标识
帧 2: msgpack 序列化主数据                   ← 请求体
帧 3+: tensor 辅助缓冲区（可选）             ← 零拷贝 tensor 数据
```

Core 端 DEALER socket 收到时，ROUTER 自动剥离了 Identity 帧：

```
帧 0: RequestType
帧 1: msgpack 主数据
帧 2+: tensor 辅助缓冲区（可选）
```

### 响应消息（Core → Client）

通过 PUSH/PULL 模式发送：

```
帧 0: msgpack 序列化的 EngineCoreOutputs
帧 1+: tensor 辅助缓冲区（可选）
```

---

## 完整请求生命周期

以一个推理请求为例：

```
1. [Client] AsyncLLM.add_request()
       ↓
2. [Client] EngineCoreClient.add_request(EngineCoreRequest)
       ↓
3. [Client] _send_input(ADD, request)
       ↓  MsgpackEncoder.encode() → 多帧 ZMQ 消息
4. [Client] input_socket.send_multipart([identity, b"\x00", data...], copy=False)
       ↓  ═══ ZMQ IPC ═══
5. [Core InputThread] input_socket.recv_multipart()
       ↓  MsgpackDecoder.decode() → EngineCoreRequest
6. [Core InputThread] preprocess_add_request(req)
       ↓  input_queue.put_nowait()
7. [Core MainThread]  input_queue.get() → _handle_client_request(ADD, request)
       ↓
8. [Core MainThread]  scheduler.schedule() → model_executor.execute_model()
       ↓  GPU 推理执行
9. [Core MainThread]  scheduler.update_from_output() → EngineCoreOutputs
       ↓  output_queue.put_nowait()
10.[Core OutputThread] output_queue.get()
       ↓  MsgpackEncoder.encode_into() → 多帧 ZMQ 消息
11.[Core OutputThread] socket.send_multipart(buffers, copy=False)
       ↓  ═══ ZMQ IPC ═══
12.[Client OutputThread] output_socket.recv_multipart()
       ↓  MsgpackDecoder.decode() → EngineCoreOutputs
13.[Client] outputs_queue.put_nowait(outputs) → 返回给 API 层
```

---

## 关键文件索引

| 文件 | 职责 |
|------|------|
| `vllm/v1/engine/core_client.py` | Client 实现（SyncMPClient, AsyncMPClient） |
| `vllm/v1/engine/core.py` | Core 实现（EngineCore, EngineCoreProc） |
| `vllm/v1/engine/__init__.py` | 消息类型定义（Request, Output, RequestType） |
| `vllm/v1/engine/utils.py` | ZMQ 地址管理、握手元数据 |
| `vllm/v1/serial_utils.py` | MsgpackEncoder / MsgpackDecoder |
| `vllm/v1/engine/async_llm.py` | AsyncLLM（异步 API 入口） |
| `vllm/v1/engine/llm_engine.py` | LLMEngine（同步 API 入口） |
