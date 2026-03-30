# BackgroundResources 的 input_socket 写入机制

## 写入路径总览

`input_socket` 是一个 **ZMQ ROUTER** 类型的 socket，由 `MPClient.__init__` 创建并绑定（`bind=True`），存储到 `self.resources.input_socket`。

数据通过 `input_socket.send_multipart()` 写入，有两个入口：

### 1. 同步客户端 (`SyncMPClient._send_input`)

`vllm/v1/engine/core_client.py:874`

```python
def _send_input(self, request_type, request):
    msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
    self.input_socket.send_multipart(msg, copy=False)
```

调用者：
- **`add_request()`** — 提交推理请求，发送 `EngineCoreRequestType.ADD`
- **`abort_requests()`** — 取消请求
- **`call_utility()`** — 同步 RPC 调用（如 `get_supported_tasks`）

### 2. 异步客户端 (`AsyncMPClient._send_input_message`)

`vllm/v1/engine/core_client.py:1110`

```python
def _send_input_message(self, message, engine, objects):
    msg = (engine,) + message
    return self.input_socket.send_multipart(msg, copy=False)
```

调用者：
- **`_send_input()`** → `add_request()`, `abort_requests()`, `call_utility_async()` 等

### 消息格式

每条消息是 ZMQ 多帧消息：

| 帧 | 内容 |
|---|---|
| 帧0 | Engine Identity（ROUTER 路由标识，决定发给哪个 EngineCore） |
| 帧1 | 请求类型（`EngineCoreRequestType` 的 1 字节值，如 `ADD`, `UTILITY`） |
| 帧2 | msgpack 序列化的请求数据 |
| 帧3+ | 可选的 tensor 辅助缓冲区（零拷贝） |

### 接收端

EngineCore 的 `process_input_sockets()` 方法（`vllm/v1/engine/core.py:1315`）在后台线程中运行，用 `zmq.Poller` 轮询 DEALER socket，通过 `input_socket.recv_multipart()` 接收消息，解析请求类型后放入内部队列供 `core_busy_loop` 消费。

### 简图

```
Client (ROUTER, bind)  ──send_multipart──▶  EngineCore (DEALER, connect)
   input_socket                              process_input_sockets() 轮询接收
```
