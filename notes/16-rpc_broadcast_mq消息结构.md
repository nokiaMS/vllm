# rpc_broadcast_mq 消息结构

## 概述

`rpc_broadcast_mq` 是 `MultiprocExecutor` 中主进程向所有 Worker 广播 RPC 请求的核心消息队列，底层实现为 `MessageQueue`（位于 `vllm/distributed/device_communicators/shm_broadcast.py`）。

消息结构分为两层：**应用层**（RPC 请求语义）和 **传输层**（共享内存序列化格式）。

---

## 1. 应用层消息（RPC 请求）

### 发送端：`MultiprocExecutor.collective_rpc`

```python
# vllm/v1/executor/multiproc_executor.py
self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
```

### 接收端：`WorkerProc.worker_busy_loop`

```python
# vllm/v1/executor/multiproc_executor.py
method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(indefinite=True)
```

### 4 元组字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `send_method` | `str` 或 `bytes` | 方法名字符串（如 `"execute_model"`），或通过 `cloudpickle.dumps()` 序列化的可调用对象 |
| `args` | `tuple` | 方法的位置参数 |
| `kwargs` | `dict` | 方法的关键字参数 |
| `output_rank` | `int \| None` | 指定哪个 rank 需要返回结果；`None` 表示所有 rank 都返回 |

### send_method 的两种形式

```python
if isinstance(method, str):        # 方法名字符串 → 直接发送
    send_method = method
else:                               # 可调用对象 → cloudpickle 序列化为 bytes
    send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
```

Worker 端对应的解析逻辑：

```python
if isinstance(method, str):         # 字符串 → 通过反射获取 Worker 上的方法
    func = getattr(self.worker, method)
else:                               # bytes → cloudpickle 反序列化为可调用对象
    func = cloudpickle.loads(method)
```

---

## 2. 传输层格式（共享内存 buffer）

`MessageQueue.enqueue` 将 Python 对象通过 `pickle.dumps` 序列化后写入共享内存，根据数据大小有两种模式。

### 2.1 正常模式（数据 < `max_chunk_bytes`）

直接写入 SHM buffer，二进制布局如下：

```
偏移    大小        内容
─────────────────────────────────────────
0       1 byte      标志位 = 0（非溢出）
1       2 bytes     buffer 数量 N（big-endian）
3       4 bytes     buffer_0 长度 L0
7       L0 bytes    buffer_0 内容（pickle 主数据）
7+L0    4 bytes     buffer_1 长度 L1        ← 可选 OOB buffer
...     L1 bytes    buffer_1 内容            ← ≥1MiB 的 PickleBuffer
...     ...         后续 OOB buffer 依次排列
```

对应代码：

```python
with self.acquire_write(timeout) as buf:
    buf[0] = 0                                          # 非溢出标志
    offset = 3
    buf[1:offset] = to_bytes_big(len(all_buffers), 2)   # buffer 数量
    for buffer in all_buffers:
        buf_len = len(buffer)
        buf_offset = offset + 4
        buf[offset:buf_offset] = to_bytes_big(buf_len, 4)  # 4字节长度前缀
        buf[buf_offset:(offset := buf_offset + buf_len)] = buffer  # buffer 内容
```

### 2.2 溢出模式（数据 >= `max_chunk_bytes`）

SHM buffer 中只写一个标志字节，实际数据走 ZMQ local socket：

```
偏移    大小        内容
─────────────────────────────────────────
0       1 byte      标志位 = 1（溢出）
```

```python
with self.acquire_write(timeout) as buf:
    buf[0] = 1  # overflow
self.local_socket.send_multipart(all_buffers, copy=False)
```

Reader 端检测到溢出后，通过 ZMQ socket 接收：

```python
with self.acquire_read(timeout, indefinite) as buf:
    overflow = buf[0] == 1
if overflow:
    obj = MessageQueue.recv(self.local_socket, timeout)
```

### 2.3 OOB (Out-of-Band) 零拷贝优化

pickle 序列化过程中，通过 `buffer_callback` 将大缓冲区拆分：

- **< 1MiB** 的 `PickleBuffer`：内联到 pickle 主数据中（返回 `True`）
- **≥ 1MiB** 的 `PickleBuffer`：作为独立 OOB buffer 追加到 `all_buffers` 列表（返回 `False`），避免内存拷贝

```python
def oob_callback(buf: PickleBuffer) -> bool:
    raw_buf = buf.raw()
    if len(raw_buf) < 1024 * 1024:    # < 1MiB，内联
        return True
    all_buffers.append(raw_buf)        # ≥ 1MiB，拆出为 OOB buffer
    nonlocal total_bytes
    total_bytes += len(raw_buf) + 4
    return False
```

反序列化时将 OOB buffer 传回 pickle：

```python
obj = pickle.loads(all_buffers[0], buffers=all_buffers[1:])
```

---

## 3. 跨节点场景

多节点部署（`nnodes_within_dp > 1`）时，`rpc_broadcast_mq` 的创建方式不同：

```python
# 单节点：从共享内存句柄创建
self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, self.worker.rank)

# 多节点：通过分布式进程组创建跨节点广播队列
self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
    external_writer_handle=input_shm_handle,
    blocking=False,
)
```

多节点模式下，底层使用 ZMQ remote socket 跨节点转发消息，消息格式与单节点一致。远程 reader 端始终通过 `zmq.Socket.recv_multipart` 接收并反序列化。

---

## 4. 完整数据流

```
MultiprocExecutor                          WorkerProc (每个 rank)
      │                                         │
      │  collective_rpc(method, args, kwargs)    │
      │                                         │
      │  ┌─────────────────────────────┐        │
      │  │ 构造 4 元组:                 │        │
      │  │ (send_method, args,         │        │
      │  │  kwargs, output_rank)       │        │
      │  └─────────────┬───────────────┘        │
      │                │                         │
      │  rpc_broadcast_mq.enqueue(tuple)         │
      │                │                         │
      │    ┌───────────▼────────────┐            │
      │    │ pickle.dumps + OOB 拆分 │            │
      │    │                        │            │
      │    │ 数据 < max_chunk_bytes? │            │
      │    │  Y → 写入 SHM buffer    │            │
      │    │  N → SHM 标记溢出 +     │            │
      │    │      ZMQ socket 发送    │            │
      │    └───────────┬────────────┘            │
      │                │                         │
      │    ════════════╪══════ SHM / ZMQ ════════╪════
      │                │                         │
      │                │    rpc_broadcast_mq.dequeue()
      │                │         ┌───────────────▼──────────┐
      │                │         │ 读取 SHM / ZMQ 接收       │
      │                │         │ pickle.loads 反序列化      │
      │                │         │ → (method, args, kwargs,  │
      │                │         │    output_rank)           │
      │                │         └───────────────┬──────────┘
      │                │                         │
      │                │         getattr(worker, method)(*args, **kwargs)
      │                │                         │
      │                │         worker_response_mq.enqueue(result)
      │                │                         │
      │    ◄═══════════╪═══════ response_mq ═════╪════
      │                                         │
      │  response_mq.dequeue() → result          │
```
