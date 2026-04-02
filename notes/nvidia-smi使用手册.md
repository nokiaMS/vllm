# nvidia-smi 使用手册

## 1. 概述

`nvidia-smi`（NVIDIA System Management Interface）是 NVIDIA 提供的命令行工具，用于管理和监控 NVIDIA GPU 设备。它基于 NVML（NVIDIA Management Library）构建，支持 Tesla、Quadro、GRID 和 GeForce 系列显卡。

主要功能：
- 查询 GPU 状态（温度、功耗、显存、利用率等）
- 管理 GPU 配置（时钟频率、计算模式、ECC 等）
- 监控进程的 GPU 资源使用
- 管理 MIG（Multi-Instance GPU）实例

---

## 2. 基础用法

### 2.1 默认输出

```bash
nvidia-smi
```

输出示例：

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB         On  | 00000000:07:00.0   Off |                    0 |
| N/A   32C    P0              63W / 400W |    1024MiB /  81920MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage       |
|=========================================================================================|
|    0   N/A  N/A     12345      C   python                                     1020MiB   |
+-----------------------------------------------------------------------------------------+
```

**字段说明：**

| 字段 | 含义 |
|------|------|
| GPU | GPU 编号（从 0 开始） |
| Name | GPU 型号 |
| Persistence-M | 持久模式（On/Off） |
| Bus-Id | PCIe 总线地址 |
| Disp.A | 显示活跃状态 |
| Volatile Uncorr. ECC | 未纠正的 ECC 错误计数 |
| Fan | 风扇转速百分比（N/A 表示被动散热） |
| Temp | GPU 核心温度（摄氏度） |
| Perf | 性能状态（P0 最高 ~ P12 最低） |
| Pwr:Usage/Cap | 当前功耗 / 功耗上限 |
| Memory-Usage | 已用显存 / 总显存 |
| GPU-Util | GPU 计算核心利用率 |
| Compute M. | 计算模式（Default/Exclusive_Process/Prohibited） |

---

## 3. 查询命令（query）

### 3.1 查询 GPU 信息

```bash
# 查询所有 GPU 的基础信息
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv

# 只查询特定 GPU（如 GPU 0 和 GPU 1）
nvidia-smi -i 0,1 --query-gpu=name,memory.used --format=csv

# 不显示表头
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 不显示单位
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

### 3.2 常用查询属性

**GPU 基础属性：**

| 属性 | 说明 |
|------|------|
| `index` | GPU 索引编号 |
| `name` | GPU 型号名称 |
| `uuid` | GPU 唯一标识符 |
| `serial` | 序列号 |
| `driver_version` | 驱动版本 |
| `pci.bus_id` | PCIe 总线 ID |
| `gpu_bus_id` | 同上（旧写法） |

**显存属性：**

| 属性 | 说明 |
|------|------|
| `memory.total` | 总显存（MiB） |
| `memory.used` | 已用显存（MiB） |
| `memory.free` | 空闲显存（MiB） |

**利用率属性：**

| 属性 | 说明 |
|------|------|
| `utilization.gpu` | GPU 核心利用率（%） |
| `utilization.memory` | 显存控制器利用率（%） |
| `encoder.stats.sessionCount` | 编码器会话数 |
| `decoder.stats.sessionCount` | 解码器会话数 |

**温度与功耗属性：**

| 属性 | 说明 |
|------|------|
| `temperature.gpu` | GPU 核心温度（C） |
| `temperature.memory` | 显存温度（C，部分型号支持） |
| `power.draw` | 当前功耗（W） |
| `power.limit` | 功耗限制（W） |
| `power.max_limit` | 最大可设功耗限制 |
| `power.min_limit` | 最小可设功耗限制 |

**时钟频率属性：**

| 属性 | 说明 |
|------|------|
| `clocks.current.graphics` | 当前图形核心频率（MHz） |
| `clocks.current.sm` | 当前 SM 频率（MHz） |
| `clocks.current.memory` | 当前显存频率（MHz） |
| `clocks.max.graphics` | 最大图形核心频率 |
| `clocks.max.memory` | 最大显存频率 |

**ECC 属性：**

| 属性 | 说明 |
|------|------|
| `ecc.mode.current` | 当前 ECC 模式 |
| `ecc.errors.corrected.volatile.total` | 已纠正的 ECC 错误数（本次启动） |
| `ecc.errors.uncorrected.volatile.total` | 未纠正的 ECC 错误数（本次启动） |

### 3.3 查询进程信息

```bash
# 查询所有使用 GPU 的进程
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

# 查询特定 GPU 上的进程
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

### 3.4 列出所有可查询属性

```bash
# 列出所有 GPU 可查询属性
nvidia-smi --help-query-gpu

# 列出所有进程可查询属性
nvidia-smi --help-query-compute-apps
```

---

## 4. 实时监控

### 4.1 循环刷新（-l / --loop）

```bash
# 每 1 秒刷新一次默认视图
nvidia-smi -l 1

# 每 2 秒刷新，以 CSV 格式输出
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv -l 2
```

### 4.2 毫秒级刷新（-lms）

```bash
# 每 500 毫秒刷新一次
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 500
```

### 4.3 dmon（设备监控模式）

```bash
# 默认每秒采样一次
nvidia-smi dmon

# 自定义采样间隔（秒）和选择指标
nvidia-smi dmon -s pucvmet -d 2

# 监控特定 GPU
nvidia-smi dmon -i 0 -s puc -d 1
```

**dmon 指标选项（-s）：**

| 选项 | 含义 |
|------|------|
| `p` | 功耗（Power） |
| `u` | 利用率（Utilization） |
| `c` | 处理器时钟（Clocks） |
| `v` | 功耗违规（Power Violations） |
| `m` | FB 显存（Frame Buffer Memory） |
| `e` | ECC 错误与 PCIe 错误 |
| `t` | 温度（Temperature） |

### 4.4 pmon（进程监控模式）

```bash
# 监控所有 GPU 上的进程
nvidia-smi pmon -d 1

# 监控指定 GPU
nvidia-smi pmon -i 0 -d 1

# 输出指定采样次数后停止
nvidia-smi pmon -c 10 -d 1
```

输出字段：GPU 编号、PID、进程类型（C=计算/G=图形）、SM 利用率、显存利用率、编解码器利用率、显存使用量、进程名。

---

## 5. 拓扑查询

### 5.1 GPU 间通信拓扑

```bash
nvidia-smi topo -m
# 或
nvidia-smi topo --matrix
```

输出示例（8 卡 A100）：

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12
...
```

**拓扑连接类型：**

| 缩写 | 含义 | 说明 |
|------|------|------|
| `X` | 自身 | — |
| `SYS` | System | 跨 NUMA 节点，通过 CPU 互联 |
| `NODE` | Node | 同 NUMA 节点，通过 CPU 互联 |
| `PHB` | PCIe Host Bridge | 同一 PCIe 主桥 |
| `PXB` | PCIe Bridge | 经过多个 PCIe 桥 |
| `PIX` | PCIe | 同一 PCIe 交换机 |
| `NV#` | NVLink | NVLink 直连，# 表示链路数 |

### 5.2 查询 GPU 与 CPU 亲和性

```bash
nvidia-smi topo -p
```

---

## 6. 管理配置

> 以下操作大多需要 root 权限。

### 6.1 持久模式（Persistence Mode）

持久模式使 NVIDIA 驱动保持加载状态，避免每次 CUDA 调用时重新初始化（可减少数百毫秒延迟）。**服务器环境强烈建议开启。**

```bash
# 开启持久模式（所有 GPU）
sudo nvidia-smi -pm 1

# 关闭持久模式
sudo nvidia-smi -pm 0

# 针对特定 GPU
sudo nvidia-smi -i 0 -pm 1
```

### 6.2 计算模式（Compute Mode）

```bash
# 默认模式：多进程共享
sudo nvidia-smi -c 0
# 或
sudo nvidia-smi --compute-mode=DEFAULT

# 独占进程模式：仅允许一个 CUDA 上下文
sudo nvidia-smi -c 3
# 或
sudo nvidia-smi --compute-mode=EXCLUSIVE_PROCESS

# 禁止计算模式：不允许任何 CUDA 上下文
sudo nvidia-smi -c 2
# 或
sudo nvidia-smi --compute-mode=PROHIBITED
```

| 模式值 | 名称 | 说明 |
|--------|------|------|
| 0 | DEFAULT | 多进程可同时使用 GPU |
| 2 | PROHIBITED | 禁止创建任何 CUDA 上下文 |
| 3 | EXCLUSIVE_PROCESS | 同一时间只允许一个进程（可多线程） |

### 6.3 功耗限制

```bash
# 查看当前功耗限制
nvidia-smi -q -d POWER

# 设置功耗限制（瓦特）
sudo nvidia-smi -pl 300

# 针对特定 GPU
sudo nvidia-smi -i 0 -pl 350
```

### 6.4 GPU 时钟频率锁定

```bash
# 锁定 GPU 核心频率（最小,最大 MHz）
sudo nvidia-smi -lgc 1200,1200

# 锁定显存频率
sudo nvidia-smi -lmc 1593,1593

# 重置为默认
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

### 6.5 ECC 管理

```bash
# 查看 ECC 状态
nvidia-smi -q -d ECC

# 开启 ECC（需重启生效）
sudo nvidia-smi -e 1

# 关闭 ECC
sudo nvidia-smi -e 0

# 重置 ECC 错误计数
sudo nvidia-smi -p 0   # 重置 volatile 计数
sudo nvidia-smi -p 1   # 重置 aggregate 计数
```

### 6.6 GPU 重置

```bash
# 重置指定 GPU（须先停止所有使用该 GPU 的进程）
sudo nvidia-smi -r -i 0
# 或
sudo nvidia-smi --gpu-reset -i 0
```

---

## 7. 详细信息查询（-q）

### 7.1 全量查询

```bash
# 查询所有 GPU 的所有信息
nvidia-smi -q

# 查询特定 GPU
nvidia-smi -q -i 0

# 输出为 XML 格式
nvidia-smi -q -x
```

### 7.2 按类别查询（-d）

```bash
nvidia-smi -q -d <SECTION>
```

常用 SECTION 值：

| SECTION | 说明 |
|---------|------|
| `MEMORY` | 显存详情 |
| `UTILIZATION` | 利用率 |
| `ECC` | ECC 错误信息 |
| `TEMPERATURE` | 温度详情（含阈值） |
| `POWER` | 功耗详情 |
| `CLOCK` | 时钟频率 |
| `COMPUTE` | 计算模式 |
| `PIDS` | 进程信息 |
| `PERFORMANCE` | 性能状态 |
| `SUPPORTED_CLOCKS` | 支持的时钟频率组合 |
| `PAGE_RETIREMENT` | 显存页面退役信息 |
| `VOLTAGE` | 电压信息 |

可组合使用：

```bash
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE
```

---

## 8. MIG（Multi-Instance GPU）管理

> 仅 A100、A30、H100 等支持 MIG 的型号可用。

### 8.1 启用/禁用 MIG

```bash
# 启用 MIG 模式（需重置 GPU 或重启）
sudo nvidia-smi -i 0 -mig 1

# 禁用 MIG 模式
sudo nvidia-smi -i 0 -mig 0
```

### 8.2 查看 MIG 配置

```bash
# 查看可用的 GPU Instance Profile
nvidia-smi mig -lgip

# 查看可用的 Compute Instance Profile
nvidia-smi mig -lcip

# 查看当前 GPU 实例
nvidia-smi mig -lgi

# 查看当前计算实例
nvidia-smi mig -lci
```

### 8.3 创建/销毁 MIG 实例

```bash
# 创建 GPU Instance（以 A100 80GB 为例）
# Profile ID 19 = 1g.10gb
sudo nvidia-smi mig -i 0 -cgi 19

# 创建 Compute Instance
sudo nvidia-smi mig -i 0 -gi 0 -cci 0

# 销毁 Compute Instance
sudo nvidia-smi mig -i 0 -gi 0 -dci 0

# 销毁 GPU Instance
sudo nvidia-smi mig -i 0 -dgi 0
```

### 8.4 常见 MIG Profile（A100 80GB）

| Profile ID | Profile Name | SM数 | 显存 |
|------------|-------------|------|------|
| 19 | 1g.10gb | 1/7 | 10GB |
| 20 | 1g.10gb+me | 1/7 | 10GB + 媒体引擎 |
| 14 | 2g.20gb | 2/7 | 20GB |
| 9 | 3g.40gb | 3/7 | 40GB |
| 5 | 4g.40gb | 4/7 | 40GB |
| 0 | 7g.80gb | 7/7 | 80GB |

---

## 9. 实用脚本与技巧

### 9.1 监控 GPU 利用率并写入日志

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw \
  --format=csv -l 5 | tee gpu_monitor.csv
```

### 9.2 查看显存占用最高的进程

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader | sort -t',' -k3 -rn | head -10
```

### 9.3 结束占用特定 GPU 的所有进程

```bash
# 查看 GPU 0 上的进程并终止
nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | xargs -I{} kill -9 {}
```

### 9.4 等待 GPU 空闲后执行任务

```bash
while [ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0) -gt 5 ]; do
    echo "GPU busy, waiting..."
    sleep 10
done
echo "GPU idle, starting job..."
python train.py
```

### 9.5 选择空闲显存最多的 GPU 运行程序

```bash
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)
echo "Using GPU $GPU_ID"
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py
```

### 9.6 多 GPU 温度告警

```bash
nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r gpu temp; do
    if [ "$temp" -gt 85 ]; then
        echo "WARNING: GPU $gpu temperature is ${temp}C!"
    fi
done
```

### 9.7 配合 watch 使用

```bash
# 每 0.5 秒刷新
watch -n 0.5 nvidia-smi

# 高亮变化
watch -d -n 1 nvidia-smi
```

---

## 10. 环境变量与 nvidia-smi 的关系

| 环境变量 | 说明 |
|----------|------|
| `CUDA_VISIBLE_DEVICES` | 限制进程可见的 GPU 编号 |
| `CUDA_DEVICE_ORDER` | GPU 排序方式（`PCI_BUS_ID` 按 PCIe 总线排序） |
| `NVIDIA_VISIBLE_DEVICES` | 容器环境中限制可见 GPU（用于 Docker） |
| `CUDA_MPS_PIPE_DIRECTORY` | MPS 管道目录 |

注意：`CUDA_VISIBLE_DEVICES` 影响 CUDA 程序看到的 GPU 编号，但**不影响** `nvidia-smi` 的输出。`nvidia-smi` 始终显示系统中所有 GPU。

---

## 11. 常见问题排查

### 11.1 "NVIDIA-SMI has failed..." 错误

```bash
# 检查驱动是否加载
lsmod | grep nvidia

# 检查内核模块日志
dmesg | grep -i nvidia

# 尝试重新加载驱动
sudo modprobe nvidia
```

常见原因：
- 内核更新后驱动未重新编译
- 安全启动（Secure Boot）阻止了内核模块加载
- 驱动与 GPU 型号不兼容

### 11.2 显存泄漏排查

```bash
# 找到占用显存但无对应进程的"僵尸"显存
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader | while IFS=',' read -r pid mem; do
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "Orphaned GPU memory: PID $pid using $mem"
    fi
done
```

### 11.3 GPU 掉卡排查

```bash
# 检查 PCIe 错误
nvidia-smi -q -d PCIE

# 检查 Xid 错误
dmesg | grep -i "xid"

# 查看 GPU 健康状态
nvidia-smi -q -d PERFORMANCE
```

### 11.4 性能状态异常（P 状态过低）

```bash
# 查看当前性能状态
nvidia-smi -q -d PERFORMANCE

# 检查是否因为功耗/温度限制降频
nvidia-smi -q -d POWER
nvidia-smi -q -d TEMPERATURE

# 检查时钟频率限制原因
nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv
```

---

## 12. 常用命令速查表

| 用途 | 命令 |
|------|------|
| 基础状态 | `nvidia-smi` |
| 持续监控（1秒） | `nvidia-smi -l 1` |
| CSV 格式查询 | `nvidia-smi --query-gpu=... --format=csv` |
| 设备监控 | `nvidia-smi dmon -d 1` |
| 进程监控 | `nvidia-smi pmon -d 1` |
| GPU 拓扑 | `nvidia-smi topo -m` |
| 详细信息 | `nvidia-smi -q` |
| 按类别查询 | `nvidia-smi -q -d MEMORY` |
| XML 输出 | `nvidia-smi -q -x` |
| 开启持久模式 | `sudo nvidia-smi -pm 1` |
| 设置功耗上限 | `sudo nvidia-smi -pl <watts>` |
| 锁定 GPU 频率 | `sudo nvidia-smi -lgc <min>,<max>` |
| 重置 GPU | `sudo nvidia-smi --gpu-reset -i <id>` |
| MIG 管理 | `nvidia-smi mig -lgip / -cgi / -dgi` |
| 列出所有可查询属性 | `nvidia-smi --help-query-gpu` |
