# vllm 编译安装指南

## 环境信息

- **服务器平台**: AutoDL GPU 云服务器
- **操作系统**: Ubuntu 22.04
- **Python**: 3.12.3 (Miniconda)
- **CUDA Toolkit**: 12.8
- **NVIDIA Driver**: 580.105.08
- **编译产物**: vllm-0.17.1rc1.dev157+gffa5d74f1.cu128

## 目录结构

```
/root/autodl-tmp/
├── code/
│   └── vllm/          # vllm 源码（从 GitHub 克隆）
└── vllm_build.log     # 编译日志
```

## 编译步骤

### 1. 配置网络加速

AutoDL 提供学术网络加速，需要在 `~/.bashrc` 中添加：

```bash
source /etc/network_turbo
```

此命令启用后可加速访问 GitHub、HuggingFace 等站点。

### 2. 安装 Docker

系统默认未安装 Docker，通过 apt 安装：

```bash
apt-get update -qq && apt-get install -y -qq docker.io
```

> 注意：AutoDL 容器环境中 Docker 官方安装脚本（get.docker.com）可能无法访问，使用 apt 源安装更可靠。

### 3. 准备源码

```bash
mkdir -p /root/autodl-tmp/code
cd /root/autodl-tmp/code
git clone https://github.com/nokiaMS/vllm.git
```

> 代码放在数据盘 `/root/autodl-tmp` 下，重装系统后数据不会丢失。

### 4. 初始化 Python 环境

AutoDL 使用 Miniconda，但默认 SSH 登录时 conda 不在 PATH 中，需要手动初始化：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
```

### 5. 安装编译依赖

使用清华镜像源加速 pip 下载：

```bash
cd /root/autodl-tmp/code/vllm
pip install -r requirements/build.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/cuda.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要依赖包括：
- **build.txt**: cmake, ninja, torch, triton, packaging, setuptools 等构建工具
- **cuda.txt**: transformers, flash-attn, huggingface-hub, openai, fastapi, uvicorn 等运行时依赖

> 安装过程中可能出现 torchvision 版本不兼容的警告，不影响编译。

### 6. 检查 CUDA 版本

vllm 编译要求 CUDA 12.8：

```bash
/usr/local/cuda/bin/nvcc --version
```

如果版本不是 12.8，需要手动安装：

```bash
cd /root/autodl-tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### 7. 编译 vllm

```bash
cd /root/autodl-tmp/code/vllm
export PATH=/usr/local/cuda/bin:$PATH
MAX_JOBS=20 pip install -e . --no-build-isolation -v
```

参数说明：
- `MAX_JOBS=20`: 并行编译任务数，加快编译速度
- `-e .`: editable 模式安装，方便后续修改源码
- `--no-build-isolation`: 不创建隔离构建环境，使用当前环境的依赖
- `-v`: 详细输出，便于排查问题

### 编译耗时与注意事项

- 编译总共需要构建 **343 个** CUDA 内核文件，耗时约 **40-60 分钟**
- 编译阶段包括：量化内核（marlin/gptq/awq）、MoE 内核、Flash Attention v2/v3 内核等
- **建议使用 nohup 后台执行**，避免 SSH 断连导致编译中断：

```bash
nohup bash -c "MAX_JOBS=20 pip install -e . --no-build-isolation -v" > /root/autodl-tmp/vllm_build.log 2>&1 &
```

查看编译进度：

```bash
tail -f /root/autodl-tmp/vllm_build.log
```

### 编译验证

编译成功后日志末尾会显示：

```
Successfully installed vllm-0.17.1rc1.dev157+gffa5d74f1.cu128
```

可通过以下命令验证安装：

```bash
python -c "import vllm; print(vllm.__version__)"
```
