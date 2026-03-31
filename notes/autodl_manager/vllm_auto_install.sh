#!/bin/bash
#
# vllm 自动编译安装脚本（适用于 AutoDL GPU 服务器）
#
# 用法:
#   方式一（本地执行）: 将脚本上传到服务器后直接运行
#     chmod +x vllm_auto_install.sh && ./vllm_auto_install.sh
#
#   方式二（远程执行）: 从本地通过 SSH 执行
#     sshpass -p '<密码>' ssh -p <端口> root@<主机> 'bash -s' < vllm_auto_install.sh
#

set -e

# ============================================================
# 配置区
# ============================================================
VLLM_REPO="https://github.com/nokiaMS/vllm.git"
CODE_DIR="/root/autodl-tmp/code"
VLLM_DIR="${CODE_DIR}/vllm"
BUILD_LOG="/root/autodl-tmp/vllm_build.log"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
MAX_JOBS=20
REQUIRED_CUDA_VERSION="12.8"

# ============================================================
# 工具函数
# ============================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_result() {
    if [ $? -ne 0 ]; then
        log_error "$1"
        exit 1
    fi
}

# ============================================================
# 步骤 1: 配置网络加速
# ============================================================
log_info "步骤 1/7: 配置网络加速..."

if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo 2>/dev/null || true

    if ! grep -q "source /etc/network_turbo" ~/.bashrc 2>/dev/null; then
        echo "source /etc/network_turbo" >> ~/.bashrc
        log_info "已将 network_turbo 添加到 ~/.bashrc"
    else
        log_info "network_turbo 已配置，跳过"
    fi
else
    log_warn "/etc/network_turbo 不存在，跳过网络加速配置"
fi

# ============================================================
# 步骤 2: 安装 Docker
# ============================================================
log_info "步骤 2/7: 检查 Docker..."

if command -v docker &>/dev/null; then
    log_info "Docker 已安装: $(docker --version)"
else
    log_info "正在安装 Docker..."
    apt-get update -qq && apt-get install -y -qq docker.io
    check_result "Docker 安装失败"
    log_info "Docker 安装完成: $(docker --version)"
fi

# ============================================================
# 步骤 3: 初始化 Python 环境
# ============================================================
log_info "步骤 3/7: 初始化 Python 环境..."

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
    log_info "Python 版本: $(python --version), pip 版本: $(pip --version)"
else
    log_error "未找到 Miniconda，请先安装 Miniconda"
    exit 1
fi

# ============================================================
# 步骤 4: 克隆 vllm 源码
# ============================================================
log_info "步骤 4/7: 准备 vllm 源码..."

mkdir -p "${CODE_DIR}"

if [ -d "${VLLM_DIR}" ]; then
    log_info "vllm 目录已存在，执行 git pull 更新..."
    cd "${VLLM_DIR}" && git pull || true
else
    log_info "正在克隆 vllm 仓库..."
    cd "${CODE_DIR}" && git clone "${VLLM_REPO}"
    check_result "vllm 仓库克隆失败"
fi

# ============================================================
# 步骤 5: 安装编译依赖
# ============================================================
log_info "步骤 5/7: 安装编译依赖..."

cd "${VLLM_DIR}"

log_info "安装 build 依赖..."
pip install -r requirements/build.txt -i "${PIP_MIRROR}"
check_result "build 依赖安装失败"

log_info "安装 cuda 依赖..."
pip install -r requirements/cuda.txt -i "${PIP_MIRROR}"
check_result "cuda 依赖安装失败"

log_info "依赖安装完成"

# ============================================================
# 步骤 6: 检查并安装 CUDA 12.8
# ============================================================
log_info "步骤 6/7: 检查 CUDA 版本..."

CUDA_VERSION=""
if [ -f /usr/local/cuda/bin/nvcc ]; then
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
fi

if [ "${CUDA_VERSION}" = "${REQUIRED_CUDA_VERSION}" ]; then
    log_info "CUDA 版本为 ${CUDA_VERSION}，满足要求"
else
    log_warn "当前 CUDA 版本为 '${CUDA_VERSION}'，需要 ${REQUIRED_CUDA_VERSION}，开始安装..."

    cd /root/autodl-tmp

    log_info "下载 CUDA ${REQUIRED_CUDA_VERSION} 安装包..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

    wget -q https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
    check_result "CUDA 安装包下载失败"

    sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update -qq
    sudo apt-get -y install cuda-toolkit-12-8
    check_result "CUDA ${REQUIRED_CUDA_VERSION} 安装失败"

    # 验证安装
    NEW_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    log_info "CUDA 已更新至 ${NEW_VERSION}"
fi

export PATH=/usr/local/cuda/bin:$PATH

# ============================================================
# 步骤 7: 编译 vllm
# ============================================================
log_info "步骤 7/7: 开始编译 vllm（共 343 个 CUDA 内核，预计 40-60 分钟）..."
log_info "编译日志: ${BUILD_LOG}"

cd "${VLLM_DIR}"

MAX_JOBS=${MAX_JOBS} pip install -e . --no-build-isolation -v 2>&1 | tee "${BUILD_LOG}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_info "============================================"
    log_info "vllm 编译安装成功！"
    log_info "版本: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '未知')"
    log_info "============================================"
else
    log_error "vllm 编译失败，请查看日志: ${BUILD_LOG}"
    exit 1
fi
