# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa
# code borrowed from https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py

import datetime  # 导入日期时间模块
import locale  # 导入区域设置模块
import os  # 导入操作系统接口模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块

# Unlike the rest of the PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python collect_env.py` or `python -m torch.utils.collect_env`
from collections import namedtuple  # 从collections导入命名元组

import regex as re  # 导入正则表达式模块（增强版）

from vllm.envs import environment_variables  # 从vllm环境模块导入环境变量

try:  # 尝试导入torch
    import torch  # 导入PyTorch

    TORCH_AVAILABLE = True  # 标记torch可用
except (ImportError, NameError, AttributeError, OSError):  # 捕获导入相关异常
    TORCH_AVAILABLE = False  # 标记torch不可用

# System Environment Information
SystemEnv = namedtuple(  # 定义系统环境信息命名元组
    "SystemEnv",  # 元组名称
    [
        "torch_version",  # PyTorch版本
        "is_debug_build",  # 是否为调试构建
        "cuda_compiled_version",  # CUDA编译版本
        "gcc_version",  # GCC版本
        "clang_version",  # Clang版本
        "cmake_version",  # CMake版本
        "os",  # 操作系统
        "libc_version",  # libc版本
        "python_version",  # Python版本
        "python_platform",  # Python平台
        "is_cuda_available",  # CUDA是否可用
        "cuda_runtime_version",  # CUDA运行时版本
        "cuda_module_loading",  # CUDA模块加载方式
        "nvidia_driver_version",  # NVIDIA驱动版本
        "nvidia_gpu_models",  # NVIDIA GPU型号
        "cudnn_version",  # cuDNN版本
        "pip_version",  # pip版本（'pip'或'pip3'）
        "pip_packages",  # pip安装的包
        "conda_packages",  # conda安装的包
        "hip_compiled_version",  # HIP编译版本
        "hip_runtime_version",  # HIP运行时版本
        "miopen_runtime_version",  # MIOpen运行时版本
        "caching_allocator_config",  # 缓存分配器配置
        "is_xnnpack_available",  # XNNPACK是否可用
        "cpu_info",  # CPU信息
        "rocm_version",  # ROCm版本（vllm特有字段）
        "vllm_version",  # vLLM版本（vllm特有字段）
        "vllm_build_flags",  # vLLM构建标志（vllm特有字段）
        "gpu_topo",  # GPU拓扑（vllm特有字段）
        "env_vars",  # 环境变量
    ],
)

DEFAULT_CONDA_PATTERNS = {  # 默认的conda包匹配模式集合
    "torch",  # PyTorch相关包
    "numpy",  # NumPy相关包
    "cudatoolkit",  # CUDA工具包
    "soumith",  # soumith相关包
    "mkl",  # MKL数学库
    "magma",  # MAGMA线性代数库
    "triton",  # Triton编译器
    "optree",  # optree库
    "nccl",  # NCCL通信库
    "transformers",  # Transformers库
    "zmq",  # ZeroMQ消息队列
    "nvidia",  # NVIDIA相关包
    "pynvml",  # Python NVML绑定
    "flashinfer-python",  # FlashInfer库
    "helion",  # Helion库
}

DEFAULT_PIP_PATTERNS = {  # 默认的pip包匹配模式集合
    "torch",  # PyTorch相关包
    "numpy",  # NumPy相关包
    "mypy",  # mypy类型检查器
    "flake8",  # flake8代码检查器
    "triton",  # Triton编译器
    "optree",  # optree库
    "onnx",  # ONNX框架
    "nccl",  # NCCL通信库
    "transformers",  # Transformers库
    "zmq",  # ZeroMQ消息队列
    "nvidia",  # NVIDIA相关包
    "pynvml",  # Python NVML绑定
    "flashinfer-python",  # FlashInfer库
    "helion",  # Helion库
}


def run(command):
    """执行命令并返回结果。

    返回 (返回码, 标准输出, 标准错误) 的元组。

    Args:
        command: 要执行的命令，可以是字符串或列表。

    Returns:
        包含 (返回码, 标准输出, 标准错误) 的元组。
    """
    shell = True if type(command) is str else False  # 如果命令是字符串则使用shell模式
    try:  # 尝试执行命令
        p = subprocess.Popen(  # 创建子进程
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell  # 捕获标准输出和标准错误
        )
        raw_output, raw_err = p.communicate()  # 等待进程完成并获取输出
        rc = p.returncode  # 获取返回码
        if get_platform() == "win32":  # 如果是Windows平台
            enc = "oem"  # 使用OEM编码
        else:  # 其他平台
            enc = locale.getpreferredencoding()  # 使用系统首选编码
        output = raw_output.decode(enc)  # 解码标准输出
        if command == "nvidia-smi topo -m":  # 如果是nvidia-smi拓扑命令
            # don't remove the leading whitespace of `nvidia-smi topo -m`
            #   because they are meaningful
            output = output.rstrip()  # 仅去除右侧空白（保留左侧有意义的空白）
        else:  # 其他命令
            output = output.strip()  # 去除两侧空白
        err = raw_err.decode(enc)  # 解码标准错误
        return rc, output, err.strip()  # 返回返回码、输出和错误

    except FileNotFoundError:  # 命令未找到异常
        cmd_str = command if isinstance(command, str) else command[0]  # 获取命令字符串
        return 127, "", f"Command not found: {cmd_str}"  # 返回127错误码


def run_and_read_all(run_lambda, command):
    """使用run_lambda执行命令；如果返回码为0则读取并返回全部输出。

    Args:
        run_lambda: 执行命令的函数。
        command: 要执行的命令。

    Returns:
        命令输出字符串，如果执行失败则返回None。
    """
    rc, out, _ = run_lambda(command)  # 执行命令
    if rc != 0:  # 如果返回码不为0
        return None  # 返回None表示失败
    return out  # 返回输出


def run_and_parse_first_match(run_lambda, command, regex):
    """使用run_lambda执行命令，返回第一个正则匹配结果。

    Args:
        run_lambda: 执行命令的函数。
        command: 要执行的命令。
        regex: 用于匹配输出的正则表达式。

    Returns:
        第一个匹配组的内容，如果无匹配则返回None。
    """
    rc, out, _ = run_lambda(command)  # 执行命令
    if rc != 0:  # 如果返回码不为0
        return None  # 返回None
    match = re.search(regex, out)  # 在输出中搜索正则匹配
    if match is None:  # 如果没有匹配
        return None  # 返回None
    return match.group(1)  # 返回第一个捕获组


def get_conda_packages(run_lambda, patterns=None):
    """获取匹配指定模式的conda包列表。

    Args:
        run_lambda: 执行命令的函数。
        patterns: 要匹配的包名模式集合，默认使用DEFAULT_CONDA_PATTERNS。

    Returns:
        匹配的conda包信息字符串，失败则返回None。
    """
    if patterns is None:  # 如果未指定匹配模式
        patterns = DEFAULT_CONDA_PATTERNS  # 使用默认模式
    conda = os.environ.get("CONDA_EXE", "conda")  # 获取conda可执行文件路径
    out = run_and_read_all(run_lambda, [conda, "list"])  # 执行conda list命令
    if out is None:  # 如果执行失败
        return out  # 返回None

    return "\n".join(  # 将匹配的行连接成字符串
        line  # 每一行
        for line in out.splitlines()  # 遍历输出的每一行
        if not line.startswith("#") and any(name in line for name in patterns)  # 过滤注释行并匹配模式
    )


def get_gcc_version(run_lambda):
    """获取GCC编译器版本。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        GCC版本字符串，未找到则返回None。
    """
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")  # 从gcc --version输出中提取版本


def get_clang_version(run_lambda):
    """获取Clang编译器版本。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        Clang版本字符串，未找到则返回None。
    """
    return run_and_parse_first_match(  # 从clang --version输出中提取版本
        run_lambda, "clang --version", r"clang version (.*)"  # 匹配clang版本号
    )


def get_cmake_version(run_lambda):
    """获取CMake构建工具版本。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        CMake版本字符串，未找到则返回None。
    """
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")  # 从cmake --version输出中提取版本


def get_nvidia_driver_version(run_lambda):
    """获取NVIDIA驱动版本。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        NVIDIA驱动版本字符串，未找到则返回None。
    """
    if get_platform() == "darwin":  # 如果是macOS平台
        cmd = "kextstat | grep -i cuda"  # 使用kextstat查找CUDA内核扩展
        return run_and_parse_first_match(  # 从输出中提取CUDA版本
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"  # 匹配NVIDIA CUDA版本
        )
    smi = get_nvidia_smi()  # 获取nvidia-smi路径
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")  # 从nvidia-smi输出中提取驱动版本


def get_gpu_info(run_lambda):
    """获取GPU信息。

    支持NVIDIA GPU和AMD ROCm GPU。
    在macOS或ROCm环境下通过PyTorch获取GPU信息，
    其他平台通过nvidia-smi获取。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        GPU型号信息字符串，未找到则返回None。
    """
    if get_platform() == "darwin" or (  # 如果是macOS平台
        TORCH_AVAILABLE  # 或者torch可用
        and hasattr(torch.version, "hip")  # 并且有HIP版本属性
        and torch.version.hip is not None  # 且HIP版本不为空（即ROCm环境）
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():  # 如果torch可用且CUDA可用
            if torch.version.hip is not None:  # 如果是HIP/ROCm环境
                prop = torch.cuda.get_device_properties(0)  # 获取设备属性
                if hasattr(prop, "gcnArchName"):  # 如果有GCN架构名称
                    gcnArch = " ({})".format(prop.gcnArchName)  # 格式化GCN架构名称
                else:  # 旧版PyTorch没有此属性
                    gcnArch = "NoGCNArchNameOnOldPyTorch"  # 标记旧版PyTorch
            else:  # 非HIP环境
                gcnArch = ""  # GCN架构名称为空
            return torch.cuda.get_device_name(None) + gcnArch  # 返回设备名称加架构信息
        return None  # CUDA不可用返回None
    smi = get_nvidia_smi()  # 获取nvidia-smi路径
    uuid_regex = re.compile(r" \(UUID: .+?\)")  # 编译UUID匹配正则
    rc, out, _ = run_lambda(smi + " -L")  # 执行nvidia-smi -L列出GPU
    if rc != 0:  # 如果执行失败
        return None  # 返回None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)  # 移除UUID以匿名化GPU信息


def get_running_cuda_version(run_lambda):
    """获取正在运行的CUDA版本。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        CUDA版本字符串，未找到则返回None。
    """
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")  # 从nvcc输出中提取CUDA版本


def get_cudnn_version(run_lambda):
    """获取cuDNN版本信息。

    返回libcudnn.so文件列表；由于难以确定实际使用的版本，
    可能返回多个候选文件路径。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        cuDNN库文件路径字符串，未找到则返回None。
    """
    if get_platform() == "win32":  # 如果是Windows平台
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")  # 获取系统根目录
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")  # 获取CUDA路径
        where_cmd = os.path.join(system_root, "System32", "where")  # where命令路径
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)  # 构建搜索cuDNN DLL的命令
    elif get_platform() == "darwin":  # 如果是macOS平台
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#install
        # https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installmac
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"  # 列出macOS上的cuDNN库文件
    else:  # Linux平台
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'  # 从ldconfig中查找cuDNN库
    rc, out, _ = run_lambda(cudnn_cmd)  # 执行查找命令
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):  # 如果输出为空或返回码异常
        l = os.environ.get("CUDNN_LIBRARY")  # 尝试从环境变量获取cuDNN库路径
        if l is not None and os.path.isfile(l):  # 如果环境变量指向有效文件
            return os.path.realpath(l)  # 返回真实路径
        return None  # 返回None
    files_set = set()  # 创建文件集合去重
    for fn in out.split("\n"):  # 遍历输出的每一行
        fn = os.path.realpath(fn)  # 解析符号链接获取真实路径
        if os.path.isfile(fn):  # 如果是有效文件
            files_set.add(fn)  # 添加到集合
    if not files_set:  # 如果集合为空
        return None  # 返回None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = sorted(files_set)  # 按字母顺序排序结果
    if len(files) == 1:  # 如果只有一个文件
        return files[0]  # 直接返回
    result = "\n".join(files)  # 将多个文件路径连接
    return "Probably one of the following:\n{}".format(result)  # 返回候选列表


def get_nvidia_smi():
    """获取nvidia-smi可执行文件的路径。

    在Windows上搜索多个可能的路径，在Linux上直接使用命令名。

    Returns:
        nvidia-smi可执行文件路径字符串。
    """
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"  # 默认nvidia-smi命令名
    if get_platform() == "win32":  # 如果是Windows平台
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")  # 获取系统根目录
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")  # 获取Program Files目录
        legacy_path = os.path.join(  # 旧版nvidia-smi路径
            program_files_root, "NVIDIA Corporation", "NVSMI", smi  # NVSMI目录下
        )
        new_path = os.path.join(system_root, "System32", smi)  # 新版nvidia-smi路径（System32下）
        smis = [new_path, legacy_path]  # 候选路径列表
        for candidate_smi in smis:  # 遍历候选路径
            if os.path.exists(candidate_smi):  # 如果路径存在
                smi = '"{}"'.format(candidate_smi)  # 用引号包裹路径
                break  # 找到后退出循环
    return smi  # 返回nvidia-smi路径


def get_rocm_version(run_lambda):
    """获取ROCm版本信息。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        ROCm版本字符串，如果不可用则返回None。
    """
    return run_and_parse_first_match(  # 从hipcc --version输出中提取HIP版本
        run_lambda, "hipcc --version", r"HIP version: (\S+)"  # 匹配HIP版本号
    )


def get_vllm_version():
    """获取vLLM版本信息。

    根据版本元组判断是否为开发构建，并格式化版本字符串。

    Returns:
        格式化的vLLM版本字符串。
    """
    from vllm import __version__, __version_tuple__  # 导入vLLM版本信息

    if __version__ == "dev":  # 如果是dev版本
        return "N/A (dev)"  # 返回N/A标识
    version_str = __version_tuple__[-1]  # 获取版本元组的最后一个元素
    if isinstance(version_str, str) and version_str.startswith("g"):  # 如果是以g开头的字符串（git哈希）
        # it's a dev build
        if "." in version_str:  # 如果包含点号（有本地修改的开发构建）
            # it's a dev build containing local changes
            git_sha = version_str.split(".")[0][1:]  # 提取git SHA（去掉前缀g）
            date = version_str.split(".")[-1][1:]  # 提取日期（去掉前缀d）
            return f"{__version__} (git sha: {git_sha}, date: {date})"  # 返回含SHA和日期的版本字符串
        else:  # 无本地修改的开发构建
            # it's a dev build without local changes
            git_sha = version_str[1:]  # type: ignore  # 提取git SHA
            return f"{__version__} (git sha: {git_sha})"  # 返回含SHA的版本字符串
    return __version__  # 返回正式版本号


def summarize_vllm_build_flags():
    """汇总vLLM的构建标志信息。

    检查CUDA架构列表和ROCm是否启用。

    Returns:
        包含CUDA架构和ROCm状态的字符串。
    """
    # This could be a static method if the flags are constant, or dynamic if you need to check environment variables, etc.
    return "CUDA Archs: {}; ROCm: {}".format(  # 格式化构建标志字符串
        os.environ.get("TORCH_CUDA_ARCH_LIST", "Not Set"),  # 获取CUDA架构列表，未设置则显示"Not Set"
        "Enabled" if os.environ.get("ROCM_HOME") else "Disabled",  # 根据ROCM_HOME判断ROCm是否启用
    )


def get_gpu_topo(run_lambda):
    """获取GPU拓扑信息。

    仅在Linux平台上可用，尝试使用nvidia-smi或rocm-smi获取。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        GPU拓扑信息字符串，不可用则返回None。
    """
    output = None  # 初始化输出为None

    if get_platform() == "linux":  # 如果是Linux平台
        output = run_and_read_all(run_lambda, "nvidia-smi topo -m")  # 尝试使用nvidia-smi获取拓扑
        if output is None:  # 如果nvidia-smi失败
            output = run_and_read_all(run_lambda, "rocm-smi --showtopo")  # 尝试使用rocm-smi获取拓扑

    return output  # 返回GPU拓扑信息


# example outputs of CPU infos
#  * linux
#    Architecture:            x86_64
#      CPU op-mode(s):        32-bit, 64-bit
#      Address sizes:         46 bits physical, 48 bits virtual
#      Byte Order:            Little Endian
#    CPU(s):                  128
#      On-line CPU(s) list:   0-31,64-95
#    Vendor ID:               GenuineIntel
#      Model name:            Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#        CPU family:          6
#        Model:               106
#        Thread(s) per core:  2
#        Core(s) per socket:  32
#        Socket(s):           2
#        Stepping:            6
#        BogoMIPS:            5799.78
#        Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr
#                             sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl
#                             xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq monitor ssse3 fma cx16
#                             pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand
#                             hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced
#                             fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap
#                             avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1
#                             xsaves wbnoinvd ida arat avx512vbmi pku ospke avx512_vbmi2 gfni vaes vpclmulqdq
#                             avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid md_clear flush_l1d arch_capabilities
#    Virtualization features:
#      Hypervisor vendor:     KVM
#      Virtualization type:   full
#    Caches (sum of all):
#      L1d:                   3 MiB (64 instances)
#      L1i:                   2 MiB (64 instances)
#      L2:                    80 MiB (64 instances)
#      L3:                    108 MiB (2 instances)
#    NUMA:
#      NUMA node(s):          2
#      NUMA node0 CPU(s):     0-31,64-95
#      NUMA node1 CPU(s):     32-63,96-127
#    Vulnerabilities:
#      Itlb multihit:         Not affected
#      L1tf:                  Not affected
#      Mds:                   Not affected
#      Meltdown:              Not affected
#      Mmio stale data:       Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
#      Retbleed:              Not affected
#      Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl and seccomp
#      Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
#      Spectre v2:            Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
#      Srbds:                 Not affected
#      Tsx async abort:       Not affected
#  * win32
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU0
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142
#
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU1
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142


def get_cpu_info(run_lambda):
    """获取CPU信息。

    根据不同平台使用不同的命令获取CPU详细信息：
    - Linux: 使用lscpu命令
    - Windows: 使用wmic命令
    - macOS: 使用sysctl命令

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        CPU信息字符串。
    """
    rc, out, err = 0, "", ""  # 初始化返回码、输出和错误
    if get_platform() == "linux":  # 如果是Linux平台
        rc, out, err = run_lambda("lscpu")  # 使用lscpu获取CPU信息
    elif get_platform() == "win32":  # 如果是Windows平台
        rc, out, err = run_lambda(  # 使用wmic获取CPU信息
            "wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID, \
        CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE"  # 获取多个CPU属性
        )
    elif get_platform() == "darwin":  # 如果是macOS平台
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")  # 使用sysctl获取CPU品牌字符串
    cpu_info = "None"  # 默认CPU信息为None
    if rc == 0:  # 如果命令执行成功
        cpu_info = out  # 使用标准输出
    else:  # 如果命令执行失败
        cpu_info = err  # 使用标准错误输出
    return cpu_info  # 返回CPU信息


def get_platform():
    """获取当前操作系统平台标识。

    Returns:
        平台标识字符串：'linux'、'win32'、'cygwin'、'darwin'或原始平台名。
    """
    if sys.platform.startswith("linux"):  # 如果是Linux系统
        return "linux"  # 返回linux
    elif sys.platform.startswith("win32"):  # 如果是Windows系统
        return "win32"  # 返回win32
    elif sys.platform.startswith("cygwin"):  # 如果是Cygwin环境
        return "cygwin"  # 返回cygwin
    elif sys.platform.startswith("darwin"):  # 如果是macOS系统
        return "darwin"  # 返回darwin
    else:  # 其他平台
        return sys.platform  # 返回原始平台标识


def get_mac_version(run_lambda):
    """获取macOS版本号。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        macOS版本字符串，未找到则返回None。
    """
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")  # 从sw_vers输出中提取版本号


def get_windows_version(run_lambda):
    """获取Windows版本信息。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        Windows版本字符串，未找到则返回None。
    """
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")  # 获取系统根目录
    wmic_cmd = os.path.join(system_root, "System32", "Wbem", "wmic")  # wmic命令路径
    findstr_cmd = os.path.join(system_root, "System32", "findstr")  # findstr命令路径
    return run_and_read_all(  # 执行wmic命令并过滤输出
        run_lambda, "{} os get Caption | {} /v Caption".format(wmic_cmd, findstr_cmd)  # 获取操作系统名称
    )


def get_lsb_version(run_lambda):
    """获取LSB（Linux标准基础）发行版描述信息。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        发行版描述字符串，未找到则返回None。
    """
    return run_and_parse_first_match(  # 从lsb_release输出中提取描述
        run_lambda, "lsb_release -a", r"Description:\t(.*)"  # 匹配Description字段
    )


def check_release_file(run_lambda):
    """从/etc/*-release文件中获取操作系统名称。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        操作系统名称字符串，未找到则返回None。
    """
    return run_and_parse_first_match(  # 从release文件中提取PRETTY_NAME
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'  # 匹配PRETTY_NAME字段
    )


def get_os(run_lambda):
    """获取操作系统详细信息。

    根据不同平台调用相应的函数获取操作系统名称和架构信息。

    Args:
        run_lambda: 执行命令的函数。

    Returns:
        操作系统信息字符串。
    """
    from platform import machine  # 导入machine函数获取CPU架构

    platform = get_platform()  # 获取平台标识

    if platform == "win32" or platform == "cygwin":  # 如果是Windows或Cygwin
        return get_windows_version(run_lambda)  # 返回Windows版本

    if platform == "darwin":  # 如果是macOS
        version = get_mac_version(run_lambda)  # 获取macOS版本
        if version is None:  # 如果获取失败
            return None  # 返回None
        return "macOS {} ({})".format(version, machine())  # 返回macOS版本和架构

    if platform == "linux":  # 如果是Linux
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)  # 尝试通过lsb_release获取
        if desc is not None:  # 如果获取成功
            return "{} ({})".format(desc, machine())  # 返回发行版名称和架构

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)  # 尝试从release文件获取
        if desc is not None:  # 如果获取成功
            return "{} ({})".format(desc, machine())  # 返回名称和架构

        return "{} ({})".format(platform, machine())  # 返回平台名和架构

    # Unknown platform
    return platform  # 返回未知平台标识


def get_python_platform():
    """获取Python平台详细信息。

    Returns:
        包含操作系统和架构信息的平台字符串。
    """
    import platform  # 导入platform模块

    return platform.platform()  # 返回平台详细信息


def get_libc_version():
    """获取libc库版本信息。

    仅在Linux平台上有意义。

    Returns:
        libc版本字符串，非Linux平台返回'N/A'。
    """
    import platform  # 导入platform模块

    if get_platform() != "linux":  # 如果不是Linux平台
        return "N/A"  # 返回N/A
    return "-".join(platform.libc_ver())  # 返回libc版本（如"glibc-2.31"）


def is_uv_venv():
    """检查当前虚拟环境是否由uv创建。

    通过检查UV环境变量或pyvenv.cfg文件中的uv标记来判断。

    Returns:
        如果是uv创建的虚拟环境则返回True，否则返回False。
    """
    if os.environ.get("UV"):  # 如果设置了UV环境变量
        return True  # 是uv环境
    pyvenv_cfg_path = os.path.join(sys.prefix, "pyvenv.cfg")  # pyvenv配置文件路径
    if os.path.exists(pyvenv_cfg_path):  # 如果配置文件存在
        with open(pyvenv_cfg_path, "r") as f:  # 打开配置文件
            return any(line.startswith("uv = ") for line in f)  # 检查是否包含uv标记
    return False  # 默认不是uv环境


def get_pip_packages(run_lambda, patterns=None):
    """获取匹配指定模式的pip包列表。

    也会找到通过conda安装的pytorch和numpy包。

    Args:
        run_lambda: 执行命令的函数。
        patterns: 要匹配的包名模式集合，默认使用DEFAULT_PIP_PATTERNS。

    Returns:
        包含pip版本和匹配包列表的元组 (pip_version, packages_string)。
    """
    if patterns is None:  # 如果未指定匹配模式
        patterns = DEFAULT_PIP_PATTERNS  # 使用默认模式

    def run_with_pip():
        """使用pip或uv pip获取包列表。"""
        try:  # 尝试检查pip是否可用
            import importlib.util  # 导入模块查找工具

            pip_spec = importlib.util.find_spec("pip")  # 查找pip模块
            pip_available = pip_spec is not None  # 判断pip是否可用
        except ImportError:  # 导入失败
            pip_available = False  # pip不可用

        if pip_available:  # 如果pip可用
            cmd = [sys.executable, "-mpip", "list", "--format=freeze"]  # 使用pip list命令
        elif is_uv_venv():  # 如果是uv虚拟环境
            print("uv is set")  # 打印提示信息
            cmd = ["uv", "pip", "list", "--format=freeze"]  # 使用uv pip list命令
        else:  # 都不可用
            raise RuntimeError(  # 抛出运行时错误
                "Could not collect pip list output (pip or uv module not available)"  # 提示pip和uv都不可用
            )

        out = run_and_read_all(run_lambda, cmd)  # 执行命令获取输出
        return "\n".join(  # 将匹配的行连接成字符串
            line for line in out.splitlines() if any(name in line for name in patterns)  # 过滤匹配的包
        )

    pip_version = "pip3" if sys.version[0] == "3" else "pip"  # 根据Python版本确定pip版本名称
    out = run_with_pip()  # 获取pip包列表
    return pip_version, out  # 返回pip版本和包列表


def get_cachingallocator_config():
    """获取PyTorch CUDA缓存分配器配置。

    Returns:
        PYTORCH_CUDA_ALLOC_CONF环境变量的值，未设置则返回空字符串。
    """
    ca_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")  # 获取缓存分配器配置
    return ca_config  # 返回配置


def get_cuda_module_loading_config():
    """获取CUDA模块加载配置。

    Returns:
        CUDA_MODULE_LOADING环境变量的值，CUDA不可用则返回'N/A'。
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():  # 如果torch和CUDA都可用
        torch.cuda.init()  # 初始化CUDA
        config = os.environ.get("CUDA_MODULE_LOADING", "")  # 获取CUDA模块加载配置
        return config  # 返回配置
    else:  # CUDA不可用
        return "N/A"  # 返回N/A


def is_xnnpack_available():
    """检查XNNPACK后端是否可用。

    Returns:
        XNNPACK可用状态字符串，torch不可用则返回'N/A'。
    """
    if TORCH_AVAILABLE:  # 如果torch可用
        import torch.backends.xnnpack  # 导入XNNPACK后端

        return str(torch.backends.xnnpack.enabled)  # type: ignore[attr-defined]  # 返回XNNPACK启用状态
    else:  # torch不可用
        return "N/A"  # 返回N/A


def get_env_vars():
    """获取与vLLM和PyTorch相关的环境变量。

    过滤掉包含敏感信息（如secret、token、api等）的环境变量，
    收集vLLM定义的环境变量和以特定前缀开头的环境变量。

    Returns:
        格式化的环境变量字符串。
    """
    env_vars = ""  # 初始化环境变量字符串
    secret_terms = ("secret", "token", "api", "access", "password")  # 敏感关键词列表
    report_prefix = (  # 需要报告的环境变量前缀列表
        "TORCH",  # PyTorch相关
        "NCCL",  # NCCL相关
        "PYTORCH",  # PyTorch相关
        "CUDA",  # CUDA相关
        "CUBLAS",  # cuBLAS相关
        "CUDNN",  # cuDNN相关
        "OMP_",  # OpenMP相关
        "MKL_",  # MKL相关
        "NVIDIA",  # NVIDIA相关
    )
    for k, v in os.environ.items():  # 遍历所有环境变量
        if any(term in k.lower() for term in secret_terms):  # 如果包含敏感关键词
            continue  # 跳过该变量
        if k in environment_variables:  # 如果是vLLM定义的环境变量
            env_vars = env_vars + "{}={}".format(k, v) + "\n"  # 添加到结果
        if k.startswith(report_prefix):  # 如果以指定前缀开头
            env_vars = env_vars + "{}={}".format(k, v) + "\n"  # 添加到结果

    return env_vars  # 返回环境变量字符串


def get_env_info():
    """收集完整的系统环境信息。

    汇总所有环境信息，包括PyTorch版本、CUDA配置、Python环境、
    GPU信息、已安装的包等，返回SystemEnv命名元组。

    Returns:
        包含所有系统环境信息的SystemEnv命名元组。
    """
    run_lambda = run  # 使用run函数作为命令执行器
    pip_version, pip_list_output = get_pip_packages(run_lambda)  # 获取pip包信息

    if TORCH_AVAILABLE:  # 如果torch可用
        version_str = torch.__version__  # 获取torch版本
        debug_mode_str = str(torch.version.debug)  # 获取调试模式状态
        cuda_available_str = str(torch.cuda.is_available())  # 获取CUDA可用状态
        cuda_version_str = torch.version.cuda  # 获取CUDA版本
        if (  # 检查是否为HIP/ROCm环境
            not hasattr(torch.version, "hip") or torch.version.hip is None  # 如果没有HIP版本（即CUDA环境）
        ):  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"  # HIP相关版本设为N/A
        else:  # HIP version（ROCm环境）

            def get_version_or_na(cfg, prefix):
                """从配置行中提取版本号，未找到则返回'N/A'。"""
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]  # 查找包含前缀的行并提取最后一个字段
                return _lst[0] if _lst else "N/A"  # 返回找到的版本或N/A

            cfg = torch._C._show_config().split("\n")  # 获取PyTorch配置信息并按行分割
            hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")  # 获取HIP运行时版本
            miopen_runtime_version = get_version_or_na(cfg, "MIOpen")  # 获取MIOpen版本
            cuda_version_str = "N/A"  # CUDA版本设为N/A
            hip_compiled_version = torch.version.hip  # 获取HIP编译版本
    else:  # torch不可用
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = "N/A"  # 所有版本设为N/A
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"  # HIP相关版本设为N/A

    sys_version = sys.version.replace("\n", " ")  # 获取Python版本并移除换行符

    conda_packages = get_conda_packages(run_lambda)  # 获取conda包信息

    rocm_version = get_rocm_version(run_lambda)  # 获取ROCm版本
    vllm_version = get_vllm_version()  # 获取vLLM版本
    vllm_build_flags = summarize_vllm_build_flags()  # 获取vLLM构建标志
    gpu_topo = get_gpu_topo(run_lambda)  # 获取GPU拓扑信息

    return SystemEnv(  # 返回系统环境信息命名元组
        torch_version=version_str,  # PyTorch版本
        is_debug_build=debug_mode_str,  # 调试构建状态
        python_version="{} ({}-bit runtime)".format(  # Python版本（含运行时位数）
            sys_version, sys.maxsize.bit_length() + 1  # 计算运行时位数
        ),
        python_platform=get_python_platform(),  # Python平台信息
        is_cuda_available=cuda_available_str,  # CUDA可用状态
        cuda_compiled_version=cuda_version_str,  # CUDA编译版本
        cuda_runtime_version=get_running_cuda_version(run_lambda),  # CUDA运行时版本
        cuda_module_loading=get_cuda_module_loading_config(),  # CUDA模块加载配置
        nvidia_gpu_models=get_gpu_info(run_lambda),  # NVIDIA GPU型号
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),  # NVIDIA驱动版本
        cudnn_version=get_cudnn_version(run_lambda),  # cuDNN版本
        hip_compiled_version=hip_compiled_version,  # HIP编译版本
        hip_runtime_version=hip_runtime_version,  # HIP运行时版本
        miopen_runtime_version=miopen_runtime_version,  # MIOpen运行时版本
        pip_version=pip_version,  # pip版本
        pip_packages=pip_list_output,  # pip包列表
        conda_packages=conda_packages,  # conda包列表
        os=get_os(run_lambda),  # 操作系统信息
        libc_version=get_libc_version(),  # libc版本
        gcc_version=get_gcc_version(run_lambda),  # GCC版本
        clang_version=get_clang_version(run_lambda),  # Clang版本
        cmake_version=get_cmake_version(run_lambda),  # CMake版本
        caching_allocator_config=get_cachingallocator_config(),  # 缓存分配器配置
        is_xnnpack_available=is_xnnpack_available(),  # XNNPACK可用状态
        cpu_info=get_cpu_info(run_lambda),  # CPU信息
        rocm_version=rocm_version,  # ROCm版本
        vllm_version=vllm_version,  # vLLM版本
        vllm_build_flags=vllm_build_flags,  # vLLM构建标志
        gpu_topo=gpu_topo,  # GPU拓扑
        env_vars=get_env_vars(),  # 环境变量
    )


env_info_fmt = """
==============================
        System Info
==============================
OS                           : {os}
GCC version                  : {gcc_version}
Clang version                : {clang_version}
CMake version                : {cmake_version}
Libc version                 : {libc_version}

==============================
       PyTorch Info
==============================
PyTorch version              : {torch_version}
Is debug build               : {is_debug_build}
CUDA used to build PyTorch   : {cuda_compiled_version}
ROCM used to build PyTorch   : {hip_compiled_version}

==============================
      Python Environment
==============================
Python version               : {python_version}
Python platform              : {python_platform}

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : {is_cuda_available}
CUDA runtime version         : {cuda_runtime_version}
CUDA_MODULE_LOADING set to   : {cuda_module_loading}
GPU models and configuration : {nvidia_gpu_models}
Nvidia driver version        : {nvidia_driver_version}
cuDNN version                : {cudnn_version}
HIP runtime version          : {hip_runtime_version}
MIOpen runtime version       : {miopen_runtime_version}
Is XNNPACK available         : {is_xnnpack_available}

==============================
          CPU Info
==============================
{cpu_info}

==============================
Versions of relevant libraries
==============================
{pip_packages}
{conda_packages}
""".strip()  # 环境信息格式化模板字符串（去除首尾空白）

# both the above code and the following code use `strip()` to
# remove leading/trailing whitespaces, so we need to add a newline
# in between to separate the two sections
env_info_fmt += "\n\n"  # 在两个模板部分之间添加换行分隔

env_info_fmt += """
==============================
         vLLM Info
==============================
ROCM Version                 : {rocm_version}
vLLM Version                 : {vllm_version}
vLLM Build Flags:
  {vllm_build_flags}
GPU Topology:
  {gpu_topo}

==============================
     Environment Variables
==============================
{env_vars}
""".strip()  # vLLM信息格式化模板字符串（去除首尾空白）


def pretty_str(envinfo):
    """将系统环境信息格式化为可读的字符串。

    对环境信息进行后处理，包括替换None值、布尔值格式化、
    添加包管理器前缀等。

    Args:
        envinfo: SystemEnv命名元组实例。

    Returns:
        格式化后的环境信息字符串。
    """
    def replace_nones(dct, replacement="Could not collect"):
        """将字典中的None值替换为指定的替换文本。"""
        for key in dct.keys():  # 遍历所有键
            if dct[key] is not None:  # 如果值不为None
                continue  # 跳过
            dct[key] = replacement  # 替换None值
        return dct  # 返回处理后的字典

    def replace_bools(dct, true="Yes", false="No"):
        """将字典中的布尔值替换为'Yes'/'No'字符串。"""
        for key in dct.keys():  # 遍历所有键
            if dct[key] is True:  # 如果值为True
                dct[key] = true  # 替换为"Yes"
            elif dct[key] is False:  # 如果值为False
                dct[key] = false  # 替换为"No"
        return dct  # 返回处理后的字典

    def prepend(text, tag="[prepend]"):
        """在文本的每一行前面添加指定的标签前缀。"""
        lines = text.split("\n")  # 按行分割文本
        updated_lines = [tag + line for line in lines]  # 为每行添加前缀
        return "\n".join(updated_lines)  # 重新连接成字符串

    def replace_if_empty(text, replacement="No relevant packages"):
        """如果文本为空则替换为默认提示信息。"""
        if text is not None and len(text) == 0:  # 如果文本非None但为空
            return replacement  # 返回替换文本
        return text  # 否则返回原文本

    def maybe_start_on_next_line(string):
        """如果字符串是多行的，在前面添加换行符。"""
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:  # 如果字符串包含多行
            return "\n{}\n".format(string)  # 在前后添加换行符
        return string  # 否则原样返回

    mutable_dict = envinfo._asdict()  # 将命名元组转换为可变字典

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(  # 处理多行GPU型号信息
        envinfo.nvidia_gpu_models  # NVIDIA GPU型号
    )

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [  # 动态CUDA字段列表
        "cuda_runtime_version",  # CUDA运行时版本
        "nvidia_gpu_models",  # NVIDIA GPU型号
        "nvidia_driver_version",  # NVIDIA驱动版本
    ]
    all_cuda_fields = dynamic_cuda_fields + ["cudnn_version"]  # 所有CUDA相关字段
    all_dynamic_cuda_fields_missing = all(  # 检查所有动态CUDA字段是否都为None
        mutable_dict[field] is None for field in dynamic_cuda_fields  # 遍历检查
    )
    if (  # 如果机器没有CUDA
        TORCH_AVAILABLE  # torch可用
        and not torch.cuda.is_available()  # 但CUDA不可用
        and all_dynamic_cuda_fields_missing  # 且所有动态CUDA字段为空
    ):
        for field in all_cuda_fields:  # 遍历所有CUDA字段
            mutable_dict[field] = "No CUDA"  # 设置为"No CUDA"
        if envinfo.cuda_compiled_version is None:  # 如果CUDA编译版本也为空
            mutable_dict["cuda_compiled_version"] = "None"  # 设为"None"字符串

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)  # 替换布尔值

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)  # 替换None值

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])  # 处理空的pip包列表
    mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])  # 处理空的conda包列表

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:  # 如果有pip包信息
        mutable_dict["pip_packages"] = prepend(  # 添加pip版本前缀
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)  # 如"[pip3] "
        )
    if mutable_dict["conda_packages"]:  # 如果有conda包信息
        mutable_dict["conda_packages"] = prepend(  # 添加conda前缀
            mutable_dict["conda_packages"], "[conda] "  # 添加"[conda] "前缀
        )
    mutable_dict["cpu_info"] = envinfo.cpu_info  # 设置CPU信息
    return env_info_fmt.format(**mutable_dict)  # 使用模板格式化并返回


def get_pretty_env_info():
    """获取格式化的系统环境信息字符串。

    Returns:
        格式化后的完整环境信息字符串。
    """
    return pretty_str(get_env_info())  # 收集并格式化环境信息


def main():
    """主函数，收集并打印系统环境信息。

    如果检测到崩溃转储文件，还会在标准错误输出中打印提示信息。
    """
    print("Collecting environment information...")  # 打印收集信息提示
    output = get_pretty_env_info()  # 获取格式化的环境信息
    print(output)  # 打印环境信息

    if (  # 检查是否有崩溃处理器
        TORCH_AVAILABLE  # torch可用
        and hasattr(torch, "utils")  # 有utils模块
        and hasattr(torch.utils, "_crash_handler")  # 有崩溃处理器
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR  # 获取小型转储目录
        if sys.platform == "linux" and os.path.exists(minidump_dir):  # 如果是Linux且目录存在
            dumps = [  # 获取所有转储文件
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)  # 构建完整路径
            ]
            latest = max(dumps, key=os.path.getctime)  # 获取最新的转储文件
            ctime = os.path.getctime(latest)  # 获取创建时间
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(  # 格式化时间戳
                "%Y-%m-%d %H:%M:%S"  # 时间格式
            )
            msg = (  # 构建提示消息
                "\n*** Detected a minidump at {} created on {}, ".format(  # 检测到转储文件的提示
                    latest, creation_time  # 文件路径和创建时间
                )
                + "if this is related to your bug please include it when you file a report ***"  # 建议包含在bug报告中
            )
            print(msg, file=sys.stderr)  # 输出到标准错误


if __name__ == "__main__":  # 如果作为主模块运行
    main()  # 执行主函数
