# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

import functools  # 导入functools模块，用于高阶函数和缓存装饰器
import json  # 导入json模块，用于JSON解析
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于操作系统接口和环境变量访问
import sys  # 导入sys模块，用于系统相关参数和函数
import tempfile  # 导入tempfile模块，用于获取临时目录路径
import uuid  # 导入uuid模块，用于生成唯一标识符
from collections.abc import Callable  # 从collections.abc导入Callable类型，用于类型标注
from typing import TYPE_CHECKING, Any, Literal  # 从typing导入类型检查相关工具

if TYPE_CHECKING:  # 仅在类型检查时执行以下代码块（运行时不执行）
    VLLM_HOST_IP: str = ""  # 分布式环境中当前节点的IP地址
    VLLM_PORT: int | None = None  # vLLM通信端口号
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()  # RPC通信的基础路径，默认为临时目录
    VLLM_USE_MODELSCOPE: bool = False  # 是否使用ModelScope替代Hugging Face Hub加载模型
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60  # 环形缓冲区满时的警告日志间隔（秒）
    VLLM_NCCL_SO_PATH: str | None = None  # NCCL动态库文件路径
    LD_LIBRARY_PATH: str | None = None  # 动态链接库搜索路径
    VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE: int = 256  # ROCm下休眠内存分配的块大小（MB）
    LOCAL_RANK: int = 0  # 分布式设置中进程的本地rank，用于确定GPU设备ID
    CUDA_VISIBLE_DEVICES: str | None = None  # 可见的CUDA设备列表
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60  # 引擎每次迭代的超时时间（秒）
    VLLM_ENGINE_READY_TIMEOUT_S: int = 600  # 引擎核心启动就绪的超时时间（秒）
    VLLM_API_KEY: str | None = None  # vLLM API服务器的API密钥
    VLLM_DEBUG_LOG_API_SERVER_RESPONSE: bool = False  # 是否记录API服务器响应的调试日志
    S3_ACCESS_KEY_ID: str | None = None  # S3访问密钥ID，用于tensorizer从S3加载模型
    S3_SECRET_ACCESS_KEY: str | None = None  # S3秘密访问密钥
    S3_ENDPOINT_URL: str | None = None  # S3端点URL
    VLLM_MODEL_REDIRECT_PATH: str | None = None  # 模型重定向路径配置文件
    VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")  # vLLM缓存文件根目录
    VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")  # vLLM配置文件根目录
    VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"  # 使用统计数据上报服务器地址
    VLLM_NO_USAGE_STATS: bool = False  # 是否禁用使用统计数据收集
    VLLM_DO_NOT_TRACK: bool = False  # 是否禁止跟踪（Do Not Track标志）
    VLLM_USAGE_SOURCE: str = "production"  # 使用来源标识
    VLLM_CONFIGURE_LOGGING: bool = True  # 是否由vLLM配置日志系统
    VLLM_LOGGING_LEVEL: str = "INFO"  # 默认日志记录级别
    VLLM_LOGGING_PREFIX: str = ""  # 日志消息前缀
    VLLM_LOGGING_STREAM: str = "ext://sys.stdout"  # 默认日志输出流
    VLLM_LOGGING_CONFIG_PATH: str | None = None  # 日志配置文件路径
    VLLM_LOGGING_COLOR: str = "auto"  # 日志颜色输出控制（auto/1/0）
    NO_COLOR: bool = False  # 标准Unix禁用ANSI颜色码标志
    VLLM_LOG_STATS_INTERVAL: float = 10.0  # 统计日志记录间隔（秒）
    VLLM_TRACE_FUNCTION: int = 0  # 是否启用函数调用追踪（调试用）
    VLLM_USE_FLASHINFER_SAMPLER: bool | None = None  # 是否使用FlashInfer采样器
    VLLM_PP_LAYER_PARTITION: str | None = None  # 流水线并行层分区策略
    VLLM_CPU_KVCACHE_SPACE: int | None = 0  # CPU后端KV缓存空间大小（GB）
    VLLM_CPU_OMP_THREADS_BIND: str = "auto"  # CPU后端OpenMP线程绑定的CPU核心ID
    VLLM_CPU_NUM_OF_RESERVED_CPU: int | None = None  # CPU后端预留的CPU核心数量
    VLLM_CPU_SGL_KERNEL: bool = False  # CPU后端是否使用SGL内核（小批量优化）
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, "xla_cache")  # XLA持久化缓存目录路径
    VLLM_XLA_CHECK_RECOMPILATION: bool = False  # 是否在每步执行后检查XLA重编译
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl", "shm"] = "auto"  # Ray编译DAG的通信通道类型
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False  # 是否启用Ray编译DAG的GPU通信重叠
    VLLM_USE_RAY_WRAPPED_PP_COMM: bool = True  # 是否使用Ray封装的流水线并行通信器
    VLLM_XLA_USE_SPMD: bool = False  # 是否为TPU后端启用SPMD模式
    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"  # 工作进程的多进程创建方式
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, "assets")  # 下载资源的缓存路径
    VLLM_ASSETS_CACHE_MODEL_CLEAN: bool = False  # 是否清理资源缓存中的模型文件
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5  # 多模态模型获取图片的超时时间（秒）
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30  # 多模态模型获取视频的超时时间（秒）
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10  # 多模态模型获取音频的超时时间（秒）
    VLLM_MEDIA_URL_ALLOW_REDIRECTS: bool = True  # 是否允许媒体URL的HTTP重定向
    VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8  # 媒体字节加载线程池的最大工作线程数
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25  # 单个音频文件的最大文件大小（MB）
    VLLM_VIDEO_LOADER_BACKEND: str = "opencv"  # 视频IO后端（opencv或identity）
    VLLM_MEDIA_CONNECTOR: str = "http"  # 媒体连接器实现（默认HTTP）
    VLLM_MM_HASHER_ALGORITHM: str = "blake3"  # 多模态内容哈希算法
    VLLM_TARGET_DEVICE: str = "cuda"  # vLLM目标设备（cuda/rocm/cpu）
    VLLM_MAIN_CUDA_VERSION: str = "12.9"  # vLLM主要CUDA版本号
    VLLM_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"  # PyTorch float32矩阵乘法精度模式
    MAX_JOBS: str | None = None  # 并行编译任务的最大数量
    NVCC_THREADS: str | None = None  # nvcc使用的线程数
    VLLM_USE_PRECOMPILED: bool = False  # 是否使用预编译的二进制文件
    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX: bool = False  # 是否跳过版本字符串中的+precompiled后缀
    VLLM_DOCKER_BUILD_CONTEXT: bool = False  # 标记setup.py是否在Docker构建上下文中运行
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False  # 引擎崩溃后API服务器是否保持存活
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None  # CMake构建类型
    VERBOSE: bool = False  # 是否在安装时打印详细日志
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False  # 是否允许最大序列长度超过模型配置值
    VLLM_RPC_TIMEOUT: int = 10000  # ms  # ZMQ客户端等待后端响应的超时时间（毫秒）
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds  # HTTP连接保活超时时间（秒）
    VLLM_PLUGINS: list[str] | None = None  # 要加载的插件名称列表
    VLLM_LORA_RESOLVER_CACHE_DIR: str | None = None  # LoRA适配器本地缓存目录
    VLLM_LORA_RESOLVER_HF_REPO_LIST: str | None = None  # 包含LoRA适配器的远程HF仓库列表
    VLLM_USE_AOT_COMPILE: bool = False  # 是否启用AOT（提前）编译
    VLLM_USE_BYTECODE_HOOK: bool = True  # 是否在TorchCompile包装器中启用字节码钩子
    VLLM_FORCE_AOT_LOAD: bool = False  # 是否强制从磁盘加载AOT编译模型
    VLLM_USE_MEGA_AOT_ARTIFACT: bool = False  # 是否启用从缓存的独立编译产物直接加载编译模型
    VLLM_USE_TRITON_AWQ: bool = False  # 是否使用Triton实现的AWQ
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False  # 是否允许运行时加载或卸载LoRA适配器
    VLLM_SKIP_P2P_CHECK: bool = False  # 是否跳过P2P连接检查
    VLLM_DISABLED_KERNELS: list[str] = []  # 应禁用的量化内核列表
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE: bool = True  # 是否启用FLA打包循环解码
    VLLM_DISABLE_PYNCCL: bool = False  # 是否禁用pynccl（改用torch.distributed）
    VLLM_USE_OINK_OPS: bool = False  # 是否启用外部Oink自定义算子
    VLLM_ROCM_USE_AITER: bool = False  # 是否启用ROCm的aiter算子（父开关）
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False  # 是否使用aiter分页注意力
    VLLM_ROCM_USE_AITER_LINEAR: bool = True  # 是否使用aiter线性层算子
    VLLM_ROCM_USE_AITER_MOE: bool = True  # 是否使用aiter MoE算子
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True  # 是否使用aiter RMS归一化算子
    VLLM_ROCM_USE_AITER_MLA: bool = True  # 是否使用aiter MLA算子
    VLLM_ROCM_USE_AITER_MHA: bool = True  # 是否使用aiter MHA算子
    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: bool = False  # 是否使用aiter FP4汇编GEMM
    VLLM_ROCM_USE_AITER_TRITON_ROPE: bool = False  # 是否使用aiter Triton ROPE
    VLLM_ROCM_USE_AITER_FP8BMM: bool = True  # 是否使用aiter Triton FP8批量矩阵乘法内核
    VLLM_ROCM_USE_AITER_FP4BMM: bool = True  # 是否使用aiter Triton FP4批量矩阵乘法内核
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False  # 是否使用aiter Triton统一注意力（V1）
    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False  # 是否使用aiter融合共享专家算子
    VLLM_ROCM_USE_AITER_TRITON_GEMM: bool = True  # 是否使用aiter Triton GEMM内核
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True  # 是否使用ROCm瘦GEMM内核
    VLLM_ROCM_FP8_PADDING: bool = True  # 是否对ROCm的FP8权重进行256字节填充
    VLLM_ROCM_MOE_PADDING: bool = True  # 是否对MoE内核的权重进行填充
    VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True  # 是否使用MI3*卡的自定义分页注意力内核
    VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool = False  # 是否使用混洗的KV缓存布局
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True  # 是否在V1代码路径中为LLM启用多进程
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1  # 批大小日志记录间隔（秒，-1表示禁用）
    VLLM_DISABLE_COMPILE_CACHE: bool = False  # 是否禁用编译缓存
    Q_SCALE_CONSTANT: int = 200  # FP8 KV缓存动态查询缩放因子的除数
    K_SCALE_CONSTANT: int = 200  # FP8 KV缓存动态键缩放因子的除数
    V_SCALE_CONSTANT: int = 100  # FP8 KV缓存动态值缩放因子的除数
    VLLM_SERVER_DEV_MODE: bool = False  # 是否启用开发模式（额外调试端点）
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128  # V1 AsyncLLM中每个异步任务处理的最大请求数
    VLLM_MLA_DISABLE: bool = False  # 是否禁用MLA注意力优化
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0  # Ray中每个worker分配的GPU数量
    VLLM_RAY_BUNDLE_INDICES: str = ""  # Ray bundle索引，控制每个worker使用的具体索引
    VLLM_CUDART_SO_PATH: str | None = None  # CUDA运行时动态库路径
    VLLM_DP_RANK: int = 0  # 数据并行中的进程rank
    VLLM_DP_RANK_LOCAL: int = -1  # 数据并行中的本地rank（默认跟随VLLM_DP_RANK）
    VLLM_DP_SIZE: int = 1  # 数据并行的世界大小
    VLLM_USE_STANDALONE_COMPILE: bool = True  # 是否启用Inductor独立编译功能
    VLLM_ENABLE_PREGRAD_PASSES: bool = False  # 是否启用Inductor的预梯度pass
    VLLM_DP_MASTER_IP: str = ""  # 数据并行中主节点的IP地址
    VLLM_DP_MASTER_PORT: int = 0  # 数据并行中主节点的端口
    VLLM_MOE_DP_CHUNK_SIZE: int = 256  # MoE数据并行/专家并行的token分发量子大小
    VLLM_ENABLE_MOE_DP_CHUNK: bool = True  # 是否启用MoE数据并行分块
    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False  # 数据并行虚拟运行时是否随机化输入
    VLLM_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"] = "strict"  # Ray数据并行rank打包策略
    VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY: str = ""  # 复制到Ray worker的额外环境变量前缀
    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""  # 复制到Ray worker的额外环境变量名
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False  # 是否在gptq/awq Marlin内核中使用atomicAdd归约
    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None  # Marlin内核的激活数据类型
    VLLM_MXFP4_USE_MARLIN: bool | None = None  # 是否在mxfp4量化方法中使用Marlin内核
    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool = False  # 是否使用DeepEPLL的NVFP4量化分发内核
    VLLM_V1_USE_OUTLINES_CACHE: bool = False  # 是否为V1启用outlines磁盘缓存
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0  # TPU前向传播的填充桶间隔
    VLLM_TPU_MOST_MODEL_LEN: int | None = None  # TPU最大模型长度
    VLLM_TPU_USING_PATHWAYS: bool = False  # 是否使用Pathways
    VLLM_USE_DEEP_GEMM: bool = True  # 是否允许使用DeepGemm内核
    VLLM_MOE_USE_DEEP_GEMM: bool = True  # 是否专门为MoE融合操作使用DeepGemm
    VLLM_USE_DEEP_GEMM_E8M0: bool = True  # Blackwell GPU上DeepGEMM是否使用E8M0缩放
    VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES: bool = True  # DeepGEMM是否创建TMA对齐的缩放张量
    VLLM_DEEP_GEMM_WARMUP: Literal[  # DeepGemm内核JIT编译预热策略
        "skip",  # 跳过预热
        "full",  # 完整预热所有可能的GEMM形状
        "relax",  # 基于启发式选择GEMM形状进行预热
    ] = "relax"  # 默认使用启发式预热
    VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool = True  # 是否使用融合的分组topk用于MoE专家选择
    VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = True  # 是否使用FlashInfer FP8块缩放GEMM
    VLLM_USE_FLASHINFER_MOE_FP16: bool = False  # 是否使用FlashInfer BF16 MoE内核
    VLLM_USE_FLASHINFER_MOE_FP8: bool = False  # 是否使用FlashInfer FP8 MoE内核
    VLLM_USE_FLASHINFER_MOE_FP4: bool = False  # 是否使用FlashInfer NVFP4 MoE内核
    VLLM_USE_FLASHINFER_MOE_INT4: bool = False  # 是否使用FlashInfer MxInt4 MoE内核
    VLLM_FLASHINFER_MOE_BACKEND: Literal["throughput", "latency", "masked_gemm"] = (  # FlashInfer MoE后端选择
        "latency"  # 默认使用低延迟模式
    )
    VLLM_FLASHINFER_ALLREDUCE_BACKEND: Literal["auto", "trtllm", "mnnvl"] = "trtllm"  # FlashInfer融合allreduce后端
    VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024  # FlashInfer工作区缓冲区大小（字节）
    VLLM_XGRAMMAR_CACHE_MB: int = 0  # xgrammar编译器缓存大小（MB）
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256  # msgspec零拷贝序列化/反序列化的张量大小阈值
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False  # 是否允许使用pickle进行不安全序列化
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION: bool = False  # 是否禁用内部请求ID的随机后缀
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"  # NIXL远程代理握手的IP地址
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600  # NIXL远程代理握手的端口
    VLLM_MOONCAKE_BOOTSTRAP_PORT: int = 8998  # Mooncake远程代理握手的端口
    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840  # NVFP4 MoE CUTLASS内核每个专家支持的最大token数
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1  # 工具解析插件的正则表达式超时时间（秒）
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16  # RPC消息队列的最大块大小（MB）
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300  # execute_model RPC调用的超时时间（秒）
    VLLM_KV_CACHE_LAYOUT: Literal["NHD", "HND"] | None = None  # KV缓存布局格式
    VLLM_COMPUTE_NANS_IN_LOGITS: bool = False  # 是否检查生成的logits中是否包含NaN
    VLLM_USE_NVFP4_CT_EMULATIONS: bool = False  # 是否为低于SM100的机器使用NVFP4仿真
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: Literal[  # ROCm MI3*卡自定义快速allreduce量化级别
        "FP", "INT8", "INT6", "INT4", "NONE"  # 可选：全精度、INT8、INT6、INT4、无量化
    ] = "NONE"  # 默认不启用
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True  # 快速allreduce是否将BF16转换为FP16
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None  # 快速allreduce允许的最大数据大小（MB）
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 480  # NixlConnector中KV缓存自动清除的超时时间（秒）
    VLLM_MORIIO_CONNECTOR_READ_MODE: bool = False  # Mori-IO连接器的读取模式控制
    VLLM_MORIIO_QP_PER_TRANSFER: int = 1  # Mori-IO连接器每次传输的队列对（QP）数量
    VLLM_MORIIO_POST_BATCH_SIZE: int = -1  # Mori-IO连接器的后处理批大小
    VLLM_MORIIO_NUM_WORKERS: int = 1  # Mori-IO连接器的工作线程数
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: int = 480  # MooncakeConnector在PD分离设置中的超时时间（秒）
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False  # 是否在CUDA图捕获期间允许GC运行
    VLLM_LOOPBACK_IP: str = ""  # 强制设置环回IP地址
    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = True  # 是否允许混合KV缓存管理器的分块局部注意力
    VLLM_ENABLE_RESPONSES_API_STORE: bool = False  # 是否启用OpenAI Responses API的存储选项
    VLLM_NVFP4_GEMM_BACKEND: str | None = None  # NVFP4 GEMM后端选择
    VLLM_HAS_FLASHINFER_CUBIN: bool = False  # 是否已预下载FlashInfer cubin文件
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: bool = False  # 是否使用FlashInfer MXFP8激活x MXFP4权重的MoE后端
    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: bool = False  # 是否使用FlashInfer BF16激活x MXFP4权重的MoE后端
    VLLM_ROCM_FP8_MFMA_PAGE_ATTN: bool = False  # 是否在ROCm分页注意力中使用FP8 MFMA
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: bool = False  # 是否使用FlashInfer CUTLASS后端的MXFP8x MXFP4 MoE
    VLLM_ALLREDUCE_USE_SYMM_MEM: bool = True  # 是否使用PyTorch对称内存进行allreduce
    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False  # 是否使用FlashInfer进行allreduce
    VLLM_TUNED_CONFIG_FOLDER: str | None = None  # 自定义调优配置文件夹路径
    VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str] = set()  # MCP工具的系统工具标签集合
    VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT: bool = False  # 是否为非harmony模型启用MCP工具调用
    VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool = False  # 是否在系统消息中注入harmony指令
    VLLM_SYSTEM_START_DATE: str | None = None  # Harmony系统消息中固定的会话起始日期
    VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool = False  # 工具调用JSON解析失败时是否自动重试
    VLLM_CUSTOM_SCOPES_FOR_PROFILING: bool = False  # 是否添加自定义性能分析作用域
    VLLM_NVTX_SCOPES_FOR_PROFILING: bool = False  # 是否添加NVTX性能分析作用域
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True  # KV缓存事件中块哈希是否使用64位整数表示
    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "VLLM_OBJECT_STORAGE_SHM_BUFFER"  # 对象存储使用的共享内存缓冲区名称
    VLLM_DEEPEP_BUFFER_SIZE_MB: int = 1024  # DeepEP使用的NVL和RDMA缓冲区大小（MB）
    VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool = False  # DeepEP高吞吐模式是否强制使用节点内核进行节点间通信
    VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL: bool = False  # DeepEP低延迟模式是否使用MNNVL进行节点间通信
    VLLM_DBO_COMM_SMS: int = 20  # DBO中分配给通信内核的SM数量
    VLLM_PATTERN_MATCH_DEBUG: str | None = None  # 自定义pass中调试模式匹配的fx.Node名称
    VLLM_DEBUG_DUMP_PATH: str | None = None  # fx图转储的目标目录
    VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = True  # 是否启用Inductor的max_autotune
    VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = True  # 是否启用Inductor的坐标下降调优
    VLLM_USE_NCCL_SYMM_MEM: bool = False  # 是否启用NCCL对称内存分配和注册
    VLLM_NCCL_INCLUDE_PATH: str | None = None  # NCCL头文件路径
    VLLM_USE_FBGEMM: bool = False  # 是否启用FBGemm内核
    VLLM_GC_DEBUG: str = ""  # GC调试配置（0=禁用，1=启用，JSON格式可配置详情）
    VLLM_DEBUG_WORKSPACE: bool = False  # 是否调试工作区分配的resize操作日志
    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool = False  # 是否禁用共享专家的并行CUDA流执行
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int = 256  # 使用独立流运行共享专家的token数量阈值
    VLLM_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"] = "binary"  # torch.compile缓存产物的保存格式
    VLLM_USE_V2_MODEL_RUNNER: bool = False  # 是否启用v2模型运行器
    VLLM_LOG_MODEL_INSPECTION: bool = False  # 是否在模型加载后记录模型检查日志
    VLLM_DEBUG_MFU_METRICS: bool = False  # 是否启用MFU指标调试日志
    VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY: bool = False  # 是否禁用CPU卸载时的pin memory
    VLLM_WEIGHT_OFFLOADING_DISABLE_UVA: bool = False  # 是否禁用CPU卸载时的UVA（统一虚拟寻址）
    VLLM_DISABLE_LOG_LOGO: bool = False  # 是否禁用服务器启动时的vLLM Logo日志
    VLLM_LORA_DISABLE_PDL: bool = False  # 是否禁用LoRA的PDL（SM100上Triton编译失败的临时方案）
    VLLM_ENABLE_CUDA_COMPATIBILITY: bool = False  # 是否启用CUDA兼容模式（旧驱动版本适配）
    VLLM_CUDA_COMPATIBILITY_PATH: str | None = None  # CUDA兼容库路径
    VLLM_ELASTIC_EP_SCALE_UP_LAUNCH: bool = False  # 是否为弹性EP启动扩容引擎
    VLLM_ELASTIC_EP_DRAIN_REQUESTS: bool = False  # 弹性EP缩容前是否等待所有请求排空
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = False  # 内存分析期间是否估算CUDA图内存使用
    VLLM_NIXL_EP_MAX_NUM_RANKS: int = 32  # NIXL EP最大rank数量


def get_default_cache_root():
    """获取默认缓存根目录路径。

    优先使用XDG_CACHE_HOME环境变量，否则默认为~/.cache。
    """
    return os.getenv(  # 获取环境变量XDG_CACHE_HOME的值
        "XDG_CACHE_HOME",  # 环境变量名
        os.path.join(os.path.expanduser("~"), ".cache"),  # 默认值：用户主目录下的.cache
    )


def get_default_config_root():
    """获取默认配置根目录路径。

    优先使用XDG_CONFIG_HOME环境变量，否则默认为~/.config。
    """
    return os.getenv(  # 获取环境变量XDG_CONFIG_HOME的值
        "XDG_CONFIG_HOME",  # 环境变量名
        os.path.join(os.path.expanduser("~"), ".config"),  # 默认值：用户主目录下的.config
    )


def maybe_convert_int(value: str | None) -> int | None:
    """将字符串值转换为整数，如果值为None则返回None。

    Args:
        value: 待转换的字符串值或None

    Returns:
        转换后的整数值或None
    """
    if value is None:  # 如果值为None
        return None  # 直接返回None
    return int(value)  # 否则将字符串转换为整数返回


def maybe_convert_bool(value: str | None) -> bool | None:
    """将字符串值转换为布尔值，如果值为None则返回None。

    Args:
        value: 待转换的字符串值或None

    Returns:
        转换后的布尔值或None
    """
    if value is None:  # 如果值为None
        return None  # 直接返回None
    return bool(int(value))  # 先转为整数再转为布尔值


def disable_compile_cache() -> bool:
    """检查是否禁用了编译缓存。

    Returns:
        如果VLLM_DISABLE_COMPILE_CACHE设置为非零值则返回True
    """
    return bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0")))  # 从环境变量获取并转换为布尔值


def use_aot_compile() -> bool:
    """判断是否应启用AOT（提前）编译。

    根据PyTorch版本、编译缓存是否禁用以及是否处于batch_invariant模式来决定。

    Returns:
        是否启用AOT编译的布尔值
    """
    from vllm.model_executor.layers.batch_invariant import (  # 延迟导入batch_invariant检查函数
        vllm_is_batch_invariant,
    )
    from vllm.utils.torch_utils import is_torch_equal_or_newer  # 延迟导入PyTorch版本检查函数

    default_value = (  # 根据条件确定默认值
        "1"  # 如果PyTorch >= 2.10.0且未禁用编译缓存，默认启用
        if is_torch_equal_or_newer("2.10.0") and not disable_compile_cache()
        else "0"  # 否则默认禁用
    )

    return (  # 返回最终判断结果
        not vllm_is_batch_invariant()  # 不处于batch_invariant模式
        and os.environ.get("VLLM_USE_AOT_COMPILE", default_value) == "1"  # 且环境变量启用
    )


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    创建一个lambda函数，用于验证环境变量的值是否在允许的选项列表中。

    Args:
        env_name: Name of the environment variable（环境变量名称）
        default: Default value if not set (can be None)（未设置时的默认值，可以为None）
        choices: List of valid string options or callable that returns list（有效选项列表或返回列表的可调用对象）
        case_sensitive: Whether validation should be case sensitive（验证是否区分大小写）

    Returns:
        Lambda function for environment_variables dict（用于environment_variables字典的Lambda函数）
    """

    def _get_validated_env() -> str | None:
        """获取并验证环境变量值的内部函数。"""
        value = os.getenv(env_name)  # 从环境中获取变量值
        if value is None:  # 如果环境变量未设置
            return default  # 返回默认值

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices  # 如果choices是可调用对象则调用获取实际选项列表

        if not case_sensitive:  # 如果不区分大小写
            check_value = value.lower()  # 将值转为小写进行比较
            check_choices = [choice.lower() for choice in actual_choices]  # 将所有选项转为小写
        else:  # 区分大小写
            check_value = value  # 直接使用原始值
            check_choices = actual_choices  # 直接使用原始选项列表

        if check_value not in check_choices:  # 如果值不在允许的选项中
            raise ValueError(  # 抛出值错误异常
                f"Invalid value '{value}' for {env_name}. "  # 无效值提示
                f"Valid options: {actual_choices}."  # 显示有效选项
            )

        return value  # 返回验证通过的值

    return _get_validated_env  # 返回验证函数


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """
    Create a lambda that validates environment variable
    containing comma-separated values against allowed choices

    创建一个lambda函数，用于验证包含逗号分隔值的环境变量是否在允许的选项列表中。

    Args:
        env_name: Name of the environment variable（环境变量名称）
        default: Default list of values if not set（未设置时的默认值列表）
        choices: List of valid string options or callable that returns list（有效选项列表或返回列表的可调用对象）
        case_sensitive: Whether validation should be case sensitive（验证是否区分大小写）

    Returns:
        Lambda function for environment_variables
        dict that returns list of strings（用于environment_variables字典的Lambda函数，返回字符串列表）
    """

    def _get_validated_env_list() -> list[str]:
        """获取并验证逗号分隔的环境变量值列表的内部函数。"""
        value = os.getenv(env_name)  # 从环境中获取变量值
        if value is None:  # 如果环境变量未设置
            return default  # 返回默认列表

        # Split comma-separated values and strip whitespace
        values = [v.strip() for v in value.split(",") if v.strip()]  # 按逗号分割并去除空白

        if not values:  # 如果分割后为空列表
            return default  # 返回默认列表

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices  # 如果choices是可调用对象则调用获取实际选项列表

        # Validate each value
        for val in values:  # 遍历每个值进行验证
            if not case_sensitive:  # 如果不区分大小写
                check_value = val.lower()  # 将值转为小写
                check_choices = [choice.lower() for choice in actual_choices]  # 将选项列表转为小写
            else:  # 区分大小写
                check_value = val  # 直接使用原始值
                check_choices = actual_choices  # 直接使用原始选项列表

            if check_value not in check_choices:  # 如果值不在允许的选项中
                raise ValueError(  # 抛出值错误异常
                    f"Invalid value '{val}' in {env_name}. "  # 无效值提示
                    f"Valid options: {actual_choices}."  # 显示有效选项
                )

        return values  # 返回验证通过的值列表

    return _get_validated_env_list  # 返回验证函数


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """
    Creates a lambda which that validates environment variable
    containing comma-separated values against allowed choices which
    returns choices as a set.

    创建一个lambda函数，用于验证包含逗号分隔值的环境变量，并将结果作为集合返回。
    """

    def _get_validated_env_set() -> set[str]:
        """获取并验证环境变量值，返回集合的内部函数。"""
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())  # 调用列表验证函数并转换为集合

    return _get_validated_env_set  # 返回验证函数


def get_vllm_port() -> int | None:
    """获取VLLM_PORT环境变量中的端口号。

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.
        如果VLLM_PORT已设置则返回端口号整数，否则返回None。

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
        如果VLLM_PORT是URI格式，提示可能是Kubernetes服务发现问题。
    """
    if "VLLM_PORT" not in os.environ:  # 如果环境变量VLLM_PORT未设置
        return None  # 返回None

    port = os.getenv("VLLM_PORT", "0")  # 获取端口值，默认为"0"

    try:  # 尝试转换为整数
        return int(port)  # 将端口字符串转换为整数返回
    except ValueError as err:  # 如果转换失败
        from urllib3.util import parse_url  # 延迟导入URL解析工具

        parsed = parse_url(port)  # 解析端口值作为URL
        if parsed.scheme:  # 如果解析出了URI scheme（如tcp://）
            raise ValueError(  # 抛出更友好的错误提示
                f"VLLM_PORT '{port}' appears to be a URI. "  # 说明值看起来是URI
                "This may be caused by a Kubernetes service discovery issue,"  # 可能是K8s服务发现问题
                "check the warning in: https://docs.vllm.ai/en/stable/serving/env_vars.html"  # 参考文档
            ) from None  # 不链接原始异常
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err  # 否则提示必须是有效整数


def get_env_or_set_default(
    env_name: str,
    default_factory: Callable[[], str],
) -> Callable[[], str]:
    """
    Create a lambda that returns an environment variable value if set,
    or generates and sets a default value using the provided factory function.

    创建一个lambda函数：如果环境变量已设置则返回其值，
    否则使用提供的工厂函数生成默认值并设置到环境变量中。
    """

    def _get_or_set_default() -> str:
        """获取环境变量值或设置默认值的内部函数。"""
        value = os.getenv(env_name)  # 获取环境变量值
        if value is not None:  # 如果环境变量已设置
            return value  # 直接返回

        default_value = default_factory()  # 调用工厂函数生成默认值
        os.environ[env_name] = default_value  # 将默认值写入环境变量
        return default_value  # 返回默认值

    return _get_or_set_default  # 返回获取或设置默认值的函数


# The start-* and end* here are used by the documentation generator  # 以下的start和end标记用于文档生成器提取使用的环境变量
# to extract the used env vars.

# --8<-- [start:env-vars-definition]  # 环境变量定义区域开始标记

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

environment_variables: dict[str, Callable[[], Any]] = {  # 环境变量字典，键为变量名，值为惰性求值的可调用对象
    # ================== Installation Time Env Vars ==================  # ================== 安装时环境变量 ==================
    # Target device of vLLM, supporting [cuda (by default),
    # rocm, cpu]
    "VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),  # vLLM目标设备，支持cuda/rocm/cpu
    # Main CUDA version of vLLM. This follows PyTorch but can be overridden.
    "VLLM_MAIN_CUDA_VERSION": lambda: os.getenv("VLLM_MAIN_CUDA_VERSION", "").lower()  # vLLM主要CUDA版本，跟随PyTorch但可覆盖
    or "12.9",  # 默认CUDA版本为12.9
    # Controls PyTorch float32 matmul precision mode within vLLM workers.
    # Valid options mirror torch.set_float32_matmul_precision
    "VLLM_FLOAT32_MATMUL_PRECISION": env_with_choices(  # 控制vLLM worker中PyTorch float32矩阵乘法精度模式
        "VLLM_FLOAT32_MATMUL_PRECISION",  # 环境变量名
        "highest",  # 默认值为最高精度
        ["highest", "high", "medium"],  # 有效选项
        case_sensitive=False,  # 不区分大小写
    ),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),  # 并行编译任务的最大数量，默认为CPU数量
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),  # nvcc使用的线程数，默认为1
    # If set, vllm will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED": lambda: os.environ.get("VLLM_USE_PRECOMPILED", "")  # 是否使用预编译的二进制文件
    .strip()  # 去除首尾空白
    .lower()  # 转为小写
    in ("1", "true")  # 检查是否为真值
    or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),  # 或者检查预编译wheel位置是否已设置
    # If set, skip adding +precompiled suffix to version string
    "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": lambda: bool(  # 是否跳过版本字符串中的+precompiled后缀
        int(os.environ.get("VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX", "0"))  # 从环境变量获取并转换
    ),
    # Used to mark that setup.py is running in a Docker build context,
    # in order to force the use of precompiled binaries.
    "VLLM_DOCKER_BUILD_CONTEXT": lambda: os.environ.get("VLLM_DOCKER_BUILD_CONTEXT", "")  # 标记是否在Docker构建上下文中运行
    .strip()  # 去除首尾空白
    .lower()  # 转为小写
    in ("1", "true"),  # 检查是否为真值
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": env_with_choices(  # CMake构建类型选择
        "CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]  # 可选：调试、发布、带调试信息的发布
    ),
    # If set, vllm will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),  # 安装时是否打印详细日志
    # Root directory for vLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    "VLLM_CONFIG_ROOT": lambda: os.path.expanduser(  # vLLM配置文件根目录
        os.getenv(  # 获取环境变量
            "VLLM_CONFIG_ROOT",  # 环境变量名
            os.path.join(get_default_config_root(), "vllm"),  # 默认为~/.config/vllm
        )
    ),
    # ================== Runtime Env Vars ==================  # ================== 运行时环境变量 ==================
    # Root directory for vLLM cache files
    # Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set
    "VLLM_CACHE_ROOT": lambda: os.path.expanduser(  # vLLM缓存文件根目录
        os.getenv(  # 获取环境变量
            "VLLM_CACHE_ROOT",  # 环境变量名
            os.path.join(get_default_cache_root(), "vllm"),  # 默认为~/.cache/vllm
        )
    ),
    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),  # 分布式环境中当前节点的IP地址
    # used in distributed environment to manually set the communication port
    # Note: if VLLM_PORT is set, and some code asks for multiple ports, the
    # VLLM_PORT will be used as the first port, and the rest will be generated
    # by incrementing the VLLM_PORT value.
    "VLLM_PORT": get_vllm_port,  # 分布式环境中手动设置的通信端口
    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    "VLLM_RPC_BASE_PATH": lambda: os.getenv(  # 前端API服务器多进程模式下IPC通信的基础路径
        "VLLM_RPC_BASE_PATH", tempfile.gettempdir()  # 默认为系统临时目录
    ),
    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE": lambda: os.environ.get(  # 是否从ModelScope加载模型（替代Hugging Face Hub）
        "VLLM_USE_MODELSCOPE", "False"  # 默认为False
    ).lower()
    == "true",  # 检查是否为"true"
    # Interval in seconds to log a warning message when the ring buffer is full
    "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(  # 环形缓冲区满时警告日志的间隔时间（秒）
        os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")  # 默认60秒
    ),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),  # CUDA工具包主目录路径
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH": lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),  # NCCL动态库文件路径
    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),  # 动态链接库搜索路径
    # flag to control the chunk size (in MB) for sleeping memory allocations under ROCm
    "VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE": lambda: int(  # ROCm下休眠内存分配的块大小（MB）
        os.environ.get("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE", "256")  # 默认256MB
    ),
    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.9 this is
    # enabled by default.
    "VLLM_USE_STANDALONE_COMPILE": lambda: os.environ.get(  # 是否启用Inductor独立编译功能
        "VLLM_USE_STANDALONE_COMPILE", "1"  # 默认启用
    )
    == "1",  # 检查是否为"1"
    # Inductor's pre-grad passes don't do anything for vLLM.
    # The pre-grad passes get run even on cache-hit and negatively impact
    # vllm cold compile times by O(1s)
    # Can remove this after the following issue gets fixed
    # https://github.com/pytorch/pytorch/issues/174502
    "VLLM_ENABLE_PREGRAD_PASSES": lambda: os.environ.get(  # 是否启用Inductor的预梯度pass（对vLLM无用，会增加编译时间）
        "VLLM_ENABLE_PREGRAD_PASSES", "0"  # 默认禁用
    )
    == "1",  # 检查是否为"1"
    # Debug pattern matching inside custom passes.
    # Should be set to the fx.Node name (e.g. 'getitem_34' or 'scaled_mm_3').
    "VLLM_PATTERN_MATCH_DEBUG": lambda: os.environ.get(  # 自定义pass中调试模式匹配的fx.Node名称
        "VLLM_PATTERN_MATCH_DEBUG", None  # 默认不启用
    ),
    # Dump fx graphs to the given directory.
    # It will override CompilationConfig.debug_dump_path if set.
    "VLLM_DEBUG_DUMP_PATH": lambda: os.environ.get("VLLM_DEBUG_DUMP_PATH", None),  # fx图转储目录，覆盖CompilationConfig.debug_dump_path
    # Feature flag to enable/disable AOT compilation. This will ensure
    # compilation is done in warmup phase and the compilation will be
    # reused in subsequent calls.
    "VLLM_USE_AOT_COMPILE": use_aot_compile,  # 是否启用AOT编译，确保在预热阶段完成编译
    # Feature flag to enable/disable bytecode in
    # TorchCompileWithNoGuardsWrapper.
    "VLLM_USE_BYTECODE_HOOK": lambda: bool(  # 是否在TorchCompile包装器中启用字节码钩子
        int(os.environ.get("VLLM_USE_BYTECODE_HOOK", "1"))  # 默认启用
    ),
    # Force vllm to always load AOT compiled models from disk. Failure
    # to load will result in a hard error when this is enabled.
    # Will be ignored when VLLM_USE_AOT_COMPILE is disabled.
    "VLLM_FORCE_AOT_LOAD": lambda: os.environ.get("VLLM_FORCE_AOT_LOAD", "0") == "1",  # 是否强制从磁盘加载AOT编译模型
    # Enable loading compiled models directly from cached standalone compile artifacts
    # without re-splitting graph modules. This reduces overhead during model
    # loading by using reconstruct_serializable_fn_from_mega_artifact.
    "VLLM_USE_MEGA_AOT_ARTIFACT": lambda: os.environ.get(  # 是否从缓存的独立编译产物直接加载编译模型
        "VLLM_USE_MEGA_AOT_ARTIFACT", "0"  # 默认禁用
    )
    == "1",  # 检查是否为"1"
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),  # 分布式设置中进程的本地rank，用于确定GPU设备ID
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),  # 分布式设置中可见的CUDA设备
    # timeout for each iteration in the engine
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": lambda: int(  # 引擎每次迭代的超时时间（秒）
        os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")  # 默认60秒
    ),
    # Timeout in seconds for waiting for engine cores to become ready
    # during startup. Default is 600 seconds (10 minutes).
    "VLLM_ENGINE_READY_TIMEOUT_S": lambda: int(  # 启动时等待引擎核心就绪的超时时间（秒）
        os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "600")  # 默认600秒（10分钟）
    ),
    # API key for vLLM API server
    "VLLM_API_KEY": lambda: os.environ.get("VLLM_API_KEY", None),  # vLLM API服务器的API密钥
    # Whether to log responses from API Server for debugging
    "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": lambda: os.environ.get(  # 是否记录API服务器的响应用于调试
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "False"  # 默认不记录
    ).lower()
    == "true",  # 检查是否为"true"
    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID": lambda: os.environ.get("S3_ACCESS_KEY_ID", None),  # S3访问密钥ID
    "S3_SECRET_ACCESS_KEY": lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),  # S3秘密访问密钥
    "S3_ENDPOINT_URL": lambda: os.environ.get("S3_ENDPOINT_URL", None),  # S3端点URL
    # Usage stats collection
    "VLLM_USAGE_STATS_SERVER": lambda: os.environ.get(  # 使用统计数据上报服务器地址
        "VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"  # 默认服务器地址
    ),
    "VLLM_NO_USAGE_STATS": lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",  # 是否禁用使用统计
    "VLLM_DO_NOT_TRACK": lambda: (  # 是否禁止跟踪（支持标准DO_NOT_TRACK环境变量）
        os.environ.get("VLLM_DO_NOT_TRACK", None)  # 检查VLLM_DO_NOT_TRACK
        or os.environ.get("DO_NOT_TRACK", None)  # 或标准DO_NOT_TRACK
        or "0"  # 默认不禁止
    )
    == "1",  # 检查是否为"1"
    "VLLM_USAGE_SOURCE": lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),  # 使用来源标识，默认为production
    # Logging configuration
    # If set to 0, vllm will not configure logging
    # If set to 1, vllm will configure logging using the default configuration
    #    or the configuration file specified by VLLM_LOGGING_CONFIG_PATH
    "VLLM_CONFIGURE_LOGGING": lambda: bool(  # 是否由vLLM配置日志系统
        int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))  # 默认启用
    ),
    "VLLM_LOGGING_CONFIG_PATH": lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),  # 日志配置文件路径
    # this is used for configuring the default logging level
    "VLLM_LOGGING_LEVEL": lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),  # 默认日志级别，转为大写
    # this is used for configuring the default logging stream
    "VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout"),  # 默认日志输出流
    # if set, VLLM_LOGGING_PREFIX will be prepended to all log messages
    "VLLM_LOGGING_PREFIX": lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),  # 日志消息前缀
    # Controls colored logging output. Options: "auto" (default, colors when terminal),
    # "1" (always use colors), "0" (never use colors)
    "VLLM_LOGGING_COLOR": lambda: os.getenv("VLLM_LOGGING_COLOR", "auto"),  # 日志颜色控制（auto/1/0）
    # Standard unix flag for disabling ANSI color codes
    "NO_COLOR": lambda: os.getenv("NO_COLOR", "0") != "0",  # 标准Unix禁用ANSI颜色码标志
    # If set, vllm will log stats at this interval in seconds
    # If not set, vllm will log stats every 10 seconds.
    "VLLM_LOG_STATS_INTERVAL": lambda: val  # 统计日志记录间隔（秒）
    if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0  # 如果值大于0则使用该值
    else 10.0,  # 否则默认10秒
    # Trace function calls
    # If set to 1, vllm will trace function calls
    # Useful for debugging
    "VLLM_TRACE_FUNCTION": lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),  # 是否追踪函数调用（调试用）
    # If set, vllm will use flashinfer sampler
    "VLLM_USE_FLASHINFER_SAMPLER": lambda: bool(  # 是否使用FlashInfer采样器
        int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"])  # 从环境变量获取并转换
    )
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ  # 仅在环境变量已设置时读取
    else None,  # 未设置时返回None
    # Pipeline stage partition strategy
    "VLLM_PP_LAYER_PARTITION": lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),  # 流水线并行阶段分区策略
    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "VLLM_CPU_KVCACHE_SPACE": lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0"))  # CPU后端KV缓存空间（GB）
    if "VLLM_CPU_KVCACHE_SPACE" in os.environ  # 仅在环境变量已设置时读取
    else None,  # 未设置时返回None，默认将设为4GB
    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "VLLM_CPU_OMP_THREADS_BIND": lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "auto"),  # CPU后端OpenMP线程绑定的核心ID
    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "VLLM_CPU_NUM_OF_RESERVED_CPU": lambda: int(  # CPU后端预留的CPU核心数量
        os.getenv("VLLM_CPU_NUM_OF_RESERVED_CPU", "0")  # 获取环境变量值
    )
    if "VLLM_CPU_NUM_OF_RESERVED_CPU" in os.environ  # 仅在环境变量已设置时读取
    else None,  # 未设置时返回None
    # (CPU backend only) whether to use SGL kernels, optimized for small batch.
    "VLLM_CPU_SGL_KERNEL": lambda: bool(int(os.getenv("VLLM_CPU_SGL_KERNEL", "0"))),  # CPU后端是否使用SGL内核
    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": env_with_choices(  # Ray编译DAG的通信通道类型
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto", ["auto", "nccl", "shm"]  # 可选：自动、NCCL、共享内存
    ),
    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph.
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM": lambda: bool(  # 是否启用Ray编译DAG的GPU通信重叠（实验性功能）
        int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))  # 默认禁用
    ),
    # If the env var is set, it uses a Ray Communicator wrapping
    # vLLM's pipeline parallelism communicator to interact with Ray's
    # Compiled Graph. Otherwise, it uses Ray's NCCL communicator.
    "VLLM_USE_RAY_WRAPPED_PP_COMM": lambda: bool(  # 是否使用Ray封装的vLLM流水线并行通信器
        int(os.getenv("VLLM_USE_RAY_WRAPPED_PP_COMM", "1"))  # 默认启用
    ),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "VLLM_WORKER_MULTIPROC_METHOD": env_with_choices(  # 工作进程的多进程创建方式
        "VLLM_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]  # 可选：spawn或fork
    ),
    # Path to the cache for storing downloaded assets
    "VLLM_ASSETS_CACHE": lambda: os.path.expanduser(  # 下载资源的缓存路径
        os.getenv(  # 获取环境变量
            "VLLM_ASSETS_CACHE",  # 环境变量名
            os.path.join(get_default_cache_root(), "vllm", "assets"),  # 默认为~/.cache/vllm/assets
        )
    ),
    # If the env var is set, we will clean model file in
    # this path $VLLM_ASSETS_CACHE/model_streamer/$model_name
    "VLLM_ASSETS_CACHE_MODEL_CLEAN": lambda: bool(  # 是否清理资源缓存中的模型文件
        int(os.getenv("VLLM_ASSETS_CACHE_MODEL_CLEAN", "0"))  # 默认不清理
    ),
    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "VLLM_IMAGE_FETCH_TIMEOUT": lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),  # 多模态模型获取图片的超时时间（秒）
    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "VLLM_VIDEO_FETCH_TIMEOUT": lambda: int(  # 多模态模型获取视频的超时时间（秒）
        os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")  # 默认30秒
    ),
    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "VLLM_AUDIO_FETCH_TIMEOUT": lambda: int(  # 多模态模型获取音频的超时时间（秒）
        os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")  # 默认10秒
    ),
    # Whether to allow HTTP redirects when fetching from media URLs.
    # Default to True
    "VLLM_MEDIA_URL_ALLOW_REDIRECTS": lambda: bool(  # 是否允许媒体URL的HTTP重定向
        int(os.getenv("VLLM_MEDIA_URL_ALLOW_REDIRECTS", "1"))  # 默认允许
    ),
    # Max number of workers for the thread pool handling
    # media bytes loading. Set to 1 to disable parallel processing.
    # Default is 8
    "VLLM_MEDIA_LOADING_THREAD_COUNT": lambda: int(  # 媒体字节加载线程池的最大工作线程数
        os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")  # 默认8个线程
    ),
    # Maximum filesize in MB for a single audio file when processing
    # speech-to-text requests. Files larger than this will be rejected.
    # Default is 25 MB
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": lambda: int(  # 单个音频文件的最大大小（MB），超过将被拒绝
        os.getenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "25")  # 默认25MB
    ),
    # Backend for Video IO
    # - "opencv": Default backend that uses OpenCV stream buffered backend.
    # - "identity": Returns raw video bytes for model processor to handle.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_VIDEO_LOADER_BACKEND": lambda: os.getenv(  # 视频IO后端选择
        "VLLM_VIDEO_LOADER_BACKEND", "opencv"  # 默认使用OpenCV
    ),
    # Media connector implementation.
    # - "http": Default connector that supports fetching media via HTTP.
    #
    # Custom implementations can be registered
    # via `@MEDIA_CONNECTOR_REGISTRY.register("my_custom_media_connector")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_MEDIA_CONNECTOR": lambda: os.getenv("VLLM_MEDIA_CONNECTOR", "http"),  # 媒体连接器实现，默认HTTP
    # Hash algorithm for multimodal content hashing.
    # - "blake3": Default, fast cryptographic hash (not FIPS 140-3 compliant)
    # - "sha256": FIPS 140-3 compliant, widely supported
    # - "sha512": FIPS 140-3 compliant, faster on 64-bit systems
    # Use sha256 or sha512 for FIPS compliance in government/enterprise deployments
    "VLLM_MM_HASHER_ALGORITHM": env_with_choices(  # 多模态内容哈希算法选择
        "VLLM_MM_HASHER_ALGORITHM",  # 环境变量名
        "blake3",  # 默认使用blake3（快速但非FIPS兼容）
        ["blake3", "sha256", "sha512"],  # 可选算法
        case_sensitive=False,  # 不区分大小写
    ),
    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "VLLM_XLA_CACHE_PATH": lambda: os.path.expanduser(  # XLA持久化缓存目录路径（仅用于TPU等XLA设备）
        os.getenv(  # 获取环境变量
            "VLLM_XLA_CACHE_PATH",  # 环境变量名
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),  # 默认路径
        )
    ),
    # If set, assert on XLA recompilation after each execution step.
    "VLLM_XLA_CHECK_RECOMPILATION": lambda: bool(  # 是否在每步执行后断言检查XLA重编译
        int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION", "0"))  # 默认不检查
    ),
    # Enable SPMD mode for TPU backend.
    "VLLM_XLA_USE_SPMD": lambda: bool(int(os.getenv("VLLM_XLA_USE_SPMD", "0"))),  # 是否为TPU后端启用SPMD模式
    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(  # 引擎崩溃后API服务器是否保持存活
        int(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))  # 默认不保持
    ),
    # If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": lambda: (  # 是否允许最大序列长度超过模型配置值
        os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower()  # 获取并规范化值
        in ("1", "true")  # 检查是否为真值
    ),
    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "VLLM_TEST_FORCE_FP8_MARLIN": lambda: (  # 是否强制使用FP8 Marlin进行FP8量化（忽略硬件支持）
        os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower()  # 获取并规范化值
        in ("1", "true")  # 检查是否为真值
    ),
    "VLLM_TEST_FORCE_LOAD_FORMAT": lambda: os.getenv(  # 测试用强制加载格式
        "VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"  # 默认为dummy
    ),
    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "VLLM_RPC_TIMEOUT": lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),  # ZMQ客户端等待后端响应的超时时间（毫秒）
    # Timeout in seconds for keeping HTTP connections alive in API server
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(  # API服务器HTTP连接保活超时时间（秒）
        os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")  # 默认5秒
    ),
    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "VLLM_PLUGINS": lambda: None  # 要加载的插件名称列表（逗号分隔）
    if "VLLM_PLUGINS" not in os.environ  # 未设置时加载所有插件
    else os.environ["VLLM_PLUGINS"].split(","),  # 设置时按逗号分割
    # a local directory to look in for unrecognized LoRA adapters.
    # only works if plugins are enabled and
    # VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    "VLLM_LORA_RESOLVER_CACHE_DIR": lambda: os.getenv(  # LoRA适配器本地缓存目录
        "VLLM_LORA_RESOLVER_CACHE_DIR", None  # 默认为None
    ),
    # A remote HF repo(s) containing one or more LoRA adapters, which
    # may be downloaded and leveraged as needed. Only works if plugins
    # are enabled and VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    # Values should be comma separated.
    "VLLM_LORA_RESOLVER_HF_REPO_LIST": lambda: os.getenv(  # 包含LoRA适配器的远程HF仓库列表（逗号分隔）
        "VLLM_LORA_RESOLVER_HF_REPO_LIST", None  # 默认为None
    ),
    # If set, vLLM will use Triton implementations of AWQ.
    "VLLM_USE_TRITON_AWQ": lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),  # 是否使用Triton实现的AWQ
    # If set, allow loading or unloading lora adapters in runtime,
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": lambda: (  # 是否允许运行时加载/卸载LoRA适配器
        os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower()  # 获取并规范化值
        in ("1", "true")  # 检查是否为真值
    ),
    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set VLLM_SKIP_P2P_CHECK=0
    # so that vLLM can verify if p2p is actually working.
    # See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "VLLM_SKIP_P2P_CHECK": lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",  # 是否跳过P2P连接检查（假设驱动报告正确）
    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "VLLM_DISABLED_KERNELS": lambda: []  # 应禁用的量化内核列表
    if "VLLM_DISABLED_KERNELS" not in os.environ  # 未设置时返回空列表
    else os.environ["VLLM_DISABLED_KERNELS"].split(","),  # 设置时按逗号分割
    "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": lambda: bool(  # 是否启用FLA打包循环解码
        int(os.getenv("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1"))  # 默认启用
    ),
    # Disable pynccl (using torch.distributed instead)
    "VLLM_DISABLE_PYNCCL": lambda: (  # 是否禁用pynccl（改用torch.distributed）
        os.getenv("VLLM_DISABLE_PYNCCL", "False").lower() in ("true", "1")  # 检查是否为真值
    ),
    # Optional: enable external Oink custom ops (e.g., Blackwell RMSNorm).
    # Disabled by default.
    "VLLM_USE_OINK_OPS": lambda: (  # 是否启用外部Oink自定义算子（如Blackwell RMSNorm）
        os.getenv("VLLM_USE_OINK_OPS", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "VLLM_ROCM_USE_AITER": lambda: (  # 是否启用ROCm aiter算子（父开关）
        os.getenv("VLLM_ROCM_USE_AITER", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # Whether to use aiter paged attention.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_PAGED_ATTN": lambda: (  # 是否使用aiter分页注意力
        os.getenv("VLLM_ROCM_USE_AITER_PAGED_ATTN", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    "VLLM_ROCM_USE_AITER_LINEAR": lambda: (  # 是否使用aiter线性层算子（如scaled_mm）
        os.getenv("VLLM_ROCM_USE_AITER_LINEAR", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Whether to use aiter moe ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MOE": lambda: (  # 是否使用aiter MoE算子
        os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in ("true", "1")  # 默认启用
    ),
    # use aiter rms norm op if aiter ops are enabled.
    "VLLM_ROCM_USE_AITER_RMSNORM": lambda: (  # 是否使用aiter RMS归一化算子
        os.getenv("VLLM_ROCM_USE_AITER_RMSNORM", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Whether to use aiter mla ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MLA": lambda: (  # 是否使用aiter MLA算子
        os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Whether to use aiter mha ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MHA": lambda: (  # 是否使用aiter MHA算子
        os.getenv("VLLM_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Whether to use aiter fp4 gemm asm.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": lambda: (  # 是否使用aiter FP4汇编GEMM
        os.getenv("VLLM_ROCM_USE_AITER_FP4_ASM_GEMM", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # Whether to use aiter rope.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_TRITON_ROPE": lambda: (  # 是否使用aiter Triton ROPE
        os.getenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # Whether to use aiter triton fp8 bmm kernel
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_FP8BMM": lambda: (  # 是否使用aiter Triton FP8批量矩阵乘法内核
        os.getenv("VLLM_ROCM_USE_AITER_FP8BMM", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Whether to use aiter triton fp4 bmm kernel
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_FP4BMM": lambda: (  # 是否使用aiter Triton FP4批量矩阵乘法内核
        os.getenv("VLLM_ROCM_USE_AITER_FP4BMM", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Use AITER triton unified attention for V1 attention
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (  # 是否使用AITER Triton统一注意力（V1）
        os.getenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()  # 获取环境变量并转小写
        in ("true", "1")  # 默认禁用
    ),
    # Whether to use aiter fusion shared experts ops.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (  # 是否使用aiter融合共享专家算子
        os.getenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower()  # 获取环境变量并转小写
        in ("true", "1")  # 默认禁用
    ),
    # Whether to use aiter triton kernels for gemm ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_TRITON_GEMM": lambda: (  # 是否使用aiter Triton GEMM内核
        os.getenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", "True").lower() in ("true", "1")  # 默认启用
    ),
    # use rocm skinny gemms
    "VLLM_ROCM_USE_SKINNY_GEMM": lambda: (  # 是否使用ROCm瘦GEMM内核
        os.getenv("VLLM_ROCM_USE_SKINNY_GEMM", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Pad the fp8 weights to 256 bytes for ROCm
    "VLLM_ROCM_FP8_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),  # 是否对ROCm的FP8权重进行256字节填充
    # Pad the weights for the moe kernel
    "VLLM_ROCM_MOE_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_MOE_PADDING", "1"))),  # 是否对MoE内核的权重进行填充
    # custom paged attention kernel for MI3* cards
    "VLLM_ROCM_CUSTOM_PAGED_ATTN": lambda: (  # MI3*卡的自定义分页注意力内核
        os.getenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in ("true", "1")  # 默认启用
    ),
    # Whether to use the shuffled kv cache layout
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": lambda: (  # 是否使用混洗的KV缓存布局
        os.getenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4 or NONE
    # Recommended for large models to get allreduce
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": env_with_choices(  # MI3*卡自定义快速allreduce的量化级别
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",  # 环境变量名
        "NONE",  # 默认不启用量化
        ["FP", "INT8", "INT6", "INT4", "NONE"],  # 可选量化级别
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": lambda: (  # 快速allreduce是否将BF16转换为FP16
        os.getenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower()  # 获取环境变量并转小写
        in ("true", "1")  # 默认启用
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": lambda: maybe_convert_int(  # 快速allreduce允许的最大数据大小（MB）
        os.environ.get("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)  # 获取环境变量，默认None
    ),
    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT": lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),  # FP8 KV缓存动态查询缩放因子的除数
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT": lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),  # FP8 KV缓存动态键缩放因子的除数
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT": lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),  # FP8 KV缓存动态值缩放因子的除数
    # If set, enable multiprocessing in LLM for the V1 code path.
    "VLLM_ENABLE_V1_MULTIPROCESSING": lambda: bool(  # 是否在V1代码路径中为LLM启用多进程
        int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))  # 默认启用
    ),
    "VLLM_LOG_BATCHSIZE_INTERVAL": lambda: float(  # 批大小日志记录间隔（秒，-1表示禁用）
        os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")  # 默认禁用
    ),
    "VLLM_DISABLE_COMPILE_CACHE": disable_compile_cache,  # 是否禁用编译缓存
    # If set, vllm will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "VLLM_SERVER_DEV_MODE": lambda: bool(int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))),  # 是否启用开发模式（额外调试端点）
    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": lambda: int(  # V1 AsyncLLM中每个异步任务处理的最大请求数
        os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")  # 默认128
    ),
    # If set, vLLM will disable the MLA attention optimizations.
    "VLLM_MLA_DISABLE": lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),  # 是否禁用MLA注意力优化
    # If set, vLLM will pick up the provided Flash Attention MLA
    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as vLLM.
    "VLLM_RAY_PER_WORKER_GPUS": lambda: float(  # Ray中每个worker分配的GPU数量
        os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")  # 默认1.0，可设为分数以允许多actor共享GPU
    ),
    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "VLLM_RAY_BUNDLE_INDICES": lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),  # Ray bundle索引（逗号分隔的整数列表）
    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable VLLM_CUDART_SO_PATH.
    "VLLM_CUDART_SO_PATH": lambda: os.getenv("VLLM_CUDART_SO_PATH", None),  # CUDA运行时动态库路径
    # Rank of the process in the data parallel setting
    "VLLM_DP_RANK": lambda: int(os.getenv("VLLM_DP_RANK", "0")),  # 数据并行中的进程rank
    # Rank of the process in the data parallel setting.
    # Defaults to VLLM_DP_RANK when not set.
    "VLLM_DP_RANK_LOCAL": lambda: int(  # 数据并行中的本地rank
        os.getenv("VLLM_DP_RANK_LOCAL", sys.modules[__name__].VLLM_DP_RANK)  # 默认跟随VLLM_DP_RANK
    ),
    # World size of the data parallel setting
    "VLLM_DP_SIZE": lambda: int(os.getenv("VLLM_DP_SIZE", "1")),  # 数据并行的世界大小
    # IP address of the master node in the data parallel setting
    "VLLM_DP_MASTER_IP": lambda: os.getenv("VLLM_DP_MASTER_IP", "127.0.0.1"),  # 数据并行主节点IP地址
    # Port of the master node in the data parallel setting
    "VLLM_DP_MASTER_PORT": lambda: int(os.getenv("VLLM_DP_MASTER_PORT", "0")),  # 数据并行主节点端口（VLLM多卡/分布式推理时，主进程通信使用的端口号）
    # In the context of executing MoE models with Data-Parallel, Expert-Parallel
    # and Batched All-to-All dispatch/combine kernels, VLLM_MOE_DP_CHUNK_SIZE
    # dictates the quantum of tokens that can be dispatched from a DP
    # rank. All DP ranks process the activations in VLLM_MOE_DP_CHUNK_SIZE
    # units.
    "VLLM_MOE_DP_CHUNK_SIZE": lambda: int(os.getenv("VLLM_MOE_DP_CHUNK_SIZE", "256")),  # MoE数据并行token分发的量子大小
    "VLLM_ENABLE_MOE_DP_CHUNK": lambda: bool(  # 是否启用MoE数据并行分块
        int(os.getenv("VLLM_ENABLE_MOE_DP_CHUNK", "1"))  # 默认启用
    ),
    # Randomize inputs during dummy runs when using Data Parallel
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": lambda: os.environ.get(  # 数据并行虚拟运行时是否随机化输入
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", "0"  # 默认不随机化
    )
    == "1",  # 检查是否为"1"
    # Strategy to pack the data parallel ranks for Ray.
    # Available options:
    # - "fill":
    #   for DP master node, allocate exactly data-parallel-size-local DP ranks,
    #   for non-master nodes, allocate as many DP ranks as can fit;
    # - "strict":
    #   allocate exactly data-parallel-size-local DP ranks to each picked node;
    # - "span":
    #   Should be used only when a single DP rank requires multiple nodes.
    #   allocate one DP rank over as many nodes as required for set world_size;
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "VLLM_RAY_DP_PACK_STRATEGY": lambda: os.getenv(  # Ray数据并行rank打包策略
        "VLLM_RAY_DP_PACK_STRATEGY", "strict"  # 默认为strict模式
    ),
    # Comma-separated *additional* prefixes of env vars to copy from the
    # driver to Ray workers.  These are merged with the built-in defaults
    # defined in ``vllm.ray.ray_env`` (VLLM_, etc.).  Example: "MYLIB_,OTHER_"
    "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": lambda: os.getenv(  # 复制到Ray worker的额外环境变量前缀
        "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", ""  # 默认为空
    ),
    # Comma-separated *additional* individual env var names to copy from
    # the driver to Ray workers.  Merged with the built-in defaults
    # defined in ``vllm.ray.ray_env`` (PYTHONHASHSEED).
    # Example: "MY_SECRET,MY_FLAG"
    "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": lambda: os.getenv(  # 复制到Ray worker的额外环境变量名
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", ""  # 默认为空
    ),
    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "VLLM_CI_USE_S3": lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",  # CI中是否通过RunAI Streamer使用S3路径加载模型
    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "VLLM_MODEL_REDIRECT_PATH": lambda: os.environ.get(  # 模型名称重定向到本地文件夹的映射文件路径
        "VLLM_MODEL_REDIRECT_PATH", None  # 默认为None
    ),
    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "VLLM_MARLIN_USE_ATOMIC_ADD": lambda: os.environ.get(  # 是否在gptq/awq Marlin内核中使用atomicAdd归约
        "VLLM_MARLIN_USE_ATOMIC_ADD", "0"  # 默认不使用
    )
    == "1",  # 检查是否为"1"
    # Whether to use marlin kernel in mxfp4 quantization method
    "VLLM_MXFP4_USE_MARLIN": lambda: maybe_convert_bool(  # 是否在mxfp4量化方法中使用Marlin内核
        os.environ.get("VLLM_MXFP4_USE_MARLIN", None)  # 获取环境变量，默认None
    ),
    # The activation dtype for marlin kernel
    "VLLM_MARLIN_INPUT_DTYPE": env_with_choices(  # Marlin内核的激活数据类型
        "VLLM_MARLIN_INPUT_DTYPE", None, ["int8", "fp8"]  # 可选：int8或fp8
    ),
    # Whether to use DeepEPLL kernels for NVFP4 quantization and dispatch method
    # only supported on Blackwell GPUs and with
    # https://github.com/deepseek-ai/DeepEP/pull/341
    "VLLM_DEEPEPLL_NVFP4_DISPATCH": lambda: bool(  # 是否使用DeepEPLL的NVFP4量化分发内核
        int(os.getenv("VLLM_DEEPEPLL_NVFP4_DISPATCH", "0"))  # 默认禁用，仅支持Blackwell GPU
    ),
    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "VLLM_V1_USE_OUTLINES_CACHE": lambda: os.environ.get(  # 是否为V1启用outlines磁盘缓存（无大小限制，不安全环境慎用）
        "VLLM_V1_USE_OUTLINES_CACHE", "0"  # 默认禁用
    )
    == "1",  # 检查是否为"1"
    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "VLLM_TPU_BUCKET_PADDING_GAP": lambda: int(  # TPU前向传播的填充桶间隔
        os.environ["VLLM_TPU_BUCKET_PADDING_GAP"]  # 从环境变量获取
    )
    if "VLLM_TPU_BUCKET_PADDING_GAP" in os.environ  # 仅在环境变量已设置时读取
    else 0,  # 默认为0
    "VLLM_TPU_MOST_MODEL_LEN": lambda: maybe_convert_int(  # TPU最大模型长度
        os.environ.get("VLLM_TPU_MOST_MODEL_LEN", None)  # 获取环境变量，可能为None
    ),
    # Whether using Pathways
    "VLLM_TPU_USING_PATHWAYS": lambda: bool(  # 是否使用Pathways（通过检查JAX_PLATFORMS中是否包含proxy）
        "proxy" in os.getenv("JAX_PLATFORMS", "").lower()  # 检查JAX平台配置
    ),
    # Allow use of DeepGemm kernels for fused moe ops.
    "VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),  # 是否允许使用DeepGemm内核
    # Allow use of DeepGemm specifically for MoE fused ops (overrides only MoE).
    "VLLM_MOE_USE_DEEP_GEMM": lambda: bool(  # 是否专门为MoE融合操作使用DeepGemm
        int(os.getenv("VLLM_MOE_USE_DEEP_GEMM", "1"))  # 默认启用
    ),
    # Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs.
    "VLLM_USE_DEEP_GEMM_E8M0": lambda: bool(  # Blackwell GPU上DeepGEMM是否使用E8M0缩放
        int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "1"))  # 默认启用
    ),
    # Whether to create TMA-aligned scale tensor when DeepGEMM is used.
    "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": lambda: bool(  # DeepGEMM是否创建TMA对齐的缩放张量
        int(os.getenv("VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "1"))  # 默认启用
    ),
    # DeepGemm JITs the kernels on-demand. The warmup attempts to make DeepGemm
    # JIT all the required kernels before model execution so there is no
    # JIT'ing in the hot-path. However, this warmup increases the engine
    # startup time by a couple of minutes.
    # Available options:
    #  - "skip"  : Skip warmup.
    #  - "full"  : Warmup deepgemm by running all possible gemm shapes the
    #   engine could encounter.
    #  - "relax" : Select gemm shapes to run based on some heuristics. The
    #   heuristic aims to have the same effect as running all possible gemm
    #   shapes, but provides no guarantees.
    "VLLM_DEEP_GEMM_WARMUP": env_with_choices(  # DeepGemm内核JIT编译预热策略
        "VLLM_DEEP_GEMM_WARMUP",  # 环境变量名
        "relax",  # 默认使用启发式预热
        [
            "skip",  # 跳过预热
            "full",  # 完整预热
            "relax",  # 启发式预热
        ],
    ),
    # Whether to use fused grouped_topk used for MoE expert selection.
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK": lambda: bool(  # 是否使用融合的分组topk进行MoE专家选择
        int(os.getenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"))  # 默认启用
    ),
    # Allow use of FlashInfer FP8 block-scale GEMM for linear layers.
    # This uses TensorRT-LLM kernels and requires SM90+ (Hopper).
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(  # 是否使用FlashInfer FP8块缩放GEMM（需要SM90+）
        int(os.getenv("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1"))  # 默认启用
    ),
    # Allow use of FlashInfer BF16 MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP16": lambda: bool(  # 是否使用FlashInfer BF16 MoE内核
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP16", "0"))  # 默认禁用
    ),
    # Allow use of FlashInfer FP8 MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP8": lambda: bool(  # 是否使用FlashInfer FP8 MoE内核
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP8", "0"))  # 默认禁用
    ),
    # Allow use of FlashInfer NVFP4 MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP4": lambda: bool(  # 是否使用FlashInfer NVFP4 MoE内核
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP4", "0"))  # 默认禁用
    ),
    # Allow use of FlashInfer MxInt4 MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_INT4": lambda: bool(  # 是否使用FlashInfer MxInt4 MoE内核
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_INT4", "0"))  # 默认禁用
    ),
    # If set to 1, use the FlashInfer
    # MXFP8 (activation) x MXFP4 (weight) MoE backend.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": lambda: bool(  # 是否使用FlashInfer MXFP8激活x MXFP4权重的MoE后端
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "0"))  # 默认禁用
    ),
    # If set to 1, use the FlashInfer CUTLASS backend for
    # MXFP8 (activation) x MXFP4 (weight) MoE.
    # This is separate from the TRTLLMGEN path controlled by
    # VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS": lambda: bool(  # 是否使用FlashInfer CUTLASS后端的MXFP8x MXFP4 MoE
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS", "0"))  # 默认禁用
    ),
    # If set to 1, use the FlashInfer
    # BF16 (activation) x MXFP4 (weight) MoE backend.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": lambda: bool(  # 是否使用FlashInfer BF16激活x MXFP4权重的MoE后端
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "0"))  # 默认禁用
    ),
    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),  # xgrammar编译器缓存大小（MB），约可存储1000个JSON schema
    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(  # msgspec零拷贝张量序列化/反序列化的大小阈值
        os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256")  # 默认256字节
    ),
    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "VLLM_ALLOW_INSECURE_SERIALIZATION": lambda: bool(  # 是否允许使用pickle进行不安全序列化
        int(os.getenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0"))  # 默认不允许
    ),
    # Temporary: skip adding random suffix to internal request IDs. May be
    # needed for KV connectors that match request IDs across instances.
    "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": lambda: bool(  # 是否禁用内部请求ID的随机后缀
        int(os.getenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0"))  # 默认不禁用
    ),
    # IP address used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_HOST": lambda: os.getenv(  # NIXL远程代理握手的IP地址
        "VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"  # 默认localhost
    ),
    # Port used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_PORT": lambda: int(  # NIXL远程代理握手的端口
        os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")  # 默认5600
    ),
    # Port used for Mooncake handshake between remote agents.
    "VLLM_MOONCAKE_BOOTSTRAP_PORT": lambda: int(  # Mooncake远程代理握手的端口
        os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "8998")  # 默认8998
    ),
    # Flashinfer MoE backend for vLLM's fused Mixture-of-Experts support.
    # Both require compute capability 10.0 or above.
    # Available options:
    # - "throughput":  [default]
    #     Uses CUTLASS kernels optimized for high-throughput batch inference.
    # - "latency":
    #     Uses TensorRT-LLM kernels optimized for low-latency inference.
    "VLLM_FLASHINFER_MOE_BACKEND": env_with_choices(  # FlashInfer MoE后端选择（需要计算能力10.0+）
        "VLLM_FLASHINFER_MOE_BACKEND",  # 环境变量名
        "latency",  # 默认使用低延迟模式
        ["throughput", "latency", "masked_gemm"],  # 可选：吞吐优化、延迟优化、掩码GEMM
    ),
    # Flashinfer fused allreduce backend.
    # "auto" will default to "mnnvl", which performs mostly same/better than "trtllm".
    # But "mnnvl" backend does not support fuse with quantization.
    # TODO: Default is "trtllm" right now because "mnnvl" has issues with cudagraph:
    # https://github.com/vllm-project/vllm/issues/35772
    # Should switch back to "auto" if the issue is resolved.
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND": env_with_choices(  # FlashInfer融合allreduce后端选择
        "VLLM_FLASHINFER_ALLREDUCE_BACKEND",  # 环境变量名
        "trtllm",  # 默认使用trtllm（因mnnvl有cudagraph问题）
        ["auto", "trtllm", "mnnvl"],  # 可选：自动、TensorRT-LLM、MNNVL
    ),
    # Control the workspace buffer size for the FlashInfer backend.
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": lambda: int(  # FlashInfer后端工作区缓冲区大小（字节）
        os.getenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(394 * 1024 * 1024))  # 默认约394MB
    ),
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE CUTLASS Kernel. This value is used to create a buffer for
    # the blockscale tensor of activations NVFP4 Quantization.
    # This is used to prevent the kernel from running out of memory.
    "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE": lambda: int(  # NVFP4 MoE CUTLASS内核每个专家支持的最大token数
        os.getenv("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")  # 默认163840
    ),
    # Specifies the thresholds of the communicated tensor sizes under which
    # vllm should use flashinfer fused allreduce. The variable should be a
    # JSON with the following format:
    #     { <world size>: <max size in mb> }
    # Unspecified world sizes will fall back to
    #     { 2: 64, 4: 1, <everything else>: 0.5 }
    "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB": lambda: json.loads(  # FlashInfer融合allreduce的张量大小阈值（JSON格式）
        os.getenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB", "{}")  # 默认为空JSON
    ),
    # MoE routing strategy selector.
    # See `RoutingSimulator.get_available_strategies()` # for available
    # strategies.
    # Custom routing strategies can be registered by
    # RoutingSimulator.register_strategy()
    # Note: custom strategies may not produce correct model outputs
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": lambda: os.environ.get(  # MoE路由策略选择器
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""  # 默认为空（使用默认策略）
    ).lower(),  # 转为小写
    # Regex timeout for use by the vLLM tool parsing plugins.
    "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(  # 工具解析插件的正则表达式超时时间（秒）
        os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")  # 默认1秒
    ),
    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "VLLM_MQ_MAX_CHUNK_BYTES_MB": lambda: int(  # RPC消息队列的最大块大小（MB）
        os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")  # 默认16MB
    ),
    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(  # 多进程执行器中execute_model RPC调用的超时时间（秒）
        os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")  # 默认300秒
    ),
    # KV Cache layout used throughout vllm.
    # Some common values are:
    # - NHD
    # - HND
    # Where N=num_blocks, H=num_heads and D=head_size. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    "VLLM_KV_CACHE_LAYOUT": env_with_choices(  # KV缓存布局格式（NHD或HND）
        "VLLM_KV_CACHE_LAYOUT", None, ["NHD", "HND"]  # 默认由后端决定
    ),
    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "VLLM_COMPUTE_NANS_IN_LOGITS": lambda: bool(  # 是否检查logits中是否包含NaN（调试用）
        int(os.getenv("VLLM_COMPUTE_NANS_IN_LOGITS", "0"))  # 默认不检查
    ),
    # Controls whether or not emulations are used for NVFP4
    # generations on machines < 100 for compressed-tensors
    # models
    "VLLM_USE_NVFP4_CT_EMULATIONS": lambda: bool(  # 是否为低于SM100的机器使用NVFP4仿真
        int(os.getenv("VLLM_USE_NVFP4_CT_EMULATIONS", "0"))  # 默认不使用
    ),
    # Time (in seconds) after which the KV cache on the producer side is
    # automatically cleared if no READ notification is received from the
    # consumer. This is only applicable when using NixlConnector in a
    # disaggregated decode-prefill setup.
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": lambda: int(  # NixlConnector中KV缓存自动清除的超时时间（秒）
        os.getenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "480")  # 默认480秒
    ),
    # Controls the read mode for the Mori-IO connector
    "VLLM_MORIIO_CONNECTOR_READ_MODE": lambda: (  # Mori-IO连接器的读取模式控制
        os.getenv("VLLM_MORIIO_CONNECTOR_READ_MODE", "False").lower() in ("true", "1")  # 默认禁用
    ),
    # Controls the QP (Queue Pair) per transfer configuration for the Mori-IO connector
    "VLLM_MORIIO_QP_PER_TRANSFER": lambda: int(  # Mori-IO连接器每次传输的队列对数量
        os.getenv("VLLM_MORIIO_QP_PER_TRANSFER", "1")  # 默认1
    ),
    # Controls the post-processing batch size for the Mori-IO connector
    "VLLM_MORIIO_POST_BATCH_SIZE": lambda: int(  # Mori-IO连接器的后处理批大小
        os.getenv("VLLM_MORIIO_POST_BATCH_SIZE", "-1")  # 默认-1
    ),
    # Controls the number of workers for Mori operations for the Mori-IO connector
    "VLLM_MORIIO_NUM_WORKERS": lambda: int(os.getenv("VLLM_MORIIO_NUM_WORKERS", "1")),  # Mori-IO连接器的工作线程数
    # Timeout (in seconds) for MooncakeConnector in PD disaggregated setup.
    "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": lambda: int(  # MooncakeConnector在PD分离设置中的超时时间（秒）
        os.getenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "480")  # 默认480秒
    ),
    # If set, it means we pre-downloaded cubin files and flashinfer will
    # read the cubin files directly.
    "VLLM_HAS_FLASHINFER_CUBIN": lambda: bool(  # 是否已预下载FlashInfer cubin文件
        int(os.getenv("VLLM_HAS_FLASHINFER_CUBIN", "0"))  # 默认未预下载
    ),
    # Supported options:
    # - "flashinfer-cudnn": use flashinfer cudnn GEMM backend
    # - "flashinfer-trtllm": use flashinfer trtllm GEMM backend
    # - "flashinfer-cutlass": use flashinfer cutlass GEMM backend
    # - "marlin": use marlin GEMM backend (for GPUs without native FP4 support)
    # - <none>: automatically pick an available backend
    "VLLM_NVFP4_GEMM_BACKEND": env_with_choices(  # NVFP4 GEMM后端选择
        "VLLM_NVFP4_GEMM_BACKEND",  # 环境变量名
        None,  # 默认自动选择
        [
            "flashinfer-cudnn",  # FlashInfer cuDNN后端
            "flashinfer-trtllm",  # FlashInfer TensorRT-LLM后端
            "flashinfer-cutlass",  # FlashInfer CUTLASS后端
            "cutlass",  # CUTLASS后端
            "marlin",  # Marlin后端（用于无原生FP4支持的GPU）
        ],
    ),
    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "VLLM_ENABLE_CUDAGRAPH_GC": lambda: bool(  # CUDA图捕获期间是否允许GC运行
        int(os.getenv("VLLM_ENABLE_CUDAGRAPH_GC", "0"))  # 默认冻结GC以加速捕获
    ),
    # Used to force set up loopback IP
    "VLLM_LOOPBACK_IP": lambda: os.getenv("VLLM_LOOPBACK_IP", ""),  # 强制设置环回IP地址
    # Used to set the process name prefix for vLLM processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "VLLM".
    "VLLM_PROCESS_NAME_PREFIX": lambda: os.getenv("VLLM_PROCESS_NAME_PREFIX", "VLLM"),  # vLLM进程名称前缀
    # Allow chunked local attention with hybrid kv cache manager.
    # Currently using the Hybrid KV cache manager with chunked local attention
    # in the Llama4 models (the only models currently using chunked local attn)
    # causes a latency regression. For this reason, we disable it by default.
    # This flag is used to allow users to enable it if they want to (to save on
    # kv-cache memory usage and enable longer contexts)
    # TODO(lucas): Remove this flag once latency regression is resolved.
    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(  # 是否允许混合KV缓存管理器的分块局部注意力
        int(os.getenv("VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "1"))  # 默认启用
    ),
    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, vLLM's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the vLLM server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "VLLM_ENABLE_RESPONSES_API_STORE": lambda: bool(  # 是否启用OpenAI Responses API的存储选项（注意：会导致内存泄漏）
        int(os.getenv("VLLM_ENABLE_RESPONSES_API_STORE", "0"))  # 默认禁用
    ),
    # If set, use the fp8 mfma in rocm paged attention.
    "VLLM_ROCM_FP8_MFMA_PAGE_ATTN": lambda: bool(  # 是否在ROCm分页注意力中使用FP8 MFMA
        int(os.getenv("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0"))  # 默认禁用
    ),
    # Whether to use pytorch symmetric memory for allreduce
    "VLLM_ALLREDUCE_USE_SYMM_MEM": lambda: bool(  # 是否使用PyTorch对称内存进行allreduce
        int(os.getenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1"))  # 默认启用
    ),
    # Whether to use FlashInfer allreduce
    "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(  # 是否使用FlashInfer进行allreduce
        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))  # 默认禁用
    ),
    # Experimental: use this to enable MCP tool calling for non harmony models
    "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": lambda: bool(  # 实验性：为非harmony模型启用MCP工具调用
        int(os.getenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0"))  # 默认禁用
    ),
    # Allows vllm to find tuned config under customized folder
    "VLLM_TUNED_CONFIG_FOLDER": lambda: os.getenv("VLLM_TUNED_CONFIG_FOLDER", None),  # 自定义调优配置文件夹路径
    # Valid values are container,code_interpreter,web_search_preview
    # ex VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=container,code_interpreter
    # If the server_label of your mcp tool is not in this list it will
    # be completely ignored.
    "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": env_set_with_choices(  # MCP工具的系统工具标签集合
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",  # 环境变量名
        default=[],  # 默认为空
        choices=["container", "code_interpreter", "web_search_preview"],  # 有效标签
    ),
    # Allows harmony instructions to be injected on system messages
    "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": lambda: bool(  # 是否在系统消息中注入harmony指令
        int(os.getenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0"))  # 默认禁用
    ),
    # Pin the conversation start date injected into the Harmony system
    # message. When unset the current date is used, which introduces
    # non-determinism (different tokens -> different model behaviour at
    # temperature=0). Set to an ISO date string, e.g. "2023-09-12",
    # for reproducible inference or testing.
    "VLLM_SYSTEM_START_DATE": lambda: os.getenv("VLLM_SYSTEM_START_DATE", None),  # 固定Harmony系统消息中的会话起始日期
    # Enable automatic retry when tool call JSON parsing fails
    # If enabled, returns an error message to the model to retry
    # If disabled (default), raises an exception and fails the request
    "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY": lambda: bool(  # 工具调用JSON解析失败时是否自动重试
        int(os.getenv("VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "0"))  # 默认不重试
    ),
    # Add optional custom scopes for profiling, disable to avoid overheads
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING": lambda: bool(  # 是否添加自定义性能分析作用域
        int(os.getenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "0"))  # 默认禁用以避免开销
    ),
    # Add optional nvtx scopes for profiling, disable to avoid overheads
    "VLLM_NVTX_SCOPES_FOR_PROFILING": lambda: bool(  # 是否添加NVTX性能分析作用域
        int(os.getenv("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))  # 默认禁用以避免开销
    ),
    # Represent block hashes in KV cache events as 64-bit integers instead of
    # raw bytes. Defaults to True for backward compatibility.
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": lambda: bool(  # KV缓存事件中块哈希是否使用64位整数
        int(os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))  # 默认启用
    ),
    # Name of the shared memory buffer used for object storage.
    # Only effective when mm_config.mm_processor_cache_type == "shm".
    # Automatically generates a unique UUID-based name per process tree
    # if not explicitly set.
    "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": get_env_or_set_default(  # 对象存储使用的共享内存缓冲区名称
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",  # 环境变量名
        lambda: f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{uuid.uuid4().hex}",  # 默认自动生成UUID名称
    ),
    # The size in MB of the buffers (NVL and RDMA) used by DeepEP
    "VLLM_DEEPEP_BUFFER_SIZE_MB": lambda: int(  # DeepEP使用的NVL和RDMA缓冲区大小（MB）
        os.getenv("VLLM_DEEPEP_BUFFER_SIZE_MB", "1024")  # 默认1024MB
    ),
    # Force DeepEP to use intranode kernel for inter-node communication in
    # high throughput mode. This is useful archive higher prefill throughput
    # on system supports multi-node nvlink (e.g GB200).
    "VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE": lambda: bool(  # DeepEP高吞吐模式是否强制使用节点内核进行节点间通信
        int(os.getenv("VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE", "0"))  # 默认禁用
    ),
    # Allow DeepEP to use MNNVL (multi-node nvlink) for internode_ll kernel,
    # turn this for better latency on GB200 like system
    "VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL": lambda: bool(  # DeepEP低延迟模式是否使用MNNVL进行节点间通信
        int(os.getenv("VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL", "0"))  # 默认禁用
    ),
    # The number of SMs to allocate for communication kernels when running DBO
    # the rest of the SMs on the device will be allocated to compute
    "VLLM_DBO_COMM_SMS": lambda: int(os.getenv("VLLM_DBO_COMM_SMS", "20")),  # DBO中分配给通信内核的SM数量
    # Enable max_autotune & coordinate_descent_tuning in inductor_config
    # to compile static shapes passed from compile_sizes in compilation_config
    # If set to 1, enable max_autotune; By default, this is enabled (1)
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": lambda: bool(  # 是否启用Inductor的max_autotune
        int(os.getenv("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"))  # 默认启用
    ),
    # If set to 1, enable coordinate_descent_tuning;
    # By default, this is enabled (1)
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": lambda: bool(  # 是否启用Inductor的坐标下降调优
        int(os.getenv("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"))  # 默认启用
    ),
    # Flag to enable NCCL symmetric memory allocation and registration
    "VLLM_USE_NCCL_SYMM_MEM": lambda: bool(  # 是否启用NCCL对称内存分配和注册
        int(os.getenv("VLLM_USE_NCCL_SYMM_MEM", "0"))  # 默认禁用
    ),
    # NCCL header path
    "VLLM_NCCL_INCLUDE_PATH": lambda: os.environ.get("VLLM_NCCL_INCLUDE_PATH", None),  # NCCL头文件路径
    # Flag to enable FBGemm kernels on model execution
    "VLLM_USE_FBGEMM": lambda: bool(int(os.getenv("VLLM_USE_FBGEMM", "0"))),  # 是否启用FBGemm内核
    # GC debug config
    # - VLLM_GC_DEBUG=0: disable GC debugger
    # - VLLM_GC_DEBUG=1: enable GC debugger with gc.collect elpased times
    # - VLLM_GC_DEBUG='{"top_objects":5}': enable GC debugger with
    #                                      top 5 collected objects
    "VLLM_GC_DEBUG": lambda: os.getenv("VLLM_GC_DEBUG", ""),  # GC调试配置
    # Debug workspace allocations.
    # logging of workspace resize operations.
    "VLLM_DEBUG_WORKSPACE": lambda: bool(int(os.getenv("VLLM_DEBUG_WORKSPACE", "0"))),  # 是否调试工作区分配的resize操作
    # Disables parallel execution of shared_experts via separate cuda stream
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(  # 是否禁用共享专家通过独立CUDA流的并行执行
        int(os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0"))  # 默认不禁用
    ),
    # Limits when we run shared_experts in a separate stream.
    # We found out that for large batch sizes, the separate stream
    # execution is not beneficial (most likely because of the input clone)
    # TODO(alexm-redhat): Tune to be more dynamic based on GPU type
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": lambda: int(  # 使用独立流运行共享专家的token数量阈值
        int(os.getenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256))  # 默认256
    ),
    # Format for saving torch.compile cache artifacts
    # - "binary": saves as binary file
    #     Safe for multiple vllm serve processes accessing the same torch compile cache.
    # - "unpacked": saves as directory structure (for inspection/debugging)
    #     NOT multiprocess safe - race conditions may occur with multiple processes.
    #     Allows viewing and setting breakpoints in Inductor's code output files.
    "VLLM_COMPILE_CACHE_SAVE_FORMAT": env_with_choices(  # torch.compile缓存产物的保存格式
        "VLLM_COMPILE_CACHE_SAVE_FORMAT", "binary", ["binary", "unpacked"]  # 可选：二进制或解包目录结构
    ),
    # Flag to enable v2 model runner.
    "VLLM_USE_V2_MODEL_RUNNER": lambda: bool(  # 是否启用v2模型运行器
        int(os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0"))  # 默认禁用
    ),
    # Log model inspection after loading.
    # If enabled, logs a transformers-style hierarchical view of the model
    # with quantization methods and attention backends.
    "VLLM_LOG_MODEL_INSPECTION": lambda: bool(  # 加载后是否记录模型检查日志
        int(os.getenv("VLLM_LOG_MODEL_INSPECTION", "0"))  # 默认禁用
    ),
    # Debug logging for --enable-mfu-metrics
    "VLLM_DEBUG_MFU_METRICS": lambda: bool(  # 是否启用MFU指标调试日志
        int(os.getenv("VLLM_DEBUG_MFU_METRICS", "0"))  # 默认禁用
    ),
    # Disable using pytorch's pin memory for CPU offloading.
    "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": lambda: bool(  # 是否禁用CPU卸载时的pin memory
        int(os.getenv("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "0"))  # 默认不禁用
    ),
    # Disable using UVA (Unified Virtual Addressing) for CPU offloading.
    "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": lambda: bool(  # 是否禁用CPU卸载时的UVA
        int(os.getenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "0"))  # 默认不禁用
    ),
    # Disable logging of vLLM logo at server startup time.
    "VLLM_DISABLE_LOG_LOGO": lambda: bool(int(os.getenv("VLLM_DISABLE_LOG_LOGO", "0"))),  # 是否禁用启动时的vLLM Logo日志
    # Disable PDL for LoRA, as enabling PDL with LoRA on SM100 causes
    # Triton compilation to fail.
    "VLLM_LORA_DISABLE_PDL": lambda: bool(int(os.getenv("VLLM_LORA_DISABLE_PDL", "0"))),  # 是否禁用LoRA的PDL
    # Enable CUDA compatibility mode for datacenter GPUs with older
    # driver versions than the CUDA toolkit major version of vLLM.
    "VLLM_ENABLE_CUDA_COMPATIBILITY": lambda: (  # 是否启用CUDA兼容模式
        os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower()  # 获取并规范化值
        in ("1", "true")  # 检查是否为真值
    ),
    # Path to the CUDA compatibility libraries when CUDA compatibility is enabled.
    "VLLM_CUDA_COMPATIBILITY_PATH": lambda: os.environ.get(  # CUDA兼容库路径
        "VLLM_CUDA_COMPATIBILITY_PATH", None  # 默认为None
    ),
    # Whether it is a scale up launch engine for elastic EP,
    # Should only be set by EngineCoreClient.
    "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": lambda: bool(  # 是否为弹性EP的扩容启动引擎
        int(os.getenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0"))  # 默认禁用
    ),
    # Whether to wait for all requests to drain before sending the
    # scaling command in elastic EP.
    "VLLM_ELASTIC_EP_DRAIN_REQUESTS": lambda: bool(  # 弹性EP缩容前是否等待所有请求排空
        int(os.getenv("VLLM_ELASTIC_EP_DRAIN_REQUESTS", "0"))  # 默认不等待
    ),
    # If set to 1, enable CUDA graph memory estimation during memory profiling.
    # This profiles CUDA graph memory usage to provide more accurate KV cache
    # memory allocation. Disabled by default to preserve existing behavior.
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": lambda: bool(  # 内存分析期间是否估算CUDA图内存使用
        int(os.getenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0"))  # 默认禁用
    ),
    # NIXL EP environment variables
    "VLLM_NIXL_EP_MAX_NUM_RANKS": lambda: int(  # NIXL EP最大rank数量
        os.getenv("VLLM_NIXL_EP_MAX_NUM_RANKS", "32")  # 默认32
    ),
}


# --8<-- [end:env-vars-definition]  # 环境变量定义区域结束标记


def __getattr__(name: str):
    """惰性获取环境变量的值。

    通过模块级别的__getattr__实现对环境变量的惰性求值。
    在enable_envs_cache()调用后（服务初始化后触发），
    所有环境变量的值将被缓存。

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:  # 如果请求的属性名在环境变量字典中
        return environment_variables[name]()  # 调用对应的lambda函数获取值
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # 否则抛出属性错误


def _is_envs_cache_enabled() -> bool:
    """检查__getattr__是否已被functools.cache包装。

    Returns:
        如果缓存已启用则返回True
    """
    global __getattr__  # 声明使用全局__getattr__
    return hasattr(__getattr__, "cache_clear")  # 检查是否有cache_clear属性（functools.cache的标志）


def enable_envs_cache() -> None:
    """启用环境变量缓存。

    启用缓存后可以避免每次访问环境变量时重新求值，提高性能。

    NOTE: Currently, it's invoked after service initialization to reduce
    runtime overhead. This also means that environment variables should NOT
    be updated after the service is initialized.

    注意：目前在服务初始化后调用以减少运行时开销。
    这意味着服务初始化后不应再更新环境变量。
    """
    if _is_envs_cache_enabled():  # 如果缓存已启用
        # Avoid wrapping functools.cache multiple times
        return  # 避免重复包装，直接返回
    # Tag __getattr__ with functools.cache
    global __getattr__  # 声明使用全局__getattr__
    __getattr__ = functools.cache(__getattr__)  # 用functools.cache包装__getattr__

    # Cache all environment variables
    for key in environment_variables:  # 遍历所有环境变量
        __getattr__(key)  # 预先缓存每个环境变量的值


def disable_envs_cache() -> None:
    """重置环境变量缓存。

    可用于在单元测试之间隔离环境。
    """
    global __getattr__  # 声明使用全局__getattr__
    # If __getattr__ is wrapped by functions.cache, unwrap the caching layer.
    if _is_envs_cache_enabled():  # 如果缓存已启用
        assert hasattr(__getattr__, "__wrapped__")  # 确认存在原始函数引用
        __getattr__ = __getattr__.__wrapped__  # 解除缓存包装，恢复原始函数


def __dir__():
    """返回模块中所有可用环境变量的名称列表。"""
    return list(environment_variables.keys())  # 返回环境变量字典的所有键


def is_set(name: str):
    """检查环境变量是否被显式设置。

    Args:
        name: 环境变量名称

    Returns:
        如果环境变量在os.environ中存在则返回True

    Raises:
        AttributeError: 如果name不是已知的环境变量
    """
    if name in environment_variables:  # 如果name在已知环境变量中
        return name in os.environ  # 检查是否在系统环境变量中存在
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # 否则抛出属性错误


def validate_environ(hard_fail: bool) -> None:
    """验证环境中是否存在未知的vLLM环境变量。

    Args:
        hard_fail: 如果为True，遇到未知变量时抛出异常；否则仅记录警告
    """
    for env in os.environ:  # 遍历所有系统环境变量
        if env.startswith("VLLM_") and env not in environment_variables:  # 如果是VLLM_前缀但未在已知列表中
            if hard_fail:  # 如果启用严格模式
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")  # 抛出异常
            else:  # 否则
                logger.warning("Unknown vLLM environment variable detected: %s", env)  # 记录警告日志


def compile_factors() -> dict[str, object]:
    """返回用于torch.compile缓存键的环境变量。

    从所有已知的vLLM环境变量开始，移除ignored_factors中的条目，
    然后对其余所有内容进行哈希。这确保缓存键在各个worker之间保持一致。

    Return env vars used for torch.compile cache keys.

    Start with every known vLLM env var; drop entries in `ignored_factors`;
    hash everything else. This keeps the cache key aligned across workers."""

    ignored_factors: set[str] = {  # 不纳入编译缓存键的环境变量集合
        "MAX_JOBS",  # 并行编译任务数（不影响编译结果）
        "VLLM_RPC_BASE_PATH",  # RPC基础路径
        "VLLM_USE_MODELSCOPE",  # ModelScope使用标志
        "VLLM_RINGBUFFER_WARNING_INTERVAL",  # 环形缓冲区警告间隔
        "VLLM_DEBUG_DUMP_PATH",  # 调试转储路径
        "VLLM_PORT",  # 通信端口
        "VLLM_CACHE_ROOT",  # 缓存根目录
        "LD_LIBRARY_PATH",  # 库路径
        "VLLM_SERVER_DEV_MODE",  # 开发模式
        "VLLM_DP_MASTER_IP",  # 数据并行主节点IP
        "VLLM_DP_MASTER_PORT",  # 数据并行主节点端口
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",  # 随机化DP虚拟输入
        "VLLM_CI_USE_S3",  # CI使用S3
        "VLLM_MODEL_REDIRECT_PATH",  # 模型重定向路径
        "VLLM_HOST_IP",  # 主机IP
        "VLLM_FORCE_AOT_LOAD",  # 强制AOT加载
        "S3_ACCESS_KEY_ID",  # S3访问密钥
        "S3_SECRET_ACCESS_KEY",  # S3秘密密钥
        "S3_ENDPOINT_URL",  # S3端点URL
        "VLLM_USAGE_STATS_SERVER",  # 统计服务器
        "VLLM_NO_USAGE_STATS",  # 禁用统计
        "VLLM_DO_NOT_TRACK",  # 禁止跟踪
        "VLLM_LOGGING_LEVEL",  # 日志级别
        "VLLM_LOGGING_PREFIX",  # 日志前缀
        "VLLM_LOGGING_STREAM",  # 日志流
        "VLLM_LOGGING_CONFIG_PATH",  # 日志配置路径
        "VLLM_LOGGING_COLOR",  # 日志颜色
        "VLLM_LOG_STATS_INTERVAL",  # 统计日志间隔
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",  # API响应调试日志
        "VLLM_TUNED_CONFIG_FOLDER",  # 调优配置文件夹
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",  # 引擎迭代超时
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",  # HTTP保活超时
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",  # 执行模型超时
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",  # 引擎死亡时保活
        "VLLM_IMAGE_FETCH_TIMEOUT",  # 图片获取超时
        "VLLM_VIDEO_FETCH_TIMEOUT",  # 视频获取超时
        "VLLM_AUDIO_FETCH_TIMEOUT",  # 音频获取超时
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS",  # 媒体URL重定向
        "VLLM_MEDIA_LOADING_THREAD_COUNT",  # 媒体加载线程数
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",  # 音频最大文件大小
        "VLLM_VIDEO_LOADER_BACKEND",  # 视频加载后端
        "VLLM_MEDIA_CONNECTOR",  # 媒体连接器
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",  # 共享内存缓冲区名
        "VLLM_ASSETS_CACHE",  # 资源缓存路径
        "VLLM_ASSETS_CACHE_MODEL_CLEAN",  # 资源缓存清理
        "VLLM_WORKER_MULTIPROC_METHOD",  # 多进程方法
        "VLLM_ENABLE_V1_MULTIPROCESSING",  # V1多进程
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",  # V1输出处理块大小
        "VLLM_CPU_KVCACHE_SPACE",  # CPU KV缓存空间
        "VLLM_CPU_MOE_PREPACK",  # CPU MoE预打包
        "VLLM_TEST_FORCE_LOAD_FORMAT",  # 测试强制加载格式
        "VLLM_ENABLE_CUDA_COMPATIBILITY",  # CUDA兼容模式
        "VLLM_CUDA_COMPATIBILITY_PATH",  # CUDA兼容路径
        "LOCAL_RANK",  # 本地rank
        "CUDA_VISIBLE_DEVICES",  # 可见CUDA设备
        "NO_COLOR",  # 禁用颜色
    }

    from vllm.config.utils import normalize_value  # 延迟导入值规范化函数

    factors: dict[str, object] = {}  # 创建存储编译因子的字典
    for factor, getter in environment_variables.items():  # 遍历所有环境变量
        if factor in ignored_factors:  # 如果在忽略列表中
            continue  # 跳过

        try:  # 尝试获取值
            raw = getter()  # 调用getter函数获取原始值
        except Exception as exc:  # pragma: no cover - defensive logging  # 捕获异常（防御性日志记录）
            logger.warning(  # 记录警告
                "Skipping environment variable %s while hashing compile factors: %s",  # 跳过该环境变量
                factor,  # 变量名
                exc,  # 异常信息
            )
            continue  # 跳过该变量

        factors[factor] = normalize_value(raw)  # 规范化值并存入字典

    ray_noset_env_vars = [  # Ray的NOSET环境变量列表（控制是否由Ray设置设备可见性）
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",  # NVIDIA GPU可见设备
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",  # AMD ROCr可见设备
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",  # AMD HIP可见设备
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",  # 华为昇腾NPU可见设备
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",  # Habana HPU可见模块
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",  # AWS Neuron可见核心
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",  # TPU可见芯片
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",  # Intel GPU设备选择
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",  # RBLN可见设备
    ]

    for var in ray_noset_env_vars:  # 遍历Ray NOSET环境变量
        factors[var] = normalize_value(os.getenv(var))  # 将规范化后的值加入因子字典

    return factors  # 返回编译因子字典
