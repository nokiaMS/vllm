# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime  # 导入datetime模块，用于获取时间戳
import json  # 导入json模块，用于JSON序列化和反序列化
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于操作系统相关功能
import platform  # 导入platform模块，用于获取平台信息
import time  # 导入time模块，用于时间相关操作
from enum import Enum  # 从enum模块导入Enum类，用于定义枚举
from pathlib import Path  # 从pathlib模块导入Path类，用于路径操作
from threading import Thread  # 从threading模块导入Thread类，用于多线程
from typing import Any  # 导入Any类型注解
from uuid import uuid4  # 从uuid模块导入uuid4函数，用于生成唯一标识符

import cpuinfo  # 导入cpuinfo库，用于获取CPU信息
import psutil  # 导入psutil库，用于获取系统资源信息
import requests  # 导入requests库，用于HTTP请求
import torch  # 导入PyTorch库

import vllm.envs as envs  # 导入vllm环境变量模块
from vllm.connections import global_http_connection  # 导入全局HTTP连接管理器
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.utils.platform_utils import cuda_get_device_properties  # 导入获取CUDA设备属性的工具函数
from vllm.utils.torch_utils import cuda_device_count_stateless  # 导入无状态获取CUDA设备数量的工具函数
from vllm.version import __version__ as VLLM_VERSION  # 导入vLLM版本号

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

_config_home = envs.VLLM_CONFIG_ROOT  # 获取vLLM配置根目录路径
_USAGE_STATS_JSON_PATH = os.path.join(_config_home, "usage_stats.json")  # 使用统计数据的JSON文件路径
_USAGE_STATS_DO_NOT_TRACK_PATH = os.path.join(_config_home, "do_not_track")  # 禁止追踪的标记文件路径
_USAGE_STATS_ENABLED = None  # 使用统计是否启用的全局标志，初始为None
_USAGE_STATS_SERVER = envs.VLLM_USAGE_STATS_SERVER  # 使用统计上报服务器地址

_GLOBAL_RUNTIME_DATA = dict[str, str | int | bool]()  # 全局运行时数据字典，用于存储每次心跳发送的数据

_USAGE_ENV_VARS_TO_COLLECT = [  # 需要收集的环境变量列表
    "VLLM_USE_MODELSCOPE",  # 是否使用ModelScope
    "VLLM_USE_FLASHINFER_SAMPLER",  # 是否使用FlashInfer采样器
    "VLLM_PP_LAYER_PARTITION",  # 流水线并行层分区配置
    "VLLM_USE_TRITON_AWQ",  # 是否使用Triton AWQ
    "VLLM_ENABLE_V1_MULTIPROCESSING",  # 是否启用V1多进程模式
]


def set_runtime_usage_data(key: str, value: str | int | bool) -> None:
    """设置全局运行时使用数据，该数据将随每次使用心跳一起发送。

    Args:
        key: 数据的键名。
        value: 数据的值，可以是字符串、整数或布尔值。
    """
    _GLOBAL_RUNTIME_DATA[key] = value  # 将键值对存入全局运行时数据字典


def is_usage_stats_enabled():
    """判断是否可以向服务器发送使用统计信息。

    判断逻辑如下：
    - 默认情况下应该启用。
    - 以下三个环境变量可以禁用：
        - VLLM_DO_NOT_TRACK=1
        - DO_NOT_TRACK=1
        - VLLM_NO_USAGE_STATS=1
    - 如果主目录中存在以下文件也会禁用：
        - $HOME/.config/vllm/do_not_track

    Returns:
        布尔值，表示使用统计是否启用。
    """
    global _USAGE_STATS_ENABLED  # 声明使用全局变量
    if _USAGE_STATS_ENABLED is None:  # 如果尚未确定启用状态
        do_not_track = envs.VLLM_DO_NOT_TRACK  # 检查VLLM_DO_NOT_TRACK环境变量
        no_usage_stats = envs.VLLM_NO_USAGE_STATS  # 检查VLLM_NO_USAGE_STATS环境变量
        do_not_track_file = os.path.exists(_USAGE_STATS_DO_NOT_TRACK_PATH)  # 检查禁止追踪标记文件是否存在

        _USAGE_STATS_ENABLED = not (do_not_track or no_usage_stats or do_not_track_file)  # 任一条件为真则禁用
    return _USAGE_STATS_ENABLED  # 返回启用状态


def _get_current_timestamp_ns() -> int:
    """获取当前UTC时间的纳秒级时间戳。

    Returns:
        当前UTC时间的纳秒级时间戳（整数）。
    """
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)  # 获取UTC时间戳并转换为纳秒


def _detect_cloud_provider() -> str:
    """检测当前运行的云服务提供商。

    首先通过读取DMI系统信息文件来识别云提供商，
    如果失败则通过环境变量进行检测。

    Returns:
        云服务提供商名称字符串，如"AWS"、"AZURE"、"GCP"等，
        无法识别时返回"UNKNOWN"。
    """
    # Try detecting through vendor file
    # 尝试通过供应商信息文件检测
    vendor_files = [  # DMI系统信息文件路径列表
        "/sys/class/dmi/id/product_version",  # 产品版本文件
        "/sys/class/dmi/id/bios_vendor",  # BIOS供应商文件
        "/sys/class/dmi/id/product_name",  # 产品名称文件
        "/sys/class/dmi/id/chassis_asset_tag",  # 机箱资产标签文件
        "/sys/class/dmi/id/sys_vendor",  # 系统供应商文件
    ]
    # Mapping of identifiable strings to cloud providers
    # 可识别字符串到云提供商的映射
    cloud_identifiers = {  # 云服务商标识字符串字典
        "amazon": "AWS",  # Amazon标识对应AWS
        "microsoft corporation": "AZURE",  # 微软标识对应Azure
        "google": "GCP",  # Google标识对应GCP
        "oraclecloud": "OCI",  # Oracle Cloud标识对应OCI
    }

    for vendor_file in vendor_files:  # 遍历所有供应商信息文件
        path = Path(vendor_file)  # 创建Path对象
        if path.is_file():  # 如果文件存在
            file_content = path.read_text().lower()  # 读取文件内容并转为小写
            for identifier, provider in cloud_identifiers.items():  # 遍历云服务商标识
                if identifier in file_content:  # 如果文件内容包含该标识
                    return provider  # 返回对应的云服务商名称

    # Try detecting through environment variables
    # 尝试通过环境变量检测
    env_to_cloud_provider = {  # 环境变量到云提供商的映射
        "RUNPOD_DC_ID": "RUNPOD",  # RunPod数据中心ID环境变量
    }
    for env_var, provider in env_to_cloud_provider.items():  # 遍历环境变量映射
        if os.environ.get(env_var):  # 如果该环境变量已设置
            return provider  # 返回对应的云服务商名称

    return "UNKNOWN"  # 无法识别时返回UNKNOWN


class UsageContext(str, Enum):
    """使用上下文枚举类，标识vLLM的不同使用场景。"""

    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"  # 未知上下文
    LLM_CLASS = "LLM_CLASS"  # 通过LLM类使用
    API_SERVER = "API_SERVER"  # 通过API服务器使用
    OPENAI_API_SERVER = "OPENAI_API_SERVER"  # 通过OpenAI兼容API服务器使用
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"  # 通过OpenAI批量运行器使用
    ENGINE_CONTEXT = "ENGINE_CONTEXT"  # 通过引擎上下文使用


class UsageMessage:
    """使用信息收集与上报类，收集平台信息并发送到使用统计服务器。"""

    def __init__(self) -> None:
        """初始化UsageMessage实例。

        初始化所有收集字段为None，包括环境信息、vLLM信息和元数据。
        注意：vLLM的服务器只支持扁平的键值对，不要使用嵌套字段。
        """
        # NOTE: vLLM's server _only_ support flat KV pair.
        # Do not use nested fields.
        # 注意：vLLM的服务器只支持扁平的键值对，不要使用嵌套字段。

        self.uuid = str(uuid4())  # 生成唯一标识符

        # Environment Information
        # 环境信息
        self.provider: str | None = None  # 云服务提供商
        self.num_cpu: int | None = None  # CPU数量
        self.cpu_type: str | None = None  # CPU型号
        self.cpu_family_model_stepping: str | None = None  # CPU家族、型号和步进信息
        self.total_memory: int | None = None  # 总内存大小
        self.architecture: str | None = None  # 系统架构
        self.platform: str | None = None  # 操作系统平台
        self.cuda_runtime: str | None = None  # CUDA运行时版本
        self.gpu_count: int | None = None  # GPU数量
        self.gpu_type: str | None = None  # GPU型号
        self.gpu_memory_per_device: int | None = None  # 每个GPU的显存大小
        self.env_var_json: str | None = None  # 环境变量JSON字符串

        # vLLM Information
        # vLLM信息
        self.model_architecture: str | None = None  # 模型架构名称
        self.vllm_version: str | None = None  # vLLM版本号
        self.context: str | None = None  # 使用上下文

        # Metadata
        # 元数据
        self.log_time: int | None = None  # 日志记录时间（纳秒时间戳）
        self.source: str | None = None  # 数据来源标识

    def report_usage(
        self,
        model_architecture: str,  # 模型架构名称
        usage_context: UsageContext,  # 使用上下文枚举值
        extra_kvs: dict[str, Any] | None = None,  # 额外的键值对数据
    ) -> None:
        """在后台线程中上报使用信息。

        启动一个守护线程来异步发送使用数据，避免阻塞主线程。

        Args:
            model_architecture: 模型架构名称。
            usage_context: 使用上下文。
            extra_kvs: 额外的键值对数据，可选。
        """
        t = Thread(  # 创建后台线程
            target=self._report_usage_worker,  # 线程目标函数
            args=(model_architecture, usage_context, extra_kvs or {}),  # 传递参数
            daemon=True,  # 设置为守护线程，主线程退出时自动结束
        )
        t.start()  # 启动线程

    def _report_usage_worker(
        self,
        model_architecture: str,  # 模型架构名称
        usage_context: UsageContext,  # 使用上下文
        extra_kvs: dict[str, Any],  # 额外的键值对数据
    ) -> None:
        """使用上报的工作线程函数。

        先执行一次性上报，然后进入持续上报循环。

        Args:
            model_architecture: 模型架构名称。
            usage_context: 使用上下文。
            extra_kvs: 额外的键值对数据。
        """
        self._report_usage_once(model_architecture, usage_context, extra_kvs)  # 执行一次性使用上报
        self._report_continuous_usage()  # 进入持续上报循环

    def _report_tpu_inference_usage(self) -> bool:
        """收集TPU推理相关的使用信息。

        尝试从tpu_inference模块获取TPU的芯片数量、类型和显存信息。

        Returns:
            如果成功收集TPU信息返回True，否则返回False。
        """
        try:
            from tpu_inference import tpu_info, utils  # 尝试导入TPU推理模块

            self.gpu_count = tpu_info.get_num_chips()  # 获取TPU芯片数量
            self.gpu_type = tpu_info.get_tpu_type()  # 获取TPU类型
            self.gpu_memory_per_device = utils.get_device_hbm_limit()  # 获取每个设备的HBM内存上限
            self.cuda_runtime = "tpu_inference"  # 设置运行时标识为tpu_inference
            return True  # 返回成功
        except Exception:  # 捕获所有异常
            return False  # 返回失败

    def _report_usage_once(
        self,
        model_architecture: str,  # 模型架构名称
        usage_context: UsageContext,  # 使用上下文
        extra_kvs: dict[str, Any],  # 额外的键值对数据
    ) -> None:
        """执行一次性使用信息上报。

        收集完整的平台信息、vLLM信息和环境变量，然后写入文件并发送到服务器。

        Args:
            model_architecture: 模型架构名称。
            usage_context: 使用上下文。
            extra_kvs: 额外的键值对数据。
        """
        # Platform information
        # 平台信息收集
        from vllm.platforms import current_platform  # 导入当前平台检测模块

        if current_platform.is_cuda_alike():  # 如果是CUDA兼容平台
            self.gpu_count = cuda_device_count_stateless()  # 获取GPU设备数量
            self.gpu_type, self.gpu_memory_per_device = cuda_get_device_properties(  # 获取GPU型号和显存
                0, ("name", "total_memory")  # 查询第一个设备的名称和总显存
            )
        if current_platform.is_cuda():  # 如果是CUDA平台
            self.cuda_runtime = torch.version.cuda  # 获取CUDA运行时版本
        if current_platform.is_tpu():  # noqa: SIM102  # 如果是TPU平台
            if not self._report_tpu_inference_usage():  # 尝试收集TPU信息
                logger.exception("Failed to collect TPU information")  # 如果失败则记录异常日志
        self.provider = _detect_cloud_provider()  # 检测云服务提供商
        self.architecture = platform.machine()  # 获取系统架构（如x86_64、aarch64）
        self.platform = platform.platform()  # 获取完整的平台信息字符串
        self.total_memory = psutil.virtual_memory().total  # 获取系统总物理内存

        info = cpuinfo.get_cpu_info()  # 获取CPU详细信息
        self.num_cpu = info.get("count", None)  # 获取CPU核心数
        self.cpu_type = info.get("brand_raw", "")  # 获取CPU品牌型号
        self.cpu_family_model_stepping = ",".join(  # 拼接CPU家族、型号和步进信息
            [
                str(info.get("family", "")),  # CPU家族
                str(info.get("model", "")),  # CPU型号
                str(info.get("stepping", "")),  # CPU步进
            ]
        )

        # vLLM information
        # vLLM信息收集
        self.context = usage_context.value  # 获取使用上下文的枚举值
        self.vllm_version = VLLM_VERSION  # 设置vLLM版本号
        self.model_architecture = model_architecture  # 设置模型架构名称

        # Environment variables
        # 环境变量收集
        self.env_var_json = json.dumps(  # 将环境变量序列化为JSON字符串
            {env_var: getattr(envs, env_var) for env_var in _USAGE_ENV_VARS_TO_COLLECT}  # 收集指定的环境变量
        )

        # Metadata
        # 元数据
        self.log_time = _get_current_timestamp_ns()  # 记录当前纳秒时间戳
        self.source = envs.VLLM_USAGE_SOURCE  # 获取使用数据来源标识

        data = vars(self)  # 将实例的所有属性转换为字典
        if extra_kvs:  # 如果有额外的键值对
            data.update(extra_kvs)  # 合并额外数据

        self._write_to_file(data)  # 将数据写入本地文件
        self._send_to_server(data)  # 将数据发送到服务器

    def _report_continuous_usage(self):
        """持续上报使用数据，每10分钟发送一次。

        这有助于收集vLLM使用的运行时间数据点。
        该函数还可以随时间发送性能指标。
        """
        while True:  # 无限循环
            time.sleep(600)  # 每10分钟（600秒）等待一次
            data = {  # 构建上报数据
                "uuid": self.uuid,  # 包含唯一标识符
                "log_time": _get_current_timestamp_ns(),  # 包含当前纳秒时间戳
            }
            data.update(_GLOBAL_RUNTIME_DATA)  # 合并全局运行时数据

            self._write_to_file(data)  # 将数据写入本地文件
            self._send_to_server(data)  # 将数据发送到服务器

    def _send_to_server(self, data: dict[str, Any]) -> None:
        """将使用数据发送到统计服务器。

        使用全局HTTP客户端发送POST请求。如果发送失败则静默忽略。

        Args:
            data: 要发送的使用数据字典。
        """
        try:
            global_http_client = global_http_connection.get_sync_client()  # 获取全局同步HTTP客户端
            global_http_client.post(_USAGE_STATS_SERVER, json=data)  # 发送POST请求到统计服务器
        except requests.exceptions.RequestException:  # 捕获HTTP请求异常
            # silently ignore unless we are using debug log
            # 静默忽略，除非开启了调试日志
            logging.debug("Failed to send usage data to server")  # 记录调试级别日志

    def _write_to_file(self, data: dict[str, Any]) -> None:
        """将使用数据写入本地JSON文件。

        以追加模式写入，每条记录占一行。

        Args:
            data: 要写入的使用数据字典。
        """
        os.makedirs(os.path.dirname(_USAGE_STATS_JSON_PATH), exist_ok=True)  # 创建目录（如果不存在）
        Path(_USAGE_STATS_JSON_PATH).touch(exist_ok=True)  # 创建文件（如果不存在）
        with open(_USAGE_STATS_JSON_PATH, "a") as f:  # 以追加模式打开文件
            json.dump(data, f)  # 将数据序列化为JSON并写入文件
            f.write("\n")  # 写入换行符，每条记录占一行


usage_message = UsageMessage()  # 创建全局UsageMessage单例实例
