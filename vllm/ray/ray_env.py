# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
import json  # 导入 JSON 序列化/反序列化模块
import os  # 导入操作系统接口模块

import vllm.envs as envs  # 导入 vLLM 环境变量配置模块
from vllm.logger import init_logger  # 从 vLLM 日志模块导入日志初始化函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

CONFIG_HOME = envs.VLLM_CONFIG_ROOT  # 获取 vLLM 配置文件根目录路径

# Env vars that should NOT be copied from the driver to Ray workers.
# 不应从驱动进程复制到 Ray Worker 的环境变量列表的配置文件路径
RAY_NON_CARRY_OVER_ENV_VARS_FILE = os.path.join(  # 拼接配置文件完整路径
    CONFIG_HOME, "ray_non_carry_over_env_vars.json"  # 配置文件名
)

try:  # 尝试加载不需要传递的环境变量配置
    if os.path.exists(RAY_NON_CARRY_OVER_ENV_VARS_FILE):  # 检查配置文件是否存在
        with open(RAY_NON_CARRY_OVER_ENV_VARS_FILE) as f:  # 打开配置文件
            RAY_NON_CARRY_OVER_ENV_VARS = set(json.load(f))  # 从 JSON 文件加载并转换为集合
    else:  # 配置文件不存在的情况
        RAY_NON_CARRY_OVER_ENV_VARS = set()  # 使用空集合作为默认值
except json.JSONDecodeError:  # 捕获 JSON 解析错误
    logger.warning(  # 记录警告日志
        "Failed to parse %s. Using an empty set for non-carry-over env vars.",  # 解析失败的警告信息
        RAY_NON_CARRY_OVER_ENV_VARS_FILE,  # 失败的文件路径
    )
    RAY_NON_CARRY_OVER_ENV_VARS = set()  # 解析失败时使用空集合

# ---------------------------------------------------------------------------
# Built-in defaults for env var propagation.
# 环境变量传播的内置默认配置。
# Users can add more via VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY and
# VLLM_RAY_EXTRA_ENV_VARS_TO_COPY (additive, not replacing).
# 用户可以通过 VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY 和
# VLLM_RAY_EXTRA_ENV_VARS_TO_COPY 添加更多（追加模式，不会替换）。
# ---------------------------------------------------------------------------
DEFAULT_ENV_VAR_PREFIXES: set[str] = {  # 默认需要传播的环境变量前缀集合
    "VLLM_",  # vLLM 相关环境变量前缀
    "LMCACHE_",  # LMCache 相关环境变量前缀
    "NCCL_",  # NCCL 通信库相关环境变量前缀
    "UCX_",  # UCX 通信框架相关环境变量前缀
    "HF_",  # HuggingFace 相关环境变量前缀
    "HUGGING_FACE_",  # HuggingFace 相关环境变量前缀（完整形式）
}

DEFAULT_EXTRA_ENV_VARS: set[str] = {  # 默认需要额外传播的单独环境变量集合
    "PYTHONHASHSEED",  # Python 哈希种子，用于确保跨进程的一致性
}


def _parse_csv(value: str) -> set[str]:
    """将逗号分隔的字符串解析为去除空白的非空字符串集合。

    Args:
        value: 逗号分隔的字符串。

    Returns:
        解析后的非空字符串集合。
    """
    return {tok.strip() for tok in value.split(",") if tok.strip()}  # 按逗号分割，去除空白，过滤空字符串


def get_env_vars_to_copy(
    exclude_vars: set[str] | None = None,  # 需要排除的环境变量集合
    additional_vars: set[str] | None = None,  # 需要额外添加的环境变量集合
    destination: str | None = None,  # 目标标签，仅用于日志消息
) -> set[str]:
    """返回需要从驱动进程复制到 Ray Actor 的环境变量名称集合。

    结果是以下来源的并集：

    1. 在 ``vllm.envs.environment_variables`` 中注册的环境变量。
    2. 在 ``os.environ`` 中匹配 ``DEFAULT_ENV_VAR_PREFIXES`` +
       ``VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY`` 前缀的环境变量。
    3. ``DEFAULT_EXTRA_ENV_VARS`` +
       ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` 中的单独变量名。
    4. 调用方提供的 *additional_vars*（例如平台特定的变量）。

    减去 *exclude_vars* 和 ``RAY_NON_CARRY_OVER_ENV_VARS`` 中的名称。

    Args:
        exclude_vars: 需要排除的环境变量（例如 Worker 特定的变量）。
        additional_vars: 需要额外复制的环境变量名称。适用于
            调用方特定的变量（例如平台环境变量）。
        destination: 仅在日志消息中使用的标签。
    """
    exclude = (exclude_vars or set()) | RAY_NON_CARRY_OVER_ENV_VARS  # 合并排除集合和配置文件中的排除列表

    # -- prefixes (built-in + user-supplied, additive) ----------------------
    # -- 前缀（内置 + 用户提供，追加模式）--------------------------------------
    prefixes = DEFAULT_ENV_VAR_PREFIXES | _parse_csv(  # 合并默认前缀和用户自定义前缀
        envs.VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY  # 用户通过环境变量指定的额外前缀
    )

    # -- collect env var names ----------------------------------------------
    # -- 收集环境变量名称 ------------------------------------------------------
    # 1. vLLM's registered env vars
    # 1. vLLM 注册的环境变量
    result = set(envs.environment_variables)  # 获取 vLLM 注册的所有环境变量名
    # 2. Prefix-matched vars present in the current environment
    # 2. 当前环境中匹配前缀的环境变量
    result |= {name for name in os.environ if any(name.startswith(p) for p in prefixes)}  # 添加所有匹配前缀的环境变量
    # 3. Individual extra vars (built-in + user-supplied, additive)
    # 3. 单独的额外变量（内置 + 用户提供，追加模式）
    result |= DEFAULT_EXTRA_ENV_VARS | _parse_csv(envs.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY)  # 合并默认额外变量和用户指定的额外变量
    # 4. Caller-supplied extra vars (e.g. platform-specific)
    # 4. 调用方提供的额外变量（例如平台特定的变量）
    result |= additional_vars or set()  # 添加调用方指定的额外变量
    # 5. Exclude worker-specific and user-blacklisted vars
    # 5. 排除 Worker 特定的和用户黑名单中的变量
    result -= exclude  # 移除需要排除的变量

    # -- logging ------------------------------------------------------------
    # -- 日志记录 --------------------------------------------------------------
    dest = f" to {destination}" if destination else ""  # 如果指定了目标则格式化日志中的目标字符串
    logger.info("Env var prefixes to copy: %s", sorted(prefixes))  # 记录要复制的环境变量前缀
    logger.info(  # 记录实际要复制的环境变量列表
        "Copying the following environment variables%s: %s",  # 日志格式字符串
        dest,  # 目标标签
        sorted(v for v in result if v in os.environ),  # 只记录当前环境中实际存在的变量
    )
    if RAY_NON_CARRY_OVER_ENV_VARS:  # 如果存在不需要传递的环境变量
        logger.info(  # 记录配置文件中排除的环境变量
            "RAY_NON_CARRY_OVER_ENV_VARS from config: %s",  # 日志格式字符串
            RAY_NON_CARRY_OVER_ENV_VARS,  # 排除的环境变量集合
        )
    logger.info(  # 记录如何排除环境变量的提示信息
        "To exclude env vars from copying, add them to %s",  # 提示用户如何排除环境变量
        RAY_NON_CARRY_OVER_ENV_VARS_FILE,  # 配置文件路径
    )

    return result  # 返回最终需要复制的环境变量名称集合
