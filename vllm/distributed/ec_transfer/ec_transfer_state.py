# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING  # 导入类型检查标记

from vllm.distributed.ec_transfer.ec_connector.base import (  # 从EC连接器基础模块导入
    ECConnectorBase,  # EC连接器基类
    ECConnectorRole,  # EC连接器角色枚举
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory  # 导入EC连接器工厂类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from vllm.config import VllmConfig  # 导入vLLM配置类

_EC_CONNECTOR_AGENT: ECConnectorBase | None = None  # 全局EC连接器代理实例，初始为None


def get_ec_transfer() -> ECConnectorBase:
    """获取EC传输连接器实例。

    Returns:
        ECConnectorBase: EC连接器实例

    Raises:
        AssertionError: 如果EC缓存传输未初始化
    """
    assert _EC_CONNECTOR_AGENT is not None, "disaggregated EC cache is not initialized"  # 断言EC连接器已初始化
    return _EC_CONNECTOR_AGENT  # 返回EC连接器实例


def has_ec_transfer() -> bool:
    """检查EC传输连接器是否已初始化。

    Returns:
        bool: 如果EC连接器已初始化则返回True
    """
    return _EC_CONNECTOR_AGENT is not None  # 返回EC连接器是否存在


def ensure_ec_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """确保EC缓存连接器已初始化。

    如果EC传输配置存在且当前实例是EC传输实例，则创建EC连接器。

    Args:
        vllm_config: vLLM配置对象
    """

    global _EC_CONNECTOR_AGENT  # 声明使用全局变量

    if vllm_config.ec_transfer_config is None:  # 如果没有EC传输配置
        return  # 直接返回

    if (
        vllm_config.ec_transfer_config.is_ec_transfer_instance  # 如果当前实例是EC传输实例
        and _EC_CONNECTOR_AGENT is None  # 且EC连接器尚未初始化
    ):
        _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(  # 通过工厂创建EC连接器
            config=vllm_config, role=ECConnectorRole.WORKER  # 指定配置和Worker角色
        )
