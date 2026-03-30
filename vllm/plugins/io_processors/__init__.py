# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging  # 导入日志模块

from vllm.config import VllmConfig  # 导入vLLM总配置类
from vllm.plugins import IO_PROCESSOR_PLUGINS_GROUP, load_plugins_by_group  # 导入IO处理器插件组名和插件加载函数
from vllm.plugins.io_processors.interface import IOProcessor  # 导入IO处理器接口类
from vllm.renderers import BaseRenderer  # 导入基础渲染器类
from vllm.utils.import_utils import resolve_obj_by_qualname  # 导入通过全限定名解析对象的工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def get_io_processor(  # 获取IO处理器实例
    vllm_config: VllmConfig,  # vLLM配置对象
    renderer: BaseRenderer,  # 渲染器实例
    plugin_from_init: str | None = None,  # 从初始化时传入的插件名称
) -> IOProcessor | None:
    """获取IO处理器实例。

    IO处理器作为插件在'vllm.io_processor_plugins'组下加载。
    类似于平台插件，这些插件注册一个返回处理器类名的函数。
    """
    # Input.Output processors are loaded as plugins under the
    # 'vllm.io_processor_plugins' group. Similar to platform
    # plugins, these plugins register a function that returns the class
    # name for the processor to install.

    if plugin_from_init:  # 如果从初始化传入了插件名称
        model_plugin = plugin_from_init  # 使用传入的插件名称
    else:
        # A plugin can be specified via the model config
        # Retrieve the model specific plugin if available
        # This is using a custom field in the hf_config for the model
        hf_config = vllm_config.model_config.hf_config.to_dict()  # 获取HuggingFace配置字典
        config_plugin = hf_config.get("io_processor_plugin")  # 从配置中获取IO处理器插件名称
        model_plugin = config_plugin  # 使用配置中的插件名称

    if model_plugin is None:  # 如果没有请求任何IO处理器插件
        logger.debug("No IOProcessor plugins requested by the model")  # 记录调试日志
        return None  # 返回None

    logger.debug("IOProcessor plugin to be loaded %s", model_plugin)  # 记录要加载的插件名称

    # Load all installed plugin in the group
    multimodal_data_processor_plugins = load_plugins_by_group(  # 加载该组中所有已安装的插件
        IO_PROCESSOR_PLUGINS_GROUP
    )

    loadable_plugins = {}  # 可加载的插件字典
    for name, func in multimodal_data_processor_plugins.items():  # 遍历所有发现的插件
        try:
            assert callable(func)  # 断言插件函数是可调用的
            processor_cls_qualname = func()  # 调用插件函数获取处理器类的全限定名
            if processor_cls_qualname is not None:  # 如果获取到了类名
                loadable_plugins[name] = processor_cls_qualname  # 添加到可加载插件字典
        except Exception:  # 捕获加载异常
            logger.warning("Failed to load plugin %s.", name, exc_info=True)  # 记录警告日志

    num_available_plugins = len(loadable_plugins.keys())  # 计算可用插件数量
    if num_available_plugins == 0:  # 如果没有可用的插件
        raise ValueError(  # 抛出值错误
            f"No IOProcessor plugins installed but one is required ({model_plugin})."
        )

    if model_plugin not in loadable_plugins:  # 如果所需的插件不在可用列表中
        raise ValueError(  # 抛出值错误
            f"The model requires the '{model_plugin}' IO Processor plugin "
            "but it is not installed. "
            f"Available plugins: {list(loadable_plugins.keys())}"
        )

    activated_plugin_cls = resolve_obj_by_qualname(loadable_plugins[model_plugin])  # 通过全限定名解析插件类

    return activated_plugin_cls(vllm_config, renderer)  # 实例化并返回IO处理器
