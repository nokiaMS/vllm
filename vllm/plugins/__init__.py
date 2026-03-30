# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging  # 导入日志模块
from collections.abc import Callable  # 导入可调用对象抽象基类
from typing import Any  # 导入Any类型提示

import vllm.envs as envs  # 导入vLLM环境变量模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# Default plugins group will be loaded in all processes(process0, engine core
# process and worker processes)
DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"  # 默认插件组，在所有进程中加载
# IO processor plugins group will be loaded in process0 only
IO_PROCESSOR_PLUGINS_GROUP = "vllm.io_processor_plugins"  # IO处理器插件组，仅在进程0中加载
# Platform plugins group will be loaded in all processes when
# `vllm.platforms.current_platform` is called and the value not initialized,
PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"  # 平台插件组，当调用current_platform且值未初始化时加载
# Stat logger plugins group will be loaded in process0 only when serve vLLM with
# async mode.
STAT_LOGGER_PLUGINS_GROUP = "vllm.stat_logger_plugins"  # 统计日志插件组，仅在异步模式的进程0中加载

# make sure one process only loads plugins once
plugins_loaded = False  # 确保每个进程只加载一次插件的标志


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:  # 按分组加载插件
    """Load plugins registered under the given entry point group.
    加载给定入口点分组下注册的插件。"""
    from importlib.metadata import entry_points  # 导入入口点发现功能

    allowed_plugins = envs.VLLM_PLUGINS  # 获取允许的插件列表

    discovered_plugins = entry_points(group=group)  # 发现指定分组的所有插件
    if len(discovered_plugins) == 0:  # 如果没有发现任何插件
        logger.debug("No plugins for group %s found.", group)  # 记录调试日志
        return {}  # 返回空字典

    # Check if the only discovered plugin is the default one
    is_default_group = group == DEFAULT_PLUGINS_GROUP  # 检查是否是默认插件组
    # Use INFO for non-default groups and DEBUG for the default group
    log_level = logger.debug if is_default_group else logger.info  # 根据分组选择日志级别

    log_level("Available plugins for group %s:", group)  # 记录可用插件信息
    for plugin in discovered_plugins:  # 遍历所有发现的插件
        log_level("- %s -> %s", plugin.name, plugin.value)  # 记录插件名称和值

    if allowed_plugins is None:  # 如果没有设置允许的插件列表
        log_level(  # 记录所有插件将被加载的信息
            "All plugins in this group will be loaded. "
            "Set `VLLM_PLUGINS` to control which plugins to load."
        )

    plugins = dict[str, Callable[[], Any]]()  # 创建插件字典
    for plugin in discovered_plugins:  # 遍历所有发现的插件
        if allowed_plugins is None or plugin.name in allowed_plugins:  # 如果插件被允许
            if allowed_plugins is not None:  # 如果设置了允许列表
                log_level("Loading plugin %s", plugin.name)  # 记录正在加载的插件

            try:
                func = plugin.load()  # 加载插件
                plugins[plugin.name] = func  # 将插件添加到字典
            except Exception:  # 捕获加载异常
                logger.exception("Failed to load plugin %s", plugin.name)  # 记录异常日志

    return plugins  # 返回已加载的插件字典


def load_general_plugins():  # 加载通用插件
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    警告：插件可能在不同进程中被多次加载。它们应该被设计为可以多次加载而不会引起问题。
    """
    global plugins_loaded  # 使用全局变量
    if plugins_loaded:  # 如果已经加载过
        return  # 直接返回
    plugins_loaded = True  # 标记为已加载

    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)  # 加载默认插件组
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():  # 遍历所有插件函数
        func()  # 执行插件函数
