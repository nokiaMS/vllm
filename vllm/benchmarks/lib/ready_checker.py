# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
"""Utilities for checking endpoint readiness."""  # 用于检查端点就绪状态的工具
"""端点就绪状态检查工具。"""

import asyncio  # 导入异步IO模块
import time  # 导入时间模块

import aiohttp  # 导入异步HTTP客户端库
from tqdm.asyncio import tqdm  # 导入异步进度条

from vllm.logger import init_logger  # 从vLLM导入日志初始化器

from .endpoint_request_func import RequestFunc, RequestFuncInput, RequestFuncOutput  # 导入请求函数相关类

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


async def wait_for_endpoint(
    request_func: RequestFunc,  # 异步请求函数
    test_input: RequestFuncInput,  # 测试用的请求输入
    session: aiohttp.ClientSession,  # HTTP客户端会话
    timeout_seconds: int = 600,  # 超时时间，默认600秒
    retry_interval: int = 5,  # 重试间隔，默认5秒
) -> RequestFuncOutput:  # 返回请求函数输出
    """
    Wait for an endpoint to become available before starting benchmarks.
    等待端点变为可用状态，然后再开始基准测试。

    Args:
        request_func: The async request function to call
            异步请求函数
        test_input: The RequestFuncInput to test with
            用于测试的请求输入
        timeout_seconds: Maximum time to wait in seconds (default: 10 minutes)
            最大等待时间（秒），默认10分钟
        retry_interval: Time between retries in seconds (default: 5 seconds)
            重试间隔时间（秒），默认5秒

    Returns:
        RequestFuncOutput: The successful response
        成功的响应结果

    Raises:
        ValueError: If the endpoint doesn't become available within the timeout
        如果端点在超时时间内未变为可用状态则抛出ValueError
    """
    deadline = time.perf_counter() + timeout_seconds  # 计算截止时间
    output = RequestFuncOutput(success=False)  # 初始化输出为失败状态
    print(f"Waiting for endpoint to become up in {timeout_seconds} seconds")  # 打印等待信息

    with tqdm(  # 创建进度条
        total=timeout_seconds,  # 总时间为超时时间
        bar_format="{desc} |{bar}| {elapsed} elapsed, {remaining} remaining",  # 进度条格式
        unit="s",  # 单位为秒
    ) as pbar:  # 使用进度条上下文管理器
        while True:  # 无限循环直到超时或成功
            # update progress bar
            remaining = deadline - time.perf_counter()  # 计算剩余时间
            elapsed = timeout_seconds - remaining  # 计算已过时间
            update_amount = min(elapsed - pbar.n, timeout_seconds - pbar.n)  # 计算进度条更新量
            pbar.update(update_amount)  # 更新进度条
            pbar.refresh()  # 刷新进度条显示
            if remaining <= 0:  # 如果已超时
                pbar.close()  # 关闭进度条
                break  # 退出循环

            # ping the endpoint using request_func
            try:  # 尝试发送请求
                output = await request_func(  # 调用异步请求函数
                    request_func_input=test_input, session=session  # 传入测试输入和会话
                )
                if output.success:  # 如果请求成功
                    pbar.close()  # 关闭进度条
                    return output  # 返回成功的输出
                else:  # 如果请求失败
                    err_last_line = str(output.error).rstrip().rsplit("\n", 1)[-1]  # 获取错误信息的最后一行
                    logger.warning("Endpoint is not ready. Error='%s'", err_last_line)  # 记录警告日志
            except aiohttp.ClientConnectorError:  # 捕获连接错误
                pass  # 忽略连接错误，继续重试

            # retry after a delay
            sleep_duration = min(retry_interval, remaining)  # 计算睡眠时间
            if sleep_duration > 0:  # 如果需要等待
                await asyncio.sleep(sleep_duration)  # 异步等待

    return output  # 返回最终输出（可能失败）
