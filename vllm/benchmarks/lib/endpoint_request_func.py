# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
"""The request function for API endpoints."""  # API端点的请求函数模块
"""API端点请求函数模块。"""

import io  # 导入IO流模块
import json  # 导入JSON处理模块
import os  # 导入操作系统接口模块
import sys  # 导入系统模块
import time  # 导入时间模块
import traceback  # 导入异常追踪模块
from collections.abc import Awaitable  # 导入可等待对象类型
from dataclasses import dataclass, field  # 导入数据类装饰器和字段
from typing import Any, Literal, Protocol  # 导入类型标注工具

import aiohttp  # 导入异步HTTP客户端库
import regex as re  # 导入正则表达式模块
from tqdm.asyncio import tqdm  # 导入异步进度条

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)  # 设置HTTP超时为6小时


class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available."""
    """流式HTTP响应处理器，通过累积数据块直到完整消息可用来处理流式响应。"""

    def __init__(self):  # 初始化方法
        """
        初始化流式响应处理器，创建空缓冲区。
        """
        self.buffer = ""  # 初始化缓冲区为空字符串

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:  # 添加数据块方法
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        """将字节数据块添加到缓冲区并返回所有完整的消息。"""
        chunk_str = chunk_bytes.decode("utf-8")  # 将字节解码为UTF-8字符串
        self.buffer += chunk_str  # 追加到缓冲区

        messages = []  # 初始化消息列表

        # Split by double newlines (SSE message separator)
        while "\n\n" in self.buffer:  # 当缓冲区中有双换行符（SSE消息分隔符）
            message, self.buffer = self.buffer.split("\n\n", 1)  # 分割出第一条消息
            message = message.strip()  # 去除首尾空白
            if message:  # 如果消息不为空
                messages.append(message)  # 添加到消息列表

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith("data: "):  # 如果缓冲区以"data: "开头
            message_content = self.buffer.removeprefix("data: ").strip()  # 移除前缀并去除空白
            if message_content == "[DONE]":  # 如果是结束标记
                messages.append(self.buffer.strip())  # 添加到消息列表
                self.buffer = ""  # 清空缓冲区
            elif message_content:  # 如果有内容
                try:  # 尝试解析JSON
                    json.loads(message_content)  # 验证是否为有效JSON
                    messages.append(self.buffer.strip())  # 如果有效则添加到消息列表
                    self.buffer = ""  # 清空缓冲区
                except json.JSONDecodeError:  # 如果JSON解析失败
                    # Incomplete JSON, wait for more chunks.
                    pass  # 不完整的JSON，等待更多数据块

        return messages  # 返回完整消息列表


@dataclass
class RequestFuncInput:
    """The input for the request function."""
    """请求函数的输入数据类。"""

    prompt: str | list[str]  # 提示文本或提示文本列表
    api_url: str  # API端点URL
    prompt_len: int  # 提示文本长度（token数）
    output_len: int  # 期望输出长度（token数）
    model: str  # 模型标识符
    model_name: str | None = None  # 模型名称，可选
    logprobs: int | None = None  # 返回的log概率数量，可选
    extra_headers: dict | None = None  # 额外的HTTP请求头，可选
    extra_body: dict | None = None  # 额外的请求体参数，可选
    multi_modal_content: dict | list[dict] | None = None  # 多模态内容，可选
    ignore_eos: bool = False  # 是否忽略结束标记
    language: str | None = None  # 语言设置，可选
    request_id: str | None = None  # 请求ID，可选


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""
    """请求函数的输出数据类，包含性能指标。"""

    generated_text: str = ""  # 生成的文本内容
    success: bool = False  # 请求是否成功
    latency: float = 0.0  # 延迟时间（秒）
    output_tokens: int = 0  # 输出的token数量
    ttft: float = 0.0  # Time to first token，首个token的延迟时间
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies，token间延迟列表
    tpot: float = 0.0  # avg next-token latencies，平均下一个token延迟
    prompt_len: int = 0  # 提示文本长度
    error: str = ""  # 错误信息
    start_time: float = 0.0  # 请求开始时间
    input_audio_duration: float = 0.0  # in seconds，输入音频时长（秒）


class RequestFunc(Protocol):
    """
    请求函数的协议定义，规定了请求函数的调用签名。
    """
    def __call__(  # 定义可调用协议
        self,
        request_func_input: RequestFuncInput,  # 请求输入
        session: aiohttp.ClientSession,  # HTTP会话
        pbar: tqdm | None = None,  # 可选的进度条
    ) -> Awaitable[RequestFuncOutput]: ...  # 返回可等待的请求输出


def _validate_api_url(
    api_url: str,  # 待验证的API URL
    api_name: str,  # API名称，用于错误消息
    expected_suffixes: str | set[str],  # 期望的URL后缀
) -> None:  # 无返回值
    """
    验证API URL是否以期望的后缀结尾。
    """
    if isinstance(expected_suffixes, str):  # 如果是单个字符串
        expected_suffixes = {expected_suffixes}  # 转换为集合

    expected_suffixes = {*expected_suffixes, "profile"}  # 添加"profile"后缀

    if not api_url.endswith(tuple(expected_suffixes)):  # 检查URL后缀
        raise ValueError(f"{api_name} URL must end with one of: {expected_suffixes}.")  # 抛出值错误


def _update_payload_common(
    payload: dict[str, Any],  # 请求负载字典
    request_func_input: RequestFuncInput,  # 请求函数输入
) -> None:  # 无返回值
    """
    更新请求负载的通用字段，如ignore_eos和extra_body。
    """
    if request_func_input.ignore_eos:  # 如果设置了忽略结束标记
        payload["ignore_eos"] = request_func_input.ignore_eos  # 添加到负载
    if request_func_input.extra_body:  # 如果有额外的请求体
        payload.update(request_func_input.extra_body)  # 合并到负载


def _update_headers_common(
    headers: dict[str, Any],  # HTTP请求头字典
    request_func_input: RequestFuncInput,  # 请求函数输入
) -> None:  # 无返回值
    """
    更新HTTP请求头的通用字段，如额外请求头和请求ID。
    """
    if request_func_input.extra_headers:  # 如果有额外请求头
        headers |= request_func_input.extra_headers  # 合并额外请求头
    if request_func_input.request_id:  # 如果有请求ID
        headers["x-request-id"] = request_func_input.request_id  # 设置请求ID头


def _get_headers(content_type: str | None = None) -> dict[str, str]:
    """
    获取HTTP请求头，包括内容类型和API密钥认证。
    """
    headers = {}  # 初始化请求头字典
    if content_type:  # 如果指定了内容类型
        headers["Content-Type"] = content_type  # 设置Content-Type
    api_key = os.environ.get("OPENAI_API_KEY")  # 从环境变量获取API密钥
    if api_key:  # 如果有API密钥
        headers["Authorization"] = f"Bearer {api_key}"  # 设置Bearer认证头
    return headers  # 返回请求头


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """The async request function for the OpenAI Completions API.
    OpenAI Completions API的异步请求函数。

    Args:
        request_func_input: The input for the request function.
            请求函数的输入
        pbar: The progress bar to display the progress.
            显示进度的进度条

    Returns:
        The output of the request function.
        请求函数的输出
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "OpenAI Completions API", "completions")  # 验证URL

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "prompt": request_func_input.prompt,  # 提示文本
        "repetition_penalty": 1.0,  # 重复惩罚系数
        "max_tokens": request_func_input.output_len,  # 最大输出token数
        "logprobs": request_func_input.logprobs,  # log概率数量
        "stream": True,  # 启用流式输出
        "stream_options": {  # 流式选项
            "include_usage": True,  # 包含使用量信息
        },
    }
    _update_payload_common(payload, request_func_input)  # 更新通用负载字段

    headers = _get_headers()  # 获取请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    output = RequestFuncOutput()  # 创建输出对象
    output.prompt_len = request_func_input.prompt_len  # 设置提示长度

    generated_text = ""  # 初始化生成文本
    st = time.perf_counter()  # 记录开始时间
    output.start_time = st  # 设置开始时间
    most_recent_timestamp = st  # 记录最近的时间戳
    try:  # 尝试发送请求
        async with session.post(url=api_url, json=payload, headers=headers) as response:  # 发送POST请求
            if response.status == 200:  # 如果响应成功
                first_chunk_received = False  # 标记是否收到首个数据块
                handler = StreamedResponseHandler()  # 创建流式响应处理器

                async for chunk_bytes in response.content.iter_any():  # 异步迭代响应内容
                    chunk_bytes = chunk_bytes.strip()  # 去除首尾空白
                    if not chunk_bytes:  # 如果数据块为空
                        continue  # 跳过

                    messages = handler.add_chunk(chunk_bytes)  # 处理数据块获取完整消息
                    for message in messages:  # 遍历消息
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):  # SSE注释以冒号开头，跳过
                            continue  # 跳过SSE注释

                        chunk = message.removeprefix("data: ")  # 移除"data: "前缀

                        if chunk != "[DONE]":  # 如果不是结束标记
                            data = json.loads(chunk)  # 解析JSON数据

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):  # 如果有choices字段
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")  # 获取生成的文本
                                timestamp = time.perf_counter()  # 记录当前时间
                                # First token
                                if not first_chunk_received:  # 如果是首个token
                                    first_chunk_received = True  # 标记已收到首个数据块
                                    ttft = time.perf_counter() - st  # 计算首token延迟
                                    output.ttft = ttft  # 设置首token延迟

                                # Decoding phase
                                else:  # 解码阶段
                                    output.itl.append(timestamp - most_recent_timestamp)  # 记录token间延迟

                                most_recent_timestamp = timestamp  # 更新最近时间戳
                                generated_text += text or ""  # 累加生成文本
                            elif usage := data.get("usage"):  # 如果有usage字段
                                output.output_tokens = usage.get("completion_tokens")  # 记录输出token数
                if first_chunk_received:  # 如果收到了有效数据
                    output.success = True  # 标记成功
                else:  # 如果没有收到任何有效数据
                    output.success = False  # 标记失败
                    output.error = (  # 设置错误信息
                        "Never received a valid chunk to calculate TTFT."  # 从未收到有效数据块
                        "This response will be marked as failed!"  # 此响应将标记为失败
                    )
                output.generated_text = generated_text  # 设置生成的文本
                output.latency = most_recent_timestamp - st  # 计算总延迟
            else:  # 如果响应状态码不是200
                output.error = response.reason or ""  # 记录错误原因
                output.success = False  # 标记失败
    except Exception:  # 捕获所有异常
        output.success = False  # 标记失败
        exc_info = sys.exc_info()  # 获取异常信息
        output.error = "".join(traceback.format_exception(*exc_info))  # 格式化异常追踪信息

    if pbar:  # 如果有进度条
        pbar.update(1)  # 更新进度
    return output  # 返回输出


def _get_chat_content(
    request_func_input: RequestFuncInput,  # 请求函数输入
    mm_position: Literal["first", "last"] = "last",  # 多模态内容位置，默认在最后
) -> list[dict[str, Any]]:  # 返回内容列表
    """
    构建聊天API的内容列表，支持文本和多模态内容。
    """
    text_contents = [{"type": "text", "text": request_func_input.prompt}]  # 创建文本内容

    mm_contents = []  # 初始化多模态内容列表
    if request_func_input.multi_modal_content:  # 如果有多模态内容
        mm_content = request_func_input.multi_modal_content  # 获取多模态内容
        if isinstance(mm_content, list):  # 如果是列表
            mm_contents.extend(request_func_input.multi_modal_content)  # 扩展列表
        elif isinstance(mm_content, dict):  # 如果是字典
            mm_contents.append(request_func_input.multi_modal_content)  # 添加到列表
        else:  # 其他类型
            raise TypeError(  # 抛出类型错误
                "multi_modal_content must be a dict or list[dict] for openai-chat"  # 必须是字典或字典列表
            )

    if mm_position == "first":  # 如果多模态内容在前
        return mm_contents + text_contents  # 多模态内容在前，文本在后

    return text_contents + mm_contents  # 默认文本在前，多模态内容在后


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
    mm_position: Literal["first", "last"] = "last",  # 多模态内容位置
) -> RequestFuncOutput:  # 返回请求输出
    """
    OpenAI Chat Completions API的异步请求函数。
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")  # 验证URL

    content = _get_chat_content(request_func_input, mm_position=mm_position)  # 构建聊天内容

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "messages": [  # 消息列表
            {"role": "user", "content": content},  # 用户消息
        ],
        "max_completion_tokens": request_func_input.output_len,  # 最大补全token数
        "stream": True,  # 启用流式输出
        "stream_options": {  # 流式选项
            "include_usage": True,  # 包含使用量信息
        },
    }
    _update_payload_common(payload, request_func_input)  # 更新通用负载字段

    headers = _get_headers("application/json")  # 获取JSON类型请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    output = RequestFuncOutput()  # 创建输出对象
    output.prompt_len = request_func_input.prompt_len  # 设置提示长度

    generated_text = ""  # 初始化生成文本
    ttft = 0.0  # 初始化首token延迟
    st = time.perf_counter()  # 记录开始时间
    output.start_time = st  # 设置开始时间
    most_recent_timestamp = st  # 记录最近时间戳
    try:  # 尝试发送请求
        async with session.post(url=api_url, json=payload, headers=headers) as response:  # 发送POST请求
            if response.status == 200:  # 如果响应成功
                handler = StreamedResponseHandler()  # 创建流式响应处理器
                async for chunk_bytes in response.content.iter_any():  # 异步迭代响应
                    chunk_bytes = chunk_bytes.strip()  # 去除首尾空白
                    if not chunk_bytes:  # 如果数据块为空
                        continue  # 跳过

                    messages = handler.add_chunk(chunk_bytes)  # 处理数据块
                    for message in messages:  # 遍历消息
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):  # 跳过SSE注释
                            continue  # 继续下一个消息

                        chunk = message.removeprefix("data: ")  # 移除前缀

                        if chunk != "[DONE]":  # 如果不是结束标记
                            timestamp = time.perf_counter()  # 记录当前时间
                            data = json.loads(chunk)  # 解析JSON

                            if choices := data.get("choices"):  # 如果有choices字段
                                content = choices[0]["delta"].get("content")  # 获取增量内容
                                # First token
                                if ttft == 0.0:  # 如果是首个token
                                    ttft = timestamp - st  # 计算首token延迟
                                    output.ttft = ttft  # 设置首token延迟

                                # Decoding phase
                                else:  # 解码阶段
                                    output.itl.append(timestamp - most_recent_timestamp)  # 记录token间延迟

                                generated_text += content or ""  # 累加生成文本
                            elif usage := data.get("usage"):  # 如果有usage字段
                                output.output_tokens = usage.get("completion_tokens")  # 记录输出token数

                            most_recent_timestamp = timestamp  # 更新最近时间戳

                output.generated_text = generated_text  # 设置生成文本
                output.success = True  # 标记成功
                output.latency = most_recent_timestamp - st  # 计算总延迟
            else:  # 如果响应失败
                output.error = response.reason or ""  # 记录错误原因
                output.success = False  # 标记失败
    except Exception:  # 捕获异常
        output.success = False  # 标记失败
        exc_info = sys.exc_info()  # 获取异常信息
        output.error = "".join(traceback.format_exception(*exc_info))  # 格式化异常信息

    if pbar:  # 如果有进度条
        pbar.update(1)  # 更新进度
    return output  # 返回输出


async def async_request_openai_audio(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    OpenAI Audio API的异步请求函数，用于音频转录和翻译。
    """
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile  # 延迟导入音频处理库

    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "OpenAI Audio API", {"transcriptions", "translations"})  # 验证URL

    content = [{"type": "text", "text": request_func_input.prompt}]  # 创建文本内容
    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "max_completion_tokens": request_func_input.output_len,  # 最大补全token数
        "stream": True,  # 启用流式输出
        "language": "en",  # 语言设置为英语
        # Flattened due to multipart/form-data
        "stream_include_usage": True,  # 流式输出包含使用量（因multipart/form-data而扁平化）
        "stream_continuous_usage_stats": True,  # 持续使用量统计
    }
    _update_payload_common(payload, request_func_input)  # 更新通用负载字段

    headers = _get_headers()  # 获取请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    # Send audio file
    def to_bytes(y, sr):  # 将音频数据转换为字节流
        """
        将音频数组转换为WAV格式的字节流。
        """
        buffer = io.BytesIO()  # 创建字节缓冲区
        soundfile.write(buffer, y, sr, format="WAV")  # 写入WAV格式
        buffer.seek(0)  # 重置缓冲区位置
        return buffer  # 返回缓冲区

    mm_audio = request_func_input.multi_modal_content  # 获取多模态音频内容
    if not isinstance(mm_audio, dict) or "audio" not in mm_audio:  # 验证音频内容格式
        raise TypeError("multi_modal_content must be a dict containing 'audio'")  # 抛出类型错误
    with to_bytes(*mm_audio["audio"]) as f:  # 转换音频为字节流
        form = aiohttp.FormData()  # 创建表单数据
        form.add_field("file", f, content_type="audio/wav")  # 添加音频文件字段
        for key, value in payload.items():  # 遍历负载项
            form.add_field(key, str(value))  # 添加到表单

        output = RequestFuncOutput()  # 创建输出对象
        output.prompt_len = request_func_input.prompt_len  # 设置提示长度
        output.input_audio_duration = soundfile.info(f).duration  # 获取音频时长
        f.seek(0)  # 重置文件位置

        generated_text = ""  # 初始化生成文本
        ttft = 0.0  # 初始化首token延迟
        st = time.perf_counter()  # 记录开始时间
        output.start_time = st  # 设置开始时间
        most_recent_timestamp = st  # 记录最近时间戳
        try:  # 尝试发送请求
            async with session.post(  # 发送POST请求
                url=api_url, data=form, headers=headers  # 使用表单数据
            ) as response:  # 获取响应
                if response.status == 200:  # 如果响应成功
                    handler = StreamedResponseHandler()  # 创建流式响应处理器

                    async for chunk_bytes in response.content.iter_any():  # 异步迭代响应
                        chunk_bytes = chunk_bytes.strip()  # 去除空白
                        if not chunk_bytes:  # 如果为空
                            continue  # 跳过

                        messages = handler.add_chunk(chunk_bytes)  # 处理数据块
                        for message in messages:  # 遍历消息
                            if type(message) is bytes:  # 如果消息是字节类型
                                message = message.decode("utf-8")  # 解码为字符串
                            chunk = message.removeprefix("data: ")  # 移除前缀
                            if chunk != "[DONE]":  # 如果不是结束标记
                                timestamp = time.perf_counter()  # 记录当前时间
                                data = json.loads(chunk)  # 解析JSON

                                if choices := data.get("choices"):  # 如果有choices
                                    content = choices[0]["delta"].get("content")  # 获取增量内容
                                    # First token
                                    if ttft == 0.0:  # 首个token
                                        ttft = timestamp - st  # 计算首token延迟
                                        output.ttft = ttft  # 设置延迟

                                    # Decoding phase
                                    else:  # 解码阶段
                                        output.itl.append(  # 记录token间延迟
                                            timestamp - most_recent_timestamp  # 时间差
                                        )

                                    generated_text += content or ""  # 累加文本
                                elif usage := data.get("usage"):  # 如果有usage
                                    output.output_tokens = usage.get(  # 记录输出token数
                                        "completion_tokens"  # 补全token数
                                    )

                                most_recent_timestamp = timestamp  # 更新时间戳

                    output.generated_text = generated_text  # 设置生成文本
                    output.success = True  # 标记成功
                    output.latency = most_recent_timestamp - st  # 计算延迟
                else:  # 响应失败
                    output.error = response.reason or ""  # 记录错误
                    output.success = False  # 标记失败
        except Exception:  # 捕获异常
            output.success = False  # 标记失败
            exc_info = sys.exc_info()  # 获取异常信息
            output.error = "".join(traceback.format_exception(*exc_info))  # 格式化异常

    if pbar:  # 如果有进度条
        pbar.update(1)  # 更新进度
    return output  # 返回输出


async def _run_pooling_request(
    session: aiohttp.ClientSession,  # HTTP会话
    api_url: str,  # API URL
    payload: dict[str, Any],  # 请求负载
    headers: dict[str, Any],  # 请求头
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    运行池化（embedding/rerank等）请求的通用函数。
    """
    output = RequestFuncOutput()  # 创建输出对象
    st = time.perf_counter()  # 记录开始时间
    output.start_time = st  # 设置开始时间
    try:  # 尝试发送请求
        async with session.post(url=api_url, headers=headers, json=payload) as response:  # 发送POST请求
            if response.status == 200:  # 如果响应成功
                output.ttft = output.latency = time.perf_counter() - st  # 计算延迟

                if payload.get("encoding_format", "float") == "bytes":  # 如果编码格式为字节
                    metadata = json.loads(response.headers["metadata"])  # 从响应头获取元数据
                    usage = metadata.get("usage", {})  # 获取使用量信息
                else:  # 默认float格式
                    data = await response.json()  # 解析JSON响应
                    usage = data.get("usage", {})  # 获取使用量信息

                output.success = True  # 标记成功
                output.generated_text = ""  # 池化请求无生成文本
                output.prompt_len = usage.get("prompt_tokens", 0)  # 获取提示token数
            else:  # 响应失败
                output.success = False  # 标记失败
                output.error = response.reason or ""  # 记录错误
    except Exception as e:  # 捕获异常
        output.success = False  # 标记失败
        output.error = str(e)  # 记录错误信息

    if pbar:  # 如果有进度条
        pbar.update(1)  # 更新进度
    return output  # 返回输出


async def async_request_openai_embeddings(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    OpenAI Embeddings API的异步请求函数。
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "OpenAI Embeddings API", "embeddings")  # 验证URL

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "input": request_func_input.prompt,  # 输入文本
        # Many embedding models have short context length,
        # this is to avoid dropping some of the requests.
        "truncate_prompt_tokens": -1,  # 截断提示token数，-1表示不截断
    }
    _update_payload_common(payload, request_func_input)  # 更新通用负载

    headers = _get_headers("application/json")  # 获取JSON请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    return await _run_pooling_request(  # 执行池化请求
        session,  # HTTP会话
        api_url,  # API URL
        payload=payload,  # 请求负载
        headers=headers,  # 请求头
        pbar=pbar,  # 进度条
    )


async def async_request_vllm_rerank(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    vLLM重排序API的异步请求函数。
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "vLLM score API", "rerank")  # 验证URL

    assert (  # 断言提示是列表且长度大于1
        isinstance(request_func_input.prompt, list)  # 提示必须是列表
        and len(request_func_input.prompt) > 1  # 且长度大于1
    )

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "query": request_func_input.prompt[0],  # 查询文本（第一个元素）
        "documents": request_func_input.prompt[1:],  # 文档列表（剩余元素）
        # Many reranker models have short context length,
        # this is to avoid dropping some of the requests.
        "truncate_prompt_tokens": -1,  # 不截断提示token
    }

    headers = _get_headers("application/json")  # 获取JSON请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    return await _run_pooling_request(  # 执行池化请求
        session,  # HTTP会话
        api_url,  # API URL
        payload=payload,  # 请求负载
        headers=headers,  # 请求头
        pbar=pbar,  # 进度条
    )


async def async_request_openai_embeddings_chat(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
    mm_position: Literal["first", "last"] = "last",  # 多模态内容位置
) -> RequestFuncOutput:  # 返回请求输出
    """
    OpenAI Embeddings API的聊天格式异步请求函数。
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "OpenAI Embeddings API", "embeddings")  # 验证URL

    content = _get_chat_content(request_func_input, mm_position=mm_position)  # 构建聊天内容

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "messages": [  # 消息列表
            {"role": "user", "content": content},  # 用户消息
        ],
        # Many embedding models have short context length,
        # this is to avoid dropping some of the requests.
        "truncate_prompt_tokens": -1,  # 不截断提示token
    }
    _update_payload_common(payload, request_func_input)  # 更新通用负载

    headers = _get_headers("application/json")  # 获取JSON请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    return await _run_pooling_request(  # 执行池化请求
        session,  # HTTP会话
        api_url,  # API URL
        payload=payload,  # 请求负载
        headers=headers,  # 请求头
        pbar=pbar,  # 进度条
    )


def _try_extract_request_idx(request_func_input: RequestFuncInput):
    """
    尝试从请求ID中提取数字索引。
    """
    if request_func_input.request_id:  # 如果有请求ID
        match = re.search(r"(\d+)$", request_func_input.request_id)  # 匹配末尾数字
        if match:  # 如果匹配成功
            try:  # 尝试转换
                return int(match.group(1))  # 返回整数索引
            except ValueError:  # 如果转换失败
                pass  # 忽略

    return None  # 返回None


def _preprocess_clip(request_func_input: RequestFuncInput):
    """
    预处理CLIP模型的请求，当有多模态内容时清空文本提示。
    """
    if request_func_input.multi_modal_content:  # 如果有多模态内容
        # Image input
        request_func_input.prompt = ""  # 清空文本提示


def _preprocess_vlm2vec(request_func_input: RequestFuncInput):
    """
    预处理VLM2Vec模型的请求，根据请求索引设置不同的提示模板。
    """
    if request_func_input.multi_modal_content:  # 如果有多模态内容
        request_idx = _try_extract_request_idx(request_func_input)  # 提取请求索引

        # Adjust the ratio manually if needed.
        use_image_only_prompt = request_idx is None or request_idx % 2 == 0  # 偶数索引使用纯图像提示

        if use_image_only_prompt:  # 纯图像输入
            # Image input
            request_func_input.prompt = "Represent the given image."  # 设置图像提示
        else:  # 文本+图像输入
            # Text+Image input
            request_func_input.prompt = (  # 设置文本+图像提示
                f"Represent the given image with the following question: "  # 提示模板
                f"{request_func_input.prompt}"  # 原始提示
            )


async def async_request_openai_embeddings_clip(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    CLIP模型的OpenAI Embeddings API异步请求函数。
    """
    _preprocess_clip(request_func_input)  # CLIP预处理

    return await async_request_openai_embeddings_chat(  # 调用聊天格式embedding请求
        request_func_input,  # 请求输入
        session,  # HTTP会话
        pbar=pbar,  # 进度条
    )


async def async_request_openai_embeddings_vlm2vec(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    VLM2Vec模型的OpenAI Embeddings API异步请求函数。
    """
    _preprocess_vlm2vec(request_func_input)  # VLM2Vec预处理

    return await async_request_openai_embeddings_chat(  # 调用聊天格式embedding请求
        request_func_input,  # 请求输入
        session,  # HTTP会话
        pbar=pbar,  # 进度条
        mm_position="first",  # 多模态内容在前
    )


async def async_request_infinity_embeddings(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    Infinity Embeddings API的异步请求函数。
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "Infinity Embeddings API", "embeddings")  # 验证URL

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
    }

    if request_func_input.prompt:  # 如果有文本提示
        payload["input"] = request_func_input.prompt  # 设置输入
    else:  # 否则使用多模态内容
        mm_content = request_func_input.multi_modal_content  # 获取多模态内容
        assert isinstance(mm_content, dict)  # 断言是字典类型

        mm_type = mm_content["type"]  # 获取多模态类型
        payload["input"] = mm_content[mm_type]["url"]  # 设置输入URL
        payload["modality"] = mm_type.split("_", 1)[0]  # 设置模态类型

    _update_payload_common(payload, request_func_input)  # 更新通用负载

    headers = _get_headers("application/json")  # 获取JSON请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    return await _run_pooling_request(  # 执行池化请求
        session,  # HTTP会话
        api_url,  # API URL
        payload=payload,  # 请求负载
        headers=headers,  # 请求头
        pbar=pbar,  # 进度条
    )


async def async_request_infinity_embeddings_clip(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    CLIP模型的Infinity Embeddings API异步请求函数。
    """
    _preprocess_clip(request_func_input)  # CLIP预处理

    return await async_request_infinity_embeddings(  # 调用Infinity embedding请求
        request_func_input,  # 请求输入
        session,  # HTTP会话
        pbar=pbar,  # 进度条
    )


async def async_request_vllm_pooling(
    request_func_input: RequestFuncInput,  # 请求输入
    session: aiohttp.ClientSession,  # HTTP会话
    pbar: tqdm | None = None,  # 可选进度条
) -> RequestFuncOutput:  # 返回请求输出
    """
    vLLM Pooling API的异步请求函数。
    """
    api_url = request_func_input.api_url  # 获取API URL
    _validate_api_url(api_url, "vLLM Pooling API", "pooling")  # 验证URL

    payload = {  # 构建请求负载
        "model": request_func_input.model_name  # 模型名称
        if request_func_input.model_name  # 优先使用模型名称
        else request_func_input.model,  # 否则使用模型ID
        "truncate_prompt_tokens": -1,  # 不截断提示token
    }

    payload = payload | request_func_input.prompt  # 合并提示内容到负载

    _update_payload_common(payload, request_func_input)  # 更新通用负载

    headers = _get_headers("application/json")  # 获取JSON请求头
    _update_headers_common(headers, request_func_input)  # 更新通用请求头

    return await _run_pooling_request(  # 执行池化请求
        session,  # HTTP会话
        api_url,  # API URL
        payload=payload,  # 请求负载
        headers=headers,  # 请求头
        pbar=pbar,  # 进度条
    )


# TODO: Add more request functions for different API protocols.
ASYNC_REQUEST_FUNCS: dict[str, RequestFunc] = {  # 异步请求函数注册表
    "vllm": async_request_openai_completions,  # vLLM后端使用OpenAI补全API
    "openai": async_request_openai_completions,  # OpenAI补全API
    "openai-chat": async_request_openai_chat_completions,  # OpenAI聊天补全API
    "openai-audio": async_request_openai_audio,  # OpenAI音频API
    "openai-embeddings": async_request_openai_embeddings,  # OpenAI嵌入API
    "openai-embeddings-chat": async_request_openai_embeddings_chat,  # OpenAI聊天格式嵌入API
    "openai-embeddings-clip": async_request_openai_embeddings_clip,  # CLIP嵌入API
    "openai-embeddings-vlm2vec": async_request_openai_embeddings_vlm2vec,  # VLM2Vec嵌入API
    # Infinity embedding server: https://github.com/michaelfeil/infinity
    "infinity-embeddings": async_request_infinity_embeddings,  # Infinity嵌入服务器
    "infinity-embeddings-clip": async_request_infinity_embeddings_clip,  # Infinity CLIP嵌入
    # (Infinity embedding server does not support vlm2vec)
    "vllm-pooling": async_request_vllm_pooling,  # vLLM池化API
    "vllm-rerank": async_request_vllm_rerank,  # vLLM重排序API
}

POOLING_BACKENDS = {  # 池化后端集合
    "openai-embeddings",  # OpenAI嵌入
    "openai-embeddings-chat",  # OpenAI聊天嵌入
    "openai-embeddings-clip",  # CLIP嵌入
    "openai-embeddings-vlm2vec",  # VLM2Vec嵌入
    "infinity-embeddings",  # Infinity嵌入
    "infinity-embeddings-clip",  # Infinity CLIP嵌入
    "vllm-pooling",  # vLLM池化
    "vllm-rerank",  # vLLM重排序
}

OPENAI_COMPATIBLE_BACKENDS = [  # OpenAI兼容后端列表
    k  # 后端名称
    for k, v in ASYNC_REQUEST_FUNCS.items()  # 遍历注册表
    if v in (async_request_openai_completions, async_request_openai_chat_completions)  # 筛选补全和聊天补全函数
]
