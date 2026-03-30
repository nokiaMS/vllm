# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, MutableMapping  # 导入映射和可变映射抽象基类
from pathlib import Path  # 导入路径处理模块

import aiohttp  # 导入异步HTTP客户端库
import requests  # 导入同步HTTP请求库
from urllib3.util import parse_url  # 导入URL解析工具函数

from vllm.version import __version__ as VLLM_VERSION  # 导入vLLM版本号


class HTTPConnection:
    """HTTP连接辅助类，用于发送HTTP请求。

    该类封装了同步和异步的HTTP客户端，提供了获取字节、文本、JSON以及下载文件等便捷方法。
    支持客户端复用以提高连接效率。
    """

    def __init__(self, *, reuse_client: bool = True) -> None:
        """初始化HTTP连接实例。

        Args:
            reuse_client: 是否复用HTTP客户端，默认为True。设为True时会缓存客户端实例以复用连接。
        """
        super().__init__()  # 调用父类初始化方法

        self.reuse_client = reuse_client  # 保存是否复用客户端的配置

        self._sync_client: requests.Session | None = None  # 同步HTTP客户端实例，初始为None
        self._async_client: aiohttp.ClientSession | None = None  # 异步HTTP客户端实例，初始为None

    def get_sync_client(self) -> requests.Session:
        """获取同步HTTP客户端。

        如果客户端尚未创建或不复用客户端，则创建新的Session实例。

        Returns:
            requests.Session: 同步HTTP会话客户端。
        """
        if self._sync_client is None or not self.reuse_client:  # 如果客户端未创建或不复用
            self._sync_client = requests.Session()  # 创建新的同步会话

        return self._sync_client  # 返回同步客户端实例

    # NOTE: We intentionally use an async function even though it is not
    # required, so that the client is only accessible inside async event loop
    async def get_async_client(self) -> aiohttp.ClientSession:
        """获取异步HTTP客户端。

        故意使用异步函数，以确保客户端仅在异步事件循环中可访问。
        如果客户端尚未创建或不复用客户端，则创建新的ClientSession实例。

        Returns:
            aiohttp.ClientSession: 异步HTTP会话客户端。
        """
        if self._async_client is None or not self.reuse_client:  # 如果客户端未创建或不复用
            self._async_client = aiohttp.ClientSession(trust_env=True)  # 创建新的异步会话，信任环境变量中的代理设置

        return self._async_client  # 返回异步客户端实例

    def _validate_http_url(self, url: str):
        """验证URL是否为合法的HTTP/HTTPS地址。

        Args:
            url: 待验证的URL字符串。

        Raises:
            ValueError: 当URL的协议不是http或https时抛出异常。
        """
        parsed_url = parse_url(url)  # 解析URL

        if parsed_url.scheme not in ("http", "https"):  # 如果协议不是http或https
            raise ValueError(  # 抛出值错误异常
                "Invalid HTTP URL: A valid HTTP URL must have scheme 'http' or 'https'."  # 错误信息：无效的HTTP URL
            )

    def _headers(self, **extras: str) -> MutableMapping[str, str]:
        """构建HTTP请求头。

        Args:
            **extras: 额外的请求头键值对。

        Returns:
            MutableMapping[str, str]: 包含User-Agent和额外请求头的可变映射。
        """
        return {"User-Agent": f"vLLM/{VLLM_VERSION}", **extras}  # 返回包含User-Agent标识和额外头信息的字典

    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        """发送同步GET请求并返回响应对象。

        Args:
            url: 请求的目标URL。
            stream: 是否以流模式接收响应，默认为False。
            timeout: 请求超时时间（秒），None表示不设超时。
            extra_headers: 额外的HTTP请求头。
            allow_redirects: 是否允许重定向，默认为True。

        Returns:
            requests.Response: HTTP响应对象。
        """
        self._validate_http_url(url)  # 验证URL合法性

        client = self.get_sync_client()  # 获取同步客户端
        extra_headers = extra_headers or {}  # 如果额外头为None则设为空字典

        return client.get(  # 发送GET请求并返回响应
            url,  # 请求URL
            headers=self._headers(**extra_headers),  # 设置请求头
            stream=stream,  # 设置是否流式传输
            timeout=timeout,  # 设置超时时间
            allow_redirects=allow_redirects,  # 设置是否允许重定向
        )

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        """发送异步GET请求并返回响应对象。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。
            extra_headers: 额外的HTTP请求头。
            allow_redirects: 是否允许重定向，默认为True。

        Returns:
            aiohttp.ClientResponse: 异步HTTP响应对象。
        """
        self._validate_http_url(url)  # 验证URL合法性

        client = await self.get_async_client()  # 异步获取异步客户端
        extra_headers = extra_headers or {}  # 如果额外头为None则设为空字典

        return client.get(  # 发送异步GET请求并返回响应
            url,  # 请求URL
            headers=self._headers(**extra_headers),  # 设置请求头
            timeout=timeout,  # 设置超时时间
            allow_redirects=allow_redirects,  # 设置是否允许重定向
        )

    def get_bytes(
        self, url: str, *, timeout: float | None = None, allow_redirects: bool = True
    ) -> bytes:
        """同步获取URL响应的字节内容。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。
            allow_redirects: 是否允许重定向，默认为True。

        Returns:
            bytes: 响应体的字节内容。
        """
        with self.get_response(  # 使用上下文管理器发送请求
            url, timeout=timeout, allow_redirects=allow_redirects  # 传入URL、超时和重定向参数
        ) as r:
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            return r.content  # 返回响应的字节内容

    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
    ) -> bytes:
        """异步获取URL响应的字节内容。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。
            allow_redirects: 是否允许重定向，默认为True。

        Returns:
            bytes: 响应体的字节内容。
        """
        async with await self.get_async_response(  # 使用异步上下文管理器发送请求
            url, timeout=timeout, allow_redirects=allow_redirects  # 传入URL、超时和重定向参数
        ) as r:
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            return await r.read()  # 异步读取并返回响应的字节内容

    def get_text(self, url: str, *, timeout: float | None = None) -> str:
        """同步获取URL响应的文本内容。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。

        Returns:
            str: 响应体的文本内容。
        """
        with self.get_response(url, timeout=timeout) as r:  # 使用上下文管理器发送请求
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            return r.text  # 返回响应的文本内容

    async def async_get_text(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        """异步获取URL响应的文本内容。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。

        Returns:
            str: 响应体的文本内容。
        """
        async with await self.get_async_response(url, timeout=timeout) as r:  # 使用异步上下文管理器发送请求
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            return await r.text()  # 异步读取并返回响应的文本内容

    def get_json(self, url: str, *, timeout: float | None = None) -> str:
        """同步获取URL响应的JSON内容。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。

        Returns:
            str: 响应体解析后的JSON对象。
        """
        with self.get_response(url, timeout=timeout) as r:  # 使用上下文管理器发送请求
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            return r.json()  # 解析并返回响应的JSON内容

    async def async_get_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        """异步获取URL响应的JSON内容。

        Args:
            url: 请求的目标URL。
            timeout: 请求超时时间（秒），None表示不设超时。

        Returns:
            str: 响应体解析后的JSON对象。
        """
        async with await self.get_async_response(url, timeout=timeout) as r:  # 使用异步上下文管理器发送请求
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            return await r.json()  # 异步解析并返回响应的JSON内容

    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        """同步下载文件并保存到指定路径。

        Args:
            url: 文件下载的目标URL。
            save_path: 文件保存的本地路径。
            timeout: 请求超时时间（秒），None表示不设超时。
            chunk_size: 每次读取的块大小（字节），默认为128。

        Returns:
            Path: 文件保存的路径。
        """
        with self.get_response(url, timeout=timeout) as r:  # 使用上下文管理器发送请求
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            with save_path.open("wb") as f:  # 以二进制写模式打开保存文件
                for chunk in r.iter_content(chunk_size):  # 按块迭代响应内容
                    f.write(chunk)  # 将每个数据块写入文件

        return save_path  # 返回文件保存路径

    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        """异步下载文件并保存到指定路径。

        Args:
            url: 文件下载的目标URL。
            save_path: 文件保存的本地路径。
            timeout: 请求超时时间（秒），None表示不设超时。
            chunk_size: 每次读取的块大小（字节），默认为128。

        Returns:
            Path: 文件保存的路径。
        """
        async with await self.get_async_response(url, timeout=timeout) as r:  # 使用异步上下文管理器发送请求
            r.raise_for_status()  # 如果响应状态码表示错误则抛出异常

            with save_path.open("wb") as f:  # 以二进制写模式打开保存文件
                async for chunk in r.content.iter_chunked(chunk_size):  # 异步按块迭代响应内容
                    f.write(chunk)  # 将每个数据块写入文件

        return save_path  # 返回文件保存路径


global_http_connection = HTTPConnection()  # 创建全局HTTP连接实例
"""
The global [`HTTPConnection`][vllm.connections.HTTPConnection] instance used
by vLLM.
"""
