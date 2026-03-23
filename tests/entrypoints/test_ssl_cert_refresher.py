# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试 SSL 证书热刷新功能，验证证书/密钥/CA 文件变更时自动重新加载，
# 以及停止刷新器后不再响应文件变更

import asyncio
import tempfile
from pathlib import Path
from ssl import SSLContext

import pytest

from vllm.entrypoints.ssl import SSLCertRefresher


# 模拟 SSL 上下文，追踪证书链和 CA 的加载次数
class MockSSLContext(SSLContext):
    def __init__(self):
        self.load_cert_chain_count = 0
        self.load_ca_count = 0

    def load_cert_chain(
        self,
        certfile,
        keyfile=None,
        password=None,
    ):
        self.load_cert_chain_count += 1

    def load_verify_locations(
        self,
        cafile=None,
        capath=None,
        cadata=None,
    ):
        self.load_ca_count += 1


# 辅助函数：创建临时文件并返回路径
def create_file() -> str:
    with tempfile.NamedTemporaryFile(dir="/tmp", delete=False) as f:
        return f.name


# 辅助函数：触碰文件以更新修改时间，触发刷新器检测
def touch_file(path: str) -> None:
    Path(path).touch()


# 测试 SSL 证书刷新器的完整生命周期：文件变更触发重载、停止后不再响应
@pytest.mark.asyncio
async def test_ssl_refresher():
    ssl_context = MockSSLContext()
    key_path = create_file()
    cert_path = create_file()
    ca_path = create_file()
    ssl_refresher = SSLCertRefresher(ssl_context, key_path, cert_path, ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 0
    assert ssl_context.load_ca_count == 0

    touch_file(key_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 1
    assert ssl_context.load_ca_count == 0

    touch_file(cert_path)
    touch_file(ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 2
    assert ssl_context.load_ca_count == 1

    ssl_refresher.stop()

    touch_file(cert_path)
    touch_file(ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 2
    assert ssl_context.load_ca_count == 1
