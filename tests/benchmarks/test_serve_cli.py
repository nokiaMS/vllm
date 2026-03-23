# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
import urllib3

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


# [中文注释] 使用openssl生成自签名证书用于SSL测试
def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed certificate for testing."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_file, key_file


# [中文注释] 支持SSL自签名证书的RemoteOpenAIServer子类，用于HTTPS基准测试
class RemoteOpenAIServerSSL(RemoteOpenAIServer):
    """RemoteOpenAIServer subclass that supports SSL with self-signed certs."""

    @property
    def url_root(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _wait_for_server(self, *, url: str, timeout: float):
        """Override to use HTTPS with SSL verification disabled."""
        # Suppress InsecureRequestWarning for self-signed certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        start = time.time()
        while True:
            try:
                if requests.get(url, verify=False).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None


# [中文注释] 启动标准HTTP模式的vLLM远程服务器fixture
@pytest.fixture(scope="function")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager", "--load-format", "dummy"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


# [中文注释] 启动带自签名SSL证书的vLLM远程服务器fixture
@pytest.fixture(scope="function")
def ssl_server():
    """Start a vLLM server with SSL enabled using a self-signed certificate."""
    with tempfile.TemporaryDirectory() as cert_dir:
        cert_file, key_file = generate_self_signed_cert(Path(cert_dir))
        args = [
            "--max-model-len",
            "1024",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--ssl-certfile",
            str(cert_file),
            "--ssl-keyfile",
            str(key_file),
        ]

        with RemoteOpenAIServerSSL(MODEL_NAME, args) as remote_server:
            yield remote_server


# [中文注释] 测试vllm bench serve CLI命令的基本服务基准测试
@pytest.mark.benchmark
def test_bench_serve(server):
    # Test default model detection and input/output len
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


# [中文注释] 测试--insecure标志与自签名证书HTTPS服务器的配合使用
@pytest.mark.benchmark
def test_bench_serve_insecure(ssl_server):
    """Test --insecure flag with an HTTPS server using a self-signed certificate."""
    base_url = f"https://{ssl_server.host}:{ssl_server.port}"
    command = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
        "--insecure",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


# [中文注释] 测试vllm bench serve的聊天完成端点基准测试（/v1/chat/completions）
@pytest.mark.benchmark
def test_bench_serve_chat(server):
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
