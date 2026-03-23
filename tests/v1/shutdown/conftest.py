# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from vllm.platforms import current_platform


# [中文注释] 测试夹具：为ROCm平台创建sitecustomize安装工厂，用于注入测试用的猴子补丁
@pytest.fixture
def rocm_sitecustomize_factory(monkeypatch, tmp_path: Path):
    """Return a function that installs a given sitecustomize payload."""
    if not current_platform.is_rocm():
        return lambda _: None

    def install(lines: Iterable[str]) -> None:
        sc = tmp_path / "sitecustomize.py"
        sc.write_text("\n".join(lines) + "\n")
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join(filter(None, [str(tmp_path), os.getenv("PYTHONPATH")])),
        )

    return install
