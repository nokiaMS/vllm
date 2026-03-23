# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

import pytest

from vllm.config.utils import get_hash_factors, hash_factors, normalize_value

# Helpers


# [中文注释] 辅助函数：检查normalize_value的输出是否以给定的完全限定名后缀结尾
def endswith_fqname(obj, suffix: str) -> bool:
    # normalize_value(type) returns fully-qualified name
    # Compare suffix to avoid brittle import paths.
    out = normalize_value(obj)
    return isinstance(out, str) and out.endswith(suffix)


# [中文注释] 辅助函数：将路径字符串转换为规范化的绝对路径
def expected_path(p_str: str = ".") -> str:
    import pathlib

    p = pathlib.Path(p_str)
    return p.expanduser().resolve().as_posix()


# Minimal dataclass to test get_hash_factors.
# Avoid importing heavy vLLM configs.
# [中文注释] 用于测试get_hash_factors的简单数据类，避免导入重量级vLLM配置
@dataclass
class SimpleConfig:
    a: object
    b: object | None = None


# [中文注释] 用于测试枚举类型规范化的虚拟枚举类
class DummyLogprobsMode(Enum):
    RAW_LOGITS = "raw_logits"


# [中文注释] 测试hash_factors生成一致的SHA-256哈希，且字典键顺序不影响结果
def test_hash_factors_deterministic():
    """Test that hash_factors produces consistent SHA-256 hashes"""
    factors = {"a": 1, "b": "test"}
    hash1 = hash_factors(factors)
    hash2 = hash_factors(factors)

    assert hash1 == hash2
    # Dict key insertion order should not affect the hash.
    factors_reordered = {"b": "test", "a": 1}
    assert hash_factors(factors_reordered) == hash1
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


@pytest.mark.parametrize(
    "inp, expected",
    [
        (None, None),
        (True, True),
        (1, 1),
        (1.0, 1.0),
        ("x", "x"),
        (b"ab", "6162"),
        (bytearray(b"ab"), "6162"),
        ([1, 2], (1, 2)),
        ({"b": 2, "a": 1}, (("a", 1), ("b", 2))),
    ],
)
# [中文注释] 参数化测试normalize_value对各种输入类型（None、布尔、数值、字符串、字节、列表、字典）的规范化
def test_normalize_value_matrix(inp, expected):
    """Parametric input→expected normalization table."""
    assert normalize_value(inp) == expected


# [中文注释] 测试枚举值规范化为(模块.类名, 值)的元组格式
def test_normalize_value_enum():
    # Enums normalize to (module.QualName, value).
    # DummyLogprobsMode uses a string payload.
    out = normalize_value(DummyLogprobsMode.RAW_LOGITS)
    assert isinstance(out, tuple)
    assert out[0].endswith("DummyLogprobsMode")
    # Expect string payload 'raw_logits'.
    assert out[1] == "raw_logits"


# [中文注释] 测试集合规范化时排序以保证确定性，不受元素插入顺序影响
def test_normalize_value_set_order_insensitive():
    # Sets are unordered; normalize_value sorts elements for determinism.
    assert normalize_value({3, 1, 2}) == normalize_value({1, 2, 3})


# [中文注释] 测试路径规范化为绝对路径字符串，稳定跨工作目录的哈希
def test_normalize_value_path_normalization():
    from pathlib import Path  # local import to avoid global dependency

    # Paths expand/resolve to absolute strings.
    # Stabilizes hashing across working dirs.
    assert normalize_value(Path(".")) == expected_path(".")


# [中文注释] 测试具有uuid()或to_json_string()方法的对象的规范化
def test_normalize_value_uuid_and_to_json():
    # Objects may normalize via uuid() or to_json_string().
    class HasUUID:
        def uuid(self):
            return "test-uuid"

    class ToJson:
        def to_json_string(self):
            return '{"x":1}'

    assert normalize_value(HasUUID()) == "test-uuid"
    assert normalize_value(ToJson()) == '{"x":1}'


@pytest.mark.parametrize(
    "bad",
    [
        (lambda x: x),
        (type("CallableInstance", (), {"__call__": lambda self: 0}))(),
        (lambda: (lambda: 0))(),  # nested function instance
    ],
)
# [中文注释] 测试函数/lambda/可调用实例等不可哈希类型应抛出TypeError
def test_error_cases(bad):
    """Inputs expected to raise TypeError."""
    # Reject functions/lambdas/callable instances
    # to avoid under-hashing.
    with pytest.raises(TypeError):
        normalize_value(bad)


# [中文注释] 测试枚举值和整数值的哈希因子区分，防止原始类型与枚举类型的哈希碰撞
def test_enum_vs_int_disambiguation():
    # int stays primitive
    nf_int = normalize_value(1)
    assert nf_int == 1

    # enum becomes ("module.QualName", value)
    nf_enum = normalize_value(DummyLogprobsMode.RAW_LOGITS)
    assert isinstance(nf_enum, tuple) and len(nf_enum) == 2
    enum_type, enum_val = nf_enum
    assert enum_type.endswith(".DummyLogprobsMode")
    assert enum_val == "raw_logits"

    # Build factor dicts from configs with int vs enum
    f_int = get_hash_factors(SimpleConfig(1), set())
    f_enum = get_hash_factors(SimpleConfig(DummyLogprobsMode.RAW_LOGITS), set())
    # The int case remains a primitive value
    assert f_int["a"] == 1
    # The enum case becomes a tagged tuple ("module.QualName", "raw_logits")
    assert isinstance(f_enum["a"], tuple) and f_enum["a"][1] == "raw_logits"
    # Factor dicts must differ so we don't collide primitives with Enums.
    assert f_int != f_enum
    # Hash digests must differ correspondingly
    assert hash_factors(f_int) != hash_factors(f_enum)

    # Hash functions produce stable hex strings
    h_int = hash_factors(f_int)
    h_enum = hash_factors(f_enum)
    assert isinstance(h_int, str) and len(h_int) == 64
    assert isinstance(h_enum, str) and len(h_enum) == 64


# [中文注释] 测试类型规范化为完全限定名字符串
def test_classes_are_types():
    """Types normalize to FQNs; include real vLLM types."""
    # Only classes allowed; functions/lambdas are rejected.
    # Canonical form is the fully-qualified name.
    assert isinstance(normalize_value(str), str)

    class LocalDummy:
        pass

    assert endswith_fqname(LocalDummy, ".LocalDummy")


# [中文注释] 测试envs.compile_factors()的哈希在多次初始化间保持稳定，使用子进程验证动态默认值被正确忽略
def test_envs_compile_factors_stable():
    """Test that envs.compile_factors() hash is stable across fresh initializations.

    Uses subprocesses to ensure env vars with dynamic defaults (like UUIDs)
    are freshly generated each time, verifying they're properly ignored.
    """
    import subprocess
    import sys

    code = """
import sys
import logging
logging.disable(logging.CRITICAL)
from vllm import envs
from vllm.config.utils import hash_factors
print(hash_factors(envs.compile_factors()))
"""

    def get_hash_in_subprocess():
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**dict(__import__("os").environ), "VLLM_LOGGING_LEVEL": "ERROR"},
        )
        return result.stdout.strip()

    hash1 = get_hash_in_subprocess()
    hash2 = get_hash_in_subprocess()

    assert hash1 == hash2, (
        "compile_factors hash differs between fresh initializations - "
        "dynamic env vars may not be properly ignored"
    )
