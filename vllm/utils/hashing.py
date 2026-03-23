# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import pickle
from _hashlib import HASH, UnsupportedDigestmodError
from collections.abc import Callable
from typing import Any

import cbor2

try:
    # It is important that this remains an optional dependency.
    # It would not be allowed in environments with strict security controls,
    # so it's best not to have it installed when not in use.
    import xxhash as _xxhash

    if not hasattr(_xxhash, "xxh3_128_digest"):
        _xxhash = None
except ImportError:  # pragma: no cover
    _xxhash = None


# 使用 pickle 序列化后进行 SHA-256 哈希，支持任意可序列化的 Python 对象
def sha256(input: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


# 使用 CBOR（跨语言二进制序列化）后进行 SHA-256 哈希
# 适用于需要与非 Python 环境保持序列化一致性的场景
def sha256_cbor(input: Any) -> bytes:
    """Hash objects using CBOR serialization and SHA-256.

    This option is useful for non-Python-dependent serialization and hashing.

    Args:
        input: Object to be serialized and hashed. Supported types include
            basic Python types and complex structures like lists, tuples, and
            dictionaries.
            Custom classes must implement CBOR serialization methods.

    Returns:
        Bytes representing the SHA-256 hash of the CBOR serialized input.
    """
    input_bytes = cbor2.dumps(input, canonical=True)
    return hashlib.sha256(input_bytes).digest()


# xxHash 128 位摘要的内部实现，xxhash 为可选依赖
def _xxhash_digest(input_bytes: bytes) -> bytes:
    if _xxhash is None:
        raise ModuleNotFoundError(
            "xxhash is required for the 'xxhash' prefix caching hash algorithms. "
            "Install it via `pip install xxhash`."
        )
    return _xxhash.xxh3_128_digest(input_bytes)


# 使用 pickle 序列化后进行 xxHash 哈希，比 SHA-256 更快，适用于前缀缓存
def xxhash(input: Any) -> bytes:
    """Hash picklable objects using xxHash."""
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return _xxhash_digest(input_bytes)


# 使用 CBOR 序列化后进行 xxHash 哈希
def xxhash_cbor(input: Any) -> bytes:
    """Hash objects serialized with CBOR using xxHash."""
    input_bytes = cbor2.dumps(input, canonical=True)
    return _xxhash_digest(input_bytes)


# 根据名称字符串获取对应的哈希函数
# 支持 sha256、sha256_cbor、xxhash、xxhash_cbor 四种算法
def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """Get a hash function by name, or raise an error if the function is not found.

    Args:
        hash_fn_name: Name of the hash function.

    Returns:
        A hash function.
    """
    if hash_fn_name == "sha256":
        return sha256
    if hash_fn_name == "sha256_cbor":
        return sha256_cbor
    if hash_fn_name == "xxhash":
        return xxhash
    if hash_fn_name == "xxhash_cbor":
        return xxhash_cbor

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")


# 安全哈希函数：优先使用 MD5，在 FIPS 受限环境中自动回退到 SHA-256
def safe_hash(data: bytes, usedforsecurity: bool = True) -> HASH:
    """Hash for configs, defaulting to md5 but falling back to sha256
    in FIPS constrained environments.

    Args:
        data: bytes
        usedforsecurity: Whether the hash is used for security purposes

    Returns:
        Hash object
    """
    try:
        return hashlib.md5(data, usedforsecurity=usedforsecurity)
    except (UnsupportedDigestmodError, ValueError):
        return hashlib.sha256(data)
