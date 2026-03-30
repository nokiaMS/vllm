# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM资产管理基础模块。
提供从S3公共存储桶下载和缓存资产文件的功能。
"""

from functools import lru_cache  # 导入LRU缓存装饰器
from pathlib import Path  # 导入路径操作类

import vllm.envs as envs  # 导入vLLM环境变量配置
from vllm.connections import global_http_connection  # 导入全局HTTP连接管理器

VLLM_S3_BUCKET_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com"  # vLLM公共资产S3存储桶URL


def get_cache_dir() -> Path:
    """Get the path to the cache for storing downloaded assets.
    获取用于存储下载资产的缓存目录路径。

    Returns:
        缓存目录的Path对象。
    """
    path = Path(envs.VLLM_ASSETS_CACHE)  # 从环境变量获取缓存路径
    path.mkdir(parents=True, exist_ok=True)  # 递归创建目录（如果不存在）

    return path  # 返回缓存目录路径


@lru_cache  # 使用LRU缓存装饰器，避免重复下载同一文件
def get_vllm_public_assets(filename: str, s3_prefix: str | None = None) -> Path:
    """
    Download an asset file from `s3://vllm-public-assets`
    and return the path to the downloaded file.
    从`s3://vllm-public-assets`下载资产文件并返回下载文件的路径。

    Args:
        filename: 要下载的文件名。
        s3_prefix: S3路径前缀（可选）。

    Returns:
        下载文件的本地路径。
    """
    asset_directory = get_cache_dir() / "vllm_public_assets"  # 构建公共资产缓存目录路径
    asset_directory.mkdir(parents=True, exist_ok=True)  # 递归创建目录（如果不存在）

    asset_path = asset_directory / filename  # 构建资产文件的完整路径
    if not asset_path.exists():  # 如果文件尚未下载
        if s3_prefix is not None:  # 如果指定了S3前缀
            filename = s3_prefix + "/" + filename  # 在文件名前添加前缀
        global_http_connection.download_file(  # 通过全局HTTP连接下载文件
            f"{VLLM_S3_BUCKET_URL}/{filename}",  # 构建完整的下载URL
            asset_path,  # 下载保存路径
            timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,  # 使用环境变量中配置的超时时间
        )

    return asset_path  # 返回资产文件的本地路径
