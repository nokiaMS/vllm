# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
def sanitize_filename(filename: str) -> str:  # 清理文件名，移除不安全字符
    """清理文件名，将不安全字符替换为安全字符。"""
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')  # 替换斜杠为下划线，双点为双下划线，去除引号
