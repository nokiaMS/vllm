# SPDX-License-Identifier: Apache-2.0  # 许可证标识符：Apache-2.0开源协议
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

try:  # 尝试从自动生成的版本文件中导入版本信息
    from ._version import __version__, __version_tuple__  # 导入版本字符串和版本元组
except Exception as e:  # 如果导入失败（例如未构建安装），捕获异常
    import warnings  # 导入警告模块

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)  # 发出运行时警告，提示读取提交哈希失败

    __version__ = "dev"  # 回退设置版本号为"dev"（开发版本）
    __version_tuple__ = (0, 0, __version__)  # 回退设置版本元组为(0, 0, "dev")


def _prev_minor_version_was(version_str):
    """检查给定的版本字符串是否匹配上一个次版本号。

    Check whether a given version matches the previous minor version.

    如果version_str匹配上一个次版本号，则返回True。
    Return True if version_str matches the previous minor version.

    例如：如果当前版本是0.7.4，且提供的version_str为'0.6'，则返回True。
    For example - return True if the current version if 0.7.4 and the
    supplied version_str is '0.6'.

    用于--show-hidden-metrics-for-version命令行参数。
    Used for --show-hidden-metrics-for-version.
    """
    # Match anything if this is a dev tree  # 如果是开发版本树，则匹配任何版本
    if __version_tuple__[0:2] == (0, 0):  # 检查主版本号和次版本号是否都为0（开发版本）
        return True  # 开发版本下直接返回True

    # Note - this won't do the right thing when we release 1.0!  # 注意：当发布1.0版本时此逻辑需要调整
    assert __version_tuple__[0] == 0  # 断言主版本号为0（当前仅支持0.x版本）
    assert isinstance(__version_tuple__[1], int)  # 断言次版本号是整数类型
    return version_str == f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"  # 比较给定版本是否等于上一个次版本号


def _prev_minor_version():
    """用于测试目的，返回上一个次版本号字符串。

    For the purpose of testing, return a previous minor version number.
    """
    # In dev tree, this will return "0.-1", but that will work fine"  # 在开发版本树中会返回"0.-1"，但不影响功能
    assert isinstance(__version_tuple__[1], int)  # 断言次版本号是整数类型
    return f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"  # 返回上一个次版本号，格式如"0.6"
