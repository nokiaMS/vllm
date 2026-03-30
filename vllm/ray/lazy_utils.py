# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明


def is_ray_initialized():
    """检查 Ray 是否已经初始化。

    尝试导入 Ray 并调用其 is_initialized() 方法，
    如果 Ray 未安装或属性不存在则返回 False。
    """
    try:  # 尝试执行以下代码块
        import ray  # 导入 Ray 分布式计算框架

        return ray.is_initialized()  # 返回 Ray 是否已初始化的布尔值
    except ImportError:  # 捕获 Ray 未安装的导入错误
        return False  # Ray 未安装，返回 False
    except AttributeError:  # 捕获属性不存在的错误
        return False  # 属性不存在，返回 False


def is_in_ray_actor():
    """检查当前是否在 Ray Actor 中运行。

    通过检查 Ray 是否已初始化且当前运行时上下文中存在 Actor ID 来判断。
    如果 Ray 未安装或属性不存在则返回 False。
    """

    try:  # 尝试执行以下代码块
        import ray  # 导入 Ray 分布式计算框架

        return (  # 返回以下条件的布尔值
            ray.is_initialized()  # 检查 Ray 是否已初始化
            and ray.get_runtime_context().get_actor_id() is not None  # 检查当前是否在 Actor 中（Actor ID 不为 None）
        )
    except ImportError:  # 捕获 Ray 未安装的导入错误
        return False  # Ray 未安装，返回 False
    except AttributeError:  # 捕获属性不存在的错误
        return False  # 属性不存在，返回 False
