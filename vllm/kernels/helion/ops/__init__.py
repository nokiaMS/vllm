# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
"""自动导入所有Helion算子模块以触发内核注册。"""

import importlib  # 导入importlib模块，用于动态导入
import pkgutil  # 导入pkgutil模块，用于遍历包中的子模块

# Automatically import all submodules so that @register_kernel  # 自动导入所有子模块，使得@register_kernel
# decorators execute and register ops with torch.ops.vllm_helion.  # 装饰器能够执行并将算子注册到torch.ops.vllm_helion
for _module_info in pkgutil.iter_modules(__path__):  # 遍历当前包下的所有子模块
    importlib.import_module(f"{__name__}.{_module_info.name}")  # 动态导入每个子模块
