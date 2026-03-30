# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs  # 导入vLLM环境变量配置模块

if envs.VLLM_USE_MODELSCOPE:  # 如果启用了ModelScope模型仓库
    try:
        # Patch here, before each import happens
        # 在此处打补丁，需在每次导入之前执行
        import modelscope  # 导入ModelScope库
        from packaging import version  # 导入版本解析工具

        # patch_hub begins from modelscope>=1.18.1
        # patch_hub功能从modelscope>=1.18.1版本开始可用
        if version.parse(modelscope.__version__) <= version.parse("1.18.0"):  # 检查ModelScope版本是否过低
            raise ImportError(  # 抛出导入错误
                "Using vLLM with ModelScope needs modelscope>=1.18.1, please "
                "install by `pip install modelscope -U`"
            )
        from modelscope.utils.hf_util import patch_hub  # 从ModelScope导入HuggingFace Hub补丁工具

        # Patch hub to download models from modelscope to speed up.
        # 对Hub打补丁，从ModelScope下载模型以加速
        patch_hub()  # 执行Hub补丁
    except ImportError as err:  # 捕获导入错误
        raise ImportError(  # 抛出友好的错误提示
            "Please install modelscope>=1.18.1 via "
            "`pip install modelscope>=1.18.1` to use ModelScope."
        ) from err
