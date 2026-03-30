# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.abstract_tool_parser import (  # 从抽象工具解析器模块导入
    ToolParser,  # 导入工具解析器基类
    ToolParserManager,  # 导入工具解析器管理器
)

__all__ = ["ToolParser", "ToolParserManager"]  # 定义模块的公开接口


"""
Register a lazy module mapping.
注册延迟加载的模块映射。

Example:
    ToolParserManager.register_lazy_module(
        name="kimi_k2",
        module_path="vllm.tool_parsers.kimi_k2_parser",
        class_name="KimiK2ToolParser",
    )
"""


_TOOL_PARSERS_TO_REGISTER = {  # 需要注册的工具解析器字典，格式为 {名称: (文件名, 类名)}
    "deepseek_v3": (  # DeepSeek V3 工具解析器
        "deepseekv3_tool_parser",  # 文件名
        "DeepSeekV3ToolParser",  # 类名
    ),
    "deepseek_v31": (  # DeepSeek V3.1 工具解析器
        "deepseekv31_tool_parser",
        "DeepSeekV31ToolParser",
    ),
    "deepseek_v32": (  # DeepSeek V3.2 工具解析器
        "deepseekv32_tool_parser",
        "DeepSeekV32ToolParser",
    ),
    "ernie45": (  # 文心4.5工具解析器
        "ernie45_tool_parser",
        "Ernie45ToolParser",
    ),
    "glm45": (  # GLM-4.5 MOE模型工具解析器
        "glm4_moe_tool_parser",
        "Glm4MoeModelToolParser",
    ),
    "glm47": (  # GLM-4.7 MOE模型工具解析器
        "glm47_moe_tool_parser",
        "Glm47MoeModelToolParser",
    ),
    "granite-20b-fc": (  # Granite 20B 函数调用工具解析器
        "granite_20b_fc_tool_parser",
        "Granite20bFCToolParser",
    ),
    "granite": (  # Granite 工具解析器
        "granite_tool_parser",
        "GraniteToolParser",
    ),
    "hermes": (  # Hermes 2 Pro 工具解析器
        "hermes_tool_parser",
        "Hermes2ProToolParser",
    ),
    "hunyuan_a13b": (  # 混元A13B工具解析器
        "hunyuan_a13b_tool_parser",
        "HunyuanA13BToolParser",
    ),
    "internlm": (  # InternLM2 工具解析器
        "internlm2_tool_parser",
        "Internlm2ToolParser",
    ),
    "jamba": (  # Jamba 工具解析器
        "jamba_tool_parser",
        "JambaToolParser",
    ),
    "kimi_k2": (  # Kimi K2 工具解析器
        "kimi_k2_tool_parser",
        "KimiK2ToolParser",
    ),
    "llama3_json": (  # Llama3 JSON 工具解析器
        "llama_tool_parser",
        "Llama3JsonToolParser",
    ),
    "llama4_json": (  # Llama4 JSON 工具解析器（复用Llama3解析器）
        "llama_tool_parser",
        "Llama3JsonToolParser",
    ),
    "llama4_pythonic": (  # Llama4 Pythonic 风格工具解析器
        "llama4_pythonic_tool_parser",
        "Llama4PythonicToolParser",
    ),
    "longcat": (  # Longcat Flash 工具解析器
        "longcat_tool_parser",
        "LongcatFlashToolParser",
    ),
    "minimax_m2": (  # MiniMax M2 工具解析器
        "minimax_m2_tool_parser",
        "MinimaxM2ToolParser",
    ),
    "minimax": (  # MiniMax 工具解析器
        "minimax_tool_parser",
        "MinimaxToolParser",
    ),
    "mistral": (  # Mistral 工具解析器
        "mistral_tool_parser",
        "MistralToolParser",
    ),
    "olmo3": (  # OLMo3 Pythonic 工具解析器
        "olmo3_tool_parser",
        "Olmo3PythonicToolParser",
    ),
    "openai": (  # OpenAI 工具解析器
        "openai_tool_parser",
        "OpenAIToolParser",
    ),
    "phi4_mini_json": (  # Phi4 Mini JSON 工具解析器
        "phi4mini_tool_parser",
        "Phi4MiniJsonToolParser",
    ),
    "pythonic": (  # Pythonic 风格工具解析器
        "pythonic_tool_parser",
        "PythonicToolParser",
    ),
    "qwen3_coder": (  # Qwen3 Coder 工具解析器
        "qwen3coder_tool_parser",
        "Qwen3CoderToolParser",
    ),
    "qwen3_xml": (  # Qwen3 XML 工具解析器
        "qwen3xml_tool_parser",
        "Qwen3XMLToolParser",
    ),
    "seed_oss": (  # Seed OSS 工具解析器
        "seed_oss_tool_parser",
        "SeedOssToolParser",
    ),
    "step3": (  # Step3 工具解析器
        "step3_tool_parser",
        "Step3ToolParser",
    ),
    "step3p5": (  # Step3.5 工具解析器
        "step3p5_tool_parser",
        "Step3p5ToolParser",
    ),
    "xlam": (  # xLAM 工具解析器
        "xlam_tool_parser",
        "xLAMToolParser",
    ),
    "gigachat3": (  # GigaChat3 工具解析器
        "gigachat3_tool_parser",
        "GigaChat3ToolParser",
    ),
    "functiongemma": (  # FunctionGemma 工具解析器
        "functiongemma_tool_parser",
        "FunctionGemmaToolParser",
    ),
}


def register_lazy_tool_parsers():  # 注册所有延迟加载的工具解析器
    """注册所有预定义的工具解析器到解析器管理器中。"""
    for name, (file_name, class_name) in _TOOL_PARSERS_TO_REGISTER.items():  # 遍历所有待注册的解析器
        module_path = f"vllm.tool_parsers.{file_name}"  # 构建模块路径
        ToolParserManager.register_lazy_module(name, module_path, class_name)  # 注册延迟加载模块


register_lazy_tool_parsers()  # 执行注册
