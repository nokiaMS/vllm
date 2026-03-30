# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager  # 从推理解析器模块导入基类和管理器

__all__ = [  # 定义模块公开接口
    "ReasoningParser",  # 推理解析器基类
    "ReasoningParserManager",  # 推理解析器管理器
]
"""
Register a lazy module mapping.
注册延迟加载的模块映射。

Example:
    ReasoningParserManager.register_lazy_module(
        name="qwen3",
        module_path="vllm.reasoning.qwen3_reasoning_parser",
        class_name="Qwen3ReasoningParser",
    )
"""


_REASONING_PARSERS_TO_REGISTER = {  # 需要注册的推理解析器字典，格式为 {名称: (文件名, 类名)}
    "deepseek_r1": (  # DeepSeek R1 推理解析器
        "deepseek_r1_reasoning_parser",  # 文件名
        "DeepSeekR1ReasoningParser",  # 类名
    ),
    "deepseek_v3": (  # DeepSeek V3 推理解析器
        "deepseek_v3_reasoning_parser",
        "DeepSeekV3ReasoningParser",
    ),
    "ernie45": (  # 文心4.5推理解析器
        "ernie45_reasoning_parser",
        "Ernie45ReasoningParser",
    ),
    "glm45": (  # GLM-4.5 推理解析器（使用DeepSeek V3带思考链的解析器）
        "deepseek_v3_reasoning_parser",
        "DeepSeekV3ReasoningWithThinkingParser",
    ),
    "openai_gptoss": (  # OpenAI GPT-OSS 推理解析器
        "gptoss_reasoning_parser",
        "GptOssReasoningParser",
    ),
    "granite": (  # Granite 推理解析器
        "granite_reasoning_parser",
        "GraniteReasoningParser",
    ),
    "holo2": (  # Holo2 推理解析器（使用DeepSeek V3带思考链的解析器）
        "deepseek_v3_reasoning_parser",
        "DeepSeekV3ReasoningWithThinkingParser",
    ),
    "hunyuan_a13b": (  # 混元A13B推理解析器
        "hunyuan_a13b_reasoning_parser",
        "HunyuanA13BReasoningParser",
    ),
    "kimi_k2": (  # Kimi K2 推理解析器
        "kimi_k2_reasoning_parser",
        "KimiK2ReasoningParser",
    ),
    "minimax_m2": (  # MiniMax M2 推理解析器
        "minimax_m2_reasoning_parser",
        "MiniMaxM2ReasoningParser",
    ),
    "minimax_m2_append_think": (  # MiniMax M2 追加思考链推理解析器
        "minimax_m2_reasoning_parser",
        "MiniMaxM2AppendThinkReasoningParser",
    ),
    "mistral": (  # Mistral 推理解析器
        "mistral_reasoning_parser",
        "MistralReasoningParser",
    ),
    "nemotron_v3": (  # Nemotron V3 推理解析器
        "nemotron_v3_reasoning_parser",
        "NemotronV3ReasoningParser",
    ),
    "olmo3": (  # OLMo3 推理解析器
        "olmo3_reasoning_parser",
        "Olmo3ReasoningParser",
    ),
    "qwen3": (  # Qwen3 推理解析器
        "qwen3_reasoning_parser",
        "Qwen3ReasoningParser",
    ),
    "seed_oss": (  # Seed OSS 推理解析器
        "seedoss_reasoning_parser",
        "SeedOSSReasoningParser",
    ),
    "step3": (  # Step3 推理解析器
        "step3_reasoning_parser",
        "Step3ReasoningParser",
    ),
    "step3p5": (  # Step3.5 推理解析器
        "step3p5_reasoning_parser",
        "Step3p5ReasoningParser",
    ),
}


def register_lazy_reasoning_parsers():  # 注册所有延迟加载的推理解析器
    """注册所有预定义的推理解析器到解析器管理器中。"""
    for name, (file_name, class_name) in _REASONING_PARSERS_TO_REGISTER.items():  # 遍历所有待注册的解析器
        module_path = f"vllm.reasoning.{file_name}"  # 构建模块路径
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)  # 注册延迟加载模块


register_lazy_reasoning_parsers()  # 执行注册
