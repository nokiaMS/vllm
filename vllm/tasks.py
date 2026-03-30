# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args  # 导入类型提示工具：Literal用于定义字面量类型，get_args用于获取泛型参数

GenerationTask = Literal["generate", "transcription", "realtime"]  # 定义生成任务类型：包括文本生成、转录和实时任务
GENERATION_TASKS: tuple[GenerationTask, ...] = get_args(GenerationTask)  # 获取所有生成任务类型的元组

PoolingTask = Literal[  # 定义池化任务类型
    "embed", "classify", "score", "token_embed", "token_classify", "plugin"  # 包括嵌入、分类、评分、token嵌入、token分类和插件
]
POOLING_TASKS: tuple[PoolingTask, ...] = get_args(PoolingTask)  # 获取所有池化任务类型的元组

# Score API handles score/rerank for:
# - "score" task (score_type: cross-encoder models)
# - "embed" task (score_type: bi-encoder models)
# - "token_embed" task (score_type: late interaction models)
ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]  # 定义评分类型：双编码器、交叉编码器、延迟交互

FrontendTask = Literal["render"]  # 定义前端任务类型：仅包含渲染任务
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)  # 获取所有前端任务类型的元组

SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]  # 定义所有支持的任务类型的联合类型
