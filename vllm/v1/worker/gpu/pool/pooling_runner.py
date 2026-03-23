# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.tasks import PoolingTask
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState


# 池化运行器：负责从模型隐藏状态中提取句子/文档级嵌入
# 当前仅支持decoder-only模型的"LAST"池化策略（取最后一个token的隐藏状态）
# 输出经L2归一化，适用于语义相似度和检索任务
# NOTE(woosuk): Currently, this class only supports the "LAST" pooling task
# on decoder-only models. How to support other pooling tasks and models
# is to be determined.
class PoolingRunner:
    def __init__(self, model: nn.Module):
        self.model = cast(VllmModelForPooling, model)

    # 检查模型是否支持池化，返回支持的任务类型列表
    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        assert "embed" in model.pooler.get_supported_tasks()
        return ["embed"]

    # 执行池化操作：提取每个请求最后一个token的隐藏状态并L2归一化
    # 同时返回有效性掩码，仅当序列长度等于prompt长度时结果有效
    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO(woosuk): Support different types of pooling tasks.
        last_hidden_states = hidden_states[input_batch.logits_indices]
        # TODO(woosuk): Make normalization optional.
        last_hidden_states = F.normalize(last_hidden_states, p=2, dim=-1)

        prompt_len = req_states.prompt_len.gpu[input_batch.idx_mapping]
        is_valid = input_batch.seq_lens == prompt_len
        return last_hidden_states, is_valid

    # 虚拟池化运行：用于CUDA Graph捕获时的预热，仅执行归一化操作
    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        F.normalize(hidden_states, p=2, dim=-1)
        return
