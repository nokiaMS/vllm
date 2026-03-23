# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.v1.outputs import LogprobsTensors


# 采样器输出数据类
# 封装单次采样步骤的所有输出：采样的 token ID、可选的 logprobs 信息、NaN 统计和采样数量
# 所有字段均为 GPU 张量，便于后续高效处理
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
