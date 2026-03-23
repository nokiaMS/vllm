# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.pooling_params import PoolingParams
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_SCORE_DOC,
    compute_maxsim_scores,
)


# 后交互（Late Interaction）评分运行器，用于ColBERT等后交互检索模型
# 设计思路：维护查询缓存，支持一个查询对多个文档的MaxSim评分
# 工作流程：1) 缓存查询请求的token级嵌入  2) 文档请求完成时，
# 从缓存取出对应查询嵌入，计算MaxSim相似度分数
class LateInteractionRunner:
    """Worker-side state and postprocessing for late-interaction scoring."""

    def __init__(self) -> None:
        # query_key -> token embeddings for late-interaction scoring.
        self._query_cache: dict[str, torch.Tensor] = {}
        # query_key -> remaining number of docs that should use this query.
        self._query_uses: dict[str, int] = {}
        # doc request id -> query key.
        self._doc_query_keys: dict[str, str] = {}

    def clear(self) -> None:
        self._query_cache.clear()
        self._query_uses.clear()
        self._doc_query_keys.clear()

    # 注册新请求：若为文档评分请求，记录其关联的查询键
    def register_request(
        self, req_id: str, pooling_params: PoolingParams | None
    ) -> None:
        mode, query_key, _ = self._parse_late_interaction_meta(pooling_params)
        if mode == LATE_INTERACTION_MODE_SCORE_DOC and query_key is not None:
            self._doc_query_keys[req_id] = query_key
        else:
            self._doc_query_keys.pop(req_id, None)

    # 请求完成回调：释放已完成文档请求对应的查询引用计数
    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        for req_id in finished_req_ids:
            query_key = self._doc_query_keys.pop(req_id, None)
            if query_key is not None:
                self._release_query_use(query_key)

    # 后处理池化输出：对已完成的请求执行后交互评分逻辑
    # 查询请求：将嵌入存入缓存，返回零值占位
    # 文档请求：从缓存取查询嵌入，计算MaxSim分数替换原始输出
    def postprocess_pooler_output(
        self,
        raw_pooler_output: PoolerOutput,
        pooling_params: list[PoolingParams],
        req_ids: list[str],
        finished_mask: list[bool],
    ) -> PoolerOutput:
        if not isinstance(raw_pooler_output, list):
            return raw_pooler_output

        num_reqs = len(pooling_params)
        if len(raw_pooler_output) != num_reqs:
            raise ValueError(
                "raw_pooler_output and pooling_params must have the same length."
            )
        if len(req_ids) != num_reqs:
            raise ValueError("req_ids and pooling_params must have the same length.")
        if len(finished_mask) != num_reqs:
            raise ValueError(
                "finished_mask and pooling_params must have the same length."
            )

        if not any(finished_mask):
            return raw_pooler_output
        if not any(p.late_interaction_params is not None for p in pooling_params):
            return raw_pooler_output

        outputs: list[torch.Tensor | None] = list(raw_pooler_output)
        score_indices: list[int] = []
        score_req_ids: list[str] = []
        score_query_keys: list[str] = []
        score_queries: list[torch.Tensor] = []
        score_docs: list[torch.Tensor] = []
        for i, (req_id, output, params, finished) in enumerate(
            zip(req_ids, outputs, pooling_params, finished_mask)
        ):
            if not finished or output is None:
                continue

            mode, query_key, query_uses = self._parse_late_interaction_meta(params)
            if mode is None:
                continue

            assert query_key is not None
            if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
                assert query_uses is not None
                # `output` can be a view into the current step's hidden-states
                # buffer, so clone it before storing across scheduling steps.
                self._query_cache[query_key] = output.clone()
                self._query_uses[query_key] = query_uses
                outputs[i] = torch.zeros((), device=output.device, dtype=torch.float32)
                continue

            if mode == LATE_INTERACTION_MODE_SCORE_DOC:
                query_output = self._query_cache.get(query_key)
                if query_output is None:
                    raise ValueError(
                        "late-interaction query cache miss for key "
                        f"{query_key!r}. Ensure query requests are executed "
                        "before their paired document requests."
                    )

                score_indices.append(i)
                score_req_ids.append(req_id)
                score_query_keys.append(query_key)
                score_queries.append(query_output)
                score_docs.append(output)
                continue

            raise ValueError(f"Unsupported late-interaction mode: {mode!r}")

        if score_indices:
            score_values = compute_maxsim_scores(score_queries, score_docs)
            for i, req_id, query_key, score in zip(
                score_indices, score_req_ids, score_query_keys, score_values
            ):
                outputs[i] = score
                self._doc_query_keys.pop(req_id, None)
                self._release_query_use(query_key)

        return outputs

    # 释放查询引用计数：计数归零时从缓存中删除查询嵌入以释放GPU内存
    def _release_query_use(self, query_key: str) -> None:
        remaining = self._query_uses.get(query_key, 1) - 1
        if remaining <= 0:
            self._query_uses.pop(query_key, None)
            self._query_cache.pop(query_key, None)
        else:
            self._query_uses[query_key] = remaining

    # 解析池化参数中的后交互元数据：提取模式（缓存查询/文档评分）、查询键和使用次数
    @staticmethod
    def _parse_late_interaction_meta(
        pooling_params: PoolingParams | None,
    ) -> tuple[str | None, str | None, int | None]:
        if pooling_params is None or pooling_params.late_interaction_params is None:
            return None, None, None

        late_interaction_params = pooling_params.late_interaction_params
        mode = late_interaction_params.mode

        query_key = late_interaction_params.query_key
        if not isinstance(query_key, str) or not query_key:
            raise ValueError(
                "late-interaction request is missing a valid query key in "
                "pooling_params.late_interaction_params."
            )

        if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
            query_uses_raw = late_interaction_params.query_uses
            if query_uses_raw is None:
                query_uses_raw = 1
            try:
                query_uses = max(1, int(query_uses_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "late-interaction query uses must be an integer value."
                ) from exc
            return mode, query_key, query_uses

        return mode, query_key, None
