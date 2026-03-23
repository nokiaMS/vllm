# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path
from typing import Any

import mteb
import numpy as np
import requests
import torch
from mteb.models import ModelMeta
from torch.utils.data import DataLoader

from tests.conftest import HfRunner
from tests.models.utils import (
    RerankModelInfo,
    get_vllm_extra_kwargs,
)

# See #19344
MTEB_RERANK_TASKS = ["NFCorpus"]
MTEB_RERANK_LANGS = ["eng"]
MTEB_RERANK_TOL = 2e-3

template_home = (
    Path(__file__).parent.parent.parent.parent.parent
    / "examples/pooling/score/template"
)

_empty_model_meta = ModelMeta(
    loader=None,
    name="vllm/model",
    revision="1",
    release_date=None,
    languages=None,
    framework=[],
    similarity_fn_name=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=None,
    training_datasets=None,
    modalities=["text"],  # 'image' can be added to evaluate multimodal models
)


# MTEB 交叉编码器的混入类：提供 MTEB 评估所需的模型元数据
class MtebCrossEncoderMixin(mteb.CrossEncoderProtocol):
    mteb_model_meta = _empty_model_meta


# vLLM MTEB 交叉编码器：将 vLLM 模型封装为 MTEB 重排序评估兼容的交叉编码器
class VllmMtebCrossEncoder(MtebCrossEncoderMixin):
    def __init__(self, vllm_model):
        self.llm = vllm_model
        self.rng = np.random.default_rng(seed=42)
        self.chat_template: str | None = getattr(vllm_model, "chat_template", None)

    def predict(
        self,
        inputs1: DataLoader[mteb.types.BatchedInput],
        inputs2: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        queries = [text for batch in inputs1 for text in batch["text"]]
        corpus = [text for batch in inputs2 for text in batch["text"]]

        # Hoping to discover potential scheduling
        # issues by randomizing the order.
        r = self.rng.permutation(len(queries))
        queries = [queries[i] for i in r]
        corpus = [corpus[i] for i in r]

        outputs = self.llm.score(
            queries,
            corpus,
            use_tqdm=False,
            chat_template=self.chat_template,
            tokenization_kwargs={"truncate_prompt_tokens": -1},
        )
        scores = np.array(outputs)
        scores = scores[np.argsort(r)]
        return scores


# 评分 API 客户端 MTEB 编码器：通过 HTTP 评分接口进行 MTEB 重排序评估
class ScoreClientMtebEncoder(MtebCrossEncoderMixin):
    mteb_model_meta = _empty_model_meta

    def __init__(self, model_name: str, url):
        self.model_name = model_name
        self.url = url

    def predict(
        self,
        inputs1: DataLoader[mteb.types.BatchedInput],
        inputs2: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        queries = [text for batch in inputs1 for text in batch["text"]]
        full_corpus = [text for batch in inputs2 for text in batch["text"]]

        outputs = []
        for query, corpus in zip(queries, full_corpus):
            outputs.append(self.get_score(query, corpus))

        scores = np.array(outputs)
        return scores

    def get_score(self, query, corpus):
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "queries": query,
                "documents": corpus,
                "truncate_prompt_tokens": -1,
            },
        ).json()
        return response["data"][0]["score"]


# 重排序 API 客户端 MTEB 编码器：通过 HTTP 重排序接口进行 MTEB 评估
class RerankClientMtebEncoder(ScoreClientMtebEncoder):
    def get_score(self, query, corpus):
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "query": query,
                "documents": [corpus],
                "truncate_prompt_tokens": -1,
            },
        ).json()
        return response["results"][0]["relevance_score"]


# HuggingFace MTEB 交叉编码器：将 HF 模型封装为 MTEB 评估兼容的交叉编码器，支持聊天模板
class HFMtebCrossEncoder(MtebCrossEncoderMixin, HfRunner):
    chat_template: str | None = None

    def __init__(self, model_name: str, dtype: str = "auto", **kwargs: Any) -> None:
        HfRunner.__init__(
            self, model_name=model_name, is_cross_encoder=True, dtype=dtype, **kwargs
        )

    @torch.no_grad
    def predict(
        self,
        inputs1: DataLoader[mteb.types.BatchedInput],
        inputs2: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        queries = [text for batch in inputs1 for text in batch["text"]]
        corpus = [text for batch in inputs2 for text in batch["text"]]

        if self.chat_template is not None:
            tokenizer = self.model.tokenizer
            prompts = []
            for query, document in zip(queries, corpus):
                conversation = [
                    {"role": "query", "content": query},
                    {"role": "document", "content": document},
                ]

                prompt = tokenizer.apply_chat_template(
                    conversation=conversation,
                    tools=None,
                    chat_template=self.chat_template,
                    tokenize=False,
                )
                prompts.append(prompt)
            outputs_list = HfRunner.classify(self, prompts)
            scores = np.array(outputs_list).squeeze(-1)
            return scores
        else:
            prompts = list(zip(queries, corpus))
            outputs_tensor = HfRunner.predict(self, prompts, show_progress_bar=False)
            return outputs_tensor.cpu().numpy()


# 辅助函数：运行 MTEB 重排序任务评估，先用 BM25 检索再用交叉编码器重排序
def run_mteb_rerank(cross_encoder: mteb.CrossEncoderProtocol, tasks, languages):
    with tempfile.TemporaryDirectory() as prediction_folder:
        bm25s = mteb.get_model("bm25s")
        eval_splits = ["test"]

        mteb_tasks: list[mteb.abstasks.AbsTaskRetrieval] = mteb.get_tasks(
            tasks=tasks, languages=languages, eval_splits=eval_splits
        )
        for task in mteb_tasks:
            if not task.data_loaded:
                task.load_data()

        mteb.evaluate(
            bm25s,
            mteb_tasks,
            prediction_folder=prediction_folder,
            show_progress_bar=False,
            # don't save results for test runs
            cache=None,
            overwrite_strategy="always",
        )

        second_stage_tasks = []
        for task in mteb_tasks:
            second_stage_tasks.append(
                task.convert_to_reranking(
                    prediction_folder,
                    top_k=10,
                )
            )

        results = mteb.evaluate(
            cross_encoder,
            second_stage_tasks,
            show_progress_bar=False,
            cache=None,
        )
        main_score = results[0].scores["test"][0]["main_score"]
    return main_score


# 核心测试函数：对重排序模型执行完整的 MTEB 评估流程，包括配置验证和分数对比
def mteb_test_rerank_models(
    vllm_runner,
    model_info: RerankModelInfo,
    hf_runner=HFMtebCrossEncoder,
    vllm_extra_kwargs=None,
    vllm_mteb_encoder=VllmMtebCrossEncoder,
    atol=MTEB_RERANK_TOL,
):
    vllm_extra_kwargs = get_vllm_extra_kwargs(model_info, vllm_extra_kwargs)

    # Maybe load chat_template.
    chat_template: str | None = None
    if model_info.chat_template_name is not None:
        chat_template = (template_home / model_info.chat_template_name).read_text()

    with vllm_runner(
        model_info.name,
        runner="pooling",
        max_model_len=None,
        max_num_seqs=8,
        **vllm_extra_kwargs,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        vllm_model.chat_template = chat_template

        # Confirm whether vllm is using the correct architecture
        if model_info.architecture:
            assert model_info.architecture in model_config.architectures

        # Score API is only enabled for num_labels == 1
        assert model_config.hf_config.num_labels == 1

        # Confirm whether the important configs in model_config are correct.
        pooler_config = model_config.pooler_config
        if model_info.seq_pooling_type is not None:
            assert pooler_config.seq_pooling_type == model_info.seq_pooling_type
        if model_info.tok_pooling_type is not None:
            assert pooler_config.tok_pooling_type == model_info.tok_pooling_type
        if model_info.attn_type is not None:
            assert model_config.attn_type == model_info.attn_type
        if model_info.is_prefix_caching_supported is not None:
            assert (
                model_config.is_prefix_caching_supported
                == model_info.is_prefix_caching_supported
            )
        if model_info.is_chunked_prefill_supported is not None:
            assert (
                model_config.is_chunked_prefill_supported
                == model_info.is_chunked_prefill_supported
            )

        vllm_main_score = run_mteb_rerank(
            vllm_mteb_encoder(vllm_model),
            tasks=MTEB_RERANK_TASKS,
            languages=MTEB_RERANK_LANGS,
        )
        vllm_dtype = model_config.dtype
        head_dtype = model_config.head_dtype

    # Accelerate mteb test by setting
    # SentenceTransformers mteb score to a constant
    if model_info.mteb_score is None:
        with hf_runner(model_info.name, dtype=model_info.hf_dtype) as hf_model:
            hf_model.chat_template = chat_template
            st_main_score = run_mteb_rerank(
                hf_model,
                tasks=MTEB_RERANK_TASKS,
                languages=MTEB_RERANK_LANGS,
            )
            st_dtype = next(hf_model.model.model.parameters()).dtype
    else:
        st_main_score = model_info.mteb_score
        st_dtype = "Constant"

    print("Model:", model_info.name)
    print("VLLM:", f"dtype:{vllm_dtype}", f"head_dtype:{head_dtype}", vllm_main_score)
    print("SentenceTransformers:", st_dtype, st_main_score)
    print("Difference:", st_main_score - vllm_main_score)

    # We are not concerned that the vllm mteb results are better
    # than SentenceTransformers, so we only perform one-sided testing.
    assert st_main_score - vllm_main_score < atol
