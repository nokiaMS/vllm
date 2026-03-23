# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# 注册 BGE-M3 稀疏嵌入处理器插件入口点
def register_bge_m3_sparse_embeddings_processor():
    return "bge_m3_sparse_processor.sparse_embeddings_processor.BgeM3SparseEmbeddingsProcessor"  # noqa: E501
