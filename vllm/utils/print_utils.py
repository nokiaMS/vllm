# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# 打印嵌入向量的摘要信息，超过 4 个元素时截断显示，用于调试时快速查看向量概览
def print_embeddings(embeds: list[float], prefix: str = "Embeddings"):
    embeds_trimmed = (str(embeds[:4])[:-1] + ", ...]") if len(embeds) > 4 else embeds
    print(f"{prefix}: {embeds_trimmed} (size={len(embeds)})")
