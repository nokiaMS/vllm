# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import zlib  # 导入zlib库，用于CRC32哈希计算
from collections.abc import Sequence  # 导入Sequence抽象基类，用于类型注解

import torch  # 导入PyTorch库

from vllm.pooling_params import LateInteractionParams, PoolingParams  # 从vllm导入晚期交互参数和池化参数类

LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"  # 晚期交互模式常量：缓存查询
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"  # 晚期交互模式常量：对文档评分


def get_late_interaction_engine_index(  # 获取晚期交互引擎索引，根据查询键将请求固定到特定引擎
    pooling_params: PoolingParams | None,  # 池化参数，可为None
    num_engines: int,  # 引擎数量
) -> int | None:  # 返回引擎索引或None
    if pooling_params is None or pooling_params.late_interaction_params is None:  # 如果池化参数为空或晚期交互参数为空
        return None  # 返回None

    late_interaction_params = pooling_params.late_interaction_params  # 获取晚期交互参数
    mode = late_interaction_params.mode  # 获取交互模式
    if mode not in (  # 如果模式不是已知的晚期交互模式
        LATE_INTERACTION_MODE_CACHE_QUERY,  # 缓存查询模式
        LATE_INTERACTION_MODE_SCORE_DOC,  # 文档评分模式
    ):
        return None  # 返回None

    query_key = late_interaction_params.query_key  # 获取查询键
    if not isinstance(query_key, str) or not query_key:  # 如果查询键不是字符串或为空
        return None  # 返回None

    # query embeddings are cached in process-local worker memory,
    # pin requests sharing the same query key to the same engine.
    return zlib.crc32(query_key.encode("utf-8")) % num_engines  # 用CRC32哈希将相同查询键的请求映射到同一个引擎


def build_late_interaction_query_params(  # 构建晚期交互查询参数，用于缓存查询嵌入
    query_key: str,  # 查询键，用于标识查询
    query_uses: int,  # 查询使用次数
) -> LateInteractionParams:  # 返回晚期交互参数对象
    return LateInteractionParams(  # 创建晚期交互参数实例
        mode=LATE_INTERACTION_MODE_CACHE_QUERY,  # 设置模式为缓存查询
        query_key=query_key,  # 设置查询键
        query_uses=max(1, int(query_uses)),  # 设置查询使用次数，至少为1
    )


def build_late_interaction_doc_params(  # 构建晚期交互文档参数，用于对文档进行评分
    query_key: str,  # 查询键，关联对应的查询嵌入
) -> LateInteractionParams:  # 返回晚期交互参数对象
    return LateInteractionParams(  # 创建晚期交互参数实例
        mode=LATE_INTERACTION_MODE_SCORE_DOC,  # 设置模式为文档评分
        query_key=query_key,  # 设置查询键
    )


def compute_maxsim_score(  # 计算单对查询和文档嵌入的MaxSim分数
    q_emb: torch.Tensor,  # 查询嵌入张量
    d_emb: torch.Tensor,  # 文档嵌入张量
) -> torch.Tensor:  # 返回MaxSim分数张量
    # compute in float32 for numerical stability
    token_scores = torch.matmul(q_emb.float(), d_emb.float().T)  # 计算查询和文档token之间的相似度矩阵，使用float32保证数值稳定性
    return token_scores.amax(dim=-1).sum()  # 对每个查询token取最大相似度，然后求和得到MaxSim分数


def compute_maxsim_scores(  # 批量计算多对查询/文档的MaxSim分数，支持小批量处理
    q_embs: Sequence[torch.Tensor],  # 查询嵌入序列
    d_embs: Sequence[torch.Tensor],  # 文档嵌入序列
    max_batch_size: int = 64,  # 最大批量大小，默认64
    max_score_matrix_elements: int = 64_000_000,  # 分数矩阵最大元素数，防止内存溢出
) -> list[torch.Tensor]:  # 返回MaxSim分数列表
    """Compute MaxSim for multiple query/doc pairs in mini-batches."""
    if len(q_embs) != len(d_embs):  # 如果查询和文档数量不匹配
        raise ValueError("q_embs and d_embs must have the same length")  # 抛出值错误

    num_pairs = len(q_embs)  # 获取查询/文档对数量
    if num_pairs == 0:  # 如果没有输入对
        return []  # 返回空列表

    if max_batch_size <= 0:  # 如果最大批量大小不合法
        raise ValueError("max_batch_size must be greater than 0")  # 抛出值错误
    if max_score_matrix_elements <= 0:  # 如果最大矩阵元素数不合法
        raise ValueError("max_score_matrix_elements must be greater than 0")  # 抛出值错误

    for q_emb, d_emb in zip(q_embs, d_embs):  # 遍历每对查询和文档嵌入进行验证
        if q_emb.ndim != 2 or d_emb.ndim != 2:  # 如果嵌入不是二维张量
            raise ValueError("Each embedding tensor must be 2-D")  # 抛出值错误
        if q_emb.shape[1] != d_emb.shape[1]:  # 如果查询和文档的嵌入维度不一致
            raise ValueError("Query and document embeddings must have same dim")  # 抛出值错误
        if q_emb.device != d_emb.device:  # 如果查询和文档不在同一设备上
            raise ValueError("Query and document embeddings must be on same device")  # 抛出值错误

    scores: list[torch.Tensor] = []  # 初始化分数列表
    start = 0  # 初始化批次起始索引
    while start < num_pairs:  # 循环处理所有查询/文档对
        end = min(start + max_batch_size, num_pairs)  # 计算当前批次的结束索引
        max_q = max(int(x.shape[0]) for x in q_embs[start:end])  # 当前批次中查询的最大token数
        max_d = max(int(x.shape[0]) for x in d_embs[start:end])  # 当前批次中文档的最大token数

        # keep score matrix bounded to avoid oversized allocations.
        while (  # 动态缩减批次大小以控制分数矩阵的内存占用
            end - start > 1  # 批次大小大于1时
            and (end - start) * max_q * max_d > max_score_matrix_elements  # 且矩阵元素总数超过限制
        ):
            end -= 1  # 缩减批次大小
            max_q = max(int(x.shape[0]) for x in q_embs[start:end])  # 重新计算最大查询token数
            max_d = max(int(x.shape[0]) for x in d_embs[start:end])  # 重新计算最大文档token数

        batch_q = q_embs[start:end]  # 获取当前批次的查询嵌入
        batch_d = d_embs[start:end]  # 获取当前批次的文档嵌入
        batch_size = end - start  # 当前批次大小
        device = batch_q[0].device  # 获取计算设备
        dim = int(batch_q[0].shape[1])  # 获取嵌入维度

        q_batch = torch.zeros(  # 创建零填充的查询批次张量
            (batch_size, max_q, dim), dtype=torch.float32, device=device  # 形状为(批次大小, 最大查询长度, 维度)
        )
        d_batch = torch.zeros(  # 创建零填充的文档批次张量
            (batch_size, max_d, dim), dtype=torch.float32, device=device  # 形状为(批次大小, 最大文档长度, 维度)
        )
        q_mask = torch.zeros((batch_size, max_q), dtype=torch.bool, device=device)  # 创建查询掩码张量，标记有效token位置
        d_mask = torch.zeros((batch_size, max_d), dtype=torch.bool, device=device)  # 创建文档掩码张量，标记有效token位置

        # copy to padded tensors
        for i, (q_emb, d_emb) in enumerate(zip(batch_q, batch_d)):  # 遍历当前批次，将数据复制到填充张量中
            q_len = int(q_emb.shape[0])  # 获取当前查询的实际长度
            d_len = int(d_emb.shape[0])  # 获取当前文档的实际长度
            q_batch[i, :q_len] = q_emb.to(device=device, dtype=torch.float32)  # 将查询嵌入复制到填充张量中
            d_batch[i, :d_len] = d_emb.to(device=device, dtype=torch.float32)  # 将文档嵌入复制到填充张量中
            q_mask[i, :q_len] = True  # 设置查询有效位置的掩码为True
            d_mask[i, :d_len] = True  # 设置文档有效位置的掩码为True

        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))  # 批量矩阵乘法计算token级相似度分数
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))  # 将文档填充位置的分数设为负无穷
        max_per_query = token_scores.amax(dim=-1)  # 对每个查询token取文档token中的最大相似度
        max_per_query.masked_fill_(~q_mask, 0.0)  # 将查询填充位置的最大值设为0
        batch_scores = max_per_query.sum(dim=-1)  # 对所有查询token的最大相似度求和，得到最终的MaxSim分数
        scores.extend(batch_scores.unbind(0))  # 将批次分数拆分并添加到结果列表中
        start = end  # 更新起始索引到下一批次

    return scores  # 返回所有查询/文档对的MaxSim分数列表
