"""
RAG 多路召回融合模块
使用 RRF (Reciprocal Rank Fusion) 融合向量召回和 BM25 召回结果
"""
from typing import List, Dict
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion) 多路召回融合

    公式: RRF_score(d) = sum(1 / (k + rank_i(d)))

    Args:
        result_lists: 多路召回的结果列表
        k: RRF 平滑参数（默认60，业界标准）

    Returns:
        融合后的文档列表，按融合分数排序
    """
    rrf_scores = defaultdict(float)
    doc_info = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            chunk_id = doc["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            # 保留文档信息（以最后一次出现的为准）
            doc_info[chunk_id] = doc

    # 按融合分数排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids:
        doc = doc_info[chunk_id].copy()
        doc["fusion_score"] = rrf_scores[chunk_id]
        results.append(doc)

    return results


def weighted_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict]:
    """
    加权分数融合（归一化后加权求和）

    Args:
        vector_results: 向量召回结果
        bm25_results: BM25召回结果
        vector_weight: 向量权重
        bm25_weight: BM25权重

    Returns:
        融合后的文档列表
    """
    # 归一化分数
    def normalize(results, score_key="score"):
        if not results:
            return results
        scores = [r[score_key] for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            for r in results:
                r["norm_score"] = 1.0
        else:
            for r in results:
                r["norm_score"] = (r[score_key] - min_s) / (max_s - min_s)
        return results

    vector_results = normalize(vector_results)
    bm25_results = normalize(bm25_results)

    fused_scores = defaultdict(float)
    doc_info = {}

    for doc in vector_results:
        chunk_id = doc["chunk_id"]
        fused_scores[chunk_id] += vector_weight * doc["norm_score"]
        doc_info[chunk_id] = doc

    for doc in bm25_results:
        chunk_id = doc["chunk_id"]
        fused_scores[chunk_id] += bm25_weight * doc["norm_score"]
        if chunk_id not in doc_info:
            doc_info[chunk_id] = doc

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids:
        doc = doc_info[chunk_id].copy()
        doc["fusion_score"] = fused_scores[chunk_id]
        results.append(doc)

    return results
