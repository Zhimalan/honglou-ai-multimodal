"""
评测指标计算模块

检索指标：Recall@5、Precision@5（基于 golden_chapters 与检索回目集合）
生成指标：事实准确 / 无幻觉 / 完整度 三项均分
专项指标：幻觉发生率（no_hallucination <= 2 的比例）
"""
from typing import List, Dict, Optional


def retrieval_metrics(
    retrieved_chapters: List[int], golden_chapters: List[int], k: int = 5
) -> Dict[str, Optional[float]]:
    """计算单条检索的 Recall@k / Precision@k。

    方案 A 无检索（retrieved_chapters 为空），Recall/Precision 记为 0。
    若 golden 为空（如纯主观题），返回 None 表示不计入检索均值。
    """
    if not golden_chapters:
        return {"recall": None, "precision": None}

    gold_set = set(golden_chapters)
    # 取前 k 个检索回目
    ret_set = set(retrieved_chapters[:k])

    if not ret_set:
        # 方案 A：未检索
        return {"recall": 0.0, "precision": 0.0}

    hit = len(gold_set & ret_set)
    recall = hit / len(gold_set)
    precision = hit / len(ret_set)
    return {"recall": recall, "precision": precision}


def hallucination_flag(no_hallucination_score: int) -> bool:
    """no_hallucination <= 2 视为存在幻觉。"""
    return no_hallucination_score <= 2


def aggregate(results: List[dict]) -> dict:
    """对一组（同一方案）评测结果做聚合统计。

    results 中每个元素应含：
        - "golden_chapters": List[int]
        - "retrieved_chapters": List[int]
        - "judge": {"factual_accuracy", "no_hallucination", "completeness"}
        - "category": str
    """
    n = len(results)
    if n == 0:
        return {}

    # 检索指标（仅 golden 非空者计入）
    recalls, precisions = [], []
    for r in results:
        m = retrieval_metrics(r["retrieved_chapters"], r["golden_chapters"])
        if m["recall"] is not None:
            recalls.append(m["recall"])
            precisions.append(m["precision"])

    # 生成指标
    fac = [r["judge"]["factual_accuracy"] for r in results]
    noh = [r["judge"]["no_hallucination"] for r in results]
    com = [r["judge"]["completeness"] for r in results]

    # 幻觉率
    hallu = sum(1 for s in noh if hallucination_flag(s)) / n

    # 分类别统计（plot / relation / poetry）
    by_cat = {}
    cats = set(r["category"] for r in results)
    for cat in cats:
        sub = [r for r in results if r["category"] == cat]
        sub_fac = [r["judge"]["factual_accuracy"] for r in sub]
        sub_noh = [r["judge"]["no_hallucination"] for r in sub]
        sub_com = [r["judge"]["completeness"] for r in sub]
        sub_hallu = sum(1 for s in sub_noh if hallucination_flag(s)) / len(sub)
        by_cat[cat] = {
            "count": len(sub),
            "factual_accuracy": sum(sub_fac) / len(sub),
            "no_hallucination": sum(sub_noh) / len(sub),
            "completeness": sum(sub_com) / len(sub),
            "hallucination_rate": sub_hallu,
        }

    return {
        "count": n,
        "recall_at_5": sum(recalls) / len(recalls) if recalls else None,
        "precision_at_5": sum(precisions) / len(precisions) if precisions else None,
        "factual_accuracy": sum(fac) / n,
        "no_hallucination": sum(noh) / n,
        "completeness": sum(com) / n,
        "hallucination_rate": hallu,
        "by_category": by_cat,
    }


def fmt(v: Optional[float], scale: float = 1.0, pct: bool = False) -> str:
    """格式化输出辅助。"""
    if v is None:
        return "N/A"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v * scale:.2f}"
