"""
BGE-Reranker 重排序模块
使用 sentence-transformers CrossEncoder 加载 BAAI/bge-reranker-base
对召回结果做交叉编码器精排
"""
import os
from typing import List, Dict
from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE 交叉编码器重排序"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        cache_dir: str = None,
        device: str = "cpu",
    ):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        print(f"[Reranker] 正在加载重排序模型: {model_name} (device={device}) ...")
        self.reranker = CrossEncoder(
            model_name,
            max_length=512,
            cache_dir=cache_dir,
            device=device,
        )
        print(f"[Reranker] 模型加载完成")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        对召回文档做交叉编码器重排序

        Args:
            query: 用户查询
            documents: 召回的文档列表（含 text 字段）
            top_k: 返回Top-K

        Returns:
            重排序后的文档列表，添加 rerank_score 字段
        """
        if not documents:
            return []

        # 构造 query-document 对
        pairs = [[query, doc["text"]] for doc in documents]

        # 计算重排序得分
        scores = self.reranker.predict(pairs)

        # 填充 rerank_score 并按分数排序
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        # 排序
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]


if __name__ == "__main__":
    ranker = BGEReranker()
    docs = [
        {"text": "林黛玉初进贾府，贾母搂着黛玉哭个不停。", "chunk_id": "1"},
        {"text": "贾宝玉在太虚幻境中游历，见到了警幻仙子。", "chunk_id": "2"},
        {"text": "薛宝钗进了贾府，众人见她端庄稳重。", "chunk_id": "3"},
    ]
    result = ranker.rerank("林黛玉进贾府", docs, top_k=2)
    for r in result:
        print(f"  score={r['rerank_score']:.4f} | {r['text'][:50]}")
