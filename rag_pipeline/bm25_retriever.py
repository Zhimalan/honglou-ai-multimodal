"""
BM25 关键词召回模块
使用 jieba 中文分词 + rank_bm25 实现
"""
import os
import json
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import pickle


class BM25Retriever:
    """BM25 关键词召回"""

    # 红楼梦高频人物名，加入jieba自定义词典提升分词质量
    CUSTOM_WORDS = [
        "贾宝玉", "林黛玉", "薛宝钗", "王熙凤", "贾母", "贾政", "贾琏",
        "史湘云", "探春", "迎春", "惜春", "李纨", "秦可卿", "妙玉",
        "袭人", "晴雯", "紫鹃", "平儿", "鸳鸯", "香菱", "薛蟠", "贾环",
        "贾兰", "贾蓉", "贾珍", "尤氏", "王夫人", "邢夫人", "赵姨娘",
        "刘姥姥", "甄士隐", "贾雨村", "北静王", "林如海", "贾赦",
        "荣国府", "宁国府", "大观园", "太虚幻境", "警幻仙子",
        "金陵十二钗", "怡红院", "潇湘馆", "蘅芜苑", "稻香村",
    ]

    def __init__(self):
        # 添加自定义词典
        for word in self.CUSTOM_WORDS:
            jieba.add_word(word)

        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        print("[BM25] 初始化完成（含红楼梦自定义词典）")

    def tokenize(self, text: str) -> List[str]:
        """jieba 中文分词"""
        return list(jieba.cut(text))

    def build_index(self, chunks: List[Dict]):
        """
        构建 BM25 索引

        Args:
            chunks: 切片列表
        """
        self.chunks = chunks
        self.tokenized_corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] 索引构建完成, 文档数: {len(chunks)}")

    def save_index(self, path: str):
        """持久化保存索引"""
        data = {
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25] 索引已保存: {path}")

    def load_index(self, path: str):
        """加载持久化索引"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] 索引已加载, 文档数: {len(self.chunks)}")

    def query(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回Top-K

        Returns:
            结果列表, 每个含 chunk_id, text, chapter_num, score
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build_index()")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "chapter_num": chunk["chapter_num"],
                "score": float(scores[idx]),
                "retriever": "bm25",
            })
        return results


if __name__ == "__main__":
    # 测试
    import os
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "chunks_rag.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    retriever = BM25Retriever()
    retriever.build_index(chunks[:200])  # 小规模测试

    results = retriever.query("林黛玉进贾府", top_k=3)
    for r in results:
        print(f"  [第{r['chapter_num']}回] score={r['score']:.4f} | {r['text'][:60]}...")
