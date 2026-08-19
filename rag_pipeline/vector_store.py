"""
Chroma 向量数据库封装
持久化存储 1620 个红楼梦文本切片，支持相似度检索
"""
import os
import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np


class ChromaStore:
    """Chroma 向量数据库封装"""

    def __init__(
        self,
        persist_path: str = None,
        collection_name: str = "honglou_chunks",
    ):
        """
        初始化 Chroma 向量库

        Args:
            persist_path: 持久化存储路径
            collection_name: 集合名称
        """
        if persist_path is None:
            persist_path = os.path.join(os.path.dirname(__file__), "chroma_db")

        self.persist_path = persist_path
        self.collection_name = collection_name

        # 创建持久化客户端
        self.client = chromadb.PersistentClient(path=persist_path)

        # 获取或创建集合（使用余弦相似度）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"[Chroma] 集合 '{collection_name}' 就绪, 当前文档数: {self.collection.count()}")

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
    ):
        """
        批量写入切片及其嵌入向量

        Args:
            chunks: 切片列表, 每个含 chunk_id, chapter_num, text, char_count
            embeddings: 对应的嵌入向量 (n, 512)
        """
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "chapter_num": c["chapter_num"],
                "char_count": c["char_count"],
                "char_start": c.get("char_start", 0),
                "char_end": c.get("char_end", 0),
            }
            for c in chunks
        ]

        # 分批写入（Chroma单次上限约5000）
        batch_size = 500
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=metadatas[i:end],
            )
            print(f"  [Chroma] 写入进度: {end}/{total}")

        print(f"[Chroma] 写入完成, 总文档数: {self.collection.count()}")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        向量相似度检索

        Args:
            query_embedding: 查询向量 (512,)
            top_k: 返回Top-K结果

        Returns:
            结果列表, 每个含 chunk_id, text, chapter_num, score
        """
        results = self.collection.query(
            query_embeddings=query_embedding.reshape(1, -1).tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "chapter_num": results["metadatas"][0][i]["chapter_num"],
                "score": 1 - results["distances"][0][i],  # cosine distance -> similarity
                "retriever": "vector",
            })
        return items

    def clear(self):
        """清空集合（重建索引时用）"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[Chroma] 集合已清空")

    def count(self) -> int:
        """返回当前文档数"""
        return self.collection.count()


if __name__ == "__main__":
    store = ChromaStore()
    print(f"当前文档数: {store.count()}")
