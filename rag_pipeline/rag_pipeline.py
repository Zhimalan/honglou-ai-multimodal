"""
红楼梦 RAG 完整流水线
整合 BGE嵌入 + Chroma向量库 + BM25关键词召回 + HyDE假设文档 + Reranker重排序

检索流程:
  用户查询
    ├─ HyDE 生成假设文档 → BGE嵌入 → Chroma向量检索 (Top-K1)
    ├─ BM25 关键词检索 (Top-K1)
    └─ RRF融合 → BGE-Reranker精排 → Top-K2 最终结果
"""
import os
import sys
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# 设置 HuggingFace 镜像（国内）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.embedding import BGEEncoder
from rag_pipeline.vector_store import ChromaStore
from rag_pipeline.bm25_retriever import BM25Retriever
from rag_pipeline.hyde import HyDEGenerator
from rag_pipeline.reranker import BGEReranker
from rag_pipeline.fusion import reciprocal_rank_fusion


@dataclass
class RAGResult:
    """RAG 检索结果"""
    query: str
    documents: List[Dict] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    elapsed_time: float = 0.0
    hyde_used: bool = False
    hyde_doc: Optional[str] = None

    def get_context(self, max_chars: int = 3000) -> str:
        """拼接检索结果为上下文文本"""
        context_parts = []
        total = 0
        for doc in self.documents:
            text = doc["text"]
            source = f"【第{doc['chapter_num']}回】"
            snippet = f"{source}{text}"
            if total + len(snippet) > max_chars:
                break
            context_parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(context_parts)

    def get_sources_str(self) -> str:
        """获取来源引用字符串"""
        chapters = sorted(set(d["chapter_num"] for d in self.documents))
        return "、".join(f"第{ch}回" for ch in chapters)


class HonglouRAG:
    """红楼梦多级增强 RAG 系统"""

    def __init__(
        self,
        chunks_path: str = None,
        chroma_path: str = None,
        bm25_index_path: str = None,
        use_hyde: bool = True,
        use_reranker: bool = True,
    ):
        """
        初始化 RAG 系统

        Args:
            chunks_path: 切片数据路径
            chroma_path: Chroma持久化路径
            bm25_index_path: BM25索引路径
            use_hyde: 是否启用HyDE
            use_reranker: 是否启用Reranker
        """
        if chunks_path is None:
            chunks_path = os.path.join(PROJECT_ROOT, "data", "processed", "chunks_rag.json")
        if chroma_path is None:
            chroma_path = os.path.join(PROJECT_ROOT, "rag_pipeline", "chroma_db")
        if bm25_index_path is None:
            bm25_index_path = os.path.join(PROJECT_ROOT, "rag_pipeline", "bm25_index.pkl")

        self.chunks_path = chunks_path
        self.chroma_path = chroma_path
        self.bm25_index_path = bm25_index_path
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker

        # 加载切片数据
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[RAG] 加载切片数据: {len(self.chunks)} 条")

        # 初始化各模块
        self.encoder = BGEEncoder()
        self.vector_store = ChromaStore(persist_path=chroma_path)
        self.bm25 = BM25Retriever()

        # 可选模块
        self.hyde = HyDEGenerator() if use_hyde else None
        self.reranker = BGEReranker() if use_reranker else None

        # 加载BM25索引
        if os.path.exists(bm25_index_path):
            self.bm25.load_index(bm25_index_path)
        else:
            print("[RAG] BM25 索引不存在，需要先运行 build_index.py")

        self._initialized = False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        recall_k: int = 20,
    ) -> RAGResult:
        """
        执行完整 RAG 检索流程

        Args:
            query: 用户查询
            top_k: 最终返回的文档数
            recall_k: 每路召回的文档数

        Returns:
            RAGResult 对象
        """
        start = time.time()
        result = RAGResult(query=query)

        # === Step 1: 生成查询向量（可选 HyDE 增强）===
        if self.use_hyde and self.hyde:
            hyde_doc = self.hyde.generate(query)
            if hyde_doc:
                result.hyde_used = True
                result.hyde_doc = hyde_doc
                query_emb = self.encoder.encode(hyde_doc)
            else:
                query_emb = self.encoder.encode_query(query)
        else:
            query_emb = self.encoder.encode_query(query)

        # === Step 2: 多路召回 ===
        # 路线A: 向量相似度检索
        vector_results = self.vector_store.query(query_emb, top_k=recall_k)
        print(f"  [向量召回] {len(vector_results)} 条")

        # 路线B: BM25 关键词检索
        bm25_results = self.bm25.query(query, top_k=recall_k)
        print(f"  [BM25召回] {len(bm25_results)} 条")

        # === Step 3: RRF 融合 ===
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        print(f"  [RRF融合] {len(fused)} 条 (去重后)")

        # 取融合后 Top recall_k 进入重排
        candidates = fused[:recall_k]

        # === Step 4: Reranker 精排 ===
        if self.use_reranker and self.reranker and len(candidates) > 0:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            result.documents = reranked
            print(f"  [Reranker精排] {len(reranked)} 条")
        else:
            result.documents = candidates[:top_k]
            print(f"  [无重排] 取Top-{top_k}")

        # === Step 5: 构建来源引用 ===
        result.sources = [
            {
                "chapter_num": d["chapter_num"],
                "chunk_id": d["chunk_id"],
                "score": d.get("rerank_score", d.get("fusion_score", 0)),
            }
            for d in result.documents
        ]

        result.elapsed_time = time.time() - start
        return result

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        max_context_chars: int = 3000,
    ) -> Dict:
        """
        检索并返回可直接喂给LLM的上下文

        Returns:
            dict: {
                "query": 原始查询,
                "context": 拼接的上下文文本,
                "sources": 来源引用,
                "elapsed_time": 耗时,
            }
        """
        result = self.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "context": result.get_context(max_chars=max_context_chars),
            "sources": result.get_sources_str(),
            "source_details": result.sources,
            "elapsed_time": result.elapsed_time,
            "hyde_used": result.hyde_used,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("红楼梦 RAG 系统测试")
    print("=" * 60)

    rag = HonglouRAG()

    test_queries = [
        "林黛玉初进贾府时见到了哪些人？",
        "贾宝玉梦游太虚幻境看到了什么判词？",
        "王熙凤是怎么协理宁国府的？",
    ]

    for q in test_queries:
        print(f"\n{'─' * 50}")
        print(f"查询: {q}")
        result = rag.retrieve_with_context(q, top_k=3)
        print(f"来源: {result['sources']}")
        print(f"耗时: {result['elapsed_time']:.2f}s")
        print(f"HyDE: {'是' if result['hyde_used'] else '否'}")
        print(f"上下文前200字:\n{result['context'][:200]}...")
