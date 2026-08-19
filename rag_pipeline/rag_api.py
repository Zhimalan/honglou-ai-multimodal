"""
RAG 对外 API 接口
供 LangGraph Agent 工具节点调用的统一入口

使用方式:
    from rag_pipeline.rag_api import rag_search
    result = rag_search("林黛玉进贾府")
    print(result['context'])
    print(result['sources'])
"""
import os
import sys

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.rag_pipeline import HonglouRAG

# 全局单例，避免重复加载模型
_rag_instance = None


def get_rag() -> HonglouRAG:
    """获取 RAG 系统单例"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = HonglouRAG(use_hyde=True, use_reranker=True)
    return _rag_instance


def rag_search(
    query: str,
    top_k: int = 5,
    max_context: int = 3000,
) -> dict:
    """
    RAG 检索接口（供 Agent 调用）

    Args:
        query: 用户查询
        top_k: 返回文档数
        max_context: 上下文最大字符数

    Returns:
        {
            "query": 原始查询,
            "context": 检索到的上下文文本,
            "sources": 来源回目引用,
            "source_details": 每条来源的详细信息,
            "elapsed_time": 检索耗时,
            "hyde_used": 是否使用了HyDE,
        }
    """
    rag = get_rag()
    return rag.retrieve_with_context(query, top_k=top_k, max_context_chars=max_context)


def rag_search_fast(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    快速 RAG 检索（不使用 HyDE 和 Reranker，仅向量+BM25）
    适用于简单查询或需要快速响应的场景

    Args:
        query: 用户查询
        top_k: 返回文档数

    Returns:
        同 rag_search
    """
    rag = get_rag()
    # 临时关闭 HyDE 和 Reranker
    original_hyde = rag.use_hyde
    original_reranker = rag.use_reranker
    rag.use_hyde = False
    rag.use_reranker = False

    try:
        result = rag.retrieve_with_context(query, top_k=top_k)
    finally:
        rag.use_hyde = original_hyde
        rag.use_reranker = original_reranker

    return result


if __name__ == "__main__":
    # 快速测试
    print("测试 rag_search...")
    result = rag_search("探春理家做了哪些改革？", top_k=3)
    print(f"来源: {result['sources']}")
    print(f"耗时: {result['elapsed_time']:.2f}s")
    print(f"上下文前200字: {result['context'][:200]}...")
