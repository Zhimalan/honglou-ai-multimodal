"""
构建 RAG 索引
1. 加载 1620 个文本切片
2. BGE 编码所有切片
3. 写入 Chroma 向量库
4. 构建 BM25 索引并持久化

运行: python rag_pipeline/build_index.py
"""
import os
import sys
import json
import time
import numpy as np

# 设置 HuggingFace 镜像（国内）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.embedding import BGEEncoder
from rag_pipeline.vector_store import ChromaStore
from rag_pipeline.bm25_retriever import BM25Retriever


def main():
    print("=" * 60)
    print("构建红楼梦 RAG 索引")
    print("=" * 60)

    chunks_path = os.path.join(PROJECT_ROOT, "data", "processed", "chunks_rag.json")
    chroma_path = os.path.join(PROJECT_ROOT, "rag_pipeline", "chroma_db")
    bm25_path = os.path.join(PROJECT_ROOT, "rag_pipeline", "bm25_index.pkl")

    # === Step 1: 加载切片 ===
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"\n[1/4] 加载切片: {len(chunks)} 条")

    # === Step 2: BGE 编码 ===
    print(f"\n[2/4] BGE 嵌入编码...")
    encoder = BGEEncoder()

    texts = [c["text"] for c in chunks]
    t0 = time.time()

    # 分批编码，避免内存溢出
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encoder.encode(batch, batch_size=32)
        all_embeddings.append(emb)
        elapsed = time.time() - t0
        progress = min(i + batch_size, len(texts))
        print(f"  编码进度: {progress}/{len(texts)} ({progress/len(texts)*100:.1f}%) | 耗时: {elapsed:.1f}s")

    all_embeddings = np.vstack(all_embeddings)
    print(f"  编码完成: shape={all_embeddings.shape}, 总耗时: {time.time()-t0:.1f}s")

    # === Step 3: 写入 Chroma ===
    print(f"\n[3/4] 写入 Chroma 向量库...")
    store = ChromaStore(persist_path=chroma_path)

    # 如果已有数据，先清空
    if store.count() > 0:
        print(f"  已有 {store.count()} 条数据，清空重建...")
        store.clear()

    store.add_chunks(chunks, all_embeddings)
    print(f"  Chroma 写入完成: {store.count()} 条")

    # === Step 4: 构建 BM25 索引 ===
    print(f"\n[4/4] 构建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    bm25.save_index(bm25_path)

    # === 验证 ===
    print(f"\n{'=' * 60}")
    print("索引构建完成！验证检索效果:")
    print(f"{'=' * 60}")

    test_query = "林黛玉初进贾府"
    print(f"\n测试查询: {test_query}")

    # 向量检索
    q_emb = encoder.encode_query(test_query)
    v_results = store.query(q_emb, top_k=3)
    print(f"\n向量召回 Top3:")
    for r in v_results:
        print(f"  [第{r['chapter_num']}回] score={r['score']:.4f} | {r['text'][:60]}...")

    # BM25 检索
    b_results = bm25.query(test_query, top_k=3)
    print(f"\nBM25召回 Top3:")
    for r in b_results:
        print(f"  [第{r['chapter_num']}回] score={r['score']:.4f} | {r['text'][:60]}...")

    print(f"\n✓ RAG 索引构建全部完成！")
    print(f"  Chroma: {chroma_path} ({store.count()} 条)")
    print(f"  BM25: {bm25_path}")


if __name__ == "__main__":
    main()
