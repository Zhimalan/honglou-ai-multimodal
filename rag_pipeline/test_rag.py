"""
RAG 系统完整测试
测试多路召回 + HyDE + Reranker 的端到端流程
"""
import os
import sys
import json

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.rag_pipeline import HonglouRAG


def main():
    print("=" * 60)
    print("红楼梦 RAG 系统完整测试")
    print("=" * 60)

    # 初始化 RAG 系统（首次运行需要下载 reranker 模型）
    rag = HonglouRAG(use_hyde=True, use_reranker=True)

    # 测试查询
    test_queries = [
        "林黛玉初进贾府时见到了哪些人？",
        "贾宝玉梦游太虚幻境看到了什么判词？",
        "王熙凤是怎么协理宁国府的？",
        "刘姥姥进大观园发生了什么有趣的事？",
        "晴雯撕扇是在什么情况下发生的？",
    ]

    results_summary = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"测试 {i}/{len(test_queries)}")
        print(f"查询: {query}")
        print(f"{'─' * 60}")

        result = rag.retrieve_with_context(query, top_k=3, max_context_chars=2000)

        print(f"\n[结果]")
        print(f"  来源: {result['sources']}")
        print(f"  耗时: {result['elapsed_time']:.2f}s")
        print(f"  HyDE: {'✓ 使用' if result['hyde_used'] else '✗ 未使用/失败'}")
        print(f"  检索文档:")
        for j, src in enumerate(result['source_details'], 1):
            print(f"    {j}. 第{src['chapter_num']}回 (score={src['score']:.4f})")
        print(f"  上下文前300字:")
        print(f"    {result['context'][:300]}...")

        results_summary.append({
            "query": query,
            "sources": result['sources'],
            "elapsed_time": result['elapsed_time'],
            "hyde_used": result['hyde_used'],
            "num_docs": len(result['source_details']),
        })

    # 汇总报告
    print(f"\n{'=' * 60}")
    print("测试汇总报告")
    print(f"{'=' * 60}")
    print(f"{'查询':<30} {'来源':<20} {'耗时':<8} {'HyDE':<6}")
    print(f"{'─' * 64}")
    for r in results_summary:
        print(f"{r['query'][:28]:<30} {r['sources'][:18]:<20} {r['elapsed_time']:.2f}s{'':>4} {'✓' if r['hyde_used'] else '✗':<6}")

    avg_time = sum(r['elapsed_time'] for r in results_summary) / len(results_summary)
    hyde_rate = sum(1 for r in results_summary if r['hyde_used']) / len(results_summary)
    print(f"\n平均耗时: {avg_time:.2f}s | HyDE成功率: {hyde_rate*100:.0f}%")

    # 保存测试报告
    report_path = os.path.join(PROJECT_ROOT, "report", "rag_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n测试报告已保存: {report_path}")


if __name__ == "__main__":
    main()
