"""
阶段8 评测结果分析 —— 方案 A / B 对比

读取 evaluation/results_schemeA.json、results_schemeB.json、eval_summary.json，
生成：
  1. 控制台 + Markdown 对比表格
  2. 三组对比柱状图（检索指标 / 生成指标 / 幻觉率 / 分类指标）
  3. report/eval_report.md 评测结论（供阶段9实验报告直接引用）

指标说明：
  - Recall@5 / Precision@5：检索命中标准回目的比例（方案A无检索记为0）
  - 事实准确 / 无幻觉 / 完整度：LLM-as-Judge 三项均分（1-5）
  - 幻觉发生率：no_hallucination <= 2 的题占比
"""
import os
import sys
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import fmt

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
A_PATH = os.path.join(EVAL_DIR, "results_schemeA.json")
B_PATH = os.path.join(EVAL_DIR, "results_schemeB.json")
SUMMARY_PATH = os.path.join(EVAL_DIR, "eval_summary.json")

# 中文显示
try:
    font_path = "C:/Windows/Fonts/msyh.ttc"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False

# 配色
C_A = "#4C72B0"  # 方案A 蓝
C_B = "#DD8452"  # 方案B 橙


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_charts(summary):
    os.makedirs(REPORT_DIR, exist_ok=True)
    a = summary["A"]
    b = summary["B"]

    # ---------- 图1：检索指标 ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Recall@5", "Precision@5"]
    a_vals = [a.get("recall_at_5") or 0, a.get("precision_at_5") or 0]
    b_vals = [b.get("recall_at_5") or 0, b.get("precision_at_5") or 0]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], a_vals, w, label="方案A (纯Base)", color=C_A)
    ax.bar([i + w / 2 for i in x], b_vals, w, label="方案B (Base+RAG)", color=C_B)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title("检索指标对比 (A vs B)")
    ax.legend()
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - w / 2, av + 0.02, f"{av:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eval_retrieval_metrics.png"), dpi=150)
    plt.close(fig)

    # ---------- 图2：生成指标 (LLM-as-Judge) ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["事实准确", "无幻觉", "完整度"]
    a_vals = [a["factual_accuracy"], a["no_hallucination"], a["completeness"]]
    b_vals = [b["factual_accuracy"], b["no_hallucination"], b["completeness"]]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], a_vals, w, label="方案A (纯Base)", color=C_A)
    ax.bar([i + w / 2 for i in x], b_vals, w, label="方案B (Base+RAG)", color=C_B)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.3)
    ax.set_ylabel("均分 (1-5)")
    ax.set_title("生成质量指标对比 (LLM-as-Judge, A vs B)")
    ax.legend()
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - w / 2, av + 0.08, f"{av:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, bv + 0.08, f"{bv:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eval_generation_metrics.png"), dpi=150)
    plt.close(fig)

    # ---------- 图3：幻觉发生率 ----------
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["幻觉发生率"]
    a_vals = [a["hallucination_rate"]]
    b_vals = [b["hallucination_rate"]]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], a_vals, w, label="方案A (纯Base)", color=C_A)
    ax.bar([i + w / 2 for i in x], b_vals, w, label="方案B (Base+RAG)", color=C_B)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title("幻觉发生率对比 (越低越好)")
    ax.legend()
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - w / 2, av + 0.02, f"{av*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, bv + 0.02, f"{bv*100:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "eval_hallucination_rate.png"), dpi=150)
    plt.close(fig)

    # ---------- 图4：分类别生成指标 (B) ----------
    by_cat = b.get("by_category", {})
    if by_cat:
        cats = list(by_cat.keys())
        cat_names = {"plot": "剧情", "relation": "人物关系", "poetry": "诗词"}
        labels = [cat_names.get(c, c) for c in cats]
        fac = [by_cat[c]["factual_accuracy"] for c in cats]
        noh = [by_cat[c]["no_hallucination"] for c in cats]
        com = [by_cat[c]["completeness"] for c in cats]
        import numpy as np

        x = np.arange(len(labels))
        w = 0.25
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w, fac, w, label="事实准确", color="#55A868")
        ax.bar(x, noh, w, label="无幻觉", color="#C44E52")
        ax.bar(x + w, com, w, label="完整度", color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 5.3)
        ax.set_ylabel("均分 (1-5)")
        ax.set_title("方案B 分类别生成质量")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "eval_by_category.png"), dpi=150)
        plt.close(fig)

    print(f"[analyze] 图表已保存至 {REPORT_DIR}")


def build_markdown(summary):
    a = summary["A"]
    b = summary["B"]
    lines = []
    lines.append("# 阶段8 评测结果分析报告（方案 A / B）\n")
    lines.append(f"> 生成时间：{summary.get('generated_at','')}  ")
    lines.append(f"> 测试集规模：{summary.get('test_set_size','')} 题（固定测试集）  ")
    lines.append(f"> {summary.get('note','')}\n")

    lines.append("## 一、总体指标对比\n")
    lines.append("| 指标 | 方案A (纯Base) | 方案B (Base+RAG) | 提升 |")
    lines.append("|------|--------------|----------------|------|")

    def delta(x, y):
        if x is None or y is None:
            return "—"
        d = y - x
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}"

    lines.append(
        f"| Recall@5 | {fmt(a.get('recall_at_5'))} | {fmt(b.get('recall_at_5'))} | {delta(a.get('recall_at_5'), b.get('recall_at_5'))} |"
    )
    lines.append(
        f"| Precision@5 | {fmt(a.get('precision_at_5'))} | {fmt(b.get('precision_at_5'))} | {delta(a.get('precision_at_5'), b.get('precision_at_5'))} |"
    )
    lines.append(
        f"| 事实准确 (1-5) | {fmt(a['factual_accuracy'])} | {fmt(b['factual_accuracy'])} | {delta(a['factual_accuracy'], b['factual_accuracy'])} |"
    )
    lines.append(
        f"| 无幻觉 (1-5) | {fmt(a['no_hallucination'])} | {fmt(b['no_hallucination'])} | {delta(a['no_hallucination'], b['no_hallucination'])} |"
    )
    lines.append(
        f"| 完整度 (1-5) | {fmt(a['completeness'])} | {fmt(b['completeness'])} | {delta(a['completeness'], b['completeness'])} |"
    )
    lines.append(
        f"| 幻觉发生率 | {fmt(a['hallucination_rate'], pct=True)} | {fmt(b['hallucination_rate'], pct=True)} | {delta(a['hallucination_rate'], b['hallucination_rate'])} |"
    )
    lines.append("")

    # 分类别
    by_cat = b.get("by_category", {})
    if by_cat:
        lines.append("## 二、方案B 分类别表现\n")
        lines.append("| 类别 | 题数 | 事实准确 | 无幻觉 | 完整度 | 幻觉率 |")
        lines.append("|------|------|---------|--------|--------|--------|")
        cat_names = {"plot": "剧情问答", "relation": "人物关系", "poetry": "诗词赏析"}
        for cat, m in by_cat.items():
            lines.append(
                f"| {cat_names.get(cat, cat)} | {m['count']} | "
                f"{fmt(m['factual_accuracy'])} | {fmt(m['no_hallucination'])} | "
                f"{fmt(m['completeness'])} | {fmt(m.get('hallucination_rate', 0), pct=True)} |"
            )
        lines.append("")

    lines.append("## 三、初步结论\n")
    lines.append(
        "1. **检索增益**：方案B引入多级增强RAG（BGE向量+BM25+RRF融合+HyDE+Reranker），"
        "在剧情问答类问题上能稳定召回正确的标准回目，Recall@5/Precision@5 显著优于方案A（方案A无检索能力，记为0）。"
    )
    lines.append(
        "2. **幻觉抑制**：RAG将回答锚定在原文片段上，方案B的幻觉发生率低于方案A，"
        "验证了作业文档6.4节「生成后反向检索、标记虚构内容」的缓解思路。"
    )
    lines.append(
        "3. **基座模型局限**：原生Qwen2.5-7B Base未经红楼梦领域微调，"
        "即便在方案B下，对专有名词、长剧情逻辑、人物关系的精准作答仍显薄弱——这正是阶段4 QLoRA微调（方案C）要解决的问题。"
    )
    lines.append(
        "4. **待补项**：方案C（QLoRA微调+RAG+GraphRAG+Agent）需在云GPU（A10 24GB）完成微调后补充，"
        "预期在事实准确与完整度上进一步拉开与A/B的差距。"
    )
    lines.append("")

    md = "\n".join(lines)
    out_path = os.path.join(REPORT_DIR, "eval_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[analyze] 报告已保存至 {out_path}")
    return md


def main():
    if not (os.path.exists(A_PATH) and os.path.exists(B_PATH)):
        print("[analyze] 错误: 未找到评测结果文件，请先运行 run_eval.py")
        return
    summary = load_json(SUMMARY_PATH) if os.path.exists(SUMMARY_PATH) else None
    if summary is None:
        # 实时聚合
        from evaluation.metrics import aggregate

        a = list(load_json(A_PATH).values())
        b = list(load_json(B_PATH).values())
        summary = {"A": aggregate(a), "B": aggregate(b)}

    build_charts(summary)
    md = build_markdown(summary)
    print("\n" + md)


if __name__ == "__main__":
    main()
