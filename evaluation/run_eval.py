"""
阶段8 评测主脚本 —— 方案 A / B 对照

固定 50 题测试集，对每题分别运行：
  - 方案 A：原生 Qwen2.5-7B Base（无 RAG）
  - 方案 B：原生 Qwen2.5-7B Base + 多级增强 RAG
并调用 LLM-as-Judge 进行三项打分，计算检索/生成/专项指标。

支持断点续跑：每次完成一题立即写入 JSON，中断后重跑会自动跳过已完成题目。

输出：
  evaluation/results_schemeA.json
  evaluation/results_schemeB.json
  evaluation/eval_summary.json   （聚合指标，供 analyze_results.py 使用）
"""
import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent_graph.model_client import ModelClient
from evaluation.schemes import run_scheme_a, run_scheme_b
from evaluation.judge import judge_one
from evaluation.metrics import aggregate

TEST_SET_PATH = os.path.join(PROJECT_ROOT, "data", "test_set", "honglou_test_50.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation")
A_PATH = os.path.join(OUT_DIR, "results_schemeA.json")
B_PATH = os.path.join(OUT_DIR, "results_schemeB.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "eval_summary.json")

# 只跑 A、B 两套（C 待阶段4微调完成再补）
SCHEMES = ["A", "B"]


def load_test_set():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def load_progress(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run():
    questions = load_test_set()
    print(f"[eval] 加载测试集: {len(questions)} 题")

    client = ModelClient(model_name="qwen2.5:7b")
    if not client.is_available:
        print("[eval] 错误: Ollama 或 qwen2.5:7b 不可用，请先启动 Ollama 服务。")
        return

    # 加载已有进度
    prog_a = load_progress(A_PATH)
    prog_b = load_progress(B_PATH)

    total = len(questions)
    start = time.time()

    for qi, q in enumerate(questions, 1):
        qid = str(q["id"])
        cat = q["category"]
        golden = q["golden_chapters"]
        ref = q["reference"]
        question = q["question"]
        print(f"\n===== [{qi}/{total}] Q{qid} ({cat}) {question} =====")

        # ---- 方案 A ----
        if qid in prog_a:
            print(f"  [A] 已完成，跳过")
        else:
            print(f"  [A] 生成中...")
            ra = run_scheme_a(question, client)
            ja = judge_one(question, ra["answer"], ref, client)
            print(f"  [A] 评分 fac={ja['factual_accuracy']} noh={ja['no_hallucination']} com={ja['completeness']}")
            prog_a[qid] = {
                "id": q["id"],
                "category": cat,
                "question": question,
                "golden_chapters": golden,
                "reference": ref,
                "answer": ra["answer"],
                "retrieved_chapters": ra["retrieved_chapters"],
                "sources": ra["sources"],
                "gen_time": round(ra["gen_time"], 2),
                "retrieval_time": ra["retrieval_time"],
                "judge": ja,
            }
            save_progress(A_PATH, prog_a)

        # ---- 方案 B ----
        if qid in prog_b:
            print(f"  [B] 已完成，跳过")
        else:
            print("  [B] 检索+生成中...")
            rb = run_scheme_b(question, client)
            jb = judge_one(question, rb["answer"], ref, client)
            print(f"  [B] 检索回目 {rb['retrieved_chapters']} | 评分 fac={jb['factual_accuracy']} noh={jb['no_hallucination']} com={jb['completeness']}")
            prog_b[qid] = {
                "id": q["id"],
                "category": cat,
                "question": question,
                "golden_chapters": golden,
                "reference": ref,
                "answer": rb["answer"],
                "retrieved_chapters": rb["retrieved_chapters"],
                "sources": rb["sources"],
                "context": rb["context"][:1500],
                "gen_time": round(rb["gen_time"], 2),
                "retrieval_time": round(rb["retrieval_time"], 2),
                "judge": jb,
            }
            save_progress(B_PATH, prog_b)

    # ---- 聚合 ----
    list_a = list(prog_a.values())
    list_b = list(prog_b.values())
    summary = {
        "A": aggregate(list_a),
        "B": aggregate(list_b),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_size": total,
        "note": "方案C（微调+RAG+GraphRAG+Agent）待阶段4完成后补充。",
    }
    save_progress(SUMMARY_PATH, summary)

    elapsed = time.time() - start
    print(f"\n[eval] 完成！耗时 {elapsed/60:.1f} 分钟")
    print(f"[eval] 方案A: {len(list_a)}/{total} 题, 方案B: {len(list_b)}/{total} 题")
    print(f"[eval] 汇总已写入: {SUMMARY_PATH}")


if __name__ == "__main__":
    run()
