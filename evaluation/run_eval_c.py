"""
阶段9 评测脚本 —— 方案 C（微调模型 + RAG）

通过 SSH 隧道连接 AutoDL 上的 LLaMA-Factory API（127.0.0.1:8000），
调用 Qwen2.5-7B-Instruct + LoRA 微调模型，
结合本地 RAG 检索上下文，对 50 题测试集进行评测。

变量对照：
  方案 A：原生 Qwen2.5-7B Base（无 RAG）
  方案 B：原生 Qwen2.5-7B Base + RAG
  方案 C：微调 Qwen2.5-7B-Instruct + RAG  ← 本脚本

唯一变量：模型是否微调（A vs B 无微调，C 有微调），RAG 检索完全一致。
"""
import os
import sys

# 强制所有 PyTorch 模型走 CPU，避免 CrossEncoder/Reranker 在 GPU 上段错误
# Ollama 作为独立进程不受影响，仍用 GPU 做 judge
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent_graph.model_client import ModelClient
from evaluation.schemes import SYSTEM_PROMPT, B_PROMPT
from rag_pipeline.rag_pipeline import HonglouRAG
from evaluation.judge import judge_one
from evaluation.metrics import aggregate

TEST_SET_PATH = os.path.join(PROJECT_ROOT, "data", "test_set", "honglou_test_50.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation")
C_PATH = os.path.join(OUT_DIR, "results_schemeC.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "eval_summary.json")

API_BASE = "http://127.0.0.1:8000/v1"


def call_finetuned_api(prompt: str, system_prompt: str, temperature=0.3, max_tokens=1024) -> str:
    """通过 LLaMA-Factory API 调用微调模型生成答案。"""
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def run_scheme_c(question: str, rag: HonglouRAG, top_k: int = 5) -> dict:
    """方案 C：微调模型 + RAG 检索上下文。"""
    # 1) RAG 检索（本地，不带 Reranker 避免 CrossEncoder 段错误）
    t0 = time.time()
    rag_result = rag.retrieve_with_context(question, top_k=top_k)
    retrieval_time = rag_result.get("elapsed_time", time.time() - t0)

    retrieved_chapters = []
    for d in rag_result.get("source_details", [])[:top_k]:
        ch = d.get("chapter_num")
        if ch is not None and ch not in retrieved_chapters:
            retrieved_chapters.append(ch)

    context = rag_result.get("context", "")
    sources = rag_result.get("sources", "")

    # 2) 通过 API 调用微调模型生成
    t1 = time.time()
    prompt = B_PROMPT.format(context=context, question=question)
    answer = call_finetuned_api(prompt, SYSTEM_PROMPT)
    gen_time = time.time() - t1

    return {
        "answer": answer,
        "retrieved_chapters": retrieved_chapters,
        "context": context,
        "sources": sources,
        "gen_time": gen_time,
        "retrieval_time": retrieval_time,
    }


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
    print(f"[eval-C] 加载测试集: {len(questions)} 题")

    # 检查 API 连通性
    try:
        req = urllib.request.Request(f"{API_BASE}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            print(f"[eval-C] API 连通: {models['data'][0]['id']}")
    except Exception as e:
        print(f"[eval-C] 错误: LLaMA-Factory API 不可用 ({e})")
        print("[eval-C] 请确保 SSH 隧道已建立: ssh -L 8000:127.0.0.1:8000 ...")
        return

    # 检查 Ollama（judge 需要）
    client = ModelClient(model_name="qwen2.5:7b")
    if not client.is_available:
        print("[eval-C] 警告: Ollama 不可用，judge 评判将跳过")
        judge_available = False
    else:
        print("[eval-C] Ollama 就绪，judge 可用")
        judge_available = True

    # 初始化 RAG 管道（禁用 Reranker 和 HyDE，避免 CrossEncoder 段错误）
    print("[eval-C] 初始化 RAG 管道 (BGE+BM25+RRF, 无 Reranker)...")
    rag = HonglouRAG(use_hyde=False, use_reranker=False)
    print("[eval-C] RAG 管道就绪")

    prog_c = load_progress(C_PATH)
    total = len(questions)
    start = time.time()
    done_count = 0

    for qi, q in enumerate(questions, 1):
        qid = str(q["id"])
        cat = q["category"]
        golden = q["golden_chapters"]
        ref = q["reference"]
        question = q["question"]
        print(f"\n===== [{qi}/{total}] Q{qid} ({cat}) =====")
        print(f"  问题: {question[:80]}")

        if qid in prog_c:
            print(f"  [C] 已完成，跳过")
            done_count += 1
            continue

        print(f"  [C] 检索+生成中...")
        try:
            rc = run_scheme_c(question, rag)
            print(f"  [C] 检索回目: {rc['retrieved_chapters']}")
            print(f"  [C] 生成耗时: {rc['gen_time']:.1f}s")
            print(f"  [C] 答案: {rc['answer'][:120]}...")

            if judge_available:
                jc = judge_one(question, rc["answer"], ref, client)
                print(f"  [C] 评分 fac={jc['factual_accuracy']} noh={jc['no_hallucination']} com={jc['completeness']}")
            else:
                jc = {
                    "factual_accuracy": 0,
                    "no_hallucination": 0,
                    "completeness": 0,
                    "reasoning": "Ollama unavailable",
                }

            prog_c[qid] = {
                "id": q["id"],
                "category": cat,
                "question": question,
                "golden_chapters": golden,
                "reference": ref,
                "answer": rc["answer"],
                "retrieved_chapters": rc["retrieved_chapters"],
                "sources": rc["sources"],
                "context": rc["context"][:1500],
                "gen_time": round(rc["gen_time"], 2),
                "retrieval_time": round(rc["retrieval_time"], 2),
                "judge": jc,
            }
            save_progress(C_PATH, prog_c)
            done_count += 1
        except Exception as e:
            print(f"  [C] 错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(3)

    # 聚合
    list_c = list(prog_c.values())

    # 更新 summary
    summary = load_progress(SUMMARY_PATH)
    summary["C"] = aggregate(list_c)
    summary["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["test_set_size"] = total
    summary["note"] = "方案C: Qwen2.5-7B-Instruct + LoRA微调 + RAG(BGE+BM25+RRF, 无Reranker). 通过SSH隧道连接AutoDL API."
    save_progress(SUMMARY_PATH, summary)

    elapsed = time.time() - start
    print(f"\n[eval-C] 完成！耗时 {elapsed/60:.1f} 分钟")
    print(f"[eval-C] 方案C: {len(list_c)}/{total} 题")
    print(f"[eval-C] 汇总已更新: {SUMMARY_PATH}")

    # 打印聚合指标
    if list_c:
        agg = aggregate(list_c)
        print(f"\n[eval-C] 聚合指标:")
        for k, v in agg.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    run()
