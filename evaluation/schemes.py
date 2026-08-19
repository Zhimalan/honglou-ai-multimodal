"""
评测方案 A / B 推理模块

方案 A：原生 Qwen2.5-7B Base（无 RAG、无微调）
方案 B：原生 Qwen2.5-7B Base + 多级增强 RAG

两个方案共用同一个基座模型与 Prompt，唯一变量是「是否注入检索上下文」，
符合作业文档 6.4 节「三组对照实验必须固定全部变量」的要求。
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent_graph.model_client import ModelClient
from rag_pipeline.rag_api import rag_search

# 系统提示词（A/B 完全一致，保证对照有效）
SYSTEM_PROMPT = (
    "你是一位研究《红楼梦》的学者型问答助手。"
    "请用准确、简洁的中文回答用户关于《红楼梦》的问题，"
    "不要编造原著中不存在的情节、人物或关系。"
)

A_PROMPT = """【问题】
{question}

请直接作答。"""

B_PROMPT = """以下是《红楼梦》原文的相关片段（按相关性排序，可能包含多个回目），请只使用与问题直接相关的片段作答。

【参考资料】
{context}

【问题】
{question}

要求：
1. 严格依据资料中与问题「最相关」的段落作答，不要引用无关的回目内容；
2. 若资料能回答，请明确指出对应的回目（如"第3回"）；
3. 若资料不足以回答，请如实说明，不要编造。"""


def run_scheme_a(question: str, client: ModelClient) -> dict:
    """方案 A：纯基座模型，无检索上下文。

    Returns:
        {
            "answer": str,
            "retrieved_chapters": [],
            "context": "",
            "sources": "",
            "gen_time": float,
            "retrieval_time": 0.0,
        }
    """
    t0 = time.time()
    answer = client.generate(
        prompt=A_PROMPT.format(question=question),
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=1024,
    )
    gen_time = time.time() - t0
    return {
        "answer": answer,
        "retrieved_chapters": [],
        "context": "",
        "sources": "",
        "gen_time": gen_time,
        "retrieval_time": 0.0,
    }


def run_scheme_b(question: str, client: ModelClient, top_k: int = 5) -> dict:
    """方案 B：基座模型 + RAG 检索上下文。

    Returns:
        {
            "answer": str,
            "retrieved_chapters": [int,...],   # Reranker 精排后 top_k 的回目编号
            "context": str,
            "sources": str,
            "gen_time": float,
            "retrieval_time": float,
        }
    """
    # 1) 检索
    t0 = time.time()
    rag_result = rag_search(question, top_k=top_k)
    retrieval_time = rag_result.get("elapsed_time", time.time() - t0)

    retrieved_chapters = []
    for d in rag_result.get("source_details", [])[:top_k]:
        ch = d.get("chapter_num")
        if ch is not None and ch not in retrieved_chapters:
            retrieved_chapters.append(ch)

    context = rag_result.get("context", "")
    sources = rag_result.get("sources", "")

    # 2) 基于上下文生成
    t1 = time.time()
    answer = client.generate(
        prompt=B_PROMPT.format(context=context, question=question),
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=1024,
    )
    gen_time = time.time() - t1

    return {
        "answer": answer,
        "retrieved_chapters": retrieved_chapters,
        "context": context,
        "sources": sources,
        "gen_time": gen_time,
        "retrieval_time": retrieval_time,
    }


if __name__ == "__main__":
    c = ModelClient()
    q = "林黛玉是如何进入贾府的？"
    print("=== 方案 A ===")
    ra = run_scheme_a(q, c)
    print("A 答案:", ra["answer"][:200])
    print("=== 方案 B ===")
    rb = run_scheme_b(q, c)
    print("B 检索回目:", rb["retrieved_chapters"])
    print("B 答案:", rb["answer"][:200])
