"""
LLM-as-Judge 评测模块

对每条回答进行三项打分（1-5 分）：
  - factual_accuracy : 事实准确性（有无明显情节/人物/关系错误）
  - no_hallucination : 无幻觉性（是否编造原著中不存在的内容）
  - completeness     : 解读完整度（是否充分回答了问题）
"""
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent_graph.model_client import ModelClient

JUDGE_SYSTEM = (
    "你是一位严谨的《红楼梦》研究专家，负责评估 AI 助手的回答质量。"
    "你只依据给定的「参考答案要点」和《红楼梦》常识进行评判，不凭空给分。"
    "必须输出严格的 JSON，不要输出任何额外文字。"
)

JUDGE_PROMPT = """【问题】
{question}

【参考答案要点】
{reference}

【待评估回答】
{answer}

请基于参考答案要点与《红楼梦》常识，对回答进行三项 1-5 分评分（5 分为最佳）：
- factual_accuracy: 是否准确回答了问题要点，有无明显情节/人物/关系错误
- no_hallucination: 是否编造了原著中根本不存在的情节、人物或关系。注意：引用了真实原文但偏离问题，不属于幻觉，仅在 factual_accuracy / completeness 中扣分
- completeness: 是否充分、贴切地回答了问题

只输出 JSON，格式如下：
{{"factual_accuracy": <int>, "no_hallucination": <int>, "completeness": <int>, "comment": "<简短评语>"}}"""


def judge_one(question: str, answer: str, reference: str, client: ModelClient) -> dict:
    """对单条回答打分。失败时返回兜底分值。"""
    if not answer or answer.startswith("[") or len(answer) < 5:
        # 生成失败/空答案，直接判低分
        return {
            "factual_accuracy": 0,
            "no_hallucination": 0,
            "completeness": 0,
            "comment": "回答为空或生成失败",
        }
    resp = client.generate_json(
        prompt=JUDGE_PROMPT.format(
            question=question, answer=answer, reference=reference
        ),
        system_prompt=JUDGE_SYSTEM,
        temperature=0.0,
    )
    if "error" in resp:
        return {
            "factual_accuracy": 0,
            "no_hallucination": 0,
            "completeness": 0,
            "comment": f"评委解析失败: {resp['error']}",
        }
    # 规整分数到 0-5 整数
    def _clip(v):
        try:
            v = int(round(float(v)))
        except Exception:
            v = 0
        return max(0, min(5, v))

    return {
        "factual_accuracy": _clip(resp.get("factual_accuracy", 0)),
        "no_hallucination": _clip(resp.get("no_hallucination", 0)),
        "completeness": _clip(resp.get("completeness", 0)),
        "comment": str(resp.get("comment", ""))[:200],
    }


if __name__ == "__main__":
    c = ModelClient()
    r = judge_one(
        "林黛玉是如何进入贾府的？",
        "林黛玉母亲去世后，贾母派人接她进京，第3回进贾府。",
        "林黛玉母亲贾敏去世后，贾母派人接她进京；第3回黛玉抛父进京都。",
        c,
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))
