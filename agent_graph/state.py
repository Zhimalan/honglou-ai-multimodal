"""
Agent 全局状态定义
LangGraph 的 State 通过 TypedDict 在节点间传递。
"""

from typing import TypedDict, Optional
from enum import Enum


class IntentType(str, Enum):
    """意图类型枚举"""
    PLOT_QA = "plot_qa"                    # 剧情/诗词问答 → RAG
    CHARACTER_RELATION = "character_relation"  # 人物关系 → GraphRAG + 绘图
    DRAWING = "drawing"                    # 明确要求绘图 → GraphRAG + ComfyUI
    GENERAL = "general"                    # 通用闲聊 → 直接 LLM


class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态，在所有节点间共享传递。

    字段说明:
        question:              用户原始问题
        intent:                意图分类结果 (IntentType 值)
        detected_characters:   从问题中识别出的人物名列表

        rag_context:           RAG 检索到的上下文文本
        rag_sources:           RAG 来源引用 (如 "第三回、第五回")
        rag_elapsed:           RAG 检索耗时

        graph_data:            GraphRAG 检索到的结构化数据 (dict)
        graph_summary:         GraphRAG 数据的自然语言摘要
        graph_elapsed:         GraphRAG 检索耗时

        draw_result:           ComfyUI 绘图结果 {success, image_path, prompt_path, ...}
        draw_elapsed:          绘图耗时

        answer:                LLM 生成的最终回答
        answer_elapsed:        答案生成耗时

        hallucination_flags:   被标记为潜在幻觉的句子列表
        hallucination_score:   幻觉评分 (0.0=无幻觉, 1.0=全幻觉)
        hallucination_checked: 是否已执行幻觉校验

        conversation_history:  对话历史 [{"role": "user"/"assistant", "content": "..."}]
        elapsed_time:          各阶段耗时统计
        error:                 错误信息 (正常流程为空)
    """
    # 输入
    question: str
    intent: str
    detected_characters: list[str]

    # RAG 检索结果
    rag_context: str
    rag_sources: str
    rag_elapsed: float

    # GraphRAG 检索结果
    graph_data: dict
    graph_summary: str
    graph_elapsed: float

    # ComfyUI 绘图结果
    draw_result: dict
    draw_elapsed: float

    # 答案生成
    answer: str
    answer_elapsed: float

    # 幻觉校验
    hallucination_flags: list[str]
    hallucination_score: float
    hallucination_checked: bool

    # 对话与元信息
    conversation_history: list[dict]
    elapsed_time: dict
    error: str
