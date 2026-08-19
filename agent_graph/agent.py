"""
LangGraph 多工具 Agent 主入口
组装状态图：意图分流 → 工具检索 → 答案生成 → 幻觉校验

图结构:
    START → intent_classifier → (条件路由)
        ├─ plot_qa → rag_tool → answer_generator → hallucination_checker → END
        ├─ character_relation → graphrag_tool → comfyui_tool → answer_generator → hallucination_checker → END
        └─ drawing → graphrag_tool → comfyui_tool → answer_generator → hallucination_checker → END
        └─ general → answer_generator → hallucination_checker → END

使用方式:
    from agent_graph.agent import HonglouAgent
    agent = HonglouAgent()
    result = agent.ask("贾宝玉和林黛玉是什么关系？")
    print(result["answer"])
"""

import os
import sys
import time
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from langgraph.graph import StateGraph, END, START

from agent_graph.state import AgentState
from agent_graph.nodes import (
    intent_classifier_node,
    rag_tool_node,
    graphrag_tool_node,
    comfyui_tool_node,
    answer_generator_node,
    hallucination_checker_node,
    route_by_intent,
)


class HonglouAgent:
    """红楼梦多模态智能解读 LangGraph Agent"""

    def __init__(self, compile_graph: bool = True):
        """
        初始化 Agent，构建 LangGraph 状态图。

        Args:
            compile_graph: 是否立即编译图（首次初始化可能需要加载模型，设为 False 可延迟）
        """
        self.graph = None
        self._elapsed = {}

        if compile_graph:
            self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 状态图"""
        print("[Agent] 构建 LangGraph 状态图...")

        workflow = StateGraph(AgentState)

        # ── 添加节点 ──
        workflow.add_node("intent_classifier", intent_classifier_node)
        workflow.add_node("rag_tool", rag_tool_node)
        workflow.add_node("graphrag_tool", graphrag_tool_node)
        workflow.add_node("comfyui_tool", comfyui_tool_node)
        workflow.add_node("answer_generator", answer_generator_node)
        workflow.add_node("hallucination_checker", hallucination_checker_node)

        # ── 设置入口 ──
        workflow.add_edge(START, "intent_classifier")

        # ── 条件路由：意图分流 ──
        workflow.add_conditional_edges(
            "intent_classifier",
            route_by_intent,
            {
                "rag_tool": "rag_tool",
                "graphrag_tool": "graphrag_tool",
                "answer_generator": "answer_generator",
            },
        )

        # ── RAG 路径 ──
        workflow.add_edge("rag_tool", "answer_generator")

        # ── GraphRAG 路径 → 绘图 → 答案生成 ──
        workflow.add_edge("graphrag_tool", "comfyui_tool")
        workflow.add_edge("comfyui_tool", "answer_generator")

        # ── 答案生成 → 幻觉校验 → END ──
        workflow.add_edge("answer_generator", "hallucination_checker")
        workflow.add_edge("hallucination_checker", END)

        # 编译
        self.graph = workflow.compile()
        print("[Agent] 状态图构建完成")

        # 打印图结构
        self._print_graph_structure()

    def _print_graph_structure(self):
        """打印图结构说明"""
        structure = """
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Agent 流程图                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  START                                                      │
│    │                                                        │
│    ▼                                                        │
│  intent_classifier (意图识别)                                │
│    │                                                        │
│    ├─ plot_qa ──────→ rag_tool ──────┐                      │
│    ├─ char_relation → graphrag_tool ─┤                      │
│    ├─ drawing ──────→ graphrag_tool ─┤                      │
│    └─ general ───────────────────────┤                      │
│                                      ▼                      │
│              (graphrag→comfyui_tool──→)                     │
│                              │                              │
│                              ▼                              │
│                    answer_generator (答案生成)                │
│                              │                              │
│                              ▼                              │
│                  hallucination_checker (幻觉校验)             │
│                              │                              │
│                              ▼                              │
│                             END                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        """
        print(structure)

    def ask(
        self,
        question: str,
        conversation_history: list = None,
    ) -> Dict[str, Any]:
        """
        执行一次完整的 Agent 问答流程。

        Args:
            question: 用户问题
            conversation_history: 对话历史 [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            完整的 AgentState 字典，包含 answer, intent, sources 等
        """
        if self.graph is None:
            self._build_graph()

        print(f"\n{'═'*60}")
        print(f"  红楼梦多模态智能解读专家系统")
        print(f"  用户问题: {question}")
        print(f"{'═'*60}")

        start_time = time.time()

        # 初始状态
        initial_state: AgentState = {
            "question": question,
            "intent": "",
            "detected_characters": [],
            "rag_context": "",
            "rag_sources": "",
            "rag_elapsed": 0,
            "graph_data": {},
            "graph_summary": "",
            "graph_elapsed": 0,
            "draw_result": {},
            "draw_elapsed": 0,
            "answer": "",
            "answer_elapsed": 0,
            "hallucination_flags": [],
            "hallucination_score": 0.0,
            "hallucination_checked": False,
            "conversation_history": conversation_history or [],
            "elapsed_time": {},
            "error": "",
        }

        # 运行图
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            print(f"\n[Agent] 运行错误: {e}")
            import traceback
            traceback.print_exc()
            final_state = initial_state.copy()
            final_state["answer"] = f"系统运行出错: {e}"
            final_state["error"] = str(e)

        total_time = time.time() - start_time

        # 记录耗时
        final_state["elapsed_time"] = {
            "total": round(total_time, 2),
            "rag": round(final_state.get("rag_elapsed", 0), 2),
            "graphrag": round(final_state.get("graph_elapsed", 0), 2),
            "draw": round(final_state.get("draw_elapsed", 0), 2),
        }

        # 打印结果摘要
        self._print_summary(final_state)

        return final_state

    def _print_summary(self, state: Dict[str, Any]):
        """打印结果摘要"""
        print(f"\n{'─'*60}")
        print(f"  Agent 执行结果摘要")
        print(f"{'─'*60}")
        print(f"  意图: {state.get('intent', '未知')}")
        print(f"  识别人物: {state.get('detected_characters', [])}")
        print(f"  RAG来源: {state.get('rag_sources', '无')}")
        print(f"  GraphRAG: {'有数据' if state.get('graph_data') else '无'}")
        print(f"  绘图: {'成功' if state.get('draw_result', {}).get('success') else '未生成/提示词模式'}")
        print(f"  幻觉校验: {'已检查' if state.get('hallucination_checked') else '未检查'}")
        print(f"  幻觉评分: {state.get('hallucination_score', 0):.2f}")
        if state.get("hallucination_flags"):
            print(f"  标记幻觉: {len(state['hallucination_flags'])} 条")
        elapsed = state.get("elapsed_time", {})
        print(f"  总耗时: {elapsed.get('total', 0):.2f}s")
        print(f"  回答长度: {len(state.get('answer', ''))} 字")
        print(f"{'─'*60}")


# ═══════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════

_agent_instance = None


def get_agent() -> HonglouAgent:
    """获取全局 Agent 单例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = HonglouAgent()
    return _agent_instance


def run_agent(question: str, conversation_history: list = None) -> Dict[str, Any]:
    """
    快捷入口：运行 Agent 问答。

    Args:
        question: 用户问题
        conversation_history: 对话历史

    Returns:
        完整结果字典
    """
    agent = get_agent()
    return agent.ask(question, conversation_history)


# ═══════════════════════════════════════════════
# 交互式命令行
# ═══════════════════════════════════════════════

def interactive_cli():
    """交互式命令行对话"""
    print("\n" + "═" * 60)
    print("  红楼梦多模态智能解读专家系统 - 交互式对话")
    print("  输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("═" * 60 + "\n")

    agent = HonglouAgent()
    history = []

    while True:
        try:
            question = input("\n📝 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "退出"):
            print("再见！")
            break

        result = agent.ask(question, history)

        # 更新对话历史
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.get("answer", "")})

        # 打印回答
        print(f"\n🤖 回答:\n{'─'*50}")
        print(result.get("answer", "（无回答）"))
        print(f"{'─'*50}")

        # 打印幻觉提示
        if result.get("hallucination_flags"):
            print(f"\n⚠️ 幻觉校验提示 (评分: {result['hallucination_score']:.2f}):")
            for flag in result["hallucination_flags"][:3]:
                print(f"  • {flag[:100]}")

        # 打印来源
        if result.get("rag_sources"):
            print(f"\n📚 来源: {result['rag_sources']}")

        # 打印绘图信息
        draw = result.get("draw_result", {})
        if draw.get("image_path"):
            print(f"🖼️ 图片已生成: {draw['image_path']}")
        elif draw.get("prompt_path"):
            print(f"🖼️ 提示词已保存: {draw['prompt_path']}")


if __name__ == "__main__":
    interactive_cli()
