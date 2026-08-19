"""
LangGraph Agent 节点函数集合
每个函数接收 AgentState，返回需要更新的 state 字段。
"""

import os
import sys
import time
import json
import re
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent_graph.state import AgentState, IntentType
from agent_graph.model_client import get_model_client
from agent_graph import prompts


# ═══════════════════════════════════════════════
# 节点 1: 意图识别
# ═══════════════════════════════════════════════

# 红楼梦核心人物名（用于规则兜底）
CORE_CHARACTERS = [
    "贾宝玉", "林黛玉", "薛宝钗", "王熙凤", "贾母", "史太君", "贾政", "王夫人",
    "贾琏", "史湘云", "秦可卿", "贾元春", "贾迎春", "贾探春", "贾惜春",
    "袭人", "晴雯", "平儿", "鸳鸯", "紫鹃", "薛蟠", "薛姨妈", "贾兰",
    "贾赦", "邢夫人", "贾珍", "尤氏", "贾蓉", "妙玉", "刘姥姥",
    "贾雨村", "甄士隐", "焦大", "香菱", "巧姐", "李纨", "赵姨娘",
    "凤姐", "黛玉", "宝钗", "宝玉", "湘云", "探春", "迎春", "惜春",
    "老太太", "宝二爷", "林妹妹", "宝姐姐", "琏二奶奶", "颦儿",
]

# 绘图关键词
DRAWING_KEYWORDS = [
    "画", "绘图", "可视化", "生成图", "关系图", "网络图",
    "画一张", "画个", "生成一张", "出图", "画图",
]

# 人物关系关键词
RELATION_KEYWORDS = [
    "关系", "什么关系", "亲戚", "亲属", "家族", "血脉", "辈分",
    "婆婆", "公公", "父亲", "母亲", "妻子", "丈夫", "儿子", "女儿",
    "哥哥", "姐姐", "弟弟", "妹妹", "叔伯", "婶娘", "表亲",
    "丫鬟", "仆人", "主子", "主仆", "夫妻", "父子", "母女",
    "是谁的", "和谁", "之间", "脉络", "族谱",
]


def _rule_based_intent(question: str) -> tuple[str, list[str]]:
    """规则兜底的意图分类（当 LLM 不可用或超时时使用）"""
    # 检测绘图关键词
    for kw in DRAWING_KEYWORDS:
        if kw in question:
            chars = [c for c in CORE_CHARACTERS if c in question]
            return IntentType.DRAWING.value, chars

    # 检测人物名
    found_chars = [c for c in CORE_CHARACTERS if c in question]

    # 检测关系关键词
    has_relation_kw = any(kw in question for kw in RELATION_KEYWORDS)

    if found_chars and has_relation_kw:
        return IntentType.CHARACTER_RELATION.value, found_chars

    # 默认走 RAG
    return IntentType.PLOT_QA.value, found_chars


def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    意图识别节点：使用 LLM 分类用户问题意图。
    当 LLM 不可用时回退到规则匹配。
    """
    question = state["question"]
    print(f"\n{'='*60}")
    print(f"[意图识别] 问题: {question}")

    client = get_model_client()
    detected_chars = []

    if client.is_available:
        prompt = prompts.INTENT_USER.format(question=question)
        result = client.generate_json(
            prompt=prompt,
            system_prompt=prompts.INTENT_SYSTEM,
            temperature=0.1,
            timeout=30,
        )

        if "error" not in result:
            intent = result.get("intent", "plot_qa")
            detected_chars = result.get("characters", [])
            reason = result.get("reason", "")

            # 验证意图合法性
            valid_intents = [e.value for e in IntentType]
            if intent not in valid_intents:
                intent = IntentType.PLOT_QA.value

            print(f"[意图识别] LLM分类: {intent} (人物: {detected_chars})")
            print(f"[意图识别] 原因: {reason}")

            return {
                "intent": intent,
                "detected_characters": detected_chars,
            }

    # 规则兜底
    intent, chars = _rule_based_intent(question)
    detected_chars = chars
    print(f"[意图识别] 规则分类: {intent} (人物: {detected_chars})")

    return {
        "intent": intent,
        "detected_characters": detected_chars,
    }


# ═══════════════════════════════════════════════
# 节点 2: RAG 检索工具
# ═══════════════════════════════════════════════

def rag_tool_node(state: AgentState) -> Dict[str, Any]:
    """RAG 检索工具节点：检索红楼梦原文片段"""
    question = state["question"]
    print(f"\n[RAG工具] 检索: {question}")

    try:
        from rag_pipeline.rag_api import rag_search
        result = rag_search(question, top_k=5, max_context=3000)

        print(f"[RAG工具] 来源: {result.get('sources', '无')}")
        print(f"[RAG工具] 耗时: {result.get('elapsed_time', 0):.2f}s")
        print(f"[RAG工具] HyDE: {'是' if result.get('hyde_used') else '否'}")

        return {
            "rag_context": result.get("context", ""),
            "rag_sources": result.get("sources", ""),
            "rag_elapsed": result.get("elapsed_time", 0),
        }
    except Exception as e:
        print(f"[RAG工具] 错误: {e}")
        return {
            "rag_context": "",
            "rag_sources": "",
            "rag_elapsed": 0,
            "error": f"RAG检索失败: {e}",
        }


# ═══════════════════════════════════════════════
# 节点 3: GraphRAG 检索工具
# ═══════════════════════════════════════════════

def graphrag_tool_node(state: AgentState) -> Dict[str, Any]:
    """GraphRAG 检索工具节点：查询人物关系知识图谱"""
    question = state["question"]
    detected_chars = state.get("detected_characters", [])
    print(f"\n[GraphRAG工具] 查询: {question}")
    print(f"[GraphRAG工具] 识别人物: {detected_chars}")

    try:
        from graphrag_index.graph_rag import GraphRAG
        graph_rag = GraphRAG()

        graph_data = {}
        summary_parts = []

        if len(detected_chars) == 0:
            # 未识别到人物，尝试自然语言查询
            result = graph_rag.query(question)
            graph_data = result
            if result.get("found"):
                if "characters" in result:
                    # 两个人物的关系路径
                    chars = result["characters"]
                    summary_parts.append(f"识别到人物: {'、'.join(chars)}")
                    path = result.get("path", {})
                    if path.get("found") and path.get("shortest_path"):
                        path_str = " → ".join(path["shortest_path"])
                        summary_parts.append(f"关系路径: {path_str}")
                        summary_parts.append(f"最短距离: {path.get('shortest_distance', 0)} 跳")
                else:
                    # 单个人物的关系概要
                    summary_parts.append(f"人物: {result.get('name', '')}")
                    relations = result.get("relations", {})
                    for rtype, rels in relations.items():
                        if rels:
                            names = "、".join(r["name"] for r in rels[:5])
                            summary_parts.append(f"{rtype}: {names}")
            else:
                summary_parts.append("未在知识图谱中识别到相关人物")

        elif len(detected_chars) == 1:
            # 单个人物：查询关系概要
            char = detected_chars[0]
            entity = graph_rag.search_entity(char)
            neighbors = graph_rag.get_neighbors(char, hop=1, top_k=15)
            relation_summary = graph_rag.get_relation_summary(char)
            subgraph = graph_rag.get_character_subgraph(char, hop=1)

            graph_data = {
                "entity": entity,
                "neighbors": neighbors,
                "relation_summary": relation_summary,
                "subgraph": subgraph,
            }

            if entity.get("found"):
                summary_parts.append(f"人物: {entity['name']}")
                summary_parts.append(f"描述: {entity.get('description', '无')}")
                summary_parts.append(f"度中心性: {entity.get('degree', 0)}")

            if relation_summary.get("found"):
                relations = relation_summary.get("relations", {})
                for rtype, rels in relations.items():
                    if rels:
                        names = "、".join(r["name"] for r in rels[:5])
                        summary_parts.append(f"{rtype}: {names}")

            if subgraph.get("found"):
                summary_parts.append(f"子图: {subgraph.get('node_count', 0)} 节点, {subgraph.get('edge_count', 0)} 边")

        else:
            # 多个人物：查询关系路径
            char1, char2 = detected_chars[0], detected_chars[1]
            path_result = graph_rag.find_relation_path(char1, char2)
            summary1 = graph_rag.get_relation_summary(char1)
            summary2 = graph_rag.get_relation_summary(char2)

            graph_data = {
                "path": path_result,
                "summary_1": summary1,
                "summary_2": summary2,
            }

            if path_result.get("found") and path_result.get("shortest_path"):
                path_str = " → ".join(path_result["shortest_path"])
                summary_parts.append(f"{char1} 与 {char2} 的关系路径: {path_str}")
                summary_parts.append(f"最短距离: {path_result.get('shortest_distance', 0)} 跳")

                # 路径详情
                for p in path_result.get("all_paths", [])[:2]:
                    for seg in p.get("segments", []):
                        summary_parts.append(
                            f"  {seg['from']} —[{seg['relation_type']}]→ {seg['to']}"
                        )
            else:
                summary_parts.append(f"未找到 {char1} 与 {char2} 之间的直接路径")

            # 补充各自的关系概要
            for name, summary in [(char1, summary1), (char2, summary2)]:
                if summary.get("found"):
                    relations = summary.get("relations", {})
                    for rtype, rels in relations.items():
                        if rels:
                            names = "、".join(r["name"] for r in rels[:3])
                            summary_parts.append(f"{name}的{rtype}: {names}")

        graph_summary = "\n".join(summary_parts)
        print(f"[GraphRAG工具] 摘要:\n{graph_summary[:300]}...")

        return {
            "graph_data": graph_data,
            "graph_summary": graph_summary,
        }
    except Exception as e:
        print(f"[GraphRAG工具] 错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            "graph_data": {},
            "graph_summary": f"GraphRAG 查询失败: {e}",
            "error": f"GraphRAG查询失败: {e}",
        }


# ═══════════════════════════════════════════════
# 节点 4: ComfyUI 绘图工具
# ═══════════════════════════════════════════════

def comfyui_tool_node(state: AgentState) -> Dict[str, Any]:
    """ComfyUI 绘图工具节点：生成人物关系可视化图"""
    question = state["question"]
    detected_chars = state.get("detected_characters", [])
    graph_data = state.get("graph_data", {})
    print(f"\n[绘图工具] 生成人物关系图...")

    try:
        from visualize.comfyui_api.comfyui_client import generate_character_relation_image

        # 确定中心人物
        if detected_chars:
            character = detected_chars[0]
        else:
            character = "贾宝玉"

        # 从 GraphRAG 数据中提取关系列表
        relations = []
        relation_summary = graph_data.get("relation_summary", {})
        if relation_summary.get("found"):
            rels = relation_summary.get("relations", {})
            for rtype, rel_list in rels.items():
                for r in rel_list[:8]:
                    relations.append({
                        "name": r["name"],
                        "relation_type": rtype,
                        "description": r.get("description", ""),
                    })

        # 也从 neighbors 中提取
        if not relations:
            neighbors = graph_data.get("neighbors", {})
            if neighbors.get("found"):
                for n in neighbors.get("neighbors", [])[:8]:
                    relations.append({
                        "name": n["name"],
                        "relation_type": n.get("relation_type", "其他"),
                        "description": n.get("description", ""),
                    })

        if not relations:
            relations = [
                {"name": "林黛玉", "relation_type": "其他", "description": ""},
                {"name": "薛宝钗", "relation_type": "其他", "description": ""},
            ]

        print(f"[绘图工具] 中心人物: {character}")
        print(f"[绘图工具] 关系数: {len(relations)}")

        result = generate_character_relation_image(character, relations)

        draw_status = "成功" if result.get("success") else "提示词已保存(ComfyUI未运行)"
        print(f"[绘图工具] 结果: {draw_status}")
        if result.get("image_path"):
            print(f"[绘图工具] 图片: {result['image_path']}")
        if result.get("prompt_path"):
            print(f"[绘图工具] 提示词: {result['prompt_path']}")

        return {
            "draw_result": result,
            "draw_elapsed": 0,
        }
    except Exception as e:
        print(f"[绘图工具] 错误: {e}")
        return {
            "draw_result": {"success": False, "message": str(e)},
            "draw_elapsed": 0,
        }


# ═══════════════════════════════════════════════
# 节点 5: 答案生成
# ═══════════════════════════════════════════════

def answer_generator_node(state: AgentState) -> Dict[str, Any]:
    """答案生成节点：根据意图和检索结果生成最终回答"""
    question = state["question"]
    intent = state.get("intent", "plot_qa")
    print(f"\n[答案生成] 意图: {intent}")

    client = get_model_client()

    if intent == IntentType.PLOT_QA.value:
        # RAG 问答
        context = state.get("rag_context", "")
        sources = state.get("rag_sources", "")

        if context:
            prompt = prompts.RAG_ANSWER_USER.format(
                question=question,
                context=context,
                sources=sources,
            )
            answer = client.generate(
                prompt=prompt,
                system_prompt=prompts.RAG_ANSWER_SYSTEM,
                temperature=0.7,
                max_tokens=2048,
            )
        else:
            answer = "抱歉，未能检索到相关原文内容。请尝试换一种方式提问。"

    elif intent in (IntentType.CHARACTER_RELATION.value, IntentType.DRAWING.value):
        # 人物关系问答
        graph_summary = state.get("graph_summary", "无数据")
        draw_result = state.get("draw_result", {})

        if intent == IntentType.DRAWING.value or draw_result:
            # 绘图请求：说明绘图结果
            draw_status = "图片已生成" if draw_result.get("success") else \
                f"提示词已保存: {draw_result.get('prompt_path', '')}"

            prompt = prompts.DRAW_ANSWER_USER.format(
                question=question,
                graph_summary=graph_summary,
                draw_status=draw_status,
            )
            answer = client.generate(
                prompt=prompt,
                system_prompt=prompts.DRAW_ANSWER_SYSTEM,
                temperature=0.7,
                max_tokens=1024,
            )
        else:
            # 纯人物关系问答
            prompt = prompts.GRAPH_ANSWER_USER.format(
                question=question,
                graph_summary=graph_summary,
            )
            answer = client.generate(
                prompt=prompt,
                system_prompt=prompts.GRAPH_ANSWER_SYSTEM,
                temperature=0.7,
                max_tokens=1024,
            )
    else:
        # 通用问答
        prompt = prompts.GENERAL_ANSWER_USER.format(question=question)
        answer = client.generate(
            prompt=prompt,
            system_prompt=prompts.GENERAL_ANSWER_SYSTEM,
            temperature=0.7,
            max_tokens=1024,
        )

    print(f"[答案生成] 回答长度: {len(answer)} 字")

    return {"answer": answer}


# ═══════════════════════════════════════════════
# 节点 6: 幻觉校验
# ═══════════════════════════════════════════════

def hallucination_checker_node(state: AgentState) -> Dict[str, Any]:
    """
    幻觉校验节点：生成答案后反向检索知识库，检测虚构内容。

    策略:
    1. 对 RAG 类回答：检查答案中的关键声明是否在检索上下文中有依据
    2. 对 GraphRAG 类回答：检查关系描述是否与图谱数据一致
    3. 使用 LLM 做二次校验
    """
    question = state["question"]
    answer = state.get("answer", "")
    intent = state.get("intent", "plot_qa")
    print(f"\n[幻觉校验] 开始校验...")

    if not answer or answer.startswith("[") or answer.startswith("抱歉"):
        # 答案本身就是错误信息，跳过校验
        print(f"[幻觉校验] 答案为错误信息，跳过校验")
        return {
            "hallucination_flags": [],
            "hallucination_score": 0.0,
            "hallucination_checked": True,
        }

    # 获取校验依据
    context = state.get("rag_context", "")
    graph_summary = state.get("graph_summary", "")

    # 如果没有依据，使用 RAG 反向检索
    if not context and not graph_summary:
        try:
            from rag_pipeline.rag_api import rag_search_fast
            result = rag_search_fast(question, top_k=3)
            context = result.get("context", "")
        except Exception:
            pass

    check_context = context or graph_summary or "无可用依据"

    client = get_model_client()
    flags = []
    score = 0.0

    if client.is_available and len(answer) > 20:
        prompt = prompts.HALLUCINATION_CHECK_USER.format(
            question=question,
            answer=answer[:2000],  # 限制长度避免超时
            context=check_context[:2000],
        )

        result = client.generate_json(
            prompt=prompt,
            system_prompt=prompts.HALLUCINATION_CHECK_SYSTEM,
            temperature=0.1,
            timeout=30,
        )

        if "error" not in result:
            has_halluc = result.get("has_hallucination", False)
            flags = result.get("flagged_sentences", [])
            score = float(result.get("confidence", 0.0)) if has_halluc else 0.0
            note = result.get("note", "")

            print(f"[幻觉校验] 有幻觉: {has_halluc}")
            print(f"[幻觉校验] 幻觉评分: {score:.2f}")
            if flags:
                print(f"[幻觉校验] 标记句子: {len(flags)} 条")
                for f in flags[:3]:
                    print(f"  - {f[:80]}")
            if note:
                print(f"[幻觉校验] 说明: {note}")
        else:
            print(f"[幻觉校验] LLM校验失败: {result.get('error', '')}")
    else:
        # 简单规则校验：检查答案中是否包含常见幻觉模式
        hallucination_patterns = [
            r"第\d+回.*?记载",  # 编造回目
            r"原文.*?写道.*?[「」]",  # 编造原文引用
        ]
        for pattern in hallucination_patterns:
            matches = re.findall(pattern, answer)
            flags.extend(matches)
        score = len(flags) / max(len(answer.split("。")), 1) if flags else 0.0
        print(f"[幻觉校验] 规则校验: 标记 {len(flags)} 条")

    return {
        "hallucination_flags": flags,
        "hallucination_score": round(score, 3),
        "hallucination_checked": True,
    }


# ═══════════════════════════════════════════════
# 路由函数
# ═══════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """根据意图分类结果路由到不同工具节点"""
    intent = state.get("intent", "plot_qa")

    if intent == IntentType.PLOT_QA.value:
        return "rag_tool"
    elif intent in (IntentType.CHARACTER_RELATION.value, IntentType.DRAWING.value):
        return "graphrag_tool"
    else:
        return "answer_generator"
