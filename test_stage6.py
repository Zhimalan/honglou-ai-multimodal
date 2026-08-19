"""
阶段6 LangGraph Agent 端到端测试
测试意图识别、RAG检索、GraphRAG检索、ComfyUI绘图、答案生成、幻觉校验全流程。
"""

import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from agent_graph.agent import HonglouAgent
from agent_graph.state import AgentState, IntentType
from agent_graph.nodes import _rule_based_intent
from agent_graph.model_client import get_model_client


# ═══════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════

TEST_CASES = [
    {
        "name": "剧情问答-林黛玉进贾府",
        "question": "林黛玉初进贾府时见到了哪些人？",
        "expected_intent": "plot_qa",
    },
    {
        "name": "剧情问答-王熙凤协理宁国府",
        "question": "王熙凤是怎么协理宁国府的？",
        "expected_intent": "plot_qa",
    },
    {
        "name": "诗词赏析-葬花吟",
        "question": "林黛玉的葬花吟表达了什么情感？",
        "expected_intent": "plot_qa",
    },
    {
        "name": "人物关系-宝玉黛玉",
        "question": "贾宝玉和林黛玉是什么关系？",
        "expected_intent": "character_relation",
    },
    {
        "name": "人物关系-王熙凤婆婆",
        "question": "王熙凤的婆婆是谁？",
        "expected_intent": "character_relation",
    },
    {
        "name": "绘图请求-贾宝玉关系图",
        "question": "帮我画一张贾宝玉的人物关系图",
        "expected_intent": "drawing",
    },
    {
        "name": "绘图请求-王熙凤网络图",
        "question": "生成一张王熙凤的关系网络图",
        "expected_intent": "drawing",
    },
]


# ═══════════════════════════════════════════════
# 测试函数
# ═══════════════════════════════════════════════

def test_rule_based_intent():
    """测试规则兜底意图分类"""
    print("\n" + "=" * 60)
    print("测试1: 规则兜底意图分类")
    print("=" * 60)

    passed = 0
    total = len(TEST_CASES)

    for tc in TEST_CASES:
        intent, chars = _rule_based_intent(tc["question"])
        ok = intent == tc["expected_intent"]
        status = "✓" if ok else "✗"
        print(f"  {status} {tc['name']}: 预期={tc['expected_intent']}, 实际={intent}, 人物={chars}")
        if ok:
            passed += 1

    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_model_client():
    """测试模型客户端"""
    print("\n" + "=" * 60)
    print("测试2: 模型客户端")
    print("=" * 60)

    client = get_model_client()
    print(f"  模型: {client.model_name}")
    print(f"  类型: {client.model_type}")
    print(f"  可用: {client.is_available}")

    if client.is_available:
        # 测试生成
        response = client.generate(
            prompt="请用一句话介绍红楼梦。",
            system_prompt="你是红楼梦专家。",
            temperature=0.3,
            max_tokens=100,
        )
        ok = len(response) > 10 and not response.startswith("[")
        print(f"  生成测试: {'✓' if ok else '✗'}")
        print(f"  回复: {response[:80]}...")

        # 测试 JSON 生成
        json_result = client.generate_json(
            prompt='请输出JSON: {"book": "红楼梦", "author": "曹雪芹"}',
            temperature=0.1,
        )
        json_ok = "error" not in json_result
        print(f"  JSON测试: {'✓' if json_ok else '✗'}")
        if json_ok:
            print(f"  JSON: {json_result}")

        return ok and json_ok
    else:
        print("  ⚠ 模型不可用，跳过生成测试")
        return True


def test_agent_full():
    """测试 Agent 完整流程"""
    print("\n" + "=" * 60)
    print("测试3: Agent 完整流程")
    print("=" * 60)

    agent = HonglouAgent()

    passed = 0
    total = len(TEST_CASES)

    for tc in TEST_CASES:
        print(f"\n{'─'*50}")
        print(f"测试: {tc['name']}")
        print(f"问题: {tc['question']}")
        print(f"{'─'*50}")

        try:
            result = agent.ask(tc["question"])

            # 检查意图
            intent = result.get("intent", "")
            intent_ok = intent == tc["expected_intent"]
            print(f"\n  意图: {'✓' if intent_ok else '✗'} (预期={tc['expected_intent']}, 实际={intent})")

            # 检查回答
            answer = result.get("answer", "")
            answer_ok = len(answer) > 20 and not answer.startswith("[")
            print(f"  回答: {'✓' if answer_ok else '✗'} ({len(answer)} 字)")

            # 检查检索
            if intent == "plot_qa":
                rag_ok = bool(result.get("rag_context"))
                print(f"  RAG检索: {'✓' if rag_ok else '✗'} (来源: {result.get('rag_sources', '无')})")
            elif intent in ("character_relation", "drawing"):
                graph_ok = bool(result.get("graph_data"))
                print(f"  GraphRAG: {'✓' if graph_ok else '✗'}")
                draw = result.get("draw_result", {})
                draw_ok = bool(draw)
                print(f"  绘图: {'✓' if draw_ok else '✗'} ({'图片' if draw.get('success') else '提示词'})")

            # 检查幻觉校验
            hall_checked = result.get("hallucination_checked", False)
            print(f"  幻觉校验: {'✓' if hall_checked else '✗'} (评分: {result.get('hallucination_score', 0):.2f})")

            # 检查耗时
            elapsed = result.get("elapsed_time", {})
            print(f"  耗时: {elapsed.get('total', 0):.1f}s")

            if intent_ok and answer_ok:
                passed += 1
                print(f"\n  >>> 测试通过 ✓")
            else:
                print(f"\n  >>> 测试失败 ✗")

        except Exception as e:
            print(f"\n  >>> 测试异常: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Agent 完整流程测试结果: {passed}/{total} 通过")
    print(f"{'='*60}")
    return passed == total


def test_graph_structure():
    """测试图结构完整性"""
    print("\n" + "=" * 60)
    print("测试4: 图结构完整性")
    print("=" * 60)

    agent = HonglouAgent()

    checks = [
        ("图已编译", agent.graph is not None),
        ("State包含question", "question" in AgentState.__annotations__),
        ("State包含intent", "intent" in AgentState.__annotations__),
        ("State包含rag_context", "rag_context" in AgentState.__annotations__),
        ("State包含graph_data", "graph_data" in AgentState.__annotations__),
        ("State包含draw_result", "draw_result" in AgentState.__annotations__),
        ("State包含answer", "answer" in AgentState.__annotations__),
        ("State包含hallucination_flags", "hallucination_flags" in AgentState.__annotations__),
        ("State包含conversation_history", "conversation_history" in AgentState.__annotations__),
    ]

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")

    # 检查节点文件
    from agent_graph import nodes
    node_checks = [
        ("intent_classifier_node", hasattr(nodes, "intent_classifier_node")),
        ("rag_tool_node", hasattr(nodes, "rag_tool_node")),
        ("graphrag_tool_node", hasattr(nodes, "graphrag_tool_node")),
        ("comfyui_tool_node", hasattr(nodes, "comfyui_tool_node")),
        ("answer_generator_node", hasattr(nodes, "answer_generator_node")),
        ("hallucination_checker_node", hasattr(nodes, "hallucination_checker_node")),
        ("route_by_intent", hasattr(nodes, "route_by_intent")),
    ]

    for name, ok in node_checks:
        print(f"  {'✓' if ok else '✗'} 节点函数: {name}")
        if ok:
            passed += 1

    total = len(checks) + len(node_checks)
    print(f"\n  结果: {passed}/{total} 通过")
    return passed == total


def test_model_switch():
    """测试模型热插拔"""
    print("\n" + "=" * 60)
    print("测试5: 模型热插拔")
    print("=" * 60)

    from agent_graph.model_client import ModelClient

    client = ModelClient(model_name="qwen2.5:7b", model_type="ollama_base")
    print(f"  初始模型: {client.model_name} ({client.model_type})")

    # 模拟切换到微调模型
    client.set_model("qwen2.5:7b-lora", model_type="ollama_finetuned")
    switch_ok = client.model_name == "qwen2.5:7b-lora" and client.model_type == "ollama_finetuned"
    print(f"  切换后: {client.model_name} ({client.model_type})")
    print(f"  热插拔: {'✓' if switch_ok else '✗'}")

    # 切换回基础模型
    client.set_model("qwen2.5:7b", "ollama_base")
    back_ok = client.model_name == "qwen2.5:7b"
    print(f"  切回: {client.model_name}")
    print(f"  恢复: {'✓' if back_ok else '✗'}")

    return switch_ok and back_ok


# ═══════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " 阶段6 LangGraph Agent 端到端测试".center(46) + "║")
    print("╚" + "═" * 58 + "╝")

    results = {}

    # 测试1: 规则兜底意图分类
    results["rule_intent"] = test_rule_based_intent()

    # 测试2: 模型客户端
    results["model_client"] = test_model_client()

    # 测试3: 图结构
    results["graph_structure"] = test_graph_structure()

    # 测试4: 模型热插拔
    results["model_switch"] = test_model_switch()

    # 测试5: Agent完整流程（耗时较长）
    print("\n" + "=" * 60)
    print("注意: Agent完整流程测试需要调用Ollama，可能需要较长时间")
    print("=" * 60)
    results["agent_full"] = test_agent_full()

    # 汇总
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " 测试汇总".center(46) + "║")
    print("╠" + "═" * 58 + "╣")
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"║  {name:30s} {status:>20s}  ║")
    total_passed = sum(1 for p in results.values() if p)
    total = len(results)
    print("╠" + "═" * 58 + "╣")
    print(f"║  总计: {total_passed}/{total} 通过{'':>28s}║")
    print("╚" + "═" * 58 + "╝")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
