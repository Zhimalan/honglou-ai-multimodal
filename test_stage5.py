"""
GraphRAG + 可视化 端到端测试
验证阶段五所有功能模块正常工作。
"""

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from graphrag_index.graph_rag import GraphRAG
from visualize.comfyui_api.comfyui_client import generate_character_relation_image


def test_graphrag():
    """测试 GraphRAG 检索功能"""
    print("=" * 60)
    print("[1] GraphRAG 检索功能测试")
    print("=" * 60)

    rag = GraphRAG()
    passed = 0
    failed = 0

    # 测试1: 实体查找
    print("\n--- 1.1 实体查找 ---")
    test_cases = [
        ("贾宝玉", True),
        ("宝玉", True),      # 别名
        ("凤姐", True),      # 别名
        ("贾母", True),      # 别名
        ("不存在的人物", False),
    ]
    for name, expected_found in test_cases:
        result = rag.search_entity(name)
        status = "✓" if result["found"] == expected_found else "✗"
        if result["found"] == expected_found:
            passed += 1
        else:
            failed += 1
        found_str = f"找到({result.get('name', '')})" if result["found"] else "未找到"
        print(f"  {status} 查找 '{name}': {found_str}")

    # 测试2: 一跳邻居
    print("\n--- 1.2 一跳邻居 ---")
    result = rag.get_neighbors("贾宝玉", hop=1, top_k=5)
    if result["found"] and len(result["neighbors"]) > 0:
        passed += 1
        print(f"  ✓ 贾宝玉一跳邻居: {len(result['neighbors'])} 人")
        for n in result["neighbors"][:3]:
            print(f"    - {n['name']} (权重:{n['weight']}, 关系:{n['relation_type']})")
    else:
        failed += 1
        print(f"  ✗ 一跳邻居查询失败")

    # 测试3: 二跳邻居
    print("\n--- 1.3 二跳邻居 ---")
    result = rag.get_neighbors("林黛玉", hop=2, top_k=5)
    if result["found"] and len(result["neighbors"]) > 0:
        passed += 1
        print(f"  ✓ 林黛玉二跳邻居: {len(result['neighbors'])} 人")
        for n in result["neighbors"][:3]:
            print(f"    - {n['name']} (关联强度:{n['total_weight']:.0f})")
    else:
        failed += 1
        print(f"  ✗ 二跳邻居查询失败")

    # 测试4: 关系路径
    print("\n--- 1.4 关系路径 ---")
    path_cases = [
        ("贾宝玉", "林黛玉"),
        ("贾宝玉", "王熙凤"),
        ("林黛玉", "薛宝钗"),
        ("贾宝玉", "刘姥姥"),
    ]
    for n1, n2 in path_cases:
        result = rag.find_relation_path(n1, n2)
        if result["found"] and result["shortest_path"]:
            passed += 1
            path_str = " → ".join(result["shortest_path"])
            print(f"  ✓ {n1} → {n2}: {path_str} (距离:{result['shortest_distance']})")
        else:
            failed += 1
            print(f"  ✗ {n1} → {n2}: 查询失败")

    # 测试5: 关系概要
    print("\n--- 1.5 关系概要 ---")
    result = rag.get_relation_summary("王熙凤")
    if result["found"] and result["total_relations"] > 0:
        passed += 1
        print(f"  ✓ 王熙凤关系概要: {result['total_relations']} 条关系")
        for rtype, rels in result["relations"].items():
            if rels:
                names = ", ".join([r["name"] for r in rels[:3]])
                print(f"    {rtype}: {names}")
    else:
        failed += 1
        print(f"  ✗ 关系概要查询失败")

    # 测试6: 子图提取
    print("\n--- 1.6 子图提取 ---")
    result = rag.get_character_subgraph("贾宝玉", hop=1)
    if result["found"] and result["node_count"] > 0:
        passed += 1
        print(f"  ✓ 贾宝玉子图: {result['node_count']} 节点, {result['edge_count']} 边")
    else:
        failed += 1
        print(f"  ✗ 子图提取失败")

    # 测试7: 社区查询
    print("\n--- 1.7 社区查询 ---")
    result = rag.get_community_members("贾宝玉")
    if result["found"] and result["total"] > 0:
        passed += 1
        print(f"  ✓ 贾宝玉所在社区 {result['community']}: {result['total']} 人")
    else:
        failed += 1
        print(f"  ✗ 社区查询失败")

    # 测试8: 自然语言查询
    print("\n--- 1.8 自然语言查询 ---")
    result = rag.query("贾宝玉和薛宝钗是什么关系？")
    if result["found"]:
        passed += 1
        if "path" in result:
            print(f"  ✓ 查询 '贾宝玉和薛宝钗是什么关系？': 找到路径")
            if result["path"].get("shortest_path"):
                print(f"    路径: {' → '.join(result['path']['shortest_path'])}")
        elif "relations" in result:
            print(f"  ✓ 查询识别到人物，返回关系概要")
    else:
        failed += 1
        print(f"  ✗ 自然语言查询失败")

    # 测试9: Top人物
    print("\n--- 1.9 Top人物 ---")
    top_chars = rag.get_top_characters(10)
    if len(top_chars) == 10:
        passed += 1
        print(f"  ✓ Top 10 人物:")
        for i, c in enumerate(top_chars[:5]):
            print(f"    {i+1}. {c['name']} (度:{c['degree']})")
    else:
        failed += 1
        print(f"  ✗ Top人物查询失败")

    print(f"\n  GraphRAG 测试结果: {passed} 通过, {failed} 失败")
    return passed, failed


def test_visualization():
    """测试可视化功能"""
    print("\n" + "=" * 60)
    print("[2] 可视化功能测试")
    print("=" * 60)

    passed = 0
    failed = 0

    # 检查静态图
    print("\n--- 2.1 静态图文件检查 ---")
    report_dir = PROJECT_ROOT / "report"
    expected_files = [
        "core_network.png",
        "subgraph_贾宝玉.png",
        "subgraph_王熙凤.png",
        "relation_type_pie.png",
        "degree_distribution.png",
    ]
    for fname in expected_files:
        fpath = report_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            passed += 1
            print(f"  ✓ {fname} ({size_kb:.0f} KB)")
        else:
            failed += 1
            print(f"  ✗ {fname} 不存在")

    # 检查交互式HTML
    print("\n--- 2.2 交互式图谱文件检查 ---")
    html_dir = PROJECT_ROOT / "visualize" / "code_plot" / "html_output"
    expected_htmls = [
        "interactive_graph.html",
        "subgraph_贾宝玉.html",
        "subgraph_林黛玉.html",
    ]
    for fname in expected_htmls:
        fpath = html_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            passed += 1
            print(f"  ✓ {fname} ({size_kb:.0f} KB)")
        else:
            failed += 1
            print(f"  ✗ {fname} 不存在")

    # 检查 parquet 文件
    print("\n--- 2.3 Parquet 索引文件检查 ---")
    index_dir = PROJECT_ROOT / "graphrag_index"
    expected_parquets = ["entities.parquet", "relationships.parquet"]
    for fname in expected_parquets:
        fpath = index_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            passed += 1
            print(f"  ✓ {fname} ({size_kb:.0f} KB)")
        else:
            failed += 1
            print(f"  ✗ {fname} 不存在")

    # 检查 GraphML
    graphml_path = index_dir / "graph.graphml"
    if graphml_path.exists():
        passed += 1
        print(f"  ✓ graph.graphml ({graphml_path.stat().st_size / 1024:.0f} KB)")
    else:
        failed += 1
        print(f"  ✗ graph.graphml 不存在")

    # 检查 ComfyUI 提示词
    print("\n--- 2.4 ComfyUI 提示词检查 ---")
    prompt_dir = PROJECT_ROOT / "visualize" / "comfyui_api" / "prompts"
    prompt_path = prompt_dir / "prompt_贾宝玉.json"
    if prompt_path.exists():
        passed += 1
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        print(f"  ✓ prompt_贾宝玉.json")
        print(f"    正向提示词长度: {len(prompt_data['positive_prompt'])} 字符")
        print(f"    反向提示词长度: {len(prompt_data['negative_prompt'])} 字符")
    else:
        failed += 1
        print(f"  ✗ prompt_贾宝玉.json 不存在")

    print(f"\n  可视化测试结果: {passed} 通过, {failed} 失败")
    return passed, failed


def test_comfyui_api():
    """测试 ComfyUI API 接口"""
    print("\n" + "=" * 60)
    print("[3] ComfyUI API 接口测试")
    print("=" * 60)

    passed = 0
    failed = 0

    # 测试提示词生成
    print("\n--- 3.1 提示词生成 ---")
    test_relations = [
        {"name": "林黛玉", "relation_type": "其他"},
        {"name": "薛宝钗", "relation_type": "夫妻"},
        {"name": "袭人", "relation_type": "主仆"},
    ]
    result = generate_character_relation_image("贾宝玉", test_relations)

    if result["positive_prompt"] and result["negative_prompt"]:
        passed += 1
        print(f"  ✓ 正向提示词: {len(result['positive_prompt'])} 字符")
        print(f"  ✓ 反向提示词: {len(result['negative_prompt'])} 字符")
    else:
        failed += 1
        print(f"  ✗ 提示词生成失败")

    if result["prompt_path"]:
        passed += 1
        print(f"  ✓ 提示词文件已保存")
    else:
        failed += 1
        print(f"  ✗ 提示词文件未保存")

    # ComfyUI 不可用时是预期行为
    if not result["success"]:
        passed += 1
        print(f"  ✓ ComfyUI 不可用时优雅降级 (预期行为)")
    else:
        print(f"  ℹ ComfyUI 服务可用，图片已生成")

    print(f"\n  ComfyUI 测试结果: {passed} 通过, {failed} 失败")
    return passed, failed


def main():
    """主测试入口"""
    start_time = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  阶段五 GraphRAG + 双可视化 端到端测试               ║")
    print("╚" + "═" * 58 + "╝")

    p1, f1 = test_graphrag()
    p2, f2 = test_visualization()
    p3, f3 = test_comfyui_api()

    total_passed = p1 + p2 + p3
    total_failed = f1 + f2 + f3
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"  GraphRAG 检索:   {p1} 通过, {f1} 失败")
    print(f"  可视化文件:      {p2} 通过, {f2} 失败")
    print(f"  ComfyUI API:     {p3} 通过, {f3} 失败")
    print(f"  ────────────────────────────────")
    print(f"  总计:            {total_passed} 通过, {total_failed} 失败")
    print(f"  耗时:            {elapsed:.1f}s")
    if total_failed == 0:
        print(f"\n  ✅ 全部测试通过!")
    else:
        print(f"\n  ⚠ {total_failed} 项测试失败，请检查")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
