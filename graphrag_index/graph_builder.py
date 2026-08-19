"""
GraphRAG 知识图谱构建器
从 edges.csv(共现权重) 和 characters.txt(人物描述) 构建红楼梦人物关系知识图谱。
推断关系类型：夫妻 / 亲子 / 主仆 / 兄妹 / 其他
输出 entities.parquet, relationships.parquet, graph.graphml
"""

import os
import re
import json
import networkx as nx
import pandas as pd
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "kg_triple"
INDEX_DIR = PROJECT_ROOT / "graphrag_index"


# ──────────────────────────────────────────────
# 1. 数据解析
# ──────────────────────────────────────────────

def parse_edges(edges_path: str) -> list[dict]:
    """解析 edges.csv: source;target;weight"""
    edges = []
    with open(edges_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) == 3:
                source, target, weight = parts
                try:
                    weight = float(weight)
                except ValueError:
                    continue
                edges.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "weight": weight,
                })
    print(f"  [edges] 解析 {len(edges)} 条边")
    return edges


def parse_characters(chars_path: str) -> dict[str, dict]:
    """解析 characters.txt: name\tdescription\tfirst_appearance\tnotes"""
    characters = {}
    with open(chars_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            first_appearance = parts[2].strip() if len(parts) > 2 else ""
            notes = parts[3].strip() if len(parts) > 3 else ""
            characters[name] = {
                "name": name,
                "description": description,
                "first_appearance": first_appearance,
                "notes": notes,
            }
    print(f"  [characters] 解析 {len(characters)} 个人物")
    return characters


# ──────────────────────────────────────────────
# 2. 关系类型推断
# ──────────────────────────────────────────────

# 已知关系映射表（从 characters.txt 的 notes 字段解析 + 红楼梦常识补充）
KNOWN_SPOUSE_PAIRS = {
    # 夫妻关系
    ("贾政", "王夫人"), ("贾赦", "邢夫人"), ("贾琏", "王熙凤"),
    ("贾琏", "尤二姐"), ("贾珍", "尤氏"), ("贾珠", "李纨"),
    ("贾蓉", "秦可卿"), ("林如海", "贾敏"), ("薛蟠", "夏金桂"),
    ("贾宝玉", "薛宝钗"), ("贾迎春", "孙绍祖"), ("贾探春", "某侍卫"),
    ("薛蝌", "邢岫烟"), ("贾代善", "史太君"),
}

KNOWN_PARENT_CHILD = {
    # 亲子关系 (parent, child)
    ("史太君", "贾赦"), ("史太君", "贾政"), ("史太君", "贾敏"),
    ("贾代善", "贾赦"), ("贾代善", "贾政"), ("贾代善", "贾敏"),
    ("贾政", "贾珠"), ("贾政", "贾宝玉"), ("贾政", "贾环"),
    ("王夫人", "贾珠"), ("王夫人", "贾宝玉"),
    ("赵姨娘", "贾环"), ("赵姨娘", "贾探春"),
    ("贾政", "贾探春"),
    ("贾赦", "贾琏"), ("邢夫人", "贾琏"),
    ("贾赦", "贾迎春"), ("贾赦", "贾琮"),
    ("贾珍", "贾蓉"),
    ("贾敬", "贾珍"), ("贾敬", "贾惜春"),
    ("贾敏", "林黛玉"), ("林如海", "林黛玉"),
    ("贾珠", "贾兰"),
    ("王熙凤", "巧姐"), ("贾琏", "巧姐"),
    ("薛姨妈", "薛蟠"), ("薛姨妈", "薛宝钗"),
    ("贾代化", "贾敬"),
}

KNOWN_MASTER_SERVANT = {
    # 主仆关系 (master, servant)
    ("贾宝玉", "袭人"), ("贾宝玉", "晴雯"), ("贾宝玉", "麝月"),
    ("贾宝玉", "秋纹"), ("贾宝玉", "碧痕"), ("贾宝玉", "紫鹃"),
    ("林黛玉", "紫鹃"), ("林黛玉", "雪雁"),
    ("薛宝钗", "莺儿"),
    ("王熙凤", "平儿"), ("王熙凤", "丰儿"),
    ("史太君", "鸳鸯"), ("史太君", "琥珀"),
    ("王夫人", "金钏"), ("王夫人", "玉钏"),
    ("贾母", "鸳鸯"),
    ("贾探春", "侍书"),
    ("贾迎春", "司棋"),
    ("贾惜春", "入画"),
    ("史湘云", "翠缕"),
    ("秦可卿", "瑞珠"),
}


def infer_relation_type(source: str, target: str, char_data: dict) -> str:
    """
    推断两个人物之间的关系类型。
    优先查已知关系表，其次从描述文本推断，最后默认为'其他'。
    """
    pair = (source, target)
    pair_rev = (target, source)

    # 查已知关系表
    if pair in KNOWN_SPOUSE_PAIRS or pair_rev in KNOWN_SPOUSE_PAIRS:
        return "夫妻"
    if pair in KNOWN_PARENT_CHILD:
        return "亲子"
    if pair_rev in KNOWN_PARENT_CHILD:
        return "亲子"
    if pair in KNOWN_MASTER_SERVANT or pair_rev in KNOWN_MASTER_SERVANT:
        return "主仆"

    # 从 characters.txt 的 notes 字段推断
    src_notes = char_data.get(source, {}).get("notes", "")
    tgt_notes = char_data.get(target, {}).get("notes", "")
    src_desc = char_data.get(source, {}).get("description", "")
    tgt_desc = char_data.get(target, {}).get("description", "")

    combined = src_notes + " " + tgt_notes + " " + src_desc + " " + tgt_desc

    # 检查夫妻关系
    if target in src_notes or source in tgt_notes:
        # 在 notes 中提到了对方
        context = ""
        if target in src_notes:
            # 找到 target 在 src_notes 中的上下文
            idx = src_notes.find(target)
            context = src_notes[max(0, idx - 3):idx + len(target) + 3]
        if source in tgt_notes:
            idx = tgt_notes.find(source)
            context = tgt_notes[max(0, idx - 3):idx + len(source) + 3]

        if any(kw in context for kw in ["妻", "嫁", "夫人"]):
            return "夫妻"
        if any(kw in context for kw in ["子", "女", "长子", "次子", "庶出", "生母"]):
            return "亲子"
        if any(kw in context for kw in ["妾", "丫鬟", "仆", "奴"]):
            return "主仆"
        if any(kw in context for kw in ["兄", "弟", "姐", "妹"]):
            return "兄妹"

    # 基于描述中的身份推断
    src_identity = char_data.get(source, {}).get("description", "")
    tgt_identity = char_data.get(target, {}).get("description", "")

    # 如果一方是丫鬟/仆人，另一方是主子
    servant_keywords = ["丫鬟", "仆", "奴", "小厮", "嬷嬷"]
    master_keywords = ["公子", "小姐", "夫人", "老爷", "太太", "奶奶"]

    src_is_servant = any(kw in src_identity for kw in servant_keywords)
    tgt_is_servant = any(kw in tgt_identity for kw in servant_keywords)
    src_is_master = any(kw in src_identity for kw in master_keywords)
    tgt_is_master = any(kw in tgt_identity for kw in master_keywords)

    if (src_is_master and tgt_is_servant) or (src_is_servant and tgt_is_master):
        return "主仆"

    return "其他"


# ──────────────────────────────────────────────
# 3. 图谱构建
# ──────────────────────────────────────────────

def build_graph(edges: list[dict], characters: dict) -> nx.Graph:
    """构建 NetworkX 无向图，节点带人物元数据，边带权重和关系类型"""
    G = nx.Graph()

    # 添加节点（带元数据）
    for name, info in characters.items():
        G.add_node(
            name,
            description=info["description"],
            first_appearance=info["first_appearance"],
            notes=info["notes"],
        )

    # 添加边（带权重和关系类型）
    for edge in edges:
        src, tgt, weight = edge["source"], edge["target"], edge["weight"]
        # 跳过不在人物表中的边（可能是地名等非人物实体）
        rel_type = infer_relation_type(src, tgt, characters)
        G.add_edge(
            src, tgt,
            weight=weight,
            relation_type=rel_type,
        )

    print(f"  [graph] 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    return G


# ──────────────────────────────────────────────
# 4. 生成 parquet 文件
# ──────────────────────────────────────────────

def generate_parquet(G: nx.Graph, characters: dict, output_dir: Path):
    """生成 entities.parquet 和 relationships.parquet"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # entities.parquet — 实体表
    entities_data = []
    for node, data in G.nodes(data=True):
        entities_data.append({
            "entity_name": node,
            "description": data.get("description", ""),
            "first_appearance": data.get("first_appearance", ""),
            "notes": data.get("notes", ""),
            "degree": G.degree(node),
            "weighted_degree": sum(d.get("weight", 1) for _, _, d in G.edges(node, data=True)),
        })
    entities_df = pd.DataFrame(entities_data)
    entities_path = output_dir / "entities.parquet"
    entities_df.to_parquet(entities_path, index=False)
    print(f"  [parquet] entities.parquet: {len(entities_df)} 行 → {entities_path}")

    # relationships.parquet — 关系表
    rel_data = []
    for src, tgt, data in G.edges(data=True):
        rel_data.append({
            "source": src,
            "target": tgt,
            "weight": data.get("weight", 0),
            "relation_type": data.get("relation_type", "其他"),
        })
    rel_df = pd.DataFrame(rel_data)
    rel_path = output_dir / "relationships.parquet"
    rel_df.to_parquet(rel_path, index=False)
    print(f"  [parquet] relationships.parquet: {len(rel_df)} 行 → {rel_path}")


# ──────────────────────────────────────────────
# 5. 社区检测（人物分组）
# ──────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> dict[str, int]:
    """使用 Louvain 算法检测社区，返回 {人物名: 社区ID}"""
    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    except (AttributeError, Exception):
        # networkx < 3.2 没有 louvain，回退到 greedy modularity
        communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))

    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    print(f"  [community] 检测到 {len(communities)} 个社区")
    for i, comm in enumerate(communities):
        members = [n for n, c in community_map.items() if c == i]
        if len(members) > 3:  # 只显示较大的社区
            print(f"    社区{i} ({len(members)}人): {', '.join(members[:8])}{'...' if len(members) > 8 else ''}")

    return community_map


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────

def build_index():
    """主入口：构建并保存 GraphRAG 索引"""
    print("=" * 60)
    print("开始构建 GraphRAG 知识图谱索引")
    print("=" * 60)

    # 1. 解析数据
    print("\n[1/5] 解析原始数据...")
    edges = parse_edges(str(DATA_DIR / "edges.csv"))
    characters = parse_characters(str(DATA_DIR / "characters.txt"))

    # 2. 构建图谱
    print("\n[2/5] 构建知识图谱...")
    G = build_graph(edges, characters)

    # 3. 社区检测
    print("\n[3/5] 社区检测...")
    community_map = detect_communities(G)
    nx.set_node_attributes(G, community_map, "community")

    # 4. 生成 parquet
    print("\n[4/5] 生成 parquet 文件...")
    generate_parquet(G, characters, INDEX_DIR)

    # 5. 保存图谱
    print("\n[5/5] 保存图谱文件...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    graphml_path = INDEX_DIR / "graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    print(f"  [save] graph.graphml → {graphml_path}")

    # 保存社区映射
    community_path = INDEX_DIR / "communities.json"
    with open(community_path, "w", encoding="utf-8") as f:
        json.dump(community_map, f, ensure_ascii=False, indent=2)
    print(f"  [save] communities.json → {community_path}")

    # 保存统计信息
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": len(set(community_map.values())),
        "top_10_by_degree": sorted(
            [(n, G.degree(n)) for n in G.nodes()],
            key=lambda x: x[1], reverse=True
        )[:10],
        "relation_type_counts": {},
    }
    for _, _, data in G.edges(data=True):
        rtype = data.get("relation_type", "其他")
        stats["relation_type_counts"][rtype] = stats["relation_type_counts"].get(rtype, 0) + 1

    stats_path = INDEX_DIR / "graph_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  [save] graph_stats.json → {stats_path}")

    print("\n" + "=" * 60)
    print("GraphRAG 索引构建完成!")
    print(f"  节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    print(f"  社区: {len(set(community_map.values()))}")
    print(f"  关系类型分布: {stats['relation_type_counts']}")
    print("=" * 60)

    return G


if __name__ == "__main__":
    build_index()
