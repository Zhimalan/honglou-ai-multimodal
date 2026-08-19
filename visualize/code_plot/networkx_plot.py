"""
NetworkX 静态人物关系图可视化
绘制红楼梦核心人物关系网络图，按关系类型区分连线颜色。
输出 PNG 到 report/ 目录。
"""

import os
import sys
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "graphrag_index"
REPORT_DIR = PROJECT_ROOT / "report"

# ─── 中文字体配置 ───
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/Deng.ttf",       # 等线
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            font_name = fm.FontProperties(fname=fp).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  [font] 使用字体: {font_name}")
            return font_name
    print("  [font] 警告: 未找到中文字体，可能显示乱码")
    return None


# ─── 关系类型颜色映射 ───
RELATION_COLORS = {
    "夫妻": "#E74C3C",   # 红色
    "亲子": "#2980B9",   # 蓝色
    "主仆": "#27AE60",   # 绿色
    "兄妹": "#8E44AD",   # 紫色
    "其他": "#BDC3C7",   # 浅灰
}

RELATION_LINESTYLES = {
    "夫妻": "-",
    "亲子": "-",
    "主仆": "--",
    "兄妹": "-.",
    "其他": ":",
}


def load_graph() -> nx.Graph:
    """加载知识图谱"""
    graph_path = INDEX_DIR / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"图谱文件不存在: {graph_path}\n请先运行 build_index.py")
    return nx.read_graphml(str(graph_path))


def plot_core_network(G: nx.Graph, top_n: int = 30, output_path: str = None):
    """
    绘制核心人物关系网络图（按度排序取 Top N）。
    """
    if output_path is None:
        output_path = str(REPORT_DIR / "core_network.png")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    # 选 Top N 节点
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
    subgraph = G.subgraph(top_nodes)

    fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # 布局：spring 布局，加权
    pos = nx.spring_layout(subgraph, k=2.5, iterations=80, weight="weight", seed=42)

    # 按关系类型分组绘制边
    edges_by_type = {}
    for src, tgt, data in subgraph.edges(data=True):
        rtype = data.get("relation_type", "其他")
        edges_by_type.setdefault(rtype, []).append((src, tgt))

    for rtype, edges in edges_by_type.items():
        color = RELATION_COLORS.get(rtype, "#BDC3C7")
        style = RELATION_LINESTYLES.get(rtype, ":")
        width = 1.5 if rtype != "其他" else 0.5
        alpha = 0.8 if rtype != "其他" else 0.3
        nx.draw_networkx_edges(
            subgraph, pos, edgelist=edges,
            edge_color=color, style=style, width=width, alpha=alpha, ax=ax,
        )

    # 节点大小：按度中心性
    degrees = dict(subgraph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 2000 * (degrees[n] / max_deg) for n in subgraph.nodes()]

    # 节点颜色：按社区着色
    communities = {}
    comm_path = INDEX_DIR / "communities.json"
    if comm_path.exists():
        with open(comm_path, "r", encoding="utf-8") as f:
            communities = json.load(f)

    community_colors = [
        "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    ]
    node_colors = [
        community_colors[communities.get(n, 0) % len(community_colors)]
        for n in subgraph.nodes()
    ]

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )

    # 标签
    labels = {n: n for n in subgraph.nodes()}
    nx.draw_networkx_labels(
        subgraph, pos, labels,
        font_size=9, font_family=plt.rcParams["font.family"],
        font_weight="bold", ax=ax,
    )

    # 图例
    legend_elements = []
    for rtype, color in RELATION_COLORS.items():
        if rtype in edges_by_type:
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=2,
                           linestyle=RELATION_LINESTYLES.get(rtype, ":"),
                           label=f"{rtype} ({len(edges_by_type[rtype])}条)")
            )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=12,
              framealpha=0.9, title="关系类型", title_fontsize=13)

    ax.set_title(f"红楼梦核心人物关系网络图 (Top {top_n})", fontsize=20, fontweight="bold", pad=20)
    ax.text(0.99, 0.01, f"节点: {subgraph.number_of_nodes()}  边: {subgraph.number_of_edges()}",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom", color="#666")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  [save] 核心人物关系图 → {output_path}")


def plot_character_subgraph(G: nx.Graph, character: str, output_path: str = None):
    """
    绘制指定人物的一阶关系子图。
    """
    if output_path is None:
        output_path = str(REPORT_DIR / f"subgraph_{character}.png")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    if character not in G.nodes():
        print(f"  [warn] 人物 {character} 不在图中")
        return

    # 一阶邻居子图
    neighbors = list(G.neighbors(character))
    if len(neighbors) > 25:
        # 按权重取前25个
        neighbors = sorted(
            neighbors,
            key=lambda n: G.edges[character, n].get("weight", 0),
            reverse=True
        )[:25]

    nodes_set = {character} | set(neighbors)
    subgraph = G.subgraph(nodes_set)

    fig, ax = plt.subplots(1, 1, figsize=(14, 11), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # 环形布局，中心人物在中间
    pos = nx.spring_layout(subgraph, k=2, iterations=60, weight="weight", seed=42)

    # 边按关系类型着色
    edges_by_type = {}
    for src, tgt, data in subgraph.edges(data=True):
        rtype = data.get("relation_type", "其他")
        edges_by_type.setdefault(rtype, []).append((src, tgt))

    for rtype, edges in edges_by_type.items():
        color = RELATION_COLORS.get(rtype, "#BDC3C7")
        width = 2.5 if rtype != "其他" else 1.0
        alpha = 0.85 if rtype != "其他" else 0.4
        nx.draw_networkx_edges(
            subgraph, pos, edgelist=edges,
            edge_color=color, width=width, alpha=alpha, ax=ax,
        )

    # 节点
    node_sizes = []
    node_colors = []
    for n in subgraph.nodes():
        if n == character:
            node_sizes.append(2000)
            node_colors.append("#E74C3C")
        else:
            deg = subgraph.degree(n)
            node_sizes.append(400 + deg * 30)
            # 按关系类型着色（与中心人物的关系）
            edge_data = G.edges[character, n] if G.has_edge(character, n) else {}
            rtype = edge_data.get("relation_type", "其他")
            node_colors.append(RELATION_COLORS.get(rtype, "#BDC3C7"))

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        subgraph, pos,
        font_size=10, font_family=plt.rcParams["font.family"],
        font_weight="bold", ax=ax,
    )

    # 图例
    legend_elements = []
    for rtype, color in RELATION_COLORS.items():
        if rtype in edges_by_type:
            legend_elements.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color, markersize=10,
                           label=f"{rtype}")
            )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11,
              framealpha=0.9, title="关系类型", title_fontsize=12)

    desc = G.nodes[character].get("description", "")
    ax.set_title(f"{character} 的人物关系图\n({desc})", fontsize=18, fontweight="bold", pad=15)
    ax.text(0.99, 0.01,
            f"直接关联: {len(neighbors)}人",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom", color="#666")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  [save] {character}关系子图 → {output_path}")


def plot_relation_type_distribution(G: nx.Graph, output_path: str = None):
    """绘制关系类型分布饼图"""
    if output_path is None:
        output_path = str(REPORT_DIR / "relation_type_pie.png")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    type_counts = Counter()
    for _, _, data in G.edges(data=True):
        rtype = data.get("relation_type", "其他")
        type_counts[rtype] += 1

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#FAFAFA")
    colors = [RELATION_COLORS.get(t, "#999") for t in type_counts.keys()]
    wedges, texts, autotexts = ax.pie(
        type_counts.values(),
        labels=type_counts.keys(),
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    ax.set_title("红楼梦人物关系类型分布", fontsize=18, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  [save] 关系类型分布饼图 → {output_path}")


def plot_degree_distribution(G: nx.Graph, output_path: str = None):
    """绘制人物度分布柱状图 (Top 20)"""
    if output_path is None:
        output_path = str(REPORT_DIR / "degree_distribution.png")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:20]
    degrees = [G.degree(n) for n in top_nodes]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars = ax.barh(range(len(top_nodes)), degrees, color="#3498DB", edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top_nodes)))
    ax.set_yticklabels(top_nodes, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("关联人物数（度）", fontsize=13)
    ax.set_title("红楼梦人物关系网络度分布 (Top 20)", fontsize=16, fontweight="bold", pad=15)

    # 在柱子上标注数值
    for bar, deg in zip(bars, degrees):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(deg), va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  [save] 度分布图 → {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("NetworkX 静态可视化生成")
    print("=" * 60)

    G = load_graph()
    print(f"  图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")

    print("[1/4] 核心人物关系网络图...")
    plot_core_network(G, top_n=30)

    print("\n[2/4] 贾宝玉人物关系子图...")
    plot_character_subgraph(G, "贾宝玉")

    print("\n[3/4] 王熙凤人物关系子图...")
    plot_character_subgraph(G, "王熙凤")

    print("\n[4/4] 关系类型分布 & 度分布...")
    plot_relation_type_distribution(G)
    plot_degree_distribution(G)

    print("\n" + "=" * 60)
    print("所有静态图生成完成! 保存在 report/ 目录")
    print("=" * 60)
