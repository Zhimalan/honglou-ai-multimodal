"""
Pyvis 交互式人物关系图可视化
生成可拖拽、缩放的交互式 HTML 网页图谱，可嵌入 Gradio 前端。
"""

import os
import sys
import json
import networkx as nx
from pathlib import Path
from pyvis.network import Network

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "graphrag_index"
REPORT_DIR = PROJECT_ROOT / "report"
HTML_DIR = PROJECT_ROOT / "visualize" / "code_plot" / "html_output"

# 关系类型颜色
RELATION_COLORS = {
    "夫妻": "#E74C3C",
    "亲子": "#2980B9",
    "主仆": "#27AE60",
    "兄妹": "#8E44AD",
    "其他": "#BDC3C7",
}

# 社区颜色
COMMUNITY_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#FF5722", "#795548", "#607D8B", "#4CAF50", "#FF9800",
]


def load_graph() -> nx.Graph:
    graph_path = INDEX_DIR / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"图谱文件不存在: {graph_path}")
    return nx.read_graphml(str(graph_path))


def load_communities() -> dict:
    comm_path = INDEX_DIR / "communities.json"
    if comm_path.exists():
        with open(comm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_interactive_graph(
    G: nx.Graph,
    communities: dict,
    top_n: int = 50,
    output_path: str = None,
    title: str = "红楼梦人物关系交互图谱",
):
    """
    生成交互式 HTML 图谱（核心人物 Top N）。
    """
    if output_path is None:
        output_path = str(HTML_DIR / "interactive_graph.html")
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    # 选 Top N 节点
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
    subgraph = G.subgraph(top_nodes)

    # 创建 Pyvis Network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        select_menu=True,
        filter_menu=True,
        notebook=False,
    )

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.4,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75
      },
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 14,
          "face": "Microsoft YaHei, sans-serif"
        },
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.3)",
          "size": 10,
          "x": 5,
          "y": 5
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none",
          "roundness": 0.5
        },
        "font": {
          "size": 0
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # 添加节点
    max_deg = max(dict(subgraph.degree()).values()) if subgraph.number_of_nodes() > 0 else 1
    for node in subgraph.nodes():
        data = subgraph.nodes[node]
        degree = subgraph.degree(node)
        comm_id = communities.get(node, 0)
        color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]

        # 节点大小按度排序
        size = 15 + 40 * (degree / max_deg)

        # tooltip 信息
        desc = data.get("description", "")
        first_app = data.get("first_appearance", "")
        tooltip = f"<b>{node}</b><br/>度: {degree}<br/>描述: {desc}<br/>首次出现: {first_app}"

        net.add_node(
            node,
            label=node,
            size=size,
            color=color,
            title=tooltip,
            shape="dot",
        )

    # 添加边（按关系类型着色）
    for src, tgt, data in subgraph.edges(data=True):
        rtype = data.get("relation_type", "其他")
        weight = data.get("weight", 0)
        color = RELATION_COLORS.get(rtype, "#BDC3C7")
        width = 1.5 if rtype != "其他" else 0.5

        # 边的 tooltip
        tooltip = f"{src} ↔ {tgt}<br/>关系: {rtype}<br/>共现权重: {weight}"

        net.add_edge(
            src, tgt,
            color=color,
            width=width,
            title=tooltip,
            value=weight,
        )

    # 添加图例说明
    legend_html = """
    <div style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.7);
         padding:15px; border-radius:8px; color:white; font-size:13px;
         font-family:'Microsoft YaHei',sans-serif; z-index:1000;">
      <h3 style="margin:0 0 10px 0;">""" + title + f"""</h3>
      <p style="margin:5px 0;">节点数: {subgraph.number_of_nodes()} | 边数: {subgraph.number_of_edges()}</p>
      <div style="margin-top:10px;">
        <b>关系类型图例:</b><br/>
        <span style="color:{RELATION_COLORS['夫妻']};">━━</span> 夫妻<br/>
        <span style="color:{RELATION_COLORS['亲子']};">━━</span> 亲子<br/>
        <span style="color:{RELATION_COLORS['主仆']};">━━</span> 主仆<br/>
        <span style="color:{RELATION_COLORS['兄妹']};">━━</span> 兄妹<br/>
        <span style="color:{RELATION_COLORS['其他']};">━━</span> 其他<br/>
      </div>
      <p style="margin-top:10px; font-size:11px; color:#aaa;">
        拖拽节点 | 滚轮缩放 | 悬停查看详情
      </p>
    </div>
    """

    # 保存 HTML
    net.save_graph(output_path)

    # 注入图例到 HTML
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", legend_html + "</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  [save] 交互式图谱 → {output_path}")
    return output_path


def generate_character_subgraph_html(
    G: nx.Graph,
    character: str,
    output_path: str = None,
    top_k: int = 25,
):
    """
    生成指定人物的交互式关系子图。
    """
    if output_path is None:
        output_path = str(HTML_DIR / f"subgraph_{character}.html")
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    if character not in G.nodes():
        print(f"  [warn] 人物 {character} 不在图中")
        return None

    # 取一阶邻居，按权重排序
    neighbors = sorted(
        G.neighbors(character),
        key=lambda n: G.edges[character, n].get("weight", 0),
        reverse=True
    )[:top_k]

    nodes_set = {character} | set(neighbors)
    subgraph = G.subgraph(nodes_set)

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#16213e",
        font_color="white",
        notebook=False,
    )

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.5,
          "springLength": 120,
          "damping": 0.4
        }
      },
      "nodes": {
        "font": {"size": 16, "face": "Microsoft YaHei, sans-serif"}
      },
      "interaction": {"hover": true}
    }
    """)

    # 中心节点
    center_desc = G.nodes[character].get("description", "")
    net.add_node(
        character,
        label=character,
        size=50,
        color="#E74C3C",
        title=f"<b>{character}</b> (中心人物)<br/>{center_desc}",
        shape="star",
    )

    # 邻居节点
    for node in neighbors:
        edge_data = G.edges[character, node]
        rtype = edge_data.get("relation_type", "其他")
        weight = edge_data.get("weight", 0)
        color = RELATION_COLORS.get(rtype, "#BDC3C7")
        desc = G.nodes[node].get("description", "")

        net.add_node(
            node,
            label=node,
            size=20 + min(weight / 30, 30),
            color=color,
            title=f"<b>{node}</b><br/>与{character}关系: {rtype}<br/>共现权重: {weight}<br/>{desc}",
            shape="dot",
        )

        net.add_edge(
            character, node,
            color=color,
            width=2 if rtype != "其他" else 1,
            title=f"{rtype} (权重:{weight})",
            value=weight,
        )

    # 添加邻居之间的关系
    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i + 1:]:
            if subgraph.has_edge(n1, n2):
                edge_data = subgraph.edges[n1, n2]
                rtype = edge_data.get("relation_type", "其他")
                weight = edge_data.get("weight", 0)
                color = RELATION_COLORS.get(rtype, "#BDC3C7")
                net.add_edge(
                    n1, n2,
                    color=color,
                    width=1,
                    title=f"{rtype} (权重:{weight})",
                    value=weight,
                )

    # 图例
    legend_html = f"""
    <div style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.7);
         padding:12px; border-radius:8px; color:white; font-size:13px;
         font-family:'Microsoft YaHei',sans-serif; z-index:1000;">
      <h3 style="margin:0 0 5px 0;">{character} 人物关系图</h3>
      <p style="margin:3px 0; font-size:12px;">{center_desc}</p>
      <div style="margin-top:8px;">
        <span style="color:{RELATION_COLORS['夫妻']};">●</span> 夫妻
        <span style="color:{RELATION_COLORS['亲子']}; margin-left:8px;">●</span> 亲子
        <span style="color:{RELATION_COLORS['主仆']}; margin-left:8px;">●</span> 主仆
        <span style="color:{RELATION_COLORS['其他']}; margin-left:8px;">●</span> 其他
      </div>
    </div>
    """

    net.save_graph(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", legend_html + "</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  [save] {character}交互式子图 → {output_path}")
    return output_path


def generate_html_for_gradio(G: nx.Graph, character: str, top_k: int = 20) -> str:
    """
    生成可嵌入 Gradio 的 HTML 字符串（不保存文件）。
    返回 HTML 字符串供 Gradio HTML 组件使用。
    """
    import tempfile
    import uuid

    if character not in G.nodes():
        return f"<p>未找到人物: {character}</p>"

    tmp_path = os.path.join(tempfile.gettempdir(), f"pyvis_{uuid.uuid4().hex[:8]}.html")
    generate_character_subgraph_html(G, character, output_path=tmp_path, top_k=top_k)

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 清理临时文件
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return html_content


if __name__ == "__main__":
    print("=" * 60)
    print("Pyvis 交互式可视化生成")
    print("=" * 60)

    G = load_graph()
    communities = load_communities()
    print(f"  图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")

    print("[1/3] 核心人物交互式图谱 (Top 50)...")
    generate_interactive_graph(G, communities, top_n=50, title="红楼梦核心人物关系交互图谱")

    print("\n[2/3] 贾宝玉交互式子图...")
    generate_character_subgraph_html(G, "贾宝玉")

    print("\n[3/3] 林黛玉交互式子图...")
    generate_character_subgraph_html(G, "林黛玉")

    print("\n" + "=" * 60)
    print("交互式图谱生成完成! 保存在 visualize/code_plot/html_output/")
    print("  用浏览器打开 .html 文件即可交互查看")
    print("=" * 60)
