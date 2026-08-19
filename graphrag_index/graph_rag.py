"""
GraphRAG 检索 API
提供实体查找、邻居查询、关系路径、社区查询等图检索功能。
供 LangGraph Agent 的 GraphRAG 工具节点调用。
"""

import os
import json
import networkx as nx
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "graphrag_index"


class GraphRAG:
    """红楼梦知识图谱检索引擎"""

    def __init__(self, graph_path: Optional[str] = None):
        if graph_path is None:
            graph_path = str(INDEX_DIR / "graph.graphml")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"图谱文件不存在: {graph_path}\n请先运行 graph_builder.py 构建索引"
            )
        self.G = nx.read_graphml(graph_path)
        self._load_communities()

        # 人物别名映射（红楼人物常有多个称呼）
        self._alias_map = self._build_alias_map()

    def _load_communities(self):
        comm_path = INDEX_DIR / "communities.json"
        if os.path.exists(comm_path):
            with open(comm_path, "r", encoding="utf-8") as f:
                self.community_map = json.load(f)
        else:
            self.community_map = {}

    def _build_alias_map(self) -> dict[str, str]:
        """构建人物别名 → 标准名 映射"""
        aliases = {}
        # 常见别名
        alias_pairs = {
            "贾母": "史太君",
            "老太太": "史太君",
            "凤姐": "王熙凤",
            "凤辣子": "王熙凤",
            "琏二奶奶": "王熙凤",
            "宝二爷": "贾宝玉",
            "宝玉": "贾宝玉",
            "颦儿": "林黛玉",
            "林妹妹": "林黛玉",
            "黛玉": "林黛玉",
            "宝姐姐": "薛宝钗",
            "蘅芜君": "薛宝钗",
            "潇湘妃子": "林黛玉",
            "怡红公子": "贾宝玉",
            "湘云": "史湘云",
            "二哥哥": "贾宝玉",
            "琏二爷": "贾琏",
            "蓉大奶奶": "秦可卿",
            "可卿": "秦可卿",
            "迎春": "贾迎春",
            "探春": "贾探春",
            "惜春": "贾惜春",
            "三姑娘": "贾探春",
            "四姑娘": "贾惜春",
            "二姑娘": "贾迎春",
            "薛大爷": "薛蟠",
            "呆霸王": "薛蟠",
            "姨妈": "薛姨妈",
            "赵姨娘": "赵姨娘",
        }
        for alias, canonical in alias_pairs.items():
            aliases[alias] = canonical
        # 也把所有图中的节点名加入（自映射）
        for node in self.G.nodes():
            aliases[node] = node
        return aliases

    def _resolve_name(self, name: str) -> Optional[str]:
        """将用户输入的人物名解析为图中的标准节点名"""
        name = name.strip()
        # 精确匹配
        if name in self.G.nodes():
            return name
        # 别名匹配
        if name in self._alias_map:
            return self._alias_map[name]
        # 模糊匹配：包含关系
        candidates = [n for n in self.G.nodes() if name in n or n in name]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # 优先选最短名（更精确）
            return min(candidates, key=len)
        return None

    # ──────────────────────────────────────────
    # 检索 API
    # ──────────────────────────────────────────

    def search_entity(self, name: str) -> dict:
        """
        查找人物实体信息。
        返回: {name, description, first_appearance, degree, community, found}
        """
        canonical = self._resolve_name(name)
        if canonical is None:
            return {"found": False, "query": name, "message": f"未找到人物: {name}"}

        data = self.G.nodes[canonical]
        degree = self.G.degree(canonical)
        community = self.community_map.get(canonical, -1)

        return {
            "found": True,
            "name": canonical,
            "query": name,
            "description": data.get("description", ""),
            "first_appearance": data.get("first_appearance", ""),
            "notes": data.get("notes", ""),
            "degree": degree,
            "community": community,
        }

    def get_neighbors(self, name: str, hop: int = 1, top_k: int = 20) -> dict:
        """
        获取人物的 N 跳邻居。
        hop=1: 直接关联人物
        hop=2: 间接关联人物（通过一个中间人）
        top_k: 返回前 K 个（按边权重排序）
        """
        canonical = self._resolve_name(name)
        if canonical is None:
            return {"found": False, "query": name, "message": f"未找到人物: {name}"}

        if hop == 1:
            neighbors = []
            for neighbor in self.G.neighbors(canonical):
                edge_data = self.G.edges[canonical, neighbor]
                neighbors.append({
                    "name": neighbor,
                    "weight": edge_data.get("weight", 0),
                    "relation_type": edge_data.get("relation_type", "其他"),
                    "description": self.G.nodes[neighbor].get("description", ""),
                })
            neighbors.sort(key=lambda x: x["weight"], reverse=True)
            neighbors = neighbors[:top_k]
            return {
                "found": True,
                "name": canonical,
                "query": name,
                "hop": hop,
                "neighbors": neighbors,
                "total": len(neighbors),
            }
        elif hop == 2:
            # 一跳邻居
            first_hop = set(self.G.neighbors(canonical))
            # 二跳邻居
            second_hop = set()
            for n1 in first_hop:
                for n2 in self.G.neighbors(n1):
                    if n2 != canonical and n2 not in first_hop:
                        second_hop.add(n2)

            # 计算二跳邻居的关联强度（通过中间人的权重之和）
            second_hop_list = []
            for n2 in second_hop:
                total_weight = 0
                paths = []
                for n1 in first_hop:
                    if self.G.has_edge(n1, n2):
                        w = self.G.edges[n1, n2].get("weight", 0)
                        total_weight += w
                        paths.append({"via": n1, "weight": w})
                second_hop_list.append({
                    "name": n2,
                    "total_weight": total_weight,
                    "description": self.G.nodes[n2].get("description", ""),
                    "paths": paths[:3],  # 只显示前3条路径
                })
            second_hop_list.sort(key=lambda x: x["total_weight"], reverse=True)
            second_hop_list = second_hop_list[:top_k]
            return {
                "found": True,
                "name": canonical,
                "query": name,
                "hop": hop,
                "neighbors": second_hop_list,
                "total": len(second_hop_list),
            }
        else:
            return {"found": False, "message": f"暂不支持 {hop} 跳查询"}

    def find_relation_path(self, name1: str, name2: str, top_k: int = 5) -> dict:
        """
        查找两个人物之间的关系路径（最短路径 + 所有可能路径）。
        返回路径及其上的关系类型。
        """
        n1 = self._resolve_name(name1)
        n2 = self._resolve_name(name2)

        if n1 is None or n2 is None:
            missing = name1 if n1 is None else name2
            return {"found": False, "message": f"未找到人物: {missing}"}

        if n1 == n2:
            return {"found": True, "path": [n1], "message": "同一人物"}

        if not nx.has_path(self.G, n1, n2):
            return {"found": True, "path": [], "message": f"{n1} 与 {n2} 之间无路径"}

        # 最短路径
        try:
            shortest = nx.shortest_path(self.G, n1, n2)
        except nx.NetworkXError:
            shortest = []

        # 所有最短路径（可能有多条）
        try:
            all_shortest = list(nx.all_shortest_paths(self.G, n1, n2))
        except nx.NetworkXError:
            all_shortest = [shortest] if shortest else []

        # 构建路径详情（每段的关系类型）
        paths_detail = []
        for path in all_shortest[:top_k]:
            segments = []
            for i in range(len(path) - 1):
                edge_data = self.G.edges[path[i], path[i + 1]]
                segments.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "relation_type": edge_data.get("relation_type", "其他"),
                    "weight": edge_data.get("weight", 0),
                })
            paths_detail.append({
                "path": path,
                "length": len(path) - 1,
                "segments": segments,
            })

        return {
            "found": True,
            "name1": n1,
            "name2": n2,
            "shortest_path": shortest,
            "shortest_distance": len(shortest) - 1 if shortest else -1,
            "all_paths": paths_detail,
            "total_paths_found": len(all_shortest),
        }

    def get_character_subgraph(self, name: str, hop: int = 1) -> dict:
        """
        获取人物子图（用于可视化）。
        返回节点和边列表，可直接用于绘图。
        """
        canonical = self._resolve_name(name)
        if canonical is None:
            return {"found": False, "message": f"未找到人物: {name}"}

        if hop == 1:
            nodes_set = {canonical} | set(self.G.neighbors(canonical))
        else:
            nodes_set = {canonical}
            frontier = {canonical}
            for _ in range(hop):
                new_frontier = set()
                for n in frontier:
                    new_frontier.update(self.G.neighbors(n))
                nodes_set.update(new_frontier)
                frontier = new_frontier

        subgraph = self.G.subgraph(nodes_set)

        nodes = []
        for node, data in subgraph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "description": data.get("description", ""),
                "degree": subgraph.degree(node),
                "is_center": node == canonical,
            })

        edges = []
        for src, tgt, data in subgraph.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "weight": data.get("weight", 0),
                "relation_type": data.get("relation_type", "其他"),
            })

        return {
            "found": True,
            "center": canonical,
            "hop": hop,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def get_community_members(self, name: str) -> dict:
        """获取与该人物同一社区的所有人物"""
        canonical = self._resolve_name(name)
        if canonical is None:
            return {"found": False, "message": f"未找到人物: {name}"}

        comm_id = self.community_map.get(canonical, -1)
        if comm_id == -1:
            return {"found": True, "name": canonical, "community": -1, "members": []}

        members = [n for n, c in self.community_map.items() if c == comm_id]
        members_info = []
        for m in members:
            data = self.G.nodes.get(m, {})
            members_info.append({
                "name": m,
                "description": data.get("description", ""),
                "degree": self.G.degree(m) if m in self.G else 0,
            })
        members_info.sort(key=lambda x: x["degree"], reverse=True)

        return {
            "found": True,
            "name": canonical,
            "community": comm_id,
            "members": members_info,
            "total": len(members_info),
        }

    def get_relation_summary(self, name: str) -> dict:
        """
        获取人物关系概要：按关系类型分组的关联人物列表。
        用于生成自然语言的人物关系描述。
        """
        canonical = self._resolve_name(name)
        if canonical is None:
            return {"found": False, "message": f"未找到人物: {name}"}

        relations = {
            "夫妻": [],
            "亲子": [],
            "主仆": [],
            "兄妹": [],
            "其他": [],
        }

        for neighbor in self.G.neighbors(canonical):
            edge_data = self.G.edges[canonical, neighbor]
            rtype = edge_data.get("relation_type", "其他")
            weight = edge_data.get("weight", 0)
            neighbor_desc = self.G.nodes[neighbor].get("description", "")
            relations.setdefault(rtype, []).append({
                "name": neighbor,
                "weight": weight,
                "description": neighbor_desc,
            })

        # 每类按权重排序
        for rtype in relations:
            relations[rtype].sort(key=lambda x: x["weight"], reverse=True)

        return {
            "found": True,
            "name": canonical,
            "description": self.G.nodes[canonical].get("description", ""),
            "relations": relations,
            "total_relations": sum(len(v) for v in relations.values()),
        }

    def get_top_characters(self, top_k: int = 20, by: str = "degree") -> list[dict]:
        """获取热门人物（按度中心性或加权度排序）"""
        if by == "degree":
            ranked = sorted(self.G.nodes(), key=lambda n: self.G.degree(n), reverse=True)
        elif by == "weighted_degree":
            ranked = sorted(
                self.G.nodes(),
                key=lambda n: sum(d.get("weight", 1) for _, _, d in self.G.edges(n, data=True)),
                reverse=True,
            )
        else:
            ranked = sorted(self.G.nodes(), key=lambda n: self.G.degree(n), reverse=True)

        result = []
        for name in ranked[:top_k]:
            data = self.G.nodes[name]
            result.append({
                "name": name,
                "degree": self.G.degree(name),
                "description": data.get("description", ""),
                "first_appearance": data.get("first_appearance", ""),
            })
        return result

    def query(self, question: str) -> dict:
        """
        简单的自然语言查询接口。
        尝试从问题中提取人物名，返回相关关系信息。
        """
        # 尝试匹配所有已知人物名
        found_chars = []
        for node in self.G.nodes():
            if node in question:
                found_chars.append(node)

        # 也检查别名
        for alias, canonical in self._alias_map.items():
            if alias in question and canonical not in found_chars:
                found_chars.append(canonical)

        # 去重
        found_chars = list(dict.fromkeys(found_chars))

        if len(found_chars) == 0:
            return {"found": False, "message": "未识别到红楼人物名"}
        elif len(found_chars) == 1:
            return self.get_relation_summary(found_chars[0])
        else:
            # 两个人物 → 查关系路径
            n1, n2 = found_chars[0], found_chars[1]
            path_result = self.find_relation_path(n1, n2)
            summary1 = self.get_relation_summary(n1)
            summary2 = self.get_relation_summary(n2)
            return {
                "found": True,
                "characters": found_chars,
                "path": path_result,
                "summary_1": summary1,
                "summary_2": summary2,
            }


if __name__ == "__main__":
    rag = GraphRAG()
    print("\n=== 测试: 查找贾宝玉 ===")
    print(json.dumps(rag.search_entity("宝玉"), ensure_ascii=False, indent=2))

    print("\n=== 测试: 贾宝玉一跳邻居 (前5) ===")
    result = rag.get_neighbors("贾宝玉", hop=1, top_k=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== 测试: 贾宝玉→林黛玉关系路径 ===")
    result = rag.find_relation_path("贾宝玉", "林黛玉")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== 测试: 贾宝玉关系概要 ===")
    result = rag.get_relation_summary("贾宝玉")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== 测试: Top 10 人物 ===")
    for c in rag.get_top_characters(10):
        print(f"  {c['name']}: 度={c['degree']}, {c['description'][:30]}")
