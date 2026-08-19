"""
GraphRAG 索引构建入口脚本
运行: python build_index.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_builder import build_index

if __name__ == "__main__":
    build_index()
