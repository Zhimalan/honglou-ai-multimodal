"""
红楼梦文本切片模块
- 段落级粗切分（按双换行）
- 递归字符细切分（chunk_size=800, overlap=150）
- 兼顾段落语义边界，避免切断完整段落
- 输出供RAG使用的标准化切片
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.data_models import TextChunk

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR  # 输出到同一目录

CHUNK_SIZE = 800           # 每段最大字符数
CHUNK_OVERLAP = 150        # 切片之间的重叠字符数
MIN_CHUNK_SIZE = 50        # 最小切片字符数


def paragraph_split(text: str) -> List[str]:
    """按段落（双换行）切分为段落列表"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    递归字符切分：在段落边界优先切分，避免切断段落
    - 优先按段落（\\n\\n）切分
    - 单段过长时按句子切分（。！？后的空白）
    - 仍过长时按硬长度切分，最后拼上overlap
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []

    chunks = []
    paragraphs = paragraph_split(text)

    current_chunk = ""

    for para in paragraphs:
        # 如果当前段落本身就超过 chunk_size
        if len(para) > chunk_size:
            # 先保存当前累积
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # 对超大段落按句子切分
            sub_chunks = split_long_paragraph(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue

        # 累加段落 + 段落间距
        candidate = current_chunk + ("\n\n" if current_chunk else "") + para

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # 保存当前切片
            if current_chunk:
                chunks.append(current_chunk)
            # 新切片从当前段落开始，带上 overlap 上下文
            overlap_text = get_overlap_tail(current_chunk, overlap)
            current_chunk = (overlap_text + "\n\n" + para) if overlap_text else para

    # 最后一段
    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk)

    return chunks


def split_long_paragraph(para: str, chunk_size: int, overlap: int) -> List[str]:
    """对超长段落按句子切分"""
    # 按中文句末标点切分，保留标点
    sentences = re.split(r"([。！？\n])", para)
    # 合并标点和句子
    merged = []
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
        if s.strip():
            merged.append(s)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        merged.append(sentences[-1])

    chunks = []
    current = ""

    for sent in merged:
        candidate = current + sent
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + sent
            else:
                # 单句就超过chunk_size，强制按字符切分
                chunks.append(sent[:chunk_size])
                current = sent[chunk_size - overlap:] if len(sent) > chunk_size else ""

    if current.strip():
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def get_overlap_tail(text: str, overlap: int) -> str:
    """取末尾 overlap 字符的语义完整片段"""
    if len(text) <= overlap:
        return text

    tail = text[-overlap:]
    # 尝试找到最近的句子边界
    for sep in ["。", "！", "？", "\n", "；"]:
        idx = tail.find(sep)
        if idx != -1 and idx < overlap - 20:
            return tail[idx + 1:]
    return tail


def chunk_chapter(chapter: Dict) -> List[TextChunk]:
    """对单回文本进行切片，返回TextChunk列表"""
    chapter_num = chapter["chapter_num"]
    chapter_title = chapter["chapter_title"]
    text = chapter["raw_text"]

    raw_chunks = recursive_chunk(text)
    chunks = []

    char_pos = 0  # 在原文中的位置追踪
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = f"ch{chapter_num:03d}_seg{idx:04d}"
        try:
            chunk = TextChunk(
                chunk_id=chunk_id,
                chapter_num=chapter_num,
                text=chunk_text.strip(),
                char_count=len(chunk_text.strip()),
                char_start=char_pos,
                char_end=char_pos + len(chunk_text),
            )
            chunks.append(chunk)
        except Exception as e:
            print(f"  ✗ {chunk_id} 校验失败: {e}")
        char_pos += len(chunk_text)

    return chunks


def run_chunking():
    """执行完整切片流水线"""
    print("=" * 60)
    print("红楼梦文本切片流水线")
    print(f"配置: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, min_chunk={MIN_CHUNK_SIZE}")
    print("=" * 60)

    # 加载清洗后的章回数据
    chapters_path = PROCESSED_DIR / "chapters_cleaned.json"
    with open(chapters_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    print(f"\n加载 {len(chapters)} 回数据，开始切片...")

    all_chunks = []
    chapter_summary = []

    for ch in chapters:
        chunks = chunk_chapter(ch)
        all_chunks.extend(chunks)
        chapter_summary.append({
            "chapter_num": ch["chapter_num"],
            "chapter_title": ch["chapter_title"],
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(c.char_count for c in chunks) // len(chunks) if chunks else 0,
        })

    # ─── 写出切片 ───
    output_path = OUTPUT_DIR / "chunks_rag.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 切片完成: 共 {len(all_chunks)} 个切片")
    print(f"  ✓ 输出: {output_path}")

    # ─── 切片统计 ───
    df = pd.DataFrame([c.model_dump() for c in all_chunks])
    print(f"\n  切片字数统计:")
    print(f"    最小: {df['char_count'].min()} 字")
    print(f"    最大: {df['char_count'].max()} 字")
    print(f"    平均: {df['char_count'].mean():.1f} 字")
    print(f"    中位数: {df['char_count'].median()} 字")

    # ─── 每回切片数 ───
    summary_path = OUTPUT_DIR / "chunk_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(chapter_summary, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ 每回切片摘要: {summary_path.name}")

    # ─── 切片预览 ───
    print("\n  切片预览（前3个）:")
    for chunk in all_chunks[:3]:
        print(f"    [{chunk.chunk_id}] {chunk.char_count}字 | {chunk.text[:60]}...")

    print("\n" + "=" * 60)
    print(f"切片流水线完成: 总切片数 {len(all_chunks)}, 平均每回 {len(all_chunks) // 120} 个")
    print("=" * 60)

    return all_chunks


if __name__ == "__main__":
    import pandas as pd  # 在这里导入避免对其他模块的影响
    run_chunking()
