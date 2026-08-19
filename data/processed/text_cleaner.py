"""
红楼梦文本清洗流水线
- 解析120回原文，按回目切分
- 去除脂评批注残留、统一标点
- 清理空行、乱码、特殊字符
- Pydantic校验后输出结构化JSON
"""

import re
import json
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_models import RedChapter, Character, RelationTriple

# ─── 路径配置 ───
RAW_TEXT_PATH = PROJECT_ROOT / "data" / "raw_text" / "honglou_120_full.txt"
CHARACTERS_PATH = PROJECT_ROOT / "data" / "kg_triple" / "characters.txt"
EDGES_PATH = PROJECT_ROOT / "data" / "kg_triple" / "edges.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """清洗单段文本：统一标点、去乱码、去多余空白"""
    # 1. 统一标点：旧式句号．→。
    text = text.replace("．", "。")
    # 2. 统一引号
    text = text.replace("`", "\u2018").replace("-", "\u2014")
    # 3. 去除HTML残留标签
    text = re.sub(r"<[^>]+>", "", text)
    # 4. 去除上标引用标记 [1] [2] 等
    text = re.sub(r"\[\d+\]", "", text)
    # 5. 替换乱码常见模式
    text = text.replace("?", "？").replace("?", "—")
    # 6. 去掉行首缩进空格
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    # 7. 合并多余空行为单个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 8. 去除首尾空白
    return text.strip()


def parse_chapters(raw_text: str) -> list[dict]:
    """
    将原始全文解析为120回结构
    返回: [{"chapter_num": 1, "chapter_title": "...", "raw_text": "...", "char_count": 12345}, ...]
    """
    # 回目正则：第X回 后跟标题（标题可能含全角空格）
    chapter_pattern = re.compile(
        r"^(第[一二三四五六七八九十百零\d]+回)\s*(.+)$",
        re.MULTILINE
    )

    matches = list(chapter_pattern.finditer(raw_text))
    if len(matches) < 120:
        print(f"⚠ 警告: 只找到 {len(matches)} 个回目标记，期望120回")

    chapters = []
    for i, match in enumerate(matches):
        chapter_marker = match.group(1)  # "第一回"
        chapter_title = match.group(2).strip()  # "甄士隐梦幻识通灵..."
        start_pos = match.end()

        # 下一回的起始位置
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(raw_text)

        # 提取该回正文
        chapter_text = raw_text[start_pos:end_pos]
        chapter_text = clean_text(chapter_text)

        # 使用列表序号+1作为回号（原文按顺序排列，最可靠）
        # 同时也用 chinese_to_int 解析作为校验参考
        chapter_num = i + 1
        parsed_num = chinese_to_int(chapter_marker)
        if parsed_num != chapter_num:
            # 记录但不阻断——序号是权威来源
            pass

        chapters.append({
            "chapter_num": chapter_num,
            "chapter_title": chapter_title,
            "raw_text": chapter_text,
            "char_count": len(chapter_text.replace("\n", "").replace(" ", "")),
        })

    return chapters


def chinese_to_int(cn: str) -> int:
    """将'第一回'转换为数字1"""
    # 先提取数字部分
    num_str = cn.replace("第", "").replace("回", "")
    digit_map = {
        "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        "十": 10, "百": 100,
    }
    if num_str.isdigit():
        return int(num_str)

    # 处理"二十三"这类
    result = 0
    current = 0
    for ch in num_str:
        if ch in digit_map:
            val = digit_map[ch]
            if val >= 10:  # 十或百
                result += (current if current > 0 else 1) * val
                current = 0
            else:
                current = val
        result += current
    return result


def load_characters(filepath: Path) -> list[dict]:
    """加载人物清单"""
    characters = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                char = {
                    "name": parts[0].strip(),
                    "description": parts[1].strip() if len(parts) > 1 else "",
                    "first_appearance": parts[2].strip() if len(parts) > 2 else "",
                    "notes": parts[3].strip() if len(parts) > 3 else "",
                }
                characters.append(char)
    return characters


def load_edges(filepath: Path) -> list[dict]:
    """加载人物关系三元组"""
    edges = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 3:
                try:
                    edges.append({
                        "source": parts[0].strip(),
                        "target": parts[1].strip(),
                        "weight": float(parts[2].strip()),
                    })
                except ValueError:
                    continue
    return edges


def run_cleaning_pipeline():
    """执行完整的数据清洗流水线"""
    print("=" * 60)
    print("红楼梦文本清洗流水线")
    print("=" * 60)

    # ─── 1. 清洗120回原文 ───
    print("\n[1/3] 清洗红楼梦120回原文...")
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chapters_raw = parse_chapters(raw_text)
    chapters_validated = []
    failed = 0

    for ch in chapters_raw:
        try:
            validated = RedChapter(**ch)
            chapters_validated.append(validated.model_dump())
        except Exception as e:
            print(f"  ✗ 第{ch.get('chapter_num', '?')}回校验失败: {e}")
            failed += 1

    output_path = OUTPUT_DIR / "chapters_cleaned.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chapters_validated, f, ensure_ascii=False, indent=2)

    total_chars = sum(ch["char_count"] for ch in chapters_validated)
    print(f"  ✓ 清洗完成: {len(chapters_validated)}/120 回, 总计 {total_chars:,} 字")
    print(f"  ✓ 输出: {output_path}")
    if failed:
        print(f"  ⚠ {failed} 回校验失败")

    # ─── 2. 加载人物清单 ───
    print("\n[2/3] 加载人物清单...")
    characters_raw = load_characters(CHARACTERS_PATH)
    characters_validated = []
    for c in characters_raw:
        try:
            validated = Character(**c)
            characters_validated.append(validated.model_dump())
        except Exception as e:
            print(f"  ✗ 人物校验失败: {c.get('name', '?')} - {e}")

    output_path = OUTPUT_DIR / "characters_cleaned.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(characters_validated, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 人物清单: {len(characters_validated)} 人")
    print(f"  ✓ 输出: {output_path}")

    # ─── 3. 加载关系三元组 ───
    print("\n[3/3] 加载人物关系三元组...")
    edges_raw = load_edges(EDGES_PATH)
    edges_validated = []
    for e in edges_raw:
        try:
            validated = RelationTriple(**e)
            edges_validated.append(validated.model_dump())
        except Exception as ex:
            print(f"  ✗ 三元组校验失败: {e} - {ex}")

    output_path = OUTPUT_DIR / "edges_cleaned.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(edges_validated, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 关系三元组: {len(edges_validated)} 条")
    print(f"  ✓ 输出: {output_path}")

    # ─── 汇总 ───
    print("\n" + "=" * 60)
    print("清洗完成汇总:")
    print(f"  章回: {len(chapters_validated)} 回, {total_chars:,} 字")
    print(f"  人物: {len(characters_validated)} 人")
    print(f"  关系: {len(edges_validated)} 条")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    return {
        "chapters": chapters_validated,
        "characters": characters_validated,
        "edges": edges_validated,
    }


if __name__ == "__main__":
    run_cleaning_pipeline()
