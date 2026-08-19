"""
SFT 问答数据生成（基于 Ollama 本地推理）
- 由于 HuggingFace 上的 honglou-qa 数据集已下架，改用本地 Ollama qwen2.5:7b 自动生成
- 每个回目生成 7 条覆盖不同类型的问答
- 输出 LLaMA-Factory 标准格式（instruction + input + output）
- 断点续传支持
"""

import json
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_models import SFTSample

# ─── 路径配置 ───
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SFT_DIR = PROJECT_ROOT / "data" / "sft_dataset"
SFT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = SFT_DIR / "honglou_sft.json"
PROGRESS_FILE = SFT_DIR / "generation_progress.json"

# ─── 模型配置 ───
MODEL_NAME = "qwen2.5:7b"
OLLAMA_TIMEOUT = 180  # 秒
QA_PER_CHAPTER = 7    # 每回生成问答对数量
TOTAL_TARGET = 840    # 目标总问答数（120回 × 7）

# ─── 多种问题模板，确保多样性 ───
QA_TEMPLATES = [
    {
        "name": "情节概述",
        "system": "你是红楼梦专家。基于提供的章节内容，回答用户关于情节的问题。回答要简洁精准，50字以内。",
        "user_template": "请根据以下章节内容，回答：这一回主要讲了什么？\n\n章节内容：\n{chapter_text}\n\n要求：1)用一句话概括核心情节；2)不超过80字。",
    },
    {
        "name": "人物相关",
        "system": "你是红楼梦人物研究专家。回答要具体、有依据，不超过80字。",
        "user_template": "根据以下章节内容，回答：这一回中出现了哪些主要人物？他们的核心特征或事件是什么？\n\n章节内容：\n{chapter_text}\n\n要求：简洁列出2-3位关键人物及其表现。",
    },
    {
        "name": "诗词解析",
        "system": "你是红楼梦诗词研究专家。回答要有诗意、有深度，简洁明了。",
        "user_template": "以下章节中可能包含诗词。请找出其中出现的一句诗词或判词，并简析其含义。\n\n章节内容：\n{chapter_text}\n\n如果没有诗词，可说明本章的主题思想。",
    },
    {
        "name": "细节问答",
        "system": "你是红楼梦细节研究专家。回答要有具体细节支撑，不超过80字。",
        "user_template": "基于以下章节内容，提一个细节问题并给出答案。问题要有深度，能体现对文本的细致理解。\n\n章节内容：\n{chapter_text}\n\n请按这个格式回答：\nQ: [你的问题]\nA: [答案]",
    },
    {
        "name": "人物性格",
        "system": "你是红楼梦人物分析专家。回答要基于文本，避免过度演绎。",
        "user_template": "根据以下章节内容，分析一位出场人物的内心活动或性格特点。\n\n章节内容：\n{chapter_text}\n\n要求：不超过100字，要结合章节具体情节。",
    },
    {
        "name": "伏笔隐喻",
        "system": "你是红楼梦艺术手法分析专家。回答要有理有据，简洁。",
        "user_template": "阅读以下章节，指出其中可能存在的伏笔、隐喻或象征。\n\n章节内容：\n{chapter_text}\n\n要求：不超过80字。",
    },
    {
        "name": "对比赏析",
        "system": "你是红楼梦文学鉴赏专家。回答要有文采，简洁。",
        "user_template": "请基于以下章节内容，从语言艺术、人物塑造或情节安排等任一角度做简短赏析。\n\n章节内容：\n{chapter_text}\n\n要求：50-100字，要有具体例子。",
    },
]


def call_ollama(prompt: str, system: str, max_retries: int = 2) -> Optional[str]:
    """调用 Ollama 生成内容（带重试 + 超时保护）"""
    for attempt in range(max_retries):
        try:
            # 使用线程池实现超时控制，防止 Ollama 卡死
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ollama.chat,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 600,
                    },
                )
                response = future.result(timeout=OLLAMA_TIMEOUT)
            return response["message"]["content"].strip()
        except FuturesTimeoutError:
            print(f"  ⚠ 第{attempt + 1}次超时（{OLLAMA_TIMEOUT}秒），重试...")
        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠ 第{attempt + 1}次失败: {e}，{wait}秒后重试...")
            time.sleep(wait)
    return None


def parse_qa_output(text: str, template_name: str) -> List[Dict]:
    """从模型输出中解析QA对"""
    qa_pairs = []

    if template_name == "细节问答":
        # 期望格式：Q: ... A: ...
        q_match = re.search(r"Q[:：]\s*(.+?)(?=A[:：])", text, re.DOTALL)
        a_match = re.search(r"A[:：]\s*(.+?)$", text, re.DOTALL)
        if q_match and a_match:
            qa_pairs.append({
                "instruction": q_match.group(1).strip(),
                "output": a_match.group(1).strip(),
            })
        else:
            # 回退为指令格式
            qa_pairs.append({
                "instruction": f"红楼梦第X回的细节理解",
                "output": text.strip()[:300],
            })
    else:
        # 其他模板：直接把输出作为对问题的回答
        # 我们需要在调用前记录问题
        qa_pairs.append({"output": text.strip()})

    return qa_pairs


def generate_qa_for_chapter(chapter: Dict, template_idx: int) -> List[Dict]:
    """对单回生成问答对"""
    template = QA_TEMPLATES[template_idx % len(QA_TEMPLATES)]
    chapter_text = chapter["raw_text"]

    # 截取章节文本前 1500 字（避免超出上下文窗口）
    text_for_prompt = chapter_text[:1500]
    if len(chapter_text) > 1500:
        text_for_prompt += "\n..."

    user_prompt = template["user_template"].format(chapter_text=text_for_prompt)
    response = call_ollama(user_prompt, template["system"])

    if not response:
        return []

    # 构造问答对
    if template["name"] == "细节问答":
        # 模型自己生成 Q 和 A
        qa_pairs = parse_qa_output(response, "细节问答")
        return qa_pairs
    else:
        # 标准格式：instruction 是固定问题模板，output 是模型回答
        if template["name"] == "情节概述":
            question = f"红楼梦第{chapter['chapter_num']}回《{chapter['chapter_title']}》主要讲了什么？"
        elif template["name"] == "人物相关":
            question = f"第{chapter['chapter_num']}回中出现了哪些主要人物，他们有什么表现？"
        elif template["name"] == "诗词解析":
            question = f"第{chapter['chapter_num']}回中有什么诗词或判词？含义是什么？"
        elif template["name"] == "人物性格":
            question = f"请分析第{chapter['chapter_num']}回中某位出场人物的内心或性格。"
        elif template["name"] == "伏笔隐喻":
            question = f"第{chapter['chapter_num']}回中有哪些伏笔或隐喻？"
        elif template["name"] == "对比赏析":
            question = f"请对第{chapter['chapter_num']}回做简要的文学赏析。"

        return [{
            "instruction": question,
            "input": "",
            "output": response,
        }]


def load_progress() -> Dict:
    """加载进度（支持断点续传）"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_chapters": [], "generated_qa": []}


def save_progress(progress: Dict):
    """保存进度"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_sft_dataset(qa_list: List[Dict]):
    """保存为LLaMA-Factory标准格式"""
    # 转换为标准 instruction 格式
    sft_data = []
    for qa in qa_list:
        try:
            sample = SFTSample(
                instruction=qa["instruction"],
                input=qa.get("input", ""),
                output=qa["output"],
            )
            sft_data.append(sample.model_dump())
        except Exception as e:
            print(f"  ✗ 跳过不合格样本: {e}")

    # 添加 system 字段（符合 LLaMA-Factory alpaca 格式）
    final_data = []
    for item in sft_data:
        final_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
            "system": "你是红楼梦研究专家，精通红楼梦的人物、情节、诗词、文化背景。",
            "history": [],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ SFT数据集保存: {OUTPUT_FILE}")
    print(f"  ✓ 总样本数: {len(final_data)}")
    return final_data


def run_sft_generation():
    """执行 SFT 数据生成流水线"""
    print("=" * 60)
    print(f"SFT 问答数据自动生成（基于 Ollama {MODEL_NAME}）")
    print(f"目标: 每回 {QA_PER_CHAPTER} 条, 总计 {TOTAL_TARGET} 条")
    print("=" * 60)

    # 检查Ollama服务
    print("\n[1/5] 检查 Ollama 服务...")
    try:
        ollama.list()
        print(f"  ✓ Ollama 服务正常")
    except Exception as e:
        print(f"  ✗ 无法连接 Ollama: {e}")
        print(f"    请确认 Ollama 已运行且 {MODEL_NAME} 模型已下载")
        return

    # 加载章节数据
    print("\n[2/5] 加载章节数据...")
    chapters_path = PROCESSED_DIR / "chapters_cleaned.json"
    with open(chapters_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)
    print(f"  ✓ 加载 {len(chapters)} 回数据")

    # 加载进度
    progress = load_progress()
    completed = set(progress["completed_chapters"])
    all_qa = progress["generated_qa"]

    if completed:
        print(f"  ↻ 已完成 {len(completed)} 回（断点续传）")

    # 开始生成
    print("\n[3/5] 生成问答对...")
    start_time = time.time()

    for ch_idx, chapter in enumerate(chapters):
        if chapter["chapter_num"] in completed:
            continue

        if len(all_qa) >= TOTAL_TARGET:
            print(f"\n  已达到目标数量 {TOTAL_TARGET}，停止生成")
            break

        print(f"\n  [{ch_idx + 1}/{len(chapters)}] 第{chapter['chapter_num']}回《{chapter['chapter_title'][:15]}》...")

        chapter_qa = []
        for tmpl_idx in range(QA_PER_CHAPTER):
            qas = generate_qa_for_chapter(chapter, tmpl_idx)
            chapter_qa.extend(qas)

        if chapter_qa:
            all_qa.extend(chapter_qa)
            completed.add(chapter["chapter_num"])
            print(f"    ✓ 生成 {len(chapter_qa)} 条，累计 {len(all_qa)}/{TOTAL_TARGET}")

            # 实时保存进度（防崩溃丢失）
            progress["completed_chapters"] = list(completed)
            progress["generated_qa"] = all_qa
            save_progress(progress)

    elapsed = time.time() - start_time
    print(f"\n  ✓ 生成完成: 共 {len(all_qa)} 条, 用时 {elapsed / 60:.1f} 分钟")

    # 保存SFT数据集
    print("\n[4/5] 保存为 LLaMA-Factory 格式...")
    final_data = save_sft_dataset(all_qa)

    # 训练集/验证集分割
    print("\n[5/5] 划分训练集 / 验证集...")
    train_ratio = 0.9
    split_idx = int(len(final_data) * train_ratio)
    train_data = final_data[:split_idx]
    val_data = final_data[split_idx:]

    train_path = SFT_DIR / "honglou_sft_train.json"
    val_path = SFT_DIR / "honglou_sft_val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 训练集: {train_path.name} ({len(train_data)} 条)")
    print(f"  ✓ 验证集: {val_path.name} ({len(val_data)} 条)")

    # 数据预览
    print("\n  数据预览（前 2 条）:")
    for item in final_data[:2]:
        print(f"\n    Instruction: {item['instruction'][:80]}")
        print(f"    Output: {item['output'][:120]}...")

    print("\n" + "=" * 60)
    print(f"SFT 生成完成: {len(final_data)} 条 (训练 {len(train_data)} + 验证 {len(val_data)})")
    print("=" * 60)

    return final_data


if __name__ == "__main__":
    run_sft_generation()
