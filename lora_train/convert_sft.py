# -*- coding: utf-8 -*-
"""将阶段2生成的 SFT 数据按 9:1 划分为训练集/验证集（alpaca 格式）。

数据已经是 instruction/input/output 标准格式，此处仅做固定种子的随机划分并落盘，
供 LLaMA-Factory 直接消费。

用法：
    python lora_train/convert_sft.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "sft_dataset" / "honglou_sft.json"
OUT_DIR = ROOT / "lora_train" / "data"
TRAIN_RATIO = 0.9
SEED = 20260819


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    # 归一化：部分单轮问答缺 input 字段，统一补空串以符合 alpaca 格式
    for x in data:
        x.setdefault("input", "")
    random.seed(SEED)
    random.shuffle(data)

    n = len(data)
    k = int(n * TRAIN_RATIO)
    train, val = data[:k], data[k:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(
        train,
        open(OUT_DIR / "honglou_qa_train.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    json.dump(
        val,
        open(OUT_DIR / "honglou_qa_val.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    # 抽样校验字段完整性（alpaca 三件套）
    assert all(set(x.keys()) >= {"instruction", "input", "output"} for x in train + val)

    print(f"总计 {n} 条 -> 训练 {len(train)} / 验证 {len(val)}")
    print(f"训练集: {OUT_DIR / 'honglou_qa_train.json'}")
    print(f"验证集: {OUT_DIR / 'honglou_qa_val.json'}")
    print("格式校验通过：instruction/input/output 字段齐全")


if __name__ == "__main__":
    main()
