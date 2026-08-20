#!/usr/bin/env python3
"""临时修复脚本：alpaca → sharegpt 格式转换 + dataset_info.json 更新"""
import json
import os

DATA_DIR = "/root/honglou-ai-multimodal/lora_train/data"

# 1. 转换训练集和验证集
for split in ["honglou_qa_train", "honglou_qa_val"]:
    path = os.path.join(DATA_DIR, f"{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sharegpt_data = []
    for item in data:
        conv = [
            {"from": "human", "value": item["instruction"]},
            {"from": "gpt", "value": item["output"]}
        ]
        # 如果有非空 input，拼到 human 消息里
        if item.get("input", "").strip():
            conv[0]["value"] = item["input"] + "\n" + item["instruction"]
        sharegpt_data.append({"conversations": conv})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    print(f"✅ {split}: {len(sharegpt_data)} 条已转换为 sharegpt 格式")

# 2. 更新 dataset_info.json
info = {
    "honglou_qa_train": {
        "file_name": "honglou_qa_train.json",
        "formatting": "sharegpt",
        "columns": {"conversations": "conversations"}
    },
    "honglou_qa_val": {
        "file_name": "honglou_qa_val.json",
        "formatting": "sharegpt",
        "columns": {"conversations": "conversations"}
    }
}
with open(os.path.join(DATA_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print("✅ dataset_info.json 已更新为 sharegpt 格式")

# 3. 验证
with open(os.path.join(DATA_DIR, "honglou_qa_train.json"), "r", encoding="utf-8") as f:
    sample = json.load(f)[0]
print(f"\n样本示例:\n{json.dumps(sample, ensure_ascii=False)[:300]}")
