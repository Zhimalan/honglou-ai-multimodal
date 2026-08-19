# -*- coding: utf-8 -*-
"""Qwen2.5-7B Base + LoRA 适配器 融合推理脚本。

用于训练完成后验证微调效果，以及为方案C评测生成回答。

部署说明：
  - 本地 8GB 显存无法跑 7B 推理，请在云 GPU(A10+) 上运行，或把融合模型
    导出为 GGUF 后走 Ollama（ollama create）再推理。
  - 接入 Agent：融合模型导出 GGUF -> ollama create honglou_finetuned ->
    agent_graph/model_client.py 中 set_model("honglou_finetuned") 一行切换。

用法：
    python lora_train/lora_inference.py \
        --base D:/honglou_ai_multimodal/models/Qwen2.5-7B \
        --lora D:/honglou_ai_multimodal/lora_train/output/checkpoint-xxx \
        --question "林黛玉进贾府是哪一回？请简述经过。"
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load(base_path: str, lora_path: str | None, quantize: bool = False):
    """加载基座模型，可选挂载 LoRA 适配器。"""
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print(f">> 已挂载 LoRA: {lora_path}")
    else:
        print(">> 仅使用基座（未挂载 LoRA）")

    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """用统一指令模板生成回答（Base 无对话对齐，需显式指令包裹）。"""
    prompt = (
        "以下是关于《红楼梦》的问题，请严格依据原著内容作答，"
        "不要编造不存在的情节或人物。\n"
        f"问题：{question}\n回答："
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5-7B + LoRA 融合推理")
    ap.add_argument("--base", required=True, help="基座模型路径")
    ap.add_argument("--lora", default=None, help="LoRA 适配器路径（可选）")
    ap.add_argument(
        "--question",
        default="林黛玉进贾府是哪一回？请简述经过。",
        help="测试问题",
    )
    ap.add_argument("--quantize", action="store_true", help="4bit 量化加载基座")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    model, tokenizer = load(args.base, args.lora, args.quantize)
    print(">> 问题:", args.question)
    print(">> 回答:", generate(model, tokenizer, args.question, args.max_new_tokens))


if __name__ == "__main__":
    main()
