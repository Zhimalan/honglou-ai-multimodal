"""
HyDE 假设文档嵌入 (Hypothetical Document Embedding)
先用 LLM 生成假设性回答，再将回答作为查询向量去检索
提升跨章节复杂问题的召回精度
"""
import os
import sys
import time
from typing import Optional
import ollama


class HyDEGenerator:
    """HyDE 假设文档生成器"""

    SYSTEM_PROMPT = """你是一位红楼梦研究专家。请根据用户的问题，生成一段简短的红楼梦原文风格的假设性回答（150-200字）。
要求：
1. 用古典白话文风格写作
2. 包含相关人物名、地名、事件
3. 即使不完全确定答案，也要生成一段合理的假设文本
4. 直接输出回答内容，不要加任何前缀说明"""

    def __init__(self, model_name: str = "qwen2.5:7b", timeout: int = 60):
        self.model_name = model_name
        self.timeout = timeout
        print(f"[HyDE] 使用模型: {model_name}")

    def generate(self, query: str) -> Optional[str]:
        """
        生成假设文档

        Args:
            query: 用户查询

        Returns:
            假设性文档文本，失败时返回 None
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                options={
                    "temperature": 0.0,
                    "top_p": 0.85,
                    "num_predict": 300,
                },
            )
            text = response["message"]["content"].strip()
            if len(text) < 20:
                return None
            return text
        except Exception as e:
            print(f"  [HyDE] 生成失败: {e}")
            return None

    def generate_embedding_query(self, query: str, encoder) -> object:
        """
        生成 HyDE 增强的查询向量

        流程:
        1. 用 LLM 生成假设文档
        2. 对假设文档做嵌入（而非原始查询）
        3. 用假设文档的向量去检索

        如果 HyDE 失败，回退到原始查询嵌入

        Args:
            query: 用户查询
            encoder: BGE 编码器

        Returns:
            增强后的查询向量 (numpy array)
        """
        hyde_doc = self.generate(query)

        if hyde_doc:
            print(f"  [HyDE] 假设文档已生成 ({len(hyde_doc)}字)")
            # 用假设文档做嵌入
            return encoder.encode(hyde_doc)
        else:
            print(f"  [HyDE] 生成失败，回退到原始查询嵌入")
            return encoder.encode_query(query)


if __name__ == "__main__":
    gen = HyDEGenerator()
    doc = gen.generate("林黛玉初进贾府时见到了哪些人？")
    print(f"假设文档:\n{doc}")
