"""
LLM 推理客户端 - 封装 Ollama 调用，支持模型热插拔。

当前使用 Ollama 原生 qwen2.5:7b (Q4_K_M 量化)。
当 QLoRA 微调完成后，只需调用 set_model() 切换到微调模型即可，
Agent 代码无需任何改动。

使用方式:
    client = ModelClient()
    response = client.generate("你好", system_prompt="你是红楼梦专家")
    client.set_model("qwen2.5:7b-lora", model_type="ollama_finetuned")
"""

import os
import sys
import time
import json
import ollama
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class ModelClient:
    """LLM 推理客户端 - 支持模型热插拔"""

    # 默认模型配置
    DEFAULT_MODEL = "qwen2.5:7b"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TIMEOUT = 120  # 秒

    def __init__(
        self,
        model_name: str = None,
        model_type: str = "ollama_base",
        temperature: float = None,
    ):
        """
        Args:
            model_name: Ollama 模型名称 (如 "qwen2.5:7b")
            model_type: 模型类型
                - "ollama_base": Ollama 原生基座模型
                - "ollama_finetuned": Ollama + LoRA 微调模型
            temperature: 生成温度
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model_type = model_type
        self.temperature = temperature or self.DEFAULT_TEMPERATURE
        self._available = None

    @property
    def is_available(self) -> bool:
        """检查 Ollama 服务和模型是否可用"""
        if self._available is None:
            try:
                models = ollama.list()
                model_names = [m.model for m in models.get("models", [])]
                self._available = self.model_name in model_names
                if not self._available:
                    print(f"  [model] 警告: 模型 {self.model_name} 未找到，可用模型: {model_names}")
            except Exception as e:
                print(f"  [model] Ollama 连接失败: {e}")
                self._available = False
        return self._available

    def set_model(self, model_name: str, model_type: str = "ollama_base"):
        """
        热切换模型。等 QLoRA 微调完成后调用此方法即可。

        Args:
            model_name: 新模型名称
            model_type: "ollama_base" 或 "ollama_finetuned"
        """
        print(f"  [model] 切换模型: {self.model_name} → {model_name} ({model_type})")
        self.model_name = model_name
        self.model_type = model_type
        self._available = None  # 重置缓存，下次调用时重新检查

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = None,
        max_tokens: int = 2048,
        timeout: int = None,
    ) -> str:
        """
        调用 LLM 生成回答。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 生成温度 (None 用默认值)
            max_tokens: 最大生成 token 数
            timeout: 超时秒数

        Returns:
            LLM 生成的文本
        """
        temp = temperature if temperature is not None else self.temperature
        timeout = timeout or self.DEFAULT_TIMEOUT

        if not self.is_available:
            return f"[模型不可用] Ollama 服务未运行或模型 {self.model_name} 未安装。"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": max_tokens,
                },
            )
            elapsed = time.time() - start
            result = response["message"]["content"]
            print(f"  [llm] 生成完成 ({elapsed:.1f}s, {len(result)} 字)")
            return result
        except Exception as e:
            return f"[生成失败] {e}"

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        timeout: int = 60,
    ) -> dict:
        """
        调用 LLM 生成 JSON 格式输出。
        用于意图分类等需要结构化输出的场景。

        Returns:
            解析后的 dict，解析失败时返回 {"error": "..."}
        """
        temp = temperature
        timeout = timeout or self.DEFAULT_TIMEOUT

        if not self.is_available:
            return {"error": "模型不可用"}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": 512,
                    "format": "json",
                },
            )
            elapsed = time.time() - start
            content = response["message"]["content"]
            result = json.loads(content)
            print(f"  [llm] JSON 生成完成 ({elapsed:.1f}s)")
            return result
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            try:
                import re
                match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                pass
            return {"error": f"JSON 解析失败: {content[:200]}"}
        except Exception as e:
            return {"error": str(e)}


# 全局单例
_client_instance: Optional[ModelClient] = None


def get_model_client() -> ModelClient:
    """获取全局 ModelClient 单例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = ModelClient()
    return _client_instance


def set_model(model_name: str, model_type: str = "ollama_base"):
    """全局切换模型（供外部调用，如评测脚本切换方案A/B/C）"""
    client = get_model_client()
    client.set_model(model_name, model_type)
