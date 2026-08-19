"""
红楼梦多模态智能解读专家系统 - Pydantic 数据模型定义
用于数据预处理阶段的结构化校验
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


class RedChapter(BaseModel):
    """红楼梦章回数据模型"""
    chapter_num: int = Field(..., ge=1, le=120, description="回目编号 1-120")
    chapter_title: str = Field(..., min_length=2, description="回目标题")
    raw_text: str = Field(..., min_length=100, description="该回原文")
    char_count: int = Field(default=0, description="该回字数")

    @field_validator("chapter_title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """回目标题不能为空且应包含中文"""
        if not re.search(r"[\u4e00-\u9fff]", v):
            raise ValueError("回目标题必须包含中文字符")
        return v.strip()

    @field_validator("raw_text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """原文不能太短（至少100字）"""
        if len(v.strip()) < 100:
            raise ValueError(f"原文过短: {len(v.strip())} 字，疑似数据缺失")
        return v.strip()


class Character(BaseModel):
    """红楼梦人物数据模型"""
    name: str = Field(..., min_length=1, description="人物姓名")
    description: str = Field(default="", description="人物描述")
    first_appearance: str = Field(default="", description="首次出场回目")
    notes: str = Field(default="", description="备注信息")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("人物姓名不能为空")
        return v


class RelationTriple(BaseModel):
    """人物关系三元组数据模型"""
    source: str = Field(..., description="人物A")
    target: str = Field(..., description="人物B")
    weight: float = Field(..., ge=0, description="交互权重")

    @field_validator("source", "target")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("人物名不能为空")
        return v


class TextChunk(BaseModel):
    """文本切片数据模型（供RAG使用）"""
    chunk_id: str = Field(..., description="切片ID，格式: ch{回号}_seg{序号}")
    chapter_num: int = Field(..., ge=1, le=120, description="所属回目")
    text: str = Field(..., min_length=50, description="切片文本")
    char_count: int = Field(default=0, description="切片字数")
    char_start: int = Field(default=0, description="在原文中的起始位置")
    char_end: int = Field(default=0, description="在原文中的结束位置")


class SFTSample(BaseModel):
    """SFT微调问答样本数据模型"""
    instruction: str = Field(..., description="问题/指令")
    input: str = Field(default="", description="上下文输入（可为空）")
    output: str = Field(..., min_length=5, description="期望回答")

    @field_validator("instruction", "output")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("内容不能为空")
        return v.strip()


if __name__ == "__main__":
    # 快速自测
    ch = RedChapter(
        chapter_num=1,
        chapter_title="甄士隐梦幻识通灵 贾雨村风尘怀闺秀",
        raw_text="此开卷第一回也。作者自云：因曾历过一番梦幻之后，" * 5,
    )
    print(f"RedChapter 校验通过: 第{ch.chapter_num}回, {ch.char_count}字")

    char = Character(name="贾宝玉", description="贾政次子", first_appearance="第二回")
    print(f"Character 校验通过: {char.name}")

    rel = RelationTriple(source="林黛玉", target="贾宝玉", weight=1020.0)
    print(f"RelationTriple 校验通过: {rel.source} -> {rel.target} ({rel.weight})")

    print("\n✓ 所有数据模型校验通过")
