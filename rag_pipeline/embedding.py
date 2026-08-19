"""
BGE 中文嵌入模型封装
模型: BAAI/bge-small-zh-v1.5 (512维, 适合CPU推理)
使用 sentence-transformers 加载，兼容性更好
"""
import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class BGEEncoder:
    """BGE 中文嵌入模型封装"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        cache_dir: str = None,
    ):
        """
        初始化 BGE 嵌入模型

        Args:
            model_name: HuggingFace 模型名
            cache_dir: 模型缓存目录
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        print(f"[BGE] 正在加载嵌入模型: {model_name} ...")
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[BGE] 模型加载完成, 输出维度: {self.dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            texts: 单条文本或文本列表
            batch_size: 批处理大小
            normalize: 是否L2归一化（用于余弦相似度）

        Returns:
            numpy 数组, shape=(n, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询文本（BGE推荐查询加前缀 "为这个句子生成表示以用于检索相关文章："）
        """
        query_prefix = "为这个句子生成表示以用于检索相关文章："
        return self.encode(query_prefix + query)


if __name__ == "__main__":
    encoder = BGEEncoder()

    # 测试编码
    texts = ["林黛玉进贾府", "贾宝玉梦游太虚幻境"]
    emb = encoder.encode(texts)
    print(f"编码结果 shape: {emb.shape}")

    # 测试相似度
    q_emb = encoder.encode_query("黛玉初进荣国府见到了谁？")
    scores = np.dot(emb, q_emb.T).flatten()
    print(f"相似度得分: {scores}")
