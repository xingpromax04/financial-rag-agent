"""Embedding providers for financial report retrieval.

默认使用确定性 hash embedding，保证项目在没有外部模型或 API key 时仍可运行。
可通过环境变量切换到 DashScope 等真实 embedding provider。
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from typing import Any

from core.rag.retriever import financial_tokenize
from core.rag.vector_store import EmbeddingFunction

try:
    from dashscope import TextEmbedding
except ImportError:  # pragma: no cover - depends on optional cloud SDK
    TextEmbedding = None  # type: ignore[assignment]


DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"


class EmbeddingConfigurationError(RuntimeError):
    """Embedding provider 配置错误。"""


def hash_embedding(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    """生成无需下载模型的确定性特征哈希向量。

    生产环境可将此函数替换为 BGE / DashScope text-embedding 等语义模型；
    这里的默认实现优先保证离线可运行和向量维度稳定。
    """

    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in financial_tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % dimensions
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


def _extract_dashscope_vectors(response: Any) -> list[list[float]]:
    """从 DashScope embedding 响应中提取向量。"""

    if isinstance(response, dict):
        embeddings = response.get("output", {}).get("embeddings", [])
    else:
        output = getattr(response, "output", {}) or {}
        embeddings = output.get("embeddings", []) if isinstance(output, dict) else []

    vectors: list[list[float]] = []
    for item in embeddings:
        if isinstance(item, dict):
            vector = item.get("embedding")
        else:
            vector = getattr(item, "embedding", None)
        if not isinstance(vector, Sequence):
            raise EmbeddingConfigurationError("DashScope embedding 响应缺少 embedding 字段")
        vectors.append([float(value) for value in vector])
    return vectors


def _dashscope_embedding(texts: list[str]) -> list[list[float]]:
    """调用 DashScope TextEmbedding 生成向量。"""

    if TextEmbedding is None:
        raise EmbeddingConfigurationError(
            "缺少 dashscope SDK，请先执行 pip install dashscope"
        )

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EmbeddingConfigurationError("缺少 DASHSCOPE_API_KEY 环境变量")

    model = os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL",
        DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
    )
    response = TextEmbedding.call(
        model=model,
        input=texts,
        api_key=api_key,
    )
    vectors = _extract_dashscope_vectors(response)
    if len(vectors) != len(texts):
        raise EmbeddingConfigurationError("DashScope embedding 返回数量与输入文本数量不一致")
    return vectors


def build_embedding_function() -> EmbeddingFunction:
    """根据 EMBEDDING_PROVIDER 构建当前使用的 embedding function。"""

    provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().casefold()
    if provider == "hash":
        return hash_embedding
    if provider == "dashscope":
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise EmbeddingConfigurationError("缺少 DASHSCOPE_API_KEY 环境变量")
        return _dashscope_embedding
    raise EmbeddingConfigurationError(
        f"不支持的 EMBEDDING_PROVIDER：{provider}，可选值为 hash 或 dashscope"
    )


__all__ = [
    "DEFAULT_DASHSCOPE_EMBEDDING_MODEL",
    "EmbeddingConfigurationError",
    "build_embedding_function",
    "hash_embedding",
]
