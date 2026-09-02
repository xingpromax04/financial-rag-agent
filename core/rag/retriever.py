"""财务报告混合检索器：向量语义检索 + BM25 + RRF 融合。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
本模块的 BM25 使用标准库实现，不强制引入 Elasticsearch 或 rank-bm25。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from core.rag.vector_store import (
    DocumentChunk,
    VectorSearchResult,
    VectorStoreProtocol,
)

Tokenizer: TypeAlias = Callable[[str], list[str]]
LOGGER = logging.getLogger(__name__)

# 数字 token 保留小数、千分位和百分号，避免 12.35% 被拆成无关词项。
_TOKEN_RE = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    r"|[A-Za-z]+(?:[._/-][A-Za-z0-9]+)*"
    r"|[\u4e00-\u9fff]+"
)
_QUERY_NUMBER_RE = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


class RetrievalError(RuntimeError):
    """混合检索或词法索引构建失败。"""


@dataclass(slots=True)
class KeywordSearchResult:
    """BM25 词法检索结果。"""

    chunk: DocumentChunk
    score: float


@dataclass(slots=True)
class HybridSearchResult:
    """混合检索结果及可解释的排名来源。"""

    chunk: DocumentChunk
    score: float
    semantic_rank: int | None
    keyword_rank: int | None
    semantic_score: float | None
    keyword_score: float | None


def financial_tokenize(text: str) -> list[str]:
    """针对中英文财报和精确数字的轻量分词。

    英文与数字作为完整 token；连续中文同时保留整词串和二元字组。
    二元字组可在不依赖专用分词词典时匹配“净利润”、“负债率”等术语。
    """

    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.casefold()):
        token = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.append(token)
            if len(token) > 1:
                tokens.extend(
                    token[index : index + 2] for index in range(len(token) - 1)
                )
        else:
            tokens.append(token.replace(",", ""))
    return tokens


class BM25Index:
    """Okapi BM25 内存索引。

    BM25 同时考虑词频、逆文档频率和文档长度。``k1`` 控制词频饱和，
    ``b`` 控制文档长度归一化强度。
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        tokenizer: Tokenizer = financial_tokenize,
        k1: float = 1.5,  # 🔧【可调参数】常用区间 1.2~2.0
        b: float = 0.75,  # 🔧【可调参数】0 表示不做长度归一化
    ) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 必须大于 0")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b 必须在 [0, 1] 内")
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        self._document_frequencies: Counter[str] = Counter()
        self._average_length = 0.0
        self._build()

    def _build(self) -> None:
        total_length = 0
        for chunk in self.chunks:
            tokens = self.tokenizer(chunk.text)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(tokens))
            self._document_frequencies.update(frequencies.keys())
            total_length += len(tokens)
        self._average_length = total_length / len(self.chunks) if self.chunks else 0.0

    def search(self, query: str, top_k: int = 10) -> list[KeywordSearchResult]:
        """返回 BM25 得分大于 0 的前 K 个切片。"""

        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        query_terms = self.tokenizer(query)
        if not query_terms or not self.chunks:
            return []

        scores = [
            self._score_document(index, query_terms)
            for index in range(len(self.chunks))
        ]
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], self.chunks[index].chunk_id),
        )
        return [
            KeywordSearchResult(chunk=self.chunks[index], score=scores[index])
            for index in ranked_indices[:top_k]
            if scores[index] > 0
        ]

    def _score_document(self, index: int, query_terms: list[str]) -> float:
        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (len(self.chunks) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_ratio = (
                document_length / self._average_length
                if self._average_length > 0
                else 0.0
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * length_ratio
            )
            score += inverse_document_frequency * (
                term_frequency * (self.k1 + 1) / denominator
            )
        return score


class HybridRetriever:
    """向量检索、BM25 与 RRF 得分融合器。"""

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        *,
        tokenizer: Tokenizer = financial_tokenize,
        semantic_weight: float = 1.0,  # 🔧【可调参数】语义召回权重
        keyword_weight: float = 1.0,  # 🔧【可调参数】精确词法召回权重
        rrf_k: int = 60,  # 🔧【可调参数】越大则前几名之间的分差越平缓
        exact_number_boost: float = 0.01,  # 🔧【可调参数】查询数字完整命中奖励
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("检索权重不能小于 0")
        if semantic_weight == 0 and keyword_weight == 0:
            raise ValueError("语义权重和关键词权重不能同时为 0")
        if rrf_k < 1:
            raise ValueError("rrf_k 必须大于等于 1")
        if exact_number_boost < 0:
            raise ValueError("exact_number_boost 不能小于 0")

        self.vector_store = vector_store
        self.tokenizer = tokenizer
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.exact_number_boost = exact_number_boost
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._bm25 = BM25Index(
            self.vector_store.all_chunks(),
            tokenizer=tokenizer,
            k1=bm25_k1,
            b=bm25_b,
        )

    def refresh(self) -> None:
        """向量库写入新切片后，重建内存 BM25 索引。"""

        chunks = self.vector_store.all_chunks()
        self._bm25 = BM25Index(
            chunks,
            tokenizer=self.tokenizer,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )
        LOGGER.info("BM25 index refreshed: chunks=%s", len(chunks))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,  # 🔧【可调参数】最终返回给 Agent 的上下文数
        semantic_k: int = 20,  # 🔧【可调参数】向量候选集大小
        keyword_k: int = 20,  # 🔧【可调参数】BM25 候选集大小
    ) -> list[HybridSearchResult]:
        """并行观念上执行两路召回，并使用 RRF 按排名融合。

        【原理说明】向量检索擅长识别“盈利能力”与“净资产收益率”的语义关联，
        但 Embedding 对证券代码、报表项目名和精确数字并不总是敏感。BM25 则可精确
        命中“600519”、“12.35%”等词项。两者融合可同时降低语义漏召回和数字漏召回。
        """

        if not query.strip():
            raise ValueError("检索问题不能为空")
        if min(top_k, semantic_k, keyword_k) < 1:
            raise ValueError("top_k、semantic_k 和 keyword_k 必须大于等于 1")

        errors: list[Exception] = []
        successful_paths = 0
        semantic_results: list[VectorSearchResult] = []
        keyword_results: list[KeywordSearchResult] = []
        if self.semantic_weight > 0:
            try:
                semantic_results = self.vector_store.search(query, top_k=semantic_k)
                successful_paths += 1
            except Exception as exc:
                errors.append(exc)
                LOGGER.warning("向量检索失败，本次降级为 BM25：%s", exc)
        if self.keyword_weight > 0:
            try:
                keyword_results = self._bm25.search(query, top_k=keyword_k)
                successful_paths += 1
            except Exception as exc:
                errors.append(exc)
                LOGGER.warning("BM25 检索失败，本次降级为向量检索：%s", exc)
        if errors and successful_paths == 0:
            raise RetrievalError("语义检索与关键词检索均未返回结果") from errors[0]

        fused = self._reciprocal_rank_fusion(query, semantic_results, keyword_results)
        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        query: str,
        semantic_results: Sequence[VectorSearchResult],
        keyword_results: Sequence[KeywordSearchResult],
    ) -> list[HybridSearchResult]:
        """使用排名而非原始分数融合，规避两类分数尺度不一致。"""

        records: dict[str, dict[str, object]] = {}
        for rank, result in enumerate(semantic_results, start=1):
            records[result.chunk.chunk_id] = {
                "chunk": result.chunk,
                "score": self.semantic_weight / (self.rrf_k + rank),
                "semantic_rank": rank,
                "keyword_rank": None,
                "semantic_score": result.score,
                "keyword_score": None,
            }

        for rank, result in enumerate(keyword_results, start=1):
            record = records.setdefault(
                result.chunk.chunk_id,
                {
                    "chunk": result.chunk,
                    "score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_score": None,
                    "keyword_score": None,
                },
            )
            record["score"] = float(record["score"]) + self.keyword_weight / (
                self.rrf_k + rank
            )
            record["keyword_rank"] = rank
            record["keyword_score"] = result.score

        query_numbers = set(_QUERY_NUMBER_RE.findall(query.replace(",", "")))
        output: list[HybridSearchResult] = []
        for record in records.values():
            chunk = record["chunk"]
            if not isinstance(chunk, DocumentChunk):
                continue
            normalized_text = chunk.text.replace(",", "")
            matched_numbers = sum(number in normalized_text for number in query_numbers)
            score = float(record["score"]) + matched_numbers * self.exact_number_boost
            output.append(
                HybridSearchResult(
                    chunk=chunk,
                    score=score,
                    semantic_rank=self._optional_int(record["semantic_rank"]),
                    keyword_rank=self._optional_int(record["keyword_rank"]),
                    semantic_score=self._optional_float(record["semantic_score"]),
                    keyword_score=self._optional_float(record["keyword_score"]),
                )
            )

        # 💡【面试加分点】RRF 只依赖各召回器的相对排名，不需要强行
        # 把“余弦相似度”与“BM25 得分”归一化到同一分布，因而对换
        # Embedding 模型或更换词法检索引擎更稳定。
        return sorted(
            output,
            key=lambda item: (
                -item.score,
                item.semantic_rank or 10**9,
                item.keyword_rank or 10**9,
                item.chunk.chunk_id,
            ),
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _optional_float(value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None


__all__ = [
    "BM25Index",
    "HybridRetriever",
    "HybridSearchResult",
    "KeywordSearchResult",
    "RetrievalError",
    "financial_tokenize",
]
