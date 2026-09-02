"""财报混合检索 Agent：召回证据并构造下游可计算数据。

运行环境：Python 3.11，Conda 环境 ``rag_311``。

输入：``AgentState.question``、向量库/BM25 中的已入库财报切片。
输出：``retrieved_context``、``financial_data`` 和工具消息的状态增量。
数据流：用户问题 -> HybridRetriever -> 文本/表格证据 -> 期间化指标字典。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from types import TracebackType
from typing import Any, Protocol

from agents.state import AgentState
from core.rag.retriever import HybridRetriever, HybridSearchResult

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


class LockProtocol(Protocol):
    """检索器与注册表并发访问所需的最小锁接口。"""

    def __enter__(self) -> Any: ...

    def __exit__(
        self,
        t: type[BaseException] | None,
        v: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...



class RAGAgentError(RuntimeError):
    """财报召回或结构化转换失败。"""


class RAGAgent:
    """将 ``HybridRetriever`` 包装成可直接注册到 LangGraph 的节点。

    ``document_registry`` 保存 Parser 阶段得到的结构化表格数据。召回完成后，
    Agent 只合并命中文档的数据，避免把未命中文档的指标污染本次上下文。
    对没有注册表数据的表格切片，再使用保守的 Markdown 表格解析作降级。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        document_registry: Mapping[str, Mapping[str, Any]] | None = None,
        lock: LockProtocol | None = None,
        top_k: int = 8,  # 🔧【可调参数】传入下游 Agent 的最终证据条数
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        self.retriever = retriever
        self.document_registry = (
            document_registry if document_registry is not None else {}
        )
        self.lock = lock
        self.top_k = top_k

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """读取问题并返回 AgentState 的增量，而不是原地修改共享状态。

        LangGraph 会依据 ``AgentState`` 中的 reducer 合并返回值。节点保持
        “输入状态、输出增量”的边界后，可独立测试，也能安全重试或回放。
        """

        question = str(state.get("question", "")).strip()
        if not question:
            raise RAGAgentError("RAG Agent 缺少用户问题")

        context_manager = self.lock if self.lock is not None else nullcontext()
        with context_manager:
            results = self.retriever.retrieve(question, top_k=self.top_k)
            financial_data = self._build_financial_data(results)
        if not results:
            raise RAGAgentError("财报库中没有可用切片，请先上传 PDF 或 DOCX")

        contexts = [self._to_context(result) for result in results]
        return {
            "retrieved_context": contexts,
            "financial_data": financial_data,
            "messages": [
                {
                    "role": "tool",
                    "name": "rag_agent",
                    "content": (
                        f"召回 {len(contexts)} 条财报证据，提取 "
                        f"{len(financial_data)} 个期间的数据。"
                    ),
                }
            ],
        }

    @staticmethod
    def _to_context(result: HybridSearchResult) -> dict[str, Any]:
        """保留融合得分和两路排名，便于 API 展示检索可解释性。"""

        return {
            "chunk_id": result.chunk.chunk_id,
            "text": result.chunk.text,
            "metadata": dict(result.chunk.metadata),
            "score": result.score,
            "semantic_rank": result.semantic_rank,
            "keyword_rank": result.keyword_rank,
            "semantic_score": result.semantic_score,
            "keyword_score": result.keyword_score,
        }

    def _build_financial_data(
        self,
        results: list[HybridSearchResult],
    ) -> dict[str, Any]:
        """将半结构化表格切片转换为 ``期间 -> 指标 -> 数值``。

        Parser 注册表是首选来源，因为它保留了 DataFrame 的完整行列关系；
        Markdown 解析只用于兼容旧数据或外部写入的向量切片。结构化注册表
        最后合并，因此同名字段以 Parser 的确定性结果为准。
        """

        structured: dict[str, Any] = {}
        for result in results:
            inferred = self._parse_markdown_table(
                result.chunk.text,
                result.chunk.metadata,
            )
            self._deep_merge(structured, inferred)

        sources = {
            str(result.chunk.metadata.get("source"))
            for result in results
            if result.chunk.metadata.get("source")
        }
        document_ids = {
            str(result.chunk.metadata.get("document_id"))
            for result in results
            if result.chunk.metadata.get("document_id")
        }
        for source, document in self.document_registry.items():
            document_id = str(document.get("document_id", ""))
            if source not in sources and document_id not in document_ids:
                continue
            registered_data = document.get("financial_data", {})
            if isinstance(registered_data, Mapping):
                self._deep_merge(structured, registered_data)
        return structured

    @classmethod
    def _parse_markdown_table(
        cls,
        text: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, dict[str, float]]:
        """保守解析表格：无法确认列与数值关系时不产生计算输入。"""

        lines = [line.strip() for line in text.splitlines() if "|" in line]
        if len(lines) < 3:
            return {}
        rows = [cls._split_markdown_row(line) for line in lines]
        if len(rows[0]) < 2 or not all(
            _SEPARATOR_CELL_RE.fullmatch(cell) for cell in rows[1]
        ):
            return {}

        headers = rows[0]
        output: dict[str, dict[str, float]] = {}
        for row in rows[2:]:
            if len(row) != len(headers):
                continue
            metric = row[0].strip()
            if not metric:
                continue
            for column, raw_value in zip(headers[1:], row[1:], strict=True):
                value = cls._parse_number(raw_value)
                if value is None:
                    continue
                year_match = _YEAR_RE.search(column)
                fallback_year = metadata.get("year")
                period = year_match.group(0) if year_match else str(
                    fallback_year or column or "current"
                )
                output.setdefault(period, {})[metric] = value
        return output

    @staticmethod
    def _split_markdown_row(line: str) -> list[str]:
        return [
            cell.strip().replace("\\|", "|")
            for cell in line.strip().strip("|").split("|")
        ]

    @staticmethod
    def _parse_number(value: Any) -> float | None:
        text = str(value).strip().replace(",", "").replace(" ", "")
        if text.casefold() in {"", "-", "--", "none", "nan", "n/a"}:
            return None
        negative = text.startswith("(") and text.endswith(")")
        percent = text.endswith("%")
        try:
            number = float(text.strip("()%"))
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        if negative:
            number = -number
        return number / 100 if percent else number

    @classmethod
    def _deep_merge(
        cls,
        target: dict[str, Any],
        incoming: Mapping[str, Any],
    ) -> None:
        """递归合并期间/报表层级，并复制叶子数据避免共享可变对象。"""

        for key, value in incoming.items():
            normalized_key = str(key)
            current = target.get(normalized_key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                cls._deep_merge(current, value)
            elif isinstance(value, Mapping):
                nested: dict[str, Any] = {}
                cls._deep_merge(nested, value)
                target[normalized_key] = nested
            else:
                target[normalized_key] = deepcopy(value)

        # 💡【面试加分点】Embedding 召回的是“相关证据”，不是可直接做
        # 除法的变量。RAG Agent 在节点边界完成证据溯源、期间对齐和数值化，
        # Calc Agent 因而只消费稳定 Schema，避免再次猜测表头与数字的对应关系。


__all__ = ["RAGAgent", "RAGAgentError"]
