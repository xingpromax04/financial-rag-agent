"""财务报告切片与 ChromaDB / FAISS 向量存储。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
可选依赖：``chromadb``、``faiss-cpu``、``numpy``、``pandas``。

数据流向：
    Word/PDF 解析结果 -> Markdown/表格感知切片 -> 文本 + Metadata
                       -> Embedding -> ChromaDB 或 FAISS -> 相似度检索
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, TypeAlias

import pandas as pd

PathLike: TypeAlias = str | Path
Backend: TypeAlias = Literal["chroma", "faiss"]
MetadataValue: TypeAlias = str | int | float | bool
EmbeddingFunction: TypeAlias = Callable[[list[str]], Sequence[Sequence[float]]]

LOGGER = logging.getLogger(__name__)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?;；])")


class VectorStoreError(RuntimeError):
    """切片或向量存储操作失败。"""


class DependencyUnavailableError(VectorStoreError):
    """当前向量存储后端缺少必要依赖。"""


@dataclass(slots=True)
class DocumentChunk:
    """可写入向量库的最小文档单元。"""

    chunk_id: str
    text: str
    metadata: dict[str, MetadataValue]


@dataclass(slots=True)
class VectorSearchResult:
    """向量检索命中结果，``score`` 越大表示越相似。"""

    chunk: DocumentChunk
    score: float


class VectorStoreProtocol(Protocol):
    """混合检索器依赖的最小向量库契约。"""

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None: ...

    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]: ...

    def all_chunks(self) -> list[DocumentChunk]: ...


def _load_module(name: str, install_name: str | None = None) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        package = install_name or name
        raise DependencyUnavailableError(
            f"缺少 {package}，请在 rag_311 环境执行 pip install {package}"
        ) from exc


def _clean_text(text: str) -> str:
    """规范化空白，但保留 Markdown 的换行结构。"""

    lines = [
        " ".join(line.split())
        for line in text.replace("\u00a0", " ").splitlines()
    ]
    output: list[str] = []
    for line in lines:
        if line or (output and output[-1]):
            output.append(line)
    return "\n".join(output).strip()


def _stable_chunk_id(source: str, index: int, text: str) -> str:
    payload = f"{source}\0{index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
    """将复合元数据转成 ChromaDB 可接受的标量类型。"""

    normalized: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = json.dumps(
                value, ensure_ascii=False, default=str, sort_keys=True
            )
    return normalized


class FinancialChunker:
    """Markdown 与表格感知的财报切片器。

    ``chunk_size`` 是每个切片的目标字符数，过小会拆散指标与上下文，
    过大会稀释 Embedding 中的关键语义并增加 LLM 输入成本。
    ``overlap`` 让相邻切片共享边界上下文，能减少关键句恰好被切断的情况，
    但过大会制造重复召回并增加向量库体积。
    """

    def __init__(
        self,
        chunk_size: int = 1200,  # 🔧【可调参数】中文财报推荐 800~1600 字符
        overlap: int = 150,  # 🔧【可调参数】通常取 chunk_size 的 10%~20%
        table_rows_per_chunk: int = 20,  # 🔧【可调参数】表头会在每个切片中重复
    ) -> None:
        if chunk_size < 200:
            raise ValueError("chunk_size 不能小于 200")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须在 [0, chunk_size) 内")
        if table_rows_per_chunk < 1:
            raise ValueError("table_rows_per_chunk 必须大于等于 1")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.table_rows_per_chunk = table_rows_per_chunk

    def chunk_markdown(
        self,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """按标题、段落和 Markdown 表格边界切片。

        【原理说明】财报不能直接每 N 个字符切一刀。这会把标题与正文、
        表头与数值行分开，使“2025 年净利润”在召回时只剩下数字而无口径。
        本实现先识别语义块，仅在单个语义块过长时才进行滑窗切分。
        """

        cleaned = _clean_text(text)
        if not cleaned:
            return []

        blocks = self._markdown_blocks(cleaned)
        pieces = self._merge_blocks(blocks)
        base_metadata = _scalar_metadata(metadata or {})
        chunks: list[DocumentChunk] = []
        for index, piece in enumerate(pieces):
            chunk_metadata = {
                **base_metadata,
                "source": source,
                "chunk_index": index,
                "chunk_type": "text",
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=_stable_chunk_id(source, index, piece),
                    text=piece,
                    metadata=chunk_metadata,
                )
            )
        return chunks

    def chunk_table(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        table_name: str,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """按整行切分 DataFrame，每块都保留表名和表头。"""

        if frame.empty and not list(frame.columns):
            return []
        columns = [str(column) for column in frame.columns]
        rows = [
            [self._display_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ]
        if not rows:
            rows = [[""] * len(columns)]

        base_metadata = _scalar_metadata(metadata or {})
        chunks: list[DocumentChunk] = []
        for offset in range(0, len(rows), self.table_rows_per_chunk):
            row_group = rows[offset : offset + self.table_rows_per_chunk]
            markdown = self._table_to_markdown(columns, row_group)
            text = f"### {table_name}\n\n{markdown}"
            index = start_index + len(chunks)
            chunk_metadata = {
                **base_metadata,
                "source": source,
                "table_name": table_name,
                "chunk_index": index,
                "chunk_type": "table",
                "row_start": offset,
                "row_end": offset + len(row_group) - 1,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=_stable_chunk_id(source, index, text),
                    text=text,
                    metadata=chunk_metadata,
                )
            )

        # 💡【面试加分点】表格切片以“整行”为原子单位，并重复表头。
        # 这使每个向量都包含指标名、期间和数值的完整对应关系，
        # 后续 Agent 还可根据 row_start/row_end 回溯原表。
        return chunks

    def chunk_parsed_document(
        self,
        parsed: Any,
        *,
        source: str,
        year: int | None = None,
        table_names: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """将 Phase 1 的 Word/PDF 解析结果转成可入库切片。"""

        common_metadata = dict(metadata or {})
        if year is not None:
            common_metadata["year"] = year

        text = self._parsed_text_to_markdown(parsed)
        chunks = self.chunk_markdown(
            text,
            source=source,
            metadata=common_metadata,
        )

        tables = list(getattr(parsed, "tables", []) or [])
        names = list(table_names or [])
        for table_index, frame in enumerate(tables):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"第 {table_index} 个表格不是 pandas.DataFrame")
            heading_path = frame.attrs.get("heading_path", [])
            default_name = (
                " / ".join(map(str, heading_path)) or f"table_{table_index + 1}"
            )
            table_name = (
                names[table_index] if table_index < len(names) else default_name
            )
            table_metadata = {
                **common_metadata,
                "table_index": table_index,
                "source_pages": frame.attrs.get("source_pages", []),
                "heading_path": heading_path,
            }
            chunks.extend(
                self.chunk_table(
                    frame,
                    source=source,
                    table_name=table_name,
                    metadata=table_metadata,
                    start_index=len(chunks),
                )
            )
        return chunks

    def _parsed_text_to_markdown(self, parsed: Any) -> str:
        paragraphs = getattr(parsed, "paragraphs", None)
        if paragraphs is not None:
            lines: list[str] = []
            for record in paragraphs:
                if isinstance(record, dict):
                    text = str(record.get("text", "")).strip()
                    level = record.get("heading_level")
                    if text and isinstance(level, int):
                        lines.append(f"{'#' * min(max(level, 1), 6)} {text}")
                    elif text:
                        lines.append(text)
                elif str(record).strip():
                    lines.append(str(record).strip())
            return "\n\n".join(lines)

        pages = getattr(parsed, "pages", None)
        if pages is not None:
            return "\n\n".join(
                f"## Page {index}\n\n{page}"
                for index, page in enumerate(pages, start=1)
                if str(page).strip()
            )
        if hasattr(parsed, "text"):
            return str(parsed.text)
        raise TypeError("无法从输入对象中读取 paragraphs、pages 或 text")

    def _markdown_blocks(self, text: str) -> list[str]:
        lines = text.splitlines()
        blocks: list[str] = []
        paragraph: list[str] = []
        index = 0

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append("\n".join(paragraph).strip())
                paragraph.clear()

        while index < len(lines):
            line = lines[index]
            if not line:
                flush_paragraph()
                index += 1
                continue
            if _HEADING_RE.match(line):
                flush_paragraph()
                blocks.append(line)
                index += 1
                continue
            if "|" in line and index + 1 < len(lines):
                next_line = lines[index + 1]
                if "|" in next_line and _TABLE_SEPARATOR_RE.match(next_line):
                    flush_paragraph()
                    table_lines = [line, next_line]
                    index += 2
                    while index < len(lines) and "|" in lines[index]:
                        table_lines.append(lines[index])
                        index += 1
                    blocks.extend(self._split_markdown_table(table_lines))
                    continue
            paragraph.append(line)
            index += 1
        flush_paragraph()
        return [block for block in blocks if block]

    def _split_markdown_table(self, lines: list[str]) -> list[str]:
        header = lines[:2]
        data_rows = lines[2:]
        if not data_rows:
            return ["\n".join(lines)]
        return [
            "\n".join(
                [
                    *header,
                    *data_rows[offset : offset + self.table_rows_per_chunk],
                ]
            )
            for offset in range(0, len(data_rows), self.table_rows_per_chunk)
        ]

    def _merge_blocks(self, blocks: list[str]) -> list[str]:
        expanded: list[str] = []
        for block in blocks:
            expanded.extend(self._split_long_block(block))

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for block in expanded:
            separator_size = 2 if current else 0
            if current and current_size + separator_size + len(block) > self.chunk_size:
                chunks.append("\n\n".join(current))
                current = self._overlap_blocks(current)
                current_size = sum(len(item) for item in current)
                current_size += max(len(current) - 1, 0) * 2
                separator_size = 2 if current else 0
            current.append(block)
            current_size += separator_size + len(block)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _split_long_block(self, block: str) -> list[str]:
        if len(block) <= self.chunk_size:
            return [block]
        lines = block.splitlines()
        if (
            len(lines) >= 2
            and "|" in lines[0]
            and _TABLE_SEPARATOR_RE.match(lines[1])
        ):
            return self._split_large_table(lines)
        sentences = [item for item in _SENTENCE_BOUNDARY_RE.split(block) if item]
        if len(sentences) == 1:
            step = self.chunk_size - self.overlap
            return [
                block[start : start + self.chunk_size]
                for start in range(0, len(block), step)
            ]

        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > self.chunk_size:
                pieces.append(current)
                current = current[-self.overlap :] if self.overlap else ""
            current += sentence
        if current:
            pieces.append(current)
        return pieces

    def _split_large_table(self, lines: list[str]) -> list[str]:
        """在不拆断单元格行的前提下，按目标字符数切分大表。"""

        header = lines[:2]
        header_text = "\n".join(header)
        pieces: list[str] = []
        current_rows: list[str] = []
        current_size = len(header_text)
        for row in lines[2:]:
            if current_rows and current_size + len(row) + 1 > self.chunk_size:
                pieces.append("\n".join([*header, *current_rows]))
                current_rows = []
                current_size = len(header_text)
            current_rows.append(row)
            current_size += len(row) + 1
        if current_rows:
            pieces.append("\n".join([*header, *current_rows]))
        return pieces or [header_text]

    def _overlap_blocks(self, blocks: list[str]) -> list[str]:
        if self.overlap == 0:
            return []
        selected: list[str] = []
        size = 0
        for block in reversed(blocks):
            if size + len(block) > self.overlap:
                break
            selected.append(block)
            size += len(block)
        return list(reversed(selected))

    @staticmethod
    def _display_value(value: Any) -> str:
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    @staticmethod
    def _table_to_markdown(columns: list[str], rows: list[list[str]]) -> str:
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header, separator, *body])


class FinancialVectorStore:
    """ChromaDB / FAISS 统一向量存储接口。"""

    def __init__(
        self,
        *,
        backend: Backend = "chroma",  # 🔧【可调参数】chroma 便于持久化，FAISS 更轻量
        collection_name: str = "financial_reports",
        persist_directory: PathLike = ".rag_store",
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        if backend not in {"chroma", "faiss"}:
            raise ValueError(f"不支持的向量库后端：{backend}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,62}", collection_name):
            raise ValueError("collection_name 需为 3~63 位字母、数字、点、下划线或连字符")

        self.backend = backend
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_function
        self._collection: Any = None
        self._faiss_index: Any = None
        self._faiss_chunks: list[DocumentChunk] = []

        if backend == "chroma":
            self._initialize_chroma()
        else:
            self._initialize_faiss()

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """批量写入切片；Chroma 使用 upsert，FAISS 跳过已存在 ID。"""

        valid_chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not valid_chunks:
            return
        if self.backend == "chroma":
            self._add_to_chroma(valid_chunks)
        else:
            self._add_to_faiss(valid_chunks)
        LOGGER.info(
            "vector store write complete: backend=%s, chunks=%s",
            self.backend,
            len(valid_chunks),
        )

    def ingest_parsed_document(
        self,
        parsed: Any,
        *,
        source: str,
        year: int | None = None,
        table_names: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        chunker: FinancialChunker | None = None,
    ) -> list[DocumentChunk]:
        """切分 Phase 1 解析结果、批量入库，并返回实际写入的切片。"""

        active_chunker = chunker or FinancialChunker()
        chunks = active_chunker.chunk_parsed_document(
            parsed,
            source=source,
            year=year,
            table_names=table_names,
            metadata=metadata,
        )
        self.add_chunks(chunks)
        return chunks

    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        """执行余弦相似度检索。"""

        if not query.strip():
            raise ValueError("检索问题不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        if self.backend == "chroma":
            return self._search_chroma(query, top_k)
        return self._search_faiss(query, top_k)

    def all_chunks(self) -> list[DocumentChunk]:
        """返回全部切片，主要用于构建 BM25 词法索引。"""

        if self.backend == "faiss":
            return list(self._faiss_chunks)
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        return [
            DocumentChunk(
                chunk_id=str(chunk_id),
                text=str(document),
                metadata=_scalar_metadata(metadata or {}),
            )
            for chunk_id, document, metadata in zip(
                ids, documents, metadatas, strict=True
            )
        ]

    def _initialize_chroma(self) -> None:
        chromadb = _load_module("chromadb")
        try:
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError("ChromaDB 初始化失败") from exc

    def _initialize_faiss(self) -> None:
        if self.embedding_function is None:
            raise ValueError("FAISS 后端必须传入 embedding_function")
        _load_module("faiss", "faiss-cpu")
        _load_module("numpy")
        self._load_faiss_files()

    def _add_to_chroma(self, chunks: Sequence[DocumentChunk]) -> None:
        documents = [chunk.text for chunk in chunks]
        payload: dict[str, Any] = {
            "ids": [chunk.chunk_id for chunk in chunks],
            "documents": documents,
            "metadatas": [_scalar_metadata(chunk.metadata) for chunk in chunks],
        }
        if self.embedding_function is not None:
            payload["embeddings"] = self._embed(documents)
        try:
            self._collection.upsert(**payload)
        except Exception as exc:
            raise VectorStoreError("ChromaDB 批量写入失败") from exc

    def _search_chroma(self, query: str, top_k: int) -> list[VectorSearchResult]:
        available = int(self._collection.count())
        if available == 0:
            return []
        payload: dict[str, Any] = {
            "n_results": min(top_k, available),
            "include": ["documents", "metadatas", "distances"],
        }
        if self.embedding_function is None:
            payload["query_texts"] = [query]
        else:
            payload["query_embeddings"] = self._embed([query])
        try:
            result = self._collection.query(**payload)
            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
        except Exception as exc:
            raise VectorStoreError("ChromaDB 检索失败") from exc

        return [
            VectorSearchResult(
                chunk=DocumentChunk(
                    chunk_id=str(chunk_id),
                    text=str(document),
                    metadata=_scalar_metadata(metadata or {}),
                ),
                score=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def _add_to_faiss(self, chunks: Sequence[DocumentChunk]) -> None:
        faiss = _load_module("faiss", "faiss-cpu")
        numpy = _load_module("numpy")
        existing_ids = {chunk.chunk_id for chunk in self._faiss_chunks}
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if not new_chunks:
            return

        vectors = numpy.asarray(
            self._embed([chunk.text for chunk in new_chunks]), dtype="float32"
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(new_chunks):
            raise VectorStoreError("embedding_function 返回的向量形状不正确")
        faiss.normalize_L2(vectors)
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
        if self._faiss_index.d != vectors.shape[1]:
            raise VectorStoreError("新向量维度与现有 FAISS 索引不一致")
        self._faiss_index.add(vectors)
        self._faiss_chunks.extend(new_chunks)
        self._persist_faiss_files()

    def _search_faiss(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self._faiss_index is None or not self._faiss_chunks:
            return []
        faiss = _load_module("faiss", "faiss-cpu")
        numpy = _load_module("numpy")
        vector = numpy.asarray(self._embed([query]), dtype="float32")
        if vector.ndim != 2 or vector.shape != (1, self._faiss_index.d):
            raise VectorStoreError("查询向量维度与 FAISS 索引不一致")
        faiss.normalize_L2(vector)
        scores, indices = self._faiss_index.search(
            vector, min(top_k, len(self._faiss_chunks))
        )
        return [
            VectorSearchResult(
                chunk=self._faiss_chunks[int(index)],
                score=float(score),
            )
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_function is None:
            raise VectorStoreError("当前操作需要 embedding_function")
        try:
            vectors = self.embedding_function(texts)
            return [list(map(float, vector)) for vector in vectors]
        except Exception as exc:
            raise VectorStoreError("Embedding 计算失败") from exc

    @property
    def _faiss_index_path(self) -> Path:
        return self.persist_directory / f"{self.collection_name}.faiss"

    @property
    def _faiss_metadata_path(self) -> Path:
        return self.persist_directory / f"{self.collection_name}.chunks.json"

    def _load_faiss_files(self) -> None:
        index_exists = self._faiss_index_path.is_file()
        metadata_exists = self._faiss_metadata_path.is_file()
        if index_exists != metadata_exists:
            raise VectorStoreError("FAISS 索引与元数据文件不完整")
        if not index_exists:
            return

        faiss = _load_module("faiss", "faiss-cpu")
        try:
            self._faiss_index = faiss.read_index(str(self._faiss_index_path))
            payload = json.loads(self._faiss_metadata_path.read_text(encoding="utf-8"))
            self._faiss_chunks = [DocumentChunk(**item) for item in payload]
        except Exception as exc:
            raise VectorStoreError("FAISS 持久化文件加载失败") from exc
        if self._faiss_index.ntotal != len(self._faiss_chunks):
            raise VectorStoreError("FAISS 向量数与元数据条数不一致")

    def _persist_faiss_files(self) -> None:
        faiss = _load_module("faiss", "faiss-cpu")
        temporary_index = self._faiss_index_path.with_suffix(".faiss.tmp")
        temporary_metadata = self._faiss_metadata_path.with_suffix(".json.tmp")
        try:
            faiss.write_index(self._faiss_index, str(temporary_index))
            temporary_metadata.write_text(
                json.dumps(
                    [asdict(chunk) for chunk in self._faiss_chunks],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_index.replace(self._faiss_index_path)
            temporary_metadata.replace(self._faiss_metadata_path)
        except Exception as exc:
            temporary_index.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            raise VectorStoreError("FAISS 持久化失败") from exc


VectorStore = FinancialVectorStore

__all__ = [
    "DocumentChunk",
    "FinancialChunker",
    "FinancialVectorStore",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreProtocol",
]
