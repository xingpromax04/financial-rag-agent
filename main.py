"""FastAPI 服务入口：财报上传、RAG 入库与多 Agent 分析。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
启动命令：``python main.py``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.audit_agent import AuditAgent
from agents.calc_agent import CalcAgent
from agents.master_agent import MasterAgent
from agents.rag_agent import RAGAgent
from agents.real_time_agent import RealTimeAgent
from agents.state import create_initial_state
from core.parsers.pdf_parser import PDFParseResult, parse_pdf
from core.parsers.word_parser import WordParseResult, parse_word
from core.llm.finalizer import build_finalizer
from core.rag.embeddings import build_embedding_function, hash_embedding
from core.rag.retriever import HybridRetriever
from core.rag.vector_store import (
    DependencyUnavailableError,
    DocumentChunk,
    FinancialChunker,
    FinancialVectorStore,
    VectorSearchResult,
    VectorStoreError,
    VectorStoreProtocol,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads")).resolve()
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", BASE_DIR / "data" / "vectors")).resolve()
REGISTRY_PATH = UPLOAD_DIR / "financial_registry.json"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


class UploadResponse(BaseModel):
    """``POST /upload`` 的结构化响应。"""

    document_id: str
    filename: str
    file_type: str
    chunks_written: int
    tables_found: int
    text_units: int
    vector_backend: str


class AnalyzeRequest(BaseModel):
    """``POST /analyze`` 请求体。"""

    question: str = Field(min_length=1, max_length=4_000)
    mode: Literal["fast", "deep"] = "fast"
    symbol: str | None = Field(default=None, max_length=32)
    calculation_code: str | None = Field(default=None, max_length=20_000)
    financial_data: dict[str, Any] | None = None


class AnalyzeResponse(BaseModel):
    """Agent 工作流的可审计输出，不包含隐式思维链。"""

    trace_id: str
    answer: str
    route: str
    workflow_trace: list[str]
    retrieved_context: list[dict[str, Any]]
    market_data: dict[str, Any]
    calculation_code: str | None
    code_output: dict[str, Any] | None
    audit_results: list[dict[str, Any]]
    errors: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    vector_backend: str
    indexed_chunks: int
    registered_documents: int

class _MemoryVectorStore(VectorStoreProtocol):
    """开发环境回退向量库；重启后数据不保留。"""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks.values())

    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        query_vector = hash_embedding([query])[0]
        scored: list[VectorSearchResult] = []
        for chunk in self._chunks.values():
            chunk_vector = hash_embedding([chunk.text])[0]
            score = sum(
                left * right
                for left, right in zip(query_vector, chunk_vector, strict=True)
            )
            scored.append(VectorSearchResult(chunk=chunk, score=score))
        return sorted(scored, key=lambda item: -item.score)[:top_k]


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "report").name.strip()
    return re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name) or "report"


def _parse_number(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().replace(",", "").replace(" ", "")
    if text.casefold() in {"", "-", "--", "none", "nan", "n/a"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    percent = text.endswith("%")
    try:
        number = float(text.strip("()%"))
    except ValueError:
        return None
    if negative:
        number = -number
    return number / 100 if percent else number


def _extract_financial_data(
    tables: Sequence[pd.DataFrame],
    fallback_year: int | None,
) -> dict[str, dict[str, float]]:
    """将“项目 + 期间列”财务表转成审计 Agent 可读的期间字典。"""

    periods: dict[str, dict[str, float]] = {}
    for frame in tables:
        if frame.empty or len(frame.columns) < 2:
            continue
        label_column = frame.columns[0]
        for value_column in frame.columns[1:]:
            year_match = re.search(r"(?:19|20)\d{2}", str(value_column))
            period = (
                year_match.group(0)
                if year_match
                else str(fallback_year or "current")
            )
            record = periods.setdefault(period, {})
            for _, row in frame.iterrows():
                label = str(row.get(label_column, "")).strip()
                number = _parse_number(row.get(value_column))
                if label and number is not None:
                    record[label] = number
    return periods


class ApplicationService:
    """串联 Parser、VectorStore、Retriever、Sandbox 和 MasterAgent。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.vector_store: VectorStoreProtocol | None = None
        self.vector_backend = "uninitialized"
        self.retriever: HybridRetriever | None = None
        self.chunker = FinancialChunker()
        self.registry: dict[str, dict[str, Any]] = {}

    def initialize(self) -> None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
        requested_backend = os.getenv("RAG_BACKEND", "chroma").casefold()
        allow_fallback = os.getenv("ALLOW_MEMORY_FALLBACK", "true").casefold() == "true"
        try:
            if requested_backend not in {"chroma", "faiss"}:
                raise ValueError("RAG_BACKEND 必须是 chroma 或 faiss")
            self.vector_store = FinancialVectorStore(
                backend=requested_backend,  # type: ignore[arg-type]
                persist_directory=VECTOR_DIR,
                embedding_function=build_embedding_function(),
            )
            self.vector_backend = requested_backend
        except (DependencyUnavailableError, ValueError, VectorStoreError) as exc:
            if not allow_fallback:
                raise
            LOGGER.warning("向量库初始化失败，降级为内存后端：%s", exc)
            self.vector_store = _MemoryVectorStore()
            self.vector_backend = "memory"
        self.retriever = HybridRetriever(self.vector_store)

    def ingest(
        self,
        file_path: Path,
        *,
        original_filename: str,
        year: int | None,
        document_id: str,
    ) -> UploadResponse:
        """同步的 CPU/IO 密集入库流程，API 层使用 to_thread 调用。"""

        if self.vector_store is None:
            raise RuntimeError("应用服务尚未初始化")
        suffix = file_path.suffix.casefold()
        parsed: WordParseResult | PDFParseResult
        if suffix == ".pdf":
            parsed = parse_pdf(file_path)
            text_units = len(parsed.pages)
        elif suffix == ".docx":
            parsed = parse_word(file_path)
            text_units = len(parsed.paragraphs)
        else:
            raise ValueError(f"不支持的文件类型：{suffix}")

        source = f"{document_id}:{original_filename}"
        chunks = self.chunker.chunk_parsed_document(
            parsed,
            source=source,
            year=year,
            metadata={"document_id": document_id, "filename": original_filename},
        )
        financial_data = _extract_financial_data(parsed.tables, year)

        with self.lock:
            self.vector_store.add_chunks(chunks)
            self.registry[source] = {
                "document_id": document_id,
                "filename": original_filename,
                "year": year,
                "stored_path": str(file_path),
                "financial_data": financial_data,
            }
            self._persist_registry()
            self.retriever = HybridRetriever(self.vector_store)

        return UploadResponse(
            document_id=document_id,
            filename=original_filename,
            file_type=suffix.removeprefix("."),
            chunks_written=len(chunks),
            tables_found=len(parsed.tables),
            text_units=text_units,
            vector_backend=self.vector_backend,
        )

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """构建本次请求的 Agent handler，并执行类型化状态图。"""

        if self.vector_store is None or self.retriever is None:
            raise RuntimeError("应用服务尚未初始化")

        rag_agent = RAGAgent(
            self.retriever,
            document_registry=self.registry,
            lock=self.lock,
        )
        master = MasterAgent(
            rag_handler=rag_agent,
            market_handler=RealTimeAgent(),
            calculation_handler=CalcAgent(),
            audit_agent=AuditAgent(
                fail_on_no_applicable_rules=request.mode == "deep"
            ),
            finalizer=build_finalizer(),
        )
        base_plan = master._default_plan(request.question)  # noqa: SLF001
        if request.financial_data and "rag" in base_plan:
            base_plan.remove("rag")
        if request.symbol and "market" not in base_plan:
            insert_at = 1 if base_plan and base_plan[0] == "rag" else 0
            base_plan.insert(insert_at, "market")
        if request.calculation_code and "calculation" not in base_plan:
            if not request.financial_data and "rag" not in base_plan:
                base_plan.insert(0, "rag")
            base_plan.append("calculation")
        if request.mode == "deep" and "audit" not in base_plan:
            base_plan.append("audit")

        master.router_classifier = lambda _question: base_plan

        state = create_initial_state(
            request.question,
            financial_data=request.financial_data,
            max_retries=0 if request.mode == "fast" else 2,
        )
        if request.symbol:
            state["market_data"] = {"requested_symbol": request.symbol}
        if request.calculation_code:
            state["calculation_code"] = request.calculation_code
        final_state = master.invoke(state)
        return AnalyzeResponse(
            trace_id=final_state["trace_id"],
            answer=final_state.get("final_answer") or "分析已完成。",
            route=final_state.get("route", ""),
            workflow_trace=final_state.get("workflow_trace", []),
            retrieved_context=final_state.get("retrieved_context", []),
            market_data=final_state.get("market_data", {}),
            calculation_code=final_state.get("calculation_code"),
            code_output=final_state.get("code_output"),
            audit_results=list(final_state.get("audit_results", [])),
            errors=final_state.get("errors", []),
        )

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not REGISTRY_PATH.is_file():
            return {}
        try:
            payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            LOGGER.exception("财务数据注册表加载失败")
            return {}

    def _persist_registry(self) -> None:
        temporary = REGISTRY_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.registry, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(REGISTRY_PATH)


SERVICE = ApplicationService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用启停钩子：避免在模块 import 阶段连接向量库。"""

    await asyncio.to_thread(SERVICE.initialize)
    yield


app = FastAPI(
    title="RAG Multi-Agent Financial Analysis API",
    version="4.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """轻量健康检查，不发起外部网络请求。"""

    chunks = len(SERVICE.vector_store.all_chunks()) if SERVICE.vector_store else 0
    degraded = SERVICE.vector_backend in {"memory", "uninitialized"}
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version="4.0.0",
        vector_backend=SERVICE.vector_backend,
        indexed_chunks=chunks,
        registered_documents=len(SERVICE.registry),
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_report(
    file: UploadFile = File(...),
    year: int | None = Form(default=None),
) -> UploadResponse:
    """上传 PDF/DOCX，解析后写入向量库。

    【API 路由说明】``async def`` 让 FastAPI 在等待客户上传时释放
    事件循环；Parser、pandas 和向量入库是阻塞任务，通过
    ``asyncio.to_thread`` 转移到工作线程，避免阻塞其他 API 请求。
    """

    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.casefold()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="仅支持 .pdf 和 .docx 文件")
    if year is not None and not 1900 <= year <= 2100:
        raise HTTPException(status_code=422, detail="year 必须在 1900~2100 之间")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        max_upload_mb = MAX_UPLOAD_BYTES // 1024 // 1024
        raise HTTPException(status_code=413, detail=f"文件不能超过 {max_upload_mb} MB")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="文件内容不是有效 PDF")
    if suffix == ".docx" and not content.startswith(b"PK"):
        raise HTTPException(status_code=415, detail="文件内容不是有效 DOCX")

    document_id = uuid4().hex
    target = UPLOAD_DIR / f"{document_id}{suffix}"
    try:
        await asyncio.to_thread(target.write_bytes, content)
        return await asyncio.to_thread(
            SERVICE.ingest,
            target,
            original_filename=filename,
            year=year,
            document_id=document_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        LOGGER.exception("财报上传入库失败")
        raise HTTPException(status_code=422, detail=f"财报解析或入库失败：{exc}") from exc


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """调用 MasterAgent 执行多 Agent 财务分析。

    POST 而非 GET 用于分析，因为问题、计算代码和表格数据都可能较大，
    且请求体需要 Pydantic 做类型验证。Agent 工作流可能包含同步网络
    SDK 与计算子进程，因此同样使用 ``to_thread`` 保护 async 事件循环。
    """

    try:
        return await asyncio.to_thread(SERVICE.analyze, request)
    except Exception as exc:
        LOGGER.exception("多 Agent 分析失败")
        raise HTTPException(status_code=500, detail=f"分析失败：{exc}") from exc


# 💡【面试加分点】API 层不直接编写 RAG/审计逻辑，而是调用
# ApplicationService 编排领域模块。这使 HTTP 协议、业务规则和存储实现
# 保持分层，后续切换 Celery 任务队列或将 Agent 拆分为微服务时无需重写路由。
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").casefold() == "true",
    )
