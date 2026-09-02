"""FastAPI API endpoint regression tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from typing import cast
import main
from main import SERVICE, app



class APIEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self._temporary_directory.name) / "uploads"
        self.vector_dir = Path(self._temporary_directory.name) / "vectors"

        self._old_upload_dir = main.UPLOAD_DIR
        self._old_vector_dir = main.VECTOR_DIR
        self._old_registry_path = main.REGISTRY_PATH
        self._old_upload_env = os.environ.get("UPLOAD_DIR")
        self._old_vector_env = os.environ.get("VECTOR_DIR")
        self._old_backend_env = os.environ.get("RAG_BACKEND")
        self._old_fallback_env = os.environ.get("ALLOW_MEMORY_FALLBACK")

        os.environ["UPLOAD_DIR"] = str(self.upload_dir)
        os.environ["VECTOR_DIR"] = str(self.vector_dir)
        os.environ["RAG_BACKEND"] = "memory"
        os.environ["ALLOW_MEMORY_FALLBACK"] = "true"
        main.UPLOAD_DIR = self.upload_dir
        main.VECTOR_DIR = self.vector_dir
        main.REGISTRY_PATH = self.upload_dir / "financial_registry.json"


        SERVICE.initialize()


    def tearDown(self) -> None:
        main.UPLOAD_DIR = self._old_upload_dir
        main.VECTOR_DIR = self._old_vector_dir
        main.REGISTRY_PATH = self._old_registry_path

        if self._old_upload_env is None:
            os.environ.pop("UPLOAD_DIR", None)
        else:
            os.environ["UPLOAD_DIR"] = self._old_upload_env

        if self._old_vector_env is None:
            os.environ.pop("VECTOR_DIR", None)
        else:
            os.environ["VECTOR_DIR"] = self._old_vector_env
        if self._old_backend_env is None:
            os.environ.pop("RAG_BACKEND", None)
        else:
            os.environ["RAG_BACKEND"] = self._old_backend_env

        if self._old_fallback_env is None:
            os.environ.pop("ALLOW_MEMORY_FALLBACK", None)
        else:
            os.environ["ALLOW_MEMORY_FALLBACK"] = self._old_fallback_env

        self._temporary_directory.cleanup()


    def test_health_returns_service_status(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["status"], {"ok", "degraded"})
        self.assertEqual(payload["version"], "4.0.0")
        self.assertIn("vector_backend", payload)
        self.assertIn("indexed_chunks", payload)
        self.assertIn("registered_documents", payload)


    def test_upload_rejects_unsupported_file_extension(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "report.txt",
                        b"not a financial report",
                        "text/plain",
                    )
                },
                data={"year": "2024"},
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "仅支持 .pdf 和 .docx 文件")


    def test_upload_rejects_invalid_pdf_signature(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "report.pdf",
                        b"this is not a real pdf",
                        "application/pdf",
                    )
                },
                data={"year": "2024"},
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "文件内容不是有效 PDF")


    def test_upload_rejects_invalid_docx_signature(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "report.docx",
                        b"this is not a real docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                data={"year": "2024"},
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "文件内容不是有效 DOCX")


    def test_analyze_rejects_empty_question(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/analyze",
                json={
                    "question": "",
                    "mode": "fast",
                },
            )

        self.assertEqual(response.status_code, 422)


    def test_analyze_returns_structured_response_with_financial_data(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/analyze",
                json={
                    "question": "计算2024年速动比率，并检查勾稽关系",
                    "mode": "fast",
                    "financial_data": {
                        "2024": {
                            "流动资产合计": 300,
                            "存货": 60,
                            "流动负债合计": 160,
                            "资产总计": 1000,
                            "负债合计": 400,
                            "所有者权益合计": 600,
                        }
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("trace_id", payload)
        self.assertIn("answer", payload)
        self.assertIn("route", payload)
        self.assertIn("workflow_trace", payload)
        self.assertIn("retrieved_context", payload)
        self.assertIn("market_data", payload)
        self.assertIn("calculation_code", payload)
        self.assertIn("code_output", payload)
        self.assertIn("audit_results", payload)
        self.assertIn("errors", payload)
        self.assertEqual(payload["code_output"]["result"]["quick_ratio"], 1.5)


    def test_analyze_reports_error_when_no_financial_context_exists(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/analyze",
                json={
                    "question": "分析2024年营业收入和净利润",
                    "mode": "fast",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("财报库中没有可用切片", "".join(payload["errors"]))
        self.assertIsNone(payload["code_output"])
        self.assertEqual(payload["retrieved_context"], [])


    @patch("main.build_finalizer")
    def test_analyze_uses_configured_llm_finalizer(self, build_finalizer_mock) -> None:
        def fake_finalizer(state):
            self.assertEqual(state["question"], "总结药明康德2025年营业收入")
            self.assertTrue(state.get("retrieved_context"))
            return "这是 Qwen 生成的最终财报总结。"

        build_finalizer_mock.return_value = fake_finalizer

        with TestClient(app) as client:
            chunk = main.DocumentChunk(
                chunk_id="income-statement",
                text="合并利润表\n营业收入 2025年度 45,456,165,774.18",

                metadata={
                    "source": "doc-1:report.pdf",
                    "filename": "report.pdf",
                    "year": 2025,
                },
            )
            vector_store = cast(main.VectorStoreProtocol, SERVICE.vector_store)
            vector_store.add_chunks([chunk])
            SERVICE.retriever = main.HybridRetriever(vector_store)

            response = client.post(
                "/analyze",
                json={
                    "question": "总结药明康德2025年营业收入",
                    "mode": "fast",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "这是 Qwen 生成的最终财报总结。")
        build_finalizer_mock.assert_called_once()












if __name__ == "__main__":
    unittest.main()
