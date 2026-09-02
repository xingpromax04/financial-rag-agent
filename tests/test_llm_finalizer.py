"""LLM finalizer tests."""

from __future__ import annotations
import os
import unittest

from core.llm.finalizer import (
    LLMConfigurationError,
    QwenFinancialFinalizer,
    build_finalizer,
)



class _LLMClientStub:
    def __init__(self) -> None:
        self.prompt = ""

    def summarize(self, prompt: str) -> str:
        self.prompt = prompt
        return "药明康德2024年营业收入为392.41亿元，净利润情况需结合利润表继续分析。"


class QwenFinancialFinalizerTests(unittest.TestCase):
    def test_finalizer_uses_context_and_financial_outputs_in_prompt(self) -> None:
        client = _LLMClientStub()
        finalizer = QwenFinancialFinalizer(client=client)

        answer = finalizer(
            {
                "question": "总结药明康德2024年营业收入和净利润情况",
                "route": "rag+calculation+audit",
                "retrieved_context": [
                    {
                        "text": (
                            "合并利润表\n"
                            "项目 2025年度 2024年度\n"
                            "营业收入 45,456,165,774.18 39,241,431,359.88\n"
                            "净利润 15,699,761,735.45 9,607,037,494.26"
                        ),
                        "metadata": {
                            "filename": "药明康德_2025年年度报告.pdf",
                            "source": "doc-1:药明康德_2025年年度报告.pdf",
                        },
                        "score": 0.05,
                    }
                ],
                "financial_data": {
                    "2024": {
                        "营业收入": 39241431359.88,
                        "净利润": 9607037494.26,
                    }
                },
                "market_data": {},
                "calculation_code": None,
                "code_output": None,
                "audit_results": [],
                "errors": [],
            }
        )

        self.assertIn("营业收入为392.41亿元", answer)
        self.assertIn("总结药明康德2024年营业收入和净利润情况", client.prompt)
        self.assertIn("合并利润表", client.prompt)
        self.assertIn("营业收入 45,456,165,774.18", client.prompt)
        self.assertIn("净利润 15,699,761,735.45", client.prompt)
        self.assertIn("药明康德_2025年年度报告.pdf", client.prompt)

    def test_finalizer_includes_errors_as_limitations(self) -> None:
        client = _LLMClientStub()
        finalizer = QwenFinancialFinalizer(client=client)

        finalizer(
            {
                "question": "分析营业收入",
                "route": "rag",
                "retrieved_context": [],
                "financial_data": {},
                "market_data": {},
                "calculation_code": None,
                "code_output": None,
                "audit_results": [],
                "errors": ["财报库中没有可用切片，请先上传 PDF 或 DOCX"],
            }
        )

        self.assertIn("限制和异常", client.prompt)
        self.assertIn("财报库中没有可用切片", client.prompt)


class FinalizerFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_provider = os.environ.get("LLM_PROVIDER")
        self._old_api_key = os.environ.get("DASHSCOPE_API_KEY")

    def tearDown(self) -> None:
        if self._old_provider is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = self._old_provider

        if self._old_api_key is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = self._old_api_key

    def test_default_provider_uses_rule_based_finalizer(self) -> None:
        os.environ.pop("LLM_PROVIDER", None)

        self.assertIsNone(build_finalizer())

    def test_unknown_provider_raises_clear_error(self) -> None:
        os.environ["LLM_PROVIDER"] = "unknown"

        with self.assertRaisesRegex(
            LLMConfigurationError,
            "不支持的 LLM_PROVIDER",
        ):
            build_finalizer()

    def test_dashscope_provider_requires_api_key(self) -> None:
        os.environ["LLM_PROVIDER"] = "dashscope"
        os.environ.pop("DASHSCOPE_API_KEY", None)

        with self.assertRaisesRegex(
            LLMConfigurationError,
            "缺少 DASHSCOPE_API_KEY",
        ):
            build_finalizer()

    def test_dashscope_provider_builds_qwen_finalizer(self) -> None:
        os.environ["LLM_PROVIDER"] = "dashscope"
        os.environ["DASHSCOPE_API_KEY"] = "test-key"

        finalizer = build_finalizer()

        self.assertIsInstance(finalizer, QwenFinancialFinalizer)


if __name__ == "__main__":
    unittest.main()
