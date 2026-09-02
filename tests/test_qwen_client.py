"""Qwen client tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core.llm.qwen_client import LLMConfigurationError, QwenClient


class QwenClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self._old_model = os.environ.get("QWEN_LLM_MODEL")

    def tearDown(self) -> None:
        if self._old_api_key is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = self._old_api_key

        if self._old_model is None:
            os.environ.pop("QWEN_LLM_MODEL", None)
        else:
            os.environ["QWEN_LLM_MODEL"] = self._old_model

    def test_qwen_client_requires_api_key(self) -> None:
        os.environ.pop("DASHSCOPE_API_KEY", None)

        with self.assertRaisesRegex(LLMConfigurationError, "缺少 DASHSCOPE_API_KEY"):
            QwenClient()

    @patch("core.llm.qwen_client.Generation")
    def test_qwen_client_calls_dashscope_generation(self, generation_class) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        os.environ["QWEN_LLM_MODEL"] = "qwen-plus"
        generation_class.call.return_value = {
            "output": {
                "text": "这是 Qwen 生成的财报总结。"
            }
        }

        client = QwenClient()
        answer = client.summarize("请总结财报。")

        self.assertEqual(answer, "这是 Qwen 生成的财报总结。")
        generation_class.call.assert_called_once()
        _, kwargs = generation_class.call.call_args
        self.assertEqual(kwargs["model"], "qwen-plus")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertIn("请总结财报。", kwargs["prompt"])

    @patch("core.llm.qwen_client.Generation")
    def test_qwen_client_raises_clear_error_for_empty_response(
        self,
        generation_class,
    ) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        generation_class.call.return_value = {"output": {"text": ""}}

        client = QwenClient()

        with self.assertRaisesRegex(LLMConfigurationError, "Qwen 返回了空摘要"):
            client.summarize("请总结财报。")


if __name__ == "__main__":
    unittest.main()
