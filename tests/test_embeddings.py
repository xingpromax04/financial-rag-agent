"""Embedding provider configuration tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core.rag.embeddings import (
    EmbeddingConfigurationError,
    build_embedding_function,
    hash_embedding,
)


class EmbeddingProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_provider = os.environ.get("EMBEDDING_PROVIDER")
        self._old_api_key = os.environ.get("DASHSCOPE_API_KEY")

    def tearDown(self) -> None:
        if self._old_provider is None:
            os.environ.pop("EMBEDDING_PROVIDER", None)
        else:
            os.environ["EMBEDDING_PROVIDER"] = self._old_provider

        if self._old_api_key is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = self._old_api_key

    def test_default_provider_uses_hash_embedding(self) -> None:
        os.environ.pop("EMBEDDING_PROVIDER", None)

        embedding_function = build_embedding_function()

        self.assertIs(embedding_function, hash_embedding)

    def test_unknown_provider_raises_clear_error(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "unknown"

        with self.assertRaisesRegex(
            EmbeddingConfigurationError,
            "不支持的 EMBEDDING_PROVIDER",
        ):
            build_embedding_function()

    def test_dashscope_provider_requires_api_key(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "dashscope"
        os.environ.pop("DASHSCOPE_API_KEY", None)

        with self.assertRaisesRegex(
            EmbeddingConfigurationError,
            "缺少 DASHSCOPE_API_KEY",
        ):
            build_embedding_function()

    def test_dashscope_provider_builds_embedding_function_when_configured(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "dashscope"
        os.environ["DASHSCOPE_API_KEY"] = "test-key"

        embedding_function = build_embedding_function()

        self.assertTrue(callable(embedding_function))
        self.assertIsNot(embedding_function, hash_embedding)

    @patch("core.rag.embeddings.TextEmbedding")
    def test_dashscope_embedding_extracts_vectors_from_response(
        self,
        text_embedding_class,
    ) -> None:
        text_embedding_class.call.return_value = {
            "output": {
                "embeddings": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }
        }

        os.environ["EMBEDDING_PROVIDER"] = "dashscope"
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        embedding_function = build_embedding_function()

        vectors = embedding_function(["营业收入增长", "净利润下降"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
