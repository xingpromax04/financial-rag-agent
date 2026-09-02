"""DashScope Qwen LLM client.

封装通义千问文本生成调用，让业务层只依赖 summarize(prompt) 这个小接口。
"""

from __future__ import annotations

import os
from typing import Any

try:
    from dashscope import Generation
except ImportError:  # pragma: no cover - depends on optional cloud SDK
    Generation = None  # type: ignore[assignment]


DEFAULT_QWEN_LLM_MODEL = "qwen-plus"


class LLMConfigurationError(RuntimeError):
    """LLM provider 配置或响应错误。"""


class QwenClient:
    """DashScope Qwen 文本总结客户端。"""

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("QWEN_LLM_MODEL", DEFAULT_QWEN_LLM_MODEL)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError("缺少 DASHSCOPE_API_KEY 环境变量")
        if Generation is None:
            raise LLMConfigurationError(
                "缺少 dashscope SDK，请先执行 pip install dashscope"
            )

    def summarize(self, prompt: str) -> str:
        """调用 Qwen 生成财报总结。"""

        response = Generation.call(
            model=self.model,
            prompt=prompt,
            api_key=self.api_key,
        )
        text = self._extract_text(response).strip()
        if not text:
            raise LLMConfigurationError("Qwen 返回了空摘要")
        return text

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, dict):
            return str(response.get("output", {}).get("text") or "")

        output = getattr(response, "output", None)
        if isinstance(output, dict):
            return str(output.get("text") or "")

        return str(getattr(output, "text", "") or "")


__all__ = ["DEFAULT_QWEN_LLM_MODEL", "LLMConfigurationError", "QwenClient"]
