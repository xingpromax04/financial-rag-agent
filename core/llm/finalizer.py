"""LLM finalizer for financial analysis workflow.

本模块只负责把 AgentState 转换为适合 LLM 总结的提示词，并调用注入的
client。真实 Qwen / DashScope 调用放在 qwen_client.py，便于单元测试
不依赖网络和 API key。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Protocol
from core.llm.qwen_client import LLMConfigurationError, QwenClient


class LLMClientProtocol(Protocol):
    """财报总结 LLM client 的最小接口。"""

    def summarize(self, prompt: str) -> str: ...


class QwenFinancialFinalizer:
    """将多 Agent 结果汇总为自然语言财报分析。"""

    def __init__(self, client: LLMClientProtocol, *, max_contexts: int = 8) -> None:
        if max_contexts < 1:
            raise ValueError("max_contexts 必须大于等于 1")
        self.client = client
        self.max_contexts = max_contexts

    def __call__(self, state: Mapping[str, Any]) -> str:
        prompt = self._build_prompt(state)
        return self.client.summarize(prompt)

    def _build_prompt(self, state: Mapping[str, Any]) -> str:
        question = str(state.get("question") or "").strip()
        route = str(state.get("route") or "")
        contexts = list(state.get("retrieved_context") or [])[: self.max_contexts]

        sections = [
            "你是严谨的中文财务分析助手。请基于提供的财报证据、结构化数据、计算结果和审计结果回答用户问题。",
            "要求：",
            "1. 只基于输入材料总结，不要编造未出现的数据。",
            "2. 涉及金额时优先换算为亿元，并保留约两位小数。",
            "3. 如果证据不足，请明确说明限制。",
            "4. 输出中文，结构清晰，适合展示在财报分析系统中。",
            "",
            f"用户问题：{question}",
            f"工作流路由：{route}",
            "",
            "检索证据：",
            self._format_contexts(contexts),
            "",
            "结构化财务数据：",
            self._to_json(state.get("financial_data") or {}),
            "",
            "市场数据：",
            self._to_json(state.get("market_data") or {}),
            "",
            "计算代码：",
            str(state.get("calculation_code") or "无"),
            "",
            "计算结果：",
            self._to_json(state.get("code_output") or {}),
            "",
            "审计结果：",
            self._to_json(state.get("audit_results") or []),
            "",
            "限制和异常：",
            self._format_errors(state.get("errors") or []),
        ]
        return "\n".join(sections)

    def _format_contexts(self, contexts: list[Any]) -> str:
        if not contexts:
            return "无"

        lines: list[str] = []
        for index, context in enumerate(contexts, start=1):
            if not isinstance(context, Mapping):
                continue
            metadata = context.get("metadata") or {}
            filename = ""
            source = ""
            if isinstance(metadata, Mapping):
                filename = str(metadata.get("filename") or "")
                source = str(metadata.get("source") or "")
            score = context.get("score")
            score_text = f"，score={score}" if isinstance(score, (int, float)) else ""
            text = str(context.get("text") or "").strip()
            lines.append(
                f"[证据{index}] 文件={filename or source or '-'}{score_text}\n{text}"
            )
        return "\n\n".join(lines) if lines else "无"

    @staticmethod
    def _format_errors(errors: Any) -> str:
        if not errors:
            return "无"
        return "\n".join(f"- {error}" for error in errors)

    @staticmethod
    def _to_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_finalizer() -> QwenFinancialFinalizer | None:
    """根据 LLM_PROVIDER 构建最终总结器。

    返回 None 表示继续使用 MasterAgent 内置规则总结。
    """

    provider = os.getenv("LLM_PROVIDER", "rule").strip().casefold()
    if provider == "rule":
        return None
    if provider == "dashscope":
        return QwenFinancialFinalizer(client=QwenClient())
    raise LLMConfigurationError(
        f"不支持的 LLM_PROVIDER：{provider}，可选值为 rule 或 dashscope"
    )


__all__ = [
    "LLMClientProtocol",
    "LLMConfigurationError",
    "QwenFinancialFinalizer",
    "build_finalizer",
]

