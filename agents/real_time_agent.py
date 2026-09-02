"""实时行情 Agent：解析证券代码并获取标准化市场快照。

运行环境：Python 3.11，Conda 环境 ``rag_311``。

输入：``AgentState.question`` 或 ``market_data.requested_symbol``。
输出：``market_data`` 和工具消息的状态增量，不修改历史财报证据。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Protocol

import pandas as pd

from agents.state import AgentState
from core.tools.market_data import MarketDataClient, RetryConfig

_REAL_TIME_TERMS = (
    "实时",
    "当前",
    "现在",
    "今日",
    "最新",
    "股价",
    "行情",
    "大盘",
    "市值",
    "涨跌",
    "成交量",
    "市盈率",
    "市净率",
    "price",
    "quote",
    "market cap",
    "p/e",
    "p/b",
)
_A_SHARE_RE = re.compile(
    r"(?<![A-Z0-9])(?:SH|SZ|BJ)?\d{6}(?:\.(?:SH|SZ|BJ|SS))?(?![A-Z0-9])",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:[.-][A-Z]{1,4})?\b")
_TICKER_STOP_WORDS = {
    "API",
    "CAGR",
    "EBIT",
    "EPS",
    "PB",
    "PE",
    "RAG",
    "ROA",
    "ROE",
}


class MarketClientProtocol(Protocol):
    """便于测试时注入离线行情客户端。"""

    def get_realtime_quote(self, symbol: str) -> pd.DataFrame: ...

    def get_financial_indicators(self, symbol: str) -> pd.DataFrame: ...


class RealTimeAgentError(RuntimeError):
    """股票代码解析或实时行情获取失败。"""


def matches_real_time_intent(question: str) -> bool:
    """供 MasterAgent 路由节点复用的确定性实时意图判定。"""

    normalized = question.casefold()
    return any(term.casefold() in normalized for term in _REAL_TIME_TERMS)


def extract_symbol(question: str) -> str | None:
    """从中文问题中提取 A 股代码或大写国际股票 Ticker。"""

    a_share_match = _A_SHARE_RE.search(question)
    if a_share_match:
        return a_share_match.group(0).upper()
    for match in _TICKER_RE.finditer(question):
        symbol = match.group(0).upper()
        if symbol not in _TICKER_STOP_WORDS:
            return symbol
    return None


class RealTimeAgent:
    """将 ``MarketDataClient`` 包装为实时数据 LangGraph 节点。

    Router 只在 ``matches_real_time_intent`` 命中时把该节点加入计划；
    节点本身再严格校验股票代码，形成“意图路由 + 参数校验”两层边界。
    显式传入 ``requested_symbol`` 的 API 请求优先于自然语言解析结果。
    """

    def __init__(
        self,
        client: MarketClientProtocol | None = None,
        *,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.client = client or MarketDataClient(
            "auto",
            retry_config=retry_config
            or RetryConfig(
                attempts=2,  # 🔧【可调参数】交互页面优先控制等待时间
                timeout_seconds=8,
            ),
        )

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """获取最新报价与基本面，并返回 JSON 安全的状态增量。"""

        symbol = self._requested_symbol(state)
        try:
            quote_frame = self.client.get_realtime_quote(symbol)
            fundamentals_frame = self.client.get_financial_indicators(symbol)
        except Exception as exc:
            raise RealTimeAgentError(f"获取 {symbol} 实时数据失败：{exc}") from exc
        if quote_frame.empty and fundamentals_frame.empty:
            raise RealTimeAgentError(f"行情服务未返回 {symbol} 的数据")

        quote = self._first_record(quote_frame)
        fundamentals = self._first_record(fundamentals_frame)
        requested_metrics = self._requested_metrics(state["question"])
        return {
            "market_data": {
                "requested_symbol": symbol,
                "requested_metrics": requested_metrics,
                "quote": quote,
                "fundamentals": fundamentals,
            },
            "messages": [
                {
                    "role": "tool",
                    "name": "real_time_agent",
                    "content": (
                        f"已更新 {symbol} 的实时行情与基本面"
                        f"（关注指标：{', '.join(requested_metrics)}）。"
                    ),
                }
            ],
        }

    @staticmethod
    def _requested_symbol(state: AgentState) -> str:
        market_data = state.get("market_data", {})
        requested = market_data.get("requested_symbol")
        symbol = str(requested).strip() if requested else extract_symbol(
            state.get("question", "")
        )
        if not symbol:
            raise RealTimeAgentError(
                "实时行情问题需要股票代码，例如 600519、AAPL 或 API symbol 字段"
            )
        return symbol.upper()

    @staticmethod
    def _requested_metrics(question: str) -> list[str]:
        metric_terms = {
            "price": ("股价", "价格", "price", "quote"),
            "change_percent": ("涨跌", "涨幅", "跌幅"),
            "volume": ("成交量", "成交额"),
            "pe_ratio": ("市盈率", "pe", "p/e"),
            "pb_ratio": ("市净率", "pb", "p/b"),
            "market_cap": ("市值", "market cap"),
            "roe": ("roe", "净资产收益率"),
        }
        normalized = question.casefold()
        requested = [
            name
            for name, terms in metric_terms.items()
            if any(term.casefold() in normalized for term in terms)
        ]
        return requested or ["price", "fundamentals"]

    @classmethod
    def _first_record(cls, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {}
        return {
            str(key): cls._json_safe(value)
            for key, value in frame.iloc[0].to_dict().items()
        }

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """将 pandas/NumPy 标量与缺失值转换为 API 可序列化数据。"""

        if value is None or value is pd.NA:
            return None
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)):
            return value

        # 💡【面试加分点】路由条件与网络请求实现被拆开：MasterAgent
        # 只判断是否需要“实时能力”，RealTimeAgent 负责代码解析、重试和
        # Schema 标准化。更换行情供应商不会改变状态图的节点与边。
        return str(value)


__all__ = [
    "RealTimeAgent",
    "RealTimeAgentError",
    "extract_symbol",
    "matches_real_time_intent",
]
