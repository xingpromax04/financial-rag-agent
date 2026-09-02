"""AkShare / yfinance 行情与基本面数据统一适配层。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
依赖：``pandas>=2.0``，以及 ``akshare`` / ``yfinance`` 中的至少一个。

数据流向：
    symbol -> 选择数据源 -> 超时与重试 -> 字段映射 -> 缺失值补全
           -> 固定 Schema 的 pandas.DataFrame -> Agent 工具调用
"""

from __future__ import annotations

import importlib
import logging
import queue
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Callable, Literal, TypeVar

import pandas as pd

Provider = Literal["auto", "akshare", "yfinance"]
T = TypeVar("T")
LOGGER = logging.getLogger(__name__)

QUOTE_COLUMNS = [
    "symbol",
    "name",
    "currency",
    "price",
    "open",
    "high",
    "low",
    "previous_close",
    "change_percent",
    "volume",
    "turnover",
    "provider",
    "timestamp",
]

FINANCIAL_COLUMNS = [
    "symbol",
    "name",
    "industry",
    "currency",
    "pe_ratio",
    "pb_ratio",
    "market_cap",
    "float_market_cap",
    "enterprise_value",
    "total_shares",
    "float_shares",
    "revenue",
    "net_income",
    "roe",
    "roa",
    "debt_to_equity",
    "debt_to_assets",
    "report_date",
    "provider",
    "timestamp",
]

_NUMERIC_COLUMNS = {
    "price",
    "open",
    "high",
    "low",
    "previous_close",
    "change_percent",
    "volume",
    "turnover",
    "pe_ratio",
    "pb_ratio",
    "market_cap",
    "float_market_cap",
    "enterprise_value",
    "total_shares",
    "float_shares",
    "revenue",
    "net_income",
    "roe",
    "roa",
    "debt_to_equity",
    "debt_to_assets",
}


class MarketDataError(RuntimeError):
    """行情服务基础异常。"""


class ProviderUnavailableError(MarketDataError):
    """数据源未安装或当前不可用。"""


class ProviderRequestError(MarketDataError):
    """数据源请求在重试后仍失败。"""


class RequestTimeoutError(MarketDataError):
    """数据源请求超过设定时间。"""


class SymbolNotFoundError(MarketDataError):
    """数据源未找到指定股票代码。"""


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """网络请求的超时与指数退避配置。"""

    attempts: int = 3  # 🔧【可调参数】包含首次请求的总尝试次数
    timeout_seconds: float = 12.0  # 🔧【可调参数】单次请求超时
    initial_backoff: float = 0.8  # 🔧【可调参数】首次重试前的等待时间
    max_backoff: float = 6.0  # 🔧【可调参数】退避时间上限
    jitter: float = 0.2  # 🔧【可调参数】随机抖动，避免并发请求同时重试

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts 必须大于等于 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if self.initial_backoff < 0 or self.max_backoff < 0 or self.jitter < 0:
            raise ValueError("退避时间与抖动值不能小于 0")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_provider(name: str) -> ModuleType:
    """延迟导入数据源，允许项目只安装实际使用的 provider。"""

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"未安装 {name}，请在 rag_311 环境执行 pip install {name}"
        ) from exc


def _is_missing(value: Any) -> bool:
    """统一判断 None、NaN、pd.NA 和常见空字符串。"""

    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "-", "--", "none", "nan", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any, *, percent: bool = False) -> float | None:
    """将逗号分隔、百分数和会计括号负数转为浮点数。"""

    if _is_missing(value):
        return None
    text = str(value).strip().replace(",", "")
    is_parenthesized = text.startswith("(") and text.endswith(")")
    has_percent_sign = text.endswith("%")
    text = text.strip("()%")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if is_parenthesized:
        number = -number
    return number / 100 if percent or has_percent_sign else number


def _first(mapping: Any, *keys: str) -> Any:
    """从多个候选字段中返回第一个非缺失值。"""

    for key in keys:
        try:
            value = mapping.get(key)
        except (AttributeError, KeyError, TypeError):
            value = None
        if not _is_missing(value):
            return value
    return None


def _cn_symbol(symbol: str) -> str:
    """将带交易所前后缀的 A 股代码规范化为 6 位数字。"""

    normalized = symbol.strip().upper()
    match = re.fullmatch(
        r"(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ|SS))?", normalized
    )
    if not match:
        raise ValueError(f"AkShare 需要 6 位 A 股代码：{symbol}")
    return match.group(1)


def _yfinance_symbol(symbol: str) -> str:
    """将纯数字 A 股代码补全为 yfinance 交易所后缀。"""

    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("股票代码不能为空")
    if re.fullmatch(r"\d{6}", normalized):
        if normalized.startswith(("5", "6", "9")):
            suffix = "SS"
        elif normalized.startswith(("4", "8")):
            suffix = "BJ"
        else:
            suffix = "SZ"
        return f"{normalized}.{suffix}"
    return normalized


def _call_with_timeout(operation: Callable[[], T], timeout: float) -> T:
    """在守护线程中执行不支持 timeout 参数的第三方接口。

    AkShare 的部分接口没有对外暴露超时参数。守护线程可以保证
    Agent 调用在设定时间后返回，不会无限卡住整个工作流。
    """

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result_queue.put((True, operation()))
        except Exception as exc:  # 异常在主线程中重新抛出
            result_queue.put((False, exc))

    worker = threading.Thread(target=run, daemon=True, name="market-data-request")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise RequestTimeoutError(f"行情请求超过 {timeout:.1f} 秒")

    succeeded, payload = result_queue.get_nowait()
    if succeeded:
        return payload  # type: ignore[return-value]
    if isinstance(payload, Exception):
        raise payload
    raise ProviderRequestError("行情请求返回未知结果")


def _execute_with_retry(
    operation: Callable[[], T],
    *,
    provider: str,
    config: RetryConfig,
) -> T:
    """执行超时控制与带抖动的指数退避重试。"""

    last_error: Exception | None = None
    for attempt in range(1, config.attempts + 1):
        try:
            return _call_with_timeout(operation, config.timeout_seconds)
        except (ProviderUnavailableError, SymbolNotFoundError, ValueError):
            # 缺少依赖、代码错误属于永久性失败，重试不会改变结果。
            raise
        except Exception as exc:
            last_error = exc
            if attempt == config.attempts:
                break
            exponential = config.initial_backoff * (2 ** (attempt - 1))
            delay = min(exponential, config.max_backoff) + random.uniform(
                0, config.jitter
            )
            LOGGER.warning(
                "%s 请求失败，%.2f 秒后第 %s/%s 次重试：%s",
                provider,
                delay,
                attempt + 1,
                config.attempts,
                exc,
            )
            time.sleep(delay)

    raise ProviderRequestError(
        f"{provider} 请求在 {config.attempts} 次尝试后失败"
    ) from last_error


def _normalize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """补齐固定 Schema，统一数值类型和缺失值表示。

    【数据标准化】AkShare 使用中文列名，yfinance 使用英文字段，
    且单位和缺失值表示不同。统一 Schema 后，上层 Agent 无需感知
    数据源差异，也不会把“未知”错误理解为数值 0。
    """

    normalized = frame.copy()
    for column in columns:
        if column not in normalized:
            normalized[column] = pd.NA
    normalized = normalized[columns]

    for column in columns:
        if column in _NUMERIC_COLUMNS:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            ).astype("Float64")
        else:
            normalized[column] = normalized[column].map(
                lambda value: pd.NA if _is_missing(value) else value
            )
    return normalized


def _has_missing(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return frame.empty or any(
        _is_missing(frame.iloc[0].get(column)) for column in columns
    )


def _fill_from_fallback(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    """仅用备用数据源填充缺失字段，不覆盖主数据源的有效值。"""

    result = primary.copy()
    for column in result.columns:
        primary_value = result.iloc[0].get(column)
        fallback_value = fallback.iloc[0].get(column)
        if _is_missing(primary_value) and not _is_missing(fallback_value):
            result.at[result.index[0], column] = fallback_value

    primary_provider = str(primary.iloc[0].get("provider"))
    fallback_provider = str(fallback.iloc[0].get("provider"))
    result.at[result.index[0], "provider"] = f"{primary_provider}+{fallback_provider}"
    return result


class MarketDataClient:
    """通过 AkShare / yfinance 获取可直接供 Agent 使用的标准化数据。"""

    def __init__(
        self,
        provider: Provider = "auto",  # 🔧【可调参数】auto 会按市场选择主备数据源
        *,
        retry_config: RetryConfig | None = None,
        fill_missing: bool = True,  # 🔧【可调参数】是否尝试从备用源补齐关键字段
    ) -> None:
        if provider not in {"auto", "akshare", "yfinance"}:
            raise ValueError(f"不支持的数据源：{provider}")
        self.provider = provider
        self.retry_config = retry_config or RetryConfig()
        self.fill_missing = fill_missing

    def get_realtime_quote(self, symbol: str) -> pd.DataFrame:
        """获取最新行情，返回单行、固定字段的 DataFrame。"""

        return self._request(
            symbol,
            calls={
                "akshare": self._quote_from_akshare,
                "yfinance": self._quote_from_yfinance,
            },
            columns=QUOTE_COLUMNS,
            completeness_fields=("price", "previous_close", "volume"),
        )

    def get_financial_indicators(self, symbol: str) -> pd.DataFrame:
        """获取估值、规模、盈利能力和偿债能力指标。"""

        return self._request(
            symbol,
            calls={
                "akshare": self._financials_from_akshare,
                "yfinance": self._financials_from_yfinance,
            },
            columns=FINANCIAL_COLUMNS,
            completeness_fields=("market_cap", "pe_ratio", "pb_ratio", "roe"),
        )

    def _provider_order(self, symbol: str) -> list[str]:
        if self.provider != "auto":
            return [self.provider]
        try:
            _cn_symbol(symbol)
        except ValueError:
            return ["yfinance", "akshare"]
        return ["akshare", "yfinance"]

    def _request(
        self,
        symbol: str,
        *,
        calls: dict[str, Callable[[str], pd.DataFrame]],
        columns: list[str],
        completeness_fields: tuple[str, ...],
    ) -> pd.DataFrame:
        """执行主备查询，并在需要时用备用源补全缺失字段。"""

        symbol = symbol.strip()
        if not symbol:
            raise ValueError("股票代码不能为空")

        primary: pd.DataFrame | None = None
        errors: list[str] = []
        for provider in self._provider_order(symbol):
            try:
                frame = _execute_with_retry(
                    lambda provider=provider: calls[provider](symbol),
                    provider=provider,
                    config=self.retry_config,
                )
                frame = _normalize_frame(frame, columns)
                if frame.empty:
                    raise SymbolNotFoundError(f"{provider} 未返回 {symbol} 的数据")

                if primary is None:
                    primary = frame
                    should_fill = (
                        self.provider == "auto"
                        and self.fill_missing
                        and _has_missing(primary, completeness_fields)
                    )
                    if should_fill:
                        continue
                    return primary
                return _fill_from_fallback(primary, frame)
            except (MarketDataError, ValueError) as exc:
                errors.append(f"{provider}: {exc}")
                LOGGER.warning("数据源 %s 查询 %s 失败：%s", provider, symbol, exc)

        if primary is not None:
            LOGGER.info("备用源未能补全数据，返回主数据源结果：%s", symbol)
            return primary
        raise MarketDataError(f"无法获取 {symbol} 的数据；" + " | ".join(errors))

    @staticmethod
    def _akshare_spot(symbol: str) -> tuple[str, pd.Series]:
        akshare = _load_provider("akshare")
        code = _cn_symbol(symbol)
        spot = akshare.stock_zh_a_spot_em()
        if "代码" not in spot.columns:
            raise ProviderRequestError("AkShare 实时行情缺少‘代码’列")
        matches = spot.loc[spot["代码"].astype(str).str.zfill(6) == code]
        if matches.empty:
            raise SymbolNotFoundError(f"未找到 A 股代码 {code}")
        return code, matches.iloc[0]

    @classmethod
    def _quote_from_akshare(cls, symbol: str) -> pd.DataFrame:
        code, row = cls._akshare_spot(symbol)
        return pd.DataFrame(
            [
                {
                    "symbol": code,
                    "name": row.get("名称"),
                    "currency": "CNY",
                    "price": _number(row.get("最新价")),
                    "open": _number(row.get("今开")),
                    "high": _number(row.get("最高")),
                    "low": _number(row.get("最低")),
                    "previous_close": _number(row.get("昨收")),
                    "change_percent": _number(row.get("涨跌幅"), percent=True),
                    "volume": _number(row.get("成交量")),
                    "turnover": _number(row.get("成交额")),
                    "provider": "akshare",
                    "timestamp": _utc_now(),
                }
            ]
        )

    @classmethod
    def _financials_from_akshare(cls, symbol: str) -> pd.DataFrame:
        code, spot_row = cls._akshare_spot(symbol)
        akshare = _load_provider("akshare")
        detail: dict[str, Any] = {}
        analysis: Any = {}

        # 基本信息和财务指标是增强数据；单个接口失败时保留已获取部分。
        try:
            info = akshare.stock_individual_info_em(symbol=code)
            if {"item", "value"}.issubset(info.columns):
                detail = dict(zip(info["item"], info["value"], strict=False))
        except Exception as exc:
            LOGGER.warning("AkShare 个股基本信息获取失败：%s", exc)

        try:
            indicators = akshare.stock_financial_analysis_indicator(
                symbol=code,
                start_year=str(datetime.now().year - 3),
            )
            if not indicators.empty:
                date_column = next(
                    (
                        column
                        for column in ("日期", "报告期", "报告日期")
                        if column in indicators.columns
                    ),
                    None,
                )
                if date_column:
                    order = pd.to_datetime(indicators[date_column], errors="coerce")
                    analysis = indicators.loc[order.sort_values().index[-1]]
                else:
                    analysis = indicators.iloc[-1]
        except Exception as exc:
            LOGGER.warning("AkShare 财务分析指标获取失败：%s", exc)

        return pd.DataFrame(
            [
                {
                    "symbol": code,
                    "name": spot_row.get("名称") or detail.get("股票简称"),
                    "industry": detail.get("行业"),
                    "currency": "CNY",
                    "pe_ratio": _number(
                        _first(spot_row, "市盈率-动态", "市盈率")
                    ),
                    "pb_ratio": _number(spot_row.get("市净率")),
                    "market_cap": _number(_first(detail, "总市值"))
                    or _number(spot_row.get("总市值")),
                    "float_market_cap": _number(_first(detail, "流通市值"))
                    or _number(spot_row.get("流通市值")),
                    "total_shares": _number(detail.get("总股本")),
                    "float_shares": _number(detail.get("流通股")),
                    "roe": _number(
                        _first(
                            analysis,
                            "净资产收益率(%)",
                            "加权净资产收益率(%)",
                        ),
                        percent=True,
                    ),
                    "roa": _number(
                        _first(
                            analysis,
                            "总资产净利润率(%)",
                            "总资产报酬率(%)",
                        ),
                        percent=True,
                    ),
                    "debt_to_assets": _number(
                        _first(analysis, "资产负债率(%)"), percent=True
                    ),
                    "report_date": _first(
                        analysis, "日期", "报告期", "报告日期"
                    ),
                    "provider": "akshare",
                    "timestamp": _utc_now(),
                }
            ]
        )

    def _quote_from_yfinance(self, symbol: str) -> pd.DataFrame:
        yfinance = _load_provider("yfinance")
        ticker_symbol = _yfinance_symbol(symbol)
        ticker = yfinance.Ticker(ticker_symbol)
        history = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
            timeout=self.retry_config.timeout_seconds,
        )
        if history.empty:
            raise SymbolNotFoundError(f"未找到股票代码 {ticker_symbol}")

        latest = history.iloc[-1]
        fast_info = ticker.fast_info
        previous_close = _first(
            fast_info, "previous_close", "regular_market_previous_close"
        )
        price = _number(_first(fast_info, "last_price")) or _number(
            latest.get("Close")
        )
        previous_close_number = _number(previous_close)
        change_percent = (
            price / previous_close_number - 1
            if price is not None and previous_close_number
            else None
        )

        return pd.DataFrame(
            [
                {
                    "symbol": ticker_symbol,
                    "currency": _first(fast_info, "currency"),
                    "price": price,
                    "open": _number(_first(fast_info, "open"))
                    or _number(latest.get("Open")),
                    "high": _number(_first(fast_info, "day_high"))
                    or _number(latest.get("High")),
                    "low": _number(_first(fast_info, "day_low"))
                    or _number(latest.get("Low")),
                    "previous_close": previous_close_number,
                    "change_percent": change_percent,
                    "volume": _number(_first(fast_info, "last_volume"))
                    or _number(latest.get("Volume")),
                    "provider": "yfinance",
                    "timestamp": _utc_now(),
                }
            ]
        )

    @staticmethod
    def _financials_from_yfinance(symbol: str) -> pd.DataFrame:
        yfinance = _load_provider("yfinance")
        ticker_symbol = _yfinance_symbol(symbol)
        info = yfinance.Ticker(ticker_symbol).get_info()
        if not info or not _first(info, "symbol", "shortName", "longName"):
            raise SymbolNotFoundError(f"未找到股票代码 {ticker_symbol}")

        return pd.DataFrame(
            [
                {
                    "symbol": info.get("symbol", ticker_symbol),
                    "name": _first(info, "shortName", "longName"),
                    "industry": info.get("industry"),
                    "currency": _first(info, "financialCurrency", "currency"),
                    "pe_ratio": _number(_first(info, "trailingPE", "forwardPE")),
                    "pb_ratio": _number(info.get("priceToBook")),
                    "market_cap": _number(info.get("marketCap")),
                    "enterprise_value": _number(info.get("enterpriseValue")),
                    "revenue": _number(info.get("totalRevenue")),
                    "net_income": _number(info.get("netIncomeToCommon")),
                    "roe": _number(info.get("returnOnEquity")),
                    "roa": _number(info.get("returnOnAssets")),
                    "debt_to_equity": _number(info.get("debtToEquity")),
                    "provider": "yfinance",
                    "timestamp": _utc_now(),
                }
            ]
        )


# 💡【面试加分点】RAG 中的财报切片提供“历史事实与管理层解释”，
# 实时市场工具则提供“当前价格与最新估值”。Agent 应先从向量库召回
# 可追溯的历史证据，再调用本模块补充时效数据，并在答案中分别标注
# “报告期”与“行情时间”，避免把不同时点的数据错当成同期口径。
def get_realtime_quote(
    symbol: str,
    provider: Provider = "auto",
    *,
    retry_config: RetryConfig | None = None,
) -> pd.DataFrame:
    """便捷函数：获取标准化最新行情。"""

    return MarketDataClient(provider, retry_config=retry_config).get_realtime_quote(
        symbol
    )


def get_financial_indicators(
    symbol: str,
    provider: Provider = "auto",
    *,
    retry_config: RetryConfig | None = None,
) -> pd.DataFrame:
    """便捷函数：获取标准化基本面财务指标。"""

    return MarketDataClient(
        provider, retry_config=retry_config
    ).get_financial_indicators(symbol)


__all__ = [
    "FINANCIAL_COLUMNS",
    "QUOTE_COLUMNS",
    "MarketDataClient",
    "MarketDataError",
    "ProviderRequestError",
    "ProviderUnavailableError",
    "RequestTimeoutError",
    "RetryConfig",
    "SymbolNotFoundError",
    "get_financial_indicators",
    "get_realtime_quote",
]
