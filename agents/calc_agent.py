"""财务计算 Agent：生成确定性公式并交给隔离沙箱执行。

运行环境：Python 3.11，Conda 环境 ``rag_311``。

输入：``financial_data``、用户问题及可选的 ``calculation_code``。
输出：计算代码、结果、stdout、耗时和工具消息的 AgentState 增量。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from agents.state import AgentState
from core.code_interpreter.calc_engine import CalculationEngine

CodeGenerator: TypeAlias = Callable[[str, Mapping[str, float]], str]
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "net_profit": ("net_profit", "净利润"),
    "total_equity": (
        "total_equity",
        "equity",
        "所有者权益合计",
        "股东权益合计",
        "净资产",
    ),
    "total_assets": ("total_assets", "assets", "资产总计", "总资产"),
    "total_liabilities": (
        "total_liabilities",
        "liabilities",
        "负债合计",
        "总负债",
    ),
    "operating_revenue": (
        "operating_revenue",
        "revenue",
        "营业收入",
        "营业总收入",
    ),
    "operating_cost": ("operating_cost", "营业成本", "营业总成本"),
    "current_assets": ("current_assets", "流动资产合计", "流动资产"),
    "inventories": ("inventories", "inventory", "存货"),
    "current_liabilities": (
        "current_liabilities",
        "流动负债合计",
        "流动负债",
    ),
}


class CalcAgentError(RuntimeError):
    """计算输入不足或无法生成确定性公式。"""


class CalcAgent:
    """连接财务数据、代码生成策略与 ``CalculationEngine``。

    ``code_generator`` 是可注入扩展点：生产环境可接入 LLM 生成公式，
    默认实现则使用白名单模板保证离线可运行。无论代码来自哪里，最终都
    必须经过 CalculationEngine 的 AST 校验和隔离子进程执行。
    """

    def __init__(
        self,
        engine: CalculationEngine | None = None,
        *,
        code_generator: CodeGenerator | None = None,
    ) -> None:
        self.engine = engine or CalculationEngine()
        self.code_generator = code_generator or self._default_code_generator

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """生成并执行公式，运行日志随 ``code_output`` 写回共享状态。"""

        question = str(state.get("question", "")).strip()
        canonical = self.canonicalize_financial_data(
            state.get("financial_data", {})
        )
        code = str(state.get("calculation_code", "")).strip()
        if not code:
            code = self.code_generator(question, canonical)

        inputs: dict[str, Any] = {
            "financial_data": state.get("financial_data", {}),
            **canonical,
        }
        execution = self.engine.execute(code, inputs=inputs)
        output = execution.to_dict()
        log_summary = (
            f"success={execution.success}, duration_ms={execution.duration_ms:.2f}"
        )
        if execution.stdout:
            log_summary += ", stdout 已捕获"
        return {
            "calculation_code": code,
            "code_output": output,
            "messages": [
                {
                    "role": "tool",
                    "name": "calc_agent",
                    "content": f"计算沙箱执行完成：{log_summary}。",
                    "metadata": {
                        "input_fields": sorted(canonical),
                        "duration_ms": execution.duration_ms,
                    },
                }
            ],
        }

    @classmethod
    def canonicalize_financial_data(
        cls,
        financial_data: Mapping[str, Any],
    ) -> dict[str, float]:
        """选取最新期间，并把中英文报表字段映射为稳定变量名。"""

        if not isinstance(financial_data, Mapping):
            raise CalcAgentError("financial_data 必须是映射类型")
        records = cls._candidate_records(financial_data)
        latest = max(records, key=cls._period_sort_key)[1] if records else {}
        normalized = {
            str(key).strip().casefold(): value for key, value in latest.items()
        }

        canonical: dict[str, float] = {}
        for field, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                number = cls._parse_number(normalized.get(alias.casefold()))
                if number is not None:
                    canonical[field] = number
                    break
        return canonical

    @classmethod
    def _candidate_records(
        cls,
        data: Mapping[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        """兼容 ``期间 -> 指标`` 和 ``报表 -> 期间 -> 指标`` 两种结构。"""

        records: list[tuple[str, dict[str, Any]]] = []
        scalar_values = {
            str(key): value
            for key, value in data.items()
            if not isinstance(value, Mapping)
        }
        if scalar_values:
            records.append(("current", scalar_values))

        for key, value in data.items():
            if not isinstance(value, Mapping):
                continue
            nested_scalars = {
                str(field): item
                for field, item in value.items()
                if not isinstance(item, Mapping)
            }
            if nested_scalars:
                records.append((str(key), nested_scalars))
            for period, record in value.items():
                if isinstance(record, Mapping):
                    records.append((str(period), dict(record)))
        return records

    @staticmethod
    def _period_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        period = item[0]
        match = _YEAR_RE.search(period)
        return (int(match.group(0)) if match else -1, period)

    @staticmethod
    def _parse_number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip().replace(",", "").replace(" ", "")
        if text.casefold() in {"", "-", "--", "none", "nan", "n/a"}:
            return None
        negative = text.startswith("(") and text.endswith(")")
        percent = text.endswith("%")
        try:
            number = float(text.strip("()%"))
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        if negative:
            number = -number
        return number / 100 if percent else number

    @staticmethod
    def _require(values: Mapping[str, float], *fields: str) -> None:
        missing = [field for field in fields if field not in values]
        if missing:
            raise CalcAgentError(f"计算缺少字段：{', '.join(missing)}")

    @classmethod
    def _default_code_generator(
        cls,
        question: str,
        values: Mapping[str, float],
    ) -> str:
        """根据问题选择可审计公式模板，结果统一写入 ``result``。"""

        normalized = question.casefold()
        if "速动比率" in normalized or "quick ratio" in normalized:
            cls._require(
                values,
                "current_assets",
                "inventories",
                "current_liabilities",
            )
            return (
                "quick_ratio = (current_assets - inventories) / "
                "current_liabilities\n"
                "result = {'quick_ratio': quick_ratio}"
            )
        if "流动比率" in normalized or "current ratio" in normalized:
            cls._require(values, "current_assets", "current_liabilities")
            return (
                "current_ratio = current_assets / current_liabilities\n"
                "result = {'current_ratio': current_ratio}"
            )
        if "roe" in normalized or "净资产收益率" in normalized:
            cls._require(values, "net_profit", "total_equity")
            return (
                "roe = net_profit / total_equity\n"
                "result = {'roe': roe}\n"
                "chart_data = {'type': 'bar', 'x': ['ROE'], 'y': [roe]}"
            )
        if "roa" in normalized or "总资产收益率" in normalized:
            cls._require(values, "net_profit", "total_assets")
            return "result = {'roa': net_profit / total_assets}"
        if "负债率" in normalized or "debt ratio" in normalized:
            cls._require(values, "total_liabilities", "total_assets")
            return (
                "result = {'debt_ratio': total_liabilities / total_assets}"
            )
        if "毛利率" in normalized or "gross margin" in normalized:
            cls._require(values, "operating_revenue", "operating_cost")
            return (
                "gross_margin = (operating_revenue - operating_cost) / "
                "operating_revenue\n"
                "result = {'gross_margin': gross_margin}"
            )
        if "净利率" in normalized or "net margin" in normalized:
            cls._require(values, "net_profit", "operating_revenue")
            return (
                "result = {'net_margin': net_profit / operating_revenue}"
            )

        # 💡【面试加分点】代码生成 Agent 与执行器必须解耦。前者负责
        # 理解指标口径并产出可审计代码，后者只负责 AST 安全校验、资源
        # 限制和确定性执行。即使未来替换 LLM，也不会削弱沙箱安全边界。
        raise CalcAgentError(
            "无法选择确定性财务公式，请在请求中提供 calculation_code"
        )


__all__ = ["CalcAgent", "CalcAgentError", "CodeGenerator"]
