"""财务勾稽关系审计 Agent 与确定性业务规则引擎。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
本模块不调用 LLM，所有算式都是可测试、可解释的确定性规则。
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from agents.state import AgentState, AuditResultRecord

LOGGER = logging.getLogger(__name__)
Severity: TypeAlias = Literal["info", "warning", "critical"]
Formula: TypeAlias = Callable[[Mapping[str, float]], tuple[float, dict[str, float]]]

_PERIOD_RE = re.compile(r"(?:19|20)\d{2}(?:[-Q年季度报中期末\d]*)", re.IGNORECASE)
_STATEMENT_KEYS = {
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "资产负债表",
    "利润表",
    "现金流量表",
}
_STATEMENT_KEY_FOLDS = {key.casefold() for key in _STATEMENT_KEYS}

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("total_assets", "assets", "资产总计", "总资产"),
    "total_liabilities": (
        "total_liabilities",
        "liabilities",
        "负债合计",
        "总负债",
    ),
    "total_equity": (
        "total_equity",
        "equity",
        "owners_equity",
        "所有者权益合计",
        "股东权益合计",
        "净资产",
    ),
    "operating_revenue": (
        "operating_revenue",
        "revenue",
        "营业收入",
        "营业总收入",
    ),
    "operating_cost": ("operating_cost", "cost_of_revenue", "营业成本"),
    "period_expenses": ("period_expenses", "期间费用"),
    "selling_expenses": ("selling_expenses", "销售费用"),
    "administrative_expenses": (
        "administrative_expenses",
        "管理费用",
    ),
    "financial_expenses": ("financial_expenses", "财务费用"),
    "operating_profit": ("operating_profit", "营业利润"),
    "total_profit": ("total_profit", "profit_before_tax", "利润总额"),
    "income_tax": ("income_tax", "income_tax_expense", "所得税费用"),
    "net_profit": ("net_profit", "净利润"),
    "beginning_cash": (
        "beginning_cash",
        "cash_at_beginning",
        "期初现金及现金等价物余额",
    ),
    "net_cash_increase": (
        "net_cash_increase",
        "现金及现金等价物净增加额",
    ),
    "ending_cash": (
        "ending_cash",
        "cash_at_end",
        "期末现金及现金等价物余额",
    ),
}


class AuditError(RuntimeError):
    """审计数据格式或规则执行异常。"""


@dataclass(frozen=True, slots=True)
class AuditRule:
    """一条线性勾稽规则的声明式定义。"""

    rule_id: str
    rule_name: str
    actual_field: str
    required_fields: tuple[str, ...]
    expected_formula: Formula
    severity: Severity = "warning"


def _balance_formula(values: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    evidence = {
        "total_liabilities": values["total_liabilities"],
        "total_equity": values["total_equity"],
    }
    return sum(evidence.values()), evidence


def _operating_profit_formula(
    values: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    evidence = {
        "operating_revenue": values["operating_revenue"],
        "operating_cost": values["operating_cost"],
        "period_expenses": values["period_expenses"],
    }
    expected = (
        evidence["operating_revenue"]
        - evidence["operating_cost"]
        - evidence["period_expenses"]
    )
    return expected, evidence


def _net_profit_formula(values: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    evidence = {
        "total_profit": values["total_profit"],
        "income_tax": values["income_tax"],
    }
    return evidence["total_profit"] - evidence["income_tax"], evidence


def _ending_cash_formula(values: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    evidence = {
        "beginning_cash": values["beginning_cash"],
        "net_cash_increase": values["net_cash_increase"],
    }
    return evidence["beginning_cash"] + evidence["net_cash_increase"], evidence


DEFAULT_AUDIT_RULES: tuple[AuditRule, ...] = (
    AuditRule(
        rule_id="balance_sheet_equation",
        rule_name="资产 = 负债 + 所有者权益",
        actual_field="total_assets",
        required_fields=("total_liabilities", "total_equity"),
        expected_formula=_balance_formula,
        severity="critical",
    ),
    AuditRule(
        rule_id="operating_profit_equation",
        rule_name="营业利润 = 营业收入 - 营业成本 - 期间费用",
        actual_field="operating_profit",
        required_fields=("operating_revenue", "operating_cost", "period_expenses"),
        expected_formula=_operating_profit_formula,
    ),
    AuditRule(
        rule_id="net_profit_equation",
        rule_name="净利润 = 利润总额 - 所得税费用",
        actual_field="net_profit",
        required_fields=("total_profit", "income_tax"),
        expected_formula=_net_profit_formula,
    ),
    AuditRule(
        rule_id="cash_reconciliation",
        rule_name="期末现金 = 期初现金 + 现金净增加额",
        actual_field="ending_cash",
        required_fields=("beginning_cash", "net_cash_increase"),
        expected_formula=_ending_cash_formula,
    ),
)


def _coerce_number(value: Any) -> float | None:
    """识别逗号、会计括号负数和常见空值。"""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace(",", "").replace(" ", "")
    if text.casefold() in {"", "-", "--", "n/a", "na", "none", "null", "nan"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").removesuffix("%")
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return -number if negative else number


def _canonicalize_record(record: Mapping[str, Any]) -> dict[str, float]:
    """将中英文报表字段映射为审计规则的标准字段。"""

    normalized_keys = {
        str(key).strip().casefold(): value for key, value in record.items()
    }
    canonical: dict[str, float] = {}
    for field, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            value = normalized_keys.get(alias.casefold())
            number = _coerce_number(value)
            if number is not None:
                canonical[field] = number
                break

    expense_fields = (
        "selling_expenses",
        "administrative_expenses",
        "financial_expenses",
    )
    if "period_expenses" not in canonical and all(
        field in canonical for field in expense_fields
    ):
        canonical["period_expenses"] = sum(canonical[field] for field in expense_fields)
    return canonical


def _flatten_statement_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if str(key).casefold() in _STATEMENT_KEY_FOLDS and isinstance(value, Mapping):
            flattened.update(value)
        elif not isinstance(value, Mapping):
            flattened[key] = value
    return flattened


def _period_records(financial_data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """兼容单期、按年份分组和 records 列表三种输入结构。"""

    records_value = financial_data.get("records")
    if isinstance(records_value, Sequence) and not isinstance(
        records_value, (str, bytes)
    ):
        records: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(records_value):
            if not isinstance(item, Mapping):
                continue
            period_value = item.get("period") or item.get("year")
            period = str(period_value or f"record_{index + 1}")
            records[period] = dict(item)
        return records

    nested_mappings = {
        str(key): value
        for key, value in financial_data.items()
        if isinstance(value, Mapping)
    }
    period_mappings = {
        key: _flatten_statement_mapping(value)
        for key, value in nested_mappings.items()
        if _PERIOD_RE.search(key)
    }
    if period_mappings:
        return period_mappings

    # 兼容 {balance_sheet: {2024: {...}}, income_statement: {2024: {...}}}
    # 这种“报表 -> 期间”结构，合并后再执行跨报表勾稽。
    statement_periods: dict[str, dict[str, Any]] = {}
    for statement_name, statement_data in nested_mappings.items():
        if statement_name.casefold() not in _STATEMENT_KEY_FOLDS:
            continue
        for period, record in statement_data.items():
            if _PERIOD_RE.search(str(period)) and isinstance(record, Mapping):
                statement_periods.setdefault(str(period), {}).update(record)
    if statement_periods:
        return statement_periods

    flattened = _flatten_statement_mapping(financial_data)
    return {"current": flattened} if flattened else {}


class AuditAgent:
    """执行跨报表勾稽校验，并作为 LangGraph 节点返回状态增量。"""

    def __init__(
        self,
        rules: Sequence[AuditRule] = DEFAULT_AUDIT_RULES,
        *,
        relative_tolerance: float = 0.005,  # 🔧【可调参数】默认允许 0.5% 相对误差
        absolute_tolerance: float = 1.0,  # 🔧【可调参数】适应“元/万元”四舍五入
        fail_on_no_applicable_rules: bool = True,
    ) -> None:
        if relative_tolerance < 0 or absolute_tolerance < 0:
            raise ValueError("审计容差不能小于 0")
        self.rules = tuple(rules)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.fail_on_no_applicable_rules = fail_on_no_applicable_rules

    def audit(self, financial_data: Mapping[str, Any]) -> list[AuditResultRecord]:
        """对所有期间和规则执行校验，缺失字段记为 skipped。"""

        if not isinstance(financial_data, Mapping):
            raise AuditError("financial_data 必须是映射类型")
        periods = _period_records(financial_data)
        if not periods:
            return []

        results: list[AuditResultRecord] = []
        for period, raw_record in periods.items():
            values = _canonicalize_record(raw_record)
            for rule in self.rules:
                results.append(self._evaluate_rule(period, values, rule))
        return results

    def _evaluate_rule(
        self,
        period: str,
        values: Mapping[str, float],
        rule: AuditRule,
    ) -> AuditResultRecord:
        required = (rule.actual_field, *rule.required_fields)
        missing = [field for field in required if field not in values]
        if missing:
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "period": period,
                "status": "skipped",
                "severity": "info",
                "actual": values.get(rule.actual_field),
                "expected": None,
                "difference": None,
                "tolerance": None,
                "message": f"缺少字段：{', '.join(missing)}",
                "evidence": {},
            }

        actual = values[rule.actual_field]
        try:
            expected, evidence = rule.expected_formula(values)
        except Exception as exc:
            raise AuditError(f"规则 {rule.rule_id} 执行失败") from exc
        difference = actual - expected
        scale = max(abs(actual), abs(expected), 1.0)
        tolerance = max(self.absolute_tolerance, self.relative_tolerance * scale)
        passed = abs(difference) <= tolerance
        return {
            "rule_id": rule.rule_id,
            "rule_name": rule.rule_name,
            "period": period,
            "status": "passed" if passed else "failed",
            "severity": "info" if passed else rule.severity,
            "actual": actual,
            "expected": expected,
            "difference": difference,
            "tolerance": tolerance,
            "message": (
                f"勾稽通过：差额 {difference:.6g}"
                if passed
                else f"勾稽失败：差额 {difference:.6g} 超过容差 {tolerance:.6g}"
            ),
            "evidence": evidence,
        }

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """LangGraph Audit Node：读取共享状态并返回审计状态增量。"""

        try:
            results = self.audit(state.get("financial_data", {}))
        except AuditError as exc:
            LOGGER.exception("财务勾稽审计异常")
            return {
                "audit_results": [],
                "audit_passed": False,
                "next_action": "retry",
                "last_failure": "audit",
                "errors": [str(exc)],
                "workflow_trace": ["audit:error"],
            }

        evaluated = [result for result in results if result["status"] != "skipped"]
        failures = [result for result in results if result["status"] == "failed"]
        passed = not failures and (
            bool(evaluated) or not self.fail_on_no_applicable_rules
        )
        summary = (
            f"审计完成：{len(evaluated)} 条有效规则，{len(failures)} 条失败"
        )

        # 💡【面试加分点】LLM 擅长理解附注和分析风险原因，但可能
        # 忽视最基本的会计恒等式。Rule Engine 先用确定性公式拦截矛盾数据，
        # 再让 LLM 对差异做语义解释，形成“规则保底 + 大模型推理”的双层审计。
        return {
            "audit_results": results,
            "audit_passed": passed,
            "next_action": "continue" if passed else "retry",
            "last_failure": None if passed else "audit",
            "messages": [{"role": "tool", "name": "audit_agent", "content": summary}],
            "workflow_trace": ["audit:passed" if passed else "audit:failed"],
        }


__all__ = [
    "AuditAgent",
    "AuditError",
    "AuditRule",
    "DEFAULT_AUDIT_RULES",
]
