"""多 Agent 财报分析系统的共享状态模型。

运行环境：Python 3.11，Conda 环境 ``rag_311``。

【状态流向】用户问题 -> Router -> 专业 Agent 节点 -> Audit -> Finalize。
每个节点只返回需要更新的字段，LangGraph 按 reducer 将增量合并回
``AgentState``，下一个节点因此能读取前序 Agent 的结果。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

AgentName = Literal["rag", "market", "calculation", "audit"]
AuditStatus = Literal["passed", "failed", "skipped"]


class AgentMessage(TypedDict, total=False):
    """统一消息格式，可后续转换为 LangChain BaseMessage。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str
    metadata: dict[str, Any]


class AuditResultRecord(TypedDict, total=False):
    """单条财务勾稽规则的可序列化结果。"""

    rule_id: str
    rule_name: str
    period: str
    status: AuditStatus
    severity: Literal["info", "warning", "critical"]
    actual: float | None
    expected: float | None
    difference: float | None
    tolerance: float | None
    message: str
    evidence: dict[str, float]


def append_items(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """LangGraph reducer：列表字段使用追加而非覆盖语义。"""

    return [*(left or []), *(right or [])]


def merge_mappings(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    """LangGraph reducer：补充检索只覆盖同名键，保留其他数据。"""

    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    """所有 Agent 通过的唯一共享状态。

    【原理说明】多 Agent 不应通过隐式全局变量或直接互相调用传值。
    显式 Shared State 使每次状态转移都可记录、可回放、可持久化，
    并能由类型检查器及测试代码验证 Agent 之间的契约。
    """

    trace_id: str
    question: str
    messages: Annotated[list[AgentMessage], append_items]

    # 业务数据：RAG 历史财报、实时市场、计算与审计结果。
    retrieved_context: Annotated[list[dict[str, Any]], append_items]
    financial_data: Annotated[dict[str, Any], merge_mappings]
    market_data: Annotated[dict[str, Any], merge_mappings]
    calculation_code: str
    code_output: dict[str, Any] | None
    audit_results: list[AuditResultRecord]
    audit_passed: bool | None

    # 编排控制：plan 是 Router 产生的节点顺序，completed_agents 防止重复执行。
    route: str
    plan: list[AgentName]
    completed_agents: list[AgentName]
    next_action: Literal["continue", "retry", "finalize"]
    last_failure: AgentName | None
    retry_count: int
    max_retries: int

    # 可观测性字段使线上问题可以定位到具体节点。
    errors: Annotated[list[str], append_items]
    workflow_trace: Annotated[list[str], append_items]
    final_answer: str | None


class StateValidationError(ValueError):
    """初始状态或节点状态违反全局契约。"""


def create_initial_state(
    question: str,
    *,
    financial_data: dict[str, Any] | None = None,
    max_retries: int = 2,  # 🔧【可调参数】防止 Agent 在失败状态下无限循环
    trace_id: str | None = None,
) -> AgentState:
    """创建字段完整、可直接交给 LangGraph 的初始状态。"""

    if not isinstance(question, str) or not question.strip():
        raise StateValidationError("用户问题不能为空")
    if max_retries < 0:
        raise StateValidationError("max_retries 不能小于 0")

    state: AgentState = {
        "trace_id": trace_id or uuid4().hex,
        "question": question.strip(),
        "messages": [{"role": "user", "content": question.strip()}],
        "retrieved_context": [],
        "financial_data": deepcopy(financial_data or {}),
        "market_data": {},
        "code_output": None,
        "audit_results": [],
        "audit_passed": None,
        "route": "",
        "plan": [],
        "completed_agents": [],
        "next_action": "continue",
        "last_failure": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "errors": [],
        "workflow_trace": ["start"],
        "final_answer": None,
    }
    validate_state(state)
    return state


def validate_state(state: AgentState) -> None:
    """对 TypedDict 不能在运行时强制的核心约束进行补充校验。"""

    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise StateValidationError("AgentState.question 必须是非空字符串")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 0)
    if not isinstance(retry_count, int) or retry_count < 0:
        raise StateValidationError("retry_count 必须是非负整数")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise StateValidationError("max_retries 必须是非负整数")
    valid_agents = {"rag", "market", "calculation", "audit"}
    if any(agent not in valid_agents for agent in state.get("plan", [])):
        raise StateValidationError("plan 包含未知 Agent")


__all__ = [
    "AgentMessage",
    "AgentName",
    "AgentState",
    "AuditResultRecord",
    "StateValidationError",
    "append_items",
    "create_initial_state",
    "merge_mappings",
    "validate_state",
]
