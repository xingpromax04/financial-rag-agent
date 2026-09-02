"""LangGraph 多 Agent 主控路由器与状态机工作流。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
推荐依赖：``langgraph>=0.2``。未安装 LangGraph 时，模块使用相同
节点和转移规则的本地状态机，便于离线测试，但不提供检查点持久化。
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal, Protocol, TypeAlias, cast

from agents.audit_agent import AuditAgent
from agents.real_time_agent import matches_real_time_intent
from agents.state import (
    AgentName,
    AgentState,
    append_items,
    create_initial_state,
    merge_mappings,
    validate_state,
)

LOGGER = logging.getLogger(__name__)
AgentHandler: TypeAlias = Callable[[AgentState], Mapping[str, Any]]
RouterClassifier: TypeAlias = Callable[[str], Sequence[AgentName]]
Finalizer: TypeAlias = Callable[[AgentState], str]
DispatchTarget: TypeAlias = Literal[
    "rag", "market", "calculation", "audit", "retry", "finalize"
]

_CALCULATION_TERMS = (
    "计算",
    "比率",
    "同比",
    "环比",
    "增长率",
    "利润率",
    "负债率",
    "roe",
    "roa",
    "cagr",
    "calculate",
)
_AUDIT_TERMS = (
    "审计",
    "勾稽",
    "校验",
    "核对",
    "平衡",
    "矛盾",
    "一致",
    "对不上",
    "等于",
    "成立",
    "恒等式",
    "audit",
    "reconcile",
    "equation",
)
_HISTORICAL_TERMS = (
    "财报",
    "年报",
    "季报",
    "报告",
    "附注",
    "营业收入",
    "净利润",
    "资产",
    "负债",
    "现金流",
    "financial statement",
    "annual report",
)


class WorkflowError(RuntimeError):
    """多 Agent 图构建或状态执行失败。"""


class CompiledWorkflow(Protocol):
    """LangGraph CompiledStateGraph 与本地回退引擎的共同接口。"""

    def invoke(
        self,
        state: AgentState,
        config: Mapping[str, Any] | None = None,
    ) -> AgentState: ...


def _contains_any(question: str, terms: Sequence[str]) -> bool:
    normalized = question.casefold()
    return any(term.casefold() in normalized for term in terms)


def _unique_agents(agents: Sequence[AgentName]) -> list[AgentName]:
    output: list[AgentName] = []
    for agent in agents:
        if agent not in output:
            output.append(agent)
    return output


def _merge_local_state(state: AgentState, update: Mapping[str, Any]) -> AgentState:
    """在本地回退引擎中复制 LangGraph Annotated reducer 的合并语义。"""

    merged = deepcopy(state)
    mutable_merged = cast(dict[str, Any], merged)
    append_fields = {"messages", "retrieved_context", "errors", "workflow_trace"}
    mapping_fields = {"financial_data", "market_data"}
    for key, value in update.items():
        if key in append_fields:
            mutable_merged[key] = append_items(mutable_merged.get(key), value)
        elif key in mapping_fields:
            mutable_merged[key] = merge_mappings(mutable_merged.get(key), value)
        else:
            mutable_merged[key] = deepcopy(value)
    validate_state(merged)
    return merged


class _LocalCompiledWorkflow:
    """无 LangGraph 时使用的同步状态机，主要服务于单元测试。"""

    def __init__(self, owner: "MasterAgent") -> None:
        self.owner = owner

    def invoke(
        self,
        state: AgentState,
        config: Mapping[str, Any] | None = None,
    ) -> AgentState:
        del config
        current = deepcopy(state)
        node_name: DispatchTarget | Literal["router", "dispatcher"] = "router"
        max_steps = max(30, (current.get("max_retries", 2) + 1) * 12)

        for _ in range(max_steps):
            if node_name == "router":
                current = _merge_local_state(current, self.owner._router_node(current))
                node_name = "dispatcher"
            elif node_name == "dispatcher":
                current = _merge_local_state(
                    current, self.owner._dispatcher_node(current)
                )
                node_name = self.owner._dispatch_route(current)
            elif node_name == "rag":
                current = _merge_local_state(current, self.owner._rag_node(current))
                node_name = "dispatcher"
            elif node_name == "market":
                current = _merge_local_state(current, self.owner._market_node(current))
                node_name = "dispatcher"
            elif node_name == "calculation":
                current = _merge_local_state(
                    current, self.owner._calculation_node(current)
                )
                node_name = "dispatcher"
            elif node_name == "audit":
                current = _merge_local_state(current, self.owner._audit_node(current))
                node_name = "dispatcher"
            elif node_name == "retry":
                current = _merge_local_state(current, self.owner._retry_node(current))
                node_name = "dispatcher"
            elif node_name == "finalize":
                return _merge_local_state(current, self.owner._finalize_node(current))

        raise WorkflowError(f"本地状态机超过最大步数 {max_steps}")


class MasterAgent:
    """动态路由 RAG、实时数据、计算与审计 Agent 的主控编排器。"""

    def __init__(
        self,
        *,
        rag_handler: AgentHandler | None = None,
        market_handler: AgentHandler | None = None,
        calculation_handler: AgentHandler | None = None,
        audit_agent: AuditAgent | None = None,
        router_classifier: RouterClassifier | None = None,
        finalizer: Finalizer | None = None,
    ) -> None:
        self.rag_handler = rag_handler
        self.market_handler = market_handler
        self.calculation_handler = calculation_handler
        self.audit_agent = audit_agent or AuditAgent()
        self.router_classifier = router_classifier
        self.finalizer = finalizer
        self._compiled: CompiledWorkflow | None = None

    def build_workflow(self, *, checkpointer: Any = None) -> CompiledWorkflow:
        """构建并编译 LangGraph StateGraph。

        【原理说明】
        - ``StateGraph(AgentState)`` 声明所有节点共享的类型化状态。
        - ``add_node`` 注册纯节点函数：输入当前状态，输出状态增量。
        - ``add_edge`` 定义无条件转移；专业 Agent 完成后都回到 Dispatcher。
        - ``add_conditional_edges`` 读取 plan、completed_agents 和重试状态，
          在运行时决定下一个节点，因此同一张图可处理不同问题。
        """

        try:
            graph_module = importlib.import_module("langgraph.graph")
        except ImportError:
            LOGGER.warning("langgraph 未安装，使用本地状态机回退引擎")
            return _LocalCompiledWorkflow(self)

        state_graph = graph_module.StateGraph(AgentState)
        state_graph.add_node("router", self._router_node)
        state_graph.add_node("dispatcher", self._dispatcher_node)
        state_graph.add_node("rag", self._rag_node)
        state_graph.add_node("market", self._market_node)
        state_graph.add_node("calculation", self._calculation_node)
        state_graph.add_node("audit", self._audit_node)
        state_graph.add_node("retry", self._retry_node)
        state_graph.add_node("finalize", self._finalize_node)

        state_graph.add_edge(graph_module.START, "router")
        state_graph.add_edge("router", "dispatcher")
        for node_name in ("rag", "market", "calculation", "audit", "retry"):
            state_graph.add_edge(node_name, "dispatcher")
        state_graph.add_conditional_edges(
            "dispatcher",
            self._dispatch_route,
            {
                "rag": "rag",
                "market": "market",
                "calculation": "calculation",
                "audit": "audit",
                "retry": "retry",
                "finalize": "finalize",
            },
        )
        state_graph.add_edge("finalize", graph_module.END)
        try:
            return cast(
                CompiledWorkflow,
                state_graph.compile(checkpointer=checkpointer),
            )
        except Exception as exc:
            raise WorkflowError("LangGraph 编译失败") from exc

    def invoke(
        self,
        question_or_state: str | AgentState,
        *,
        max_retries: int = 2,  # 🔧【可调参数】计算/审计失败后的最大补充检索次数
        config: Mapping[str, Any] | None = None,
    ) -> AgentState:
        """运行工作流并返回最终共享状态。"""

        state = (
            create_initial_state(question_or_state, max_retries=max_retries)
            if isinstance(question_or_state, str)
            else deepcopy(question_or_state)
        )
        validate_state(state)
        if self._compiled is None:
            self._compiled = self.build_workflow()

        run_config = dict(config or {})
        run_config.setdefault(
            "recursion_limit",
            max(30, (state.get("max_retries", max_retries) + 1) * 12),
        )
        try:
            result = self._compiled.invoke(state, config=run_config)
        except Exception as exc:
            if isinstance(exc, WorkflowError):
                raise
            raise WorkflowError("Multi-Agent 工作流执行失败") from exc
        validate_state(result)
        return result

    def _default_plan(self, question: str) -> list[AgentName]:
        # 实时意图词由 RealTimeAgent 统一维护，避免路由与节点规则漂移。
        needs_market = matches_real_time_intent(question)
        needs_calculation = _contains_any(question, _CALCULATION_TERMS)
        needs_audit = _contains_any(question, _AUDIT_TERMS) or needs_calculation
        needs_history = _contains_any(question, _HISTORICAL_TERMS)
        needs_rag = (
            needs_history or needs_calculation or needs_audit or not needs_market
        )

        plan: list[AgentName] = []
        if needs_rag:
            plan.append("rag")
        if needs_market:
            plan.append("market")
        if needs_calculation:
            plan.append("calculation")
        if needs_audit:
            plan.append("audit")
        return plan or ["rag"]

    def _router_node(self, state: AgentState) -> dict[str, Any]:
        """Router Node：将用户意图转换为有序 Agent 计划。"""

        question = state["question"]
        raw_plan = (
            self.router_classifier(question)
            if self.router_classifier is not None
            else self._default_plan(question)
        )
        valid_agents = {"rag", "market", "calculation", "audit"}
        if any(agent not in valid_agents for agent in raw_plan):
            raise WorkflowError("Router 返回了未知 Agent")
        plan = _unique_agents(list(raw_plan))
        if not plan:
            plan = ["rag"]
        return {
            "route": "+".join(plan),
            "plan": plan,
            "completed_agents": [],
            "next_action": "continue",
            "workflow_trace": [f"router:{'+'.join(plan)}"],
        }

    @staticmethod
    def _dispatcher_node(state: AgentState) -> dict[str, Any]:
        """Dispatcher Node 不执行业务，只留下可观测的调度轨迹。"""

        del state
        return {"workflow_trace": ["dispatcher"]}

    @staticmethod
    def _dispatch_route(state: AgentState) -> DispatchTarget:
        """Conditional Edge：选择重试、下一个专业节点或结束。"""

        if state.get("next_action") == "retry":
            if state.get("retry_count", 0) < state.get("max_retries", 0):
                return "retry"
            return "finalize"
        completed = set(state.get("completed_agents", []))
        for agent in state.get("plan", []):
            if agent not in completed:
                return agent
        return "finalize"

    def _rag_node(self, state: AgentState) -> dict[str, Any]:
        return self._run_specialist("rag", self.rag_handler, state)

    def _market_node(self, state: AgentState) -> dict[str, Any]:
        return self._run_specialist("market", self.market_handler, state)

    def _calculation_node(self, state: AgentState) -> dict[str, Any]:
        return self._run_specialist("calculation", self.calculation_handler, state)

    def _audit_node(self, state: AgentState) -> dict[str, Any]:
        update = dict(self.audit_agent(state))
        if update.get("next_action") != "retry":
            completed = list(state.get("completed_agents", []))
            if "audit" not in completed:
                completed.append("audit")
            update["completed_agents"] = completed
        return update

    def _run_specialist(
        self,
        agent: AgentName,
        handler: AgentHandler | None,
        state: AgentState,
    ) -> dict[str, Any]:
        """统一执行专业 Agent，限制其只能修改自己拥有的业务字段。"""

        if handler is None:
            return self._failure_update(agent, f"{agent} handler 未配置")
        try:
            raw_update = handler(deepcopy(state))
            if not isinstance(raw_update, Mapping):
                raise TypeError("Agent handler 必须返回 Mapping")
            update = self._sanitize_handler_update(agent, raw_update)
        except Exception as exc:
            LOGGER.exception("%s Agent 执行失败", agent)
            return self._failure_update(agent, f"{agent} Agent 执行失败：{exc}")

        if agent == "calculation":
            output = update.get("code_output")
            if hasattr(output, "to_dict"):
                output = output.to_dict()
                update["code_output"] = output
            if not isinstance(output, Mapping) or not bool(output.get("success")):
                message = (
                    output.get("error_message", "未返回有效计算结果")
                    if isinstance(output, Mapping)
                    else "未返回有效计算结果"
                )
                return {
                    **update,
                    **self._failure_update(agent, str(message)),
                }

        completed = list(state.get("completed_agents", []))
        if agent not in completed:
            completed.append(agent)
        return {
            **update,
            "completed_agents": completed,
            "next_action": "continue",
            "last_failure": None,
            "workflow_trace": [f"{agent}:completed"],
        }

    @staticmethod
    def _sanitize_handler_update(
        agent: AgentName,
        update: Mapping[str, Any],
    ) -> dict[str, Any]:
        allowed_fields: dict[AgentName, set[str]] = {
            "rag": {"retrieved_context", "financial_data", "messages"},
            "market": {"market_data", "financial_data", "messages"},
            "calculation": {
                "calculation_code",
                "code_output",
                "financial_data",
                "messages",
            },
            "audit": {"audit_results", "audit_passed", "messages"},
        }
        return {
            key: value
            for key, value in update.items()
            if key in allowed_fields[agent]
        }

    @staticmethod
    def _failure_update(agent: AgentName, message: str) -> dict[str, Any]:
        return {
            "next_action": "retry",
            "last_failure": agent,
            "errors": [message],
            "workflow_trace": [f"{agent}:failed"],
        }

    def _retry_node(self, state: AgentState) -> dict[str, Any]:
        """Retry Node：增加计数，并重置需要重跑的节点。"""

        failure = state.get("last_failure")
        completed = list(state.get("completed_agents", []))
        plan = list(state.get("plan", []))

        if failure in {"calculation", "audit"}:
            # 计算/勾稽失败通常意味着数据不全，回到 RAG 做补充检索。
            for agent in ("rag", "calculation", "audit"):
                if agent in completed:
                    completed.remove(agent)
            if "rag" not in plan:
                plan.insert(0, "rag")
        elif failure in completed:
            completed.remove(failure)

        retry_count = state.get("retry_count", 0) + 1
        return {
            "plan": plan,
            "completed_agents": completed,
            "retry_count": retry_count,
            "next_action": "continue",
            "last_failure": None,
            "audit_results": [],
            "audit_passed": None,
            "workflow_trace": [f"retry:{retry_count}:{failure or 'unknown'}"],
        }

    def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        """Finalize Node：生成最终文本，并将工作流转移到 END。"""

        if self.finalizer is not None:
            try:
                answer = self.finalizer(deepcopy(state))
            except Exception as exc:
                raise WorkflowError("Finalizer 执行失败") from exc
        else:
            answer = self._default_final_answer(state)
        return {
            "final_answer": answer,
            "next_action": "finalize",
            "messages": [{"role": "assistant", "content": answer}],
            "workflow_trace": ["finalize"],
        }

    @staticmethod
    def _default_final_answer(state: AgentState) -> str:
        parts = [f"已完成工作流：{state.get('route', '未路由')}。"]
        if state.get("retrieved_context"):
            parts.append(f"召回 {len(state['retrieved_context'])} 条历史财报上下文。")
        if state.get("market_data"):
            parts.append("已获取实时市场数据。")
        code_output = state.get("code_output") or {}
        if code_output.get("success"):
            parts.append(f"计算结果：{code_output.get('result')}。")
        audit_results = state.get("audit_results", [])
        failed_rules = [
            item for item in audit_results if item.get("status") == "failed"
        ]
        if failed_rules:
            parts.append(f"风险提示：{len(failed_rules)} 条财务勾稽规则未通过。")
        elif state.get("audit_passed"):
            parts.append("财务勾稽校验已通过。")
        if state.get("errors"):
            parts.append(f"工作流异常记录：{state['errors'][-1]}")

        # 💡【面试加分点】“中央 Dispatcher + 共享状态”避免 Agent 相互
        # 直接调用形成网状耦合；Retry 又是图中的显式节点，因此每次重试
        # 都有 trace、上限和状态转移记录，不会变成难以定位的隐式递归。
        return "".join(parts)


def build_default_workflow(
    *,
    rag_handler: AgentHandler,
    market_handler: AgentHandler,
    calculation_handler: AgentHandler,
    audit_agent: AuditAgent | None = None,
    checkpointer: Any = None,
) -> CompiledWorkflow:
    """便捷函数：使用三个专业 handler 构建已编译工作流。"""

    master = MasterAgent(
        rag_handler=rag_handler,
        market_handler=market_handler,
        calculation_handler=calculation_handler,
        audit_agent=audit_agent,
    )
    return master.build_workflow(checkpointer=checkpointer)


__all__ = [
    "AgentHandler",
    "CompiledWorkflow",
    "MasterAgent",
    "WorkflowError",
    "build_default_workflow",
]
