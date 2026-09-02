"""Streamlit 财务报告多 Agent 分析面板。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
启动命令：``streamlit run app.py``。
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="LedgerMind | Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #fbfcfd; color: #182026; }
    [data-testid="stSidebar"] { background: #f2f4f5; border-right: 1px solid #dfe3e6; }
    .block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #15252d; letter-spacing: 0; }
    h1 { font-size: 1.85rem !important; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe3e6;
        border-radius: 6px;
        padding: 0.7rem 0.8rem;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #e3e7e9;
        border-radius: 6px;
        background: #ffffff;
    }
    .backend-ok { color: #176b55; font-weight: 600; }
    .backend-warn { color: #9a5a13; font-weight: 600; }
    .small-label { color: #5e6b73; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_session_state() -> None:
    """初始化跨 rerun 持久的用户会话数据。

    【Streamlit 状态管理】Streamlit 在控件交互后会从顶部重新执行
    整个脚本，普通局部变量会丢失。``st.session_state`` 按浏览器会话
    保存对话、上传记录和选项，因此 rerun 不会清空用户工作台。
    """

    defaults = {
        "messages": [],
        "documents": [],
        "analysis_mode": "fast",
        "api_url": DEFAULT_API_URL,
        "symbol": "",
        "calculation_code": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("detail") or payload)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


@st.cache_data(ttl=10, show_spinner=False)
def _health(api_url: str) -> dict[str, Any]:
    with httpx.Client(timeout=5) as client:
        response = client.get(f"{api_url}/health")
        response.raise_for_status()
        return response.json()


def _upload_report(
    api_url: str,
    uploaded_file: Any,
    year: int | None,
) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {"year": str(year)} if year else {}
    with httpx.Client(timeout=180) as client:
        response = client.post(f"{api_url}/upload", files=files, data=data)
    if response.is_error:
        raise RuntimeError(_error_detail(response))
    return response.json()


def _analyze(
    api_url: str,
    question: str,
    mode: str,
    symbol: str,
    calculation_code: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question, "mode": mode}
    if symbol.strip():
        payload["symbol"] = symbol.strip()
    if calculation_code.strip():
        payload["calculation_code"] = calculation_code.strip()
    with httpx.Client(timeout=240) as client:
        response = client.post(f"{api_url}/analyze", json=payload)
    if response.is_error:
        raise RuntimeError(_error_detail(response))
    return response.json()


def _render_chart(chart_data: Any) -> None:
    if not isinstance(chart_data, dict):
        return
    x_values = chart_data.get("x")
    y_values = chart_data.get("y")
    if not isinstance(x_values, list) or not isinstance(y_values, list):
        return
    if len(x_values) != len(y_values) or not x_values:
        return
    frame = pd.DataFrame({"label": x_values, "value": y_values}).set_index("label")
    if chart_data.get("type") == "line":
        st.line_chart(frame, color="#176b55")
    else:
        st.bar_chart(frame, color="#a4463a")


def _render_analysis(payload: dict[str, Any]) -> None:
    contexts = payload.get("retrieved_context") or []
    audit_results = payload.get("audit_results") or []
    failed_audits = [item for item in audit_results if item.get("status") == "failed"]
    market_data = payload.get("market_data") or {}
    code_output = payload.get("code_output") or {}

    metric_columns = st.columns(4)
    metric_columns[0].metric("路由", payload.get("route") or "-")
    metric_columns[1].metric("检索证据", len(contexts))
    metric_columns[2].metric("审计异常", len(failed_audits))
    metric_columns[3].metric("轨迹步数", len(payload.get("workflow_trace") or []))

    tabs = st.tabs(["检索证据", "计算与图表", "勾稽审计", "Agent 轨迹"])
    with tabs[0]:
        if market_data:
            st.subheader("实时市场数据")
            quote = market_data.get("quote") or {}
            fundamentals = market_data.get("fundamentals") or {}
            if quote:
                st.dataframe(pd.DataFrame([quote]), use_container_width=True)
            if fundamentals:
                st.dataframe(pd.DataFrame([fundamentals]), use_container_width=True)
        for index, context in enumerate(contexts, start=1):
            metadata = context.get("metadata") or {}
            label = (
                metadata.get("table_name")
                or metadata.get("filename")
                or f"Chunk {index}"
            )
            score = context.get("score")
            score_label = f" | {score:.4f}" if isinstance(score, (int, float)) else ""
            with st.expander(f"{label}{score_label}", expanded=index == 1):
                st.markdown(context.get("text") or "")
                st.json(metadata, expanded=False)

    with tabs[1]:
        calculation_code = payload.get("calculation_code")
        if calculation_code:
            st.code(calculation_code, language="python", line_numbers=True)
        if code_output:
            if code_output.get("success"):
                st.json(code_output.get("result"), expanded=True)
                _render_chart(code_output.get("chart_data"))
            else:
                st.error(code_output.get("error_message") or "计算失败")

    with tabs[2]:
        if audit_results:
            audit_frame = pd.DataFrame(audit_results)
            preferred = [
                "period",
                "rule_name",
                "status",
                "severity",
                "actual",
                "expected",
                "difference",
                "message",
            ]
            visible = [column for column in preferred if column in audit_frame.columns]
            st.dataframe(
                audit_frame[visible], use_container_width=True, hide_index=True
            )

    with tabs[3]:
        trace = payload.get("workflow_trace") or []
        if trace:
            st.code("\n".join(trace), language="text")
        st.caption(f"Trace ID: {payload.get('trace_id', '-')}")

    for error in payload.get("errors") or []:
        st.warning(error)


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("分析工作台")
        try:
            health = _health(st.session_state.api_url)
            css_class = "backend-ok" if health.get("status") == "ok" else "backend-warn"
            st.markdown(
                f'<span class="{css_class}">{health.get("vector_backend", "-")}</span>'
                '<span class="small-label"> · '
                f'{health.get("indexed_chunks", 0)} chunks</span>',
                unsafe_allow_html=True,
            )
        except (httpx.HTTPError, RuntimeError):
            st.markdown(
                '<span class="backend-warn">API offline</span>',
                unsafe_allow_html=True,
            )

        mode_options = ["fast", "deep"]

        def format_mode(value: str) -> str:
            return "极速分析" if value == "fast" else "深度审计"

        if hasattr(st, "segmented_control"):
            mode_label = st.segmented_control(
                "分析模式",
                options=mode_options,
                format_func=format_mode,
                default=st.session_state.analysis_mode,
            )
        else:
            mode_label = st.radio(
                "分析模式",
                options=mode_options,
                format_func=format_mode,
                index=mode_options.index(st.session_state.analysis_mode),
                horizontal=True,
            )
        if mode_label:
            st.session_state.analysis_mode = mode_label

        st.divider()
        uploaded_file = st.file_uploader("财务报告", type=["pdf", "docx"])
        year_value = st.number_input(
            "报告年份",
            min_value=1900,
            max_value=2100,
            value=2024,
            step=1,
        )
        if st.button("上传并入库", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("请选择 PDF 或 DOCX 文件")
            else:
                try:
                    with st.spinner("正在解析与切片…"):
                        result = _upload_report(
                            st.session_state.api_url,
                            uploaded_file,
                            int(year_value),
                        )
                    st.session_state.documents.append(result)
                    _health.clear()
                    st.success(f"{result['filename']} 已入库")
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(str(exc))

        if st.session_state.documents:
            st.subheader("本次会话文档")
            for document in st.session_state.documents:
                st.caption(
                    f"{document['filename']} · {document['chunks_written']} chunks"
                )

        st.divider()
        st.session_state.symbol = st.text_input(
            "股票代码",
            value=st.session_state.symbol,
            placeholder="600519 / AAPL",
        )
        with st.expander("计算代码"):
            st.session_state.calculation_code = st.text_area(
                "Python",
                value=st.session_state.calculation_code,
                height=150,
                label_visibility="collapsed",
                placeholder="result = net_profit / total_equity",
            )


def main() -> None:
    _initialize_session_state()
    _render_sidebar()

    st.title("LedgerMind")
    st.caption("RAG Multi-Agent Financial Analysis")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("analysis"):
                _render_analysis(message["analysis"])

    prompt = st.chat_input("输入财报、行情或勾稽问题")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        execution_status = st.status("Agent workflow running", expanded=True)
        try:
            payload = _analyze(
                st.session_state.api_url,
                prompt,
                st.session_state.analysis_mode,
                st.session_state.symbol,
                st.session_state.calculation_code,
            )
            execution_status.write(" → ".join(payload.get("workflow_trace") or []))
            execution_status.update(label="Agent workflow complete", state="complete")
            answer = payload.get("answer") or "分析已完成。"
            st.markdown(answer)
            _render_analysis(payload)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "analysis": payload}
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            execution_status.update(label="Agent workflow failed", state="error")
            message = f"请求失败：{exc}"
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message}
            )


# 💡【面试加分点】前端不直接 import Agent 或向量库，只依赖
# FastAPI 的稳定 JSON 契约。这种前后端分离让 Streamlit 可被 React/Vue
# 替换，也允许后端独立水平扩容和接入身份认证、限流与审计日志。
if __name__ == "__main__":
    main()
