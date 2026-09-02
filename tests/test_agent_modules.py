"""独立 Agent 节点及 LangGraph 组合流程的离线回归测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from agents.audit_agent import AuditAgent
from agents.calc_agent import CalcAgent
from agents.master_agent import MasterAgent
from agents.rag_agent import RAGAgent
from agents.real_time_agent import RealTimeAgent, extract_symbol
from agents.state import create_initial_state
from core.rag.retriever import HybridSearchResult
from core.rag.vector_store import DocumentChunk


class _RetrieverStub:
    def __init__(self) -> None:
        self.result = HybridSearchResult(
            chunk=DocumentChunk(
                chunk_id="quick-ratio-table",
                text=(
                    "### 资产负债表\n\n"
                    "| 项目 | 2024年 |\n"
                    "| --- | --- |\n"
                    "| 流动资产合计 | 300 |\n"
                    "| 存货 | 60 |\n"
                    "| 流动负债合计 | 160 |"
                ),
                metadata={
                    "source": "doc-1:report.docx",
                    "chunk_type": "table",
                    "year": 2024,
                },
            ),
            score=0.03,
            semantic_rank=1,
            keyword_rank=1,
            semantic_score=0.9,
            keyword_score=3.2,
        )

    def retrieve(self, query: str, *, top_k: int = 8):
        del query, top_k
        return [self.result]


class _MarketClientStub:
    def get_realtime_quote(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{"symbol": symbol, "price": 100.0, "volume": pd.NA}]
        )

    def get_financial_indicators(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": symbol, "pe_ratio": 12.5}])


class AgentModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_agent = RAGAgent(_RetrieverStub())  # type: ignore[arg-type]

    def test_rag_agent_structures_markdown_table(self) -> None:
        update = self.rag_agent(create_initial_state("查询速动比率数据"))
        self.assertEqual(update["financial_data"]["2024"]["存货"], 60.0)
        self.assertEqual(len(update["retrieved_context"]), 1)

    def test_real_time_agent_normalizes_missing_values(self) -> None:
        agent = RealTimeAgent(_MarketClientStub())
        state = create_initial_state("查询 AAPL 最新股价")
        update = agent(state)
        self.assertEqual(extract_symbol(state["question"]), "AAPL")
        self.assertEqual(update["market_data"]["quote"]["price"], 100.0)
        self.assertIsNone(update["market_data"]["quote"]["volume"])

    def test_calc_agent_executes_quick_ratio_in_sandbox(self) -> None:
        state = create_initial_state(
            "计算2024年速动比率",
            financial_data={
                "2024": {
                    "流动资产合计": 300,
                    "存货": 60,
                    "流动负债合计": 160,
                }
            },
        )
        update = CalcAgent()(state)
        self.assertTrue(update["code_output"]["success"])
        self.assertEqual(update["code_output"]["result"]["quick_ratio"], 1.5)

    def test_real_langgraph_composes_modular_agents(self) -> None:
        master = MasterAgent(
            rag_handler=self.rag_agent,
            calculation_handler=CalcAgent(),
            audit_agent=AuditAgent(fail_on_no_applicable_rules=False),
        )
        final_state = master.invoke(
            create_initial_state("计算2024年速动比率", max_retries=0)
        )
        self.assertEqual(final_state["route"], "rag+calculation+audit")
        self.assertEqual(
            final_state["code_output"]["result"]["quick_ratio"],
            1.5,
        )
        self.assertEqual(final_state["next_action"], "finalize")


if __name__ == "__main__":
    unittest.main()
