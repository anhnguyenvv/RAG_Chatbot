"""Unit tests for app.rag.agent module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.prompts import AGENT_SYSTEM_PROMPT
from app.rag import agent as agent_mod
from app.rag.agent import ReactRAGAgent

try:
    import mongomock

    _HAS_MONGOMOCK = True
except ImportError:
    _HAS_MONGOMOCK = False


def _make_memory(**kwargs):
    """Create a MongoSessionMemoryStore backed by mongomock."""
    if not _HAS_MONGOMOCK:
        pytest.skip("mongomock required for agent memory tests")

    import app.storage.memory as mem_mod

    original = mem_mod.MongoClient
    mem_mod.MongoClient = mongomock.MongoClient
    try:
        store = mem_mod.MongoSessionMemoryStore(
            mongo_uri="mongodb://localhost",
            db_name="test_db",
            **kwargs,
        )
    finally:
        mem_mod.MongoClient = original
    return store


# ---------------------------------------------------------------------------
# AGENT_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_vietnamese_persona(self):
        assert "Khoa Cong Nghe Thong Tin" in AGENT_SYSTEM_PROMPT or "FIT" in AGENT_SYSTEM_PROMPT

    def test_contains_contact_info(self):
        assert "info@fit.hcmus.edu.vn" in AGENT_SYSTEM_PROMPT
        assert "(028) 62884499" in AGENT_SYSTEM_PROMPT

    def test_contains_tool_usage_rules(self):
        assert "qdrant_search" in AGENT_SYSTEM_PROMPT
        assert "fit_website_search" in AGENT_SYSTEM_PROMPT

    def test_contains_clarification_instructions(self):
        assert "khóa tuyển sinh" in AGENT_SYSTEM_PROMPT or "niên khóa" in AGENT_SYSTEM_PROMPT.lower()
        assert "chuyên ngành" in AGENT_SYSTEM_PROMPT.lower()
        assert "hỏi lại" in AGENT_SYSTEM_PROMPT.lower() or "HỎI LẠI" in AGENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# ReactRAGAgent -- helper methods
# ---------------------------------------------------------------------------

class TestExtractThoughtProcess:
    def _make_agent(self):
        memory = _make_memory()
        mock_llm = MagicMock()
        with patch.object(agent_mod, "create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent = ReactRAGAgent(
                llm=mock_llm,
                tools=[],
                memory_store=memory,
                max_iterations=3,
            )
        return agent

    def test_extract_tool_calls(self):
        agent = self._make_agent()
        msg = AIMessage(content="", tool_calls=[
            {"name": "qdrant_search", "args": {"query": "CNTT"}, "id": "1"},
        ])
        steps = agent._extract_thought_process([msg])
        assert len(steps) == 1
        assert steps[0]["type"] == "tool_call"
        assert steps[0]["tool"] == "qdrant_search"

    def test_extract_tool_results(self):
        agent = self._make_agent()
        msg = ToolMessage(content="[Doc 1] CNTT has 130 credits", name="qdrant_search", tool_call_id="1")
        steps = agent._extract_thought_process([msg])
        assert len(steps) == 1
        assert steps[0]["type"] == "tool_result"
        assert steps[0]["tool"] == "qdrant_search"

    def test_extract_empty_messages(self):
        agent = self._make_agent()
        steps = agent._extract_thought_process([])
        assert steps == []


class TestExtractSources:
    def _make_agent(self):
        memory = _make_memory()
        mock_llm = MagicMock()
        with patch.object(agent_mod, "create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent = ReactRAGAgent(
                llm=mock_llm,
                tools=[],
                memory_store=memory,
            )
        return agent

    def test_extract_from_qdrant_tool_result(self):
        agent = self._make_agent()
        content = "[Doc 1] | Nganh: CNTT\nContent here\n\n---\n\n[Doc 2] | Nganh: KTPM\nMore content"
        msg = ToolMessage(content=content, name="qdrant_search", tool_call_id="1")
        sources = agent._extract_sources([msg])
        assert len(sources) == 2
        assert "[Doc 1]" in sources[0]["header"]

    def test_ignore_non_qdrant_tools(self):
        agent = self._make_agent()
        msg = ToolMessage(content="website data", name="fit_website_search", tool_call_id="1")
        sources = agent._extract_sources([msg])
        assert sources == []


# ---------------------------------------------------------------------------
# ReactRAGAgent.run -- with mocked agent
# ---------------------------------------------------------------------------

class TestReactAgentRun:
    def _make_agent_with_mock(self, invoke_return):
        memory = _make_memory()
        mock_llm = MagicMock()

        with patch.object(agent_mod, "create_react_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = invoke_return
            mock_create.return_value = mock_graph

            agent = ReactRAGAgent(llm=mock_llm, tools=[], memory_store=memory)
        return agent, memory

    def test_run_returns_expected_fields(self):
        agent, _ = self._make_agent_with_mock({
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="Day la cau tra loi"),
            ]
        })
        result = agent.run("test query", session_id="s1")

        assert "result" in result
        assert result["result"] == "Day la cau tra loi"
        assert result["route"] == "react_agent"
        assert "confidence" in result
        assert "timings" in result
        assert "request_id" in result

    def test_run_no_final_answer(self):
        agent, _ = self._make_agent_with_mock({"messages": []})
        result = agent.run("test")
        assert "Khong the tao cau tra loi" in result["result"]

    def test_run_handles_exception(self):
        memory = _make_memory()
        mock_llm = MagicMock()

        with patch.object(agent_mod, "create_react_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = RuntimeError("LLM failed")
            mock_create.return_value = mock_graph
            agent = ReactRAGAgent(llm=mock_llm, tools=[], memory_store=memory)

        result = agent.run("test")
        assert "loi" in result["result"].lower() or "error" in result.get("error", "").lower()
        assert result["confidence"] == 0.0

    def test_debug_includes_thought_process(self):
        agent, _ = self._make_agent_with_mock({
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="", tool_calls=[
                    {"name": "qdrant_search", "args": {"query": "q"}, "id": "1"}
                ]),
                ToolMessage(content="[Doc 1]\nresult", name="qdrant_search", tool_call_id="1"),
                AIMessage(content="Final answer"),
            ]
        })
        result = agent.run("q", debug=True)

        assert "thought_process" in result
        assert len(result["thought_process"]) > 0
        assert "message_count" in result

    def test_session_touched_on_run(self):
        agent, memory = self._make_agent_with_mock({
            "messages": [AIMessage(content="ok")]
        })
        agent.run("test", session_id="sess-42")
        sessions = memory.list_sessions()
        assert any(s["session_id"] == "sess-42" for s in sessions)

    def test_confidence_increases_with_tool_calls(self):
        agent, _ = self._make_agent_with_mock({
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="", tool_calls=[
                    {"name": "qdrant_search", "args": {"query": "q"}, "id": "1"}
                ]),
                ToolMessage(content="[Doc 1]\nresult", name="qdrant_search", tool_call_id="1"),
                AIMessage(content="", tool_calls=[
                    {"name": "fit_website_search", "args": {"query": "q"}, "id": "2"}
                ]),
                ToolMessage(content="website data", name="fit_website_search", tool_call_id="2"),
                AIMessage(content="Comprehensive answer"),
            ]
        })
        result = agent.run("q")

        assert result["confidence"] >= 0.6
        assert result["needs_clarification"] is False

    def test_messages_persisted_to_memory(self):
        agent, memory = self._make_agent_with_mock({
            "messages": [
                HumanMessage(content="test query"),
                AIMessage(content="test answer"),
            ]
        })
        agent.run("test query", session_id="s-persist")
        messages = memory.get_messages("s-persist")
        assert len(messages) == 2
        assert messages[0].content == "test query"
        assert messages[1].content == "test answer"

    def test_context_extracted_from_conversation(self):
        agent, memory = self._make_agent_with_mock({
            "messages": [
                HumanMessage(content="CNTT K2023 chinh quy"),
                AIMessage(content="Chuong trinh dao tao CNTT K2023"),
            ]
        })
        agent.run("CNTT K2023 chinh quy", session_id="s-ctx")
        ctx = memory.get_context("s-ctx")
        assert ctx.get("nganh") == "Cong nghe thong tin"
        assert ctx.get("khoa") == "K2023"
        assert ctx.get("he_dao_tao") == "chinh quy"
