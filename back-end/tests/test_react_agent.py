"""Unit tests for app.react_agent module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Import memory and react_agent directly
APP_DIR = Path(__file__).resolve().parents[1] / "app"

if "_app_memory" not in sys.modules:
    spec = importlib.util.spec_from_file_location("_app_memory", str(APP_DIR / "memory.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_app_memory"] = mod
    spec.loader.exec_module(mod)

# Make app importable as a package with the modules we need
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Prevent app/__init__.py from importing api.py (which needs fastapi)
# by pre-registering a dummy app package
import types
if "app" not in sys.modules:
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(APP_DIR)]
    sys.modules["app"] = app_pkg

# Now import directly
sys.modules["app.memory"] = sys.modules["_app_memory"]
from app.react_agent import ReactRAGAgent, SYSTEM_PROMPT

react_agent_mod = sys.modules["app.react_agent"]
SessionMemoryStore = sys.modules["_app_memory"].SessionMemoryStore


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_vietnamese_persona(self):
        assert "Khoa Công Nghệ Thông Tin" in SYSTEM_PROMPT

    def test_contains_contact_info(self):
        assert "info@fit.hcmus.edu.vn" in SYSTEM_PROMPT
        assert "(028) 62884499" in SYSTEM_PROMPT

    def test_contains_tool_usage_rules(self):
        assert "qdrant_search" in SYSTEM_PROMPT
        assert "fit_website_search" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# ReactRAGAgent — helper methods
# ---------------------------------------------------------------------------

class TestExtractThoughtProcess:
    def _make_agent(self):
        memory = SessionMemoryStore()
        mock_llm = MagicMock()
        with patch.object(react_agent_mod, "create_react_agent") as mock_create:
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
        memory = SessionMemoryStore()
        mock_llm = MagicMock()
        with patch.object(react_agent_mod, "create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent = ReactRAGAgent(
                llm=mock_llm,
                tools=[],
                memory_store=memory,
            )
        return agent

    def test_extract_from_qdrant_tool_result(self):
        agent = self._make_agent()
        content = "[Doc 1] | Ngành: CNTT\nContent here\n\n---\n\n[Doc 2] | Ngành: KTPM\nMore content"
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
# ReactRAGAgent.run — with mocked agent
# ---------------------------------------------------------------------------

class TestReactAgentRun:
    def _make_agent_with_mock(self, invoke_return):
        memory = SessionMemoryStore()
        mock_llm = MagicMock()

        with patch.object(react_agent_mod, "create_react_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = invoke_return
            mock_create.return_value = mock_graph

            agent = ReactRAGAgent(llm=mock_llm, tools=[], memory_store=memory)
        return agent, memory

    def test_run_returns_expected_fields(self):
        agent, _ = self._make_agent_with_mock({
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="Đây là câu trả lời"),
            ]
        })
        result = agent.run("test query", session_id="s1")

        assert "result" in result
        assert result["result"] == "Đây là câu trả lời"
        assert result["route"] == "react_agent"
        assert "confidence" in result
        assert "timings" in result
        assert "request_id" in result

    def test_run_no_final_answer(self):
        agent, _ = self._make_agent_with_mock({"messages": []})
        result = agent.run("test")
        assert "Không thể tạo câu trả lời" in result["result"]

    def test_run_handles_exception(self):
        memory = SessionMemoryStore()
        mock_llm = MagicMock()

        with patch.object(react_agent_mod, "create_react_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = RuntimeError("LLM failed")
            mock_create.return_value = mock_graph
            agent = ReactRAGAgent(llm=mock_llm, tools=[], memory_store=memory)

        result = agent.run("test")
        assert "Đã xảy ra lỗi" in result["result"]
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
