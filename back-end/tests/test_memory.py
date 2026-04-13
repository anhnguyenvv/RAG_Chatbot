"""Unit tests for app.memory module."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import memory module directly
APP_DIR = Path(__file__).resolve().parents[1] / "app"
spec = importlib.util.spec_from_file_location("_app_memory", str(APP_DIR / "memory.py"))
memory_mod = importlib.util.module_from_spec(spec)
sys.modules["_app_memory"] = memory_mod
spec.loader.exec_module(memory_mod)
SessionMemoryStore = memory_mod.SessionMemoryStore


# ---------------------------------------------------------------------------
# SessionMemoryStore — basic operations
# ---------------------------------------------------------------------------

class TestSessionMemoryStoreBasic:
    def test_touch_creates_session(self):
        store = SessionMemoryStore()
        store.touch_session("s1")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["message_count"] == 0

    def test_touch_updates_last_access(self):
        store = SessionMemoryStore()
        store.touch_session("s1")
        first_access = store._sessions["s1"]["last_access"]
        time.sleep(0.01)
        store.touch_session("s1")
        second_access = store._sessions["s1"]["last_access"]
        assert second_access > first_access

    def test_increment_message_count(self):
        store = SessionMemoryStore()
        store.touch_session("s1")
        assert store.increment_message_count("s1") == 1
        assert store.increment_message_count("s1") == 2

    def test_increment_nonexistent_session(self):
        store = SessionMemoryStore()
        assert store.increment_message_count("nope") == 0

    def test_clear_session(self):
        store = SessionMemoryStore()
        store.touch_session("s1")
        assert store.clear_session("s1") is True
        assert store.list_sessions() == []

    def test_clear_nonexistent(self):
        store = SessionMemoryStore()
        assert store.clear_session("nope") is False


# ---------------------------------------------------------------------------
# Summary operations
# ---------------------------------------------------------------------------

class TestSessionMemoryStoreSummary:
    def test_get_summary_empty(self):
        store = SessionMemoryStore()
        assert store.get_session_summary("s1") == ""

    def test_update_and_get_summary(self):
        store = SessionMemoryStore()
        store.touch_session("s1")
        store.update_summary("s1", "User asked about CNTT program")
        assert store.get_session_summary("s1") == "User asked about CNTT program"

    def test_update_summary_nonexistent_session(self):
        store = SessionMemoryStore()
        store.update_summary("nope", "something")
        assert store.get_session_summary("nope") == ""


# ---------------------------------------------------------------------------
# TTL cleanup
# ---------------------------------------------------------------------------

class TestSessionMemoryStoreTTL:
    def test_expired_sessions_cleaned(self):
        store = SessionMemoryStore(session_ttl_seconds=0)
        store.touch_session("s1")
        time.sleep(0.01)
        sessions = store.list_sessions()
        assert len(sessions) == 0

    def test_active_sessions_kept(self):
        store = SessionMemoryStore(session_ttl_seconds=3600)
        store.touch_session("s1")
        sessions = store.list_sessions()
        assert len(sessions) == 1


# ---------------------------------------------------------------------------
# Message trimming
# ---------------------------------------------------------------------------

class TestPrepareMessagesWithSummary:
    def test_few_messages_no_trim(self):
        store = SessionMemoryStore(max_recent_turns=3)
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
        ]
        result = store.prepare_messages_with_summary(messages, "")
        assert len(result) == 2

    def test_many_messages_trimmed(self):
        store = SessionMemoryStore(max_recent_turns=2)
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
            AIMessage(content="A2"),
            HumanMessage(content="Q3"),
            AIMessage(content="A3"),
        ]
        result = store.prepare_messages_with_summary(messages, "")
        assert len(result) == 4  # keep 2*2=4 recent

    def test_summary_prepended(self):
        store = SessionMemoryStore(max_recent_turns=2)
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
        ]
        result = store.prepare_messages_with_summary(messages, "Previous context about CNTT")
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert "CNTT" in result[0].content

    def test_no_summary_no_system_msg(self):
        store = SessionMemoryStore(max_recent_turns=2)
        messages = [HumanMessage(content="Q1")]
        result = store.prepare_messages_with_summary(messages, "")
        assert not any(isinstance(m, SystemMessage) for m in result)


# ---------------------------------------------------------------------------
# should_summarize
# ---------------------------------------------------------------------------

class TestShouldSummarize:
    def test_below_threshold(self):
        store = SessionMemoryStore(max_recent_turns=3)
        messages = [HumanMessage(content="Q1"), AIMessage(content="A1")]
        assert store.should_summarize(messages) is False

    def test_above_threshold(self):
        store = SessionMemoryStore(max_recent_turns=2)
        messages = [
            HumanMessage(content="Q1"), AIMessage(content="A1"),
            HumanMessage(content="Q2"), AIMessage(content="A2"),
            HumanMessage(content="Q3"),
        ]
        assert store.should_summarize(messages) is True


# ---------------------------------------------------------------------------
# build_summary_prompt
# ---------------------------------------------------------------------------

class TestBuildSummaryPrompt:
    def test_without_existing_summary(self):
        store = SessionMemoryStore()
        messages = [
            HumanMessage(content="Điều kiện tốt nghiệp CNTT?"),
            AIMessage(content="Cần 130 tín chỉ."),
        ]
        prompt = store.build_summary_prompt(messages, "")
        assert "Người dùng" in prompt
        assert "Trợ lý" in prompt
        assert "130 tín chỉ" in prompt

    def test_with_existing_summary(self):
        store = SessionMemoryStore()
        messages = [HumanMessage(content="Môn nào khó?")]
        prompt = store.build_summary_prompt(messages, "Previous: asked about CNTT")
        assert "Previous: asked about CNTT" in prompt
        assert "Môn nào khó" in prompt
