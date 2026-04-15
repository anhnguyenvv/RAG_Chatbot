"""Unit tests for app.storage.memory module (MongoSessionMemoryStore)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# We mock pymongo so tests don't need a real MongoDB instance.
# mongomock provides a drop-in replacement for pymongo.MongoClient.
try:
    import mongomock

    _HAS_MONGOMOCK = True
except ImportError:
    _HAS_MONGOMOCK = False


def _make_store(**kwargs):
    """Create a MongoSessionMemoryStore backed by mongomock."""
    if not _HAS_MONGOMOCK:
        pytest.skip("mongomock required for memory tests (pip install mongomock)")

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
# MongoSessionMemoryStore -- basic operations
# ---------------------------------------------------------------------------


class TestMongoSessionMemoryStoreBasic:
    def test_touch_creates_session(self):
        store = _make_store()
        store.touch_session("s1")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["message_count"] == 0

    def test_touch_updates_last_access(self):
        store = _make_store()
        store.touch_session("s1")
        doc1 = store._col.find_one({"_id": "s1"})
        first_access = doc1["last_access"]
        time.sleep(0.01)
        store.touch_session("s1")
        doc2 = store._col.find_one({"_id": "s1"})
        second_access = doc2["last_access"]
        assert second_access > first_access

    def test_increment_message_count(self):
        store = _make_store()
        store.touch_session("s1")
        assert store.increment_message_count("s1") == 1
        assert store.increment_message_count("s1") == 2

    def test_increment_nonexistent_session(self):
        store = _make_store()
        assert store.increment_message_count("nope") == 0

    def test_clear_session(self):
        store = _make_store()
        store.touch_session("s1")
        assert store.clear_session("s1") is True
        assert store.list_sessions() == []

    def test_clear_nonexistent(self):
        store = _make_store()
        assert store.clear_session("nope") is False


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------


class TestMongoSessionMemoryStoreMessages:
    def test_add_and_get_messages(self):
        store = _make_store()
        store.touch_session("s1")
        store.add_message("s1", HumanMessage(content="Hello"))
        store.add_message("s1", AIMessage(content="Hi there"))

        messages = store.get_messages("s1")
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        assert isinstance(messages[1], AIMessage)

    def test_add_messages_batch(self):
        store = _make_store()
        store.touch_session("s1")
        store.add_messages("s1", [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
        ])
        messages = store.get_messages("s1")
        assert len(messages) == 3

    def test_get_recent_messages(self):
        store = _make_store(max_recent_turns=2)
        store.touch_session("s1")
        store.add_messages("s1", [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
            AIMessage(content="A2"),
            HumanMessage(content="Q3"),
            AIMessage(content="A3"),
        ])
        recent = store.get_recent_messages("s1")
        assert len(recent) == 4  # 2 turns * 2
        assert recent[0].content == "Q2"

    def test_get_messages_empty(self):
        store = _make_store()
        assert store.get_messages("nonexistent") == []


# ---------------------------------------------------------------------------
# Summary operations
# ---------------------------------------------------------------------------


class TestMongoSessionMemoryStoreSummary:
    def test_get_summary_empty(self):
        store = _make_store()
        assert store.get_session_summary("s1") == ""

    def test_update_and_get_summary(self):
        store = _make_store()
        store.touch_session("s1")
        store.update_summary("s1", "User asked about CNTT program")
        assert store.get_session_summary("s1") == "User asked about CNTT program"

    def test_update_summary_nonexistent_session(self):
        store = _make_store()
        store.update_summary("nope", "something")
        assert store.get_session_summary("nope") == ""


# ---------------------------------------------------------------------------
# Context operations
# ---------------------------------------------------------------------------


class TestMongoSessionMemoryStoreContext:
    def test_get_context_empty(self):
        store = _make_store()
        assert store.get_context("s1") == {}

    def test_update_and_get_context(self):
        store = _make_store()
        store.touch_session("s1")
        store.update_context("s1", {"nganh": "CNTT", "khoa": "K2023"})
        ctx = store.get_context("s1")
        assert ctx["nganh"] == "CNTT"
        assert ctx["khoa"] == "K2023"

    def test_context_merge(self):
        store = _make_store()
        store.touch_session("s1")
        store.update_context("s1", {"nganh": "CNTT"})
        store.update_context("s1", {"khoa": "K2024"})
        ctx = store.get_context("s1")
        assert ctx["nganh"] == "CNTT"
        assert ctx["khoa"] == "K2024"

    def test_update_context_empty_values_ignored(self):
        store = _make_store()
        store.touch_session("s1")
        store.update_context("s1", {"nganh": "", "khoa": "K2023"})
        ctx = store.get_context("s1")
        assert "nganh" not in ctx
        assert ctx["khoa"] == "K2023"


# ---------------------------------------------------------------------------
# Message trimming
# ---------------------------------------------------------------------------


class TestPrepareMessagesWithSummary:
    def test_few_messages_no_trim(self):
        store = _make_store(max_recent_turns=3)
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
        ]
        result = store.prepare_messages_with_summary(messages, "")
        assert len(result) == 2

    def test_many_messages_trimmed(self):
        store = _make_store(max_recent_turns=2)
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
        store = _make_store(max_recent_turns=2)
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
        ]
        result = store.prepare_messages_with_summary(messages, "Previous context about CNTT")
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert "CNTT" in result[0].content

    def test_no_summary_no_system_msg(self):
        store = _make_store(max_recent_turns=2)
        messages = [HumanMessage(content="Q1")]
        result = store.prepare_messages_with_summary(messages, "")
        assert not any(isinstance(m, SystemMessage) for m in result)


# ---------------------------------------------------------------------------
# should_summarize
# ---------------------------------------------------------------------------


class TestShouldSummarize:
    def test_below_threshold(self):
        store = _make_store(max_recent_turns=3)
        messages = [HumanMessage(content="Q1"), AIMessage(content="A1")]
        assert store.should_summarize(messages) is False

    def test_above_threshold(self):
        store = _make_store(max_recent_turns=2)
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
        store = _make_store()
        messages = [
            HumanMessage(content="Dieu kien tot nghiep CNTT?"),
            AIMessage(content="Can 130 tin chi."),
        ]
        prompt = store.build_summary_prompt(messages, "")
        assert "Nguoi dung" in prompt
        assert "Tro ly" in prompt
        assert "130 tin chi" in prompt

    def test_with_existing_summary(self):
        store = _make_store()
        messages = [HumanMessage(content="Mon nao kho?")]
        prompt = store.build_summary_prompt(messages, "Previous: asked about CNTT")
        assert "Previous: asked about CNTT" in prompt
        assert "Mon nao kho" in prompt
