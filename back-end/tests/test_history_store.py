"""Unit tests for app.history_store module."""

from __future__ import annotations

import pytest


class TestChatHistoryStore:
    def test_add_and_get_entry(self, history_store_cls, tmp_db_path):
        store = history_store_cls(db_path=tmp_db_path)
        entry_id = store.add_entry(
            source="qdrant",
            query="test query",
            answer="test answer",
            source_documents=[{"page_content": "doc1"}],
        )
        assert isinstance(entry_id, int)
        assert entry_id > 0

        entry = store.get_entry(entry_id)
        assert entry is not None
        assert entry["source"] == "qdrant"
        assert entry["query"] == "test query"
        assert entry["answer"] == "test answer"
        assert entry["source_documents"] == [{"page_content": "doc1"}]

    def test_get_nonexistent_entry(self, history_store_cls, tmp_db_path):
        store = history_store_cls(db_path=tmp_db_path)
        assert store.get_entry(999) is None

    def test_list_entries_ordered(self, history_store_cls, tmp_db_path):
        store = history_store_cls(db_path=tmp_db_path)
        store.add_entry("qdrant", "q1", "a1", [])
        store.add_entry("qdrant", "q2", "a2", [])
        store.add_entry("qdrant", "q3", "a3", [])

        entries = store.list_entries(limit=10)
        assert len(entries) == 3
        assert entries[0]["query"] == "q3"
        assert entries[2]["query"] == "q1"

    def test_list_entries_limit(self, history_store_cls, tmp_db_path):
        store = history_store_cls(db_path=tmp_db_path)
        for i in range(10):
            store.add_entry("qdrant", f"q{i}", f"a{i}", [])

        entries = store.list_entries(limit=3)
        assert len(entries) == 3

    def test_unicode_content(self, history_store_cls, tmp_db_path):
        store = history_store_cls(db_path=tmp_db_path)
        entry_id = store.add_entry(
            source="qdrant",
            query="Điều kiện tốt nghiệp ngành CNTT?",
            answer="Sinh viên cần hoàn thành 130 tín chỉ và đạt GPA >= 2.0",
            source_documents=[{"nganh": "CNTT", "loai": "quy chế"}],
        )
        entry = store.get_entry(entry_id)
        assert "tốt nghiệp" in entry["query"]
        assert "130 tín chỉ" in entry["answer"]

    def test_creates_db_directory(self, history_store_cls, tmp_path):
        nested_path = tmp_path / "subdir" / "deep" / "history.db"
        store = history_store_cls(db_path=str(nested_path))
        entry_id = store.add_entry("qdrant", "q", "a", [])
        assert entry_id > 0
        assert nested_path.exists()
