"""Shared fixtures for backend tests."""

from __future__ import annotations

from typing import Any

import pytest

from app.config.config import BackendConfig, _to_bool
from app.storage.history import ChatHistoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend_config():
    return BackendConfig(
        generate_model_name="gemini",
        google_api_key="test-google-key",
        retrieval_top_k=5,
        rerank_top_k=3,
        enable_reranker=True,
        official_site_allowlist="fit.hcmus.edu.vn",
        agent_max_iterations=3,
        memory_max_recent_turns=3,
        memory_session_ttl=300,
        max_context_tokens=1000,
    )


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_history.db")


@pytest.fixture
def history_store_cls():
    return ChatHistoryStore


@pytest.fixture
def config_cls():
    return BackendConfig


@pytest.fixture
def to_bool_fn():
    return _to_bool


# ---------------------------------------------------------------------------
# Mock LangChain document
# ---------------------------------------------------------------------------

class FakeDocument:
    """Minimal Document-like object for testing."""

    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def to_json(self):
        return {"kwargs": {"page_content": self.page_content, "metadata": dict(self.metadata)}}


@pytest.fixture
def fake_docs():
    return [
        FakeDocument(
            page_content="Chuong trinh dao tao nganh CNTT gom 130 tin chi.",
            metadata={"source": "ctdt_cntt.txt", "nganh": "CNTT", "loai_van_ban": "chuong trinh dao tao"},
        ),
        FakeDocument(
            page_content="Dieu kien tot nghiep: hoan thanh tat ca hoc phan va dat GPA >= 2.0.",
            metadata={"source": "quy_che.txt", "nganh": "CNTT", "loai_van_ban": "quy che"},
        ),
        FakeDocument(
            page_content="Sinh vien can dat chuan ngoai ngu TOEIC 450.",
            metadata={"source": "ngoai_ngu.txt", "nganh": "", "loai_van_ban": "quy che"},
        ),
    ]


@pytest.fixture
def mock_retriever_fn(fake_docs):
    def retriever(source: str, query: str, metadata_filter: dict | None = None):
        if metadata_filter:
            # Simple filtering for tests: filter by nganh if provided
            filtered = [
                doc for doc in fake_docs
                if all(
                    doc.metadata.get(k, "") == v
                    for k, v in metadata_filter.items()
                    if v
                )
            ]
            return filtered if filtered else fake_docs  # fallback to all
        return fake_docs
    return retriever


@pytest.fixture
def mock_rerank_fn(fake_docs):
    def rerank(query: str, documents: list, top_k: int = 3):
        return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(documents[:top_k])]
    return rerank
