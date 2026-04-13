"""Shared fixtures for backend tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BACKEND_DIR / "app"


def import_module_from_file(name: str, filepath: Path):
    """Import a module directly from file path, bypassing package __init__."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-import modules that only need stdlib + python-dotenv
_config_mod = import_module_from_file("_app_config", APP_DIR / "config.py")
_history_mod = import_module_from_file("_app_history_store", APP_DIR / "history_store.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend_config():
    return _config_mod.BackendConfig(
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
    return _history_mod.ChatHistoryStore


@pytest.fixture
def config_cls():
    return _config_mod.BackendConfig


@pytest.fixture
def to_bool_fn():
    return _config_mod._to_bool


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
            page_content="Chương trình đào tạo ngành CNTT gồm 130 tín chỉ.",
            metadata={"source": "ctdt_cntt.txt", "nganh": "CNTT", "loai_van_ban": "chương trình đào tạo"},
        ),
        FakeDocument(
            page_content="Điều kiện tốt nghiệp: hoàn thành tất cả học phần và đạt GPA >= 2.0.",
            metadata={"source": "quy_che.txt", "nganh": "CNTT", "loai_van_ban": "quy chế"},
        ),
        FakeDocument(
            page_content="Sinh viên cần đạt chuẩn ngoại ngữ TOEIC 450.",
            metadata={"source": "ngoai_ngu.txt", "nganh": "", "loai_van_ban": "quy chế"},
        ),
    ]


@pytest.fixture
def mock_retriever_fn(fake_docs):
    def retriever(source: str, query: str):
        return fake_docs
    return retriever


@pytest.fixture
def mock_rerank_fn(fake_docs):
    def rerank(query: str, documents: list, top_k: int = 3):
        return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(documents[:top_k])]
    return rerank
