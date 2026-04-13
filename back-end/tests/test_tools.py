"""Unit tests for app.tools module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

# Import tools module directly to avoid app/__init__.py
APP_DIR = Path(__file__).resolve().parents[1] / "app"
spec = importlib.util.spec_from_file_location("_app_tools", str(APP_DIR / "tools.py"))
tools_mod = importlib.util.module_from_spec(spec)
sys.modules["_app_tools"] = tools_mod
spec.loader.exec_module(tools_mod)

_format_docs_for_agent = tools_mod._format_docs_for_agent
_token_overlap_score = tools_mod._token_overlap_score
create_qdrant_search_tool = tools_mod.create_qdrant_search_tool
create_fit_website_tool = tools_mod.create_fit_website_tool


# ---------------------------------------------------------------------------
# _token_overlap_score
# ---------------------------------------------------------------------------

class TestTokenOverlapScore:
    def test_identical_query_and_content(self):
        score = _token_overlap_score("hello world", "hello world")
        assert score == 1.0

    def test_partial_overlap(self):
        score = _token_overlap_score("hello world foo", "hello bar baz")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = _token_overlap_score("alpha beta", "gamma delta")
        assert score == 0.0

    def test_empty_query(self):
        score = _token_overlap_score("", "some content here")
        assert score == 0.0

    def test_single_char_tokens_ignored(self):
        score = _token_overlap_score("a b c", "a b c d")
        assert score == 0.0

    def test_case_insensitive(self):
        score = _token_overlap_score("Hello WORLD", "hello world")
        assert score == 1.0

    def test_vietnamese_tokens(self):
        score = _token_overlap_score("chương trình đào tạo", "chương trình đào tạo ngành CNTT")
        assert score > 0.5


# ---------------------------------------------------------------------------
# _format_docs_for_agent
# ---------------------------------------------------------------------------

class TestFormatDocsForAgent:
    def test_empty_docs(self):
        result = _format_docs_for_agent([])
        assert "Không tìm thấy" in result

    def test_single_doc(self, fake_docs):
        result = _format_docs_for_agent(fake_docs[:1])
        assert "[Doc 1]" in result
        assert "CNTT" in result
        assert "130 tín chỉ" in result

    def test_multiple_docs_separated(self, fake_docs):
        result = _format_docs_for_agent(fake_docs)
        assert "[Doc 1]" in result
        assert "[Doc 2]" in result
        assert "[Doc 3]" in result
        assert "---" in result

    def test_top_k_limits(self, fake_docs):
        result = _format_docs_for_agent(fake_docs, top_k=1)
        assert "[Doc 1]" in result
        assert "[Doc 2]" not in result

    def test_metadata_displayed(self, fake_docs):
        result = _format_docs_for_agent(fake_docs[:1])
        assert "Ngành: CNTT" in result
        assert "Loại: chương trình đào tạo" in result
        assert "Nguồn: ctdt_cntt.txt" in result

    def test_missing_metadata_graceful(self):
        class BareDoc:
            page_content = "bare content"
            metadata = {}

        result = _format_docs_for_agent([BareDoc()])
        assert "[Doc 1]" in result
        assert "Nguồn: unknown" in result


# ---------------------------------------------------------------------------
# create_qdrant_search_tool
# ---------------------------------------------------------------------------

class TestQdrantSearchTool:
    def test_returns_tool_with_correct_name(self, mock_retriever_fn, mock_rerank_fn):
        tool = create_qdrant_search_tool(mock_retriever_fn, mock_rerank_fn, rerank_top_k=3)
        assert hasattr(tool, "invoke")
        assert tool.name == "qdrant_search"

    def test_tool_invocation_returns_docs(self, mock_retriever_fn, mock_rerank_fn):
        tool = create_qdrant_search_tool(mock_retriever_fn, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke("điều kiện tốt nghiệp")
        assert "[Doc 1]" in result
        assert "CNTT" in result

    def test_tool_empty_results(self, mock_rerank_fn):
        def empty_retriever(source, query):
            return []

        tool = create_qdrant_search_tool(empty_retriever, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke("xyz không tồn tại")
        assert "Không tìm thấy" in result

    def test_tool_handles_exception(self, mock_rerank_fn):
        def error_retriever(source, query):
            raise ConnectionError("DB unavailable")

        tool = create_qdrant_search_tool(error_retriever, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke("test")
        assert "Lỗi" in result

    def test_tool_has_description(self, mock_retriever_fn, mock_rerank_fn):
        tool = create_qdrant_search_tool(mock_retriever_fn, mock_rerank_fn)
        assert "FIT" in tool.description or "HCMUS" in tool.description


# ---------------------------------------------------------------------------
# create_fit_website_tool
# ---------------------------------------------------------------------------

class TestFITWebsiteTool:
    def test_returns_tool_with_correct_name(self):
        tool = create_fit_website_tool("fit.hcmus.edu.vn")
        assert hasattr(tool, "invoke")
        assert tool.name == "fit_website_search"

    def test_empty_allowlist(self):
        tool = create_fit_website_tool("")
        result = tool.invoke("test query")
        assert "Không có website" in result

    def test_tool_has_description(self):
        tool = create_fit_website_tool("fit.hcmus.edu.vn")
        assert "FIT" in tool.description or "HCMUS" in tool.description
