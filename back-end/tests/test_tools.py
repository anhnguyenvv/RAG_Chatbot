"""Unit tests for app.rag.tools module."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from app.rag.reranker import token_overlap_score
from app.rag.tools import (
    _format_docs_for_agent,
    create_fit_website_tool,
    create_qdrant_search_tool,
)


# ---------------------------------------------------------------------------
# token_overlap_score
# ---------------------------------------------------------------------------

class TestTokenOverlapScore:
    def test_identical_query_and_content(self):
        score = token_overlap_score("hello world", "hello world")
        assert score == 1.0

    def test_partial_overlap(self):
        score = token_overlap_score("hello world foo", "hello bar baz")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = token_overlap_score("alpha beta", "gamma delta")
        assert score == 0.0

    def test_empty_query(self):
        score = token_overlap_score("", "some content here")
        assert score == 0.0

    def test_single_char_tokens_ignored(self):
        score = token_overlap_score("a b c", "a b c d")
        assert score == 0.0

    def test_case_insensitive(self):
        score = token_overlap_score("Hello WORLD", "hello world")
        assert score == 1.0

    def test_vietnamese_tokens(self):
        score = token_overlap_score("chuong trinh dao tao", "chuong trinh dao tao nganh CNTT")
        assert score > 0.5


# ---------------------------------------------------------------------------
# _format_docs_for_agent
# ---------------------------------------------------------------------------

class TestFormatDocsForAgent:
    def test_empty_docs(self):
        result = _format_docs_for_agent([])
        assert "tim thay" in result.lower() or "khong" in result.lower()

    def test_single_doc(self, fake_docs):
        result = _format_docs_for_agent(fake_docs[:1])
        assert "[Doc 1]" in result
        assert "CNTT" in result

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
        assert "CNTT" in result
        assert "Nguon:" in result or "ctdt_cntt.txt" in result

    def test_missing_metadata_graceful(self):
        class BareDoc:
            page_content = "bare content"
            metadata = {}

        result = _format_docs_for_agent([BareDoc()])
        assert "[Doc 1]" in result
        assert "unknown" in result


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
        result = tool.invoke("dieu kien tot nghiep")
        assert "[Doc 1]" in result
        assert "CNTT" in result

    def test_tool_invocation_with_metadata_filter(self, mock_retriever_fn, mock_rerank_fn):
        tool = create_qdrant_search_tool(mock_retriever_fn, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke({"query": "dieu kien tot nghiep", "nganh": "CNTT"})
        assert "[Doc 1]" in result

    def test_tool_invocation_with_he_dao_tao(self, mock_retriever_fn, mock_rerank_fn):
        tool = create_qdrant_search_tool(mock_retriever_fn, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke({"query": "chuong trinh", "he_dao_tao": "chinh quy"})
        assert "[Doc" in result or "khong" in result.lower()

    def test_tool_empty_results(self, mock_rerank_fn):
        def empty_retriever(source, query, metadata_filter=None):
            return []

        tool = create_qdrant_search_tool(empty_retriever, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke("xyz khong ton tai")
        assert "khong" in result.lower() or "Khong" in result

    def test_tool_handles_exception(self, mock_rerank_fn):
        def error_retriever(source, query, metadata_filter=None):
            raise ConnectionError("DB unavailable")

        tool = create_qdrant_search_tool(error_retriever, mock_rerank_fn, rerank_top_k=3)
        result = tool.invoke("test")
        assert "Loi" in result or "loi" in result.lower()

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
        assert "khong" in result.lower() or "Khong" in result

    def test_tool_has_description(self):
        tool = create_fit_website_tool("fit.hcmus.edu.vn")
        assert "FIT" in tool.description or "HCMUS" in tool.description
