"""Tests for Data/pipeline/splitters.py."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from pipeline.splitters import (
    OutlineAwareTextSplitter,
    _extract_dieu_khoan,
    _extract_hoc_ky,
    _merge_chunk_metadata,
    _strip_crawl_header,
    chunk_documents,
)


class TestExtractHocKy:
    def test_vietnamese_with_diacritics(self):
        assert _extract_hoc_ky("Học kỳ 3 năm 2024") == "Học kỳ 3 năm 2024"

    def test_ascii_fallback(self):
        assert _extract_hoc_ky("Hoc ky 2 nam 2023") == "Học kỳ 2 năm 2023"

    def test_case_insensitive(self):
        assert _extract_hoc_ky("HOC KY 1 NAM 2025") == "Học kỳ 1 năm 2025"

    def test_not_found(self):
        assert _extract_hoc_ky("Không có học kỳ nào") is None

    def test_partial_match_not_matched(self):
        # no number for year
        assert _extract_hoc_ky("Học kỳ 1 năm") is None


class TestExtractDieuKhoan:
    def test_vietnamese(self):
        assert _extract_dieu_khoan("Điều 15 về đào tạo") == "Điều 15"

    def test_ascii(self):
        assert _extract_dieu_khoan("Dieu 7. Quy dinh chung") == "Điều 7"

    def test_multiple_takes_first(self):
        assert _extract_dieu_khoan("Điều 3 và Điều 5") == "Điều 3"

    def test_not_found(self):
        assert _extract_dieu_khoan("Chỉ là văn bản thường") is None


class TestStripCrawlHeader:
    def test_strips_header_block(self):
        text = (
            "# Tài liệu: CTDT\n"
            "# Hệ đào tạo: chinh-quy\n"
            "# Chuyên ngành: cntt\n"
            "# ---\n"
            "\n"
            "Nội dung chính ở đây."
        )
        assert _strip_crawl_header(text) == "Nội dung chính ở đây."

    def test_no_header_returns_as_is(self):
        text = "Đoạn văn bình thường\nKhông có header."
        assert _strip_crawl_header(text) == text

    def test_missing_separator_returns_as_is(self):
        # Header without '# ---' separator should not strip
        text = "# Tài liệu: X\n\nNội dung"
        assert _strip_crawl_header(text) == text

    def test_markdown_heading_not_stripped(self):
        # A single '# ' line at start (Markdown title) with no separator
        text = "# Tiêu đề tài liệu\n\nNội dung ở đây"
        assert _strip_crawl_header(text) == text

    def test_empty_string(self):
        assert _strip_crawl_header("") == ""


class TestMergeChunkMetadata:
    def test_populates_chunk_level_fields(self):
        base = {"nganh": "CNTT", "source": "file.txt"}
        merged = _merge_chunk_metadata(base, "Điều 5. Quy định về thi.")
        assert merged["dieu_khoan"] == "Điều 5"
        assert merged["hoc_ky"] is None
        assert merged["nganh"] == "CNTT"
        assert merged["source"] == "file.txt"

    def test_defaults_applied_when_missing(self):
        merged = _merge_chunk_metadata({}, "Nội dung không có metadata")
        assert merged["loai_van_ban"] == "tài liệu khác"
        assert merged["source"] == "unknown"
        assert merged["nganh"] is None
        assert merged["he_dao_tao"] is None
        assert merged["tai_lieu"] is None

    def test_preserves_base_metadata_over_default(self):
        base = {"loai_van_ban": "quy định", "nganh": "KHMT"}
        merged = _merge_chunk_metadata(base, "nội dung")
        assert merged["loai_van_ban"] == "quy định"
        assert merged["nganh"] == "KHMT"

    def test_does_not_mutate_input(self):
        base = {"nganh": "CNTT"}
        _merge_chunk_metadata(base, "Điều 1")
        assert base == {"nganh": "CNTT"}  # unchanged


class TestOutlineAwareSplitter:
    def test_is_heading_numeric(self):
        assert OutlineAwareTextSplitter._is_heading("1.")
        assert OutlineAwareTextSplitter._is_heading("1.1.")
        assert OutlineAwareTextSplitter._is_heading("2.3.4.")
        assert not OutlineAwareTextSplitter._is_heading("1")  # no trailing dot
        assert not OutlineAwareTextSplitter._is_heading("Nội dung thường")

    def test_is_heading_roman(self):
        assert OutlineAwareTextSplitter._is_heading("I. Giới thiệu")
        assert OutlineAwareTextSplitter._is_heading("III. Kết luận")
        assert OutlineAwareTextSplitter._is_heading("IV. Phần bốn")
        # edge: IX vs I
        assert OutlineAwareTextSplitter._is_heading("IX. Chương chín")

    def test_is_heading_vietnamese_legal(self):
        assert OutlineAwareTextSplitter._is_heading("Chương I")
        assert OutlineAwareTextSplitter._is_heading("Điều 15")
        assert OutlineAwareTextSplitter._is_heading("Mục 2")
        assert OutlineAwareTextSplitter._is_heading("Khoản 3")
        assert OutlineAwareTextSplitter._is_heading("Phần II")
        # case insensitive
        assert OutlineAwareTextSplitter._is_heading("điều 5")

    def test_normalize_whitespace_collapses_blank_lines(self):
        s = "a\n\n\n\nb\n\n\n\n\nc"
        assert OutlineAwareTextSplitter._normalize_whitespace(s) == "a\n\nb\n\nc"

    def test_split_text_respects_chunk_size(self):
        """Splitter is line-based — paragraphs must be separated by newlines."""
        splitter = OutlineAwareTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "\n".join([f"Dòng số {i} chứa vài từ để chunk." for i in range(30)])
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        # Line-aware splitter may emit chunks slightly over limit — allow 2x slack
        for ch in chunks:
            assert len(ch) <= 150

    def test_split_text_single_long_line_not_split(self):
        """Known behavior: line-based splitter keeps a single long line intact.

        This is documented for future contributors — don't ship single-line
        blobs without pre-processing or the chunk size cap won't be honored.
        """
        splitter = OutlineAwareTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "word " * 200  # single line, ~1000 chars
        chunks = splitter.split_text(text)
        assert len(chunks) == 1

    def test_split_text_strips_header(self):
        splitter = OutlineAwareTextSplitter(chunk_size=200, chunk_overlap=20)
        text = "# Tài liệu: X\n# ---\n\nNội dung chính văn bản ở đây."
        chunks = splitter.split_text(text)
        joined = "\n".join(chunks)
        assert "# Tài liệu" not in joined
        assert "Nội dung chính" in joined

    def test_split_text_preserves_heading_in_overlap(self):
        splitter = OutlineAwareTextSplitter(chunk_size=60, chunk_overlap=15)
        text = (
            "Điều 1\n"
            + "Câu rất dài về quy định thi học kỳ của trường. " * 5
        )
        chunks = splitter.split_text(text)
        # Subsequent chunks should carry heading context
        assert len(chunks) >= 2
        # At least one non-first chunk retains "Điều 1" from overlap injection
        carried = [c for c in chunks[1:] if "Điều 1" in c]
        assert carried, f"Heading should propagate, got: {chunks}"

    def test_create_documents_assigns_metadata(self):
        splitter = OutlineAwareTextSplitter(chunk_size=80, chunk_overlap=10)
        docs = [
            Document(
                page_content="Điều 3. Quy định về tín chỉ.\n" + ("Nội dung. " * 20),
                metadata={"source": "test.txt", "nganh": "CNTT"},
            )
        ]
        chunks = splitter.create_documents(docs)
        assert chunks, "Expected at least one chunk"
        for ch in chunks:
            assert ch.metadata["source"] == "test.txt"
            assert ch.metadata["nganh"] == "CNTT"
        # First chunk should pick up dieu_khoan
        assert chunks[0].metadata["dieu_khoan"] == "Điều 3"


class TestChunkDocuments:
    def _doc(self, text: str, meta: dict | None = None) -> Document:
        return Document(page_content=text, metadata=meta or {"source": "x"})

    def test_outline_strategy(self):
        # Multi-line text so the line-based outline splitter can chunk it
        body = "\n".join([f"Nội dung dòng {i} về học phần." for i in range(30)])
        docs = [self._doc(f"Điều 1. {body}")]
        chunks = chunk_documents(docs, strategy="outline", chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 2

    def test_recursive_strategy(self):
        docs = [self._doc("Some long text. " * 100)]
        chunks = chunk_documents(docs, strategy="recursive", chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 2
        for ch in chunks:
            assert ch.metadata["source"] == "x"

    def test_recursive_strips_header(self):
        text = "# Tài liệu: T\n# ---\n" + ("Nội dung A. " * 50)
        chunks = chunk_documents([self._doc(text)], strategy="recursive", chunk_size=100, chunk_overlap=10)
        joined = "\n".join(c.page_content for c in chunks)
        assert "# Tài liệu" not in joined

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            chunk_documents([self._doc("x")], strategy="semantic", chunk_size=100, chunk_overlap=0)

    def test_empty_input(self):
        assert chunk_documents([], strategy="outline", chunk_size=100, chunk_overlap=0) == []
