"""Tests for Data/WebDownloads/llm_ocr_pdf.py — pure helper functions.

We don't test the OCR backends (PaddleOCR, Qwen, Gemini, GPT-4o, Ollama) as
unit tests: they hit real models / APIs and require GPU + API keys.
"""

from __future__ import annotations

import pytest

import llm_ocr_pdf as ocr


class TestCleanText:
    def test_nfc_normalization(self):
        # "cà" composed vs decomposed — should both normalize to NFC
        decomposed = "ca\u0300"
        composed = "cà"
        assert ocr.clean_text(decomposed) == composed

    def test_strips_control_chars(self):
        text = "hello\x00\x01world\x7f"
        assert ocr.clean_text(text) == "helloworld"

    def test_collapses_spaces_and_tabs(self):
        assert ocr.clean_text("a    b\t\t\tc") == "a b c"

    def test_collapses_many_newlines(self):
        text = "a\n\n\n\n\n\nb"  # 6 newlines
        result = ocr.clean_text(text)
        # 4+ newlines → 3 (one blank line between blocks)
        assert "\n" * 4 not in result
        assert result == "a\n\n\nb"

    def test_nbsp_replaced_with_space(self):
        assert ocr.clean_text("a\u00a0b") == "a b"

    def test_preserves_vietnamese_diacritics(self):
        assert ocr.clean_text("Đại học Khoa học Tự nhiên") == "Đại học Khoa học Tự nhiên"

    def test_strips_leading_trailing_whitespace(self):
        assert ocr.clean_text("   hello world   ") == "hello world"

    def test_empty_string(self):
        assert ocr.clean_text("") == ""


class TestBuildHeader:
    def test_full_metadata(self):
        meta = {
            "title": "CTDT CNTT 2025",
            "he_dao_tao": "chinh-quy",
            "nganh": "cntt",
            "year": "2025",
            "url": "https://x.com/a.pdf",
        }
        header = ocr._build_header(meta)
        assert header.startswith("# Tài liệu: CTDT CNTT 2025")
        assert "# Hệ đào tạo: chinh-quy" in header
        assert "# Chuyên ngành: cntt" in header
        assert "# Năm: 2025" in header
        assert "# Nguồn: https://x.com/a.pdf" in header
        assert header.endswith("# ---\n\n")

    def test_empty_meta_returns_empty(self):
        assert ocr._build_header({}) == ""
        assert ocr._build_header(None) == ""

    def test_partial_meta(self):
        header = ocr._build_header({"title": "X"})
        assert "# Tài liệu: X" in header
        assert "# Hệ đào tạo: " in header  # empty values kept
        assert header.endswith("# ---\n\n")


class TestDispatchOcr:
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Model không hỗ trợ"):
            ocr._dispatch_ocr(pdf_path=None, model="unknown-model", verbose=False)


class TestLoadManifest:
    def test_missing_returns_empty(self, tmp_path):
        assert ocr._load_manifest(tmp_path) == {}

    def test_valid_json(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            '{"a.pdf": {"title": "X"}}', encoding="utf-8"
        )
        assert ocr._load_manifest(tmp_path) == {"a.pdf": {"title": "X"}}

    def test_invalid_json_returns_empty(self, tmp_path, capsys):
        (tmp_path / "manifest.json").write_text("{invalid", encoding="utf-8")
        result = ocr._load_manifest(tmp_path)
        assert result == {}
        out = capsys.readouterr().out
        assert "WARN" in out or "lỗi" in out.lower()
