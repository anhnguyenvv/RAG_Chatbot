"""Tests for Data/pipeline/loaders.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.loaders import (
    _build_base_metadata,
    _extract_issue_year,
    _extract_program_name,
    _infer_document_type,
    _parse_crawl_filename,
    _parse_crawl_header,
    load_txt_documents,
)


class TestParseCrawlHeader:
    def test_full_header(self):
        content = (
            "# Tài liệu: CTDT CNTT 2025\n"
            "# Hệ đào tạo: chinh-quy\n"
            "# Chuyên ngành: cntt\n"
            "# Năm: 2025\n"
            "# Nguồn: https://fit.hcmus.edu.vn/x.pdf\n"
            "# ---\n"
            "\n"
            "Đây là nội dung chính.\n"
            "Dòng 2."
        )
        header, body = _parse_crawl_header(content)
        assert header == {
            "Tài liệu": "CTDT CNTT 2025",
            "Hệ đào tạo": "chinh-quy",
            "Chuyên ngành": "cntt",
            "Năm": "2025",
            "Nguồn": "https://fit.hcmus.edu.vn/x.pdf",
        }
        assert body.startswith("Đây là nội dung chính.")
        assert body.endswith("Dòng 2.")

    def test_no_header_returns_empty_dict(self):
        content = "Chỉ là văn bản\nkhông có metadata"
        header, body = _parse_crawl_header(content)
        assert header == {}
        assert body == content

    def test_missing_separator(self):
        content = "# Tài liệu: X\n# Chuyên ngành: cntt\n\nNội dung không có sep"
        header, body = _parse_crawl_header(content)
        # No '# ---' → body_start = len(lines) → body is empty after lstrip
        assert header == {"Tài liệu": "X", "Chuyên ngành": "cntt"}
        assert body == ""

    def test_empty_value_kept(self):
        content = "# Tài liệu: \n# ---\n\nbody"
        header, _ = _parse_crawl_header(content)
        assert header == {"Tài liệu": ""}

    def test_malformed_line_skipped(self):
        # Lines without ":" are silently ignored but included in key parsing
        content = "# Tài liệu: X\n# no-colon-line\n# ---\nbody"
        header, body = _parse_crawl_header(content)
        assert "Tài liệu" in header
        assert body == "body"


class TestParseCrawlFilename:
    def test_standard_format(self):
        result = _parse_crawl_filename(
            "chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt"
        )
        assert result == {
            "he_dao_tao": "chinh-quy",
            "nganh_slug": "cntt",
            "year": "2025",
            "doc_type_slug": "chuong-trinh-dao-tao",
        }

    def test_extra_double_underscore_in_doc_type(self):
        # doc_type_slug should capture everything after the 3rd __
        result = _parse_crawl_filename(
            "chinh-quy__khmt__2023__quyet-dinh__ban-hanh.txt"
        )
        assert result["doc_type_slug"] == "quyet-dinh__ban-hanh"

    def test_too_few_parts_returns_empty(self):
        assert _parse_crawl_filename("invalid-name.txt") == {}
        assert _parse_crawl_filename("a__b__c.txt") == {}

    def test_legacy_filename_returns_empty(self):
        assert _parse_crawl_filename("CNTT.txt") == {}
        assert _parse_crawl_filename(
            "chuong_trinh_dao_tao_nganh-cong_nghe_thong_tin.txt"
        ) == {}


class TestBuildBaseMetadata:
    def _mkpath(self, name: str) -> Path:
        return Path(f"/tmp/{name}")

    def test_crawl_format_with_header_and_filename(self):
        filename = "chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt"
        header = {
            "Tài liệu": "CTDT CNTT",
            "Hệ đào tạo": "chinh-quy",
            "Chuyên ngành": "cntt",
            "Năm": "2025",
            "Nguồn": "https://fit.hcmus.edu.vn/x.pdf",
        }
        meta = _build_base_metadata(self._mkpath(filename), "body", header)
        assert meta["source"] == "https://fit.hcmus.edu.vn/x.pdf"
        assert meta["file_name"] == filename
        assert meta["he_dao_tao"] == "chinh-quy"
        assert meta["nganh"] == "Công nghệ thông tin"
        assert meta["loai_van_ban"] == "chương trình đào tạo"
        assert meta["nam_ban_hanh"] == 2025
        assert meta["tai_lieu"] == "CTDT CNTT"

    def test_unknown_year_becomes_none(self):
        filename = "chinh-quy__cntt__unknown__chuong-trinh-dao-tao.txt"
        meta = _build_base_metadata(self._mkpath(filename), "body")
        assert meta["nam_ban_hanh"] is None

    def test_non_numeric_year_becomes_none(self):
        filename = "chinh-quy__cntt__NaN__chuong-trinh-dao-tao.txt"
        meta = _build_base_metadata(self._mkpath(filename), "body")
        assert meta["nam_ban_hanh"] is None

    def test_unknown_nganh_slug_falls_back_to_slug(self):
        filename = "chinh-quy__xyz-unknown__2025__chuong-trinh-dao-tao.txt"
        meta = _build_base_metadata(self._mkpath(filename), "body")
        # Returns the slug itself when not in map
        assert meta["nganh"] == "xyz-unknown"

    def test_unknown_doc_type_humanized(self):
        filename = "chinh-quy__cntt__2025__custom-doc-type.txt"
        meta = _build_base_metadata(self._mkpath(filename), "body")
        assert meta["loai_van_ban"] == "custom doc type"

    def test_legacy_format_fallback(self):
        filename = "chuong_trinh_dao_tao_nganh-cong_nghe_thong_tin.txt"
        content = "Ngành Công nghệ thông tin. Ngày 07/09/2023"
        meta = _build_base_metadata(self._mkpath(filename), content)
        assert meta["loai_van_ban"] == "chương trình đào tạo"
        assert meta["nganh"] == "Công nghệ thông tin"
        assert meta["nam_ban_hanh"] == 2023
        assert meta["he_dao_tao"] is None

    def test_header_priority_for_year(self):
        filename = "chinh-quy__cntt__2020__chuong-trinh-dao-tao.txt"
        header = {"Năm": "2025"}
        meta = _build_base_metadata(self._mkpath(filename), "body", header)
        assert meta["nam_ban_hanh"] == 2025  # header wins over filename

    def test_filename_source_fallback(self):
        filename = "chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt"
        meta = _build_base_metadata(self._mkpath(filename), "body", header_meta={})
        # When "Nguồn" missing, source falls back to filename
        assert meta["source"] == filename


class TestInferDocumentType:
    def test_prefix_match(self):
        assert _infer_document_type("chuong_trinh_dao_tao_x.txt", "") == "chương trình đào tạo"
        assert _infer_document_type("quy_dinh_dao_tao_x.txt", "") == "quy định"

    def test_legacy_filenames(self):
        assert _infer_document_type("QuyChe.txt", "") == "quy định"
        assert _infer_document_type("DKTN.txt", "") == "tốt nghiệp"
        assert _infer_document_type("CNTT.txt", "") == "chương trình đào tạo"

    def test_content_fallback_to_quy_dinh(self):
        assert _infer_document_type("misc.txt", "Các quy định chung") == "quy định"

    def test_default_tai_lieu_khac(self):
        assert _infer_document_type("unknown.txt", "nothing special") == "tài liệu khác"


class TestExtractIssueYear:
    def test_date_format(self):
        assert _extract_issue_year("ban hành ngày 07/09/2023") == 2023

    def test_khoa_tuyen(self):
        assert _extract_issue_year("Khóa tuyển: 2022 cho ngành X") == 2022

    def test_fallback_first_year(self):
        assert _extract_issue_year("tài liệu năm 2019 cập nhật 2021") == 2019

    def test_no_year(self):
        assert _extract_issue_year("không có năm nào") is None


class TestExtractProgramName:
    def test_slug_match(self):
        assert _extract_program_name(
            "chuong_trinh_dao_tao_nganh-cong_nghe_thong_tin.txt", ""
        ) == "Công nghệ thông tin"

    def test_legacy_filename(self):
        assert _extract_program_name("CNTT.txt", "") == "Công nghệ thông tin"
        assert _extract_program_name("HTTT.txt", "") == "Hệ thống thông tin"

    def test_content_extraction(self):
        assert _extract_program_name("misc.txt", "Ngành Khoa học máy tính chuyên") == "Khoa học máy tính chuyên"

    def test_not_found(self):
        assert _extract_program_name("unknown.txt", "no signal") is None


class TestLoadTxtDocuments:
    def test_raises_for_missing_dir(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_txt_documents(str(missing))

    def test_raises_for_empty_dir(self, tmp_path):
        with pytest.raises(ValueError, match="No .txt files"):
            load_txt_documents(str(tmp_path))

    def test_loads_crawl_format(self, tmp_path):
        filename = "chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt"
        content = (
            "# Tài liệu: CTDT CNTT\n"
            "# Hệ đào tạo: chinh-quy\n"
            "# Chuyên ngành: cntt\n"
            "# Năm: 2025\n"
            "# Nguồn: https://fit.hcmus.edu.vn/x.pdf\n"
            "# ---\n"
            "\n"
            "Nội dung chính tài liệu."
        )
        (tmp_path / filename).write_text(content, encoding="utf-8")

        docs = load_txt_documents(str(tmp_path))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content.startswith("Nội dung chính")
        assert "# Tài liệu" not in doc.page_content  # header stripped
        assert doc.metadata["nganh"] == "Công nghệ thông tin"
        assert doc.metadata["nam_ban_hanh"] == 2025
        assert doc.metadata["he_dao_tao"] == "chinh-quy"

    def test_loads_legacy_format(self, tmp_path):
        (tmp_path / "QuyChe.txt").write_text(
            "Quy chế đào tạo. Ngày 24/09/2021", encoding="utf-8"
        )
        docs = load_txt_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["loai_van_ban"] == "quy định"
        assert docs[0].metadata["nam_ban_hanh"] == 2021

    def test_handles_utf8_sig(self, tmp_path):
        # BOM-prefixed file should still load
        (tmp_path / "file.txt").write_bytes(
            "\ufeffNội dung có BOM".encode("utf-8")
        )
        docs = load_txt_documents(str(tmp_path))
        assert len(docs) == 1
        # TextLoader's utf-8 fallback strips BOM when encoding=utf-8-sig
        assert "Nội dung có BOM" in docs[0].page_content

    def test_multiple_files_sorted(self, tmp_path):
        (tmp_path / "b.txt").write_text("content B", encoding="utf-8")
        (tmp_path / "a.txt").write_text("content A", encoding="utf-8")
        docs = load_txt_documents(str(tmp_path))
        assert [d.metadata["file_name"] for d in docs] == ["a.txt", "b.txt"]
