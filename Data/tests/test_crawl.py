"""Tests for Data/WebDownloads/crawl_fit_pdfs.py — pure helper functions."""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

import crawl_fit_pdfs as crawl


class TestSlugify:
    def test_vietnamese_diacritics_stripped(self):
        assert crawl.slugify("Đề cương Công nghệ thông tin") == "de-cuong-cong-nghe-thong-tin"

    def test_preserves_numbers(self):
        assert crawl.slugify("Khóa 2023-2024") == "khoa-2023-2024"

    def test_lowercase(self):
        assert crawl.slugify("ĐẠI HỌC KHOA HỌC") == "dai-hoc-khoa-hoc"

    def test_underscores_become_dashes(self):
        """Regression guard for T02: slugify must convert `_` to `-`."""
        assert crawl.slugify("hello_world_abc") == "hello-world-abc"

    def test_mixed_underscore_and_space(self):
        assert crawl.slugify("foo_ bar __baz") == "foo-bar-baz"

    def test_multiple_spaces_collapsed(self):
        assert crawl.slugify("a   b   c") == "a-b-c"

    def test_strips_leading_trailing_dashes(self):
        assert crawl.slugify("--- hello ---") == "hello"

    def test_drops_special_chars(self):
        assert crawl.slugify("hello @#$% world!") == "hello-world"


class TestDetectNganh:
    def test_cntt(self):
        assert crawl.detect_nganh("Chương trình đào tạo Công nghệ thông tin") == "cntt"

    def test_khmt(self):
        assert crawl.detect_nganh("Khoa học máy tính 2024") == "khmt"

    def test_ktpm(self):
        assert crawl.detect_nganh("Kỹ thuật phần mềm khóa 2023") == "ktpm"

    def test_default_chung(self):
        assert crawl.detect_nganh("Tài liệu tổng quát không nêu ngành") == "chung"


class TestDetectYear:
    def test_4_digit_year(self):
        assert crawl.detect_year("Khóa tuyển 2025") == "2025"

    def test_from_page_desc(self):
        assert crawl.detect_year("tài liệu A", page_desc="K2022") == "2022"

    def test_no_year_returns_unknown(self):
        assert crawl.detect_year("tài liệu không năm") == "unknown"


class TestDetectDocType:
    def test_chuong_trinh(self):
        assert crawl.detect_doc_type("Chương trình đào tạo 2025") == "chuong-trinh-dao-tao"

    def test_quyet_dinh(self):
        slug = crawl.detect_doc_type("Quyết định ban hành CTDT")
        # Should match one of the mapped doc-type slugs
        assert "quyet-dinh" in slug

    def test_fallback_slugified(self):
        result = crawl.detect_doc_type("Tài liệu đặc biệt lạ")
        # Fallback slugifies title and truncates to 60 chars
        assert result == crawl.slugify("Tài liệu đặc biệt lạ")[:60]


class TestBuildBasename:
    def test_format(self):
        result = crawl.build_basename(
            "chinh-quy", "Chương trình đào tạo Công nghệ thông tin 2025", "K2025"
        )
        assert result == "chinh-quy__cntt__2025__chuong-trinh-dao-tao"

    def test_unknown_year(self):
        result = crawl.build_basename("vlvh", "Tài liệu lạ", "")
        assert "vlvh__chung__unknown__" in result


class TestIsPdfLink:
    @pytest.mark.parametrize("href", [
        "file.pdf", "/path/doc.pdf", "https://x.com/a.pdf", "FILE.PDF",
    ])
    def test_pdf_extensions(self, href):
        assert crawl.is_pdf_link(href) is True

    @pytest.mark.parametrize("href", [
        "/doc.pdf?forcedownload=true",
        "https://x.com/report.pdf?v=2&dl=1",
        "/a.PDF?foo=bar",
    ])
    def test_pdf_with_query_string(self, href):
        """T03 regression guard: query strings must not break detection."""
        assert crawl.is_pdf_link(href) is True

    def test_linkclick_aspx(self):
        assert crawl.is_pdf_link("/LinkClick.aspx?fileticket=xyz&tabid=36") is True

    @pytest.mark.parametrize("href", [
        "file.html", "/home", "https://x.com", "#section", "javascript:void(0)",
    ])
    def test_non_pdf(self, href):
        assert crawl.is_pdf_link(href) is False


class TestExtractPdfLinks:
    def test_extracts_direct_pdfs(self):
        html = """
        <html><body>
            <a href="/doc1.pdf">Tài liệu A</a>
            <a href="/doc2.pdf" title="Tài liệu B">Short</a>
            <a href="/page.html">Not a PDF</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = crawl.extract_pdf_links(soup, "https://fit.hcmus.edu.vn/page")
        assert len(links) == 2
        titles = {link["title"] for link in links}
        assert "Tài liệu A" in titles
        assert any("doc2.pdf" in link["url"] for link in links)

    def test_deduplicates_by_url(self):
        html = """
        <html><body>
            <a href="/doc.pdf">First</a>
            <a href="/doc.pdf">Second</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = crawl.extract_pdf_links(soup, "https://fit.hcmus.edu.vn/")
        assert len(links) == 1

    def test_strips_forcedownload_when_url_already_matches(self):
        """Regression guard: forcedownload query param stripped after pass.

        Current behavior: `is_pdf_link` runs BEFORE query strip, so URLs like
        `/doc.pdf?forcedownload=true` are filtered out (endswith check fails).
        Links matching via `linkclick.aspx` or ending in `.pdf` cleanly DO
        get the forcedownload param stripped. This test covers the latter.
        """
        html = '<a href="/doc.pdf">Doc</a>'
        soup = BeautifulSoup(html, "html.parser")
        links = crawl.extract_pdf_links(soup, "https://fit.hcmus.edu.vn/")
        assert len(links) == 1
        assert "forcedownload" not in links[0]["url"].lower()

    def test_forcedownload_in_query_is_detected(self):
        """Regression guard for T03: `is_pdf_link` handles query strings."""
        html = '<a href="/doc.pdf?forcedownload=true">X</a>'
        soup = BeautifulSoup(html, "html.parser")
        links = crawl.extract_pdf_links(soup, "https://fit.hcmus.edu.vn/")
        assert len(links) == 1
        assert "forcedownload" not in links[0]["url"].lower()

    def test_skips_fragments_and_js(self):
        html = """
        <a href="#top">Top</a>
        <a href="javascript:void(0)">JS</a>
        <a href="/real.pdf">Real</a>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = crawl.extract_pdf_links(soup, "https://fit.hcmus.edu.vn/")
        assert len(links) == 1
        assert "real.pdf" in links[0]["url"]

    def test_fallback_to_parent_text(self):
        html = """
        <td>Tên tài liệu trong cell
            <a href="/doc.pdf"></a>
        </td>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = crawl.extract_pdf_links(soup, "https://x/")
        assert len(links) == 1
        assert "Tên tài liệu" in links[0]["title"]


class TestDiscoverSubpageTabids:
    def test_extracts_tabids_same_domain(self):
        html = """
        <a href="/vn/Default.aspx?tabid=100">Tab 100</a>
        <a href="/vn/Default.aspx?tabid=200">Tab 200</a>
        <a href="https://other.com/?tabid=300">External</a>
        """
        soup = BeautifulSoup(html, "html.parser")
        tabids = crawl.discover_subpage_tabids(soup, "https://fit.hcmus.edu.vn/")
        assert 100 in tabids
        assert 200 in tabids
        assert 300 not in tabids  # other domain

    def test_ignores_invalid_tabid(self):
        html = '<a href="?tabid=abc">Bad</a>'
        soup = BeautifulSoup(html, "html.parser")
        tabids = crawl.discover_subpage_tabids(soup, "https://fit.hcmus.edu.vn/")
        assert tabids == []

    def test_sorted(self):
        html = """
        <a href="?tabid=50">A</a>
        <a href="?tabid=10">B</a>
        <a href="?tabid=30">C</a>
        """
        soup = BeautifulSoup(html, "html.parser")
        tabids = crawl.discover_subpage_tabids(soup, "https://fit.hcmus.edu.vn/")
        assert tabids == [10, 30, 50]


class TestDeduplicateBasenames:
    def test_unique_names_untouched(self):
        pdfs = [{"basename": "a"}, {"basename": "b"}, {"basename": "c"}]
        out = crawl.deduplicate_basenames(pdfs)
        assert [p["basename"] for p in out] == ["a", "b", "c"]

    def test_duplicates_get_suffix(self):
        pdfs = [{"basename": "same"}, {"basename": "same"}, {"basename": "same"}]
        out = crawl.deduplicate_basenames(pdfs)
        assert [p["basename"] for p in out] == ["same", "same--2", "same--3"]

    def test_empty_list(self):
        assert crawl.deduplicate_basenames([]) == []
