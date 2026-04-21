from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config — nạp từ crawl_config.json cùng thư mục
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent / "crawl_config.json"

# Thư mục lưu PDF thô. OCR task sẽ đọc từ đây và ghi .txt cùng tên sang pdf_text/
PDF_DIR = Path(__file__).resolve().parent.parent / "Database" / "pdf_raw"
MANIFEST_PATH = PDF_DIR / "manifest.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Nạp file crawl_config.json."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


_CONFIG = load_config()

BASE_URL: str = _CONFIG["base_url"]
ENTRY_PAGE = f"{BASE_URL}/vn/Default.aspx?tabid={_CONFIG['entry_tabid']}"

HEADERS = {
    "User-Agent": _CONFIG["user_agent"],
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5",
}

REQUEST_DELAY: float = _CONFIG["request_delay"]
REQUEST_TIMEOUT: int = _CONFIG["request_timeout"]

# Các trang chứa link CTĐT theo hệ đào tạo: tabid → (hệ đào tạo, mô tả)
CTDT_PAGES: dict[int, tuple[str, str]] = {
    int(entry["tabid"]): (entry["he_dao_tao"], entry["desc"])
    for entry in _CONFIG["ctdt_pages"]
}

SUBPAGE_KEYWORDS: list[str] = _CONFIG["subpage_keywords"]

# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

_VIET_MAP = str.maketrans({
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
    "À": "A", "Á": "A", "Ả": "A", "Ã": "A", "Ạ": "A",
    "Ă": "A", "Ằ": "A", "Ắ": "A", "Ẳ": "A", "Ẵ": "A", "Ặ": "A",
    "Â": "A", "Ầ": "A", "Ấ": "A", "Ẩ": "A", "Ẫ": "A", "Ậ": "A",
    "È": "E", "É": "E", "Ẻ": "E", "Ẽ": "E", "Ẹ": "E",
    "Ê": "E", "Ề": "E", "Ế": "E", "Ể": "E", "Ễ": "E", "Ệ": "E",
    "Ì": "I", "Í": "I", "Ỉ": "I", "Ĩ": "I", "Ị": "I",
    "Ò": "O", "Ó": "O", "Ỏ": "O", "Õ": "O", "Ọ": "O",
    "Ô": "O", "Ồ": "O", "Ổ": "O", "Ỗ": "O", "Ộ": "O",
    "Ơ": "O", "Ờ": "O", "Ớ": "O", "Ở": "O", "Ỡ": "O", "Ợ": "O",
    "Ù": "U", "Ú": "U", "Ủ": "U", "Ũ": "U", "Ụ": "U",
    "Ư": "U", "Ừ": "U", "Ứ": "U", "Ử": "U", "Ữ": "U", "Ự": "U",
    "Ỳ": "Y", "Ý": "Y", "Ỷ": "Y", "Ỹ": "Y", "Ỵ": "Y",
    "Đ": "D",
})


def slugify(text: str) -> str:
    """Chuyển text tiếng Việt thành slug ASCII.

    - Drop diacritics → ASCII
    - Underscore/whitespace collapsed into single dash
    - Non-alphanumeric stripped
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().translate(_VIET_MAP)
    # Preserve underscore so the next step can convert it to dash
    text = re.sub(r"[^a-z0-9_\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Detect chuyên ngành + năm từ tên tài liệu
# ---------------------------------------------------------------------------

_NGANH_PATTERNS: list[tuple[str, str]] = [
    (pattern, slug) for pattern, slug in _CONFIG["nganh_patterns"]
]

_DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
    (pattern, slug) for pattern, slug in _CONFIG["doc_type_patterns"]
]


def detect_nganh(title: str) -> str:
    """Detect chuyên ngành từ tên tài liệu."""
    for pattern, slug in _NGANH_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return slug
    return "chung"


def detect_year(title: str, page_desc: str = "") -> str:
    """Detect năm đào tạo từ tên tài liệu hoặc mô tả trang."""
    m = re.search(r"(?:khóa|khoá|K\.?\s*)?(?:tuyển\s+)?(\d{4})", title)
    if m:
        return m.group(1)
    m = re.search(r"K?(\d{4})", page_desc)
    if m:
        return m.group(1)
    return "unknown"


def detect_doc_type(title: str) -> str:
    """Detect loại tài liệu từ tên."""
    for pattern, slug in _DOC_TYPE_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return slug
    return slugify(title)[:60]


def build_basename(he_dao_tao: str, title: str, page_desc: str) -> str:
    """Tạo basename chuẩn hóa (không đuôi file).

    Format: {he_dao_tao}__{chuyen_nganh}__{nam}__{ten_tai_lieu}
    """
    nganh = detect_nganh(title)
    year = detect_year(title, page_desc)
    doc_type = detect_doc_type(title)
    return "__".join([he_dao_tao, nganh, year, doc_type])


_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch and parse an HTML page."""
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type.lower() and "text" not in content_type.lower():
            return None
        resp.encoding = resp.apparent_encoding or "utf-8"
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [ERROR] Fetch failed {url}: {e}")
        return None


def is_pdf_link(href: str) -> bool:
    """Check if a link points to a PDF.

    Handles URLs with query strings (`/doc.pdf?forcedownload=true`) by
    inspecting the parsed path, not the raw href.
    """
    lower = href.lower()
    # Inspect parsed path so query string doesn't break detection.
    try:
        path = urlparse(href).path.lower()
    except Exception:
        path = lower
    if path.endswith(".pdf"):
        return True
    if "linkclick.aspx" in lower and "fileticket" in lower:
        return True
    return False


def extract_pdf_links(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Extract all PDF download links from a page."""
    links = []
    seen_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        if not is_pdf_link(href):
            continue

        full_url = urljoin(page_url, href)
        # Strip forcedownload regardless of position (? or &, first or later).
        clean_url = re.sub(
            r"[?&]forcedownload=true",
            lambda m: "?" if m.group(0).startswith("?") else "",
            full_url,
            flags=re.IGNORECASE,
        )
        # Collapse leftover trailing "?" or "?&" from the substitution.
        clean_url = re.sub(r"\?&", "?", clean_url)
        clean_url = re.sub(r"\?$", "", clean_url)

        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        title = a_tag.get_text(strip=True)
        if not title or len(title) < 3:
            title = a_tag.get("title", "")
        if not title:
            parent = a_tag.find_parent(["td", "li", "div", "p"])
            if parent:
                title = parent.get_text(strip=True)[:200]

        size_text = ""
        next_sib = a_tag.next_sibling
        if next_sib and isinstance(next_sib, str):
            size_match = re.search(r"[\d.]+\s*[KMG]B", next_sib, re.IGNORECASE)
            if size_match:
                size_text = size_match.group()

        links.append({
            "url": clean_url,
            "title": title or "untitled",
            "size": size_text,
        })

    return links


def discover_subpage_tabids(soup: BeautifulSoup, page_url: str) -> list[int]:
    """Discover additional tabid sub-pages linked from this page."""
    tabids = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(page_url, href)
        parsed = urlparse(full_url)
        if "fit.hcmus.edu.vn" not in parsed.netloc:
            continue
        qs = parse_qs(parsed.query)
        if "tabid" in qs:
            try:
                tid = int(qs["tabid"][0])
                tabids.add(tid)
            except ValueError:
                pass
    return sorted(tabids)


def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF file."""
    try:
        resp = _session.get(url, timeout=60, stream=True, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            first_bytes = next(resp.iter_content(chunk_size=8), b"")
            if not first_bytes.startswith(b"%PDF"):
                print(f"  [SKIP] Not a PDF: {url} (Content-Type: {content_type})")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(first_bytes)
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed {url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

def crawl_all_pages(
    year_filter: list[str] | None = None,
) -> list[dict]:
    """Crawl tất cả trang CTĐT, thu thập link PDF."""
    all_pdfs: list[dict] = []
    visited_tabids: set[int] = set()
    seen_urls: set[str] = set()

    queue = list(CTDT_PAGES.items())

    print(f"\n{'='*70}")
    print(f"  CRAWL PDF CHƯƠNG TRÌNH ĐÀO TẠO - FIT HCMUS")
    print(f"  Số trang cần crawl: {len(queue)}")
    if year_filter:
        print(f"  Lọc theo năm: {', '.join(year_filter)}")
    print(f"{'='*70}\n")

    while queue:
        tabid, (he_dao_tao, desc) = queue.pop(0)

        if tabid in visited_tabids:
            continue
        visited_tabids.add(tabid)

        page_url = f"{BASE_URL}/vn/Default.aspx?tabid={tabid}"
        print(f"[PAGE] tabid={tabid} | {he_dao_tao} | {desc}")

        soup = fetch_page(page_url)
        if soup is None:
            continue

        pdf_links = extract_pdf_links(soup, page_url)
        print(f"  → Tìm thấy {len(pdf_links)} PDF links")

        for pdf_info in pdf_links:
            url = pdf_info["url"]
            title = pdf_info["title"]

            if url in seen_urls:
                continue
            seen_urls.add(url)

            year = detect_year(title, desc)
            if year_filter and year not in year_filter and "unknown" not in year_filter:
                continue

            basename = build_basename(he_dao_tao, title, desc)
            pdf_info["he_dao_tao"] = he_dao_tao
            pdf_info["page_desc"] = desc
            pdf_info["basename"] = basename
            pdf_info["year"] = year
            pdf_info["nganh"] = detect_nganh(title)

            all_pdfs.append(pdf_info)
            print(f"    📄 {basename}.pdf")
            print(f"       ← {title[:80]}")

        new_tabids = discover_subpage_tabids(soup, page_url)
        for tid in new_tabids:
            if tid not in visited_tabids and tid not in {t for t, _ in queue}:
                if tid not in CTDT_PAGES:
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        if f"tabid={tid}" in href:
                            link_text = a_tag.get_text(strip=True).lower()
                            if any(kw in link_text for kw in SUBPAGE_KEYWORDS):
                                queue.append((tid, (he_dao_tao, f"Auto: {link_text[:50]}")))
                                break

        time.sleep(REQUEST_DELAY)

    return all_pdfs


def deduplicate_basenames(pdfs: list[dict]) -> list[dict]:
    """Ensure unique basenames by appending suffix if needed."""
    name_counts: dict[str, int] = {}
    for pdf in pdfs:
        name = pdf["basename"]
        if name in name_counts:
            name_counts[name] += 1
            pdf["basename"] = f"{name}--{name_counts[name]}"
        else:
            name_counts[name] = 1
    return pdfs


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Manifest cũ không đọc được ({e}) — tạo mới")
        return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Crawl PDF CTĐT từ FIT HCMUS (chỉ tải, không OCR)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ liệt kê, không tải")
    parser.add_argument("--years", nargs="*", help="Chỉ crawl một số năm (vd: 2023 2024)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Thư mục lưu PDF (mặc định: {PDF_DIR})",
    )
    args = parser.parse_args()

    pdf_dir: Path = args.output_dir if args.output_dir else PDF_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pdf_dir / "manifest.json"

    # --- Step 1: Crawl all pages ---
    pdfs = crawl_all_pages(year_filter=args.years)
    pdfs = deduplicate_basenames(pdfs)

    print(f"\n{'='*70}")
    print(f"  TỔNG KẾT CRAWL")
    print(f"  Tổng PDF tìm thấy: {len(pdfs)}")
    print(f"{'='*70}")

    if args.dry_run:
        print("\n[DRY RUN] Danh sách PDF sẽ được tải:\n")
        for i, pdf in enumerate(pdfs, 1):
            print(f"  {i:3d}. {pdf['basename']}.pdf")
            print(f"       URL: {pdf['url']}")
            print(f"       Title: {pdf['title'][:80]}")
            print(f"       Hệ: {pdf['he_dao_tao']} | Ngành: {pdf['nganh']} | Năm: {pdf['year']}")
            print()
        return

    # --- Step 2: Download PDF + cập nhật manifest ---
    manifest = load_manifest(manifest_path)
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, pdf in enumerate(pdfs, 1):
        pdf_filename = f"{pdf['basename']}.pdf"
        pdf_path = pdf_dir / pdf_filename

        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"[{i}/{len(pdfs)}] SKIP (đã tồn tại): {pdf_filename}")
            manifest[pdf_filename] = {
                "title": pdf["title"],
                "url": pdf["url"],
                "he_dao_tao": pdf["he_dao_tao"],
                "nganh": pdf["nganh"],
                "year": pdf["year"],
                "page_desc": pdf["page_desc"],
                "size": pdf.get("size", ""),
            }
            skip_count += 1
            continue

        print(f"[{i}/{len(pdfs)}] Downloading: {pdf['title'][:60]}...")
        ok = download_pdf(pdf["url"], pdf_path)
        if not ok:
            error_count += 1
            continue

        manifest[pdf_filename] = {
            "title": pdf["title"],
            "url": pdf["url"],
            "he_dao_tao": pdf["he_dao_tao"],
            "nganh": pdf["nganh"],
            "year": pdf["year"],
            "page_desc": pdf["page_desc"],
            "size": pdf.get("size", ""),
        }
        print(f"  ✓ Saved: {pdf_filename} ({pdf_path.stat().st_size // 1024} KB)")
        success_count += 1
        time.sleep(REQUEST_DELAY)

    save_manifest(manifest_path, manifest)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  KẾT QUẢ CRAWL")
    print(f"  Thành công: {success_count}")
    print(f"  Bỏ qua (đã có): {skip_count}")
    print(f"  Lỗi: {error_count}")
    print(f"  PDF dir: {pdf_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*70}")
    print(f"\n  Bước tiếp theo — chạy OCR:")
    print(f"    python llm_ocr_pdf.py --input-dir {pdf_dir}")


if __name__ == "__main__":
    main()
