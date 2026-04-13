"""
Crawl PDF files về chương trình đào tạo từ website FIT HCMUS.

Quy trình:
  1. Bắt đầu từ trang gốc (tabid=97), duyệt tất cả trang con liên quan đến CTĐT
  2. Thu thập tất cả link PDF (LinkClick.aspx + .pdf trực tiếp)
  3. Tải PDF về thư mục tạm
  4. Trích xuất text bằng PyMuPDF (không cần Tesseract cho PDF có text layer)
  5. Nếu PDF là ảnh scan (không có text), dùng OCR fallback
  6. Lưu file .txt vào Data/Database/ với tên chuẩn hóa

Tên file output:
  {he_dao_tao}__{chuyen_nganh}__{nam}__{ten_tai_lieu}.txt

  Ví dụ:
  - chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt
  - chinh-quy__cu-nhan-tai-nang__2024__chuong-trinh-dao-tao.txt
  - tu-xa__cntt__2023__chuong-trinh-dao-tao.txt
  - chinh-quy__chung__2022__quyet-dinh-ban-hanh-ctdt.txt

Cài đặt:
  pip install requests beautifulsoup4 pymupdf

Sử dụng:
  python crawl_fit_pdfs.py
  python crawl_fit_pdfs.py --dry-run          # Chỉ liệt kê, không tải
  python crawl_fit_pdfs.py --skip-ocr         # Bỏ qua OCR cho PDF scan
  python crawl_fit_pdfs.py --years 2023 2024  # Chỉ crawl một số năm
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.fit.hcmus.edu.vn"
ENTRY_PAGE = f"{BASE_URL}/vn/Default.aspx?tabid=97"

OUTPUT_DIR = Path(__file__).resolve().parent / "Database" / "pdf_crawled"
TEMP_DIR = Path(__file__).resolve().parent / ".tmp_pdfs"

HEADERS = {
    "User-Agent": "RAG-Chatbot-DataPipeline/2.0 (FIT HCMUS Crawler)",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5",
}

REQUEST_DELAY = 1.0  # giây giữa các request
REQUEST_TIMEOUT = 30

# Các trang chứa link CTĐT theo hệ đào tạo
# Key: tabid → (hệ đào tạo, mô tả)
CTDT_PAGES = {
    # === CHÍNH QUY ===
    289: ("chinh-quy", "Trang tổng CTĐT chính quy"),
    290: ("chinh-quy", "Trước 2005"),
    291: ("chinh-quy", "2005-2007"),
    830: ("chinh-quy", "K2011"),
    910: ("chinh-quy", "K2012"),
    955: ("chinh-quy", "K2013"),
    957: ("chinh-quy", "K2014"),
    956: ("chinh-quy", "K2015"),
    961: ("chinh-quy", "K2016"),
    1106: ("chinh-quy", "K2017-2018"),
    1129: ("chinh-quy", "K2019"),
    1133: ("chinh-quy", "K2020"),
    1172: ("chinh-quy", "K2021"),
    1283: ("chinh-quy", "K2022"),
    1284: ("chinh-quy", "K2023"),
    1287: ("chinh-quy", "K2024"),
    1310: ("chinh-quy", "K2025"),
    562: ("chinh-quy", "Bảng chuyển đổi HP"),
    987: ("chinh-quy", "Học phần tiên quyết"),
    # === TỐT NGHIỆP ===
    1064: ("chinh-quy", "Tốt nghiệp - Biểu mẫu"),
    1065: ("chinh-quy", "Tốt nghiệp - Quy trình"),
    # === HỘI ĐỒNG ĐẠO TẠO ===
    740: ("hoi-dong-dao-tao", "CTĐT HCĐH"),
    # === CAO ĐẲNG ===
    501: ("cao-dang", "CTĐT Cao đẳng"),
    # === ĐÀO TẠO TỪ XA ===
    960: ("tu-xa", "Trang tổng CTĐT từ xa"),
    1286: ("tu-xa", "ĐTTX K2023"),
    1160: ("tu-xa", "ĐTTX K2022"),
    1137: ("tu-xa", "ĐTTX K2021"),
    1138: ("tu-xa", "ĐTTX K2020"),
    1139: ("tu-xa", "ĐTTX K2016-2019"),
    1289: ("tu-xa", "ĐTTX K2010"),
    1291: ("tu-xa", "ĐTTX K2011"),
    1292: ("tu-xa", "ĐTTX K2012"),
    1294: ("tu-xa", "ĐTTX K2013"),
    1295: ("tu-xa", "ĐTTX K2014"),
    1296: ("tu-xa", "ĐTTX K2015"),
    1301: ("tu-xa", "ĐTTX Chuyển đổi HP"),
}

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
    "Ô": "O", "Ồ": "O", "Ố": "O", "Ổ": "O", "Ỗ": "O", "Ộ": "O",
    "Ơ": "O", "Ờ": "O", "Ớ": "O", "Ở": "O", "Ỡ": "O", "Ợ": "O",
    "Ù": "U", "Ú": "U", "Ủ": "U", "Ũ": "U", "Ụ": "U",
    "Ư": "U", "Ừ": "U", "Ứ": "U", "Ử": "U", "Ữ": "U", "Ự": "U",
    "Ỳ": "Y", "Ý": "Y", "Ỷ": "Y", "Ỹ": "Y", "Ỵ": "Y",
    "Đ": "D",
})


def slugify(text: str) -> str:
    """Chuyển text tiếng Việt thành slug ASCII."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().translate(_VIET_MAP)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Detect chuyên ngành + năm từ tên tài liệu
# ---------------------------------------------------------------------------

_NGANH_PATTERNS = [
    (r"công\s*nghệ\s*thông\s*tin", "cntt"),
    (r"hệ\s*thống\s*thông\s*tin", "httt"),
    (r"khoa\s*học\s*máy\s*tính", "khmt"),
    (r"kỹ\s*thuật\s*phần\s*mềm", "ktpm"),
    (r"trí\s*tuệ\s*nhân\s*tạo", "ttnt"),
    (r"cử\s*nhân\s*tài\s*năng", "cu-nhan-tai-nang"),
    (r"CNTT", "cntt"),
    (r"HTTT", "httt"),
    (r"KHMT", "khmt"),
    (r"KTPM", "ktpm"),
    (r"TTNT", "ttnt"),
    (r"CNTN", "cu-nhan-tai-nang"),
]

_DOC_TYPE_PATTERNS = [
    (r"quyết\s*định\s*ban\s*hành", "quyet-dinh-ban-hanh"),
    (r"bảng\s*chuyển\s*đổi", "bang-chuyen-doi-hoc-phan"),
    (r"chuyển\s*đổi\s*h[ọo]c\s*ph[ầa]n", "bang-chuyen-doi-hoc-phan"),
    (r"chương\s*trình\s*đào\s*tạo", "chuong-trinh-dao-tao"),
    (r"CTĐT", "chuong-trinh-dao-tao"),
    (r"điều\s*chỉnh", "dieu-chinh"),
    (r"bổ\s*sung", "bo-sung"),
    (r"liên\s*thông", "lien-thong"),
    (r"CV\s*chuyển\s*đổi", "cong-van-chuyen-doi"),
    (r"DSHP", "danh-sach-hoc-phan"),
]


def detect_nganh(title: str) -> str:
    """Detect chuyên ngành từ tên tài liệu."""
    for pattern, slug in _NGANH_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return slug
    return "chung"


def detect_year(title: str, page_desc: str = "") -> str:
    """Detect năm đào tạo từ tên tài liệu hoặc mô tả trang."""
    # Tìm trong title trước
    m = re.search(r"(?:khóa|khoá|K\.?\s*)?(?:tuyển\s+)?(\d{4})", title)
    if m:
        return m.group(1)
    # Tìm trong page description
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


def build_filename(he_dao_tao: str, title: str, page_desc: str) -> str:
    """Tạo tên file chuẩn hóa từ metadata.

    Format: {he_dao_tao}__{chuyen_nganh}__{nam}__{ten_tai_lieu}.txt
    """
    nganh = detect_nganh(title)
    year = detect_year(title, page_desc)
    doc_type = detect_doc_type(title)

    parts = [he_dao_tao, nganh, year, doc_type]
    filename = "__".join(parts) + ".txt"

    # Deduplicate nếu đã tồn tại
    return filename


# ---------------------------------------------------------------------------
# Web helpers
# ---------------------------------------------------------------------------

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
    """Check if a link points to a PDF."""
    lower = href.lower()
    if lower.endswith(".pdf"):
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
        # Bỏ forcedownload param để lấy URL gốc
        clean_url = re.sub(r"&forcedownload=true", "", full_url, flags=re.IGNORECASE)

        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        # Lấy title từ text của link hoặc title attribute
        title = a_tag.get_text(strip=True)
        if not title or len(title) < 3:
            title = a_tag.get("title", "")
        if not title:
            # Thử lấy từ parent element
            parent = a_tag.find_parent(["td", "li", "div", "p"])
            if parent:
                title = parent.get_text(strip=True)[:200]

        # Lấy file size nếu có
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


# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF file."""
    try:
        resp = _session.get(url, timeout=60, stream=True, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            # Check magic bytes
            first_bytes = next(resp.iter_content(chunk_size=8), b"")
            if not first_bytes.startswith(b"%PDF"):
                print(f"  [SKIP] Not a PDF: {url} (Content-Type: {content_type})")
                return False
            # Write first bytes + rest
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


# Ký tự có dấu tiếng Việt (Unicode, dải \u1ea0-\u1ef9 là toàn bộ chữ Việt có dấu)
_VIET_ACCENTED = re.compile(
    "[\u00e0\u00e1\u00e2\u00e3\u00e8\u00e9\u00ea\u00ec\u00ed\u00f2\u00f3\u00f4\u00f5\u00f9\u00fa\u00fd"
    "\u0103\u0111\u01b0\u01a1\u1ea0-\u1ef9]",
    re.UNICODE,
)


def _is_vietnamese_text(text: str, min_ratio: float = 0.015) -> bool:
    """
    Kiểm tra text có chứa dấu tiếng Việt Unicode không.

    PDF dùng font encoding cũ (VNI/TCVN3/ABC) sẽ trả về text ASCII nhưng
    mất hoàn toàn dấu thanh. Nếu tỷ lệ ký tự có dấu < min_ratio thì coi
    là bị garbled encoding → cần OCR.

    Returns:
        True  → text hợp lệ, có dấu tiếng Việt
        False → nghi ngờ garbled encoding / không phải UTF-8 Việt
    """
    if not text or len(text) < 20:
        return False
    accented = len(_VIET_ACCENTED.findall(text))
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False
    return (accented / alpha) >= min_ratio


def _load_llm_ocr():
    """Lazy import llm_ocr_pdf từ cùng thư mục."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_ocr_pdf",
            Path(__file__).parent / "llm_ocr_pdf.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"  [WARN] Không load được llm_ocr_pdf: {e}")
        return None


def extract_text_from_pdf(
    pdf_path: Path,
    enable_ocr: bool = True,
    ocr_model: str = "qwen",
) -> str:

    try:
        import fitz
    except ImportError:
        print("  [ERROR] PyMuPDF not installed. Run: pip install pymupdf")
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        text_pages: list[str | None] = []
        scan_count = 0

        for page_num in range(total_pages):
            t = doc[page_num].get_text("text").strip()
            if t and len(t) > 30:
                text_pages.append(t)
            else:
                text_pages.append(None)
                scan_count += 1

        doc.close()
    except Exception as e:
        print(f"  [ERROR] PDF open failed {pdf_path.name}: {e}")
        return ""

    scan_ratio = scan_count / max(total_pages, 1)
    print(f"  [PDF] {total_pages} trang | {scan_count} scan ({scan_ratio:.0%})")

    # --- Nếu phần lớn là scan → dùng OCR pipeline ---
    if enable_ocr and scan_ratio > 0.3:
        llm_ocr = _load_llm_ocr()
        if llm_ocr:
            print(f"  [OCR] Dùng pipeline: {ocr_model}")
            try:
                return llm_ocr.extract_with_llm_fallback(
                    pdf_path, model=ocr_model, verbose=True
                )
            except Exception as e:
                print(f"  [WARN] OCR pipeline lỗi: {e} — fallback tesseract")


    # --- PDF có text layer → ghép lại ---
    parts = []
    for i, t in enumerate(text_pages):
        if t:
            parts.append(t)
        elif enable_ocr:
            parts.append(f"[Trang {i + 1}: ảnh scan]")
    return "\n\n".join(parts)



# ---------------------------------------------------------------------------
# Clean extracted text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean and normalize extracted Vietnamese text."""
    text = unicodedata.normalize("NFC", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix common PDF extraction artifacts
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("–", "-")
    # Remove repeated short lines (headers/footers)
    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 80:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    repeated = {line for line, count in line_counts.items() if count > 5}
    if repeated:
        lines = [line for line in lines if line.strip() not in repeated]
        text = "\n".join(lines)

    return text.strip()


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

            filename = build_filename(he_dao_tao, title, desc)
            pdf_info["he_dao_tao"] = he_dao_tao
            pdf_info["page_desc"] = desc
            pdf_info["filename"] = filename
            pdf_info["year"] = year
            pdf_info["nganh"] = detect_nganh(title)

            all_pdfs.append(pdf_info)
            print(f"    📄 {filename}")
            print(f"       ← {title[:80]}")

        # Discover thêm sub-pages chưa biết
        new_tabids = discover_subpage_tabids(soup, page_url)
        for tid in new_tabids:
            if tid not in visited_tabids and tid not in {t for t, _ in queue}:
                # Chỉ thêm nếu link nằm trong scope CTĐT
                if tid not in CTDT_PAGES:
                    # Check xem link text có liên quan CTĐT không
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        if f"tabid={tid}" in href:
                            link_text = a_tag.get_text(strip=True).lower()
                            if any(kw in link_text for kw in [
                                "chương trình", "ctđt", "khóa", "đào tạo",
                                "curriculum", "đttx",
                            ]):
                                queue.append((tid, (he_dao_tao, f"Auto: {link_text[:50]}")))
                                break

        time.sleep(REQUEST_DELAY)

    return all_pdfs


def deduplicate_filenames(pdfs: list[dict]) -> list[dict]:
    """Ensure unique filenames by appending suffix if needed."""
    name_counts: dict[str, int] = {}
    for pdf in pdfs:
        name = pdf["filename"]
        if name in name_counts:
            name_counts[name] += 1
            base = name.rsplit(".", 1)[0]
            pdf["filename"] = f"{base}--{name_counts[name]}.txt"
        else:
            name_counts[name] = 1
    return pdfs


def main():
    parser = argparse.ArgumentParser(description="Crawl PDF CTĐT từ FIT HCMUS")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ liệt kê, không tải")
    parser.add_argument("--skip-ocr", action="store_true", help="Bỏ qua OCR cho PDF scan")
    parser.add_argument("--years", nargs="*", help="Chỉ crawl một số năm (vd: 2023 2024)")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục output")
    parser.add_argument(
        "--ocr-model",
        choices=["qwen", "paddle-only", "gemini", "gpt4o", "tesseract"],
        default="qwen",
        help="Model OCR cho PDF scan (mặc định: qwen = PaddleOCR + Qwen2.5-VL)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Crawl all pages ---
    pdfs = crawl_all_pages(year_filter=args.years)
    pdfs = deduplicate_filenames(pdfs)

    print(f"\n{'='*70}")
    print(f"  TỔNG KẾT CRAWL")
    print(f"  Tổng PDF tìm thấy: {len(pdfs)}")
    print(f"{'='*70}")

    if args.dry_run:
        print("\n[DRY RUN] Danh sách PDF sẽ được tải:\n")
        for i, pdf in enumerate(pdfs, 1):
            print(f"  {i:3d}. {pdf['filename']}")
            print(f"       URL: {pdf['url']}")
            print(f"       Title: {pdf['title'][:80]}")
            print(f"       Hệ: {pdf['he_dao_tao']} | Ngành: {pdf['nganh']} | Năm: {pdf['year']}")
            print()
        return

    # --- Step 2: Download + Extract ---
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, pdf in enumerate(pdfs, 1):
        txt_path = output_dir / pdf["filename"]
        if txt_path.exists():
            print(f"\n[{i}/{len(pdfs)}] SKIP (đã tồn tại): {pdf['filename']}")
            skip_count += 1
            continue

        print(f"\n[{i}/{len(pdfs)}] Downloading: {pdf['title'][:60]}...")

        # Download PDF
        pdf_filename = hashlib.md5(pdf["url"].encode()).hexdigest() + ".pdf"
        pdf_path = TEMP_DIR / pdf_filename

        if not pdf_path.exists():
            ok = download_pdf(pdf["url"], pdf_path)
            if not ok:
                error_count += 1
                continue
            time.sleep(REQUEST_DELAY)

        # Extract text
        print(f"  Extracting text... (ocr-model: {args.ocr_model})")
        raw_text = extract_text_from_pdf(
            pdf_path,
            enable_ocr=not args.skip_ocr,
            ocr_model=args.ocr_model,
        )
        if not raw_text or len(raw_text.strip()) < 50:
            print(f"  [WARN] Extracted text too short ({len(raw_text)} chars)")
            error_count += 1
            continue

        # Clean text
        cleaned = clean_text(raw_text)

        # Add metadata header
        header = (
            f"# Tài liệu: {pdf['title']}\n"
            f"# Hệ đào tạo: {pdf['he_dao_tao']}\n"
            f"# Chuyên ngành: {pdf['nganh']}\n"
            f"# Năm: {pdf['year']}\n"
            f"# Nguồn: {pdf['url']}\n"
            f"# ---\n\n"
        )

        txt_path.write_text(header + cleaned, encoding="utf-8")
        print(f"  ✓ Saved: {pdf['filename']} ({len(cleaned):,} chars)")
        success_count += 1

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  KẾT QUẢ")
    print(f"  Thành công: {success_count}")
    print(f"  Bỏ qua (đã có): {skip_count}")
    print(f"  Lỗi: {error_count}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    # Cleanup temp
    if TEMP_DIR.exists():
        print(f"\n  [INFO] File PDF tạm lưu tại: {TEMP_DIR}")
        print(f"  Xóa bằng: rm -rf {TEMP_DIR}")


if __name__ == "__main__":
    main()
