"""
Crawl PDF files về chương trình đào tạo từ website FIT HCMUS.

Quy trình:
  1. Bắt đầu từ trang gốc (tabid=97), duyệt tất cả trang con liên quan đến CTĐT
  2. Thu thập tất cả link PDF (LinkClick.aspx + .pdf trực tiếp)
  3. Tải PDF về thư mục tạm
  4. Trích xuất text bằng PyMuPDF (không cần Tesseract cho PDF có text layer)
  5. Nếu PDF là ảnh scan (không có text), dùng OCR fallback
  6. Lưu file .txt vào Data/Database/pdf_crawled/ với tên chuẩn hóa

Tên file output:
  {he_dao_tao}__{chuyen_nganh}__{nam}__{ten_tai_lieu}.txt

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
import logging
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from llm_ocr_pdf import clean_text as _normalize_text
from llm_ocr_pdf import extract_with_llm_fallback

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

REQUEST_DELAY = 1.0            # giây giữa các request
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
CHUNK_SIZE = 8192

# Ngưỡng heuristic cho extract & clean
MIN_TEXT_LEN_PER_PAGE = 30     # page có >30 ký tự coi là có text layer
OCR_SCAN_RATIO_THRESHOLD = 0.3 # >30% page là scan → kích hoạt OCR
MIN_EXTRACTED_LEN = 50         # text extract <50 ký tự → coi như fail
MAX_HEADER_FOOTER_LEN = 80     # dòng ngắn hơn mức này mới check header/footer
MIN_HEADER_FOOTER_REPEAT = 5   # lặp nhiều hơn mức này → xem là header/footer
DOC_TYPE_SLUG_MAXLEN = 60

# Các trang chứa link CTĐT theo hệ đào tạo
# Key: tabid → (hệ đào tạo, mô tả)
CTDT_PAGES: dict[int, tuple[str, str]] = {
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

logger = logging.getLogger("fit_crawler")


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Chuyển text tiếng Việt thành slug ASCII (dạng kebab-case)."""
    # đ/Đ không phân rã qua NFD nên xử lý trước
    text = text.replace("đ", "d").replace("Đ", "D")
    # Tách dấu khỏi ký tự gốc rồi loại bỏ combining marks
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Detect chuyên ngành + năm từ tên tài liệu
# ---------------------------------------------------------------------------

_NGANH_PATTERNS: list[tuple[str, str]] = [
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

_DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
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
    return slugify(title)[:DOC_TYPE_SLUG_MAXLEN]


def build_filename(he_dao_tao: str, title: str, page_desc: str) -> str:
    """Tạo tên file theo format {he}__{nganh}__{nam}__{doc_type}.txt (có thể trùng, dedupe sau)."""
    parts = [
        he_dao_tao,
        detect_nganh(title),
        detect_year(title, page_desc),
        detect_doc_type(title),
    ]
    return "__".join(parts) + ".txt"


# ---------------------------------------------------------------------------
# Web helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch và parse HTML page. Trả về None nếu fail hoặc response không phải HTML."""
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Fetch failed %s: %s", url, e)
        return None

    content_type = resp.headers.get("Content-Type", "").lower()
    if "html" not in content_type and "text" not in content_type:
        return None
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def is_pdf_link(href: str) -> bool:
    """Check if a link points to a PDF."""
    lower = href.lower()
    if lower.endswith(".pdf"):
        return True
    if "linkclick.aspx" in lower and "fileticket" in lower:
        return True
    return False


def _extract_link_title(a_tag) -> str:
    """Lấy title hiển thị cho một <a>: text, attr title, hoặc text của parent."""
    title = a_tag.get_text(strip=True)
    if title and len(title) >= 3:
        return title
    title = a_tag.get("title", "")
    if title:
        return title
    parent = a_tag.find_parent(["td", "li", "div", "p"])
    if parent:
        return parent.get_text(strip=True)[:200]
    return ""


def extract_pdf_links(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Extract tất cả link PDF trong một trang, đã dedupe theo URL."""
    links: list[dict] = []
    seen_urls: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        if not is_pdf_link(href):
            continue

        full_url = urljoin(page_url, href)
        clean_url = re.sub(r"&forcedownload=true", "", full_url, flags=re.IGNORECASE)
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        links.append({
            "url": clean_url,
            "title": _extract_link_title(a_tag) or "untitled",
        })

    return links


def discover_subpage_tabids(soup: BeautifulSoup, page_url: str) -> list[int]:
    """Discover các tabid trang con link từ trang hiện tại."""
    tabids: set[int] = set()
    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(page_url, a_tag["href"])
        parsed = urlparse(full_url)
        if "fit.hcmus.edu.vn" not in parsed.netloc:
            continue
        qs = parse_qs(parsed.query)
        if "tabid" not in qs:
            continue
        try:
            tabids.add(int(qs["tabid"][0]))
        except ValueError:
            pass
    return sorted(tabids)


# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> bool:
    """Download PDF về `dest`. Check magic bytes nếu Content-Type không khẳng định."""
    try:
        resp = _session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Download failed %s: %s", url, e)
        return False

    content_type = resp.headers.get("Content-Type", "").lower()
    prefix = b""
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        prefix = next(resp.iter_content(chunk_size=8), b"")
        if not prefix.startswith(b"%PDF"):
            logger.info("Not a PDF: %s (Content-Type: %s)", url, content_type)
            return False

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            if prefix:
                f.write(prefix)
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
    except OSError as e:
        logger.error("Write failed %s: %s", dest, e)
        return False
    return True


def _load_llm_ocr():
    """Lazy import llm_ocr_pdf từ cùng thư mục."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_ocr_pdf",
            Path(__file__).parent / "llm_ocr_pdf.py",
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Không load được llm_ocr_pdf: %s", e)
        return None


def extract_text_from_pdf(
    pdf_path: Path,
    enable_ocr: bool = True,
    ocr_model: str = "qwen",
) -> str:
    """Extract text từ PDF. Dùng PyMuPDF nếu có text layer, fallback OCR nếu phần lớn là ảnh scan."""
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install pymupdf")
        return ""

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:  # fitz raises nhiều loại lỗi khác nhau
        logger.error("PDF open failed %s: %s", pdf_path.name, e)
        return ""

    try:
        total_pages = len(doc)
        text_pages: list[str | None] = []
        scan_count = 0
        for page_num in range(total_pages):
            t = doc[page_num].get_text("text").strip()
            if t and len(t) > MIN_TEXT_LEN_PER_PAGE:
                text_pages.append(t)
            else:
                text_pages.append(None)
                scan_count += 1
    finally:
        doc.close()

    scan_ratio = scan_count / max(total_pages, 1)
    logger.info("PDF %s: %d trang | %d scan (%.0f%%)",
                pdf_path.name, total_pages, scan_count, scan_ratio * 100)

    # Phần lớn là scan → dùng OCR pipeline
    if enable_ocr and scan_ratio > OCR_SCAN_RATIO_THRESHOLD:
        llm_ocr = _load_llm_ocr()
        if llm_ocr is not None:
            logger.info("OCR pipeline: %s", ocr_model)
            try:
                return llm_ocr.extract_with_llm_fallback(
                    pdf_path, model=ocr_model, verbose=True
                )
            except Exception as e:
                logger.warning("OCR pipeline lỗi: %s — dùng text layer có sẵn", e)

    # PDF có text layer → ghép lại
    parts: list[str] = []
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
    """Normalize và loại bỏ header/footer lặp từ text extract."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\u00a0", " ").replace("–", "-")

    # Remove dòng ngắn lặp lại nhiều lần (thường là header/footer)
    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < MAX_HEADER_FOOTER_LEN:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    repeated = {s for s, c in line_counts.items() if c > MIN_HEADER_FOOTER_REPEAT}
    if repeated:
        lines = [line for line in lines if line.strip() not in repeated]
        text = "\n".join(lines)

    return text.strip()


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

def _is_ctdt_related(link_text: str) -> bool:
    """Trang có nằm trong scope CTĐT không (dựa trên anchor text)."""
    keywords = ("chương trình", "ctđt", "khóa", "đào tạo", "curriculum", "đttx")
    return any(kw in link_text.lower() for kw in keywords)


def crawl_all_pages(year_filter: list[str] | None = None) -> list[dict]:
    """Crawl tất cả trang CTĐT, thu thập link PDF."""
    all_pdfs: list[dict] = []
    visited_tabids: set[int] = set()
    seen_urls: set[str] = set()
    queue: list[tuple[int, tuple[str, str]]] = list(CTDT_PAGES.items())

    logger.info("=" * 70)
    logger.info("CRAWL PDF CTĐT - FIT HCMUS | %d trang đầu vào", len(queue))
    if year_filter:
        logger.info("Lọc theo năm: %s", ", ".join(year_filter))
    logger.info("=" * 70)

    while queue:
        tabid, (he_dao_tao, desc) = queue.pop(0)
        if tabid in visited_tabids:
            continue
        visited_tabids.add(tabid)

        page_url = f"{BASE_URL}/vn/Default.aspx?tabid={tabid}"
        logger.info("[PAGE] tabid=%s | %s | %s", tabid, he_dao_tao, desc)

        soup = fetch_page(page_url)
        if soup is None:
            continue

        pdf_links = extract_pdf_links(soup, page_url)
        logger.info("  -> %d PDF links", len(pdf_links))

        for pdf_info in pdf_links:
            url = pdf_info["url"]
            title = pdf_info["title"]
            if url in seen_urls:
                continue
            seen_urls.add(url)

            year = detect_year(title, desc)
            if year_filter and year not in year_filter and "unknown" not in year_filter:
                continue

            pdf_info.update({
                "he_dao_tao": he_dao_tao,
                "page_desc": desc,
                "filename": build_filename(he_dao_tao, title, desc),
                "year": year,
                "nganh": detect_nganh(title),
            })
            all_pdfs.append(pdf_info)
            logger.info("    [FOUND] %s  <-  %s", pdf_info["filename"], title[:80])

        # Discover sub-pages mới
        queued_tabids = {t for t, _ in queue}
        for tid in discover_subpage_tabids(soup, page_url):
            if tid in visited_tabids or tid in queued_tabids or tid in CTDT_PAGES:
                continue
            # Chỉ nhận nếu link text liên quan CTĐT
            for a_tag in soup.find_all("a", href=True):
                if f"tabid={tid}" not in a_tag["href"]:
                    continue
                link_text = a_tag.get_text(strip=True)
                if _is_ctdt_related(link_text):
                    queue.append((tid, (he_dao_tao, f"Auto: {link_text[:50]}")))
                    break

        time.sleep(REQUEST_DELAY)

    return all_pdfs


def deduplicate_filenames(pdfs: list[dict]) -> list[dict]:
    """Đảm bảo filename unique bằng suffix --N."""
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


def _write_output(pdf: dict, cleaned: str, txt_path: Path) -> None:
    header = (
        f"# Tài liệu: {pdf['title']}\n"
        f"# Hệ đào tạo: {pdf['he_dao_tao']}\n"
        f"# Chuyên ngành: {pdf['nganh']}\n"
        f"# Năm: {pdf['year']}\n"
        f"# Nguồn: {pdf['url']}\n"
        f"# ---\n\n"
    )
    txt_path.write_text(header + cleaned, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Log level DEBUG")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Crawl all pages
    pdfs = deduplicate_filenames(crawl_all_pages(year_filter=args.years))

    logger.info("=" * 70)
    logger.info("TỔNG KẾT CRAWL | Tổng PDF: %d", len(pdfs))
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("[DRY RUN] Danh sách PDF sẽ tải:")
        for i, pdf in enumerate(pdfs, 1):
            logger.info(
                "  %3d. %s | hệ=%s ngành=%s năm=%s",
                i, pdf["filename"], pdf["he_dao_tao"], pdf["nganh"], pdf["year"],
            )
            logger.debug("       URL: %s", pdf["url"])
        return

    # Step 2: Download + Extract
    success = skip = error = 0
    for i, pdf in enumerate(pdfs, 1):
        txt_path = output_dir / pdf["filename"]
        if txt_path.exists():
            logger.info("[%d/%d] SKIP (đã tồn tại): %s", i, len(pdfs), pdf["filename"])
            skip += 1
            continue

        logger.info("[%d/%d] Downloading: %s", i, len(pdfs), pdf["title"][:60])

        pdf_filename = hashlib.md5(pdf["url"].encode()).hexdigest() + ".pdf"
        pdf_path = TEMP_DIR / pdf_filename
        if not pdf_path.exists():
            if not download_pdf(pdf["url"], pdf_path):
                error += 1
                continue
            time.sleep(REQUEST_DELAY)

        logger.info("  Extracting text (ocr-model=%s)...", args.ocr_model)
        raw_text = extract_text_from_pdf(
            pdf_path,
            enable_ocr=not args.skip_ocr,
            ocr_model=args.ocr_model,
        )
        if not raw_text or len(raw_text.strip()) < MIN_EXTRACTED_LEN:
            logger.warning("  Extracted text too short (%d chars)", len(raw_text))
            error += 1
            continue

        _write_output(pdf, clean_text(raw_text), txt_path)
        logger.info("  [OK] Saved: %s (%d chars)", pdf["filename"], len(raw_text))
        success += 1

    logger.info("=" * 70)
    logger.info("KẾT QUẢ | OK=%d | SKIP=%d | ERR=%d", success, skip, error)
    logger.info("Output: %s", output_dir)
    logger.info("Temp PDF: %s (xoá thủ công nếu muốn)", TEMP_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
