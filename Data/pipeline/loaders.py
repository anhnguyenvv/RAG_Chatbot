import logging
from pathlib import Path
import re
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy mappings (kept for backward compatibility)
# ---------------------------------------------------------------------------

PROGRAM_BY_SLUG = {
    "cong_nghe_thong_tin": "Công nghệ thông tin",
    "he_thong_thong_tin": "Hệ thống thông tin",
    "khoa_hoc_may_tinh": "Khoa học máy tính",
    "ky_thuat_phan_mem": "Kỹ thuật phần mềm",
    "tri_tue_nhan_tao": "Trí tuệ nhân tạo",
    "toan_truong": "Toàn trường",
    "da_nganh": "Đa ngành",
}

DOC_TYPE_BY_PREFIX = {
    "chuong_trinh_dao_tao": "chương trình đào tạo",
    "quy_dinh_dao_tao": "quy định",
    "quy_dinh_ngoai_ngu": "quy định",
    "de_cuong_mon_hoc": "đề cương môn học",
    "dieu_kien_tot_nghiep": "tốt nghiệp",
    "lien_thong_dai_hoc_thac_si": "liên thông",
}


# ---------------------------------------------------------------------------
# Crawl-format mappings  (crawl_fit_pdfs naming convention)
# {he_dao_tao}__{nganh}__{year}__{doc_type}.txt
# ---------------------------------------------------------------------------

_NGANH_SLUG_TO_NAME: dict[str, str] = {
    "cntt": "Công nghệ thông tin",
    "httt": "Hệ thống thông tin",
    "khmt": "Khoa học máy tính",
    "ktpm": "Kỹ thuật phần mềm",
    "ttnt": "Trí tuệ nhân tạo",
    "cu-nhan-tai-nang": "Cử nhân tài năng",
    "chung": "Chung",
}

_DOC_TYPE_SLUG_TO_LOAI: dict[str, str] = {
    "chuong-trinh-dao-tao": "chương trình đào tạo",
    "quyet-dinh-ban-hanh": "quyết định ban hành",
    "quyet-dinh-ban-hanh-ctdt": "quyết định ban hành",
    "bang-chuyen-doi-hoc-phan": "bảng chuyển đổi học phần",
    "danh-sach-hoc-phan": "danh sách học phần",
    "dieu-chinh": "điều chỉnh",
    "bo-sung": "bổ sung",
    "lien-thong": "liên thông",
    "cong-van-chuyen-doi": "công văn chuyển đổi",
}


# ---------------------------------------------------------------------------
# Crawl-format parsers
# ---------------------------------------------------------------------------

def _parse_crawl_header(content: str) -> tuple[dict, str]:
    """Parse metadata header block produced by crawl_fit_pdfs.py.

    The header looks like:
        # Tài liệu: ...
        # Hệ đào tạo: ...
        # Chuyên ngành: ...
        # Năm: ...
        # Nguồn: ...
        # ---
        <body text>

    Returns:
        (header_dict, body_text) — header_dict is empty if no header found.
    """
    lines = content.split("\n")

    # Quick check: must start with "# "
    if not lines or not lines[0].startswith("# "):
        return {}, content

    header: dict[str, str] = {}
    body_start = len(lines)  # fallback: no separator found

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "# ---":
            body_start = i + 1
            break
        if stripped.startswith("# "):
            kv = stripped[2:].split(":", 1)
            if len(kv) == 2:
                header[kv[0].strip()] = kv[1].strip()

    body = "\n".join(lines[body_start:]).lstrip("\n").strip()
    return header, body


def _parse_crawl_filename(filename: str) -> dict:
    """Parse crawl_fit_pdfs naming convention.

    Pattern: {he_dao_tao}__{nganh}__{year}__{doc_type}.txt
    Example: chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt

    Returns empty dict if pattern doesn't match.
    """
    name = filename.rsplit(".", 1)[0]
    parts = name.split("__")
    if len(parts) < 4:
        return {}
    return {
        "he_dao_tao": parts[0],
        "nganh_slug": parts[1],
        "year": parts[2],
        "doc_type_slug": "__".join(parts[3:]),  # join remainder in case of extra __
    }


# ---------------------------------------------------------------------------
# Legacy helpers (kept unchanged for backward compat)
# ---------------------------------------------------------------------------

def _infer_document_type(file_name: str, content: str) -> str:
    lowered = file_name.lower()

    for prefix, doc_type in DOC_TYPE_BY_PREFIX.items():
        if lowered.startswith(prefix):
            return doc_type

    if lowered in {"quyche.txt", "nn.txt"}:
        return "quy định"
    if lowered == "monhoc.txt":
        return "đề cương môn học"
    if lowered == "dktn.txt":
        return "tốt nghiệp"
    if lowered == "bsms.txt":
        return "liên thông"
    if lowered in {"cntt.txt", "httt.txt", "khmt.txt", "ktpm.txt", "ttnt.txt"}:
        return "chương trình đào tạo"

    lowered_content = content.lower()
    if any(token in lowered_content for token in ("quy dinh", "quy định", "quy che", "quy chế")):
        return "quy định"
    return "tài liệu khác"


def _extract_program_name(file_name: str, content: str) -> str | None:
    lowered = file_name.lower()

    slug_match = re.search(r"nganh-([a-z0-9_]+)", lowered)
    if slug_match:
        slug = slug_match.group(1)
        if slug in PROGRAM_BY_SLUG:
            return PROGRAM_BY_SLUG[slug]
        return slug.replace("_", " ").strip().title()

    legacy_map = {
        "cntt.txt": "Công nghệ thông tin",
        "httt.txt": "Hệ thống thông tin",
        "khmt.txt": "Khoa học máy tính",
        "ktpm.txt": "Kỹ thuật phần mềm",
        "ttnt.txt": "Trí tuệ nhân tạo",
    }
    if lowered in legacy_map:
        return legacy_map[lowered]

    match = re.search(r"(?:Nganh|Ngành)\s+([^\n\r]+)", content, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip(" .:-")
        if value:
            return value
    return None


def _extract_issue_year(content: str) -> int | None:
    date_match = re.search(r"(?:ngay|ngày)\s+\d{1,2}\s*/\s*\d{1,2}\s*/\s*(\d{4})", content, flags=re.IGNORECASE)
    if date_match:
        return int(date_match.group(1))

    recruit_match = re.search(r"(?:Khoa\s+tuyen|Khóa\s+tuyển)\s*:\s*(\d{4})", content, flags=re.IGNORECASE)
    if recruit_match:
        return int(recruit_match.group(1))

    first_year = re.search(r"\b(19|20)\d{2}\b", content)
    if first_year:
        return int(first_year.group(0))
    return None


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def _build_base_metadata(
    txt_file: Path,
    content: str,
    header_meta: dict | None = None,
) -> dict:
    """Build document metadata.

    Supports two formats:
    1. crawl_fit_pdfs format — parses ``# Key: Value`` header + ``__``-separated filename.
    2. Legacy format  — falls back to existing inference heuristics.
    """
    filename = txt_file.name
    crawl_parts = _parse_crawl_filename(filename)

    # ---- Crawl format -------------------------------------------------------
    if crawl_parts or header_meta:
        hm = header_meta or {}

        # nganh: header "# Chuyên ngành:" → human readable name
        nganh_slug = hm.get("Chuyên ngành") or crawl_parts.get("nganh_slug", "")
        nganh = _NGANH_SLUG_TO_NAME.get(nganh_slug, nganh_slug or None)

        # loai_van_ban: filename doc_type slug → human readable
        doc_type_slug = crawl_parts.get("doc_type_slug", "")
        loai_van_ban = _DOC_TYPE_SLUG_TO_LOAI.get(
            doc_type_slug, doc_type_slug.replace("-", " ") or "tài liệu khác"
        )
        if not loai_van_ban:
            loai_van_ban = "tài liệu khác"

        # nam_ban_hanh: header "# Năm:" takes priority
        year_str = hm.get("Năm") or crawl_parts.get("year", "")
        try:
            nam_ban_hanh = int(year_str) if year_str and year_str != "unknown" else None
        except (ValueError, TypeError):
            nam_ban_hanh = None

        # source: URL from header, fallback to filename
        source = hm.get("Nguồn") or filename

        return {
            "source": source,
            "file_name": filename,
            "tai_lieu": hm.get("Tài liệu"),
            "he_dao_tao": hm.get("Hệ đào tạo") or crawl_parts.get("he_dao_tao"),
            "nganh": nganh,
            "loai_van_ban": loai_van_ban,
            "nam_ban_hanh": nam_ban_hanh,
            "hoc_ky": None,
            "dieu_khoan": None,
        }

    # ---- Legacy format ------------------------------------------------------
    return {
        "source": filename,
        "file_name": filename,
        "tai_lieu": None,
        "he_dao_tao": None,
        "nganh": _extract_program_name(filename, content),
        "hoc_ky": None,
        "loai_van_ban": _infer_document_type(filename, content),
        "nam_ban_hanh": _extract_issue_year(content),
        "dieu_khoan": None,
    }


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_file_documents(txt_file: Path) -> List[Document]:
    encodings = ["utf-8", "utf-8-sig"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            loader = TextLoader(str(txt_file), encoding=encoding)
            return loader.load()
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return []


def load_txt_documents(source_dir: str) -> List[Document]:
    data_path = Path(source_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    logger.info("Scanning source directory for txt files", extra={"source_dir": source_dir})
    documents: List[Document] = []
    txt_files = sorted(data_path.glob("*.txt"))
    logger.info("Found txt files", extra={"txt_file_count": len(txt_files)})

    for txt_file in txt_files:
        before_count = len(documents)
        for doc in _load_file_documents(txt_file):
            # Parse and strip crawl header block if present
            header_meta, body = _parse_crawl_header(doc.page_content)
            if header_meta:
                doc.page_content = body  # use body-only content for chunking

            base_metadata = _build_base_metadata(
                txt_file=txt_file,
                content=doc.page_content,
                header_meta=header_meta or None,
            )
            doc.metadata = {**doc.metadata, **base_metadata}
            documents.append(doc)

        logger.debug(
            "Loaded txt file",
            extra={
                "file_name": txt_file.name,
                "loaded_docs": len(documents) - before_count,
            },
        )

    if not documents:
        raise ValueError(f"No .txt files found in {source_dir}")

    logger.info("Finished loading documents", extra={"document_count": len(documents)})
    return documents
