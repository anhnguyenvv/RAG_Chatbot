import logging
from pathlib import Path
import re
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


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


def _infer_document_type(file_name: str, content: str) -> str:
    lowered = file_name.lower()

    # New naming convention: <loai>_nganh-...__...
    for prefix, doc_type in DOC_TYPE_BY_PREFIX.items():
        if lowered.startswith(prefix):
            return doc_type

    # Legacy short file names (backward compatibility)
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

    # New naming convention: ..._nganh-<slug>__...
    slug_match = re.search(r"nganh-([a-z0-9_]+)", lowered)
    if slug_match:
        slug = slug_match.group(1)
        if slug in PROGRAM_BY_SLUG:
            return PROGRAM_BY_SLUG[slug]
        return slug.replace("_", " ").strip().title()

    # Legacy short file names (backward compatibility)
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


def _build_base_metadata(txt_file: Path, content: str) -> dict:
    return {
        "source": txt_file.name,
        "nganh": _extract_program_name(txt_file.name, content),
        "hoc_ky": None,
        "loai_van_ban": _infer_document_type(txt_file.name, content),
        "nam_ban_hanh": _extract_issue_year(content),
        "dieu_khoan": None,
    }


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
            base_metadata = _build_base_metadata(txt_file=txt_file, content=doc.page_content)
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
