import logging
import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def _extract_hoc_ky(text: str) -> str | None:
    match = re.search(r"(?:Hoc\s*ky|Học\s*kỳ)\s*(\d+)\s*(?:nam|năm)\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return f"Học kỳ {match.group(1)} năm {match.group(2)}"


def _extract_dieu_khoan(text: str) -> str | None:
    match = re.search(r"(?:Dieu|Điều)\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return f"Điều {match.group(1)}"


def _merge_chunk_metadata(base_metadata: dict, chunk_text: str) -> dict:
    metadata = dict(base_metadata)
    chunk_hoc_ky = _extract_hoc_ky(chunk_text)
    chunk_dieu_khoan = _extract_dieu_khoan(chunk_text)

    if chunk_hoc_ky is not None:
        metadata["hoc_ky"] = chunk_hoc_ky
    else:
        metadata.setdefault("hoc_ky", None)

    if chunk_dieu_khoan is not None:
        metadata["dieu_khoan"] = chunk_dieu_khoan
    else:
        metadata.setdefault("dieu_khoan", None)

    metadata.setdefault("nganh", None)
    metadata.setdefault("loai_van_ban", "tài liệu khác")
    metadata.setdefault("nam_ban_hanh", None)
    metadata.setdefault("source", "unknown")
    return metadata


class OutlineAwareTextSplitter:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _is_heading(line: str) -> bool:
        return bool(re.match(r"^\d+(\.\d+)*\.", line.strip()))

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"(\n{2,})", "\n\n", text)

    @staticmethod
    def _nearest_split_point(text: str) -> int:
        split_point = max(text.rfind(" "), text.rfind("!"), text.rfind("?"))
        return split_point if split_point != -1 else len(text)

    def split_text(self, text: str) -> List[str]:
        text = self._normalize_whitespace(text)
        lines = text.splitlines()

        chunks: List[str] = []
        current_lines: List[str] = []
        current_len = 0
        last_heading = ""

        for line in lines:
            line_len = len(line)
            if current_len + line_len > self.chunk_size and current_lines:
                chunk = "\n".join(current_lines).strip()

                if len(chunk) > self.chunk_size:
                    split_point = self._nearest_split_point(chunk[: self.chunk_size])
                    chunks.append(chunk[: split_point + 1].strip())
                    overlap_text = chunk[split_point + 1 :]
                else:
                    chunks.append(chunk)
                    overlap_text = chunk[-self.chunk_overlap :]

                overlap_lines = [line for line in overlap_text.splitlines() if line.strip()]

                if last_heading and (not overlap_lines or not self._is_heading(overlap_lines[0])):
                    overlap_lines.insert(0, last_heading)

                current_lines = overlap_lines
                current_len = sum(len(item) + 1 for item in current_lines)

            current_lines.append(line)
            current_len += line_len + 1

            if self._is_heading(line):
                last_heading = line

        if current_lines:
            tail = "\n".join(current_lines).strip()
            if tail:
                chunks.append(tail)

        return chunks

    def create_documents(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        for doc in docs:
            base_metadata = dict(doc.metadata)
            for chunk in self.split_text(doc.page_content):
                metadata = _merge_chunk_metadata(base_metadata=base_metadata, chunk_text=chunk)
                out.append(Document(page_content=chunk, metadata=metadata))
        return out


def chunk_documents(
    documents: List[Document],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    logger.info(
        "Chunking documents",
        extra={
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "input_document_count": len(documents),
        },
    )

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunked_documents: List[Document] = []
        for original_doc in documents:
            base_metadata = dict(original_doc.metadata)
            chunks = splitter.split_text(original_doc.page_content)
            for chunk in chunks:
                metadata = _merge_chunk_metadata(base_metadata=base_metadata, chunk_text=chunk)
                chunked_documents.append(Document(page_content=chunk, metadata=metadata))
        logger.info("Chunking completed", extra={"output_chunk_count": len(chunked_documents)})
        return chunked_documents

    if strategy == "outline":
        splitter = OutlineAwareTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_documents = splitter.create_documents(documents)
        logger.info("Chunking completed", extra={"output_chunk_count": len(chunked_documents)})
        return chunked_documents

    raise ValueError("Unknown chunk strategy")
