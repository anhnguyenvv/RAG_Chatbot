"""Pipeline smoke tests — end-to-end load + chunk with real code (no external calls).

Vector-store and embeddings are mocked. Network-dependent steps
(Qdrant upsert, HuggingFace model download) are explicitly bypassed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.loaders import load_txt_documents
from pipeline.splitters import chunk_documents


SAMPLE_FILES: dict[str, str] = {
    "chinh-quy__cntt__2025__chuong-trinh-dao-tao.txt": (
        "# Tài liệu: CTDT CNTT 2025\n"
        "# Hệ đào tạo: chinh-quy\n"
        "# Chuyên ngành: cntt\n"
        "# Năm: 2025\n"
        "# Nguồn: https://fit.hcmus.edu.vn/ctdt-cntt.pdf\n"
        "# ---\n"
        "\n"
        "Điều 1. Quy định chung\n"
        + ("Đây là đoạn nội dung về học phần đại cương ngành Công nghệ thông tin. " * 20)
        + "\n\nĐiều 2. Yêu cầu tốt nghiệp\n"
        + ("Sinh viên phải hoàn thành ít nhất 140 tín chỉ theo chương trình. " * 20)
    ),
    "chinh-quy__khmt__2024__chuong-trinh-dao-tao.txt": (
        "# Tài liệu: CTDT KHMT 2024\n"
        "# Hệ đào tạo: chinh-quy\n"
        "# Chuyên ngành: khmt\n"
        "# Năm: 2024\n"
        "# Nguồn: https://fit.hcmus.edu.vn/ctdt-khmt.pdf\n"
        "# ---\n"
        "\n"
        "Chương I. Tổng quan chương trình\n"
        + ("Nội dung mô tả chương trình Khoa học máy tính. " * 30)
    ),
    "QuyChe.txt": (  # Legacy format
        "Quy chế đào tạo Trường Đại học Khoa học Tự nhiên. Ngày 24/09/2021.\n"
        + ("Điều khoản chi tiết quy định việc tổ chức đào tạo. " * 25)
    ),
}


@pytest.fixture
def sample_source_dir(tmp_path):
    for name, content in SAMPLE_FILES.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


class TestLoadAndChunk:
    def test_load_all_files(self, sample_source_dir):
        docs = load_txt_documents(str(sample_source_dir))
        assert len(docs) == 3

    def test_crawl_format_metadata_propagates(self, sample_source_dir):
        docs = load_txt_documents(str(sample_source_dir))
        cntt = next(d for d in docs if "cntt" in d.metadata["file_name"])
        assert cntt.metadata["nganh"] == "Công nghệ thông tin"
        assert cntt.metadata["nam_ban_hanh"] == 2025
        assert cntt.metadata["he_dao_tao"] == "chinh-quy"
        assert cntt.metadata["loai_van_ban"] == "chương trình đào tạo"
        # Header block must be stripped from page_content
        assert "# Tài liệu:" not in cntt.page_content

    def test_legacy_format_falls_back(self, sample_source_dir):
        docs = load_txt_documents(str(sample_source_dir))
        legacy = next(d for d in docs if d.metadata["file_name"] == "QuyChe.txt")
        assert legacy.metadata["loai_van_ban"] == "quy định"
        assert legacy.metadata["nam_ban_hanh"] == 2021

    def test_outline_chunking_preserves_metadata(self, sample_source_dir):
        docs = load_txt_documents(str(sample_source_dir))
        chunks = chunk_documents(docs, strategy="outline", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > len(docs)  # chunking should increase count
        # Every chunk must carry source + file_name metadata
        for ch in chunks:
            assert "source" in ch.metadata
            assert "file_name" in ch.metadata
            assert "nganh" in ch.metadata
            assert "loai_van_ban" in ch.metadata

    def test_outline_chunking_extracts_dieu_khoan(self, sample_source_dir):
        docs = load_txt_documents(str(sample_source_dir))
        chunks = chunk_documents(docs, strategy="outline", chunk_size=200, chunk_overlap=40)
        # At least one chunk from CTDT CNTT should pick up "Điều 1" or "Điều 2"
        cntt_chunks = [c for c in chunks if "cntt" in c.metadata["file_name"]]
        has_dieu_khoan = [c for c in cntt_chunks if c.metadata.get("dieu_khoan")]
        assert has_dieu_khoan, "Expected dieu_khoan metadata on at least one chunk"

    def test_recursive_chunking_also_works(self, sample_source_dir):
        docs = load_txt_documents(str(sample_source_dir))
        chunks = chunk_documents(docs, strategy="recursive", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > len(docs)


class TestPipelineWithMockedVectorStore:
    """Full pipeline with vector_store + embeddings mocked.

    Verifies that `build_vector_index` wires load → chunk → embed → upsert
    correctly and passes the right config through.
    """

    def test_build_vector_index_happy_path(self, sample_source_dir, monkeypatch):
        # Must mock the pipeline module (it imports upsert_documents_qdrant as a name)
        # Patch `upsert_documents_qdrant` inside the `pipeline.pipeline` namespace.
        from pipeline import pipeline as pipeline_mod

        fake_upsert = MagicMock(return_value="FAKE_VECTORSTORE")
        fake_embeddings = MagicMock()
        fake_create_embeddings = MagicMock(return_value=fake_embeddings)

        monkeypatch.setattr(pipeline_mod, "upsert_documents_qdrant", fake_upsert)
        monkeypatch.setattr(pipeline_mod, "create_embeddings", fake_create_embeddings)

        from pipeline.config import PipelineConfig
        cfg = PipelineConfig(
            source_dir=str(sample_source_dir),
            collection_name="test_coll",
            qdrant_url="http://test",
            qdrant_api_key="test-key",
            chunk_size=300,
            chunk_overlap=50,
            chunk_strategy="outline",
        )

        result = pipeline_mod.build_vector_index(cfg)

        # Verify pipeline produced reasonable output
        assert result["raw_documents"]
        assert result["chunked_documents"]
        assert len(result["chunked_documents"]) >= len(result["raw_documents"])
        assert result["vector_store"] == "FAKE_VECTORSTORE"

        # Verify upsert called with correct args
        fake_upsert.assert_called_once()
        call_kwargs = fake_upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_coll"
        assert call_kwargs["qdrant_url"] == "http://test"
        assert call_kwargs["documents"] == result["chunked_documents"]

    def test_missing_qdrant_config_raises(self, sample_source_dir):
        from pipeline import pipeline as pipeline_mod
        from pipeline.config import PipelineConfig

        cfg = PipelineConfig(source_dir=str(sample_source_dir))  # no qdrant creds
        with pytest.raises(ValueError, match="QDRANT"):
            pipeline_mod.build_vector_index(cfg)

    def test_missing_source_dir_raises(self, tmp_path, monkeypatch):
        from pipeline import pipeline as pipeline_mod
        from pipeline.config import PipelineConfig

        monkeypatch.setattr(pipeline_mod, "upsert_documents_qdrant", MagicMock())
        monkeypatch.setattr(pipeline_mod, "create_embeddings", MagicMock())

        cfg = PipelineConfig(
            source_dir=str(tmp_path / "nonexistent"),
            qdrant_url="http://test",
            qdrant_api_key="test-key",
        )
        with pytest.raises(FileNotFoundError):
            pipeline_mod.build_vector_index(cfg)
