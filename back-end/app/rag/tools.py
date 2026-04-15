"""LangChain tools for the ReAct RAG agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from app.rag.reranker import token_overlap_score

logger = logging.getLogger(__name__)


def _format_docs_for_agent(documents: list[Any], top_k: int = 5) -> str:
    """Format retrieved documents into a readable string for the agent."""
    if not documents:
        return "Khong tim thay tai lieu nao phu hop."

    blocks: list[str] = []
    for idx, doc in enumerate(documents[:top_k], start=1):
        content = getattr(doc, "page_content", str(doc))
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("url") or "unknown"
        nganh = metadata.get("nganh", "")
        doc_type = metadata.get("loai_van_ban", "")

        header_parts = [f"[Doc {idx}]"]
        if nganh:
            header_parts.append(f"Nganh: {nganh}")
        if doc_type:
            header_parts.append(f"Loai: {doc_type}")
        header_parts.append(f"Nguon: {source}")

        blocks.append(f"{' | '.join(header_parts)}\n{content}")

    return "\n\n---\n\n".join(blocks)


def create_qdrant_search_tool(
    retriever_fn: Any,
    rerank_fn: Any,
    rerank_top_k: int = 5,
):
    """Create a tool that searches the Qdrant vector database."""

    @tool
    def qdrant_search(
        query: str,
        nganh: str = "",
        khoa: str = "",
        he_dao_tao: str = "",
    ) -> str:
        """Tim kiem trong co so du lieu chuong trinh dao tao, quy che, de cuong mon hoc cua Khoa CNTT (FIT) truong HCMUS.
        Dung tool nay khi can tra cuu thong tin ve: chuong trinh dao tao cac nganh, quy che dao tao,
        dieu kien tot nghiep, de cuong mon hoc, tin chi, hoc phan, ngoai ngu, lien thong dai hoc - thac si.

        Args:
            query: Cau hoi hoac tu khoa tim kiem bang tieng Viet.
            nganh: Ten chuyen nganh de loc ket qua (VD: "Cong nghe thong tin", "Ky thuat phan mem"). De trong neu khong biet.
            khoa: Khoa tuyen sinh / nien khoa (VD: "K2023"). De trong neu khong biet.
            he_dao_tao: He dao tao (VD: "chinh quy", "tu xa"). De trong neu khong biet.
        """
        try:
            # Build metadata filter from provided context
            metadata_filter: dict[str, str] = {}
            if nganh:
                metadata_filter["nganh"] = nganh
            if he_dao_tao:
                metadata_filter["he_dao_tao"] = he_dao_tao
            # khoa is used for context but not as direct Qdrant filter
            # (data uses nam_ban_hanh / file-level metadata, not "khoa" field)

            docs = retriever_fn(
                source="qdrant",
                query=query,
                metadata_filter=metadata_filter or None,
            )
            if not docs:
                # Fallback: retry without filter if no results found
                if metadata_filter:
                    logger.info("qdrant_search: no results with filter %s, retrying without filter", metadata_filter)
                    docs = retriever_fn(source="qdrant", query=query)
                if not docs:
                    return "Khong tim thay tai lieu nao trong co so du lieu."

            reranked_pairs = rerank_fn(query=query, documents=docs, top_k=rerank_top_k)
            reranked_docs = [doc for doc, _ in reranked_pairs]
            logger.debug(
                "qdrant_search: %d docs retrieved, %d after rerank, filter=%s",
                len(docs), len(reranked_docs), metadata_filter or "none",
            )
            return _format_docs_for_agent(reranked_docs, top_k=rerank_top_k)
        except Exception as exc:
            logger.error("qdrant_search failed: %s", exc, exc_info=True)
            return f"Loi khi tim kiem: {exc}"

    return qdrant_search


def create_fit_website_tool(
    official_site_allowlist: str,
):
    """Create a tool that searches the FIT HCMUS website."""

    @tool
    def fit_website_search(query: str) -> str:
        """Tim kiem thong tin tren website chinh thuc cua Khoa CNTT (FIT) HCMUS.
        Dung tool nay khi can thong tin moi nhat, thong bao, tin tuc, lich tuyen sinh,
        su kien, hoac thong tin khong co trong co so du lieu noi bo.
        Input la cau hoi hoac tu khoa tim kiem bang tieng Viet."""
        allowlist = [h.strip() for h in official_site_allowlist.split(",") if h.strip()]
        if not allowlist:
            return "Khong co website nao duoc cau hinh."

        try:
            import httpx
        except ImportError:
            return "Thieu thu vien httpx de crawl website."

        results: list[str] = []
        for host in allowlist[:2]:
            url = f"https://{host}"
            try:
                response = httpx.get(url, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                html = response.text
            except Exception:
                logger.debug("Failed to fetch %s", url, exc_info=True)
                continue

            text = ""
            try:
                import trafilatura

                extracted = trafilatura.extract(html)
                if extracted:
                    text = extracted
            except Exception:
                pass

            if not text:
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(html, "html.parser")
                    text = " ".join(soup.get_text(" ").split())
                except Exception:
                    pass

            if not text:
                continue

            clipped = text[:3000]
            score = token_overlap_score(query, clipped)
            results.append(f"[Website: {url} | Relevance: {score:.2f}]\n{clipped}")

        if not results:
            return "Khong the truy cap website FIT HCMUS."

        return "\n\n---\n\n".join(results)

    return fit_website_search
