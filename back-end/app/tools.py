"""LangChain tools for the ReAct RAG agent."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool


def _format_docs_for_agent(documents: list[Any], top_k: int = 5) -> str:
    """Format retrieved documents into a readable string for the agent."""
    if not documents:
        return "Không tìm thấy tài liệu nào phù hợp."

    blocks: list[str] = []
    for idx, doc in enumerate(documents[:top_k], start=1):
        content = getattr(doc, "page_content", str(doc))
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("url") or "unknown"
        nganh = metadata.get("nganh", "")
        doc_type = metadata.get("loai_van_ban", "")

        header_parts = [f"[Doc {idx}]"]
        if nganh:
            header_parts.append(f"Ngành: {nganh}")
        if doc_type:
            header_parts.append(f"Loại: {doc_type}")
        header_parts.append(f"Nguồn: {source}")

        blocks.append(f"{' | '.join(header_parts)}\n{content}")

    return "\n\n---\n\n".join(blocks)


def _token_overlap_score(query: str, content: str) -> float:
    q_tokens = {t for t in re.split(r"\W+", query.lower()) if len(t) > 1}
    if not q_tokens:
        return 0.0
    c_tokens = set(re.split(r"\W+", content.lower()))
    return len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))


def create_qdrant_search_tool(
    retriever_fn: Any,
    rerank_fn: Any,
    rerank_top_k: int = 5,
):
    """Create a tool that searches the Qdrant vector database."""

    @tool
    def qdrant_search(query: str) -> str:
        """Tìm kiếm trong cơ sở dữ liệu chương trình đào tạo, quy chế, đề cương môn học của Khoa CNTT (FIT) trường HCMUS.
        Dùng tool này khi cần tra cứu thông tin về: chương trình đào tạo các ngành, quy chế đào tạo,
        điều kiện tốt nghiệp, đề cương môn học, tín chỉ, học phần, ngoại ngữ, liên thông đại học - thạc sĩ.
        Input là câu hỏi hoặc từ khóa tìm kiếm bằng tiếng Việt."""
        try:
            docs = retriever_fn(source="qdrant", query=query)
            if not docs:
                return "Không tìm thấy tài liệu nào trong cơ sở dữ liệu."

            reranked_pairs = rerank_fn(query=query, documents=docs, top_k=rerank_top_k)
            reranked_docs = [doc for doc, _ in reranked_pairs]
            return _format_docs_for_agent(reranked_docs, top_k=rerank_top_k)
        except Exception as exc:
            return f"Lỗi khi tìm kiếm: {exc}"

    return qdrant_search


def create_fit_website_tool(
    official_site_allowlist: str,
):
    """Create a tool that searches the FIT HCMUS website."""

    @tool
    def fit_website_search(query: str) -> str:
        """Tìm kiếm thông tin trên website chính thức của Khoa CNTT (FIT) HCMUS.
        Dùng tool này khi cần thông tin mới nhất, thông báo, tin tức, lịch tuyển sinh,
        sự kiện, hoặc thông tin không có trong cơ sở dữ liệu nội bộ.
        Input là câu hỏi hoặc từ khóa tìm kiếm bằng tiếng Việt."""
        allowlist = [h.strip() for h in official_site_allowlist.split(",") if h.strip()]
        if not allowlist:
            return "Không có website nào được cấu hình."

        try:
            import httpx
        except ImportError:
            return "Thiếu thư viện httpx để crawl website."

        results: list[str] = []
        for host in allowlist[:2]:
            url = f"https://{host}"
            try:
                response = httpx.get(url, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                html = response.text
            except Exception:
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
            score = _token_overlap_score(query, clipped)
            results.append(f"[Website: {url} | Relevance: {score:.2f}]\n{clipped}")

        if not results:
            return "Không thể truy cập website FIT HCMUS."

        return "\n\n---\n\n".join(results)

    return fit_website_search
