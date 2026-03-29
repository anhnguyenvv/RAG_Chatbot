import logging
from collections import deque
from html import unescape
from typing import Iterable, List
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    cleaned, _ = urldefrag(url.strip())
    return cleaned.rstrip("/")


def _is_allowed_url(url: str, allowed_domains: set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    hostname = parsed.hostname or ""
    return hostname in allowed_domains


def _extract_main_text(html: str) -> tuple[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    text = soup.get_text(separator="\n")
    lines = [unescape(line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    compact_text = "\n".join(lines)
    return compact_text, title


def crawl_website_documents(
    start_urls: Iterable[str],
    allowed_domains: Iterable[str],
    max_pages: int = 30,
    timeout_seconds: int = 15,
) -> List[Document]:
    normalized_domains = {domain.strip().lower() for domain in allowed_domains if domain.strip()}
    if not normalized_domains:
        raise ValueError("allowed_domains must not be empty")
    if max_pages <= 0:
        raise ValueError("max_pages must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    queue: deque[str] = deque()
    visited: set[str] = set()

    for start_url in start_urls:
        normalized = _normalize_url(start_url)
        if normalized and _is_allowed_url(normalized, normalized_domains):
            queue.append(normalized)

    if not queue:
        raise ValueError("No valid start_urls after filtering by allowed_domains")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "RAG-Chatbot-DataPipeline/1.0 (+https://fit.hcmus.edu.vn)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    documents: List[Document] = []

    logger.info(
        "Starting website crawl",
        extra={
            "seed_count": len(queue),
            "allowed_domains": sorted(normalized_domains),
            "max_pages": max_pages,
        },
    )

    while queue and len(visited) < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = session.get(current_url, timeout=timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Skipping URL due to request error", extra={"url": current_url, "error": str(exc)})
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            logger.debug("Skipping non-HTML content", extra={"url": current_url, "content_type": content_type})
            continue

        text, title = _extract_main_text(response.text)
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": current_url,
                        "title": title,
                        "nganh": None,
                        "hoc_ky": None,
                        "loai_van_ban": "web",
                        "nam_ban_hanh": None,
                        "dieu_khoan": None,
                    },
                )
            )

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            candidate = _normalize_url(urljoin(current_url, link["href"]))
            if candidate in visited:
                continue
            if _is_allowed_url(candidate, normalized_domains):
                queue.append(candidate)

    logger.info(
        "Website crawl completed",
        extra={
            "visited_pages": len(visited),
            "web_documents": len(documents),
        },
    )

    return documents