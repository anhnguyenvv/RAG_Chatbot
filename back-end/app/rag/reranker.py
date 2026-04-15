"""Document reranking using cosine similarity."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def token_overlap_score(query: str, content: str) -> float:
    """Calculate token overlap between query and content."""
    q_tokens = {t for t in re.split(r"\W+", query.lower()) if len(t) > 1}
    if not q_tokens:
        return 0.0
    c_tokens = set(re.split(r"\W+", content.lower()))
    return len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """Reranks documents using embedding-based cosine similarity."""

    def __init__(self, embeddings: Any, enable_reranker: bool = True) -> None:
        self.embeddings = embeddings
        self.enable_reranker = enable_reranker

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int = 5,
    ) -> list[tuple[Any, float | None]]:
        """Rerank documents by cosine similarity to query.

        Returns list of (document, score) tuples.
        """
        if not documents:
            return []

        top_n = max(1, min(top_k, len(documents)))

        if not self.enable_reranker:
            logger.debug("Reranker disabled, returning top %d docs", top_n)
            return [(doc, None) for doc in documents[:top_n]]

        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents(
                [doc.page_content for doc in documents]
            )
            scored = []
            for doc, emb in zip(documents, doc_embeddings):
                score = cosine_similarity(query_embedding, emb)
                scored.append((doc, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            logger.debug(
                "Reranked %d -> %d docs, top score=%.4f",
                len(documents), top_n,
                scored[0][1] if scored else 0.0,
            )
            return scored[:top_n]
        except Exception:
            logger.warning("Reranking failed, returning unranked docs", exc_info=True)
            return [(doc, None) for doc in documents[:top_n]]
