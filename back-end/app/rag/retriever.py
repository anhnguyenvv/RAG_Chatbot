# pyright: reportMissingImports=false
"""Document retrieval from Qdrant vector store."""

from __future__ import annotations

import logging
import threading
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Default single-flight coalescing timeout in seconds — guards against waiter
# deadlock if the winner hangs indefinitely.
_COALESCE_WAIT_TIMEOUT = 30.0


# Optional Prometheus metrics — degrade gracefully if prometheus_client is missing.
try:
    from prometheus_client import Counter

    _CACHE_HITS = Counter(
        "retriever_cache_hits_total",
        "Number of retrieval query cache hits.",
        ["source"],
    )
    _CACHE_MISSES = Counter(
        "retriever_cache_misses_total",
        "Number of retrieval query cache misses.",
        ["source"],
    )
    _CACHE_COALESCED = Counter(
        "retriever_cache_coalesced_total",
        "Number of cache misses coalesced into an in-flight request.",
        ["source"],
    )
except Exception:  # pragma: no cover - metrics are best-effort
    _CACHE_HITS = None
    _CACHE_MISSES = None
    _CACHE_COALESCED = None


class RetrieverManager:
    """Manages Qdrant retrievers and vector stores with caching."""

    def __init__(self, pipeline_config: Any, backend_config: Any, embeddings: Any) -> None:
        self.pipeline_config = pipeline_config
        self.backend_config = backend_config
        self.embeddings = embeddings
        self._retriever_cache: dict[str, Any] = {}
        self._vectorstore_cache: dict[str, Any] = {}

        # `getattr` fallbacks keep backward compatibility with older BackendConfig
        # instances (e.g. test fixtures built before these fields existed).
        cache_enabled = getattr(backend_config, "retrieval_cache_enabled", True)
        ttl = getattr(backend_config, "retrieval_cache_ttl", 600)
        maxsize = getattr(backend_config, "retrieval_cache_maxsize", 500)
        # TTLCache raises on ttl<=0 or maxsize<=0 — treat either as "disabled".
        self._query_cache: TTLCache | None = (
            TTLCache(maxsize=maxsize, ttl=ttl)
            if cache_enabled and maxsize > 0 and ttl > 0
            else None
        )
        self._query_cache_lock = threading.Lock()
        # Single-flight coalescing: keyed by cache_key, value is an Event the
        # winner sets when compute finishes (success or failure).
        self._inflight: dict[tuple, threading.Event] = {}

    @staticmethod
    def _build_cache_key(
        source: str,
        query: str,
        metadata_filter: dict[str, str] | None,
    ) -> tuple:
        # Empty-valued filter entries are dropped: they match the no-filter path
        # in `_retrieve_with_filter` (line where conditions end up empty) and
        # should therefore share the same cache entry.
        filter_key: tuple = ()
        if metadata_filter:
            filter_key = tuple(sorted((k, v) for k, v in metadata_filter.items() if v))
        return (source, query.strip(), filter_key)

    def _normalize_source(self, source: str) -> str:
        if source in {"fit_web", "auto"}:
            return "qdrant"
        return source

    def _create_vector_store(self, source: str):
        """Create a QdrantVectorStore instance for the given source."""
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        source = self._normalize_source(source)

        if not self.pipeline_config.qdrant_url or not self.pipeline_config.qdrant_api_key:
            raise ValueError("Missing Qdrant config: QDRANT_URL and QDRANT_API_KEY")

        client = QdrantClient(
            url=self.pipeline_config.qdrant_url,
            api_key=self.pipeline_config.qdrant_api_key,
            prefer_grpc=False,
        )

        db = QdrantVectorStore(
            client=client,
            embedding=self.embeddings,
            collection_name=self.pipeline_config.collection_name,
        )
        logger.info(
            "Created vector store for source=%s collection=%s",
            source, self.pipeline_config.collection_name,
        )
        return db

    def get_vector_store(self, source: str):
        """Get QdrantVectorStore instance (cached)."""
        key = self._normalize_source(source)
        cached = self._vectorstore_cache.get(key)
        if cached is not None:
            return cached

        db = self._create_vector_store(source)
        self._vectorstore_cache[key] = db
        return db

    def get_retriever(self, source: str):
        """Get retriever (cached). Used for queries without metadata filter."""
        key = self._normalize_source(source)
        cached = self._retriever_cache.get(key)
        if cached is not None:
            return cached

        db = self.get_vector_store(source)
        retriever = db.as_retriever(search_kwargs={"k": self.backend_config.retrieval_top_k})
        self._retriever_cache[key] = retriever
        return retriever

    def retrieve(
        self,
        source: str,
        query: str,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[Any]:
        """Retrieve documents for query, optionally filtered by metadata.

        Args:
            source: Data source name (e.g. "qdrant").
            query: Search query text.
            metadata_filter: Optional dict of metadata fields to filter on.
                Supported keys: ``nganh``, ``he_dao_tao``, ``khoa``, etc.
                Values must match exactly in Qdrant payload.
        """
        normalized = self._normalize_source(source)
        cache_key = None
        is_winner = False
        waiter_event: threading.Event | None = None

        if self._query_cache is not None and query and query.strip():
            cache_key = self._build_cache_key(normalized, query, metadata_filter)
            with self._query_cache_lock:
                cached = self._query_cache.get(cache_key)
                if cached is not None:
                    if _CACHE_HITS is not None:
                        _CACHE_HITS.labels(source=normalized).inc()
                    logger.debug(
                        "Cache hit for source=%s docs=%d", normalized, len(cached)
                    )
                    # Return a shallow copy so downstream in-place mutations
                    # (sort, pop, etc.) cannot poison the cached entry.
                    return list(cached)
                if _CACHE_MISSES is not None:
                    _CACHE_MISSES.labels(source=normalized).inc()

                # Single-flight coalescing: if another thread is already
                # computing this key, wait for it instead of issuing a
                # duplicate upstream call.
                existing = self._inflight.get(cache_key)
                if existing is not None:
                    waiter_event = existing
                else:
                    waiter_event = threading.Event()
                    self._inflight[cache_key] = waiter_event
                    is_winner = True

        # Losers wait for the winner, then try the cache again.
        if waiter_event is not None and not is_winner:
            if _CACHE_COALESCED is not None:
                _CACHE_COALESCED.labels(source=normalized).inc()
            waiter_event.wait(timeout=_COALESCE_WAIT_TIMEOUT)
            if cache_key is not None:
                with self._query_cache_lock:
                    cached = self._query_cache.get(cache_key) if self._query_cache is not None else None
                if cached is not None:
                    return list(cached)
            logger.debug(
                "Coalesce waiter timed out or winner failed; falling through "
                "for source=%s", normalized,
            )

        try:
            if metadata_filter:
                docs = self._retrieve_with_filter(source, query, metadata_filter)
            else:
                retriever = self.get_retriever(source)
                if hasattr(retriever, "invoke"):
                    docs = retriever.invoke(query)
                else:
                    docs = retriever.get_relevant_documents(query)
                logger.debug(
                    "Retrieved %d docs for source=%s (no filter)", len(docs), source,
                )

            if cache_key is not None and self._query_cache is not None:
                # Store a shallow copy to decouple the cached list from
                # the caller's reference.
                with self._query_cache_lock:
                    self._query_cache[cache_key] = list(docs)
            return docs
        finally:
            # Winner always signals waiters (success OR failure) and cleans up.
            if is_winner and cache_key is not None:
                with self._query_cache_lock:
                    self._inflight.pop(cache_key, None)
                assert waiter_event is not None
                waiter_event.set()

    def clear_query_cache(self) -> int:
        """Clear the retrieval query cache (useful after reindexing).

        Returns the number of entries evicted (0 if cache is disabled).
        """
        if self._query_cache is None:
            return 0
        with self._query_cache_lock:
            size = len(self._query_cache)
            self._query_cache.clear()
        logger.info("Query cache cleared, evicted %d entries", size)
        return size

    def cache_stats(self) -> dict[str, Any]:
        """Return a snapshot of cache configuration and current size."""
        if self._query_cache is None:
            return {"enabled": False, "size": 0, "maxsize": 0, "ttl": 0}
        with self._query_cache_lock:
            size = len(self._query_cache)
        return {
            "enabled": True,
            "size": size,
            "maxsize": self._query_cache.maxsize,
            "ttl": self._query_cache.ttl,
        }

    def _retrieve_with_filter(
        self,
        source: str,
        query: str,
        metadata_filter: dict[str, str],
    ) -> list[Any]:
        """Retrieve documents using Qdrant metadata filter."""
        from qdrant_client import models

        conditions = []
        for key, value in metadata_filter.items():
            if value:
                conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value),
                    )
                )

        if not conditions:
            return self.retrieve(source=source, query=query)

        qdrant_filter = models.Filter(must=conditions)
        db = self.get_vector_store(source)

        docs = db.similarity_search(
            query=query,
            k=self.backend_config.retrieval_top_k,
            filter=qdrant_filter,
        )
        logger.debug(
            "Retrieved %d docs for source=%s with filter=%s",
            len(docs), source, metadata_filter,
        )
        return docs
