# pyright: reportMissingImports=false
"""Document retrieval from Qdrant vector store."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RetrieverManager:
    """Manages Qdrant retrievers and vector stores with caching."""

    def __init__(self, pipeline_config: Any, backend_config: Any, embeddings: Any) -> None:
        self.pipeline_config = pipeline_config
        self.backend_config = backend_config
        self.embeddings = embeddings
        self._retriever_cache: dict[str, Any] = {}
        self._vectorstore_cache: dict[str, Any] = {}

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
        if metadata_filter:
            return self._retrieve_with_filter(source, query, metadata_filter)

        # No filter — use cached retriever
        retriever = self.get_retriever(source)
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)
        logger.debug("Retrieved %d docs for source=%s (no filter)", len(docs), source)
        return docs

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
