import logging
import importlib
from typing import List

from langchain_core.documents import Document
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


def upsert_documents_qdrant(
    documents: List[Document],
    embeddings,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    force_recreate: bool = True,
):
    if not documents:
        raise ValueError("No documents to index")

    logger.info(
        "Upserting documents to Qdrant",
        extra={
            "collection_name": collection_name,
            "document_count": len(documents),
            "qdrant_url": qdrant_url,
            "force_recreate": force_recreate,
        },
    )

    qdrant_module = importlib.util.find_spec("langchain_qdrant")
    if qdrant_module is None:
        raise RuntimeError(
            "Missing dependency: langchain-qdrant. "
            "Install it with: pip install langchain-qdrant==0.1.4"
        )
    QdrantVectorStore = getattr(importlib.import_module("langchain_qdrant"), "QdrantVectorStore")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)

    if force_recreate:
        try:
            client.delete_collection(collection_name=collection_name)
            logger.info("Deleted existing collection before re-index", extra={"collection_name": collection_name})
        except Exception:
            logger.debug("Collection did not exist before re-index", extra={"collection_name": collection_name})

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        prefer_grpc=False,
    )
    logger.info("Finished upserting documents to Qdrant", extra={"collection_name": collection_name})
    return vector_store


def similarity_search(vector_store, query: str, top_k: int = 5):
    logger.debug("Running similarity search", extra={"top_k": top_k, "query_preview": query[:120]})
    docs = vector_store.similarity_search(query, k=top_k)
    logger.debug("Similarity search completed", extra={"result_count": len(docs)})
    return docs
