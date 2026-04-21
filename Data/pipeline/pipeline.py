import logging
from pathlib import Path

from .config import PipelineConfig
from .embeddings import create_embeddings
from .loaders import load_txt_documents
from .splitters import chunk_documents
from .vector_store import similarity_search, upsert_documents_qdrant


logger = logging.getLogger(__name__)


def build_vector_index(config: PipelineConfig):
    logger.info("Starting vector index build")
    config.validate()
    logger.debug(
        "Pipeline config validated",
        extra={
            "source_dir": config.source_dir,
            "collection_name": config.collection_name,
            "chunk_strategy": config.chunk_strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": config.embedding_model_name,
        },
    )

    logger.info("Loading raw documents")
    raw_documents = []

    if not Path(config.source_dir).exists():
        raise FileNotFoundError(f"Source directory not found: {config.source_dir}")

    local_documents = load_txt_documents(config.source_dir)
    raw_documents.extend(local_documents)
    logger.info("Loaded local txt documents", extra={"local_document_count": len(local_documents)})

    if not raw_documents:
        raise ValueError("No documents were loaded from local files")

    logger.info("Loaded raw documents", extra={"raw_document_count": len(raw_documents)})

    logger.info("Chunking documents")
    chunked_documents = chunk_documents(
        raw_documents,
        strategy=config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    logger.info("Chunking completed", extra={"chunked_document_count": len(chunked_documents)})

    logger.info("Creating embeddings client")
    embeddings = create_embeddings(
        model_name=config.embedding_model_name,
        api_key=config.huggingface_api_key,
    )

    logger.info("Indexing chunks to vector store")
    vector_store = upsert_documents_qdrant(
        documents=chunked_documents,
        embeddings=embeddings,
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
        collection_name=config.collection_name,
        force_recreate=config.qdrant_force_recreate,
    )

    logger.info("Vector index build finished", extra={"collection_name": config.collection_name})

    return {
        "raw_documents": raw_documents,
        "chunked_documents": chunked_documents,
        "vector_store": vector_store,
    }


def run_smoke_search(vector_store, query: str, top_k: int = 5):
    logger.info("Running smoke search", extra={"top_k": top_k, "query_preview": query[:120]})
    return similarity_search(vector_store, query=query, top_k=top_k)
