"""Dependency wiring — builds the RAG service with all its components."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import BackendConfig
from app.rag.agent import ReactRAGAgent
from app.rag.generator import get_prompt_template
from app.rag.llm import create_embeddings, create_llm
from app.rag.reranker import Reranker
from app.rag.retriever import RetrieverManager
from app.rag.tools import create_fit_website_tool, create_qdrant_search_tool
from app.services.rag_service import RAGService
from app.storage.memory import MongoSessionMemoryStore

logger = logging.getLogger(__name__)


def build_rag_service(backend_config: BackendConfig, pipeline_config: Any) -> RAGService:
    """Wire all RAG components and return the service."""

    # Models
    embeddings = create_embeddings(pipeline_config)
    llm = create_llm(backend_config.generate_model_name)
    prompt = get_prompt_template()

    # Retrieval + Reranking
    retriever_mgr = RetrieverManager(pipeline_config, backend_config, embeddings)
    reranker = Reranker(embeddings, enable_reranker=backend_config.enable_reranker)

    # Memory (MongoDB-backed)
    memory_store = MongoSessionMemoryStore(
        mongo_uri=backend_config.mongodb_uri,
        db_name=backend_config.mongodb_db_name,
        max_recent_turns=backend_config.memory_max_recent_turns,
        max_token_limit=backend_config.max_context_tokens,
        session_ttl_seconds=backend_config.memory_session_ttl,
    )

    # Agent tools
    tools = [
        create_qdrant_search_tool(
            retriever_fn=retriever_mgr.retrieve,
            rerank_fn=reranker.rerank,
            rerank_top_k=backend_config.rerank_top_k,
        ),
        create_fit_website_tool(
            official_site_allowlist=backend_config.official_site_allowlist,
        ),
    ]

    # Agent
    react_agent = ReactRAGAgent(
        llm=llm,
        tools=tools,
        memory_store=memory_store,
        max_iterations=backend_config.agent_max_iterations,
    )

    logger.info("All RAG components wired successfully")

    return RAGService(
        retriever_mgr=retriever_mgr,
        reranker=reranker,
        llm=llm,
        prompt=prompt,
        react_agent=react_agent,
        memory_store=memory_store,
        backend_config=backend_config,
    )
