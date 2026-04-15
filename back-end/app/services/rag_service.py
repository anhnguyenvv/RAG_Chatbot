"""RAG service — thin orchestrator routing queries to classic or agentic pipelines."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import PromptTemplate

from app.rag.agent import ReactRAGAgent
from app.rag.generator import build_sources_payload, format_context, generate_answer
from app.rag.reranker import Reranker
from app.rag.retriever import RetrieverManager
from app.storage.memory import MongoSessionMemoryStore

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates classic and agentic RAG pipelines."""

    def __init__(
        self,
        retriever_mgr: RetrieverManager,
        reranker: Reranker,
        llm: Any,
        prompt: PromptTemplate,
        react_agent: ReactRAGAgent,
        memory_store: MongoSessionMemoryStore,
        backend_config: Any,
    ) -> None:
        self.retriever_mgr = retriever_mgr
        self.reranker = reranker
        self.llm = llm
        self.prompt = prompt
        self.react_agent = react_agent
        self.memory_store = memory_store
        self.backend_config = backend_config
        logger.info("RAGService initialized")

    def query(
        self,
        source: str,
        query: str,
        mode: str = "classic",
        session_id: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        requested_mode = (mode or "classic").strip().lower()
        logger.info("Processing query mode=%s source=%s", requested_mode, source)

        if requested_mode == "agentic":
            return self.react_agent.run(
                query=query,
                session_id=session_id,
                debug=debug,
            )

        return self._query_classic(source=source, query=query)

    def _query_classic(self, source: str, query: str) -> dict[str, Any]:
        retrieved_docs = self.retriever_mgr.retrieve(source=source, query=query)
        reranked_pairs = self.reranker.rerank(
            query=query,
            documents=retrieved_docs,
            top_k=self.backend_config.rerank_top_k,
        )
        reranked_docs = [doc for doc, _ in reranked_pairs]

        context = format_context(reranked_docs)
        answer = generate_answer(
            llm=self.llm,
            prompt=self.prompt,
            context=context,
            question=query,
        )
        sources_payload = build_sources_payload(reranked_pairs)

        return {
            "result": answer,
            "source_documents": sources_payload,
        }
