from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Protocol
import uuid


@dataclass
class AgenticState:
    request_id: str
    source: str
    mode: str
    user_query: str
    session_id: str | None = None
    created_at: str = ""
    normalized_query: str = ""
    retrieval_query: str = ""
    answer_query: str = ""
    history_signals: dict[str, Any] = field(default_factory=dict)
    query_intents: list[str] = field(default_factory=list)
    route: str = "fast"
    retrieved_docs_raw: list[Any] = field(default_factory=list)
    reranked_docs: list[Any] = field(default_factory=list)
    reranked_pairs: list[tuple[Any, float | None]] = field(default_factory=list)
    context_blocks: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    draft_answer: str = ""
    verified_answer: str = ""
    source_documents: list[dict[str, Any]] = field(default_factory=list)
    citations_prepared: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    needs_clarification: bool = False
    fallback_reason: str = ""
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    rerank_stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


class AgenticNode(Protocol):
    name: str

    def run(self, state: AgenticState) -> AgenticState:
        ...


def _sanitize_query(value: str) -> str:
    collapsed = " ".join(value.split())
    blocked_tokens = ["ignore previous", "system prompt", "bypass", "jailbreak"]
    lowered = collapsed.lower()
    for token in blocked_tokens:
        lowered = lowered.replace(token, "")
    return " ".join(lowered.split())


def _token_overlap_score(query: str, content: str) -> float:
    query_tokens = {token for token in query.lower().split() if len(token) > 1}
    if not query_tokens:
        return 0.0
    content_tokens = set(content.lower().split())
    common = len(query_tokens.intersection(content_tokens))
    return common / max(1, len(query_tokens))


class IntakeNode:
    name = "intake"

    def run(self, state: AgenticState) -> AgenticState:
        state.created_at = datetime.now(timezone.utc).isoformat()
        state.normalized_query = _sanitize_query(state.user_query)
        if not state.normalized_query:
            state.fallback_reason = "empty_query"
        return state


class RewriteNode:
    name = "rewrite"

    DOMAIN_SYNONYMS = {
        "hoc phan": ["mon hoc", "hoc phan"],
        "ngoai ngu": ["tieng anh", "chuan ngoai ngu"],
        "khoa luan": ["tot nghiep", "de tai"],
    }

    def run(self, state: AgenticState) -> AgenticState:
        base = state.normalized_query or state.user_query
        retrieval_terms = [base]
        lowered = base.lower()
        intents = []
        for key, values in self.DOMAIN_SYNONYMS.items():
            if key in lowered or any(v in lowered for v in values):
                retrieval_terms.extend(values)
                intents.append(key)

        state.query_intents = intents
        state.retrieval_query = " | ".join(dict.fromkeys(retrieval_terms))
        state.answer_query = base
        return state


class RouterNode:
    name = "router"

    def run(self, state: AgenticState) -> AgenticState:
        q = (state.retrieval_query or "").lower()
        out_of_scope_words = ["thoi tiet", "gia vang", "bong da", "chinh tri the gioi"]
        if any(token in q for token in out_of_scope_words):
            state.route = "out_of_scope_precheck"
            return state

        deep_words = ["so sanh", "khac nhau", "dieu kien", "va", "hoac", "hoc ky", "quy che"]
        state.route = "deep" if any(token in q for token in deep_words) else "fast"
        return state


class RetrievalNode:
    name = "retrieval"

    def __init__(
        self,
        dense_retrieve_fn: Callable[[str, str, int], list[Any]],
        sparse_retrieve_fn: Callable[[str, str, int], list[Any]],
        web_retrieve_fn: Callable[[str, int], list[Any]],
        dense_top_k: int,
        sparse_top_k: int,
        web_top_k: int,
    ) -> None:
        self.dense_retrieve_fn = dense_retrieve_fn
        self.sparse_retrieve_fn = sparse_retrieve_fn
        self.web_retrieve_fn = web_retrieve_fn
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.web_top_k = web_top_k

    @staticmethod
    def _dedup_docs(documents: list[Any]) -> list[Any]:
        deduped: list[Any] = []
        seen: set[str] = set()
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            key = str(metadata.get("source") or metadata.get("url") or getattr(doc, "page_content", "")[:160])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    def run(self, state: AgenticState) -> AgenticState:
        dense_docs = self.dense_retrieve_fn(state.source, state.retrieval_query, self.dense_top_k)
        sparse_docs = self.sparse_retrieve_fn(state.source, state.retrieval_query, self.sparse_top_k)

        web_docs: list[Any] = []
        if state.source in {"fit_web", "auto"} or state.route in {"deep", "out_of_scope_precheck"}:
            web_docs = self.web_retrieve_fn(state.retrieval_query, self.web_top_k)

        docs = self._dedup_docs([*dense_docs, *sparse_docs, *web_docs])
        state.retrieved_docs_raw = docs
        state.retrieval_stats = {
            "retrieved_count": len(docs),
            "dense_count": len(dense_docs),
            "sparse_count": len(sparse_docs),
            "web_count": len(web_docs),
            "path": "hybrid",
        }

        if not docs and not state.fallback_reason:
            state.fallback_reason = "no_retrieval_evidence"
        return state


class RerankNode:
    name = "rerank"

    def __init__(self, rerank_fn: Callable[[str, list[Any], int], list[tuple[Any, float | None]]], top_k: int) -> None:
        self.rerank_fn = rerank_fn
        self.top_k = top_k

    def run(self, state: AgenticState) -> AgenticState:
        pairs = self.rerank_fn(state.answer_query, state.retrieved_docs_raw, self.top_k)
        state.reranked_pairs = pairs
        state.reranked_docs = [doc for doc, _ in pairs]
        state.rerank_stats = {
            "candidates": len(state.retrieved_docs_raw),
            "selected": len(pairs),
            "method": "cross_encoder_or_cosine_fallback",
        }
        return state


class ContextBuilderNode:
    name = "context_builder"

    def __init__(self, max_context_tokens: int) -> None:
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        return max(1, len(value) // 4)

    def run(self, state: AgenticState) -> AgenticState:
        blocks: list[dict[str, Any]] = []
        token_budget = self.max_context_tokens
        for idx, (doc, score) in enumerate(state.reranked_pairs, start=1):
            content = getattr(doc, "page_content", "")
            est_tokens = self._estimate_tokens(content)
            if est_tokens > token_budget:
                continue
            token_budget -= est_tokens
            metadata = getattr(doc, "metadata", {}) or {}
            block = {
                "index": idx,
                "text": content,
                "score": score,
                "source": metadata.get("source") or metadata.get("url") or "unknown",
                "metadata": metadata,
            }
            blocks.append(block)

        state.context_blocks = blocks
        state.citations_prepared = [
            {
                "doc_index": block["index"],
                "source": block["source"],
                "score": block["score"],
            }
            for block in blocks
        ]

        state.context = "\n\n".join(f"[Doc {b['index']}]\n{b['text']}" for b in blocks)
        return state


class AnswerNode:
    name = "answer"

    def __init__(self, generate_fn: Callable[[str, str], str]) -> None:
        self.generate_fn = generate_fn

    def run(self, state: AgenticState) -> AgenticState:
        if not state.context_blocks:
            state.draft_answer = (
                "Mình chưa có đủ bằng chứng trong kho dữ liệu để trả lời chắc chắn. "
                "Bạn có thể nêu rõ hơn ngành/chuyên ngành hoặc học kỳ để mình truy xuất chính xác hơn."
            )
            state.citations = []
            return state

        state.draft_answer = self.generate_fn(state.context, state.answer_query)
        state.citations = list(state.citations_prepared)
        return state


class CriticNode:
    name = "critic"

    def __init__(self, confidence_threshold: float, critic_enabled: bool) -> None:
        self.confidence_threshold = confidence_threshold
        self.critic_enabled = critic_enabled

    def run(self, state: AgenticState) -> AgenticState:
        if not self.critic_enabled:
            state.verified_answer = state.draft_answer
            return state

        doc_count = len(state.reranked_pairs)
        state.verified_answer = state.draft_answer
        base = 0.35 + 0.08 * doc_count
        if state.citations:
            base += 0.1
        state.confidence = min(0.95, base)
        state.needs_clarification = doc_count == 0
        if state.confidence < self.confidence_threshold:
            state.needs_clarification = True
            if not state.fallback_reason:
                state.fallback_reason = "low_confidence"

        if doc_count == 0 and not state.fallback_reason:
            state.fallback_reason = "no_evidence"
        return state


class ResponseAdapterNode:
    name = "response_adapter"

    def run(self, state: AgenticState) -> AgenticState:
        if state.needs_clarification and state.fallback_reason in {"no_evidence", "low_confidence", "no_retrieval_evidence"}:
            state.verified_answer = (
                f"{state.verified_answer}\n\n"
                "Mình cần bạn làm rõ thêm phạm vi câu hỏi (ngành, khóa tuyển, học kỳ hoặc văn bản liên quan) "
                "để trả lời chính xác hơn."
            )
        return state


class AgenticPipeline:
    def __init__(
        self,
        dense_retrieve_fn: Callable[[str, str, int], list[Any]],
        sparse_retrieve_fn: Callable[[str, str, int], list[Any]],
        web_retrieve_fn: Callable[[str, int], list[Any]],
        rerank_fn: Callable[[str, list[Any], int], list[tuple[Any, float | None]]],
        generate_fn: Callable[[str, str], str],
        source_payload_fn: Callable[[list[tuple[Any, float | None]]], list[dict[str, Any]]],
        dense_top_k: int,
        sparse_top_k: int,
        web_top_k: int,
        rerank_top_k: int,
        max_context_tokens: int,
        router_confidence_threshold: float,
        critic_enabled: bool,
    ) -> None:
        self.source_payload_fn = source_payload_fn
        self.nodes: list[AgenticNode] = [
            IntakeNode(),
            RewriteNode(),
            RouterNode(),
            RetrievalNode(
                dense_retrieve_fn=dense_retrieve_fn,
                sparse_retrieve_fn=sparse_retrieve_fn,
                web_retrieve_fn=web_retrieve_fn,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                web_top_k=web_top_k,
            ),
            RerankNode(rerank_fn=rerank_fn, top_k=rerank_top_k),
            ContextBuilderNode(max_context_tokens=max_context_tokens),
            AnswerNode(generate_fn=generate_fn),
            CriticNode(confidence_threshold=router_confidence_threshold, critic_enabled=critic_enabled),
            ResponseAdapterNode(),
        ]

    def run(self, source: str, query: str, mode: str = "agentic", session_id: str | None = None) -> dict[str, Any]:
        state = AgenticState(
            request_id=str(uuid.uuid4()),
            source=source,
            mode=mode,
            user_query=query,
            session_id=session_id,
        )

        for node in self.nodes:
            start = perf_counter()
            state = node.run(state)
            state.timings[node.name] = perf_counter() - start

        state.source_documents = self.source_payload_fn(state.reranked_pairs)

        return {
            "result": state.verified_answer,
            "source_documents": state.source_documents,
            "confidence": state.confidence,
            "route": state.route,
            "retrieval_stats": state.retrieval_stats,
            "rerank_stats": state.rerank_stats,
            "citations": state.citations,
            "needs_clarification": state.needs_clarification,
            "fallback_reason": state.fallback_reason,
            "timings": state.timings,
            "request_id": state.request_id,
            "state": {
                "request_id": state.request_id,
                "source": state.source,
                "mode": state.mode,
                "user_query": state.user_query,
                "normalized_query": state.normalized_query,
                "retrieval_query": state.retrieval_query,
                "answer_query": state.answer_query,
                "history_signals": state.history_signals,
                "route": state.route,
                "retrieved_docs_raw": len(state.retrieved_docs_raw),
                "reranked_docs": len(state.reranked_docs),
                "context_blocks": len(state.context_blocks),
                "draft_answer": state.draft_answer,
                "verified_answer": state.verified_answer,
                "citations": state.citations,
                "confidence": state.confidence,
                "needs_clarification": state.needs_clarification,
                "fallback_reason": state.fallback_reason,
                "timings": state.timings,
            },
        }