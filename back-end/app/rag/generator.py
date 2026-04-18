"""Answer generation and context formatting."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import PromptTemplate

from app.config.prompts import CLASSIC_RAG_TEMPLATE

logger = logging.getLogger(__name__)


def get_prompt_template() -> PromptTemplate:
    """Create the classic RAG prompt template."""
    return PromptTemplate(
        template=CLASSIC_RAG_TEMPLATE,
        input_variables=["context", "question"],
    )


def format_context(documents: list[Any]) -> str:
    """Format documents into numbered context blocks."""
    if not documents:
        return ""
    blocks = []
    for idx, doc in enumerate(documents, start=1):
        blocks.append(f"[Doc {idx}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def normalize_llm_output(output: Any) -> str:
    """Extract text from LLM output (handles both string and AIMessage)."""
    if hasattr(output, "content"):
        return str(output.content)
    return str(output)


def generate_answer(llm: Any, prompt: PromptTemplate, context: str, question: str) -> str:
    """Generate answer from context using LLM."""
    prompt_text = prompt.format(context=context, question=question)
    result = normalize_llm_output(llm.invoke(prompt_text))
    logger.debug("Generated answer: %d chars", len(result))
    return result


def build_sources_payload(
    reranked_pairs: list[tuple[Any, float | None]],
) -> list[dict[str, Any]]:
    """Convert reranked document pairs to API response payload."""
    sources_payload = []
    for doc, score in reranked_pairs:
        payload = doc.to_json()["kwargs"]
        payload.setdefault("metadata", {})
        payload["metadata"]["rerank_score"] = score
        sources_payload.append(payload)
    return sources_payload
