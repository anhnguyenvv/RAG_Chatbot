# pyright: reportMissingImports=false
"""Factory functions for creating LLM and embedding instances."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_embeddings(pipeline_config: Any):
    """Create HuggingFace embeddings from pipeline config."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    logger.info("Loading embeddings: %s", pipeline_config.embedding_model_name)
    return HuggingFaceEndpointEmbeddings(
        model=pipeline_config.embedding_model_name,
        huggingfacehub_api_token=pipeline_config.huggingface_api_key,
    )


def create_llm(model_name: str, max_new_tokens: int = 384):
    """Create LLM instance (Gemini or VLLM)."""
    if model_name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Using Gemini model: gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    try:
        from langchain_community.llms import VLLM

        logger.info("Using VLLM model: %s", model_name)
        return VLLM(
            model=model_name,
            trust_remote_code=True,
            max_new_tokens=max_new_tokens,
            top_k=10,
            top_p=0.95,
            temperature=0.4,
            dtype="half",
        )
    except ImportError:
        raise ValueError(
            f"VLLM model '{model_name}' requested but langchain-community is not installed. "
            "Install with: pip install langchain-community"
        )
