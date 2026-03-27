import logging
import importlib
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


def _resolve_hf_embeddings_class():
    try:
        module = importlib.import_module("langchain_huggingface")
        return getattr(module, "HuggingFaceEmbeddings", None)
    except ImportError:
        return None


class CompatibleHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        if api_key:
            logger.info(
                "HUGGINGFACE_API_KEY is set but local embedding backend does not require it",
                extra={"model_name": self.model_name},
            )
        self._client = None
        self._local_model = None

        hf_embeddings_class = _resolve_hf_embeddings_class()
        if hf_embeddings_class is not None:
            self._client = hf_embeddings_class(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            logger.warning(
                "langchain_huggingface is not installed, using sentence-transformers fallback",
                extra={"model_name": self.model_name},
            )
            self._local_model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._client is not None:
            vectors = self._client.embed_documents(texts)
        else:
            vectors = self._local_model.encode(texts, normalize_embeddings=True).tolist()
        if len(vectors) != len(texts):
            logger.warning(
                "Embedding count mismatch",
                extra={"expected": len(texts), "actual": len(vectors)},
            )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if self._client is not None:
            return self._client.embed_query(text)
        return self._local_model.encode(text, normalize_embeddings=True).tolist()


def create_embeddings(model_name: str, api_key: str) -> Embeddings:
    logger.info("Creating langchain_huggingface embedding client", extra={"model_name": model_name})
    return CompatibleHuggingFaceEmbeddings(model_name=model_name, api_key=api_key)
