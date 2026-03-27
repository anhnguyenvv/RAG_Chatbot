# pyright: reportMissingImports=false

import math
import re
from typing import Any

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain.llms import VLLM
from qdrant_client import QdrantClient

from .agentic import AgenticPipeline


class LLMServe:
    def __init__(self, backend_config, pipeline_config) -> None:
        self.backend_config = backend_config
        self.pipeline_config = pipeline_config
        self.retriever_cache: dict[str, Any] = {}
        self.embeddings = self._load_embeddings()
        self.llm = self._load_llm(self.backend_config.generate_model_name)
        self.prompt = self._load_prompt_template()
        self.agentic_pipeline = AgenticPipeline(
            dense_retrieve_fn=self._retrieve_dense_documents,
            sparse_retrieve_fn=self._retrieve_sparse_documents,
            web_retrieve_fn=self._retrieve_web_documents,
            rerank_fn=self._rerank_documents,
            generate_fn=self._generate_answer_from_context,
            source_payload_fn=self._build_sources_payload,
            dense_top_k=self.backend_config.retrieval_dense_top_k,
            sparse_top_k=self.backend_config.retrieval_sparse_top_k,
            web_top_k=self.backend_config.retrieval_web_top_k,
            rerank_top_k=self.backend_config.rerank_top_k,
            max_context_tokens=self.backend_config.max_context_tokens,
            router_confidence_threshold=self.backend_config.router_confidence_threshold,
            critic_enabled=self.backend_config.critic_enabled,
        )

    def _load_embeddings(self):
        return HuggingFaceInferenceAPIEmbeddings(
            model_name=self.pipeline_config.embedding_model_name,
            api_key=self.pipeline_config.huggingface_api_key,
        )

    def _load_retriever(self, source: str):
        if source == "wiki":
            return WikipediaRetriever(
                lang="vi",
                doc_content_chars_max=800,
                top_k_results=self.backend_config.retrieval_top_k,
            )

        if source in {"fit_web", "auto"}:
            source = "qdrant"

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
        return db.as_retriever(search_kwargs={"k": self.backend_config.retrieval_top_k})

    def _load_llm(self, model_name: str, max_new_tokens: int = 384):
        if model_name == "gemini":
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        return VLLM(
            model=model_name,
            trust_remote_code=True,
            max_new_tokens=max_new_tokens,
            top_k=10,
            top_p=0.95,
            temperature=0.4,
            dtype="half",
        )

    def _load_prompt_template(self):
        query_template = """
            <|system|>
            You are required to answer the question based only on the provided context.
            - The answer must be accurate, complete, and relevant to the question.
            - If multiple context passages contain relevant information, combine them to form a comprehensive answer.
            - Only use the information provided in the context. Do not add any external knowledge.
            - If the context does not contain enough information, return:
            \"Vui lòng liên lạc Khoa Công Nghệ Thông Tin, trường Đại học Khoa Học Tự Nhiên - Đại học Quốc Gia TP.Hồ Chí Minh để giải đáp:
            Địa chỉ: Phòng I.54, toà nhà I, 227 Nguyễn Văn Cừ, Q.5, TP.HCM
            Điện thoại: (028) 62884499
            Email: info@fit.hcmus.edu.vn\"
            - The final answer must be written in Vietnamese.
            </s>
            <|user|>
            Context:
            {context}
            ---
            Question: {question}
            </s>
            <|assistant|>
        """
        return PromptTemplate(template=query_template, input_variables=["context", "question"])

    def _get_retriever(self, source: str):
        cached = self.retriever_cache.get(source)
        if cached is not None:
            return cached

        retriever = self._load_retriever(source)
        self.retriever_cache[source] = retriever
        return retriever

    def _retrieve_documents(self, source: str, query: str):
        retriever = self._get_retriever(source)
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        return retriever.get_relevant_documents(query)

    def _retrieve_dense_documents(self, source: str, query: str, top_k: int) -> list[Any]:
        previous_top_k = self.backend_config.retrieval_top_k
        self.backend_config.retrieval_top_k = top_k
        try:
            return list(self._retrieve_documents(source=source, query=query))
        finally:
            self.backend_config.retrieval_top_k = previous_top_k

    @staticmethod
    def _token_overlap_score(query: str, content: str) -> float:
        q_tokens = {t for t in re.split(r"\W+", query.lower()) if len(t) > 1}
        if not q_tokens:
            return 0.0
        c_tokens = set(re.split(r"\W+", content.lower()))
        return len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))

    def _retrieve_sparse_documents(self, source: str, query: str, top_k: int) -> list[Any]:
        # Sparse retrieval skeleton: lexical scoring over a wider dense pool.
        candidate_k = max(top_k * 3, self.backend_config.retrieval_dense_top_k)
        candidates = self._retrieve_dense_documents(source=source, query=query, top_k=candidate_k)
        if not candidates:
            return []

        scored = []
        for doc in candidates:
            content = getattr(doc, "page_content", "")
            score = self._token_overlap_score(query=query, content=content)
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, score in scored[:top_k] if score > 0.0]

    def _retrieve_web_documents(self, query: str, top_k: int) -> list[Any]:
        allowlist = [host.strip() for host in self.backend_config.official_site_allowlist.split(",") if host.strip()]
        if not allowlist:
            return []

        try:
            import httpx
        except Exception:
            return []

        documents: list[Any] = []
        timeout_seconds = 8.0
        for host in allowlist[:1]:
            url = f"https://{host}"
            try:
                response = httpx.get(url, timeout=timeout_seconds)
                response.raise_for_status()
                html = response.text
            except Exception:
                continue

            text = ""
            try:
                import trafilatura

                extracted = trafilatura.extract(html)
                if extracted:
                    text = extracted
            except Exception:
                text = ""

            if not text:
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(html, "html.parser")
                    text = " ".join(soup.get_text(" ").split())
                except Exception:
                    text = ""

            if not text:
                continue

            clipped = text[:4000]
            score = self._token_overlap_score(query=query, content=clipped)
            metadata = {
                "source": url,
                "url": url,
                "source_type": "official_web",
                "score": score,
            }
            documents.append(Document(page_content=clipped, metadata=metadata))

        documents.sort(key=lambda d: float((d.metadata or {}).get("score", 0.0)), reverse=True)
        return documents[:top_k]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _rerank_documents(self, query: str, documents: list[Any], top_k: int | None = None):
        if not documents:
            return []

        selected_top_k = top_k if top_k is not None else self.backend_config.rerank_top_k
        top_n = max(1, min(selected_top_k, len(documents)))

        if not self.backend_config.enable_reranker:
            return [(doc, None) for doc in documents[:top_n]]

        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
            scored = []
            for doc, emb in zip(documents, doc_embeddings):
                score = self._cosine_similarity(query_embedding, emb)
                scored.append((doc, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_n]
        except Exception:
            return [(doc, None) for doc in documents[:top_n]]

    @staticmethod
    def _format_context(documents: list[Any]) -> str:
        if not documents:
            return ""
        blocks = []
        for idx, doc in enumerate(documents, start=1):
            blocks.append(f"[Doc {idx}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _normalize_llm_output(output: Any) -> str:
        if hasattr(output, "content"):
            return str(output.content)
        return str(output)

    def _generate_answer_from_context(self, context: str, question: str) -> str:
        prompt_text = self.prompt.format(context=context, question=question)
        return self._normalize_llm_output(self.llm.invoke(prompt_text))

    @staticmethod
    def _build_sources_payload(reranked_pairs: list[tuple[Any, float | None]]) -> list[dict[str, Any]]:
        sources_payload = []
        for doc, score in reranked_pairs:
            payload = doc.to_json()["kwargs"]
            payload.setdefault("metadata", {})
            payload["metadata"]["rerank_score"] = score
            sources_payload.append(payload)
        return sources_payload

    def _query_classic(self, source: str, query: str) -> dict[str, Any]:
        retrieved_docs = self._retrieve_documents(source=source, query=query)
        reranked_pairs = self._rerank_documents(query=query, documents=retrieved_docs, top_k=self.backend_config.rerank_top_k)
        reranked_docs = [doc for doc, _ in reranked_pairs]

        context = self._format_context(reranked_docs)
        answer = self._generate_answer_from_context(context=context, question=query)
        sources_payload = self._build_sources_payload(reranked_pairs)

        return {
            "result": answer,
            "source_documents": sources_payload,
        }

    def query(
        self,
        source: str,
        query: str,
        mode: str = "classic",
        session_id: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        requested_mode = (mode or "classic").strip().lower()

        if requested_mode == "agentic":
            if self.backend_config.enable_agentic_pipeline:
                output = self.agentic_pipeline.run(source=source, query=query, mode=requested_mode, session_id=session_id)
                if not debug:
                    output.pop("state", None)
                    output.pop("timings", None)
                return output

            classic_output = self._query_classic(source=source, query=query)
            classic_output["route"] = "classic_fallback"
            classic_output["fallback_reason"] = "agentic_disabled"
            return classic_output

        return self._query_classic(source=source, query=query)
