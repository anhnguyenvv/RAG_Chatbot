"""FastAPI application and route definitions."""

import hmac
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config.config import load_configs
from app.core.dependencies import build_rag_service
from app.storage.history import ChatHistoryStore

logger = logging.getLogger(__name__)

VALID_SOURCES = ["qdrant", "fit_web", "auto"]
VALID_MODES = ["classic", "agentic"]


def create_app() -> FastAPI:
    backend_config, pipeline_config = load_configs()
    logger.info(
        "Initializing RAG Backend API",
        extra={
            "collection": pipeline_config.collection_name,
            "model": backend_config.generate_model_name,
        },
    )

    rag_service = build_rag_service(backend_config, pipeline_config)
    history_store = ChatHistoryStore(db_path=backend_config.chat_history_db_path)

    # --- Rate Limiter ---
    limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(title="RAG Backend API")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    Instrumentator().instrument(app).expose(app)

    @app.get("/")
    @limiter.limit(backend_config.rate_limit_default)
    def read_root(request: Request):
        return {
            "message": "API RAG is running",
            "collection": pipeline_config.collection_name,
            "embedding_model": pipeline_config.embedding_model_name,
            "modes": VALID_MODES,
            "sources": VALID_SOURCES,
        }

    @app.get("/rag/{source}")
    @limiter.limit(backend_config.rate_limit_rag)
    async def rag_query(
        request: Request,
        source: str,
        q: str | None = None,
        mode: str = "classic",
        session_id: str | None = None,
        debug: bool = False,
    ):
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail=f"Invalid source. Must be one of: {VALID_SOURCES}")

        normalized_mode = mode.strip().lower()
        if normalized_mode not in VALID_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {VALID_MODES}")

        if not q:
            raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

        logger.info(
            "RAG query received",
            extra={"source": source, "mode": normalized_mode, "session_id": session_id, "query_len": len(q)},
        )

        try:
            output = rag_service.query(
                source=source,
                query=q,
                mode=normalized_mode,
                session_id=session_id,
                debug=debug,
            )
            sources = output.get("source_documents", [])

            history_id = history_store.add_entry(
                source=source,
                query=q,
                answer=output["result"],
                source_documents=sources,
            )

            response_payload = {
                "result": output["result"],
                "source_documents": sources,
                "history_id": history_id,
            }
            optional_fields = [
                "confidence",
                "route",
                "retrieval_stats",
                "rerank_stats",
                "citations",
                "needs_clarification",
                "fallback_reason",
                "timings",
                "request_id",
                "thought_process",
                "message_count",
            ]
            for field in optional_fields:
                if field in output:
                    response_payload[field] = output[field]

            logger.info(
                "RAG query completed",
                extra={"source": source, "mode": normalized_mode, "result_len": len(output["result"]), "num_sources": len(sources)},
            )
            return JSONResponse(content=jsonable_encoder(response_payload))
        except Exception as exc:
            logger.exception("RAG query failed", extra={"source": source, "query": q[:100]})
            raise HTTPException(status_code=500, detail=f"An error occurred: {exc}") from exc

    @app.get("/history")
    @limiter.limit(backend_config.rate_limit_default)
    def get_history(request: Request, limit: int = 50):
        data = history_store.list_entries(limit=limit)
        return JSONResponse(content=jsonable_encoder(data))

    @app.get("/history/{entry_id}")
    @limiter.limit(backend_config.rate_limit_default)
    def get_history_item(request: Request, entry_id: int):
        entry = history_store.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="History entry not found")
        return JSONResponse(content=jsonable_encoder(entry))

    @app.get("/sessions")
    @limiter.limit(backend_config.rate_limit_default)
    def list_sessions(request: Request):
        sessions = rag_service.memory_store.list_sessions()
        return JSONResponse(content=jsonable_encoder(sessions))

    @app.delete("/sessions/{session_id}")
    @limiter.limit(backend_config.rate_limit_default)
    def clear_session(request: Request, session_id: str):
        cleared = rag_service.memory_store.clear_session(session_id)
        if not cleared:
            raise HTTPException(status_code=404, detail="Session not found")
        logger.info("Session cleared", extra={"session_id": session_id})
        return JSONResponse(content={"message": f"Session '{session_id}' cleared"})

    # --- Admin endpoints (gated by ADMIN_API_KEY env var) ---
    def _require_admin(x_admin_key: str | None) -> None:
        expected = backend_config.admin_api_key
        if not expected:
            # Admin endpoints are disabled when no key is configured.
            raise HTTPException(status_code=403, detail="Admin endpoints disabled")
        if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
            raise HTTPException(status_code=401, detail="Invalid admin credentials")

    @app.get("/admin/cache/stats")
    @limiter.limit(backend_config.rate_limit_default)
    def cache_stats(request: Request, x_admin_key: str | None = Header(default=None)):
        _require_admin(x_admin_key)
        return JSONResponse(content=rag_service.retriever_mgr.cache_stats())

    @app.post("/admin/cache/clear")
    @limiter.limit(backend_config.rate_limit_default)
    def cache_clear(request: Request, x_admin_key: str | None = Header(default=None)):
        _require_admin(x_admin_key)
        evicted = rag_service.retriever_mgr.clear_query_cache()
        logger.info("Admin cleared retrieval cache", extra={"evicted": evicted})
        return JSONResponse(content={"evicted": evicted})

    return app
