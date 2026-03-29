from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .config import load_configs
from .history_store import ChatHistoryStore
from .llm_service import LLMServe

VALID_SOURCES = ["qdrant", "wiki", "fit_web", "auto"]
VALID_MODES = ["classic", "agentic"]


def create_app() -> FastAPI:
    backend_config, pipeline_config = load_configs()

    # Validate qdrant config only for qdrant source usage.
    rag_service = LLMServe(backend_config=backend_config, pipeline_config=pipeline_config)
    history_store = ChatHistoryStore(db_path=backend_config.chat_history_db_path)

    app = FastAPI(title="RAG Backend API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Expose Prometheus metrics at /metrics for monitoring stack.
    Instrumentator().instrument(app).expose(app)

    @app.get("/")
    def read_root():
        return {
            "message": "API RAG is running",
            "collection": pipeline_config.collection_name,
            "embedding_model": pipeline_config.embedding_model_name,
        }

    @app.get("/rag/{source}")
    async def rag_query(
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
                "state",
            ]
            for field in optional_fields:
                if field in output:
                    response_payload[field] = output[field]

            return JSONResponse(content=jsonable_encoder(response_payload))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"An error occurred: {exc}") from exc

    @app.get("/history")
    def get_history(limit: int = 50):
        data = history_store.list_entries(limit=limit)
        return JSONResponse(content=jsonable_encoder(data))

    @app.get("/history/{entry_id}")
    def get_history_item(entry_id: int):
        entry = history_store.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="History entry not found")
        return JSONResponse(content=jsonable_encoder(entry))

    return app
