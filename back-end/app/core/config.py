# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
import os
import sys

from dotenv import load_dotenv


@dataclass
class BackendConfig:
    generate_model_name: str = "gemini"
    google_api_key: str = ""
    ngrok_token: str = ""
    ngrok_static_domain: str = ""
    chat_history_db_path: str = "./data/chat_history.db"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    enable_reranker: bool = True
    enable_agentic_pipeline: bool = False
    router_confidence_threshold: float = 0.55
    retrieval_dense_top_k: int = 20
    retrieval_sparse_top_k: int = 20
    retrieval_web_top_k: int = 5
    rerank_model_name: str = ""
    rerank_candidates: int = 40
    max_context_tokens: int = 2800
    critic_enabled: bool = True
    official_site_allowlist: str = "fit.hcmus.edu.vn"
    agent_max_iterations: int = 5
    memory_max_recent_turns: int = 3
    memory_session_ttl: int = 1800
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "rag_chatbot"


def _to_bool(value: str, default: bool) -> bool:
    if value is None:
        return default

    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _add_data_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "Data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))


def load_configs():
    load_dotenv()

    backend_dir = Path(__file__).resolve().parents[2]
    raw_db_path = os.getenv("CHAT_HISTORY_DB_PATH", "./data/chat_history.db")
    db_path = Path(raw_db_path)
    if not db_path.is_absolute():
        db_path = (backend_dir / db_path).resolve()

    backend_config = BackendConfig(
        generate_model_name=os.getenv("GENERATE_MODEL_NAME", "gemini"),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        ngrok_token=os.getenv("NGROK_TOKEN", ""),
        ngrok_static_domain=os.getenv("NGROK_STATIC_DOMAIN", ""),
        chat_history_db_path=str(db_path),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
        enable_reranker=_to_bool(os.getenv("ENABLE_RERANKER", "true"), True),
        enable_agentic_pipeline=_to_bool(os.getenv("ENABLE_AGENTIC_PIPELINE", "false"), False),
        router_confidence_threshold=float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.55")),
        retrieval_dense_top_k=int(os.getenv("RETRIEVAL_DENSE_TOP_K", "20")),
        retrieval_sparse_top_k=int(os.getenv("RETRIEVAL_SPARSE_TOP_K", "20")),
        retrieval_web_top_k=int(os.getenv("RETRIEVAL_WEB_TOP_K", "5")),
        rerank_model_name=os.getenv("RERANK_MODEL_NAME", ""),
        rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "40")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "2800")),
        critic_enabled=_to_bool(os.getenv("CRITIC_ENABLED", "true"), True),
        official_site_allowlist=os.getenv("OFFICIAL_SITE_ALLOWLIST", "fit.hcmus.edu.vn"),
        agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
        memory_max_recent_turns=int(os.getenv("MEMORY_MAX_RECENT_TURNS", "3")),
        memory_session_ttl=int(os.getenv("MEMORY_SESSION_TTL", "1800")),
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_db_name=os.getenv("MONGODB_DB_NAME", "rag_chatbot"),
    )

    _add_data_path()
    from pipeline.config import PipelineConfig

    pipeline_config = PipelineConfig.from_env()
    return backend_config, pipeline_config
