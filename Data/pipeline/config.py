from dataclasses import dataclass
import os


def _to_bool(value: str, default: bool) -> bool:
    if value is None:
        return default

    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass
class PipelineConfig:
    source_dir: str = "./Database"
    collection_name: str = "ITUS_mpnet_600v1"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    chunk_strategy: str = "outline"
    chunk_size: int = 600
    chunk_overlap: int = 200
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_force_recreate: bool = True
    huggingface_api_key: str = ""

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            source_dir=os.getenv("PIPELINE_SOURCE_DIR", "./Database"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "ITUS_mpnet_600v1"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            chunk_strategy=os.getenv("CHUNK_STRATEGY", "outline"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            qdrant_force_recreate=_to_bool(os.getenv("QDRANT_FORCE_RECREATE", "true"), True),
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY", ""),
        )

    def validate(self) -> None:
        required = {
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_API_KEY": self.qdrant_api_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "Missing required environment variables: "
                f"{joined}. "
                "Set them in Data/.env (copy from Data/.env.example) "
                "or pass via CLI flags: --qdrant-url, --qdrant-api-key."
            )

        if self.chunk_strategy not in {"recursive", "outline"}:
            raise ValueError("CHUNK_STRATEGY must be one of: recursive, outline")
