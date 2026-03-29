import argparse
import importlib
import logging
from pathlib import Path

from dotenv import load_dotenv

config_module = importlib.import_module("pipeline.config")
pipeline_module = importlib.import_module("pipeline.pipeline")

PipelineConfig = config_module.PipelineConfig
build_vector_index = pipeline_module.build_vector_index
run_smoke_search = pipeline_module.run_smoke_search


def parse_args():
    parser = argparse.ArgumentParser(description="Build Qdrant vector index for RAG dataset")
    parser.add_argument("--source-dir", default=None, help="Path to folder containing .txt files")
    parser.add_argument("--collection", default=None, help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL override")
    parser.add_argument("--qdrant-api-key", default=None, help="Qdrant API key override")
    parser.add_argument("--hf-api-key", default=None, help="Hugging Face API key override")
    parser.add_argument("--chunk-strategy", choices=["recursive", "outline"], default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument(
        "--enable-web-crawl",
        action="store_true",
        help="Enable crawling website pages and merge them with local data",
    )
    parser.add_argument(
        "--web-start-urls",
        default=None,
        help="Comma-separated seed URLs for crawler, e.g. https://it.hcmus.edu.vn,https://it.hcmus.edu.vn/tin-tuc",
    )
    parser.add_argument(
        "--web-allowed-domains",
        default=None,
        help="Comma-separated allowed domains for crawler, e.g. it.hcmus.edu.vn",
    )
    parser.add_argument("--web-max-pages", type=int, default=None, help="Maximum pages to crawl")
    parser.add_argument("--web-timeout-seconds", type=int, default=None, help="HTTP timeout per page")
    parser.add_argument(
        "--query",
        default="Các môn học Học kỳ 2 năm 3 ngành hệ thống thông tin",
        help="Smoke test query after indexing",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for pipeline debugging",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main():
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")
    load_dotenv()
    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline run", extra={"log_level": args.log_level})

    config = PipelineConfig.from_env()

    if args.source_dir:
        config.source_dir = args.source_dir
    if args.collection:
        config.collection_name = args.collection
    if args.qdrant_url:
        config.qdrant_url = args.qdrant_url
    if args.qdrant_api_key:
        config.qdrant_api_key = args.qdrant_api_key
    if args.hf_api_key:
        config.huggingface_api_key = args.hf_api_key
    if args.chunk_strategy:
        config.chunk_strategy = args.chunk_strategy
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap
    if args.enable_web_crawl:
        config.enable_web_crawl = True
    if args.web_start_urls:
        config.web_start_urls = [url.strip() for url in args.web_start_urls.split(",") if url.strip()]
    if args.web_allowed_domains:
        config.web_allowed_domains = [d.strip() for d in args.web_allowed_domains.split(",") if d.strip()]
    if args.web_max_pages:
        config.web_max_pages = args.web_max_pages
    if args.web_timeout_seconds:
        config.web_timeout_seconds = args.web_timeout_seconds

    if not Path(config.source_dir).exists() and not config.enable_web_crawl:
        raise FileNotFoundError(f"Source dir does not exist: {config.source_dir}")

    result = build_vector_index(config)

    print(f"Loaded documents: {len(result['raw_documents'])}")
    print(f"Chunked documents: {len(result['chunked_documents'])}")
    print(f"Collection indexed: {config.collection_name}")

    found_docs = run_smoke_search(result["vector_store"], query=args.query, top_k=args.top_k)
    print("\nSmoke test retrieval:")
    for i, doc in enumerate(found_docs, start=1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"{i}. source={doc.metadata.get('source', 'unknown')} | {preview}")


if __name__ == "__main__":
    main()
