"""Data quality eval — scans Database/pdf_crawled/*.txt and reports stats.

Run: python Data/tests/data_quality_report.py

Produces:
  - Overall file count, size distribution
  - Header compliance (crawl-format `# ---` separator)
  - Metadata coverage (nganh, year, he_dao_tao)
  - Duplicate detection (by content hash)
  - Empty/near-empty file flagging
  - Chunk distribution (via outline splitter)
  - Distribution by he_dao_tao / nganh / nam_ban_hanh
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

# Allow running from anywhere
_DATA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DATA_DIR))

# Avoid triggering sentence_transformers import chain by stubbing
import types
sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
sys.modules["sentence_transformers"].SentenceTransformer = type(  # type: ignore[attr-defined]
    "_Stub", (), {"__init__": lambda *a, **k: None}
)
sys.modules.setdefault("pipeline.web_crawler", types.ModuleType("pipeline.web_crawler"))
sys.modules["pipeline.web_crawler"].crawl_website_documents = lambda *a, **k: []  # type: ignore[attr-defined]

from pipeline.loaders import load_txt_documents, _parse_crawl_header, _parse_crawl_filename  # noqa: E402
from pipeline.splitters import chunk_documents  # noqa: E402


CRAWL_DIR = _DATA_DIR / "Database" / "pdf_crawled"
MIN_CHARS_NON_EMPTY = 200
DUPLICATE_CHAR_THRESHOLD = 50  # prefix length for near-duplicate detection


def report(title: str, body: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")
    print(body)


def format_kv(items: list[tuple[str, int]], total: int, top_n: int = 10) -> str:
    lines = []
    for k, v in items[:top_n]:
        pct = v / total * 100 if total else 0
        bar = "█" * int(pct / 2)  # 50% = 25 chars
        lines.append(f"  {str(k):<35} {v:>5} ({pct:5.1f}%) {bar}")
    if len(items) > top_n:
        lines.append(f"  ... +{len(items) - top_n} more")
    return "\n".join(lines)


def main() -> int:
    if not CRAWL_DIR.exists():
        print(f"ERROR: {CRAWL_DIR} not found")
        return 1

    files = sorted(CRAWL_DIR.glob("*.txt"))
    if not files:
        print(f"ERROR: no .txt files in {CRAWL_DIR}")
        return 1

    n = len(files)
    print(f"Scanning {n} files in {CRAWL_DIR.name}/\n")

    # --- File size + content stats ---
    sizes, char_counts, line_counts = [], [], []
    has_header_count = 0
    empty_body_count = 0
    near_empty_count = 0
    content_hashes: dict[str, list[str]] = defaultdict(list)
    prefix_hashes: dict[str, list[str]] = defaultdict(list)
    body_empty_files: list[str] = []

    for f in files:
        size = f.stat().st_size
        content = f.read_text(encoding="utf-8", errors="replace")
        chars = len(content)
        lines = content.count("\n") + 1

        sizes.append(size)
        char_counts.append(chars)
        line_counts.append(lines)

        header, body = _parse_crawl_header(content)
        if header:
            has_header_count += 1

        if not body or len(body.strip()) < 10:
            empty_body_count += 1
            body_empty_files.append(f.name)
        elif len(body.strip()) < MIN_CHARS_NON_EMPTY:
            near_empty_count += 1

        # Full content hash (exact duplicates)
        content_hashes[hashlib.md5(content.encode()).hexdigest()].append(f.name)
        # Body prefix hash (near duplicates)
        if body:
            prefix = body.strip()[:DUPLICATE_CHAR_THRESHOLD]
            if prefix:
                prefix_hashes[hashlib.md5(prefix.encode()).hexdigest()].append(f.name)

    report(
        "FILE STATS",
        (
            f"  Files              : {n}\n"
            f"  Total bytes        : {sum(sizes):,}  ({sum(sizes) / 1024 / 1024:.1f} MB)\n"
            f"  Size (min / median / mean / max, bytes):\n"
            f"    {min(sizes):,} / {int(median(sizes)):,} / {int(mean(sizes)):,} / {max(sizes):,}\n"
            f"  Chars (median / mean / max): "
            f"{int(median(char_counts)):,} / {int(mean(char_counts)):,} / {max(char_counts):,}\n"
            f"  Lines (median / mean / max): "
            f"{int(median(line_counts))} / {int(mean(line_counts))} / {max(line_counts)}"
        ),
    )

    report(
        "HEADER COMPLIANCE",
        (
            f"  Crawl-format header parsed : {has_header_count:>5} / {n} "
            f"({has_header_count/n*100:.1f}%)\n"
            f"  Empty body (<10 chars)     : {empty_body_count:>5}\n"
            f"  Near-empty (<{MIN_CHARS_NON_EMPTY} chars)      : {near_empty_count:>5}"
        )
        + ("\n  Empty-body files:\n    " + "\n    ".join(body_empty_files[:10])
           if body_empty_files else ""),
    )

    # --- Duplicate detection ---
    exact_dups = {h: names for h, names in content_hashes.items() if len(names) > 1}
    near_dups = {h: names for h, names in prefix_hashes.items() if len(names) > 1}
    dup_lines = []
    for h, names in list(exact_dups.items())[:5]:
        dup_lines.append(f"  EXACT: {names}")
    for h, names in list(near_dups.items())[:5]:
        if h not in exact_dups:
            dup_lines.append(f"  NEAR:  {names[:3]}{'...' if len(names) > 3 else ''}  (+{len(names)-3})")
    report(
        "DUPLICATE DETECTION",
        (
            f"  Exact duplicate groups : {len(exact_dups)}\n"
            f"  Near-duplicate groups  : {len(near_dups) - len(exact_dups)} "
            f"(same first {DUPLICATE_CHAR_THRESHOLD} chars of body)\n"
            + ("\n".join(dup_lines) if dup_lines else "  None")
        ),
    )

    # --- Metadata distribution via loader ---
    try:
        docs = load_txt_documents(str(CRAWL_DIR))
    except Exception as e:
        print(f"\n[WARN] load_txt_documents failed: {e}")
        return 0

    he_dao_tao_dist = Counter()
    nganh_dist = Counter()
    year_dist = Counter()
    loai_van_ban_dist = Counter()
    missing_year = 0
    missing_nganh = 0
    missing_he_dao_tao = 0

    for d in docs:
        m = d.metadata
        he_dao_tao_dist[m.get("he_dao_tao") or "<missing>"] += 1
        nganh_dist[m.get("nganh") or "<missing>"] += 1
        year_dist[str(m.get("nam_ban_hanh") or "<missing>")] += 1
        loai_van_ban_dist[m.get("loai_van_ban") or "<missing>"] += 1
        missing_year += 1 if m.get("nam_ban_hanh") is None else 0
        missing_nganh += 1 if not m.get("nganh") else 0
        missing_he_dao_tao += 1 if not m.get("he_dao_tao") else 0

    report(
        "METADATA COVERAGE",
        (
            f"  Documents                  : {len(docs)}\n"
            f"  Missing nam_ban_hanh       : {missing_year} ({missing_year/len(docs)*100:.1f}%)\n"
            f"  Missing nganh              : {missing_nganh} ({missing_nganh/len(docs)*100:.1f}%)\n"
            f"  Missing he_dao_tao         : {missing_he_dao_tao} ({missing_he_dao_tao/len(docs)*100:.1f}%)"
        ),
    )

    report(
        "HỆ ĐÀO TẠO DISTRIBUTION",
        format_kv(he_dao_tao_dist.most_common(), len(docs)),
    )
    report(
        "NGÀNH DISTRIBUTION",
        format_kv(nganh_dist.most_common(), len(docs)),
    )
    report(
        "NĂM BAN HÀNH DISTRIBUTION",
        format_kv(sorted(year_dist.items(), key=lambda x: (x[0] == "<missing>", x[0])), len(docs)),
    )
    report(
        "LOẠI VĂN BẢN DISTRIBUTION",
        format_kv(loai_van_ban_dist.most_common(), len(docs)),
    )

    # --- Chunk distribution ---
    try:
        chunks = chunk_documents(docs, strategy="outline", chunk_size=1000, chunk_overlap=200)
        chunk_lens = [len(c.page_content) for c in chunks]
        short_chunks = sum(1 for ln in chunk_lens if ln < 100)
        oversized = sum(1 for ln in chunk_lens if ln > 1500)
        report(
            "CHUNK DISTRIBUTION (outline, size=1000, overlap=200)",
            (
                f"  Total chunks             : {len(chunks)}\n"
                f"  Chunks / doc (avg)       : {len(chunks)/len(docs):.1f}\n"
                f"  Chunk size (min/median/mean/max): "
                f"{min(chunk_lens)} / {int(median(chunk_lens))} / "
                f"{int(mean(chunk_lens))} / {max(chunk_lens)}\n"
                f"  Short chunks (<100 chars): {short_chunks} ({short_chunks/len(chunks)*100:.1f}%)\n"
                f"  Oversized (>1500 chars)  : {oversized} ({oversized/len(chunks)*100:.1f}%)"
            ),
        )
    except Exception as e:
        print(f"\n[WARN] Chunking failed: {e}")

    # --- Summary grade ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    issues = []
    if missing_year / len(docs) > 0.1:
        issues.append(f"- {missing_year} docs missing nam_ban_hanh ({missing_year/len(docs)*100:.1f}%)")
    if missing_nganh / len(docs) > 0.1:
        issues.append(f"- {missing_nganh} docs missing nganh ({missing_nganh/len(docs)*100:.1f}%)")
    if empty_body_count > 0:
        issues.append(f"- {empty_body_count} files with empty body")
    if len(exact_dups) > 0:
        issues.append(f"- {len(exact_dups)} exact-duplicate groups")
    if has_header_count / n < 0.95:
        issues.append(
            f"- only {has_header_count/n*100:.1f}% of files have crawl-format header"
        )

    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ No major data quality issues.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
