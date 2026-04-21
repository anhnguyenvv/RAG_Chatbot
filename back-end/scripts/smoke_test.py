#!/usr/bin/env python3
"""
Smoke test for RAG Chatbot FIT-HCMUS deployment.

Runs all smoke test cases from deploy-checklist.md §5 against a running backend.
Exit code 0 = all pass, 1 = any fail.

Usage:
    # Local
    python scripts/smoke_test.py

    # Staging, with admin key
    python scripts/smoke_test.py \\
        --base-url https://staging.fit-chatbot.example.com \\
        --admin-key "$ADMIN_API_KEY"

    # CI-friendly JSON output
    python scripts/smoke_test.py --json > smoke-result.json

Env vars (optional):
    SMOKE_BASE_URL     default: http://localhost:8000
    SMOKE_ADMIN_KEY    required for /admin/* checks
    SMOKE_TIMEOUT      default: 30 (seconds)

Dependencies:
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("ERROR: missing dependency 'requests'. Install with: pip install requests\n")
    sys.exit(2)


# ---------------------------------------------------------------------------
# ANSI colors (auto-disable when not tty)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
GREEN = "\033[92m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
YELLOW = "\033[93m" if _USE_COLOR else ""
BLUE = "\033[94m" if _USE_COLOR else ""
GREY = "\033[90m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class SmokeConfig:
    base_url: str
    admin_key: Optional[str]
    timeout: int
    session_id: str
    skip_rate_limit: bool


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    detail: str = ""
    error: Optional[str] = None
    skipped: bool = False


CheckFn = Callable[[SmokeConfig], Tuple[bool, str]]


# ---------------------------------------------------------------------------
# Individual checks — each returns (passed, detail_string)
# ---------------------------------------------------------------------------
def _has_answer_like(data: dict) -> bool:
    """RAG endpoints may return {answer|response|text|result|content|...}."""
    if not isinstance(data, dict):
        return False
    for key in ("answer", "response", "text", "result", "content", "message"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return True
    # Some impls nest the answer under "data"
    if isinstance(data.get("data"), dict):
        return _has_answer_like(data["data"])
    return False


def check_health(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(f"{cfg.base_url}/", timeout=cfg.timeout)
    return r.status_code == 200, f"status={r.status_code}"


def check_rag_classic(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(
        f"{cfg.base_url}/rag/qdrant",
        params={"q": "Nganh CNTT hoc may nam", "mode": "classic"},
        timeout=cfg.timeout,
    )
    if r.status_code != 200:
        return False, f"status={r.status_code}, body={r.text[:200]}"
    try:
        data = r.json()
    except ValueError:
        return False, "response is not JSON"
    return _has_answer_like(data), f"status=200, payload_bytes={len(r.content)}"


def check_rag_agentic(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(
        f"{cfg.base_url}/rag/qdrant",
        params={
            "q": "Nganh CNTT co may chuyen nganh?",
            "mode": "agentic",
            "session_id": cfg.session_id,
        },
        timeout=cfg.timeout * 2,  # agentic can iterate up to 5x
    )
    if r.status_code != 200:
        return False, f"status={r.status_code}, body={r.text[:200]}"
    try:
        data = r.json()
    except ValueError:
        return False, "response is not JSON"
    return _has_answer_like(data), f"status=200, session={cfg.session_id}"


def check_rag_fit_web(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(
        f"{cfg.base_url}/rag/fit_web",
        params={
            "q": "Lich dang ky hoc phan",
            "mode": "agentic",
            "session_id": cfg.session_id,
        },
        timeout=cfg.timeout * 2,
    )
    if r.status_code != 200:
        return False, f"status={r.status_code}, body={r.text[:200]}"
    return True, "status=200"


def check_sessions_list(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(f"{cfg.base_url}/sessions", timeout=cfg.timeout)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    found = cfg.session_id in r.text
    return found, f"session_id {'FOUND' if found else 'NOT found'} in /sessions"


def check_session_delete(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.delete(
        f"{cfg.base_url}/sessions/{cfg.session_id}", timeout=cfg.timeout
    )
    return r.status_code in (200, 204), f"status={r.status_code}"


def check_metrics(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(f"{cfg.base_url}/metrics", timeout=cfg.timeout)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    is_prom = ("# HELP" in r.text) or ("# TYPE" in r.text)
    return is_prom, f"status=200, size={len(r.text)}B, prom_format={is_prom}"


def check_admin_no_key(cfg: SmokeConfig) -> Tuple[bool, str]:
    r = requests.get(f"{cfg.base_url}/admin/cache/stats", timeout=cfg.timeout)
    # Without X-Admin-Key, must NOT return 200
    return r.status_code in (401, 403), f"status={r.status_code} (expect 401/403)"


def check_admin_with_key(cfg: SmokeConfig) -> Tuple[bool, str]:
    if not cfg.admin_key:
        raise RuntimeError("SKIPPED: --admin-key not provided")
    r = requests.get(
        f"{cfg.base_url}/admin/cache/stats",
        headers={"X-Admin-Key": cfg.admin_key},
        timeout=cfg.timeout,
    )
    return r.status_code == 200, f"status={r.status_code}"


def check_rate_limit(cfg: SmokeConfig) -> Tuple[bool, str]:
    """
    RATE_LIMIT_RAG=10/minute. Fire 11 fast requests and expect a 429.
    We hit classic mode with a throwaway query to keep latency down.
    """
    statuses = []
    hit_429 = False
    first_429_at = None
    for i in range(1, 12):
        try:
            r = requests.get(
                f"{cfg.base_url}/rag/qdrant",
                params={"q": f"smoke-ratelimit-{i}-{uuid.uuid4().hex[:6]}", "mode": "classic"},
                timeout=cfg.timeout,
            )
            statuses.append(r.status_code)
            if r.status_code == 429 and not hit_429:
                hit_429 = True
                first_429_at = i
                break
        except requests.exceptions.Timeout:
            statuses.append("timeout")
    detail = f"seen 429 at req #{first_429_at}" if hit_429 else f"no 429 in 11 reqs, statuses={statuses}"
    return hit_429, detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_check(name: str, fn: CheckFn, cfg: SmokeConfig) -> TestResult:
    start = time.time()
    try:
        passed, detail = fn(cfg)
        return TestResult(name, passed, (time.time() - start) * 1000, detail=detail)
    except RuntimeError as e:
        # Convention: RuntimeError with 'SKIPPED' → soft skip (counted as pass)
        msg = str(e)
        if msg.startswith("SKIPPED"):
            return TestResult(name, True, (time.time() - start) * 1000, detail=msg, skipped=True)
        return TestResult(name, False, (time.time() - start) * 1000, error=msg)
    except requests.exceptions.ConnectionError as e:
        return TestResult(name, False, (time.time() - start) * 1000, error=f"connection refused: {e}")
    except requests.exceptions.Timeout:
        return TestResult(name, False, (time.time() - start) * 1000, error="request timed out")
    except Exception as e:
        return TestResult(name, False, (time.time() - start) * 1000, error=f"{type(e).__name__}: {e}")


def build_test_plan(cfg: SmokeConfig):
    # Order matters: agentic creates session → list → fit_web → delete → rate limit last
    plan = [
        ("GET /  (health)", check_health),
        ("GET /rag/qdrant ?mode=classic", check_rag_classic),
        ("GET /rag/qdrant ?mode=agentic (creates session)", check_rag_agentic),
        ("GET /rag/fit_web ?mode=agentic", check_rag_fit_web),
        ("GET /sessions (contains our session)", check_sessions_list),
        ("GET /metrics (Prometheus format)", check_metrics),
        ("GET /admin/cache/stats without key → 401/403", check_admin_no_key),
        ("GET /admin/cache/stats with key → 200", check_admin_with_key),
        ("DELETE /sessions/{id}", check_session_delete),
    ]
    if not cfg.skip_rate_limit:
        plan.append(("Rate limit 10/min on /rag/qdrant → 429", check_rate_limit))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for RAG Chatbot FIT-HCMUS deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SMOKE_BASE_URL", "http://localhost:8000"),
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--admin-key",
        default=os.environ.get("SMOKE_ADMIN_KEY"),
        help="X-Admin-Key for /admin/* endpoints",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("SMOKE_TIMEOUT", "30")),
        help="Per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--session-id",
        default=f"smoke-{uuid.uuid4().hex[:8]}",
        help="Session ID used for agentic test (default: smoke-<random>)",
    )
    parser.add_argument(
        "--skip-rate-limit",
        action="store_true",
        help="Skip rate-limit test (saves ~6s but reduces coverage)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON result to stdout (CI-friendly)",
    )
    args = parser.parse_args()

    cfg = SmokeConfig(
        base_url=args.base_url.rstrip("/"),
        admin_key=args.admin_key,
        timeout=args.timeout,
        session_id=args.session_id,
        skip_rate_limit=args.skip_rate_limit,
    )

    if not args.json:
        print(f"{BOLD}{BLUE}RAG Chatbot FIT-HCMUS — Smoke Test{RESET}")
        print(f"  {GREY}Base URL   :{RESET} {cfg.base_url}")
        print(f"  {GREY}Session ID :{RESET} {cfg.session_id}")
        print(f"  {GREY}Admin key  :{RESET} {'yes' if cfg.admin_key else f'{YELLOW}no (admin-with-key will skip){RESET}'}")
        print(f"  {GREY}Timeout    :{RESET} {cfg.timeout}s")
        print(f"  {GREY}Rate-limit :{RESET} {'SKIPPED' if cfg.skip_rate_limit else 'enabled'}")
        print()

    plan = build_test_plan(cfg)
    results: list[TestResult] = []

    for name, fn in plan:
        if not args.json:
            sys.stdout.write(f"  {name:<52} ... ")
            sys.stdout.flush()
        res = run_check(name, fn, cfg)
        results.append(res)
        if not args.json:
            if res.skipped:
                status = f"{YELLOW}SKIP{RESET}"
            elif res.passed:
                status = f"{GREEN}PASS{RESET}"
            else:
                status = f"{RED}FAIL{RESET}"
            tail = res.detail if res.passed else (res.error or res.detail)
            print(f"{status} {GREY}({res.duration_ms:>6.0f}ms){RESET}  {tail}")

    # Tally (skipped counts as passed for exit-code purposes)
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)

    if args.json:
        out = {
            "base_url": cfg.base_url,
            "session_id": cfg.session_id,
            "total": total,
            "passed": passed,
            "failed": len(failed),
            "skipped": skipped,
            "all_passed": not failed,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "skipped": r.skipped,
                    "duration_ms": round(r.duration_ms, 1),
                    "detail": r.detail,
                    "error": r.error,
                }
                for r in results
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print()
        summary_color = GREEN if not failed else RED
        print(f"{BOLD}{summary_color}Result: {passed}/{total} passed"
              + (f" ({skipped} skipped)" if skipped else "")
              + RESET)
        if failed:
            print(f"\n{BOLD}{RED}Failed checks:{RESET}")
            for r in failed:
                print(f"  - {r.name}")
                print(f"      {r.error or r.detail}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
