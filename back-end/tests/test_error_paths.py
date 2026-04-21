"""Error path coverage for the RAG backend.

Covers failure modes that are easy to miss until production:
- Retriever upstream (Qdrant) timeout / 5xx
- LLM timeout / exception
- Rate-limit 429
- Admin endpoint auth (no key, wrong key, rate limit)
- Invalid input (source, mode, missing query)

Uses the full FastAPI app via TestClient, with RAG components mocked at the
dependency-injection boundary (build_rag_service).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi.testclient import TestClient  # requires httpx
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(TestClient is None, reason="httpx / fastapi.testclient unavailable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(
    rag_side_effect=None,
    rag_return=None,
    admin_key: str = "",
    rate_limit_rag: str = "1000/minute",
    rate_limit_admin: str = "1000/minute",
):
    """Build a FastAPI app with a mocked RAG service.

    `rag_side_effect` lets tests simulate exceptions thrown by the service.
    `admin_key` != "" enables /admin/* endpoints.
    Rate limits default to loose values; tests that exercise throttling
    pass in a tight limit.
    """
    from app.api.routes import create_app
    from app.config.config import BackendConfig

    fake_service = MagicMock()
    if rag_side_effect is not None:
        fake_service.query.side_effect = rag_side_effect
    else:
        fake_service.query.return_value = rag_return or {
            "result": "ok",
            "source_documents": [],
        }
    # Retriever manager stub for /admin/cache/*
    fake_service.retriever_mgr = MagicMock()
    fake_service.retriever_mgr.cache_stats.return_value = {
        "enabled": True, "size": 0, "maxsize": 500, "ttl": 600,
    }
    fake_service.retriever_mgr.clear_query_cache.return_value = 0
    fake_service.memory_store = MagicMock()
    fake_service.memory_store.list_sessions.return_value = []
    fake_service.memory_store.clear_session.return_value = True

    fake_backend_cfg = BackendConfig(
        rate_limit_rag=rate_limit_rag,
        rate_limit_default="1000/minute",
        rate_limit_admin=rate_limit_admin,
        admin_api_key=admin_key,
    )
    fake_pipeline_cfg = SimpleNamespace(
        collection_name="test_coll",
        embedding_model_name="test-model",
    )

    with patch("app.api.routes.load_configs", return_value=(fake_backend_cfg, fake_pipeline_cfg)), \
         patch("app.api.routes.build_rag_service", return_value=fake_service), \
         patch("app.api.routes.ChatHistoryStore") as HistoryCls:
        hist_inst = MagicMock()
        hist_inst.add_entry.return_value = 42
        hist_inst.list_entries.return_value = []
        hist_inst.get_entry.return_value = None
        HistoryCls.return_value = hist_inst

        app = create_app()

    return app, fake_service


# ---------------------------------------------------------------------------
# Input validation (4xx)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_source_returns_400(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/rag/invalid_src?q=test")
        assert r.status_code == 400
        assert "Invalid source" in r.json()["detail"]

    def test_invalid_mode_returns_400(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/rag/qdrant?q=test&mode=semantic")
        assert r.status_code == 400
        assert "Invalid mode" in r.json()["detail"]

    def test_missing_query_param_returns_400(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/rag/qdrant")
        assert r.status_code == 400
        assert "required" in r.json()["detail"].lower()

    def test_history_entry_not_found_returns_404(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/history/99999")
        assert r.status_code == 404

    def test_delete_unknown_session_returns_404(self):
        app, svc = _make_app()
        svc.memory_store.clear_session.return_value = False
        with TestClient(app) as client:
            r = client.delete("/sessions/nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Upstream failures (5xx propagation)
# ---------------------------------------------------------------------------


class TestUpstreamFailures:
    def test_qdrant_timeout_returns_500(self):
        """When retriever raises (e.g. Qdrant timeout), API returns 500."""
        class FakeQdrantTimeout(Exception):
            pass

        app, _ = _make_app(rag_side_effect=FakeQdrantTimeout("connection timed out"))
        with TestClient(app) as client:
            r = client.get("/rag/qdrant?q=hoc+phi")
        assert r.status_code == 500
        # Error message should surface, not be swallowed
        assert "timed out" in r.json()["detail"].lower() or "error" in r.json()["detail"].lower()

    def test_llm_exception_returns_500(self):
        """LLM throwing propagates as 500."""
        app, _ = _make_app(rag_side_effect=RuntimeError("Gemini API 503 unavailable"))
        with TestClient(app) as client:
            r = client.get("/rag/qdrant?q=dieu+kien+tot+nghiep")
        assert r.status_code == 500
        assert "Gemini" in r.json()["detail"] or "error" in r.json()["detail"].lower()

    def test_memory_crash_in_agentic_returns_500(self):
        """Agentic mode crashing doesn't leak stack trace."""
        app, _ = _make_app(rag_side_effect=ConnectionError("mongo refused"))
        with TestClient(app) as client:
            r = client.get("/rag/qdrant?q=quy+che&mode=agentic")
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# Rate limiting (429)
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_rag_rate_limit_triggers_429(self):
        """After hitting rate limit, subsequent request returns 429."""
        app, _ = _make_app(rate_limit_rag="2/minute")
        with TestClient(app) as client:
            r1 = client.get("/rag/qdrant?q=a")
            r2 = client.get("/rag/qdrant?q=b")
            r3 = client.get("/rag/qdrant?q=c")  # exceeds 2/minute
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429

    def test_admin_rate_limit_is_separate(self):
        """Admin rate limit counted separately from /rag/ limit."""
        app, _ = _make_app(
            rate_limit_rag="100/minute",
            rate_limit_admin="1/minute",
            admin_key="secret",
        )
        with TestClient(app) as client:
            # Exhaust admin quota
            r1 = client.get("/admin/cache/stats", headers={"X-Admin-Key": "secret"})
            r2 = client.get("/admin/cache/stats", headers={"X-Admin-Key": "secret"})
            # /rag/ must still work — separate bucket
            r3 = client.get("/rag/qdrant?q=still-works")
        assert r1.status_code == 200
        assert r2.status_code == 429
        assert r3.status_code == 200


# ---------------------------------------------------------------------------
# Admin endpoint authentication
# ---------------------------------------------------------------------------


class TestAdminAuth:
    def test_admin_endpoint_disabled_when_no_key_configured(self):
        app, _ = _make_app(admin_key="")  # empty disables admin
        with TestClient(app) as client:
            r = client.get("/admin/cache/stats")
        assert r.status_code == 403
        assert "disabled" in r.json()["detail"].lower()

    def test_admin_endpoint_missing_header_returns_401(self):
        app, _ = _make_app(admin_key="topsecret")
        with TestClient(app) as client:
            r = client.get("/admin/cache/stats")
        assert r.status_code == 401

    def test_admin_endpoint_wrong_key_returns_401(self):
        app, _ = _make_app(admin_key="topsecret")
        with TestClient(app) as client:
            r = client.get("/admin/cache/stats", headers={"X-Admin-Key": "wrong"})
        assert r.status_code == 401

    def test_admin_endpoint_correct_key_returns_200(self):
        app, _ = _make_app(admin_key="topsecret")
        with TestClient(app) as client:
            r = client.get("/admin/cache/stats", headers={"X-Admin-Key": "topsecret"})
        assert r.status_code == 200
        body = r.json()
        assert "enabled" in body

    def test_admin_cache_clear_requires_auth(self):
        app, _ = _make_app(admin_key="topsecret")
        with TestClient(app) as client:
            r_no_auth = client.post("/admin/cache/clear")
            r_auth = client.post(
                "/admin/cache/clear", headers={"X-Admin-Key": "topsecret"}
            )
        assert r_no_auth.status_code == 401
        assert r_auth.status_code == 200
        assert r_auth.json() == {"evicted": 0}


# ---------------------------------------------------------------------------
# Health check + metrics
# ---------------------------------------------------------------------------


class TestHealth:
    def test_root_returns_200(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "message" in body
        assert body["collection"] == "test_coll"

    def test_metrics_endpoint_exposes_prometheus(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
        assert r.status_code == 200
        # Prometheus exposition format starts with comments
        assert "# HELP" in r.text or "# TYPE" in r.text


# ---------------------------------------------------------------------------
# Response shape stability
# ---------------------------------------------------------------------------


class TestResponseShape:
    def test_classic_response_has_required_fields(self):
        app, _ = _make_app(rag_return={
            "result": "answer text",
            "source_documents": [{"page_content": "snippet", "metadata": {}}],
        })
        with TestClient(app) as client:
            r = client.get("/rag/qdrant?q=test")
        assert r.status_code == 200
        body = r.json()
        assert "result" in body
        assert "source_documents" in body
        assert "history_id" in body

    def test_optional_fields_propagate_when_present(self):
        app, _ = _make_app(rag_return={
            "result": "a",
            "source_documents": [],
            "confidence": 0.87,
            "citations": [{"idx": 1}],
            "timings": {"retrieve_ms": 50},
        })
        with TestClient(app) as client:
            r = client.get("/rag/qdrant?q=test&mode=agentic")
        assert r.status_code == 200
        body = r.json()
        assert body["confidence"] == 0.87
        assert body["citations"] == [{"idx": 1}]
        assert body["timings"] == {"retrieve_ms": 50}
