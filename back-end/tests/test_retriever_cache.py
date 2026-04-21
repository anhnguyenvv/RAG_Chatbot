"""Tests for RetrieverManager in-memory TTL query cache."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag.retriever import RetrieverManager


def _make_manager(enabled: bool = True, ttl: int = 600, maxsize: int = 500) -> RetrieverManager:
    backend_config = SimpleNamespace(
        retrieval_top_k=5,
        retrieval_cache_enabled=enabled,
        retrieval_cache_ttl=ttl,
        retrieval_cache_maxsize=maxsize,
    )
    pipeline_config = SimpleNamespace(
        qdrant_url="http://dummy",
        qdrant_api_key="dummy",
        collection_name="dummy",
    )
    return RetrieverManager(pipeline_config, backend_config, embeddings=object())


def test_cache_hit_skips_retriever_call():
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc-a", "doc-b"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    first = manager.retrieve("qdrant", "chuong trinh CNTT")
    second = manager.retrieve("qdrant", "chuong trinh CNTT")

    assert first == ["doc-a", "doc-b"]
    assert second == first
    assert fake_retriever.invoke.call_count == 1


def test_cache_key_differs_by_filter():
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc-no-filter"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)
    manager._retrieve_with_filter = MagicMock(return_value=["doc-filtered"])

    a = manager.retrieve("qdrant", "abc")
    b = manager.retrieve("qdrant", "abc", metadata_filter={"nganh": "CNTT"})
    c = manager.retrieve("qdrant", "abc", metadata_filter={"nganh": "CNTT"})

    assert a == ["doc-no-filter"]
    assert b == ["doc-filtered"]
    assert c == b
    assert fake_retriever.invoke.call_count == 1
    assert manager._retrieve_with_filter.call_count == 1


def test_cache_respects_source_normalization():
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", "abc")
    manager.retrieve("auto", "abc")
    manager.retrieve("fit_web", "abc")

    # "auto" and "fit_web" normalize to "qdrant" → single cache entry
    assert fake_retriever.invoke.call_count == 1


def test_cache_disabled_always_hits_retriever():
    manager = _make_manager(enabled=False)
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", "abc")
    manager.retrieve("qdrant", "abc")

    assert fake_retriever.invoke.call_count == 2
    assert manager._query_cache is None


def test_clear_query_cache():
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", "abc")
    manager.clear_query_cache()
    manager.retrieve("qdrant", "abc")

    assert fake_retriever.invoke.call_count == 2


def test_ttl_expiry_causes_refetch():
    manager = _make_manager(ttl=60)
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", "abc")
    # Fast-forward beyond TTL by running expire() with a future timestamp.
    manager._query_cache.expire(manager._query_cache.timer() + 120)
    manager.retrieve("qdrant", "abc")

    assert fake_retriever.invoke.call_count == 2


@pytest.mark.parametrize("bad_query", ["", "   ", "\t\n"])
def test_empty_or_whitespace_query_bypasses_cache(bad_query):
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["doc"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", bad_query)
    manager.retrieve("qdrant", bad_query)

    # Whitespace-only queries should never populate the cache.
    assert fake_retriever.invoke.call_count == 2
    assert len(manager._query_cache) == 0


def test_cache_hit_returns_isolated_copy():
    """Mutating the returned list must not corrupt the cached entry."""
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["a", "b", "c"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    first = manager.retrieve("qdrant", "abc")
    first.append("MUTATED")
    first.sort(reverse=True)

    second = manager.retrieve("qdrant", "abc")
    assert second == ["a", "b", "c"]
    assert second is not first


def test_zero_ttl_disables_cache():
    manager = _make_manager(ttl=0)
    assert manager._query_cache is None


def test_zero_maxsize_disables_cache():
    manager = _make_manager(maxsize=0)
    assert manager._query_cache is None


def test_maxsize_eviction():
    manager = _make_manager(maxsize=2)
    fake_retriever = MagicMock()
    fake_retriever.invoke.side_effect = lambda q: [f"doc-{q}"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", "q1")
    manager.retrieve("qdrant", "q2")
    manager.retrieve("qdrant", "q3")  # evicts LRU

    assert len(manager._query_cache) == 2
    # q1 was evicted — re-fetching it must hit the retriever again.
    manager.retrieve("qdrant", "q1")
    assert fake_retriever.invoke.call_count == 4


def test_clear_query_cache_returns_evicted_count():
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.side_effect = lambda q: [f"doc-{q}"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    manager.retrieve("qdrant", "q1")
    manager.retrieve("qdrant", "q2")
    evicted = manager.clear_query_cache()
    assert evicted == 2
    assert manager.clear_query_cache() == 0  # already empty


def test_clear_query_cache_when_disabled():
    manager = _make_manager(enabled=False)
    assert manager.clear_query_cache() == 0


def test_cache_stats_enabled_and_disabled():
    manager = _make_manager(ttl=120, maxsize=10)
    stats = manager.cache_stats()
    assert stats == {"enabled": True, "size": 0, "maxsize": 10, "ttl": 120}

    disabled = _make_manager(enabled=False)
    assert disabled.cache_stats() == {"enabled": False, "size": 0, "maxsize": 0, "ttl": 0}


def test_concurrent_retrieve_is_thread_safe():
    """Concurrent identical queries must not exceed one upstream call per miss
    and must not raise under lock contention."""
    manager = _make_manager()
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = ["shared"]
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    # Warm the cache so all concurrent callers hit it (exercises the read lock).
    manager.retrieve("qdrant", "hotpath")

    def worker():
        return manager.retrieve("qdrant", "hotpath")

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(worker) for _ in range(64)]
        results = [f.result() for f in as_completed(futures)]

    assert all(r == ["shared"] for r in results)
    # Only the warm-up call should have hit the retriever.
    assert fake_retriever.invoke.call_count == 1


# --- T05: single-flight coalescing --------------------------------------


def test_single_flight_coalesces_concurrent_misses():
    """N concurrent misses for the same key trigger exactly 1 upstream call."""
    import threading as _th

    manager = _make_manager()
    fake_retriever = MagicMock()

    start_gate = _th.Event()
    compute_done = _th.Event()
    compute_started = _th.Event()
    call_counter = {"n": 0}
    counter_lock = _th.Lock()

    def slow_invoke(query):
        with counter_lock:
            call_counter["n"] += 1
        compute_started.set()
        # Block until main thread has dispatched all waiters.
        start_gate.wait(timeout=5.0)
        compute_done.set()
        return ["doc-coalesced"]

    fake_retriever.invoke.side_effect = slow_invoke
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    # Kick off 32 concurrent callers for same key. First one wins, others coalesce.
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(manager.retrieve, "qdrant", "coalesce-me") for _ in range(32)]
        # Wait for the winner to enter compute, then let all complete.
        compute_started.wait(timeout=5.0)
        start_gate.set()
        results = [f.result(timeout=10.0) for f in futures]

    assert all(r == ["doc-coalesced"] for r in results)
    # Exactly 1 upstream call despite 32 concurrent callers.
    assert call_counter["n"] == 1
    # Inflight map must be drained after winner completes.
    assert manager._inflight == {}


def test_single_flight_different_keys_run_parallel():
    """Different keys must NOT block each other."""
    import threading as _th

    manager = _make_manager()
    fake_retriever = MagicMock()
    barrier = _th.Barrier(3, timeout=5.0)

    def invoke_with_barrier(query):
        barrier.wait()  # each distinct query must reach barrier independently
        return [f"doc-{query}"]

    fake_retriever.invoke.side_effect = invoke_with_barrier
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(manager.retrieve, "qdrant", f"key-{i}") for i in range(3)]
        results = [f.result(timeout=10.0) for f in futures]

    # If single-flight blocked different keys, the barrier would time out.
    assert sorted(r[0] for r in results) == ["doc-key-0", "doc-key-1", "doc-key-2"]
    assert fake_retriever.invoke.call_count == 3


def test_single_flight_winner_failure_does_not_deadlock_waiters():
    """If winner's upstream call raises, waiters must still progress (not deadlock)."""
    import threading as _th

    manager = _make_manager()
    fake_retriever = MagicMock()

    attempt_lock = _th.Lock()
    attempts = {"n": 0}

    def maybe_fail(query):
        with attempt_lock:
            attempts["n"] += 1
            n = attempts["n"]
        if n == 1:
            raise RuntimeError("boom")
        return ["doc-recovered"]

    fake_retriever.invoke.side_effect = maybe_fail
    manager.get_retriever = MagicMock(return_value=fake_retriever)

    # First call: winner fails. Second call: succeeds after fallthrough.
    with pytest.raises(RuntimeError, match="boom"):
        manager.retrieve("qdrant", "flaky")

    # Inflight cleaned up even on exception.
    assert manager._inflight == {}

    # Subsequent call should succeed.
    result = manager.retrieve("qdrant", "flaky")
    assert result == ["doc-recovered"]
