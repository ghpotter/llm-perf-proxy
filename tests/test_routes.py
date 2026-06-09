"""
tests/test_routes.py
====================
Smoke tests for the FastAPI routes using an in-memory DB and a stubbed backend.
These confirm that the app wires up correctly and routes return the right shapes
without needing a live LLM or network.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

import llm_perf_proxy.db as db
import llm_perf_proxy.worker as worker
from llm_perf_proxy import main
from llm_perf_proxy.backends.base import MetricsRecord
from llm_perf_proxy.backends.ollama import OllamaBackend

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def test_app(monkeypatch):
    """
    Stand up the FastAPI app with:
      - an in-memory SQLite database
      - a no-op metrics worker (so tests don't need a running event loop task)
      - OllamaBackend set as the active backend
    """
    # In-memory DB
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await db._migrate(conn)
    monkeypatch.setattr(db, "conn", conn)

    # Active backend
    monkeypatch.setattr(main, "active_backend", OllamaBackend())

    yield main.app

    await conn.close()


@pytest.fixture
async def client(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac


# ── /health ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    with patch("llm_perf_proxy.main.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
        mock_cls.return_value = mock_http

        resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["proxy"] == "ok"
    assert "backend" in body
    assert "stored_records" in body


@pytest.mark.asyncio
async def test_health_stored_records_zero_initially(client):
    with patch("llm_perf_proxy.main.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
        mock_cls.return_value = mock_http

        resp = await client.get("/health")

    assert resp.json()["stored_records"] == 0


# ── /metrics/summary ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_summary_empty(client):
    resp = await client.get("/metrics/summary")
    assert resp.status_code == 200
    assert resp.json()["total_requests"] == 0


@pytest.mark.asyncio
async def test_metrics_summary_with_data(client, monkeypatch):
    # Seed two records directly into the in-memory DB
    for i in range(2):
        await db.insert(
            MetricsRecord(
                timestamp=f"2026-06-08T00:0{i}:00Z",
                backend="ollama",
                endpoint="chat",
                model="phi3",
                wall_duration_ns=5_000_000_000,
                ttft_ns=None,
                total_duration=None,
                load_duration=None,
                prompt_eval_count=10,
                prompt_eval_duration=None,
                eval_count=100,
                eval_duration=4_000_000_000,
            )
        )

    resp = await client.get("/metrics/summary")
    assert resp.status_code == 200
    body = resp.json()

    assert body["overall"]["requests"] == 2
    assert body["overall"]["total_tokens"] == 200
    assert "by_backend" in body
    assert "by_model" in body
    assert "ollama" in body["by_backend"]


# ── /api/chat ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_bad_json_returns_400(client):
    resp = await client.post(
        "/api/chat",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_generate_bad_json_returns_400(client):
    resp = await client.post(
        "/api/generate",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_generate_non_ollama_backend_returns_501(test_app, monkeypatch):
    mock_backend = MagicMock()
    mock_backend.name = "openai"
    monkeypatch.setattr(main, "active_backend", mock_backend)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        resp = await ac.post(
            "/api/generate",
            json={"model": "gpt-4o", "prompt": "Why is the sky blue?"},
        )

    assert resp.status_code == 501
    assert "openai" in resp.json()["detail"]
    assert "/api/chat" in resp.json()["detail"]


# ── /metrics/timeseries ───────────────────────────────────────────────────────


def _ts_record(timestamp, eval_count, eval_duration, backend="ollama", model="phi3"):
    return MetricsRecord(
        timestamp=timestamp,
        backend=backend,
        endpoint="chat",
        model=model,
        wall_duration_ns=5_000_000_000,
        ttft_ns=None,
        total_duration=None,
        load_duration=None,
        prompt_eval_count=10,
        prompt_eval_duration=None,
        eval_count=eval_count,
        eval_duration=eval_duration,
    )


@pytest.mark.asyncio
async def test_timeseries_hour_buckets(client):
    # 2 records in hour 10, 1 in hour 11
    for ts, ec, ed in [
        ("2026-06-09T10:00:00Z", 100, 4_000_000_000),
        ("2026-06-09T10:30:00Z", 80, 3_000_000_000),
        ("2026-06-09T11:15:00Z", 120, 5_000_000_000),
    ]:
        await db.insert(_ts_record(ts, ec, ed))

    resp = await client.get(
        "/metrics/timeseries",
        params={"since": "2026-06-09T00:00:00Z", "window": "hour"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["window"] == "hour"
    assert len(body["buckets"]) == 2

    bucket_10 = next(b for b in body["buckets"] if "T10:" in b["ts"])
    assert bucket_10["requests"] == 2
    assert bucket_10["total_tokens"] == 180


@pytest.mark.asyncio
async def test_timeseries_since_excludes_old_records(client):
    await db.insert(_ts_record("2020-01-01T00:00:00Z", 100, 4_000_000_000))

    resp = await client.get(
        "/metrics/timeseries",
        params={"since": "2025-01-01T00:00:00Z", "window": "day"},
    )
    assert resp.status_code == 200
    assert resp.json()["buckets"] == []


@pytest.mark.asyncio
async def test_timeseries_backend_filter(client):
    await db.insert(_ts_record("2026-06-09T10:00:00Z", 100, 4_000_000_000, backend="ollama"))
    await db.insert(_ts_record("2026-06-09T10:00:00Z", 80, None, backend="openai", model="gpt-4o"))

    resp = await client.get(
        "/metrics/timeseries",
        params={"since": "2026-06-09T00:00:00Z", "window": "hour", "backend": "ollama"},
    )
    assert resp.status_code == 200
    buckets = resp.json()["buckets"]
    assert len(buckets) == 1
    assert buckets[0]["requests"] == 1


@pytest.mark.asyncio
async def test_timeseries_invalid_window_returns_400(client):
    resp = await client.get("/metrics/timeseries", params={"window": "week"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_timeseries_relative_since_accepted(client):
    resp = await client.get("/metrics/timeseries", params={"since": "24h"})
    assert resp.status_code == 200
    assert "buckets" in resp.json()
