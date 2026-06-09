"""
main.py — FastAPI application, lifespan, and routes
=====================================================
This file is intentionally thin: app setup, lifespan wiring, and route
handlers only. Business logic lives in the modules it imports.

    config.py          — environment variables
    backends/          — OllamaBackend, OpenAIBackend, AnthropicBackend
    db.py              — SQLite connection and persistence
    worker.py          — async metrics queue and background drain task
"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

import llm_perf_proxy.db as db
import llm_perf_proxy.worker as worker
from llm_perf_proxy.backends import MetricsRecord, OllamaBackend, build_backend
from llm_perf_proxy.backends.base import Backend

log = logging.getLogger("llm-proxy")

# Module-level reference set during lifespan so routes can call it.
active_backend: Backend | None = None

# ── Timeseries helpers ────────────────────────────────────────────────────────

_RELATIVE_RE = re.compile(r"^(\d+)([mhd])$", re.IGNORECASE)

_WINDOW_FMT: dict[str, str] = {
    "minute": "%Y-%m-%dT%H:%M:00",
    "hour": "%Y-%m-%dT%H:00:00",
    "day": "%Y-%m-%d",
}


def _parse_since(s: str) -> str:
    """Convert a relative shorthand (e.g. '24h', '7d') to an ISO-8601 string."""
    m = _RELATIVE_RE.match(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = {"m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n)}[unit]
        return (datetime.now(UTC) - delta).isoformat()
    return s


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_backend
    active_backend = build_backend()
    log.info("Active backend: %s", active_backend.name)

    await db.init()

    worker_task = asyncio.create_task(worker.run())
    yield

    # Graceful shutdown: cancel the worker, then flush remaining records
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await worker.queue.join()
    await db.close()


app = FastAPI(
    title="LLM Performance Proxy",
    description="Multi-backend observability proxy. Set BACKEND=ollama|openai|anthropic.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


async def _proxy_stream(endpoint: str, body: dict, request_headers: dict):
    """
    Delegates to the active backend's stream() generator.
    Bytes are forwarded to the client; the terminal MetricsRecord is
    intercepted and handed off to the worker queue.
    """
    async for item in active_backend.stream(endpoint, body, request_headers):
        if isinstance(item, MetricsRecord):
            worker.enqueue(item)
        else:
            yield item


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness check. Reports active backend and persisted record count."""
    backend_status = {"name": active_backend.name}

    if isinstance(active_backend, OllamaBackend):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{active_backend.base_url}/api/tags")
                backend_status["reachable"] = r.status_code == 200
        except Exception:
            backend_status["reachable"] = False
    else:
        backend_status["reachable"] = True  # remote APIs assumed reachable

    async with db.conn.execute("SELECT COUNT(*) FROM requests") as cur:
        row = await cur.fetchone()

    return {
        "proxy": "ok",
        "backend": backend_status,
        "queued_metrics": worker.queue.qsize(),
        "stored_records": row[0] if row else 0,
    }


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """
    Streaming proxy for the chat endpoint.

    Ollama:    { "model": "phi3",   "messages": [...] }
    OpenAI:    { "model": "gpt-4o", "messages": [...] }
    Anthropic: { "model": "claude-sonnet-4-20250514", "messages": [...], "max_tokens": 1024 }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    body["stream"] = True
    return StreamingResponse(
        _proxy_stream("chat", body, dict(request.headers)),
        media_type="application/x-ndjson",
    )


@app.post("/api/generate")
async def proxy_generate(request: Request):
    """
    Streaming proxy for Ollama's raw completion endpoint.
    Returns 501 for non-Ollama backends — they have no equivalent raw-completion
    API and silently forwarding an Ollama-style body would produce a confusing
    upstream error. Use /api/chat instead.
    """
    if not isinstance(active_backend, OllamaBackend):
        raise HTTPException(
            status_code=501,
            detail=(
                f"/api/generate is not supported for the '{active_backend.name}' backend. "
                "Use /api/chat instead."
            ),
        )
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    body["stream"] = True
    return StreamingResponse(
        _proxy_stream("generate", body, dict(request.headers)),
        media_type="application/x-ndjson",
    )


@app.get("/metrics/summary")
async def metrics_summary():
    """
    SQL-backed aggregated stats — survives restarts.

    Returns five slices of the same dataset:
        overall              — single global roll-up
        by_backend           — GROUP BY backend
        by_endpoint          — GROUP BY endpoint
        by_model             — GROUP BY model, sorted by avg_tps desc
        by_model_and_endpoint — nested cross-tab { model: { endpoint: stats } }

    For remote backends (openai/anthropic), eval_duration is NULL so TPS
    is calculated from wall_duration_ns — a conservative end-to-end figure
    that includes network RTT rather than pure generation speed.
    """
    _METRICS_COLS = """
        COUNT(*)                                                          AS requests,
        SUM(eval_count)                                                   AS total_tokens,
        AVG(CAST(eval_count AS REAL)
            / (COALESCE(eval_duration, wall_duration_ns) / 1e9))          AS avg_tps,
        MAX(CAST(eval_count AS REAL)
            / (COALESCE(eval_duration, wall_duration_ns) / 1e9))          AS peak_tps,
        AVG(wall_duration_ns / 1e6)                                       AS avg_wall_ms,
        AVG(CASE WHEN ttft_ns IS NOT NULL THEN ttft_ns / 1e6 END)         AS avg_ttft_ms,
        AVG(CASE WHEN total_duration IS NOT NULL
                 THEN total_duration / 1e6 END)                           AS avg_total_ms,
        AVG(CASE WHEN prompt_eval_duration IS NOT NULL
                 THEN prompt_eval_duration / 1e6 END)                     AS avg_prompt_ms
    """
    _WHERE = "WHERE eval_count > 0 AND COALESCE(eval_duration, wall_duration_ns) > 0"

    queries = {
        "by_backend": (
            f"SELECT backend,  {_METRICS_COLS} FROM requests {_WHERE} GROUP BY backend  ORDER BY backend",
            "backend",
        ),
        "by_endpoint": (
            f"SELECT endpoint, {_METRICS_COLS} FROM requests {_WHERE} GROUP BY endpoint ORDER BY endpoint",
            "endpoint",
        ),
        "by_model": (
            f"SELECT model,    {_METRICS_COLS} FROM requests {_WHERE} GROUP BY model    ORDER BY avg_tps DESC",
            "model",
        ),
    }

    def _round_row(row) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in dict(row).items()}

    async with db.conn.execute(f"SELECT {_METRICS_COLS} FROM requests {_WHERE}") as cur:
        global_row = await cur.fetchone()

    if not global_row or global_row["requests"] == 0:
        return {"message": "No metrics captured yet.", "total_requests": 0}

    result: dict = {"overall": _round_row(global_row)}

    for label, (sql, key) in queries.items():
        async with db.conn.execute(sql) as cur:
            rows = await cur.fetchall()
        result[label] = {row[key]: _round_row(row) for row in rows}

    async with db.conn.execute(
        f"SELECT model, endpoint, {_METRICS_COLS} FROM requests {_WHERE} "
        "GROUP BY model, endpoint ORDER BY model, endpoint"
    ) as cur:
        me_rows = await cur.fetchall()

    cross_tab: dict = {}
    for row in me_rows:
        cross_tab.setdefault(row["model"], {})[row["endpoint"]] = _round_row(row)

    result["by_model_and_endpoint"] = cross_tab
    result["queue_depth"] = worker.queue.qsize()
    return result


@app.get("/metrics/timeseries")
async def metrics_timeseries(
    since: str = Query("24h", description="ISO-8601 timestamp or relative shorthand (1h, 24h, 7d)"),
    window: str = Query("hour", description="Bucket size: minute | hour | day"),
    backend: str | None = Query(None, description="Filter to a specific backend"),
    model: str | None = Query(None, description="Filter to a specific model"),
):
    """
    Time-bucketed performance stats. Returns one row per bucket within the
    requested window, ordered oldest-first (suitable for chart rendering).

    Bucket timestamps follow strftime output:
        minute → "2026-06-09T14:35:00"
        hour   → "2026-06-09T14:00:00"
        day    → "2026-06-09"
    """
    if window not in _WINDOW_FMT:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window '{window}'. Use: {', '.join(_WINDOW_FMT)}.",
        )

    fmt = _WINDOW_FMT[window]
    since_ts = _parse_since(since)

    conditions = [
        "eval_count > 0",
        "COALESCE(eval_duration, wall_duration_ns) > 0",
        "timestamp >= ?",
    ]
    params: list = [since_ts]

    if backend:
        conditions.append("backend = ?")
        params.append(backend)
    if model:
        conditions.append("model = ?")
        params.append(model)

    where_clause = " AND ".join(conditions)

    # fmt comes only from _WINDOW_FMT — not user input — so f-string is safe.
    sql = f"""
        SELECT
            strftime('{fmt}', timestamp)                                            AS ts,
            COUNT(*)                                                                AS requests,
            SUM(eval_count)                                                         AS total_tokens,
            AVG(CAST(eval_count AS REAL)
                / (COALESCE(eval_duration, wall_duration_ns) / 1e9))                AS avg_tps,
            MAX(CAST(eval_count AS REAL)
                / (COALESCE(eval_duration, wall_duration_ns) / 1e9))                AS peak_tps,
            AVG(wall_duration_ns / 1e6)                                             AS avg_wall_ms,
            AVG(CASE WHEN ttft_ns IS NOT NULL THEN ttft_ns / 1e6 END)               AS avg_ttft_ms
        FROM requests
        WHERE {where_clause}
        GROUP BY ts
        ORDER BY ts ASC
    """

    async with db.conn.execute(sql, params) as cur:
        rows = await cur.fetchall()

    def _round_bucket(row) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in dict(row).items()}

    return {
        "window": window,
        "since": since_ts,
        "filters": {"backend": backend, "model": model},
        "buckets": [_round_bucket(r) for r in rows],
    }
