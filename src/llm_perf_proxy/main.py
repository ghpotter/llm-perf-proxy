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
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import llm_perf_proxy.db as db
import llm_perf_proxy.worker as worker
from llm_perf_proxy.backends import MetricsRecord, OllamaBackend, build_backend
from llm_perf_proxy.backends.base import Backend

log = logging.getLogger("llm-proxy")

# Module-level reference set during lifespan so routes can call it.
active_backend: Backend | None = None


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
                 THEN prompt_eval_duration / 1e6 END)                     AS avg_prompt_ms,
        SUM(estimated_cost_usd)                                           AS total_cost_usd,
        AVG(estimated_cost_usd)                                           AS avg_cost_usd
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
        result = {}
        for k, v in dict(row).items():
            if isinstance(v, float):
                result[k] = round(v, 6) if k.endswith("_usd") else round(v, 2)
            else:
                result[k] = v
        return result

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
