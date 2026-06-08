"""
LLM Performance & Cost Observability Proxy
==========================================
A zero-overhead async reverse proxy that sits in front of Ollama,
streams responses to the client, and captures performance metrics
asynchronously via a background worker queue.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
METRICS_QUEUE_MAX_SIZE = 1000  # back-pressure limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("llm-proxy")

# ---------------------------------------------------------------------------
# In-memory metrics store
# TODO: Replace with SQLite/Postgres later
# ---------------------------------------------------------------------------

metrics_store: list[dict] = []


# ---------------------------------------------------------------------------
# Background metrics worker
# ---------------------------------------------------------------------------

metrics_queue: asyncio.Queue = asyncio.Queue(maxsize=METRICS_QUEUE_MAX_SIZE)


async def metrics_worker():
    """
    Drains the metrics queue and persists records.
    Runs as a long-lived background task — completely decoupled from the
    request/response path so it never adds latency to the client.
    """
    log.info("Metrics worker started.")
    while True:
        record = await metrics_queue.get()
        try:
            _persist_metrics(record)
        except Exception as exc:
            log.error("Failed to persist metrics record: %s", exc)
        finally:
            metrics_queue.task_done()


def _persist_metrics(record: dict):
    """
    Persist a single metrics record.
    Currently appends to an in-memory list and logs a summary line.
    TODO: swap this for an async SQLite / ClickHouse write.
    """
    metrics_store.append(record)

    eval_count = record.get("eval_count", 0)
    eval_duration_ns = record.get("eval_duration", 0)
    total_duration_ms = record.get("total_duration", 0) / 1_000_000

    tps = (
        eval_count / (eval_duration_ns / 1_000_000_000)
        if eval_duration_ns > 0
        else 0.0
    )

    log.info(
        "METRICS | model=%-20s tokens=%4d  tps=%6.1f  total_ms=%7.1f  "
        "prompt_ms=%6.1f  gen_ms=%6.1f",
        record.get("model", "unknown"),
        eval_count,
        tps,
        total_duration_ms,
        record.get("prompt_eval_duration", 0) / 1_000_000,
        eval_duration_ns / 1_000_000,
    )


# ---------------------------------------------------------------------------
# App lifespan — starts the worker before accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(metrics_worker())
    log.info("Proxy ready. Forwarding to %s", OLLAMA_BASE_URL)
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    log.info("Metrics worker stopped.")


app = FastAPI(
    title="LLM Performance Proxy",
    description="Zero-overhead observability layer for local LLM inference.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Proxy liveness check. Also probes Ollama availability."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "proxy": "ok",
        "ollama": "ok" if ollama_ok else "unreachable",
        "queued_metrics": metrics_queue.qsize(),
        "stored_records": len(metrics_store),
    }


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """
    Streaming proxy for Ollama's /api/chat endpoint.

    Flow:
      1. Read & validate the incoming JSON body.
      2. Force stream=True so Ollama sends newline-delimited JSON chunks.
      3. Open a persistent httpx stream to Ollama.
      4. Yield each raw chunk back to the client immediately (zero buffering).
      5. On the final chunk (done=True), extract perf fields and enqueue them
         for async persistence — never blocking the response path.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # Always request streaming from Ollama regardless of what client sent
    body["stream"] = True
    model = body.get("model", "unknown")
    wall_start = time.monotonic()

    async def stream_and_capture():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as ollama_response:

                if ollama_response.status_code != 200:
                    error_body = await ollama_response.aread()
                    yield error_body
                    return

                async for raw_line in ollama_response.aiter_lines():
                    if not raw_line:
                        continue

                    # Yield raw bytes to client immediately
                    yield (raw_line + "\n").encode()

                    # Parse only to check for the terminal metrics chunk
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("done") is True:
                        wall_elapsed_ns = int(
                            (time.monotonic() - wall_start) * 1_000_000_000
                        )
                        record = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model,
                            "wall_duration_ns": wall_elapsed_ns,
                            # Ollama native perf fields
                            "total_duration": chunk.get("total_duration"),
                            "prompt_eval_duration": chunk.get("prompt_eval_duration"),
                            "eval_duration": chunk.get("eval_duration"),
                            "eval_count": chunk.get("eval_count"),
                            "prompt_eval_count": chunk.get("prompt_eval_count"),
                        }
                        # Non-blocking enqueue — drop if queue is full under load
                        try:
                            metrics_queue.put_nowait(record)
                        except asyncio.QueueFull:
                            log.warning("Metrics queue full — dropping record for model %s", model)

    return StreamingResponse(
        stream_and_capture(),
        media_type="application/x-ndjson",
    )


@app.get("/metrics/summary")
async def metrics_summary():
    """
    Returns aggregated performance stats across all captured requests.
    TODO: replace with time-windowed SQL queries.
    """
    if not metrics_store:
        return {"message": "No metrics captured yet.", "total_requests": 0}

    valid = [
        r for r in metrics_store
        if r.get("eval_duration") and r.get("eval_count")
    ]

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    tps_list = [
        r["eval_count"] / (r["eval_duration"] / 1_000_000_000) for r in valid
    ]
    total_ms_list = [
        r["total_duration"] / 1_000_000 for r in valid if r.get("total_duration")
    ]
    prompt_ms_list = [
        r["prompt_eval_duration"] / 1_000_000
        for r in valid if r.get("prompt_eval_duration")
    ]

    return {
        "total_requests": len(metrics_store),
        "requests_with_metrics": len(valid),
        "avg_tokens_per_second": round(avg(tps_list), 2),
        "avg_total_latency_ms": round(avg(total_ms_list), 2),
        "avg_prompt_eval_ms": round(avg(prompt_ms_list), 2),
        "peak_tokens_per_second": round(max(tps_list), 2) if tps_list else 0,
        "total_tokens_generated": sum(r.get("eval_count", 0) for r in metrics_store),
    }