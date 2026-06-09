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

import aiosqlite
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
METRICS_QUEUE_MAX_SIZE = 1000  # back-pressure limit
DB_PATH = "metrics.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("llm-proxy")

# ---------------------------------------------------------------------------
# Database — shared async connection (opened in lifespan)
# ---------------------------------------------------------------------------

db: aiosqlite.Connection | None = None


async def _init_db(conn: aiosqlite.Connection) -> None:
    """
    Idempotent schema migration — safe to run on every startup.
    CREATE TABLE IF NOT EXISTS means existing data is never touched.

    All Ollama durations are stored in nanoseconds (raw from the API) so
    no precision is lost; the summary endpoint converts to ms on read.
    """
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            endpoint            TEXT    NOT NULL,
            model               TEXT    NOT NULL,
            wall_duration_ns    INTEGER,
            total_duration      INTEGER,
            load_duration       INTEGER,
            prompt_eval_count   INTEGER,
            prompt_eval_duration INTEGER,
            eval_count          INTEGER,
            eval_duration       INTEGER
        )
    """)
    # Index for the most common query patterns
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_model    ON requests (model)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_endpoint ON requests (endpoint)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests (timestamp)"
    )
    await conn.commit()
    log.info("Database ready at %s", DB_PATH)


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
            await _persist_metrics(record)
        except Exception as exc:
            log.error("Failed to persist metrics record: %s", exc)
        finally:
            metrics_queue.task_done()


async def _persist_metrics(record: dict) -> None:
    """
    Write a single metrics record to SQLite via the shared connection.
    Uses a parameterised INSERT — no string formatting, no injection risk.
    The log line is retained so live tail (``uvicorn`` stdout) still works
    as a lightweight real-time monitor without opening the DB.
    """
    await db.execute(
        """
        INSERT INTO requests (
            timestamp, endpoint, model,
            wall_duration_ns, total_duration, load_duration,
            prompt_eval_count, prompt_eval_duration,
            eval_count, eval_duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("timestamp"),
            record.get("endpoint"),
            record.get("model"),
            record.get("wall_duration_ns"),
            record.get("total_duration"),
            record.get("load_duration"),
            record.get("prompt_eval_count"),
            record.get("prompt_eval_duration"),
            record.get("eval_count"),
            record.get("eval_duration"),
        ),
    )
    await db.commit()

    eval_count       = record.get("eval_count") or 0
    eval_duration_ns = record.get("eval_duration") or 0
    tps = eval_count / (eval_duration_ns / 1_000_000_000) if eval_duration_ns > 0 else 0.0

    log.info(
        "METRICS | endpoint=%-8s  model=%-20s  tokens=%4d  tps=%6.1f  "
        "total_ms=%7.1f  prompt_ms=%6.1f  gen_ms=%6.1f",
        record.get("endpoint", "?"),
        record.get("model", "unknown"),
        eval_count,
        tps,
        (record.get("total_duration") or 0) / 1_000_000,
        (record.get("prompt_eval_duration") or 0) / 1_000_000,
        eval_duration_ns / 1_000_000,
    )


# ---------------------------------------------------------------------------
# App lifespan — starts the worker before accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = await aiosqlite.connect(DB_PATH)
    # Return rows as dict-like objects so column names are accessible by name
    db.row_factory = aiosqlite.Row
    await _init_db(db)

    worker_task = asyncio.create_task(metrics_worker())
    log.info("Proxy ready. Forwarding to %s", OLLAMA_BASE_URL)
    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # Flush any records still in the queue before closing
    await metrics_queue.join()
    await db.close()
    log.info("Database closed. Metrics worker stopped.")


app = FastAPI(
    title="LLM Performance Proxy",
    description="Zero-overhead observability layer for local LLM inference.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Shared streaming helper
# ---------------------------------------------------------------------------

async def _proxy_stream(ollama_path: str, body: dict, endpoint: str):
    """
    Generic async generator that:
      - Opens a streaming POST to ``ollama_path`` with ``body``.
      - Yields every raw NDJSON line back to the caller immediately.
      - On the terminal chunk (done=True) builds a metrics record tagged
        with ``endpoint`` and enqueues it for async persistence.

    Both /api/chat and /api/generate share this path — they differ only
    in which Ollama URL they target and which content field carries text,
    but the metrics fields on the final chunk are identical for both.
    """
    model = body.get("model", "unknown")
    wall_start = time.monotonic()

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}{ollama_path}",
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

                # Yield raw bytes to client before doing any parsing
                yield (raw_line + "\n").encode()

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
                        "endpoint": endpoint,   # "chat" | "generate"
                        "model": model,
                        "wall_duration_ns": wall_elapsed_ns,
                        # Ollama native perf fields (nanoseconds)
                        "total_duration": chunk.get("total_duration"),
                        "load_duration": chunk.get("load_duration"),
                        "prompt_eval_duration": chunk.get("prompt_eval_duration"),
                        "prompt_eval_count": chunk.get("prompt_eval_count"),
                        "eval_duration": chunk.get("eval_duration"),
                        "eval_count": chunk.get("eval_count"),
                    }
                    try:
                        metrics_queue.put_nowait(record)
                    except asyncio.QueueFull:
                        log.warning(
                            "Metrics queue full — dropping record "
                            "[endpoint=%s model=%s]",
                            endpoint, model,
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

    async with db.execute("SELECT COUNT(*) FROM requests") as cursor:
        row = await cursor.fetchone()
        stored_records = row[0] if row else 0

    return {
        "proxy": "ok",
        "ollama": "ok" if ollama_ok else "unreachable",
        "queued_metrics": metrics_queue.qsize(),
        "stored_records": stored_records,
    }


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """
    Streaming proxy for Ollama's /api/chat endpoint.

    Expects the standard Ollama chat body:
        { "model": "phi3", "messages": [{"role": "user", "content": "..."}] }

    Streams NDJSON chunks back to the client. Each intermediate chunk carries
    a ``message.content`` delta; the final chunk (done=True) carries perf
    metrics which are captured asynchronously without blocking the stream.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    body["stream"] = True  # enforce streaming regardless of client preference

    return StreamingResponse(
        _proxy_stream("/api/chat", body, endpoint="chat"),
        media_type="application/x-ndjson",
    )


@app.post("/api/generate")
async def proxy_generate(request: Request):
    """
    Streaming proxy for Ollama's /api/generate endpoint.

    Expects the standard Ollama generate body:
        { "model": "phi3", "prompt": "Why is the sky blue?" }

    Identical proxy mechanics to /api/chat. Intermediate chunks carry a
    ``response`` string field (raw text delta); the final chunk (done=True)
    carries the same perf metrics schema and is captured asynchronously.

    Key schema differences vs /api/chat:
        /api/chat   → chunk["message"]["content"]  (delta per chunk)
        /api/generate → chunk["response"]          (delta per chunk)
    The proxy is transparent to both — it streams raw bytes without
    inspecting the content field, so either schema passes through untouched.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    body["stream"] = True  # enforce streaming regardless of client preference

    return StreamingResponse(
        _proxy_stream("/api/generate", body, endpoint="generate"),
        media_type="application/x-ndjson",
    )


@app.get("/metrics/summary")
async def metrics_summary():
    """
    Aggregated performance stats backed by SQL — survives restarts.

    All heavy aggregation is pushed into SQLite so Python only handles
    serialisation. Four queries, each a GROUP BY slice of the same dataset:

        overall          — single global roll-up row
        by_endpoint      — GROUP BY endpoint   (chat vs generate)
        by_model         — GROUP BY model       (phi3 vs phi4 vs …)
        by_model_endpoint — GROUP BY model, endpoint  (cross-tab)
    """
    # Reusable SELECT expression block — identical across all four queries.
    # Defining it once avoids drift if the metric list ever changes.
    _METRICS_COLS = """
            COUNT(*)                                         AS requests,
            SUM(eval_count)                                  AS total_tokens,
            AVG(CAST(eval_count AS REAL)
                / (eval_duration / 1e9))                     AS avg_tps,
            MAX(CAST(eval_count AS REAL)
                / (eval_duration / 1e9))                     AS peak_tps,
            AVG(total_duration       / 1e6)                  AS avg_total_ms,
            AVG(prompt_eval_duration / 1e6)                  AS avg_prompt_ms
    """
    _WHERE = "WHERE eval_duration > 0 AND eval_count > 0"

    GLOBAL_SQL = f"SELECT {_METRICS_COLS} FROM requests {_WHERE}"

    PER_ENDPOINT_SQL = f"""
        SELECT endpoint, {_METRICS_COLS}
        FROM requests {_WHERE}
        GROUP BY endpoint
        ORDER BY endpoint
    """

    PER_MODEL_SQL = f"""
        SELECT model, {_METRICS_COLS}
        FROM requests {_WHERE}
        GROUP BY model
        ORDER BY avg_tps DESC
    """

    PER_MODEL_ENDPOINT_SQL = f"""
        SELECT model, endpoint, {_METRICS_COLS}
        FROM requests {_WHERE}
        GROUP BY model, endpoint
        ORDER BY model, endpoint
    """

    def _round_row(row) -> dict:
        return {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in dict(row).items()
        }

    async with db.execute(GLOBAL_SQL) as cur:
        global_row = await cur.fetchone()

    if not global_row or global_row["requests"] == 0:
        return {"message": "No metrics captured yet.", "total_requests": 0}

    async with db.execute(PER_ENDPOINT_SQL) as cur:
        endpoint_rows = await cur.fetchall()

    async with db.execute(PER_MODEL_SQL) as cur:
        model_rows = await cur.fetchall()

    async with db.execute(PER_MODEL_ENDPOINT_SQL) as cur:
        model_endpoint_rows = await cur.fetchall()

    # Build the cross-tab as { model: { endpoint: stats } }
    cross_tab: dict = {}
    for row in model_endpoint_rows:
        cross_tab.setdefault(row["model"], {})[row["endpoint"]] = _round_row(row)

    return {
        "overall": _round_row(global_row),
        "by_endpoint": {
            row["endpoint"]: _round_row(row) for row in endpoint_rows
        },
        "by_model": {
            row["model"]: _round_row(row) for row in model_rows
        },
        "by_model_and_endpoint": cross_tab,
        "queue_depth": metrics_queue.qsize(),
    }