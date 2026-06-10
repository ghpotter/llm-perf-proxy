"""
worker.py — async metrics queue and background worker
=======================================================
Decouples the request/response path from database writes.

The proxy enqueues a MetricsRecord with put_nowait() (non-blocking).
The worker drains the queue in a separate asyncio Task, calling db.insert()
and logging a summary line. Even if a write takes tens of milliseconds, the
client's stream is never affected.
"""

import asyncio
import logging

import llm_perf_proxy.db as db
from llm_perf_proxy.backends.base import MetricsRecord
from llm_perf_proxy.config import METRICS_QUEUE_MAX_SIZE, compute_cost

log = logging.getLogger("llm-proxy.worker")

queue: asyncio.Queue = asyncio.Queue(maxsize=METRICS_QUEUE_MAX_SIZE)


def enqueue(record: MetricsRecord) -> None:
    """
    Non-blocking enqueue. Drops the record and logs a warning if the queue
    is full — preferable to blocking the response path under sustained load.
    """
    try:
        queue.put_nowait(record)
    except asyncio.QueueFull:
        log.warning(
            "Metrics queue full — dropping record [backend=%s model=%s]",
            record.get("backend"),
            record.get("model"),
        )


async def run() -> None:
    """
    Long-lived background task. Drain the queue indefinitely, persisting
    each record and emitting a one-line log summary.

    Started in app lifespan; cancelled (then flushed via queue.join()) on
    shutdown so in-flight records are never silently dropped.
    """
    log.info("Metrics worker started.")
    while True:
        record: MetricsRecord = await queue.get()
        try:
            cost = compute_cost(
                record.get("model"),
                record.get("prompt_eval_count"),
                record.get("eval_count"),
            )
            if cost is not None:
                record["estimated_cost_usd"] = cost
            await db.insert(record)
            _log_record(record)
        except Exception as exc:
            log.error("Failed to persist metrics record: %s", exc)
        finally:
            queue.task_done()


def _log_record(record: MetricsRecord) -> None:
    """Emit a structured one-liner to stdout for live tail monitoring."""
    eval_count = record.get("eval_count") or 0
    wall_ns = record.get("wall_duration_ns") or 0
    ttft_ns = record.get("ttft_ns")
    # Use Ollama's native gen duration when available; fall back to wall clock
    gen_ns = record.get("eval_duration") or wall_ns
    tps = eval_count / (gen_ns / 1e9) if gen_ns > 0 else 0.0
    ttft_ms = f"{ttft_ns / 1e6:.1f}" if ttft_ns is not None else "n/a"
    cost = record.get("estimated_cost_usd")
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"

    log.info(
        "METRICS | backend=%-9s  endpoint=%-8s  model=%-24s  "
        "tokens=%4d  tps=%6.1f  wall_ms=%7.1f  ttft_ms=%s  cost=%s",
        record.get("backend", "?"),
        record.get("endpoint", "?"),
        record.get("model", "unknown"),
        eval_count,
        tps,
        wall_ns / 1e6,
        ttft_ms,
        cost_str,
    )
