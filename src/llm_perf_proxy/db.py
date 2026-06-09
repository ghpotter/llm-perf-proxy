"""
db.py — SQLite persistence layer
==================================
Owns the shared async database connection, schema migration, and the
single write function called by the metrics worker.

Nothing in this module knows about HTTP, FastAPI, or which backend is
active — it only knows how to store and query MetricsRecord dicts.
"""

import logging

import aiosqlite

from llm_perf_proxy.backends.base import MetricsRecord
from llm_perf_proxy.config import DB_PATH

log = logging.getLogger("llm-proxy.db")

# Shared connection — opened in lifespan, injected via init() below.
# Using a module-level variable (rather than passing conn everywhere) keeps
# call sites in worker.py and routes clean without a dependency-injection
# framework.
conn: aiosqlite.Connection | None = None


async def init() -> None:
    """
    Open the database connection and run the idempotent schema migration.
    Must be called once during app startup before any writes occur.
    """
    global conn
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    await _migrate(conn)
    log.info("Database ready at %s", DB_PATH)


async def close() -> None:
    """Flush WAL and close the connection. Called during shutdown."""
    if conn:
        await conn.close()
        log.info("Database connection closed.")


async def _migrate(c: aiosqlite.Connection) -> None:
    """
    Idempotent schema migration — safe to run on every startup.

    CREATE TABLE/INDEX IF NOT EXISTS means existing data is never touched.
    ALTER TABLE … ADD COLUMN handles upgrades from earlier schema versions
    (the except-pass swallows the 'column already exists' error from SQLite).

    All Ollama durations are stored raw in nanoseconds so no precision is
    lost; the summary endpoint converts to ms on read.
    """
    await c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp            TEXT    NOT NULL,
            backend              TEXT    NOT NULL DEFAULT 'ollama',
            endpoint             TEXT    NOT NULL,
            model                TEXT    NOT NULL,
            wall_duration_ns     INTEGER,
            ttft_ns              INTEGER,
            total_duration       INTEGER,
            load_duration        INTEGER,
            prompt_eval_count    INTEGER,
            prompt_eval_duration INTEGER,
            eval_count           INTEGER,
            eval_duration        INTEGER
        )
    """)

    # Upgrade existing databases that predate the backend/ttft columns
    for col, typedef in [
        ("backend", "TEXT NOT NULL DEFAULT 'ollama'"),
        ("ttft_ns", "INTEGER"),
    ]:
        try:
            await c.execute(f"ALTER TABLE requests ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already exists

    for idx, col in [
        ("idx_requests_backend", "backend"),
        ("idx_requests_model", "model"),
        ("idx_requests_endpoint", "endpoint"),
        ("idx_requests_timestamp", "timestamp"),
    ]:
        await c.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON requests ({col})")

    await c.commit()


async def insert(record: MetricsRecord) -> None:
    """
    Write one MetricsRecord to the database.
    Parameterised — no string formatting, no injection risk.
    """
    await conn.execute(
        """
        INSERT INTO requests (
            timestamp, backend, endpoint, model,
            wall_duration_ns, ttft_ns,
            total_duration, load_duration,
            prompt_eval_count, prompt_eval_duration,
            eval_count, eval_duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("timestamp"),
            record.get("backend"),
            record.get("endpoint"),
            record.get("model"),
            record.get("wall_duration_ns"),
            record.get("ttft_ns"),
            record.get("total_duration"),
            record.get("load_duration"),
            record.get("prompt_eval_count"),
            record.get("prompt_eval_duration"),
            record.get("eval_count"),
            record.get("eval_duration"),
        ),
    )
    await conn.commit()
