"""
tests/test_db.py
================
Integration tests for the database layer.
Uses a real aiosqlite in-memory database (:memory:) so tests are fast,
isolated, and leave no files on disk.
"""

import aiosqlite
import pytest

import llm_perf_proxy.db as db
from llm_perf_proxy.backends.base import MetricsRecord


@pytest.fixture
async def test_db(monkeypatch):
    """
    Fixture: open an in-memory DB, run the migration, inject it into
    the db module, and close it after the test.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await db._migrate(conn)

    monkeypatch.setattr(db, "conn", conn)
    yield conn

    await conn.close()


@pytest.mark.asyncio
async def test_schema_creates_table(test_db):
    async with test_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='requests'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "requests table should exist after migration"


@pytest.mark.asyncio
async def test_schema_idempotent(test_db):
    """Running _migrate a second time on the same DB must not raise."""
    await db._migrate(test_db)


@pytest.mark.asyncio
async def test_insert_ollama_record(test_db):
    record = MetricsRecord(
        timestamp="2026-06-08T00:00:00Z",
        backend="ollama",
        endpoint="chat",
        model="phi3",
        wall_duration_ns=5_000_000_000,
        ttft_ns=None,
        total_duration=4_900_000_000,
        load_duration=80_000_000,
        prompt_eval_count=18,
        prompt_eval_duration=300_000_000,
        eval_count=120,
        eval_duration=4_600_000_000,
    )
    await db.insert(record)

    async with test_db.execute("SELECT * FROM requests") as cur:
        row = await cur.fetchone()

    assert row["backend"] == "ollama"
    assert row["model"] == "phi3"
    assert row["eval_count"] == 120
    assert row["eval_duration"] == 4_600_000_000
    assert row["ttft_ns"] is None


@pytest.mark.asyncio
async def test_insert_openai_record(test_db):
    record = MetricsRecord(
        timestamp="2026-06-08T00:01:00Z",
        backend="openai",
        endpoint="chat",
        model="gpt-4o",
        wall_duration_ns=2_100_000_000,
        ttft_ns=480_000_000,
        total_duration=None,
        load_duration=None,
        prompt_eval_count=12,
        prompt_eval_duration=None,
        eval_count=85,
        eval_duration=None,
    )
    await db.insert(record)

    async with test_db.execute("SELECT * FROM requests WHERE backend='openai'") as cur:
        row = await cur.fetchone()

    assert row["backend"] == "openai"
    assert row["eval_count"] == 85
    assert row["eval_duration"] is None
    assert row["ttft_ns"] == 480_000_000


@pytest.mark.asyncio
async def test_row_count_after_multiple_inserts(test_db):
    for i in range(5):
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
                eval_count=50 + i,
                eval_duration=2_000_000_000,
            )
        )

    async with test_db.execute("SELECT COUNT(*) AS n FROM requests") as cur:
        row = await cur.fetchone()
    assert row["n"] == 5


@pytest.mark.asyncio
async def test_summary_sql_coalesce(test_db):
    """
    The summary query uses COALESCE(eval_duration, wall_duration_ns).
    Verify it computes TPS correctly for both an Ollama row (eval_duration set)
    and an OpenAI row (eval_duration NULL, wall_duration_ns used instead).
    """
    await db.insert(
        MetricsRecord(
            timestamp="2026-06-08T00:00:00Z",
            backend="ollama",
            endpoint="chat",
            model="phi3",
            wall_duration_ns=5_000_000_000,
            ttft_ns=None,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=100,
            eval_duration=4_000_000_000,
        )
    )
    await db.insert(
        MetricsRecord(
            timestamp="2026-06-08T00:01:00Z",
            backend="openai",
            endpoint="chat",
            model="gpt-4o",
            wall_duration_ns=2_000_000_000,
            ttft_ns=400_000_000,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=80,
            eval_duration=None,
        )
    )

    async with test_db.execute("""
        SELECT model,
               CAST(eval_count AS REAL)
               / (COALESCE(eval_duration, wall_duration_ns) / 1e9) AS tps
        FROM requests
        ORDER BY model
    """) as cur:
        rows = {r["model"]: r["tps"] async for r in cur}

    # phi3: 100 tokens / 4s = 25 t/s
    assert abs(rows["phi3"] - 25.0) < 0.1

    # gpt-4o: 80 tokens / 2s = 40 t/s (wall-clock fallback)
    assert abs(rows["gpt-4o"] - 40.0) < 0.1