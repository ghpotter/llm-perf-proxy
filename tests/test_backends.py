"""
tests/test_backends.py
======================
Unit tests for backend MetricsRecord construction.

All network I/O is replaced with an in-memory async iterator so these
tests run offline with no LLM, no Ollama, and no API keys.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_perf_proxy.backends.anthropic import AnthropicBackend
from llm_perf_proxy.backends.base import MetricsRecord
from llm_perf_proxy.backends.ollama import OllamaBackend
from llm_perf_proxy.backends.openai import OpenAIBackend

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_ollama_stream(chunks: list[dict]):
    """Yield NDJSON lines as an async iterator, mimicking aiter_lines()."""

    async def _gen():
        for chunk in chunks:
            yield json.dumps(chunk)

    return _gen()


def _make_sse_stream(payloads: list[str]):
    """Yield SSE lines as an async iterator."""

    async def _gen():
        for p in payloads:
            yield p

    return _gen()


def _mock_http_stream(lines_gen, status=200):
    """
    Build a mock httpx streaming context manager that yields lines
    from lines_gen via aiter_lines().
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.aiter_lines = MagicMock(return_value=lines_gen)

    mock_stream_cm = MagicMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_cm)

    mock_client_cm = MagicMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_client_cm


# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_CHUNKS = [
    {"model": "phi3", "message": {"role": "assistant", "content": "Hello"}, "done": False},
    {"model": "phi3", "message": {"role": "assistant", "content": " world"}, "done": False},
    {
        "model": "phi3",
        "done": True,
        "total_duration": 5_000_000_000,
        "load_duration": 80_000_000,
        "prompt_eval_count": 18,
        "prompt_eval_duration": 300_000_000,
        "eval_count": 120,
        "eval_duration": 4_700_000_000,
    },
]


@pytest.mark.asyncio
async def test_ollama_yields_bytes_then_record():
    backend = OllamaBackend()
    lines = _make_ollama_stream(OLLAMA_CHUNKS)

    with patch(
        "llm_perf_proxy.backends.ollama.httpx.AsyncClient", return_value=_mock_http_stream(lines)
    ):
        items = [
            item async for item in backend.stream("chat", {"model": "phi3", "messages": []}, {})
        ]

    bytes_items = [i for i in items if isinstance(i, bytes)]
    record_items = [i for i in items if isinstance(i, MetricsRecord)]

    assert len(bytes_items) == 3, "Expected one bytes chunk per NDJSON line"
    assert len(record_items) == 1, "Expected exactly one MetricsRecord"


@pytest.mark.asyncio
async def test_ollama_record_fields():
    backend = OllamaBackend()
    lines = _make_ollama_stream(OLLAMA_CHUNKS)

    with patch(
        "llm_perf_proxy.backends.ollama.httpx.AsyncClient", return_value=_mock_http_stream(lines)
    ):
        items = [
            item async for item in backend.stream("chat", {"model": "phi3", "messages": []}, {})
        ]

    record = next(i for i in items if isinstance(i, MetricsRecord))

    assert record["backend"] == "ollama"
    assert record["model"] == "phi3"
    assert record["eval_count"] == 120
    assert record["eval_duration"] == 4_700_000_000
    assert record["prompt_eval_count"] == 18
    assert record["prompt_eval_duration"] == 300_000_000
    assert record["load_duration"] == 80_000_000
    assert record["ttft_ns"] is None  # Ollama doesn't measure TTFT


@pytest.mark.asyncio
async def test_ollama_tps_math():
    """Sanity-check the TPS formula used in worker._log_record."""
    eval_count = 120
    eval_duration_ns = 4_700_000_000
    tps = eval_count / (eval_duration_ns / 1e9)
    assert 25.0 < tps < 26.0, f"Unexpected TPS: {tps}"


# ── OpenAI ────────────────────────────────────────────────────────────────────

OPENAI_SSE_LINES = [
    'data: {"id":"c1","choices":[{"delta":{"role":"assistant"},"index":0}],"usage":null}',
    'data: {"id":"c1","choices":[{"delta":{"content":"Hello"},"index":0}],"usage":null}',
    'data: {"id":"c1","choices":[{"delta":{"content":" world"},"index":0}],"usage":null}',
    'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop","index":0}],'
    '"usage":{"prompt_tokens":12,"completion_tokens":85,"total_tokens":97}}',
    "data: [DONE]",
]


@pytest.mark.asyncio
async def test_openai_record_fields():
    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend.api_key = "sk-test"
    backend.base_url = "https://api.openai.com/v1"

    lines = _make_sse_stream(OPENAI_SSE_LINES)

    with patch(
        "llm_perf_proxy.backends.openai.httpx.AsyncClient", return_value=_mock_http_stream(lines)
    ):
        items = [
            item async for item in backend.stream("chat", {"model": "gpt-4o", "messages": []}, {})
        ]

    record = next(i for i in items if isinstance(i, MetricsRecord))

    assert record["backend"] == "openai"
    assert record["model"] == "gpt-4o"
    assert record["eval_count"] == 85
    assert record["prompt_eval_count"] == 12
    assert record["eval_duration"] is None  # not available from OpenAI
    assert record["ttft_ns"] is not None
    assert record["ttft_ns"] > 0


@pytest.mark.asyncio
async def test_openai_ttft_captured_on_first_content_chunk():
    """TTFT should be set on the first chunk with non-empty content."""
    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend.api_key = "sk-test"
    backend.base_url = "https://api.openai.com/v1"

    lines = _make_sse_stream(OPENAI_SSE_LINES)

    with patch(
        "llm_perf_proxy.backends.openai.httpx.AsyncClient", return_value=_mock_http_stream(lines)
    ):
        items = [
            item async for item in backend.stream("chat", {"model": "gpt-4o", "messages": []}, {})
        ]

    record = next(i for i in items if isinstance(i, MetricsRecord))
    # TTFT must be less than wall time — it's a sub-interval
    assert record["ttft_ns"] < record["wall_duration_ns"]


# ── Anthropic ─────────────────────────────────────────────────────────────────

ANTHROPIC_SSE_LINES = [
    "event: message_start",
    'data: {"type":"message_start","message":{"usage":{"input_tokens":21}}}',
    "event: content_block_start",
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
    "event: message_delta",
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":102}}',
    "event: message_stop",
    'data: {"type":"message_stop"}',
]


@pytest.mark.asyncio
async def test_anthropic_record_fields():
    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend.api_key = "sk-ant-test"
    backend.base_url = "https://api.anthropic.com"

    lines = _make_sse_stream(ANTHROPIC_SSE_LINES)

    with patch(
        "llm_perf_proxy.backends.anthropic.httpx.AsyncClient", return_value=_mock_http_stream(lines)
    ):
        items = [
            item
            async for item in backend.stream(
                "chat",
                {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 1024},
                {},
            )
        ]

    record = next(i for i in items if isinstance(i, MetricsRecord))

    assert record["backend"] == "anthropic"
    assert record["model"] == "claude-sonnet-4-20250514"
    assert record["eval_count"] == 102
    assert record["prompt_eval_count"] == 21
    assert record["eval_duration"] is None
    assert record["ttft_ns"] is not None
    assert record["ttft_ns"] > 0


@pytest.mark.asyncio
async def test_anthropic_ttft_on_first_text_delta():
    """TTFT should fire on the first content_block_delta with non-empty text."""
    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend.api_key = "sk-ant-test"
    backend.base_url = "https://api.anthropic.com"

    lines = _make_sse_stream(ANTHROPIC_SSE_LINES)

    with patch(
        "llm_perf_proxy.backends.anthropic.httpx.AsyncClient", return_value=_mock_http_stream(lines)
    ):
        items = [
            item
            async for item in backend.stream(
                "chat",
                {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 1024},
                {},
            )
        ]

    record = next(i for i in items if isinstance(i, MetricsRecord))
    assert record["ttft_ns"] < record["wall_duration_ns"]