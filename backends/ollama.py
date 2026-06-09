"""
backends/ollama.py — Ollama backend
=====================================
Streams from a local Ollama instance and extracts its native nanosecond-
precision performance fields from the terminal chunk (done=True).
"""

import json
import time
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx

from backends.base import Backend, MetricsRecord
from config import OLLAMA_BASE_URL


def _ollama_path(endpoint: str) -> str:
    """Map proxy endpoint name to Ollama API path."""
    return "/api/chat" if endpoint == "chat" else "/api/generate"


class OllamaBackend(Backend):
    """
    Streams from a local Ollama instance.

    Ollama returns nanosecond-precision perf fields on the terminal chunk
    (done=True), so we use those directly — no wall-clock approximation needed.
    Every raw NDJSON line is yielded to the client *before* being parsed,
    so there is genuinely zero added latency on the hot path.
    """

    name = "ollama"

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL

    async def stream(
        self,
        endpoint: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[bytes | MetricsRecord]:
        model = body.get("model", "unknown")
        wall_start = time.monotonic()

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}{_ollama_path(endpoint)}",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status_code != 200:
                    yield await resp.aread()
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    yield (raw_line + "\n").encode()

                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("done") is True:
                        wall_ns = int((time.monotonic() - wall_start) * 1e9)
                        yield MetricsRecord(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            backend="ollama",
                            endpoint=endpoint,
                            model=model,
                            wall_duration_ns=wall_ns,
                            ttft_ns=None,
                            total_duration=chunk.get("total_duration"),
                            load_duration=chunk.get("load_duration"),
                            prompt_eval_count=chunk.get("prompt_eval_count"),
                            prompt_eval_duration=chunk.get("prompt_eval_duration"),
                            eval_count=chunk.get("eval_count"),
                            eval_duration=chunk.get("eval_duration"),
                        )