"""
backends/anthropic.py — Anthropic Claude backend
==================================================
Streams from the Anthropic Messages API using its SSE event/data format.
"""

import json
import time
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx

from src.llm_perf_proxy.backends.base import Backend, MetricsRecord
from src.llm_perf_proxy.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL


class AnthropicBackend(Backend):
    """
    Streams from the Anthropic Messages API.

    Anthropic's SSE format uses paired ``event:`` / ``data:`` lines.
    Key events and what we extract from each:

        message_start       → usage.input_tokens  (prompt token count)
        content_block_delta → first text delta marks TTFT
        message_delta       → usage.output_tokens (completion token count)
        message_stop        → stream is complete

    The stream is passed through verbatim so Anthropic SDK clients that
    parse SSE events (e.g. streaming tool-use) continue to work correctly.

    Measured fields:
        wall_duration_ns  — monotonic ns, request start → stream end
        ttft_ns           — monotonic ns, request start → first text delta
        eval_count        — output_tokens from message_delta.usage
        prompt_eval_count — input_tokens from message_start.message.usage

    Ollama-specific fields are None.
    """

    name = "anthropic"

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.api_key  = ANTHROPIC_API_KEY
        self.base_url = ANTHROPIC_BASE_URL

    async def stream(
        self,
        endpoint: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[bytes | MetricsRecord]:
        model = body.get("model", "unknown")
        body["stream"] = True

        out_headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        wall_start = time.monotonic()
        ttft_ns: int | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=body,
                headers=out_headers,
            ) as resp:
                if resp.status_code != 200:
                    yield await resp.aread()
                    return

                event_type: str | None = None

                async for raw_line in resp.aiter_lines():
                    yield (raw_line + "\n").encode()

                    if raw_line.startswith("event:"):
                        event_type = raw_line[6:].strip()
                        continue

                    if not raw_line.startswith("data:"):
                        continue

                    try:
                        chunk = json.loads(raw_line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    if event_type == "message_start":
                        usage = chunk.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens")

                    elif event_type == "content_block_delta":
                        if ttft_ns is None:
                            delta = chunk.get("delta", {})
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                ttft_ns = int((time.monotonic() - wall_start) * 1e9)

                    elif event_type == "message_delta":
                        usage = chunk.get("usage", {})
                        output_tokens = usage.get("output_tokens")

        wall_ns = int((time.monotonic() - wall_start) * 1e9)
        yield MetricsRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            backend="anthropic",
            endpoint=endpoint,
            model=model,
            wall_duration_ns=wall_ns,
            ttft_ns=ttft_ns,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=input_tokens,
            prompt_eval_duration=None,
            eval_count=output_tokens,
            eval_duration=None,
        )