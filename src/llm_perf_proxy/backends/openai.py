"""
backends/openai.py — OpenAI-compatible backend
================================================
Works with any OpenAI-compatible API: OpenAI, Groq, Together, LiteLLM, etc.
Override OPENAI_BASE_URL in .env to point at a different host.
"""

import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx

from llm_perf_proxy.backends.base import Backend, MetricsRecord
from llm_perf_proxy.config import OPENAI_API_KEY, OPENAI_BASE_URL


class OpenAIBackend(Backend):
    """
    Streams from an OpenAI-compatible SSE API.

    Wire format: ``data: {…}\\n\\n`` lines, terminated by ``data: [DONE]``.
    Token counts arrive in ``usage`` on the last content chunk when
    ``stream_options.include_usage`` is set to true — we inject this
    automatically so clients don't need to remember.

    Measured fields (server-side timing is not exposed by OpenAI):
        wall_duration_ns  — monotonic ns, request start → stream end
        ttft_ns           — monotonic ns, request start → first content token
        eval_count        — usage.completion_tokens
        prompt_eval_count — usage.prompt_tokens

    Ollama-specific fields (total_duration, eval_duration, etc.) are None.
    """

    name = "openai"

    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.api_key = OPENAI_API_KEY
        self.base_url = OPENAI_BASE_URL

    async def stream(
        self,
        endpoint: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[bytes | MetricsRecord]:
        model = body.get("model", "unknown")
        body.setdefault("stream_options", {})["include_usage"] = True
        body["stream"] = True

        out_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        wall_start = time.monotonic()
        ttft_ns: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers=out_headers,
            ) as resp:
                if resp.status_code != 200:
                    yield await resp.aread()
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line or not raw_line.startswith("data:"):
                        continue

                    payload = raw_line[5:].strip()

                    if payload == "[DONE]":
                        yield (raw_line + "\n").encode()
                        break

                    yield (raw_line + "\n").encode()

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # Capture TTFT on first chunk that carries actual content
                    if ttft_ns is None:
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        if delta.get("content"):
                            ttft_ns = int((time.monotonic() - wall_start) * 1e9)

                    # Usage arrives on the final content chunk
                    if chunk.get("usage"):
                        prompt_tokens = chunk["usage"].get("prompt_tokens")
                        completion_tokens = chunk["usage"].get("completion_tokens")

        wall_ns = int((time.monotonic() - wall_start) * 1e9)
        yield MetricsRecord(
            timestamp=datetime.now(UTC).isoformat(),
            backend="openai",
            endpoint=endpoint,
            model=model,
            wall_duration_ns=wall_ns,
            ttft_ns=ttft_ns,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=prompt_tokens,
            prompt_eval_duration=None,
            eval_count=completion_tokens,
            eval_duration=None,
        )