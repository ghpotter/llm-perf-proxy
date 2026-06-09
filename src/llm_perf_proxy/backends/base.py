"""
backends/base.py — shared types for all LLM backends
======================================================
Defines the ``MetricsRecord`` data container and the ``Backend`` abstract
base class that every concrete backend must implement.

Keeping these in their own file means:
  - ``db.py`` and ``worker.py`` can import ``MetricsRecord`` without pulling
    in any backend logic or HTTP clients.
  - New backends only need to import from here, not from each other.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class MetricsRecord(dict):
    """
    A dict subclass used as the terminal sentinel in each backend's stream().

    Fields populated by every backend:
        timestamp           ISO-8601 UTC string
        backend             "ollama" | "openai" | "anthropic"
        endpoint            "chat" | "generate"
        model               model name string
        wall_duration_ns    monotonic ns from request start to stream end
        prompt_eval_count   input / prompt token count
        eval_count          output / completion token count

    Fields populated by Ollama only (None for remote backends):
        total_duration       ns — Ollama end-to-end including load
        load_duration        ns — model cold-start time
        prompt_eval_duration ns — time spent processing the prompt (TTFT proxy)
        eval_duration        ns — time spent generating tokens

    Fields populated by remote backends only (None for Ollama):
        ttft_ns              ns — wall-clock time to first content token
    """


class Backend(ABC):
    """
    Abstract base class for all LLM backends.

    Subclasses implement ``stream()``, an async generator that must:

      1. Forward the request to the upstream API.
      2. Yield each raw response chunk as ``bytes`` immediately, so the
         proxy can pass it to the client with zero added buffering.
      3. After the upstream stream ends, yield exactly one ``MetricsRecord``
         as its final item.

    The proxy's ``_proxy_stream`` helper separates bytes from MetricsRecord
    using ``isinstance`` — bytes are forwarded to the client, the record is
    enqueued for async persistence. Backends never touch the queue directly.
    """

    #: Identifier stored in every MetricsRecord and DB row this backend emits.
    name: str

    @abstractmethod
    async def stream(
        self,
        endpoint: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[bytes | MetricsRecord]:
        """Yield response bytes, then one MetricsRecord."""
