"""
backends/__init__.py
====================
Public surface of the backends package.

Imports here define what ``from backends import …`` exposes, and keep
main.py free of per-backend import lines. Adding a new backend means:
  1. Create backends/newprovider.py with a class that extends Backend.
  2. Import and register it in _build_backend() below.
  3. Add its env vars to config.py and .env.example.
Nothing else needs to change.
"""

from llm_perf_proxy.backends.anthropic import AnthropicBackend
from llm_perf_proxy.backends.base import Backend, MetricsRecord
from llm_perf_proxy.backends.ollama import OllamaBackend
from llm_perf_proxy.backends.openai import OpenAIBackend
from llm_perf_proxy.config import BACKEND


def build_backend() -> Backend:
    """
    Instantiate and return the backend selected by the BACKEND env var.
    Fails fast at startup with a clear message if the value is unrecognised
    or a required API key is missing — better than a cryptic error mid-request.
    """
    match BACKEND:
        case "ollama":
            return OllamaBackend()
        case "openai":
            return OpenAIBackend()
        case "anthropic":
            return AnthropicBackend()
        case _:
            raise RuntimeError(
                f"Unknown BACKEND '{BACKEND}'. Valid options: ollama, openai, anthropic."
            )


__all__ = [
    "Backend",
    "MetricsRecord",
    "OllamaBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "build_backend",
]
