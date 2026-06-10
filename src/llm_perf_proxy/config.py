"""
config.py — centralised configuration
======================================
All os.getenv() calls live here so every other module imports a typed
constant rather than scattering getenv() across the codebase.

To add a setting: add it here, import it where needed. To audit what
the app reads from the environment, read this file — nowhere else.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Logging (configured early so _load_price_table can emit warnings) ─────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Proxy ─────────────────────────────────────────────────────────────────────

#: Which backend to forward requests to. One of: ollama | openai | anthropic
BACKEND: str = os.getenv("BACKEND", "ollama").lower()

#: SQLite database file path (relative to cwd or absolute)
DB_PATH: str = os.getenv("DB_PATH", "metrics.db")

#: Maximum number of unprocessed records allowed in the async metrics queue.
#: Requests that would exceed this are dropped with a warning rather than
#: blocking the response path.
METRICS_QUEUE_MAX_SIZE: int = int(os.getenv("METRICS_QUEUE_MAX_SIZE", "1000"))

# ── Backend-specific ──────────────────────────────────────────────────────────

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# ── Model pricing ─────────────────────────────────────────────────────────────
# Keyed by model-name prefix. Values are (input_$/1M_tokens, output_$/1M_tokens).
# Longest prefix wins so "gpt-4o-mini" is matched before "gpt-4o".
# Anthropic pricing: https://www.anthropic.com/pricing (cached 2026-05-26)
# OpenAI pricing:    https://platform.openai.com/pricing

_PRICE_TABLE_DEFAULTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "o1-mini": (3.00, 12.00),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "o3": (10.00, 40.00),
    "o4-mini": (1.10, 4.40),
}

#: Optional path to a JSON file that overrides or extends the built-in price table.
#: Format: { "model-prefix": [input_$/1M, output_$/1M], ... }
#: Copy prices.example.json to prices.json and edit to add or override entries.
PRICES_PATH: str = os.getenv("PRICES_PATH", "prices.json")


def _load_price_table() -> dict[str, tuple[float, float]]:
    table = dict(_PRICE_TABLE_DEFAULTS)
    prices_file = Path(PRICES_PATH)
    if prices_file.exists():
        log = logging.getLogger("llm-proxy")
        try:
            overrides: dict = json.loads(prices_file.read_text())
            for k, v in overrides.items():
                table[k] = (float(v[0]), float(v[1]))
            log.info("Loaded %d price entries from %s", len(overrides), prices_file)
        except Exception as exc:
            log.warning("Failed to load %s: %s — using built-in prices", prices_file, exc)
    return table


_PRICE_TABLE: dict[str, tuple[float, float]] = _load_price_table()


def compute_cost(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Return estimated USD cost, or None if the model is unknown or tokens unavailable."""
    if not model or prompt_tokens is None or completion_tokens is None:
        return None
    # Sort longest key first so "gpt-4o-mini" is checked before "gpt-4o"
    for key in sorted(_PRICE_TABLE, key=len, reverse=True):
        if model.startswith(key):
            input_rate, output_rate = _PRICE_TABLE[key]
            return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
    return None


# ── Startup validation ────────────────────────────────────────────────────────

_VALID_BACKENDS = {"ollama", "openai", "anthropic"}


def validate() -> None:
    """Raise ValueError with a clear message if the configuration is unusable."""
    if BACKEND not in _VALID_BACKENDS:
        raise ValueError(
            f"BACKEND={BACKEND!r} is not valid. Choose one of: {', '.join(sorted(_VALID_BACKENDS))}"
        )
    if BACKEND == "openai" and not OPENAI_API_KEY:
        raise ValueError("BACKEND=openai requires OPENAI_API_KEY to be set.")
    if BACKEND == "anthropic" and not ANTHROPIC_API_KEY:
        raise ValueError("BACKEND=anthropic requires ANTHROPIC_API_KEY to be set.")
