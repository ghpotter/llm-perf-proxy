"""
config.py — centralised configuration
======================================
All os.getenv() calls live here so every other module imports a typed
constant rather than scattering getenv() across the codebase.

To add a setting: add it here, import it where needed. To audit what
the app reads from the environment, read this file — nowhere else.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

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

OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

ANTHROPIC_API_KEY: str  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)