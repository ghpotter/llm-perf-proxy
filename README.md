# llm-perf-proxy

An asynchronous LLM performance and cost observability proxy. Sits in front of a local or remote LLM, streams responses back to the client with zero added latency, and captures performance metrics asynchronously via a background worker queue persisted to SQLite.

```
client  →  llm-perf-proxy :8000  →  Ollama / OpenAI / Anthropic
                ↓ (non-blocking)
           metrics queue  →  SQLite
```

## Features

- **Zero-overhead streaming** — raw response bytes are forwarded to the client before any parsing occurs
- **Async metrics capture** — a background worker drains a queue into SQLite, completely decoupled from the request path
- **Three backends** — Ollama (local), OpenAI-compatible APIs, and Anthropic
- **SQLite persistence** — metrics survive restarts; schema migration runs automatically on startup
- **REST summary endpoint** — aggregated stats sliced by backend, endpoint, model, and cross-tabbed combinations
- **CI/CD** — GitHub Actions pipeline with lint, type-check, tests, and Docker image push to GHCR

---

## Requirements

- Python 3.12+
- One of:
  - [Ollama](https://ollama.com) running locally (`ollama serve`)
  - An OpenAI API key
  - An Anthropic API key

---

## Quickstart

```bash
git clone https://github.com/<your-username>/llm-perf-proxy.git
cd llm-perf-proxy

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .

cp .env.example .env        # then edit .env to set BACKEND and any API keys
uvicorn llm_perf_proxy.main:app --reload
```

The proxy is now running on `http://localhost:8000`.

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and set the values for your chosen backend.

| Variable | Default | Description |
|---|---|---|
| `BACKEND` | `ollama` | Active backend: `ollama`, `openai`, or `anthropic` |
| `DB_PATH` | `metrics.db` | SQLite database file path |
| `METRICS_QUEUE_MAX_SIZE` | `1000` | Max queued records before dropping under load |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

**Ollama**

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |

**OpenAI** (also works with Groq, Together, LiteLLM, etc. — set `OPENAI_BASE_URL`)

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | *(required)* |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |

**Anthropic**

| Variable | Default |
|---|---|
| `ANTHROPIC_API_KEY` | *(required)* |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` |

---

## API

### `POST /api/chat`

Streaming chat proxy. Forwards to the active backend's chat endpoint.

```bash
# Ollama
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3", "messages": [{"role": "user", "content": "Why is the sky blue?"}]}'

# OpenAI
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Why is the sky blue?"}]}'

# Anthropic
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

### `POST /api/generate`

Raw completion proxy (Ollama only).

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3", "prompt": "Why is the sky blue?"}'
```

### `GET /health`

Liveness check. Reports active backend reachability and total persisted record count.

```json
{
  "proxy": "ok",
  "backend": { "name": "ollama", "reachable": true },
  "queued_metrics": 0,
  "stored_records": 42
}
```

### `GET /metrics/summary`

SQL-aggregated performance stats across all captured requests. Returns five slices of the same dataset.

```json
{
  "overall": {
    "requests": 42,
    "total_tokens": 4821,
    "avg_tps": 26.4,
    "peak_tps": 31.2,
    "avg_wall_ms": 4820.5,
    "avg_ttft_ms": null,
    "avg_total_ms": 4750.1,
    "avg_prompt_ms": 298.3
  },
  "by_backend": {
    "ollama": { "requests": 30, "avg_tps": 24.1, "avg_ttft_ms": null },
    "openai": { "requests": 8,  "avg_tps": 41.7, "avg_ttft_ms": 395.2 },
    "anthropic": { "requests": 4, "avg_tps": 31.9, "avg_ttft_ms": 620.1 }
  },
  "by_endpoint": { "chat": {...}, "generate": {...} },
  "by_model": {
    "gpt-4o":  { "avg_tps": 42.9, "total_tokens": 1240 },
    "phi3":    { "avg_tps": 24.1, "total_tokens": 3581 }
  },
  "by_model_and_endpoint": {
    "phi3": {
      "chat":     { "avg_tps": 24.3 },
      "generate": { "avg_tps": 23.8 }
    }
  },
  "queue_depth": 0
}
```

**Metric field reference**

| Field | Unit | Notes |
|---|---|---|
| `avg_tps` / `peak_tps` | tokens/sec | Uses Ollama's native `eval_duration` when available; falls back to `wall_duration_ns` for remote backends |
| `avg_wall_ms` | ms | End-to-end wall clock, including network RTT |
| `avg_ttft_ms` | ms | Time to first token. Measured for OpenAI and Anthropic; `null` for Ollama (use `avg_prompt_ms` instead) |
| `avg_total_ms` | ms | Ollama's internal total duration. `null` for remote backends |
| `avg_prompt_ms` | ms | Ollama's prompt evaluation time (proxy for TTFT). `null` for remote backends |

---

## Project structure

```
llm-perf-proxy/
├── src/
│   └── llm_perf_proxy/
│       ├── main.py          # FastAPI app, lifespan, routes
│       ├── config.py        # All environment variables
│       ├── db.py            # SQLite connection, schema migration, insert
│       ├── worker.py        # Async metrics queue and background drain task
│       └── backends/
│           ├── base.py      # Backend ABC + MetricsRecord
│           ├── ollama.py
│           ├── openai.py
│           └── anthropic.py
├── tests/
│   ├── test_backends.py     # Unit tests — MetricsRecord construction per backend
│   ├── test_db.py           # Integration tests — schema migration, insert, SQL aggregation
│   └── test_routes.py       # Smoke tests — route shapes using in-memory DB
├── .github/workflows/
│   ├── ci.yml               # Lint, type-check, test on every push
│   └── cd.yml               # Build and push Docker image on merge to main
├── .env.example
├── Dockerfile
└── pyproject.toml
```

---

## Development

```bash
pip install -e ".[dev]"       # install with dev dependencies

pytest tests/ -v              # run tests
ruff check .                  # lint
ruff format .                 # format
mypy src/llm_perf_proxy/      # type-check
```

To prevent lint errors from reaching CI, install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.12
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## Docker

```bash
# Build
docker build -t llm-perf-proxy .

# Run (Ollama backend, persisting the DB to a named volume)
docker run -p 8000:8000 \
  -e BACKEND=ollama \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v llm-proxy-data:/app/data \
  llm-perf-proxy

# Run (OpenAI backend)
docker run -p 8000:8000 \
  -e BACKEND=openai \
  -e OPENAI_API_KEY=sk-... \
  -v llm-proxy-data:/app/data \
  llm-perf-proxy
```

Pre-built images are published to GHCR on every merge to `main`:

```bash
docker pull ghcr.io/<your-username>/llm-perf-proxy:latest
```

---

## How metrics are captured

Each backend's `stream()` method is an async generator with a specific contract: it yields raw response bytes first (forwarded to the client immediately), then yields exactly one `MetricsRecord` as its final item. The proxy intercepts the record and hands it to the worker queue without the client ever seeing it.

**Ollama** — uses the native nanosecond-precision fields Ollama appends to the terminal chunk (`eval_duration`, `prompt_eval_duration`, `total_duration`, `load_duration`). No wall-clock approximation needed.

**OpenAI** — measures wall clock and TTFT with `time.monotonic()`. Token counts come from `usage.completion_tokens` and `usage.prompt_tokens` on the final SSE chunk (requires `stream_options.include_usage: true`, injected automatically).

**Anthropic** — same wall-clock approach. Input tokens come from `message_start`, output tokens from `message_delta`, and TTFT is captured on the first `content_block_delta` with non-empty text.