# ── Stage 1: dependency installation ─────────────────────────────────────────
# Use a full image to compile any C-extension wheels, then copy only the
# installed packages into the slim runtime image.
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user — never run application code as root in a container
RUN adduser --disabled-password --no-create-home appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY pyproject.toml .
COPY src/ ./src/

# SQLite database file will live here; mount a volume in production so it
# persists across container restarts:
#   docker run -v llm-proxy-data:/app/data -e DB_PATH=/app/data/metrics.db …
RUN mkdir -p /app/data && chown appuser /app/data
ENV DB_PATH=/app/data/metrics.db

USER appuser

EXPOSE 8000

# Use 1 worker (SQLite is single-writer); increase only when swapping to Postgres
CMD ["uvicorn", "llm_perf_proxy.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
