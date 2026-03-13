# ============================================================
# Dockerfile — Systematic Alpha Research Pipeline
# ============================================================
# Multi-stage build:
#   Stage 1 (builder) : install Python deps into a venv
#   Stage 2 (runtime) : copy only the venv + source code
#
# Usage:
#   docker build -t systematic-alpha .
#   docker run --rm -v $(pwd)/data:/app/data systematic-alpha \
#       python main.py --mode research --tickers SPY QQQ
#
# Pass secrets via environment variables (never bake into the image):
#   docker run --env-file .env systematic-alpha python main.py --mode live --ticker SPY
#
# ============================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build dependencies (for LightGBM, scipy, matplotlib C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt .

# Install into an isolated venv inside the image
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="systematic-alpha"
LABEL description="Systematic Alpha Research Pipeline — ML-driven quant trading system"

# Runtime system lib for LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set PATH so Python commands find the venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# MLflow tracking (file-based by default — can override via env)
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy source code
COPY --chown=appuser:appuser . .

# Remove development artifacts
RUN rm -rf venv/ .git/ __pycache__/ *.pyc

# Create writable directories for data, models, logs
RUN mkdir -p data models logs mlruns && \
    chown -R appuser:appuser /app

USER appuser

# Default command: show help
CMD ["python", "main.py", "--help"]

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import config; print('OK')" || exit 1
