# =============================================================
# InkFlow-AI — Production Dockerfile
# =============================================================
# Base image: Python 3.12 slim (Debian Bookworm)
# Dependency manager: uv (deterministic, fast)
# Entry point: FastAPI via Uvicorn
# =============================================================

FROM python:3.12-slim

# -----------------------------------------------------------------
# Runtime environment
# -----------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Use uv-managed environment inside the container
    UV_SYSTEM_PYTHON=1 \
    # Suppress uv's progress bar for cleaner CI/CD build logs
    UV_NO_PROGRESS=1

# -----------------------------------------------------------------
# System dependencies (minimal, only what is strictly required)
# -----------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # uv uses curl for self-update checks; also useful for health probes
    curl \
    # Required by Pillow for image processing (libwebp, libjpeg headers)
    libffi-dev \
    # Required to compile native extensions for some packages
    gcc \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# Install uv from the official distroless image
# -----------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# -----------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------
# Install dependencies BEFORE copying application source
# This maximises Docker layer cache — a source-only change will
# not trigger a slow dependency reinstall.
# -----------------------------------------------------------------
COPY pyproject.toml uv.lock ./

# --frozen  -> honour the lockfile exactly (production-safe)
# --no-dev  -> skip development-only dependency groups
RUN uv sync --frozen --no-dev

# -----------------------------------------------------------------
# Copy application source (order matters for cache efficiency)
# -----------------------------------------------------------------
COPY app.py        ./
COPY src/          ./src/
COPY static/       ./static/
COPY templates/    ./templates/

# -----------------------------------------------------------------
# Outputs directory: created here so the path exists even if the
# Docker Compose bind-mount is absent. The bind-mount in
# docker-compose.yml overlays this directory at runtime so that
# generated blogs, images, and metadata persist on the host.
# -----------------------------------------------------------------
RUN mkdir -p /app/outputs

# -----------------------------------------------------------------
# Expose application port
# -----------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------
# Production startup command
#
# host 0.0.0.0 — required so Docker can route external traffic
#   to the container (127.0.0.1 would be unreachable from outside)
# reload is off by default — never enable hot-reload in production
# -----------------------------------------------------------------
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]