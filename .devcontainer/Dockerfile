FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra dev --extra nlp --extra db --extra ml 2>/dev/null || uv sync --extra dev

RUN uv run python -m spacy download ru_core_news_sm 2>/dev/null || true

ENV PATH="/workspace/.venv/bin:$PATH"
