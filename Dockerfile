FROM python:3.12-slim

# System deps for pypdf (no extra needed), python-docx (lxml → gcc), spaCy (gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project manifest and source first (layer caching)
COPY pyproject.toml .
COPY src/ src/

# Copy data and config (may not exist yet — COPY with || true not supported,
# so we create placeholders to make COPY happy)
COPY data/ data/
COPY config/ config/

# Install uv, then project dependencies (no dev extras)
RUN pip install --no-cache-dir uv
RUN uv sync --no-dev

# Download the Russian spaCy model used by preprocessing
RUN uv run python -m spacy download ru_core_news_sm

# Non-root user for security
RUN useradd -m -u 1000 aiqyn
USER aiqyn

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "aiqyn.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
