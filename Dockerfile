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

# Install pre-downloaded spaCy Russian model from vendor/ (offline-friendly).
# Populate vendor/ on the host first: ./scripts/download_assets.sh
# If the wheel is missing we emit a warning and skip — the pipeline
# degrades gracefully (F-06 disabled, F-12 falls back to surface tokens).
COPY vendor/ vendor/
RUN if ls vendor/spacy/*.whl >/dev/null 2>&1; then \
      uv pip install --python .venv/bin/python vendor/spacy/*.whl; \
    else \
      echo "WARN: vendor/spacy/ is empty — spaCy model not installed."; \
      echo "      Run ./scripts/download_assets.sh on host and rebuild."; \
    fi

# Non-root user for security. Ensure /app and the venv are owned by it
# so runtime operations (logging, SQLite writes) don't need root.
RUN useradd -m -u 1000 aiqyn && chown -R aiqyn:aiqyn /app
USER aiqyn

# Put the installed venv on PATH so uvicorn/aiqyn are resolved directly,
# without `uv run` (which re-syncs and would need write access to .venv).
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "aiqyn.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
