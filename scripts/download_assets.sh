#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# download_assets.sh
#   Fetch everything needed for an offline Docker build into ./vendor/.
#   Run once on a machine with network, then `docker compose build` can
#   complete without any outbound requests.
#
# Assets collected:
#   - spaCy ru_core_news_sm wheel  → vendor/spacy/
#
# Usage:
#   ./scripts/download_assets.sh                 # default version
#   SPACY_MODEL_VERSION=3.8.0 ./scripts/...      # pin spaCy model version
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="${ROOT}/vendor"
SPACY_DIR="${VENDOR_DIR}/spacy"

mkdir -p "${SPACY_DIR}"

SPACY_VERSION="${SPACY_MODEL_VERSION:-3.8.0}"
SPACY_WHEEL="ru_core_news_sm-${SPACY_VERSION}-py3-none-any.whl"
SPACY_URL="https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-${SPACY_VERSION}/${SPACY_WHEEL}"
SPACY_FILE="${SPACY_DIR}/${SPACY_WHEEL}"

# ── helpers ──────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[assets]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m   %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[fail]\033[0m   %s\n' "$*" >&2; exit 1; }

fetch() {
  local url="$1" dst="$2"
  curl -fL --retry 5 --retry-delay 3 --retry-connrefused \
       --connect-timeout 30 --max-time 600 \
       -o "${dst}" "${url}"
}

# ── spaCy model ──────────────────────────────────────────────────────
if [[ -f "${SPACY_FILE}" ]]; then
  log "spaCy wheel already present ($(stat -f '%z' "${SPACY_FILE}" 2>/dev/null \
       || stat -c '%s' "${SPACY_FILE}") bytes) — skip"
else
  log "downloading spaCy ru_core_news_sm ${SPACY_VERSION}..."
  if ! fetch "${SPACY_URL}" "${SPACY_FILE}"; then
    warn "primary download failed; retrying via pip index"
    rm -f "${SPACY_FILE}"
    # Last-ditch fallback: let pip resolve from PyPI (works if wheel mirrored)
    if ! pip download --no-deps --dest "${SPACY_DIR}" \
         "ru_core_news_sm==${SPACY_VERSION}" 2>/dev/null; then
      fail "could not fetch spaCy model. Retry later or pin another version."
    fi
  fi
  log "spaCy wheel saved to vendor/spacy/${SPACY_WHEEL}"
fi

# ── summary ──────────────────────────────────────────────────────────
log "done. Contents of vendor/:"
find "${VENDOR_DIR}" -type f -exec ls -lh {} \; | awk '{printf "    %-10s %s\n", $5, $9}'

log "next: docker compose -f docker/docker-compose.yml build"
