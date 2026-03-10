"""FastAPI application for Aiqyn REST API."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from aiqyn import __version__
from aiqyn.api.models import (
    AnalyzeRequest,
    DeleteResponse,
    HealthResponse,
    HistoryEntryResponse,
    StatusResponse,
)
from aiqyn.config import get_config
from aiqyn.core.analyzer import TextAnalyzer
from aiqyn.models.manager import get_model_manager
from aiqyn.schemas import AnalysisResult
from aiqyn.storage.database import HistoryRepository
from aiqyn.utils.file_reader import read_text_from_file, supported_extensions
from aiqyn.web.router import router as web_router

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_analyzer: TextAnalyzer | None = None
_repository: HistoryRepository | None = None

_STATIC_DIR = Path(__file__).parent.parent / "web" / "static"


def _get_analyzer() -> TextAnalyzer:
    if _analyzer is None:
        raise RuntimeError("TextAnalyzer not initialized — startup event not fired")
    return _analyzer


def _get_repository() -> HistoryRepository:
    if _repository is None:
        raise RuntimeError("HistoryRepository not initialized — startup event not fired")
    return _repository


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _analyzer, _repository

    log.info("api_startup", version=__version__)
    config = get_config()

    _analyzer = TextAnalyzer(config=config, use_llm=True, load_spacy=True)
    _repository = HistoryRepository()

    log.info("api_ready")
    yield

    log.info("api_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="Aiqyn API",
        description="Offline AI-generated text detector for Russian language",
        version=__version__,
        lifespan=lifespan,
    )

    # Allow localhost on any port (for local desktop usage and dev tools)
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://localhost(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files (CSS, JS assets if needed in the future)
    if _STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Web UI routes (must be included before API routes so / is not shadowed)
    application.include_router(web_router)

    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/status", response_model=StatusResponse, tags=["system"])
    async def status() -> StatusResponse:
        manager = get_model_manager()

        # Check Ollama availability without loading a model
        ollama_available = False
        try:
            from aiqyn.models.ollama_runner import OllamaRunner
            runner = OllamaRunner(
                model=get_config().ollama_model,
                base_url=get_config().ollama_base_url,
            )
            ollama_available = runner.is_available()
            runner.close()
        except Exception as exc:
            log.debug("ollama_check_failed", error=str(exc))

        return StatusResponse(
            status="ok",
            version=__version__,
            model=manager.model_name,
            ollama_available=ollama_available,
        )

    @application.post("/analyze", response_model=AnalysisResult, tags=["analysis"])
    async def analyze(request: AnalyzeRequest) -> AnalysisResult:
        """Analyze text for AI-generated content.

        Returns the full AnalysisResult including per-segment scores and feature breakdown.
        Saves the result to local history automatically.
        """
        analyzer = _get_analyzer()
        repo = _get_repository()

        # Override use_llm per-request: create a lightweight analyzer if LLM is disabled
        active_analyzer = analyzer
        if not request.use_llm and analyzer._use_llm:
            config = get_config()
            active_analyzer = TextAnalyzer(config=config, use_llm=False, load_spacy=True)

        try:
            result: AnalysisResult = active_analyzer.analyze(request.text)
        except Exception as exc:
            log.error("analyze_failed", error=str(exc), exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Persist to history (best-effort — do not fail the request on DB error)
        try:
            repo.save(request.text, result)
        except Exception as exc:
            log.warning("history_save_failed", error=str(exc))

        return result

    @application.post("/upload", tags=["analysis"])
    async def upload_file(file: UploadFile) -> dict[str, str]:
        """Extract text from an uploaded file (.txt, .docx, .pdf).

        Saves to a temporary file, reads with the existing file_reader utility,
        then cleans up. Returns {"text": "<extracted text>"}.
        """
        # Validate extension early to avoid unnecessary disk I/O
        original_name = file.filename or "upload"
        suffix = Path(original_name).suffix.lower()
        if suffix not in supported_extensions():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported file type: {suffix!r}. "
                    f"Allowed: {', '.join(supported_extensions())}"
                ),
            )

        # Stream the upload into a named temp file so file_reader can use Path
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                content = await file.read()
                tmp.write(content)
        except Exception as exc:
            log.error("upload_write_failed", error=str(exc))
            raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc

        try:
            text = read_text_from_file(tmp_path)
        except (ValueError, ImportError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.error("upload_read_failed", error=str(exc), exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to extract text from file") from exc
        finally:
            # Always clean up the temp file
            tmp_path.unlink(missing_ok=True)

        if not text.strip():
            raise HTTPException(status_code=422, detail="Extracted text is empty")

        log.info("upload_ok", filename=original_name, chars=len(text))
        return {"text": text}

    @application.get("/history", response_model=list[HistoryEntryResponse], tags=["history"])
    async def list_history(
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[HistoryEntryResponse]:
        """Return recent analysis history entries."""
        repo = _get_repository()
        # HistoryRepository.list() accepts only limit; we handle offset in-memory.
        # SQLite table is small (desktop app), so fetching limit+offset is acceptable.
        entries = repo.list(limit=limit + offset)
        sliced = entries[offset:]
        return [
            HistoryEntryResponse(
                id=e.id,
                created_at=e.created_at,
                text_preview=e.text_preview,
                overall_score=e.overall_score,
                verdict=e.verdict,
                confidence=e.confidence,
                word_count=e.word_count,
                model_used=e.model_used,
            )
            for e in sliced
        ]

    @application.get("/history/{entry_id}", tags=["history"])
    async def get_history_entry(entry_id: int) -> dict:
        """Return full result JSON for a history entry."""
        repo = _get_repository()
        entry = repo.get(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"History entry {entry_id} not found")
        import json
        return json.loads(entry.result_json)

    @application.delete("/history/{entry_id}", response_model=DeleteResponse, tags=["history"])
    async def delete_history(entry_id: int) -> DeleteResponse:
        """Delete a history entry by ID."""
        repo = _get_repository()
        deleted = repo.delete(entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"History entry {entry_id} not found")
        log.info("history_entry_deleted", id=entry_id)
        return DeleteResponse(deleted=True, id=entry_id)


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn "aiqyn.api.app:app")
# ---------------------------------------------------------------------------

app = create_app()
