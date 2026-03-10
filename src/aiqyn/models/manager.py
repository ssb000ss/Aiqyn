"""ModelManager — singleton for llama-cpp-python Llama instance."""

from __future__ import annotations

import threading
from pathlib import Path

import structlog

from aiqyn.config import get_config

log = structlog.get_logger(__name__)


class ModelManager:
    """Manages a single Llama instance (lazy-loaded, thread-safe)."""

    _instance: "ModelManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._llm = None
                    cls._instance._model_path = None
                    cls._instance._llm_lock = threading.Lock()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    def load(self, model_path: Path | None = None) -> bool:
        """Load LLM. Returns True on success, False if model not found."""
        try:
            from llama_cpp import Llama
        except ImportError:
            log.warning("llama_cpp_not_installed", hint="pip install llama-cpp-python")
            return False

        config = get_config()
        path = model_path or config.resolve_model_path()

        if path is None or not path.exists():
            log.warning("model_not_found", searched=str(path))
            return False

        with self._llm_lock:
            if self._llm is not None and self._model_path == path:
                log.debug("model_already_loaded", path=str(path))
                return True

            log.info("model_loading", path=str(path))
            self._llm = Llama(
                model_path=str(path),
                n_ctx=config.context_size,
                n_gpu_layers=config.gpu_layers,
                logits_all=True,
                verbose=False,
            )
            self._model_path = path
            log.info("model_loaded", path=str(path))
            return True

    def get_llm(self) -> "object | None":
        return self._llm

    def unload(self) -> None:
        with self._llm_lock:
            self._llm = None
            self._model_path = None
            log.info("model_unloaded")


def get_model_manager() -> ModelManager:
    return ModelManager()
