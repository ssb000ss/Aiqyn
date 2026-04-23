"""ModelManager — unified backend for LLM inference (Ollama or llama-cpp)."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Literal

import structlog

log = structlog.get_logger(__name__)


class ModelManager:
    """Singleton. Manages Ollama runner or llama-cpp Llama instance."""

    _instance: ModelManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._ollama: object | None = None
                    obj._ollama_secondary: object | None = None
                    obj._secondary_model_name: str | None = None
                    obj._llama: object | None = None
                    obj._backend: Literal["ollama", "llama_cpp", "none"] = "none"
                    obj._model_name: str | None = None
                    obj._llm_lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def load_ollama(
        self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"
    ) -> bool:
        from aiqyn.models.ollama_runner import OllamaRunner
        runner = OllamaRunner(model=model, base_url=base_url)
        if not runner.is_available():
            log.warning("ollama_not_available", url=base_url)
            runner.close()
            return False

        models = runner.list_models()
        if model not in models:
            # try prefix match
            match = next((m for m in models if m.startswith(model.split(":")[0])), None)
            if match:
                log.info("ollama_model_alias", requested=model, found=match)
                model = match
                runner = OllamaRunner(model=model, base_url=base_url)
            else:
                log.warning("ollama_model_not_found", model=model, available=models)
                runner.close()
                return False

        with self._llm_lock:
            if self._ollama:
                with contextlib.suppress(Exception):
                    self._ollama.close()  # type: ignore[union-attr]
            self._ollama = runner
            self._backend = "ollama"
            self._model_name = model
            log.info("ollama_loaded", model=model)
        return True

    def load_ollama_secondary(
        self,
        model: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
    ) -> bool:
        """Load a second OllamaRunner for Binoculars-style dual-model scoring (f16).

        Fails silently — callers must check get_ollama_secondary() for None.
        The primary model must already be loaded before calling this.
        """
        from aiqyn.models.ollama_runner import OllamaRunner

        runner = OllamaRunner(model=model, base_url=base_url)
        if not runner.is_available():
            runner.close()
            return False

        models = runner.list_models()
        if model not in models:
            match = next((m for m in models if m.startswith(model.split(":")[0])), None)
            if match:
                log.info("ollama_secondary_model_alias", requested=model, found=match)
                model = match
                runner.close()
                runner = OllamaRunner(model=model, base_url=base_url)
            else:
                log.warning("ollama_secondary_model_not_found", model=model, available=models)
                runner.close()
                return False

        with self._llm_lock:
            if self._ollama_secondary:
                with contextlib.suppress(Exception):
                    self._ollama_secondary.close()  # type: ignore[union-attr]
            self._ollama_secondary = runner
            self._secondary_model_name = model
            log.info("ollama_secondary_loaded", model=model)
        return True

    def get_ollama_secondary(self) -> object | None:
        """Return the secondary OllamaRunner, or None if not loaded."""
        return self._ollama_secondary

    def load_llama_cpp(self, model_path: Path) -> bool:
        try:
            from llama_cpp import Llama
        except ImportError:
            log.warning("llama_cpp_not_installed")
            return False

        if not model_path.exists():
            log.warning("gguf_not_found", path=str(model_path))
            return False

        from aiqyn.config import get_config
        cfg = get_config()
        with self._llm_lock:
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=cfg.context_size,
                n_gpu_layers=cfg.gpu_layers,
                logits_all=True,
                verbose=False,
            )
            self._backend = "llama_cpp"
            self._model_name = model_path.name
            log.info("llama_cpp_loaded", model=model_path.name)
        return True

    def auto_load(self) -> bool:
        """Try Ollama first (primary + secondary), then llama-cpp from config path."""
        from aiqyn.config import get_config
        cfg = get_config()

        if self.load_ollama(model=cfg.ollama_model, base_url=cfg.ollama_base_url):
            # Attempt secondary runner — failure is non-fatal
            if cfg.ollama_secondary_model:
                self.load_ollama_secondary(
                    model=cfg.ollama_secondary_model,
                    base_url=cfg.ollama_base_url,
                )
            return True

        path = cfg.resolve_model_path()
        if path:
            return self.load_llama_cpp(path)
        log.warning("no_llm_backend_available")
        return False

    def get_ollama(self) -> object | None:
        return self._ollama if self._backend == "ollama" else None

    def get_llama(self) -> object | None:
        return self._llama if self._backend == "llama_cpp" else None

    def unload(self) -> None:
        with self._llm_lock:
            if self._ollama:
                with contextlib.suppress(Exception):
                    self._ollama.close()  # type: ignore[union-attr]
                self._ollama = None
            if self._ollama_secondary:
                with contextlib.suppress(Exception):
                    self._ollama_secondary.close()  # type: ignore[union-attr]
                self._ollama_secondary = None
                self._secondary_model_name = None
            self._llama = None
            self._backend = "none"
            self._model_name = None
            log.info("model_unloaded")


def get_model_manager() -> ModelManager:
    return ModelManager()
