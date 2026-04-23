"""Tests for ModelManager secondary Ollama runner support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiqyn.models.manager import ModelManager


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset ModelManager singleton state between tests."""
    original = ModelManager._instance
    ModelManager._instance = None
    yield
    # Unload if loaded
    try:
        mgr = ModelManager()
        mgr.unload()
    except Exception:
        pass
    ModelManager._instance = original


def _make_runner_mock(available: bool = True, models: list[str] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.is_available.return_value = available
    mock.list_models.return_value = models or ["qwen3:1.7b"]
    mock.close.return_value = None
    return mock


# ---------------------------------------------------------------------------
# get_ollama_secondary — initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_secondary_is_none_by_default(self) -> None:
        mgr = ModelManager()
        assert mgr.get_ollama_secondary() is None

    def test_secondary_model_name_is_none(self) -> None:
        mgr = ModelManager()
        assert mgr._secondary_model_name is None


# ---------------------------------------------------------------------------
# load_ollama_secondary
# ---------------------------------------------------------------------------

class TestLoadSecondary:
    # OllamaRunner is imported locally inside load_ollama_secondary, so patch at source module
    def test_returns_false_when_ollama_unavailable(self) -> None:
        mgr = ModelManager()
        with patch(
            "aiqyn.models.ollama_runner.OllamaRunner",
            return_value=_make_runner_mock(available=False),
        ):
            result = mgr.load_ollama_secondary("qwen3:1.7b")
        assert result is False
        assert mgr.get_ollama_secondary() is None

    def test_returns_false_when_model_not_found(self) -> None:
        """Use a model name with no prefix overlap with any available model."""
        mgr = ModelManager()
        with patch(
            "aiqyn.models.ollama_runner.OllamaRunner",
            return_value=_make_runner_mock(available=True, models=["llama3:8b"]),
        ):
            result = mgr.load_ollama_secondary("mistral:7b")
        assert result is False

    def test_returns_true_when_model_found(self) -> None:
        mgr = ModelManager()
        with patch(
            "aiqyn.models.ollama_runner.OllamaRunner",
            return_value=_make_runner_mock(available=True, models=["qwen3:1.7b"]),
        ):
            result = mgr.load_ollama_secondary("qwen3:1.7b")
        assert result is True
        assert mgr.get_ollama_secondary() is not None

    def test_secondary_runner_stored(self) -> None:
        mgr = ModelManager()
        mock_runner = _make_runner_mock(available=True, models=["qwen3:1.7b"])
        with patch("aiqyn.models.ollama_runner.OllamaRunner", return_value=mock_runner):
            mgr.load_ollama_secondary("qwen3:1.7b")
        assert mgr.get_ollama_secondary() is mock_runner

    def test_replaces_existing_secondary(self) -> None:
        """Calling load_ollama_secondary twice replaces the first runner."""
        mgr = ModelManager()
        mock1 = _make_runner_mock(available=True, models=["qwen3:1.7b"])
        mock2 = _make_runner_mock(available=True, models=["qwen3:1.7b"])

        with patch("aiqyn.models.ollama_runner.OllamaRunner", side_effect=[mock1, mock2]):
            mgr.load_ollama_secondary("qwen3:1.7b")
            mgr.load_ollama_secondary("qwen3:1.7b")

        assert mgr.get_ollama_secondary() is mock2
        mock1.close.assert_called()


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_closes_secondary(self) -> None:
        mgr = ModelManager()
        mock_runner = _make_runner_mock(available=True, models=["qwen3:1.7b"])
        with patch("aiqyn.models.ollama_runner.OllamaRunner", return_value=mock_runner):
            mgr.load_ollama_secondary("qwen3:1.7b")

        mgr.unload()
        mock_runner.close.assert_called()
        assert mgr.get_ollama_secondary() is None
        assert mgr._secondary_model_name is None

    def test_unload_without_secondary_does_not_raise(self) -> None:
        mgr = ModelManager()
        mgr.unload()  # secondary is None — should not raise


# ---------------------------------------------------------------------------
# auto_load integration
# ---------------------------------------------------------------------------

class TestAutoLoad:
    # get_config is imported locally inside auto_load, patch at source module
    def test_auto_load_attempts_secondary_after_primary(self) -> None:
        """auto_load should call load_ollama_secondary after primary succeeds."""
        mgr = ModelManager()

        with patch.object(mgr, "load_ollama", return_value=True) as mock_primary, \
             patch.object(mgr, "load_ollama_secondary", return_value=True) as mock_secondary, \
             patch("aiqyn.config.get_config") as mock_cfg:
            mock_cfg.return_value.ollama_model = "qwen3:8b"
            mock_cfg.return_value.ollama_base_url = "http://localhost:11434"
            mock_cfg.return_value.ollama_secondary_model = "qwen3:1.7b"
            mock_cfg.return_value.resolve_model_path.return_value = None
            mgr.auto_load()

        mock_primary.assert_called_once()
        mock_secondary.assert_called_once_with(model="qwen3:1.7b", base_url="http://localhost:11434")

    def test_auto_load_skips_secondary_when_primary_fails(self) -> None:
        """If primary Ollama fails, secondary should not be attempted."""
        mgr = ModelManager()

        with patch.object(mgr, "load_ollama", return_value=False), \
             patch.object(mgr, "load_ollama_secondary") as mock_secondary, \
             patch("aiqyn.config.get_config") as mock_cfg:
            mock_cfg.return_value.ollama_model = "qwen3:8b"
            mock_cfg.return_value.ollama_base_url = "http://localhost:11434"
            mock_cfg.return_value.ollama_secondary_model = "qwen3:1.7b"
            mock_cfg.return_value.resolve_model_path.return_value = None
            mgr.auto_load()

        mock_secondary.assert_not_called()

    def test_auto_load_skips_secondary_when_empty_model_name(self) -> None:
        """Empty ollama_secondary_model should skip secondary loading."""
        mgr = ModelManager()

        with patch.object(mgr, "load_ollama", return_value=True), \
             patch.object(mgr, "load_ollama_secondary") as mock_secondary, \
             patch("aiqyn.config.get_config") as mock_cfg:
            mock_cfg.return_value.ollama_model = "qwen3:8b"
            mock_cfg.return_value.ollama_base_url = "http://localhost:11434"
            mock_cfg.return_value.ollama_secondary_model = ""  # disabled
            mgr.auto_load()

        mock_secondary.assert_not_called()
