"""Tests for new OllamaRunner methods: score_window, get_sentence_embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiqyn.models.ollama_runner import OllamaRunner


@pytest.fixture
def runner() -> OllamaRunner:
    return OllamaRunner(model="qwen3:8b")


# ---------------------------------------------------------------------------
# score_window
# ---------------------------------------------------------------------------

class TestScoreWindow:
    def test_delegates_to_score_continuation(self, runner: OllamaRunner) -> None:
        """score_window must delegate to _score_continuation without transformation."""
        with patch.object(runner, "_score_continuation", return_value=-1.5) as mock_sc:
            result = runner.score_window("prefix text", "target text")
        mock_sc.assert_called_once_with("prefix text", "target text")
        assert result == -1.5

    def test_returns_float(self, runner: OllamaRunner) -> None:
        with patch.object(runner, "_score_continuation", return_value=-2.0):
            result = runner.score_window("a", "b")
        assert isinstance(result, float)

    def test_propagates_exception(self, runner: OllamaRunner) -> None:
        """Exceptions from _score_continuation should propagate to caller."""
        with patch.object(runner, "_score_continuation", side_effect=RuntimeError("net error")):
            with pytest.raises(RuntimeError, match="net error"):
                runner.score_window("p", "t")


# ---------------------------------------------------------------------------
# get_sentence_embeddings
# ---------------------------------------------------------------------------

class TestGetSentenceEmbeddings:
    def _mock_client_post(self, response_body: dict) -> MagicMock:
        mock_response = MagicMock()
        mock_response.json.return_value = response_body
        mock_response.raise_for_status.return_value = None
        return mock_response

    def test_returns_embeddings_for_sentences(self, runner: OllamaRunner) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_resp = self._mock_client_post({"embeddings": [vec]})
        with patch.object(runner._client, "post", return_value=mock_resp):
            result = runner.get_sentence_embeddings(["hello", "world"])
        assert result == [vec, vec]

    def test_returns_empty_on_http_error(self, runner: OllamaRunner) -> None:
        with patch.object(runner._client, "post", side_effect=Exception("timeout")):
            result = runner.get_sentence_embeddings(["text"])
        assert result == []

    def test_returns_empty_when_embeddings_key_missing(self, runner: OllamaRunner) -> None:
        mock_resp = self._mock_client_post({"embeddings": []})
        with patch.object(runner._client, "post", return_value=mock_resp):
            result = runner.get_sentence_embeddings(["text"])
        assert result == []

    def test_returns_empty_on_raise_for_status_error(self, runner: OllamaRunner) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        with patch.object(runner._client, "post", return_value=mock_resp):
            result = runner.get_sentence_embeddings(["text"])
        assert result == []

    def test_uses_embed_model_parameter(self, runner: OllamaRunner) -> None:
        """embed_model kwarg should be sent in the request payload."""
        vec = [0.5, 0.6]
        mock_resp = self._mock_client_post({"embeddings": [vec]})
        with patch.object(runner._client, "post", return_value=mock_resp) as mock_post:
            runner.get_sentence_embeddings(["text"], embed_model="custom-embed")
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "custom-embed"

    def test_empty_input_returns_empty_list(self, runner: OllamaRunner) -> None:
        result = runner.get_sentence_embeddings([])
        assert result == []

    def test_stops_on_first_error(self, runner: OllamaRunner) -> None:
        """If first sentence fails, return [] without calling for subsequent sentences."""
        with patch.object(runner._client, "post", side_effect=Exception("err")):
            result = runner.get_sentence_embeddings(["a", "b", "c"])
        assert result == []
