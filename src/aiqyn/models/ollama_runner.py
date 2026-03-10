"""OllamaRunner — Ollama HTTP API wrapper for LLM-based feature extraction."""

from __future__ import annotations

import math
from typing import Iterator

import httpx
import structlog

log = structlog.get_logger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaRunner:
    """Calls Ollama API to compute logprobs and perplexity-like scores."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def is_available(self) -> bool:
        try:
            r = self._client.get("/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = self._client.get("/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def compute_pseudo_perplexity(self, text: str, max_words: int = 300) -> float:
        """Compute pseudo-perplexity via sliding window prediction.

        Strategy: split text into (prefix, target) pairs, ask the model to
        continue from prefix, collect logprobs of generated tokens that match
        the target text. Average negative log-prob → pseudo-perplexity.

        This is NOT true perplexity but strongly correlates with AI-ness.
        """
        words = text.split()
        if len(words) < 15:
            return 20.0

        words = words[:max_words]
        prefix_size = min(30, len(words) // 3)
        window_size = min(15, (len(words) - prefix_size) // 3)

        if window_size < 3:
            return self._compression_proxy(text)

        total_logprob = 0.0
        n_samples = 0

        positions = list(range(prefix_size, len(words) - window_size, window_size))
        positions = positions[:8]  # max 8 windows to stay fast

        for pos in positions:
            prefix_text = " ".join(words[max(0, pos - prefix_size):pos])
            target_text = " ".join(words[pos:pos + window_size])

            try:
                lp = self._score_continuation(prefix_text, target_text)
                total_logprob += lp
                n_samples += 1
            except Exception as exc:
                log.debug("ollama_window_failed", pos=pos, error=str(exc))

        if n_samples == 0:
            return self._compression_proxy(text)

        avg_neg_logprob = -total_logprob / n_samples
        perplexity = math.exp(min(avg_neg_logprob, 10.0))
        log.debug("pseudo_perplexity", value=perplexity, windows=n_samples)
        return perplexity

    def _score_continuation(self, prefix: str, target: str) -> float:
        """Score how well the model predicts the target text from the prefix.

        Uses word-overlap between the model's greedy continuation and the
        actual target.  This avoids the greedy-logprob collapse (with
        temperature=0 the model always picks the top token so logprob≈0 for
        every text regardless of origin).

        Returns a pseudo log-prob value in range ~ [-4, 0]:
            near 0     → high overlap → predictable → AI-like
            very negative → low overlap → unpredictable → human-like
        """
        prompt = f"Продолжи текст: {prefix}"
        n_words = len(target.split())
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": n_words + 5,
                "num_ctx": 2048,
            },
        }
        if "qwen3" in self.model:
            payload["think"] = False  # disable extended thinking for speed

        r = self._client.post("/api/generate", json=payload)
        r.raise_for_status()
        generated = r.json().get("response", "").strip()

        # Word-level overlap: fraction of target words that appear in generated
        gen_words = set(generated.lower().split())
        tgt_words = [w.lower() for w in target.split() if w.isalpha()]
        if not gen_words or not tgt_words:
            return -3.0  # neutral fallback

        overlap = sum(1 for w in tgt_words if w in gen_words) / len(tgt_words)

        # Map overlap [0, 1] → pseudo log-prob ~ [-4, 0]:
        # high overlap (AI text) → near 0
        # low overlap (human text) → near -4
        return math.log(max(overlap, 0.018))  # log(0.018) ≈ -4.0

    def get_token_ranks(self, text: str) -> list[float]:
        """Get normalized rank scores for text tokens via Ollama.

        Returns list of values 0–1 where 0 = top-1 prediction (AI-like),
        1 = unexpected token (human-like).
        """
        words = text.split()[:100]
        prefix_size = 10
        ranks: list[float] = []

        for pos in range(prefix_size, min(len(words), prefix_size + 40), 3):
            prefix = " ".join(words[max(0, pos - prefix_size):pos])
            try:
                payload = {
                    "model": self.model,
                    "prompt": prefix,
                    "stream": False,
                    "logprobs": True,
                    "options": {"temperature": 0.0, "num_predict": 3, "num_ctx": 512},
                }
                if "qwen3" in self.model:
                    payload["think"] = False
                r = self._client.post("/api/generate", json=payload)
                data = r.json()
                lps = [e["logprob"] for e in data.get("logprobs", [])]
                if lps:
                    # High logprob (near 0) → top prediction → AI-like → rank ≈ 0
                    avg_lp = sum(lps) / len(lps)
                    # Map [-10, 0] → [1, 0]
                    rank = max(0.0, min(1.0, -avg_lp / 10.0))
                    ranks.append(rank)
            except Exception:
                pass

        return ranks

    @staticmethod
    def _compression_proxy(text: str) -> float:
        """Fallback: zlib compression ratio as perplexity proxy.

        Lower compression ratio → more compressible → more AI-like → lower perplexity.
        """
        import zlib
        encoded = text.encode("utf-8")
        if len(encoded) < 20:
            return 20.0
        compressed = zlib.compress(encoded, level=9)
        ratio = len(compressed) / len(encoded)
        # Typical AI: 0.45–0.55, Human: 0.60–0.75
        # Map to pseudo-perplexity: 5–50
        perplexity = 5.0 + ratio * 60.0
        return perplexity

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaRunner":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
