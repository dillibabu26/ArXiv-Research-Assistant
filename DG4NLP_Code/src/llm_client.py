"""
One LLM interface for both integration options:

  Option A - hosted API: any provider exposing the OpenAI-compatible
             `/chat/completions` endpoint. Pros: strong models, no local hardware.
             Cons: costs money per token, needs internet, models can change or be
             retired (hurting reproducibility), data leaves your machine.
  Option B - local open model via Ollama (also OpenAI-compatible, at
             http://localhost:11434/v1). Pros: free, private, a fixed model file
             (reproducible). Cons: needs RAM (roughly 2-8 GB for 1-8B parameter models),
             slower on CPU, smaller models are less accurate.

  Option C - local instruction-tuned model run directly with Hugging Face transformers
             (LLM_BACKEND=local; default google/flan-t5-large, ~3 GB download once).
             Pros: nothing to install beyond requirements.txt, no server, no key, fully
             reproducible. Cons: a much smaller model than API LLMs, 512-token input limit,
             cannot reliably produce JSON, so it receives prompts in its native
             "OPTIONS:" format (see llm_classification.build_messages_local).

Settings come from `.env` (LLM_BACKEND, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY).
With LLM_BACKEND=none, `available` is False and callers skip LLM features.

Every request/response is cached on disk (outputs/results/llm_cache.jsonl), so
re-running an experiment does not repeat paid calls and every reported LLM result
can be audited. The API key is never logged or cached.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import config
from src.utils import Timer, get_logger

logger = get_logger()
CACHE_FILENAME = "llm_cache.jsonl"
_DEFAULT = object()
LOCAL_MAX_INPUT_TOKENS = 512


@lru_cache(maxsize=2)
def _load_local_model(name: str):
    """Load a local seq2seq instruction model once (cached). Returns (model, tokenizer, device)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.utils import get_device

    device = get_device(config.FORCE_CPU)
    logger.info("Loading local LLM %s on %s (first time downloads the weights)", name, device)
    tok = AutoTokenizer.from_pretrained(name)
    tok.truncation_side = "left"   # keep the END of the prompt (the abstract + question) if too long
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
    model.eval()
    return model, tok, device


def flatten_messages(messages: list[dict]) -> str:
    """Turn chat messages into one prompt string for a non-chat model.
    Assistant turns (few-shot answers) become 'Answer: ...' lines."""
    parts = []
    for m in messages:
        if m["role"] == "assistant":
            parts.append(f"Answer: {m['content']}")
        else:
            parts.append(m["content"])
    return "\n\n".join(parts) + "\n\nAnswer:"


class LLMClient:
    def __init__(self, backend: str | None = None, base_url: str | None = None,
                 model: str | None = None, api_key: str | None = None,
                 timeout: int | None = None, cache_file=_DEFAULT) -> None:
        self.backend = (backend or config.LLM_BACKEND).lower()
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model if model is not None else config.LLM_MODEL
        if self.backend == "local":
            self.model = self.model or config.LLM_LOCAL_MODEL
            self.base_url = "local-transformers"
        self._local_generate = None   # injectable for tests
        self._api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.timeout = timeout or config.LLM_TIMEOUT_SECONDS
        if cache_file is _DEFAULT:  # resolved at run time so tests can redirect RESULTS_DIR
            cache_file = config.RESULTS_DIR / CACHE_FILENAME
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: dict[str, dict] = {}
        self.last_error: str | None = None
        if self.cache_file and self.cache_file.exists():
            for line in self.cache_file.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue

    # -- status --------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.backend in ("openai", "local") and bool(self.model)

    @property
    def available(self) -> bool:
        """True if configured (a connection is attempted only when chat() is called)."""
        return self.configured

    def describe(self) -> str:
        if not self.configured:
            return "LLM disabled (LLM_BACKEND=none or LLM_MODEL not set in .env)"
        if self.backend == "local":
            return f"{self.model} (local, transformers)"
        return f"{self.model} via {self.base_url}"

    def check_connection(self) -> tuple[bool, str]:
        """Send a tiny request to verify the endpoint and model respond."""
        if not self.configured:
            return False, self.describe()
        reply = self.chat([{"role": "user", "content": "Reply with the word OK."}], use_cache=False)
        return (reply is not None), (self.last_error or "connected")

    # -- requests -------------------------------------------------------------
    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict], temperature: float = config.LLM_TEMPERATURE,
             max_tokens: int = 300, use_cache: bool = True, retries: int = 2) -> str | None:
        """
        Return the assistant's reply text, or None on failure (error in `last_error`).
        temperature=0 asks for the most likely tokens, making outputs as repeatable as
        the provider allows (not guaranteed bit-identical across runs or versions).
        """
        self.last_error = None
        if not self.configured:
            self.last_error = self.describe()
            return None
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        key = self._key(payload)
        if use_cache and key in self._cache:
            return self._cache[key]["response"]

        if self.backend == "local":
            try:
                with Timer() as t:
                    text = self._generate_local(messages, max_tokens)
            except Exception as e:  # missing transformers/torch, download failure, OOM
                self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning("Local LLM failed: %s", self.last_error)
                return None
            self._store(key, messages, text, t.elapsed)
            return text

        import requests

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        for attempt in range(retries + 1):
            try:
                with Timer() as t:
                    r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                      headers=headers, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                self._store(key, messages, text, t.elapsed)
                return text
            except Exception as e:  # network error, timeout, bad JSON, HTTP error
                self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < retries:
                    time.sleep(2 ** attempt)
        logger.warning("LLM request failed: %s", self.last_error)
        return None

    def _store(self, key: str, messages: list[dict], text: str, seconds: float) -> None:
        rec = {"key": key, "model": self.model, "base_url": self.base_url,
               "messages": messages, "response": text, "latency_s": round(seconds, 3)}
        self._cache[key] = rec
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _generate_local(self, messages: list[dict], max_tokens: int) -> str:
        """Greedy decoding (the deterministic equivalent of temperature 0)."""
        if self._local_generate is not None:          # test hook
            return self._local_generate(flatten_messages(messages), max_tokens)
        import torch

        model, tok, device = _load_local_model(self.model)
        enc = tok(flatten_messages(messages), max_length=LOCAL_MAX_INPUT_TOKENS, truncation=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=min(max_tokens, 64), do_sample=False, num_beams=1)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    def cached_latency(self, messages: list[dict], temperature: float = config.LLM_TEMPERATURE,
                       max_tokens: int = 300) -> float | None:
        """Latency recorded when this exact request was first made (for timing tables)."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        rec = self._cache.get(self._key(payload))
        return rec.get("latency_s") if rec else None
