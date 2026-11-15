"""LLM response parsing, client behaviour (mocked HTTP) and prompt construction."""

import json

import pandas as pd
import pytest

from src.llm_classification import build_messages, classify_with_llm, parse_llm_json
from src.llm_client import LLMClient
from src.llm_title_generation import (PROMPT_STYLES, build_messages as title_messages,
                                      generate_title_llm, select_few_shot_examples)

LABELS = ["cs.CL", "cs.CV", "quant-ph", "hep-th"]


@pytest.mark.parametrize("raw,cat,method", [
    ('{"category": "cs.CL", "explanation": "NLP task", "confidence": "high"}', "cs.CL", "json"),
    ('```json\n{"category": "quant-ph", "explanation": "qubits"}\n```', "quant-ph", "json"),
    ('Sure! Here is the answer: {"category": "cs.cv", "explanation": "images"} Hope it helps.', "cs.CV", "json"),
    ("{'category': 'hep-th', 'explanation': 'strings', 'confidence': 'low'}", "hep-th", "json"),
    ("The best category is quant-ph because of qubits.", "quant-ph", "label_search"),
])
def test_valid_responses(raw, cat, method):
    out = parse_llm_json(raw, LABELS)
    assert out["valid"] and out["category"] == cat and out["parse_method"] == method


@pytest.mark.parametrize("raw,method", [
    ('{"category": "physics.optics", "explanation": "x"}', "json_invalid_label"),
    ("It could be cs.CL or cs.CV.", "failed"),
    ("I cannot decide.", "failed"),
    ("", "empty"), (None, "empty"),
])
def test_invalid_responses_are_never_guessed(raw, method):
    out = parse_llm_json(raw, LABELS)
    assert not out["valid"] and out["category"] is None and out["parse_method"] == method


def test_label_search_does_not_match_substrings():
    # 'cs.CL' must not be found inside 'cs.CLX' or similar longer codes
    assert parse_llm_json("Category: cs.CLX", LABELS)["valid"] is False


def test_confidence_only_accepts_known_values():
    out = parse_llm_json('{"category": "cs.CL", "confidence": "93%"}', LABELS)
    assert out["valid"] and out["confidence"] is None


def test_classification_prompt_lists_all_labels_and_truncates():
    msgs = build_messages("word " * 1000, LABELS)
    user = msgs[-1]["content"]
    assert all(l in user for l in LABELS) and "JSON" in user
    assert user.count("word") <= 360


def test_unconfigured_client_returns_none():
    c = LLMClient(backend="none", cache_file=None)
    assert not c.available
    assert c.chat([{"role": "user", "content": "hi"}]) is None
    assert "disabled" in c.last_error.lower()


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _ok(text):
    return _Resp(200, {"choices": [{"message": {"content": text}}]})


def test_client_success_cache_and_no_key_leak(monkeypatch, tmp_path):
    import requests

    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, headers))
        return _ok('{"category": "cs.CL", "explanation": "text"}')

    monkeypatch.setattr(requests, "post", fake_post)
    cache = tmp_path / "cache.jsonl"
    c = LLMClient(backend="openai", base_url="http://x/v1", model="m", api_key="SECRET", cache_file=cache)
    res = classify_with_llm("A study of machine translation for low resource languages.", c, LABELS)
    assert res["category"] == "cs.CL" and res["error"] is None
    assert calls[0][0] == "http://x/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer SECRET"
    classify_with_llm("A study of machine translation for low resource languages.", c, LABELS)
    assert len(calls) == 1                                   # second call served from cache
    assert "SECRET" not in cache.read_text()
    c2 = LLMClient(backend="openai", base_url="http://x/v1", model="m", cache_file=cache)
    assert c2.chat(build_messages("A study of machine translation for low resource languages.", LABELS),
                   max_tokens=200) is not None and len(calls) == 1   # persisted cache


def test_client_retries_then_fails_gracefully(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(503, {}))
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = LLMClient(backend="openai", base_url="http://x/v1", model="m", cache_file=None)
    assert c.chat([{"role": "user", "content": "hi"}], retries=2) is None
    assert "503" in c.last_error


def test_title_prompt_styles(monkeypatch):
    examples = [{"abstract": "Example abstract text.", "title": "Example Title", "id": "1"}]
    zs = title_messages("Abstract here.", "zero_shot")
    fs = title_messages("Abstract here.", "few_shot", examples)
    ins = title_messages("Abstract here.", "instructed")
    assert len(zs) == 1 and len(ins) == 2 and len(fs) == 4
    assert fs[2] == {"role": "assistant", "content": "Example Title"}
    assert "do not add results" in ins[-1]["content"]
    with pytest.raises(ValueError):
        title_messages("x", "chain_of_thought")
    assert set(PROMPT_STYLES) == {"zero_shot", "instructed", "few_shot"}


def test_generate_title_llm_postprocesses(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _ok('Title: "Bond-Angle Message Passing."'))
    c = LLMClient(backend="openai", base_url="http://x/v1", model="m", cache_file=None)
    assert generate_title_llm("Some abstract.", c, "instructed") == "Bond-Angle Message Passing"


def test_few_shot_examples_come_from_given_training_frame(labelled_df):
    df = labelled_df.assign(title_words=10, abstract_words=100)
    ex = select_few_shot_examples(df, n=3)
    assert len(ex) == 3 and set(e["id"] for e in ex) <= set(df["id"])


def test_few_shot_examples_cover_different_fields():
    rows = []
    for label, n in [("cs.LG", 9), ("cs.CV", 8), ("quant-ph", 7), ("hep-ph", 6), ("math.AP", 5)]:
        for i in range(n):
            rows.append({"id": f"{label}-{i}", "label": label, "title": "A title of ten words for testing purposes here",
                         "abstract": "word " * 100, "title_words": 10, "abstract_words": 100})
    ex = select_few_shot_examples(pd.DataFrame(rows), n=3)
    labels = [e["id"].rsplit("-", 1)[0] for e in ex]
    assert labels == ["cs.LG", "quant-ph", "math.AP"]     # one per field, most frequent first


# --- local (transformers) backend -------------------------------------------------
@pytest.mark.parametrize("raw,cat,method", [
    ("cs.CL", "cs.CL", "label_search"),
    ("Computation and Language", "cs.CL", "description_search"),
    ("quantum physics", "quant-ph", "description_search"),
    ("High Energy Physics - Theory", "hep-th", "description_search"),
])
def test_parser_accepts_category_names(raw, cat, method):
    labels = ["cs.CL", "cs.CV", "quant-ph", "hep-th", "hep-ph"]
    out = parse_llm_json(raw, labels)
    assert out["valid"] and out["category"] == cat and out["parse_method"] == method


def test_parser_rejects_two_category_names():
    out = parse_llm_json("Computation and Language or Quantum Physics", ["cs.CL", "quant-ph"])
    assert not out["valid"]


def test_local_backend_end_to_end_with_fake_model(tmp_path):
    from src.llm_client import flatten_messages

    seen = []
    c = LLMClient(backend="local", model="fake-t5", cache_file=tmp_path / "c.jsonl")
    c._local_generate = lambda prompt, n: (seen.append(prompt) or "quant-ph")
    assert c.available and "local" in c.describe()
    res = classify_with_llm("We entangle two qubits with photons in a cavity. " * 5, c, LABELS)
    assert res["category"] == "quant-ph" and res["valid"] and res["latency_s"] is not None
    assert "OPTIONS:" in seen[0] and seen[0].endswith("Answer:")
    # cached: no second generation
    classify_with_llm("We entangle two qubits with photons in a cavity. " * 5, c, LABELS)
    assert len(seen) == 1
    # few-shot answers become 'Answer:' lines
    flat = flatten_messages([{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "T1"},
                             {"role": "user", "content": "Q2"}])
    assert flat == "Q1\n\nAnswer: T1\n\nQ2\n\nAnswer:"


def test_local_backend_failure_is_reported_not_raised(tmp_path):
    c = LLMClient(backend="local", model="fake-t5", cache_file=None)

    def boom(prompt, n):
        raise RuntimeError("no weights")

    c._local_generate = boom
    assert c.chat([{"role": "user", "content": "hi"}]) is None and "no weights" in c.last_error
