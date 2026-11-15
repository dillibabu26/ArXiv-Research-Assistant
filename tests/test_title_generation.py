"""Input validation, baselines, FLAN-T5 generation loop (mocked) and title metrics."""

import sys
import types
from contextlib import nullcontext

import pytest

from src.evaluation import rouge_scores, unsupported_term_rate
from src.title_generation import (centroid_title, generate_title, generate_titles, lead_title,
                                  postprocess_title, split_sentences, validate_abstract)

ABSTRACT = ("Graph neural networks have become the standard tool for learning on molecules. "
            "We propose a message passing architecture that encodes bond angles. "
            "Experiments on two benchmarks show improved accuracy for property prediction.")


class TestValidation:
    def test_valid(self):
        assert validate_abstract(ABSTRACT) == (True, "OK")

    @pytest.mark.parametrize("text", ["", "   ", None, 42])
    def test_empty_or_not_text(self, text):
        assert validate_abstract(text)[0] is False

    def test_too_short(self):
        ok, msg = validate_abstract("Only five words are here.")
        assert not ok and "too short" in msg

    def test_too_long(self):
        ok, msg = validate_abstract("word " * 3000)
        assert not ok and "too long" in msg

    def test_not_natural_language(self):
        assert validate_abstract("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22")[0] is False


def test_sentence_split_and_baselines():
    assert len(split_sentences(ABSTRACT)) == 3
    assert lead_title(ABSTRACT) == "Graph neural networks have become the standard tool for learning on molecules"
    assert len(centroid_title(ABSTRACT).split()) <= 12


@pytest.mark.parametrize("raw,clean", [
    ('Title: "Deep Learning for Molecules."', "Deep Learning for Molecules"),
    ("**A Study**\nextra line", "A Study"), ("", "")])
def test_postprocess_title(raw, clean):
    assert postprocess_title(raw) == clean


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        self.seen.append((texts, max_length, truncation))
        return _FakeBatch(input_ids=[[1]] * len(texts))

    def batch_decode(self, outputs, skip_special_tokens=True):
        return [f"Title: Generated title {i}." for i in range(len(outputs))]


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.params = None

    def generate(self, input_ids, **params):
        self.params = params
        return list(range(len(input_ids)))


@pytest.fixture
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=nullcontext))


def test_generate_titles_batches_truncates_and_postprocesses(fake_torch):
    import config

    tok, model = _FakeTokenizer(), _FakeModel()
    out = generate_titles([ABSTRACT] * 10, model, tok, "cpu", batch_size=4)
    assert out[0] == "Generated title 0" and len(out) == 10
    assert len(tok.seen) == 3                                   # 4 + 4 + 2
    texts, max_len, trunc = tok.seen[0]
    assert texts[0].startswith(config.TITLE_PROMPT) and max_len == config.TITLE_MAX_INPUT_TOKENS and trunc
    assert model.params["num_beams"] == config.TITLE_GENERATION_PARAMS["num_beams"]


def test_generate_title_rejects_invalid_input(fake_torch):
    with pytest.raises(ValueError):
        generate_title("too short", model=_FakeModel(), tokenizer=_FakeTokenizer())


def test_rouge_identical_and_disjoint():
    s = rouge_scores(["graph neural networks for molecules", "a b c"],
                     ["graph neural networks for molecules", "x y z"])
    assert s.loc[0, "rouge1"] == pytest.approx(1.0) and s.loc[0, "rougeL"] == pytest.approx(1.0)
    assert s.loc[1, "rouge1"] == 0.0


def test_unsupported_term_rate():
    assert unsupported_term_rate(ABSTRACT, "Message passing for molecule property prediction") == 0.0
    assert unsupported_term_rate(ABSTRACT, "Quantum cryptography protocols") == 1.0
