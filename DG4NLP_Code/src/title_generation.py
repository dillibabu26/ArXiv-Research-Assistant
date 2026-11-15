"""
Abstract-to-title generation.

Main method: FLAN-T5 (Chung et al., 2022), an instruction-tuned sequence-to-sequence
transformer, used zero-shot (no fine-tuning) so it runs on a CPU laptop.

Two deterministic extractive baselines give a lower bound that needs no model:
  * lead:     the abstract's first sentence, truncated to TITLE_BASELINE_WORDS words
  * centroid: the sentence most similar (TF-IDF cosine) to the whole abstract, truncated

Why title generation is not ordinary summarisation: a summary condenses the content of a
text into a shorter text of the same kind (sentences), whereas a title is a short noun
phrase (median ~9 words on arXiv) that names the topic, method or main contribution. It
often uses words or formulations that do not appear verbatim in the abstract and omits
results entirely, so it is more abstractive and compressive, and n-gram overlap (ROUGE)
with the one reference title is only a partial measure of quality.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

import config
from src.preprocessing import clean_for_transformer
from src.utils import get_device, get_logger

logger = get_logger()

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
TITLE_BASELINE_WORDS = 12


# ---------------------------------------------------------------------------
# Input validation (shared with the Streamlit UI)
# ---------------------------------------------------------------------------
def validate_abstract(text: str, min_words: int = config.MIN_INPUT_WORDS,
                      max_chars: int = config.MAX_INPUT_CHARS) -> tuple[bool, str]:
    """Return (is_valid, message). Rejects empty, too short, too long or non-text input."""
    if not isinstance(text, str) or not text.strip():
        return False, "Please enter an abstract."
    text = text.strip()
    if len(text) > max_chars:
        return False, f"The abstract is too long ({len(text):,} characters; maximum {max_chars:,})."
    n_words = len(text.split())
    if n_words < min_words:
        return False, f"The abstract is too short ({n_words} words; at least {min_words} needed)."
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(1, len(text)) < 0.5:
        return False, "The input does not look like natural-language text."
    return True, "OK"


# ---------------------------------------------------------------------------
# Extractive baselines
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    text = clean_for_transformer(text)
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _truncate_words(s: str, n: int) -> str:
    words = s.rstrip(".").split()
    return " ".join(words[:n])


def lead_title(abstract: str, max_words: int = TITLE_BASELINE_WORDS) -> str:
    """First sentence, truncated. Abstracts often open with the topic."""
    sents = split_sentences(abstract)
    return _truncate_words(sents[0], max_words) if sents else ""


def centroid_title(abstract: str, max_words: int = TITLE_BASELINE_WORDS) -> str:
    """Sentence closest to the abstract as a whole (TF-IDF cosine), truncated."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sents = split_sentences(abstract)
    if len(sents) <= 1:
        return lead_title(abstract, max_words)
    try:
        vec = TfidfVectorizer(stop_words="english").fit(sents + [abstract])
    except ValueError:  # only stop words
        return lead_title(abstract, max_words)
    S, A = vec.transform(sents), vec.transform([abstract])
    sims = np.asarray(S.multiply(A).sum(axis=1)).ravel()
    return _truncate_words(sents[int(np.argmax(sims))], max_words)


BASELINES = {"lead_baseline": lead_title, "centroid_baseline": centroid_title}


# ---------------------------------------------------------------------------
# FLAN-T5
# ---------------------------------------------------------------------------
def transformers_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True, "ok"
    except Exception as e:  # ImportError or broken install
        return False, f"{type(e).__name__}: {e}"


@lru_cache(maxsize=2)
def load_title_model(model_name: str = config.TITLE_MODEL_NAME, device: str | None = None):
    """
    Load tokenizer + model once (cached). The first call downloads the weights from
    the Hugging Face Hub (~1 GB for flan-t5-base, ~300 MB for flan-t5-small).
    Returns (model, tokenizer, device).
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or get_device(config.FORCE_CPU)
    logger.info("Loading %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def postprocess_title(text: str) -> str:
    """Strip quotes, 'Title:' prefixes and trailing full stops; keep the first line."""
    text = (text or "").strip().split("\n")[0]
    text = re.sub(r"^\s*(?:title\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"“”\'*').strip()
    return text.rstrip(".").strip()


def generate_titles(abstracts: list[str], model=None, tokenizer=None, device: str | None = None,
                    batch_size: int = 8, **gen_params) -> list[str]:
    """
    Generate titles for a list of abstracts with FLAN-T5.

    Generation parameters (config.TITLE_GENERATION_PARAMS):
      max_new_tokens       - upper bound on title length in tokens
      num_beams            - beam search keeps this many candidate sequences (better
                             than greedy decoding for short outputs, slower)
      no_repeat_ngram_size - forbids repeating any 3-gram (prevents loops)
      early_stopping       - stop when all beams have finished
    Inputs longer than TITLE_MAX_INPUT_TOKENS are truncated (T5 was trained on 512).
    """
    import torch

    if model is None or tokenizer is None:
        model, tokenizer, device = load_title_model()
    params = {**config.TITLE_GENERATION_PARAMS, **gen_params}
    outputs: list[str] = []
    for i in range(0, len(abstracts), batch_size):
        batch = [config.TITLE_PROMPT + clean_for_transformer(a) for a in abstracts[i:i + batch_size]]
        enc = tokenizer(batch, max_length=config.TITLE_MAX_INPUT_TOKENS, truncation=True,
                        padding=True, return_tensors="pt").to(device or model.device)
        with torch.no_grad():
            out = model.generate(**enc, **params)
        outputs += [postprocess_title(t) for t in tokenizer.batch_decode(out, skip_special_tokens=True)]
    return outputs


def generate_title(abstract: str, **kwargs) -> str:
    """Validate then generate a single title (raises ValueError for invalid input)."""
    ok, msg = validate_abstract(abstract)
    if not ok:
        raise ValueError(msg)
    return generate_titles([abstract], **kwargs)[0]
