"""
LLM title generation with three prompting strategies, compared on the same abstracts:

  zero_shot  - a one-line instruction only.
  instructed - detailed rules: length, academic style, use only claims stated in the
               abstract (to discourage invented findings), no quotes or prefixes.
  few_shot   - the instructed rules plus three (abstract, title) example pairs taken
               from the TRAINING split, so no test paper is ever shown to the model.
"""

from __future__ import annotations

import pandas as pd

import config
from src.llm_client import LLMClient
from src.title_generation import postprocess_title

PROMPT_STYLES = ("zero_shot", "instructed", "few_shot")
MAX_ABSTRACT_WORDS = 350

INSTRUCTIONS = (
    "Write a title for the scientific abstract below.\n"
    "Rules:\n"
    "1. 6-15 words, in the style of an academic paper title.\n"
    "2. Name the main topic and method or contribution.\n"
    "3. Use only information stated in the abstract; do not add results, numbers, "
    "datasets or claims that are not in it.\n"
    "4. Output the title only: no quotation marks, no 'Title:' prefix, no full stop."
)


def _shorten(abstract: str) -> str:
    words = abstract.split()
    return " ".join(words[:MAX_ABSTRACT_WORDS]) + (" ..." if len(words) > MAX_ABSTRACT_WORDS else "")


def select_few_shot_examples(train: pd.DataFrame, n: int = 3, seed: int = config.SEED) -> list[dict]:
    """
    Pick n training examples with typical title lengths, from n different BROAD FIELDS
    (computer science, mathematics/statistics, physics) where possible, so the prompt is
    not biased towards one field. Within a field the most frequent category is used.
    """
    pool = train[(train["title_words"].between(7, 14)) & (train["abstract_words"].between(80, 200))]

    def field(label: str) -> str:
        top = label.split(".")[0]
        return top if top in {"cs", "math", "stat", "eess"} else "physics"

    cats, used = [], set()
    for c in pool["label"].value_counts().index:          # most frequent first
        if field(c) not in used:
            cats.append(c); used.add(field(c))
        if len(cats) == n:
            break
    for c in pool["label"].value_counts().index:          # fewer fields than n: fill up
        if len(cats) == n:
            break
        if c not in cats:
            cats.append(c)
    examples = []
    for i, c in enumerate(cats):
        row = pool[pool["label"] == c].sample(1, random_state=seed + i).iloc[0]
        examples.append({"abstract": row["abstract"], "title": row["title"], "id": row["id"]})
    return examples


def build_messages(abstract: str, style: str, examples: list[dict] | None = None) -> list[dict]:
    if style not in PROMPT_STYLES:
        raise ValueError(f"style must be one of {PROMPT_STYLES}")
    abstract = _shorten(abstract)
    if style == "zero_shot":
        return [{"role": "user", "content": f"Generate a title for this abstract:\n\n{abstract}"}]
    messages = [{"role": "system", "content": "You are an expert scientific editor."}]
    if style == "few_shot":
        for ex in examples or []:
            messages.append({"role": "user", "content": f"{INSTRUCTIONS}\n\nAbstract:\n{_shorten(ex['abstract'])}"})
            messages.append({"role": "assistant", "content": ex["title"]})
    messages.append({"role": "user", "content": f"{INSTRUCTIONS}\n\nAbstract:\n{abstract}"})
    return messages


def generate_title_llm(abstract: str, client: LLMClient, style: str = "instructed",
                       examples: list[dict] | None = None) -> str | None:
    """Return a cleaned title, or None if the request failed."""
    raw = client.chat(build_messages(abstract, style, examples), max_tokens=60)
    return postprocess_title(raw) if raw else None
