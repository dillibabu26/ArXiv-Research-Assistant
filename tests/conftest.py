"""
Shared synthetic fixtures. Tests never need the real arXiv dataset, models or network.
"""

import random
import sys
from pathlib import Path

# Make `import config` / `import src...` work even when pytest is started from a
# different folder (e.g. the parent folder created when the zip was extracted).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

TOPICS = {
    "cs.CL": ["language", "translation", "parsing", "sentiment", "tokens", "corpus", "dialogue"],
    "cs.CV": ["image", "pixels", "segmentation", "detection", "camera", "video", "convolutional"],
    "quant-ph": ["qubit", "entanglement", "quantum", "photon", "decoherence", "circuit", "superposition"],
}
FILLER = ["we", "study", "a", "method", "for", "the", "problem", "and", "show", "results", "on", "data"]


def make_abstract(label: str, rng: random.Random, n_words: int = 40) -> str:
    words = [rng.choice(TOPICS[label]) if rng.random() < 0.4 else rng.choice(FILLER) for _ in range(n_words)]
    return ("We " + " ".join(words) + ". This second sentence discusses " +
            " ".join(rng.choice(TOPICS[label]) for _ in range(5)) + ".")


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """90 papers, 30 per class, with unique IDs and abstracts."""
    rng = random.Random(0)
    rows = []
    for i in range(90):
        label = list(TOPICS)[i % 3]
        rows.append({"id": f"1901.{i:05d}", "title": f"A study of {TOPICS[label][0]} number {i}",
                     "abstract": make_abstract(label, rng) + f" Paper {i}.",
                     "categories": f"{label} stat.ML" if i % 2 else label,
                     "authors": "A. Author", "comments": None, "doi": None,
                     "update_date": "2019-01-01", "source": "test"})
    return pd.DataFrame(rows)


@pytest.fixture
def labelled_df(synthetic_df) -> pd.DataFrame:
    from src.preprocessing import build_sample, clean_dataframe

    clean, _ = clean_dataframe(synthetic_df)
    return build_sample(clean, sorted(TOPICS), max_per_class=1000)
