"""
Text cleaning, label preparation and dataset splitting.

Two text views are produced because the two model families need different input:

* `clean_for_tfidf`  - for bag-of-words models. TF-IDF treats every distinct string
  as a separate feature, so normalisation (lower-casing, removing punctuation and
  stop words, optionally stemming) merges variants and shrinks the vocabulary.
* `clean_for_transformer` - for FLAN-T5, sentence embeddings and LLMs. These models
  use sub-word tokenisers trained on natural text, so casing, punctuation and
  function words carry meaning and are kept; only formatting noise is removed.

Scientific text needs care: hyphenated terms ("self-supervised"), alphanumeric
terms ("3d", "co2", "gpt-2") and abbreviations are kept as single tokens; inline
LaTeX maths is replaced by a placeholder token instead of being deleted.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

import config
from src.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
INLINE_MATH_RE = re.compile(r"\$[^$]{1,300}\$")
LATEX_WITH_ARG_RE = re.compile(
    r"\\(?:emph|textit|textbf|textrm|textsc|mathrm|mathbf|mathcal|mathbb|text|it|bf)\s*\{([^{}]*)\}")
LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
WHITESPACE_RE = re.compile(r"\s+")

# Tokens: words that may contain inner hyphens/digits (self-supervised, gpt-2, 3d).
TOKENIZER = RegexpTokenizer(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# scikit-learn's built-in English stop-word list is used (rather than the NLTK corpus)
# because it needs no download and is identical on every machine -> reproducible.
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | {"paper", "propose", "proposed", "show", "using",
                                               "based", "results", "study", "new", "approach"}
PLACEHOLDERS = {"mathexpr", "urltoken"}

_stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including the hard line breaks arXiv abstracts contain)."""
    if not isinstance(text, str):
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_for_transformer(text: str) -> str:
    """
    Light cleaning for transformer models and display: unwrap LaTeX formatting commands
    (\\emph{x} -> x), unescape common symbols and normalise whitespace. Case, punctuation
    and maths are kept because sub-word tokenisers use them.
    """
    if not isinstance(text, str):
        return ""
    text = LATEX_WITH_ARG_RE.sub(r"\1", text)
    text = text.replace("\\%", "%").replace("~", " ").replace("``", '"').replace("''", '"')
    return normalise_whitespace(text)


@lru_cache(maxsize=None)
def _lemmatizer():
    """WordNet lemmatiser, loaded lazily. Requires `nltk.download('wordnet')`."""
    from nltk.stem import WordNetLemmatizer

    lem = WordNetLemmatizer()
    lem.lemmatize("tests")  # raises LookupError now if WordNet data is missing
    return lem


def tokenize(text: str) -> list[str]:
    """Tokenise already lower-cased text, keeping hyphenated and alphanumeric terms."""
    return TOKENIZER.tokenize(text)


def clean_for_tfidf(text: str, remove_stopwords: bool = True, stem: bool = False,
                    lemmatize: bool = False) -> str:
    """
    Normalise text for TF-IDF features.

    Steps: URLs -> 'urltoken'; inline maths -> 'mathexpr'; unwrap LaTeX formatting;
    drop other LaTeX commands; lower-case; tokenise; drop pure numbers and
    single characters; optionally drop stop words; optionally stem or lemmatise.

    Args:
        remove_stopwords: drop high-frequency function words.
        stem: Porter stemming (fast, crude: 'networks' -> 'network', 'studies' -> 'studi').
        lemmatize: WordNet lemmatisation (dictionary-based; needs NLTK 'wordnet' data).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = URL_RE.sub(" urltoken ", text)
    text = INLINE_MATH_RE.sub(" mathexpr ", text)
    text = LATEX_WITH_ARG_RE.sub(r" \1 ", text)
    text = LATEX_CMD_RE.sub(" ", text)
    tokens = tokenize(text.lower())
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stem:
        tokens = [t if t in PLACEHOLDERS else _stemmer.stem(t) for t in tokens]
    elif lemmatize:
        lem = _lemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def split_categories(categories: str) -> list[str]:
    """Split a category string (space- or comma-separated) into a list."""
    if not isinstance(categories, str):
        return []
    return [c.strip() for c in re.split(r"[\s,]+", categories) if c.strip()]


def extract_primary_category(categories: str) -> str | None:
    """
    Return the primary arXiv category: the FIRST listed category.

    arXiv stores the author-chosen primary category first, followed by any
    cross-lists (e.g. 'cs.CL cs.AI cs.LG' -> 'cs.CL'). Category codes are case-
    sensitive identifiers (cs.LG, hep-ph, cond-mat.mes-hall), so they are only
    stripped, not lower-cased.
    """
    cats = split_categories(categories)
    return cats[0] if cats else None


def get_archive(category: str | None) -> str | None:
    """Top-level archive of a category: 'cs.LG' -> 'cs', 'hep-ph' -> 'hep-ph'."""
    if not category:
        return None
    return category.split(".")[0]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def _norm_for_dedup(s: str) -> str:
    return WHITESPACE_RE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def clean_dataframe(raw: pd.DataFrame, min_abstract_words: int = config.MIN_ABSTRACT_WORDS
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove unusable records and add label/length columns.

    Returns (clean_df, steps_df) where steps_df records how many rows each step removed.
    Duplicate abstracts are removed (re-submissions of the same work) so the same text
    cannot end up in both the training and test sets (data leakage).
    """
    df = raw.copy()
    steps = []

    def record(step: str, before: int) -> None:
        steps.append({"step": step, "removed": before - len(df), "remaining": len(df)})

    n = len(df); df = df.drop_duplicates("id"); record("duplicate IDs", n)
    n = len(df)
    df = df[df["title"].notna() & df["abstract"].notna() & df["categories"].notna()]
    df = df[(df["title"].str.strip() != "") & (df["abstract"].str.strip() != "")]
    record("missing title/abstract/categories", n)
    n = len(df)
    df = df[~df["abstract"].str.contains(
        r"\b(?:paper|article|submission|manuscript) (?:has been|is) withdrawn\b",
        case=False, regex=True)]
    record("withdrawal notices", n)
    n = len(df)
    df = df.assign(_abs_norm=df["abstract"].map(_norm_for_dedup))
    df = df.drop_duplicates("_abs_norm").drop(columns="_abs_norm")
    record("duplicate abstracts", n)

    df["title"] = df["title"].map(clean_for_transformer)
    df["abstract"] = df["abstract"].map(clean_for_transformer)
    df["abstract_words"] = df["abstract"].str.split().str.len()
    df["title_words"] = df["title"].str.split().str.len()
    n = len(df); df = df[df["abstract_words"] >= min_abstract_words]
    record(f"abstracts < {min_abstract_words} words", n)

    df["primary_category"] = df["categories"].map(extract_primary_category)
    df["n_categories"] = df["categories"].map(lambda c: len(split_categories(c)))
    df["archive"] = df["primary_category"].map(get_archive)
    n = len(df); df = df[df["primary_category"].notna()]; record("no primary category", n)
    return df.reset_index(drop=True), pd.DataFrame(steps)


def select_top_classes(df: pd.DataFrame, n_classes: int = config.N_CLASSES) -> list[str]:
    """Return the n most frequent primary categories (ties broken alphabetically)."""
    counts = df["primary_category"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    return counts.head(n_classes).index.tolist()


def build_sample(df: pd.DataFrame, labels: list[str], max_per_class: int = config.MAX_PER_CLASS,
                 seed: int = config.SEED) -> pd.DataFrame:
    """
    Keep papers whose primary category is in `labels` and randomly sample at most
    `max_per_class` per class (fixed seed). Capping reduces imbalance, but classes
    with fewer papers keep all of them, so some imbalance remains by design.
    """
    sub = df[df["primary_category"].isin(labels)]
    parts = [g.sample(n=min(len(g), max_per_class), random_state=seed)
             for _, g in sub.groupby("primary_category", sort=True)]
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample["label"] = sample["primary_category"]
    sample["text_tfidf"] = sample["abstract"].map(clean_for_tfidf)
    return sample


def split_dataset(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                  val_size: float = config.VAL_SIZE, seed: int = config.SEED,
                  label_col: str = "label") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    The test set is separated first; the validation set is then taken from the
    remainder, with its size rescaled so that it is `val_size` of the whole dataset.
    Stratification keeps class proportions equal in all three splits.
    """
    stratify = df[label_col] if df[label_col].value_counts().min() >= 3 else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    rel_val = val_size / (1.0 - test_size)
    stratify_tv = train_val[label_col] if stratify is not None else None
    train, val = train_test_split(train_val, test_size=rel_val, random_state=seed, stratify=stratify_tv)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def make_eval_subset(test: pd.DataFrame, size: int = config.EVAL_SUBSET_SIZE,
                     seed: int = config.SEED, label_col: str = "label") -> pd.DataFrame:
    """A fixed, class-balanced subset of the TEST set shared by every model comparison."""
    n_classes = test[label_col].nunique()
    per_class = max(1, size // n_classes)
    parts = [g.sample(n=min(len(g), per_class), random_state=seed)
             for _, g in test.groupby(label_col, sort=True)]
    # Shuffle so that any prefix (e.g. `--limit 50`) still contains every class.
    sub = pd.concat(parts).sample(frac=1.0, random_state=seed)
    order = sub.groupby(label_col).cumcount()
    return sub.assign(_o=order).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any paper ID or abstract appears in more than one split."""
    for col in ("id", "abstract"):
        a, b, c = set(train[col]), set(val[col]), set(test[col])
        overlap = (a & b) | (a & c) | (b & c)
        if overlap:
            raise AssertionError(f"{len(overlap)} overlapping '{col}' values between splits")


def class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Counts and percentages per class, plus the imbalance ratio (largest / smallest)."""
    counts = df[label_col].value_counts()
    out = pd.DataFrame({"count": counts, "percent": (counts / counts.sum() * 100).round(2)})
    out.attrs["imbalance_ratio"] = float(counts.max() / counts.min()) if len(counts) else np.nan
    return out
