"""
Central configuration for the DG4NLP project.

Every path, random seed, sample size and hyperparameter lives here so that:
  * experiments are reproducible (one place to read the settings used), and
  * no other module hardcodes values that might drift out of sync.

Secrets (API keys) are NOT stored here; they are read from the `.env` file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (all relative to this file, so the project works from any location)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Default Kaggle filename; data_loader also auto-discovers other .json/.jsonl/.csv/.tsv(.zip) files.
RAW_DATA_FILENAME = "arxiv-metadata-oai-snapshot.json"

# Restrict to papers first submitted in one year (new-style IDs start with YYMM).
# This keeps memory bounded when streaming the full Kaggle file, and makes the Kaggle
# snapshot and the GitHub archive fallback select the same population of papers.
# Set to None to use every year (needs much more RAM).
# None = use every year. The Mendeley dataset (default source) is mostly 2025-26 computer
# science; set e.g. "2025" to keep one year. With the full Kaggle file, set a year to bound memory.
YEAR_FILTER = None
MAX_RAW_RECORDS = None    # optional hard cap for quick experiments, e.g. 20_000

# Processed artefacts (produced by `python train.py prepare`)
SAMPLE_FILE = PROCESSED_DATA_DIR / "sample.parquet"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
VAL_FILE = PROCESSED_DATA_DIR / "val.parquet"
TEST_FILE = PROCESSED_DATA_DIR / "test.parquet"
EVAL_SUBSET_FILE = PROCESSED_DATA_DIR / "eval_subset.parquet"
LABELS_FILE = PROCESSED_DATA_DIR / "labels.json"
RUN_CONFIG_FILE = RESULTS_DIR / "run_config.json"

ALL_DIRS = [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSIFICATION_MODELS_DIR,
    EMBEDDINGS_DIR, FIGURES_DIR, RESULTS_DIR,
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data sampling and labels
# ---------------------------------------------------------------------------
N_CLASSES = 10            # keep the N most frequent primary sub-categories
MAX_PER_CLASS = 5_000     # cap per class -> ~50k papers; limits imbalance and RAM
MIN_ABSTRACT_WORDS = 20   # drop near-empty abstracts
EDA_TOP_CATEGORIES = 15   # "Top 15 research categories" figure

# Split proportions (of the de-duplicated sample)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EVAL_SUBSET_SIZE = 300    # shared, fixed test subset for ML vs FLAN-T5 vs LLM

# ---------------------------------------------------------------------------
# TF-IDF + classifier grid (searched on the validation set only)
# ---------------------------------------------------------------------------
TFIDF_GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [20_000, 50_000],
    "min_df": [2, 5],
    "max_df": [0.9],
}
CLASSIFIERS = ["naive_bayes", "logistic_regression", "linear_svm"]

# ---------------------------------------------------------------------------
# Title generation (FLAN-T5)
# ---------------------------------------------------------------------------
TITLE_MODEL_NAME = "google/flan-t5-base"
TITLE_PROMPT = "Generate a concise academic title for this scientific abstract: "
TITLE_MAX_INPUT_TOKENS = 512
TITLE_GENERATION_PARAMS = {
    "max_new_tokens": 32,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

# Abstract input validation (shared by the pipeline and the Streamlit UI)
MIN_INPUT_WORDS = 20
MAX_INPUT_CHARS = 5_000

# ---------------------------------------------------------------------------
# Deep learning (python train.py deep)
# ---------------------------------------------------------------------------
DL_TRANSFORMER = "distilbert-base-uncased"   # 66M-parameter distilled BERT
DL_PER_CLASS = 600        # training papers per class for fine-tuning (6,000 total; CPU budget)
DL_VAL_SIZE = 1000        # validation papers used to pick the best epoch
DL_EPOCHS = 2
DL_BATCH_SIZE = 16
DL_LR = 5e-5
DL_MAX_LEN = 256          # sub-word tokens; covers the median abstract (~190 tokens)

# ---------------------------------------------------------------------------
# Embeddings / optional RAG
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5

# ---------------------------------------------------------------------------
# Environment-driven settings (.env)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

LLM_BACKEND = os.getenv("LLM_BACKEND", "none").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()   # never printed or logged
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_LOCAL_MODEL = os.getenv("LLM_LOCAL_MODEL", "google/flan-t5-large").strip()   # for LLM_BACKEND=local
LLM_TEMPERATURE = 0.0     # deterministic-as-possible outputs for evaluation

FORCE_CPU = os.getenv("FORCE_CPU", "0").strip() == "1"


def ensure_dirs() -> None:
    """Create every output directory if it does not already exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def public_settings() -> dict:
    """Return the non-secret settings, for logging into run_config.json."""
    return {
        "SEED": SEED,
        "N_CLASSES": N_CLASSES,
        "MAX_PER_CLASS": MAX_PER_CLASS,
        "MIN_ABSTRACT_WORDS": MIN_ABSTRACT_WORDS,
        "TEST_SIZE": TEST_SIZE,
        "VAL_SIZE": VAL_SIZE,
        "EVAL_SUBSET_SIZE": EVAL_SUBSET_SIZE,
        "TFIDF_GRID": {k: [list(v) if isinstance(v, tuple) else v for v in vals]
                       for k, vals in TFIDF_GRID.items()},
        "TITLE_MODEL_NAME": TITLE_MODEL_NAME,
        "TITLE_GENERATION_PARAMS": TITLE_GENERATION_PARAMS,
        "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
        "LLM_BACKEND": LLM_BACKEND,
        "LLM_MODEL": LLM_MODEL,
        "LLM_TEMPERATURE": LLM_TEMPERATURE,
        "FORCE_CPU": FORCE_CPU,
    }
