"""
Shared utilities: seeding, timing, JSON I/O, logging, device selection,
library-version reporting and NLTK resource download.

Kept dependency-light: torch is imported lazily so that modules which only
need scikit-learn do not pay the cost of importing PyTorch.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOGGER_NAME = "dg4nlp"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a configured logger that prints timestamped messages once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # avoid duplicate lines if a library configures the root logger
    return logger


def set_seed(seed: int) -> None:
    """
    Seed every random number generator used in the project.

    Python's `random`, NumPy and (if installed) PyTorch each keep their own
    generator, so all three must be seeded for runs to be repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(force_cpu: bool = False) -> str:
    """Return 'cuda' if a GPU is available and allowed, otherwise 'cpu'."""
    if force_cpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Timer:
    """Simple wall-clock timer. Use as `with Timer() as t: ...; t.elapsed`."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


@contextmanager
def log_duration(task: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log how long a block of code takes."""
    logger = logger or get_logger()
    logger.info("Starting: %s", task)
    with Timer() as t:
        yield
    logger.info("Finished: %s (%.1f s)", task, t.elapsed)


def save_json(obj: Any, path: str | Path) -> None:
    """Write an object to a UTF-8 JSON file, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


CORE_PACKAGES = [
    ("pandas", "pandas"), ("numpy", "numpy"), ("pyarrow", "pyarrow"),
    ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
    ("scikit-learn", "sklearn"), ("nltk", "nltk"), ("joblib", "joblib"),
    ("torch", "torch"), ("transformers", "transformers"),
    ("sentencepiece", "sentencepiece"),
    ("sentence-transformers", "sentence_transformers"),
    ("rouge-score", "rouge_score"), ("streamlit", "streamlit"),
    ("requests", "requests"), ("python-dotenv", "dotenv"),
    ("tqdm", "tqdm"), ("pytest", "pytest"),
]
OPTIONAL_PACKAGES = [("faiss-cpu", "faiss")]


def get_library_versions(packages=None) -> dict[str, str]:
    """
    Return {package: version} for each package, or 'NOT INSTALLED'.
    Recorded in run_config.json so results can be tied to exact versions.
    """
    packages = packages or CORE_PACKAGES + OPTIONAL_PACKAGES
    versions = {}
    for display_name, import_name in packages:
        try:
            module = importlib.import_module(import_name)
            versions[display_name] = getattr(module, "__version__", "installed")
        except Exception:  # ImportError, or a broken install raising other errors
            versions[display_name] = "NOT INSTALLED"
    return versions


NLTK_RESOURCES = {
    # resource name -> path used by nltk.data.find
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",   # required by newer NLTK versions
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data(quiet: bool = True) -> dict[str, bool]:
    """Download the NLTK resources the project needs (no-op if present)."""
    import nltk

    status = {}
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
            status[name] = True
        except LookupError:
            try:
                status[name] = bool(nltk.download(name, quiet=quiet))
            except Exception:
                status[name] = False
    return status
