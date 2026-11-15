"""
Shared Streamlit helpers: cached loaders, artefact status, input widgets.

Caching:
  * st.cache_resource - heavy objects created once per server (models, index, client)
  * st.cache_data     - data frames / JSON read from disk (returns a copy each time)
The app only READS artefacts written by train.py; it never trains or fabricates results.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from src import classification as C
from src.title_generation import validate_abstract

R = config.RESULTS_DIR


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = R / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        return None


def load_labels() -> list[str]:
    return load_json_file(str(config.LABELS_FILE)) or []


@st.cache_resource(show_spinner="Loading classifier ...")
def load_classifier(key: str):
    return C.load_model(key)


@st.cache_resource(show_spinner="Loading retrieval index ...")
def load_retriever():
    from src.rag import load_retriever as _load

    return _load()


@st.cache_resource(show_spinner=False)
def get_llm_client():
    from src.llm_client import LLMClient

    return LLMClient()


@st.cache_resource(show_spinner="Loading FLAN-T5 (first time downloads the model) ...")
def load_flan_t5():
    from src.title_generation import load_title_model

    return load_title_model()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def artefact_status() -> dict:
    from src.rag import index_info
    from src.title_generation import transformers_available

    client = get_llm_client()
    tf_ok, tf_msg = transformers_available()
    return {
        "data": config.SAMPLE_FILE.exists(),
        "ml_models": C.available_models(),
        "flan_t5": tf_ok,
        "flan_t5_msg": tf_msg,
        "llm": client.available,
        "llm_desc": client.describe(),
        "index": index_info(),
        "results": (R / "classification_metrics_test.csv").exists(),
    }


def missing_artefact(message: str, command: str) -> None:
    st.warning(f"{message}\n\nRun in the VS Code terminal:\n\n`{command}`")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
def example_papers() -> pd.DataFrame | None:
    """Real held-out test papers used as examples (never training papers)."""
    return load_parquet(str(config.EVAL_SUBSET_FILE))


def abstract_input(key: str = "abstract", with_title: bool = False) -> tuple[str, str]:
    """
    Text area shared across pages through session_state, plus an
    'example' button that loads a random held-out test abstract.
    """
    st.session_state.setdefault("abstract", "")
    st.session_state.setdefault("original_title", "")
    st.session_state.setdefault("example_label", "")
    ex = example_papers()
    cols = st.columns([1, 1, 3])
    if ex is not None and cols[0].button("Load example abstract", key=f"{key}_example",
                                         help="A real paper from the held-out test set"):
        row = ex.sample(1).iloc[0]
        st.session_state["abstract"] = row["abstract"]
        st.session_state["original_title"] = row["title"]
        st.session_state["example_label"] = f"{row['id']} (true category: {row['label']})"
    if cols[1].button("Clear", key=f"{key}_clear"):
        st.session_state["abstract"] = ""
        st.session_state["original_title"] = ""
        st.session_state["example_label"] = ""
    if st.session_state["example_label"]:
        cols[2].caption(f"Example: arXiv:{st.session_state['example_label']}")
    text = st.text_area("Scientific abstract", key="abstract", height=220,
                        placeholder="Paste an abstract (at least 20 words) ...")
    title = ""
    if with_title:
        title = st.text_input("Original title (optional, for ROUGE comparison)", key="original_title")
    return text, title


def check_abstract(text: str) -> bool:
    ok, msg = validate_abstract(text)
    if not ok:
        st.error(msg)
    return ok


def show_figure(name: str, caption: str | None = None) -> None:
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not generated yet.")


def arxiv_link(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"
