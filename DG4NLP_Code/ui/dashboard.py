"""Page 1 - Home dashboard: project introduction, dataset statistics, system status."""

from __future__ import annotations

import streamlit as st

import config
from src.classification import MODEL_DISPLAY_NAMES
from ui.components import artefact_status, load_json_file, missing_artefact, show_figure


def render() -> None:
    st.title("ArXiv Research Assistant")
    st.markdown(
        "An MSc NLP project that classifies scientific abstracts into arXiv categories, "
        "generates academic titles, compares traditional machine learning with "
        "transformer/LLM approaches, and retrieves related papers with source references.")

    summary = load_json_file(str(config.RESULTS_DIR / "eda_summary.json"))
    status = artefact_status()
    if not summary:
        missing_artefact("No processed dataset found.", "python train.py prepare")
        return

    st.subheader("Dataset")
    c = st.columns(4)
    c[0].metric("Papers loaded", f"{summary['load_report']['kept']:,}")
    c[1].metric("After cleaning", f"{summary['clean_records']:,}")
    c[2].metric("Modelling sample", f"{summary['sample_size']:,}")
    c[3].metric("Primary categories (all)", summary["eda"]["top_categories"]["n_primary_categories"])
    src = summary["load_report"]["source"]
    source_names = {"mendeley_csv": "Mendeley arXiv Scientific Research Paper Dataset",
                    "kaggle_jsonl": "Kaggle arXiv snapshot",
                    "archive_tsv": "ArXiV Archive (export of the arXiv OAI metadata)"}
    st.caption(f"Source: {source_names.get(src, src)} · papers from "
               f"{config.YEAR_FILTER or 'all years'} · {len(summary['labels'])} target classes · splits "
               + " / ".join(f"{k} {v:,}" for k, v in summary["split_sizes"].items()))

    st.subheader("Available models and components")
    ml = status["ml_models"]
    idx = status["index"]
    rows = [
        ("Traditional ML classifiers", "Ready: " + ", ".join(MODEL_DISPLAY_NAMES[m] for m in ml) if ml
         else "Not trained: `python train.py classify`"),
        ("FLAN-T5 title generation", "Available (loads on first use)" if status["flan_t5"]
         else f"Unavailable: {status['flan_t5_msg'][:80]}"),
        ("Extractive title baselines", "Ready (no model needed)"),
        ("LLM classification / titles", status["llm_desc"] if status["llm"] else
         "Disabled; set LLM_BACKEND in .env (optional)"),
        ("Retrieval index", f"{idx['backend']} over {idx['n_documents']:,} papers" if idx
         else "Not built: `python train.py index`"),
    ]
    st.table({"Component": [r[0] for r in rows], "Status": [r[1] for r in rows]})

    st.subheader("What the system can do")
    st.markdown(
        "- **Classify** an abstract with Naive Bayes, Logistic Regression, Linear SVM or an LLM, "
        "and compare their predictions.\n"
        "- **Generate a title** with FLAN-T5, an LLM (three prompt styles) or extractive baselines, "
        "with ROUGE against the original title.\n"
        "- **Compare models** using metrics measured on held-out test data.\n"
        "- **Explore papers** by keyword or meaning and find similar papers.\n"
        "- **Ask questions** answered only from retrieved papers, with arXiv IDs.")

    st.subheader("Exploratory data analysis")
    tabs = st.tabs(["Categories", "Abstract length", "Title length", "Missing values",
                    "Frequent words", "Class imbalance", "Word frequency"])
    figs = ["01_top_categories.png", "02_abstract_length.png", "03_title_length.png",
            "04_missing_values.png", "05_most_frequent_words.png", "06_category_imbalance.png",
            "07_word_frequency_analysis.png"]
    for tab, fig in zip(tabs, figs):
        with tab:
            show_figure(fig)
