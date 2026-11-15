"""Page 2 - Research paper classification."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.classification import MODEL_DISPLAY_NAMES, predict_with_scores
from src.llm_classification import CATEGORY_DESCRIPTIONS, classify_with_llm
from src.utils import Timer
from ui.components import (abstract_input, artefact_status, check_abstract, get_llm_client,
                           load_classifier, load_csv, load_labels, missing_artefact)

LLM_KEY = "llm"


def _model_options(status: dict) -> dict[str, str]:
    opts = {MODEL_DISPLAY_NAMES[m]: m for m in status["ml_models"]}
    if status["llm"]:
        opts["LLM (zero-shot prompt)"] = LLM_KEY
    return opts


def _run(key: str, text: str, labels: list[str]) -> dict:
    with Timer() as t:
        if key == LLM_KEY:
            res = classify_with_llm(text, get_llm_client(), labels)
            out = {"prediction": res["category"] or "INVALID", "scores": None,
                   "explanation": res["explanation"], "confidence": res["confidence"],
                   "error": res["error"], "parse": res["parse_method"]}
        else:
            pipe = load_classifier(key)
            scores = predict_with_scores(pipe, text, top_k=3)
            out = {"prediction": scores.iloc[0]["category"], "scores": scores}
    out["ms"] = t.elapsed * 1000
    return out


def render() -> None:
    st.title("Research paper classification")
    status = artefact_status()
    if not status["ml_models"]:
        missing_artefact("No trained classifiers found.", "python train.py classify")
        return
    labels = load_labels()
    st.caption("Predicts the paper's primary arXiv category among the "
               f"{len(labels)} classes the models were trained on: " + ", ".join(labels))

    text, _ = abstract_input("cls")
    options = _model_options(status)
    choice = st.selectbox("Model", list(options))
    compare = st.checkbox("Also compare predictions across all models")

    if st.button("Predict category", type="primary") and check_abstract(text):
        key = options[choice]
        with st.spinner("Classifying ..."):
            res = _run(key, text, labels)
        st.session_state["cls_result"] = (choice, res)
        if compare:
            rows = []
            for name, k in options.items():
                r = res if k == key else _run(k, text, labels)
                rows.append({"model": name, "prediction": r["prediction"],
                             "time (ms)": round(r["ms"], 1)})
            st.session_state["cls_compare"] = pd.DataFrame(rows)
        else:
            st.session_state.pop("cls_compare", None)

    if "cls_result" in st.session_state:
        choice_done, res = st.session_state["cls_result"]
        pred = res["prediction"]
        st.success(f"**{choice_done}** predicts: **{pred}** — {CATEGORY_DESCRIPTIONS.get(pred, '')}")
        st.caption(f"Inference time: {res['ms']:.1f} ms")
        if res.get("scores") is not None:
            st.markdown("Top-3 categories")
            st.dataframe(res["scores"], hide_index=True, width="stretch")
        else:
            st.markdown(f"**Explanation:** {res.get('explanation') or '—'}")
            st.caption(f"Self-reported confidence: {res.get('confidence') or 'not given'} "
                       "(the model's own statement, not a calibrated probability) · "
                       f"parse method: {res.get('parse')}")
            if res.get("error"):
                st.error(f"LLM error: {res['error']}")
        if st.session_state.get("example_label"):
            st.info(f"Ground truth for this example: {st.session_state['example_label']}")

        if "cls_compare" in st.session_state:
            st.markdown("**Comparison across models**")
            cmp = st.session_state["cls_compare"]
            st.dataframe(cmp, hide_index=True, width="stretch")
            agree = cmp["prediction"].nunique() == 1
            st.caption("All models agree." if agree else "Models disagree — the abstract may span several fields.")

    with st.expander("Model details (measured on the held-out test set)"):
        m = load_csv("classification_metrics_test.csv")
        if m is not None:
            st.dataframe(m[["model", "accuracy", "macro_f1", "weighted_f1", "val_macro_f1",
                            "tfidf_params", "clf_params"]].round(4), hide_index=True, width="stretch")
            st.caption("All three use a TF-IDF pipeline fitted on training data only; "
                       "hyperparameters were chosen on the validation set.")
