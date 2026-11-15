"""Page 4 - Model comparison. Shows ONLY values measured by train.py."""

from __future__ import annotations

import streamlit as st

import config
from src.classification import MODEL_DISPLAY_NAMES
from ui.components import load_csv, load_json_file, missing_artefact, show_figure

FMT_COLS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def render() -> None:
    st.title("Model comparison")
    st.caption("Every number on this page is read from `outputs/results/`, written by `train.py`.")
    test = load_csv("classification_metrics_test.csv")
    if test is None:
        missing_artefact("No evaluation results yet.", "python train.py classify")
        return

    st.subheader("Traditional ML — full test set")
    t = test.rename(columns={"single_ms_median": "latency ms (median, 1 abstract)",
                             "batch_ms_per_doc": "ms/abstract (batch)"})
    st.dataframe(t[["model"] + FMT_COLS + ["ms/abstract (batch)", "latency ms (median, 1 abstract)", "n"]].round(4),
                 hide_index=True, width="stretch")
    show_figure("classification_comparison.png")

    st.subheader("Traditional ML vs LLM — shared evaluation subset")
    comp = load_csv("comparison_classification.csv")
    llm_status = load_json_file(str(config.RESULTS_DIR / "llm_status.json")) or {}
    if comp is not None:
        st.dataframe(comp.round(4), hide_index=True, width="stretch")
        st.caption(f"Same {int(comp['n'].iloc[0])} class-balanced test abstracts for every model. "
                   "With this sample size, differences of about ±3 percentage points are within noise.")
    if llm_status.get("status") != "run":
        st.info(f"LLM experiments not run yet ({llm_status.get('reason', 'no results')}). "
                "Configure `.env`, then run `python train.py llm` and `python train.py compare`.")
    c = st.columns(2)
    with c[0]:
        show_figure("comparison_classification.png")
    with c[1]:
        show_figure("comparison_inference_time.png")

    st.subheader("Statistical significance")
    ci = load_csv("significance_ci.csv")
    mc = load_csv("significance_mcnemar.csv")
    if ci is None:
        st.info("Not computed yet: run `python train.py stats`.")
    else:
        show = ci.assign(**{"95% CI": ci.apply(lambda r: f"[{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)})
        st.dataframe(show[["model", "split", "metric", "estimate", "95% CI", "n"]].round(4),
                     hide_index=True, width="stretch")
        st.caption(f"Percentile bootstrap with {int(ci['n_boot'].iloc[0])} resamples of the evaluation set.")
    if mc is not None:
        st.dataframe(mc.rename(columns={"a_only_correct": "only A correct", "b_only_correct": "only B correct"})
                     .round(4), hide_index=True, width="stretch")
        st.caption("McNemar's test compares two models on the same papers using only the papers where "
                   "exactly one model is correct. p < 0.05 means the difference in error rates is "
                   "unlikely to be due to chance.")

    st.subheader("Confusion matrices")
    options = {MODEL_DISPLAY_NAMES[k]: k for k in MODEL_DISPLAY_NAMES
               if (config.FIGURES_DIR / f"confusion_matrix_{k}.png").exists()}
    if options:
        pick = st.selectbox("Model", list(options))
        show_figure(f"confusion_matrix_{options[pick]}.png",
                    "Row-normalised: each row shows how papers of one true category were classified.")
    reports = load_csv("classification_reports.csv")
    if reports is not None:
        with st.expander("Per-class precision / recall / F1"):
            st.dataframe(reports.rename(columns={"Unnamed: 0": "class"}).round(4),
                         hide_index=True, width="stretch")

    st.subheader("Title generation")
    tm = load_csv("title_metrics.csv")
    if tm is not None:
        from src.evaluation import method_label

        shown = tm.assign(method=tm["method"].map(method_label)).sort_values("rougeL", ascending=False)
        st.dataframe(shown.round(4), hide_index=True, width="stretch")
        gen = load_csv("titles_generated.csv")
        if gen is not None:
            st.caption(f"Similarity method: {gen['similarity_method'].iloc[0]}. "
                       "unsupported_rate = share of title words absent from the abstract.")
        show_figure("comparison_titles.png")
        rc = load_json_file(str(config.RESULTS_DIR / "run_config.json")) or {}
        st.caption("FLAN-T5 status: " + str(rc.get("stages", {}).get("titles", {}).get("status", {}).get("flan_t5", "unknown")))
        if gen is not None:
            with st.expander("Example generated titles"):
                st.dataframe(gen[["method", "original", "generated", "rougeL"]].head(30),
                             hide_index=True, width="stretch")

    st.subheader("Retrieval")
    rm = load_json_file(str(config.RESULTS_DIR / "retrieval_metrics.json"))
    if rm:
        th = rm["evidence_threshold"]
        c = st.columns(5)
        c[0].metric("Recall@1", f"{rm['recall@1']:.3f}")
        c[1].metric("Recall@10", f"{rm['recall@10']:.3f}")
        c[2].metric("MRR@10", f"{rm['mrr@10']:.3f}")
        c[3].metric("Category P@10", f"{rm['category_precision@10']:.3f}")
        c[4].metric("Latency (ms)", f"{rm['median_latency_ms']:.1f}")
        st.caption(f"Backend: {rm['backend']}. Known-item search with {rm['n_queries']} paper titles. "
                   f"Evidence threshold {th['threshold']} ({th['threshold_rule']}): "
                   f"{th['in_domain_answered']:.0%} of scientific queries answered, "
                   f"{th['out_domain_refused']:.0%} of off-topic questions refused.")
