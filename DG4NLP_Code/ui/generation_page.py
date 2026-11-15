"""Page 3 - Title generation."""

from __future__ import annotations

import streamlit as st

from src.evaluation import rouge_scores, unsupported_term_rate
from src.llm_title_generation import generate_title_llm, select_few_shot_examples
from src.title_generation import BASELINES, generate_titles
from src.utils import Timer
from ui.components import (abstract_input, artefact_status, check_abstract, get_llm_client,
                           load_parquet)

import config

METHODS = {
    "FLAN-T5 (google/flan-t5-base, zero-shot)": "flan_t5",
    "LLM — zero-shot prompt": "llm_zero_shot",
    "LLM — detailed instructions": "llm_instructed",
    "LLM — few-shot (3 training examples)": "llm_few_shot",
    "Baseline — first sentence (lead)": "lead_baseline",
    "Baseline — most central sentence": "centroid_baseline",
}


def _generate(method: str, text: str) -> str:
    if method in BASELINES:
        return BASELINES[method](text)
    if method == "flan_t5":
        from ui.components import load_flan_t5

        model, tok, device = load_flan_t5()
        return generate_titles([text], model, tok, device)[0]
    style = method.replace("llm_", "")
    examples = None
    if style == "few_shot":
        train = load_parquet(str(config.TRAIN_FILE))
        examples = select_few_shot_examples(train) if train is not None else []
    client = get_llm_client()
    title = generate_title_llm(text, client, style, examples)
    if title is None:
        raise RuntimeError(client.last_error or "LLM request failed")
    return title


def render() -> None:
    st.title("Title generation")
    st.caption("Generates a concise academic title from an abstract. ROUGE is shown when "
               "the original title is provided.")
    status = artefact_status()
    available = {k: v for k, v in METHODS.items()
                 if (v != "flan_t5" or status["flan_t5"]) and (not v.startswith("llm_") or status["llm"])}
    hidden = set(METHODS) - set(available)
    text, original = abstract_input("gen", with_title=True)
    method_label = st.selectbox("Generation method", list(available))
    if hidden:
        st.caption("Unavailable here: " + "; ".join(sorted(hidden)) +
                   (". FLAN-T5 needs `torch` and `transformers`." if not status["flan_t5"] else "") +
                   (" LLM methods need LLM settings in `.env`." if not status["llm"] else ""))

    if st.button("Generate title", type="primary") and check_abstract(text):
        try:
            with st.spinner("Generating ..."), Timer() as t:
                title = _generate(available[method_label], text)
            st.session_state["gen_result"] = (method_label, title, t.elapsed)
        except Exception as e:
            st.error(f"Generation failed: {type(e).__name__}: {e}")

    if "gen_result" in st.session_state:
        label, title, secs = st.session_state["gen_result"]
        st.markdown(f"### {title or '(empty output)'}")
        st.caption(f"{label} · {secs:.2f} s")
        rate = unsupported_term_rate(text, title)
        st.caption(f"Title words not found in the abstract: {rate:.0%} "
                   "(a simple check for invented content; lower is safer)")
        if original.strip():
            st.markdown(f"**Original title:** {original}")
            s = rouge_scores([original], [title]).iloc[0]
            c = st.columns(3)
            c[0].metric("ROUGE-1 F1", f"{s['rouge1']:.3f}")
            c[1].metric("ROUGE-2 F1", f"{s['rouge2']:.3f}")
            c[2].metric("ROUGE-L F1", f"{s['rougeL']:.3f}")
            st.caption("ROUGE for a single example; see Model Comparison for averages over the test subset.")
