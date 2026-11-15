"""Page 6 - Research assistant chatbot (retrieval-grounded; never invents citations)."""

from __future__ import annotations

import streamlit as st

import config
from src.rag import answer_question
from ui.components import arxiv_link, get_llm_client, load_json_file, load_retriever, missing_artefact

MODE_LABELS = {"llm_grounded": "LLM answer grounded in retrieved papers",
               "extractive": "Extractive answer (quoted sentences from retrieved papers; no LLM)",
               "refused": "Refused: no sufficiently relevant evidence"}


def _sources_md(sources, cited) -> str:
    lines = []
    for i, r in enumerate(sources.itertuples(), start=1):
        mark = " **(cited)**" if i in cited else ""
        lines.append(f"[{i}] [arXiv:{r.id}]({arxiv_link(r.id)}) — {r.title} "
                     f"(`{r.label}`, score {r.score:.3f}){mark}")
    return "\n\n".join(lines)


def render() -> None:
    st.title("Research assistant chatbot")
    try:
        retriever = load_retriever()
    except FileNotFoundError:
        missing_artefact("The chatbot needs a retrieval index.", "python train.py index")
        return
    client = get_llm_client()
    metrics = load_json_file(str(config.RESULTS_DIR / "retrieval_metrics.json")) or {}
    threshold = metrics.get("evidence_threshold", {}).get("threshold")
    st.caption(f"Answers come only from the {len(retriever.docs):,} loaded papers "
               f"({retriever.backend} retrieval). "
               + (f"LLM: {client.describe()}." if client.available else
                  "No LLM configured, so answers are extractive quotes from the retrieved papers.")
               + (f" Evidence threshold: {threshold}." if threshold is not None else ""))

    st.session_state.setdefault("chat", [])
    if st.button("Clear conversation"):
        st.session_state["chat"] = []
    for turn in st.session_state["chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask about the loaded papers, e.g. 'How is BERT used for relation extraction?'")
    if not question:
        return
    if len(question.split()) < 2:
        st.warning("Please ask a longer question.")
        return
    st.session_state["chat"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Retrieving papers ..."):
        res = answer_question(question, retriever, client if client.available else None,
                              k=config.RAG_TOP_K, threshold=threshold)
        body = f"{res['answer']}\n\n_{MODE_LABELS[res['mode']]}_"
        if res.get("invalid_citations"):
            body += f"\n\n:warning: Removed citations to non-retrieved sources: {res['invalid_citations']}"
        if res.get("llm_error"):
            body += f"\n\n:warning: LLM unavailable ({res['llm_error']}); fell back to extractive answer."
        if not res["sources"].empty:
            body += "\n\n**Retrieved sources**\n\n" + _sources_md(res["sources"], res["cited"])
        st.markdown(body)
    st.session_state["chat"].append({"role": "assistant", "content": body})
