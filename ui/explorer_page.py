"""Page 5 - Research paper explorer: search, filter, view, similar papers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from ui.components import arxiv_link, load_labels, load_parquet, load_retriever, missing_artefact


def _keyword_search(df: pd.DataFrame, query: str, cats: list[str], n: int) -> pd.DataFrame:
    if cats:
        df = df[df["label"].isin(cats)]
    if query.strip():
        mask = (df["title"].str.contains(query, case=False, regex=False)
                | df["abstract"].str.contains(query, case=False, regex=False))
        df = df[mask]
    return df.head(n)


def _show(results: pd.DataFrame, retriever, allow_similar: bool) -> None:
    for _, r in results.iterrows():
        score = f" · score {r['score']:.3f}" if "score" in r and pd.notna(r.get("score")) else ""
        with st.expander(f"{r['title']}  —  {r['label']}{score}"):
            st.markdown(f"[arXiv:{r['id']}]({arxiv_link(r['id'])}) · categories: `{r['categories']}`")
            if isinstance(r.get("authors"), str):
                st.caption(r["authors"][:300])
            st.write(r["abstract"])
            if allow_similar and st.button("Find similar papers", key=f"sim_{r['id']}"):
                st.session_state["similar_to"] = (r["id"], r["title"])


def render() -> None:
    st.title("Research paper explorer")
    try:
        retriever = load_retriever()
    except FileNotFoundError:
        retriever = None
    docs = retriever.docs if retriever else load_parquet(str(config.SAMPLE_FILE))
    if docs is None:
        missing_artefact("No papers loaded.", "python train.py prepare")
        return
    mode_note = (f"Ranked search ({retriever.backend}) over {len(docs):,} papers (train + validation)."
                 if retriever else f"Keyword filter over {len(docs):,} papers. Build an index for ranked "
                                   "and similar-paper search: `python train.py index`.")
    st.caption(mode_note)

    c = st.columns([3, 2, 1])
    query = c[0].text_input("Search titles and abstracts", placeholder="e.g. graph neural networks for molecules")
    cats = c[1].multiselect("Categories", load_labels())
    n = c[2].number_input("Results", 5, 50, 10, step=5)

    if retriever and query.strip():
        results = retriever.search(query, k=int(n), category=cats or None)
    else:
        results = _keyword_search(docs, query, cats, int(n))
    st.write(f"{len(results)} result(s)")
    _show(results, retriever, allow_similar=retriever is not None)

    if retriever and "similar_to" in st.session_state:
        pid, title = st.session_state["similar_to"]
        st.subheader(f"Papers similar to: {title}")
        _show(retriever.similar_to_paper(pid, k=5), retriever, allow_similar=False)
