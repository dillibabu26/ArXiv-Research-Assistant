"""
ArXiv Research Assistant - Streamlit entry point.

    streamlit run app.py      ->  http://localhost:8501

This file only sets up the page and navigation; each page lives in ui/.
Run `python train.py all` first so models and results exist.
"""

import streamlit as st

st.set_page_config(page_title="ArXiv Research Assistant", page_icon="📄", layout="wide")

from ui import (chatbot_page, classification_page, comparison_page, dashboard,  # noqa: E402
                explorer_page, generation_page)

PAGES = {
    "Home dashboard": dashboard,
    "Paper classification": classification_page,
    "Title generation": generation_page,
    "Model comparison": comparison_page,
    "Paper explorer": explorer_page,
    "Research assistant chatbot": chatbot_page,
}

with st.sidebar:
    st.header("ArXiv Research Assistant")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("DG4NLP MSc project · runs locally · results shown are measured by train.py")

try:
    PAGES[page].render()
except Exception as e:  # last-resort guard: show a helpful message instead of a stack trace
    st.error(f"Something went wrong on this page: {type(e).__name__}: {e}")
    st.caption("Check the terminal for details. If artefacts are missing, run `python train.py all`.")
