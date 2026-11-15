"""Retrieval, grounded answering, citation validation and a Streamlit smoke test."""

from pathlib import Path

import pytest

from src.rag import answer_question, build_index, validate_citations
from src.title_generation import validate_abstract


@pytest.fixture
def retriever(labelled_df, tmp_path):
    return build_index(labelled_df, backend="tfidf", out_dir=tmp_path)


def test_ui_validation_messages():
    assert validate_abstract("")[1] == "Please enter an abstract."
    assert "too short" in validate_abstract("short text only")[1]


def test_tfidf_search_ranks_topic_papers_first(retriever):
    res = retriever.search("qubit entanglement photon", k=5)
    assert len(res) == 5 and (res["label"] == "quant-ph").all()
    assert list(res["rank"]) == [1, 2, 3, 4, 5]
    assert res["score"].is_monotonic_decreasing


def test_category_filter_and_similar_papers(retriever, labelled_df):
    res = retriever.search("image segmentation", k=3, category="cs.CL")
    assert (res["label"] == "cs.CL").all()
    pid = labelled_df.iloc[0]["id"]
    sim = retriever.similar_to_paper(pid, k=3)
    assert pid not in set(sim["id"]) and len(sim) == 3


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ").empty


def test_validate_citations_removes_out_of_range():
    cleaned, valid, invalid = validate_citations("Claim [1] and [3] and fake [9].", n_sources=3)
    assert valid == [1, 3] and invalid == [9] and "[9]" not in cleaned


def test_extractive_answer_cites_only_retrieved_sources(retriever):
    res = answer_question("How is entanglement of qubits studied?", retriever, client=None, k=3)
    assert res["mode"] == "extractive" and not res["refused"]
    assert res["cited"] and all(1 <= c <= 3 for c in res["cited"])
    assert set(res["sources"]["id"]) <= set(retriever.docs["id"])


def test_refuses_below_evidence_threshold(retriever):
    res = answer_question("banana bread recipe", retriever, client=None, threshold=0.99)
    assert res["refused"] and res["mode"] == "refused"


def test_llm_grounded_answer_with_invalid_citation(retriever, monkeypatch):
    import requests

    from src.llm_client import LLMClient

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Qubits are entangled [1], see also [7]."}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    client = LLMClient(backend="openai", base_url="http://x/v1", model="m", cache_file=None)
    res = answer_question("qubit entanglement", retriever, client=client, k=3)
    assert res["mode"] == "llm_grounded" and res["cited"] == [1] and res["invalid_citations"] == [7]
    assert "[7]" not in res["answer"]


def test_streamlit_app_starts_without_exceptions():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=60)
    at.run()
    assert not at.exception
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"page '{page}' raised {at.exception[0].value}"


def test_multi_category_filter_returns_k_results(retriever):
    res = retriever.search("image segmentation", k=6, category=["cs.CL", "quant-ph"])
    assert len(res) == 6 and set(res["label"]) <= {"cs.CL", "quant-ph"}


def test_index_does_not_contain_titles(retriever, labelled_df):
    """Known-item evaluation must not match a title query against an indexed copy of itself."""
    from src.rag import _doc_text

    texts = _doc_text(labelled_df)
    assert all(t == a for t, a in zip(texts, labelled_df["abstract"]))
