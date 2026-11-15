"""
Paper retrieval and Retrieval-Augmented Generation (RAG).

Ordinary prompting asks an LLM to answer from what it memorised during training, so
it can invent papers, results or citations. RAG first RETRIEVES relevant documents
(here: abstracts from the loaded dataset) and then asks the model to answer USING ONLY
those documents, citing them. Answers become traceable to real arXiv IDs, and the
system can refuse when nothing relevant is retrieved (Lewis et al., 2020).

Two retrieval back-ends share one interface:
  * dense  - sentence-transformers embeddings (all-MiniLM-L6-v2, 384-d, L2-normalised)
             searched with FAISS inner product (= cosine) or NumPy if FAISS is missing.
             Matches meaning ("car" ~ "automobile").
  * tfidf  - TF-IDF cosine over the abstracts. Needs no model download; matches words.
`build_index(backend="auto")` tries dense first and falls back to TF-IDF, recording
which was used.

Answering:
  * With an LLM: numbered retrieved abstracts are given as context; the model must
    cite them as [1], [2] and say when the context is insufficient. Citation numbers
    are validated against the retrieved set; invalid ones are removed and reported.
  * Without an LLM: an extractive answer quotes the retrieved sentences most similar
    to the question, each with its source - nothing is generated, so nothing is invented.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from src.llm_client import LLMClient
from src.title_generation import split_sentences
from src.utils import get_device, get_logger, load_json, save_json

logger = get_logger()

META_COLUMNS = ["id", "title", "label", "categories", "abstract", "authors"]
INDEX_META_FILE = config.EMBEDDINGS_DIR / "index_meta.json"
DOCS_FILE = config.EMBEDDINGS_DIR / "documents.parquet"


@lru_cache(maxsize=1)
def get_encoder(model_name: str = config.EMBEDDING_MODEL_NAME):
    """Load the sentence-transformers encoder once (downloads ~90 MB on first use)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=get_device(config.FORCE_CPU))


def _doc_text(df: pd.DataFrame) -> list[str]:
    """
    Text that is indexed: the ABSTRACT only. Titles are shown as metadata but not indexed,
    so the known-item evaluation (title as query -> find that paper) cannot simply match
    the query against an identical copy of itself, which would inflate Recall@k.
    """
    return df["abstract"].fillna("").tolist()


class Retriever:
    """Top-K search over a fixed document collection."""

    def __init__(self, docs: pd.DataFrame, backend: str, vectorizer=None, matrix=None,
                 embeddings: np.ndarray | None = None, faiss_index=None) -> None:
        self.docs = docs.reset_index(drop=True)
        self.backend = backend
        self.vectorizer, self.matrix = vectorizer, matrix
        self.embeddings, self.faiss_index = embeddings, faiss_index

    def _scores_tfidf(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel()

    def search(self, query: str, k: int = config.RAG_TOP_K,
               category: str | list[str] | None = None) -> pd.DataFrame:
        """
        Return the k most similar documents with a cosine `score` in [-1, 1].
        `category` restricts results to one category or a list of categories; the filter is
        applied BEFORE ranking, so k results are returned whenever k matching papers exist.
        """
        if not query or not query.strip():
            return self.docs.head(0).assign(score=[], rank=[])
        if self.backend == "dense":
            q = get_encoder().encode([query], normalize_embeddings=True).astype(np.float32)
            if self.faiss_index is not None and category is None:
                scores, idx = self.faiss_index.search(q, k)
                scores, idx = scores[0], idx[0]
            else:
                all_scores = self.embeddings @ q[0]
                scores, idx = self._top(all_scores, k, category)
        else:
            scores, idx = self._top(self._scores_tfidf(query), k, category)
        out = self.docs.iloc[idx].copy()
        out["score"] = np.round(scores, 4)
        out["rank"] = range(1, len(out) + 1)
        return out.reset_index(drop=True)

    def _top(self, scores: np.ndarray, k: int, category: str | list[str] | None):
        if category:
            cats = [category] if isinstance(category, str) else list(category)
            scores = np.where(np.isin(self.docs["label"].values, cats), scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        idx = np.argpartition(-scores, k - 1)[:k] if k > 0 else np.array([], dtype=int)
        idx = idx[np.argsort(-scores[idx])]
        return scores[idx], idx

    def similar_to_paper(self, paper_id: str, k: int = 5) -> pd.DataFrame:
        """Papers most similar to an existing paper (excluding itself)."""
        row = self.docs[self.docs["id"] == paper_id]
        if row.empty:
            return self.docs.head(0)
        res = self.search(_doc_text(row)[0], k=k + 1)
        return res[res["id"] != paper_id].head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Building / loading
# ---------------------------------------------------------------------------
def build_index(docs: pd.DataFrame, backend: str = "auto",
                out_dir: Path = config.EMBEDDINGS_DIR) -> Retriever:
    """Build and save a retrieval index. backend: 'auto' | 'dense' | 'tfidf'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = docs[[c for c in META_COLUMNS if c in docs.columns]].reset_index(drop=True)
    docs.to_parquet(out_dir / DOCS_FILE.name, index=False)
    texts = _doc_text(docs)
    meta = {"n_documents": len(docs)}
    if backend in ("auto", "dense"):
        try:
            enc = get_encoder()
            emb = enc.encode(texts, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True).astype(np.float32)
            np.save(out_dir / "embeddings.npy", emb)
            index = None
            try:
                import faiss

                index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
                faiss.write_index(index, str(out_dir / "faiss.index"))
                meta["vector_store"] = "faiss IndexFlatIP (exact cosine)"
            except ImportError:
                meta["vector_store"] = "numpy (faiss not installed)"
            meta.update(backend="dense", model=config.EMBEDDING_MODEL_NAME, dim=int(emb.shape[1]))
            save_json(meta, out_dir / INDEX_META_FILE.name)
            return Retriever(docs, "dense", embeddings=emb, faiss_index=index)
        except Exception as e:
            if backend == "dense":
                raise
            logger.warning("Dense embeddings unavailable (%s: %s); using TF-IDF retrieval.",
                           type(e).__name__, str(e)[:120])
            meta["dense_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
                          min_df=2, max_features=200_000, dtype=np.float32)
    matrix = vec.fit_transform(texts)
    joblib.dump({"vectorizer": vec, "matrix": matrix}, out_dir / "tfidf_index.joblib")
    meta.update(backend="tfidf", vector_store="scipy sparse matrix (exact cosine)",
                n_features=int(matrix.shape[1]))
    save_json(meta, out_dir / INDEX_META_FILE.name)
    return Retriever(docs, "tfidf", vectorizer=vec, matrix=matrix)


def load_retriever(out_dir: Path = config.EMBEDDINGS_DIR) -> Retriever:
    meta_path = out_dir / INDEX_META_FILE.name
    if not meta_path.exists():
        raise FileNotFoundError("No retrieval index. Run `python train.py index` first.")
    meta = load_json(meta_path)
    docs = pd.read_parquet(out_dir / DOCS_FILE.name)
    if meta["backend"] == "dense":
        emb = np.load(out_dir / "embeddings.npy")
        index = None
        if (out_dir / "faiss.index").exists():
            try:
                import faiss

                index = faiss.read_index(str(out_dir / "faiss.index"))
            except ImportError:
                pass
        return Retriever(docs, "dense", embeddings=emb, faiss_index=index)
    blob = joblib.load(out_dir / "tfidf_index.joblib")
    return Retriever(docs, "tfidf", vectorizer=blob["vectorizer"], matrix=blob["matrix"])


def index_info(out_dir: Path = config.EMBEDDINGS_DIR) -> dict | None:
    p = out_dir / INDEX_META_FILE.name
    return load_json(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
OUT_OF_DOMAIN_QUERIES = [
    "What is the best recipe for banana bread?", "Who won the football world cup in 2018?",
    "How do I renew my passport?", "Best hiking trails in the Lake District",
    "How to train a puppy to sit", "What time does the supermarket close on Sunday?",
    "Tips for growing tomatoes on a balcony", "How much does a used car cost?",
    "What should I wear to a wedding?", "How do I fix a leaking kitchen tap?",
    "Ideas for a child's birthday party", "Which smartphone has the best camera?",
    "How to make cold brew coffee", "Where can I buy cheap flights to Spain?",
    "How do you knit a scarf?", "What is a good bedtime routine for toddlers?",
    "How to remove red wine stains from carpet", "Recommend a romantic comedy film",
    "How do I change a car tyre?", "What are the opening hours of the British Museum?",
]


def evaluate_retrieval(retriever: Retriever, queries: pd.DataFrame, k_values=(1, 5, 10)) -> dict:
    """
    Known-item retrieval: each query is a paper's TITLE and the relevant document is
    that paper. Reports Recall@k and MRR@10, plus category precision@10 (share of the
    top 10 results sharing the query paper's primary category).
    """
    import time

    ranks, cat_prec, lat = [], [], []
    for _, row in queries.iterrows():
        t0 = time.perf_counter(); res = retriever.search(row["title"], k=max(k_values))
        lat.append(time.perf_counter() - t0)
        hit = res.index[res["id"] == row["id"]].tolist()
        ranks.append(hit[0] + 1 if hit else None)
        cat_prec.append(float((res["label"] == row["label"]).mean()))
    out = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in k_values}
    out["mrr@10"] = float(np.mean([1.0 / r if r else 0.0 for r in ranks]))
    out["category_precision@10"] = float(np.mean(cat_prec))
    out["median_latency_ms"] = float(np.median(lat) * 1000)
    out["n_queries"] = len(queries)
    return out


def calibrate_threshold(retriever: Retriever, in_domain: list[str],
                        out_domain: list[str] = OUT_OF_DOMAIN_QUERIES,
                        min_in_domain_recall: float = 0.95) -> dict:
    """
    Choose the minimum top-1 similarity needed to answer, from measured score
    distributions of in-domain scientific queries and out-of-domain everyday questions.

    Two candidates are reported:
      * best_balanced - maximises balanced accuracy (in-domain answered + OOD refused) / 2
      * threshold     - the value USED by the app: the highest threshold that still lets
                        `min_in_domain_recall` (95%) of in-domain queries through, because
                        wrongly refusing a genuine question is the more costly error here.
    """
    s_in = np.array([retriever.search(q, k=1)["score"].max() for q in in_domain])
    s_out = np.array([retriever.search(q, k=1)["score"].max() for q in out_domain])
    best_t, best_ba = 0.0, -1.0
    for t in np.unique(np.concatenate([s_in, s_out])):
        ba = 0.5 * ((s_in >= t).mean() + (s_out < t).mean())
        if ba > best_ba:
            best_t, best_ba = float(t), float(ba)
    used = float(np.quantile(s_in, 1 - min_in_domain_recall))
    return {"threshold": round(used, 4),
            "threshold_rule": f"keep {min_in_domain_recall:.0%} of in-domain queries answerable",
            "in_domain_answered": round(float((s_in >= used).mean()), 4),
            "out_domain_refused": round(float((s_out < used).mean()), 4),
            "best_balanced_threshold": round(best_t, 4), "best_balanced_accuracy": round(best_ba, 4),
            "in_domain_median": round(float(np.median(s_in)), 4),
            "out_domain_median": round(float(np.median(s_out)), 4),
            "n_in": len(s_in), "n_out": len(s_out)}


# ---------------------------------------------------------------------------
# Grounded answering
# ---------------------------------------------------------------------------
RAG_SYSTEM = ("You answer questions about scientific papers using ONLY the numbered paper "
              "abstracts provided. Cite every claim with the paper number in square brackets, "
              "e.g. [2]. If the abstracts do not contain the answer, say exactly: "
              "'The retrieved papers do not contain enough information to answer this.' "
              "Never mention papers that are not in the list.")


def _context_block(hits: pd.DataFrame) -> str:
    parts = []
    for i, r in enumerate(hits.itertuples(), start=1):
        abstract = " ".join(str(r.abstract).split()[:220])
        parts.append(f"[{i}] arXiv:{r.id} | {r.title}\n{abstract}")
    return "\n\n".join(parts)


def validate_citations(answer: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """Return (answer with invalid citations removed, valid numbers used, invalid numbers)."""
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", answer or "")]
    valid = sorted({c for c in cited if 1 <= c <= n_sources})
    invalid = sorted({c for c in cited if not 1 <= c <= n_sources})
    cleaned = answer
    for c in invalid:
        cleaned = cleaned.replace(f"[{c}]", "")
    return cleaned, valid, invalid


def extractive_answer(question: str, hits: pd.DataFrame, n_sentences: int = 3) -> str:
    """Quote the retrieved sentences most similar to the question, with their source number."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sents = [(i, s) for i, a in enumerate(hits["abstract"], start=1) for s in split_sentences(a)]
    if not sents:
        return ""
    vec = TfidfVectorizer(stop_words="english").fit([question] + [s for _, s in sents])
    sims = (vec.transform([s for _, s in sents]) @ vec.transform([question]).T).toarray().ravel()
    best = np.argsort(-sims)[:n_sentences]
    return " ".join(f"“{sents[j][1]}” [{sents[j][0]}]" for j in sorted(best))


def answer_question(question: str, retriever: Retriever, client: LLMClient | None = None,
                    k: int = config.RAG_TOP_K, threshold: float | None = None) -> dict:
    """
    Retrieve, then answer with an LLM (if available) or extractively.
    Returns {answer, mode, sources (DataFrame), cited, invalid_citations, refused}.
    """
    hits = retriever.search(question, k=k)
    result = {"answer": "", "mode": "", "sources": hits, "cited": [], "invalid_citations": [],
              "refused": False}
    if hits.empty or (threshold is not None and hits["score"].max() < threshold):
        result.update(refused=True, mode="refused",
                      answer="No sufficiently relevant papers were found in the loaded dataset, "
                             "so I cannot answer this from the available evidence.")
        return result
    if client is not None and client.available:
        messages = [{"role": "system", "content": RAG_SYSTEM},
                    {"role": "user", "content": f"Papers:\n\n{_context_block(hits)}\n\nQuestion: {question}"}]
        reply = client.chat(messages, max_tokens=400)
        if reply:
            cleaned, valid, invalid = validate_citations(reply, len(hits))
            result.update(answer=cleaned, mode="llm_grounded", cited=valid, invalid_citations=invalid)
            return result
        result["llm_error"] = client.last_error
    ans = extractive_answer(question, hits)
    _, valid, _ = validate_citations(ans, len(hits))
    result.update(answer=ans, mode="extractive", cited=valid)
    return result
