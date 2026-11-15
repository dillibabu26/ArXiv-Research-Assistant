"""
Shared evaluation functions so every model is measured in exactly the same way.

Classification: accuracy, macro / weighted precision-recall-F1, per-class report,
confusion matrix, inference time.
Title generation: ROUGE-1 / ROUGE-2 / ROUGE-L F1 and (optionally) semantic similarity.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config

PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
MUTED = "#52514e"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                           labels: Sequence[str] | None = None) -> dict:
    """
    Accuracy plus macro and weighted averages.

    Macro averaging computes the metric per class and takes the unweighted mean, so
    every category counts equally regardless of its size; weighted averaging weights
    each class by its support (number of true examples). Predictions outside `labels`
    (e.g. an invalid LLM answer) count as errors for the true class.
    """
    labels = list(labels) if labels is not None else sorted(set(y_true))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f), "weighted_f1": float(wf),
            "n": int(len(y_true))}


def classification_report_df(y_true, y_pred, labels) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(4)


def plot_confusion_matrix(y_true, y_pred, labels, title: str, path, normalize: bool = True) -> np.ndarray:
    """Row-normalised confusion matrix (each row sums to 1 = recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = shown[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}" if normalize else str(v), ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "#0b0b0b")
    ax.set_xlabel("predicted category"); ax.set_ylabel("true category")
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)
    return cm


def time_inference(fn: Callable[[list[str]], object], inputs: list[str],
                   single_calls: int = 200) -> dict:
    """
    Measure inference time in two ways:
      * batch: total time for all inputs / number of inputs (throughput)
      * single: median latency of `single_calls` one-abstract calls (what a UI user feels)
    A warm-up call is made first so one-off costs (lazy imports, memory allocation) are
    not attributed to whichever model happens to be timed first.
    """
    fn(inputs[:5])
    t0 = time.perf_counter(); fn(inputs); batch = (time.perf_counter() - t0) / max(1, len(inputs))
    lat = []
    for text in inputs[:single_calls]:
        t0 = time.perf_counter(); fn([text]); lat.append(time.perf_counter() - t0)
    return {"batch_ms_per_doc": round(batch * 1000, 4),
            "single_ms_median": round(float(np.median(lat)) * 1000, 3) if lat else None}


def plot_metric_comparison(df: pd.DataFrame, metrics: list[str], title: str, path,
                           label_col: str = "model") -> None:
    """Grouped horizontal bars: one group per model, one bar per metric."""
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(df) * len(metrics) / 2 + 1))
    colors = [PRIMARY, SECONDARY, "#1baf7a", "#6250d6"]
    h = 0.8 / len(metrics)
    y = np.arange(len(df))
    for k, m in enumerate(metrics):
        ax.barh(y + k * h, df[m], h * 0.92, color=colors[k % len(colors)], label=METRIC_LABELS.get(m, m.replace("_", " ")))
        for yi, v in zip(y + k * h, df[m]):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y + h * (len(metrics) - 1) / 2, df[label_col])
    ax.set_xlim(0, 1.08); ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(metrics), fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e5e5e2"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------
def rouge_scores(references: Sequence[str], predictions: Sequence[str]) -> pd.DataFrame:
    """
    Per-example ROUGE-1, ROUGE-2 and ROUGE-L F1 (Lin, 2004), with Porter stemming.
    ROUGE-N counts overlapping n-grams; ROUGE-L uses the longest common subsequence.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for ref, pred in zip(references, predictions):
        s = scorer.score(str(ref), str(pred or ""))
        rows.append({"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure,
                     "rougeL": s["rougeL"].fmeasure})
    return pd.DataFrame(rows)


def semantic_similarity(references: Sequence[str], predictions: Sequence[str]) -> tuple[np.ndarray, str]:
    """
    Cosine similarity between each reference and generated title.

    Uses the sentence-transformers encoder when it is installed and downloadable
    (meaning-level similarity). Otherwise falls back to TF-IDF cosine fitted on the
    titles themselves, which measures word overlap only; the method used is returned
    so it can be reported honestly.
    """
    refs = [str(r) for r in references]
    preds = [str(p or "") for p in predictions]
    try:
        from src.rag import get_encoder

        enc = get_encoder()
        a = enc.encode(refs, normalize_embeddings=True)
        b = enc.encode(preds, normalize_embeddings=True)
        return np.sum(a * b, axis=1), f"sentence-transformers ({config.EMBEDDING_MODEL_NAME})"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(stop_words="english").fit(refs + preds)
        a, b = vec.transform(refs), vec.transform(preds)
        return np.asarray(a.multiply(b).sum(axis=1)).ravel(), "tfidf-cosine (fallback)"


def unsupported_term_rate(abstract: str, title: str) -> float:
    """
    Share of the generated title's content words (stemmed, stop words removed) that do
    NOT occur in the abstract. A simple, transparent hallucination proxy: a high value
    means the title introduces vocabulary the source does not contain. It cannot detect
    false claims made with the abstract's own words, so it is a lower bound on risk.
    """
    from src.preprocessing import clean_for_tfidf

    t = set(clean_for_tfidf(title or "", stem=True).split()) - {"mathexpr", "urltoken"}
    if not t:
        return 0.0
    a = set(clean_for_tfidf(abstract or "", stem=True).split())
    return len(t - a) / len(t)


# Readable names for charts and tables (CSV files keep the short method keys).
METHOD_LABELS = {
    "flan_t5": "FLAN-T5 (zero-shot)",
    "lead_baseline": "First-sentence baseline",
    "centroid_baseline": "Central-sentence baseline",
    "llm_zero_shot": "LLM: zero-shot prompt",
    "llm_instructed": "LLM: detailed instructions",
    "llm_few_shot": "LLM: few-shot prompt",
}
METRIC_LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
                 "accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted F1"}


def method_label(key: str) -> str:
    return METHOD_LABELS.get(key, key)


def summarise_title_scores(df: pd.DataFrame, group_col: str = "method") -> pd.DataFrame:
    cols = [c for c in ["rouge1", "rouge2", "rougeL", "similarity", "unsupported_rate", "words",
                        "seconds_per_title"] if c in df.columns]
    out = df.groupby(group_col)[cols].mean().round(4)
    out["n"] = df.groupby(group_col).size()
    return out


# ---------------------------------------------------------------------------
# Statistical uncertainty and significance
# ---------------------------------------------------------------------------
def _metrics_from_codes(t: np.ndarray, p: np.ndarray, k: int) -> tuple[float, float]:
    """Accuracy and macro-F1 from integer-coded labels (fast path for bootstrapping).
    Predictions outside the label set are coded k and count as errors."""
    acc = float(np.mean(t == p))
    tp = np.bincount(t[t == p], minlength=k + 1)[:k]
    pred_n = np.bincount(p, minlength=k + 1)[:k]
    true_n = np.bincount(t, minlength=k + 1)[:k]
    prec = np.divide(tp, pred_n, out=np.zeros(k), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.zeros(k), where=true_n > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return acc, float(f1.mean())


def bootstrap_ci(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str],
                 metric: str = "macro_f1", n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = config.SEED) -> tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for accuracy or macro-F1.

    The test set is resampled with replacement `n_boot` times and the metric is
    recomputed on each resample; the 2.5th and 97.5th percentiles give a 95% interval
    (Efron and Tibshirani, 1993). Returns (point estimate, lower, upper).
    """
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError("metric must be 'accuracy' or 'macro_f1'")
    labels = list(labels)
    code = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    t = np.array([code.get(v, k) for v in y_true])
    p = np.array([code.get(v, k) for v in y_pred])
    j = 0 if metric == "accuracy" else 1
    point = classification_metrics(y_true, y_pred, labels)[metric]
    rng = np.random.default_rng(seed)
    n = len(t)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = _metrics_from_codes(t[idx], p[idx], k)[j]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mcnemar_test(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    """
    McNemar's test for two classifiers evaluated on the SAME examples (Dietterich, 1998).

    Only discordant pairs matter: b = A right & B wrong, c = A wrong & B right.
    Under H0 (equal error rates) b and c are equally likely. An exact two-sided binomial
    test is used when b + c < 25, otherwise the chi-squared statistic with continuity
    correction, (|b - c| - 1)^2 / (b + c), with 1 degree of freedom.
    """
    import math

    y_true, a, b_ = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    a_ok, b_ok = a == y_true, b_ == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "statistic": 0.0, "p_value": 1.0, "method": "no disagreements"}
    if n < 25:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        return {"a_only_correct": b, "b_only_correct": c, "statistic": float(k), "p_value": p, "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))  # survival function of chi-squared with 1 df
    return {"a_only_correct": b, "b_only_correct": c, "statistic": float(stat), "p_value": float(p),
            "method": "chi-squared (continuity-corrected)"}
