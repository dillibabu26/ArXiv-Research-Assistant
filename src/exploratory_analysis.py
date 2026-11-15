"""
Exploratory data analysis: the seven required figures plus summary tables.

Every function saves a PNG under outputs/figures/ and returns the numbers it plotted,
so the interpretation written in the report is based on the measured values
(saved in outputs/results/eda_summary.json).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.preprocessing import split_categories

# Plot style: one accent hue for single series, a second for comparisons, recessive grid.
PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
MUTED = "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e2",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
})


def _save(fig, name: str, out_dir: Path = config.FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# 1 ---------------------------------------------------------------------------
def plot_top_categories(df: pd.DataFrame, n: int = config.EDA_TOP_CATEGORIES) -> dict:
    """Top-n primary categories across the whole loaded population."""
    counts = df["primary_category"].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=PRIMARY, height=0.7)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("papers (primary category)")
    ax.set_title(f"Top {n} primary arXiv categories (n = {len(df):,} papers)")
    ax.grid(axis="y", visible=False)
    _save(fig, "01_top_categories")
    share = float(counts.sum() / len(df))
    return {"top_categories": counts.to_dict(), "top_n_share": round(share, 4),
            "n_primary_categories": int(df["primary_category"].nunique())}


# 2 & 3 -----------------------------------------------------------------------
def _length_hist(series: pd.Series, title: str, xlabel: str, clip: int, name: str,
                 marker: int | None = None, marker_label: str = "") -> dict:
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s.clip(upper=clip), bins=min(60, clip), color=PRIMARY, edgecolor="white", linewidth=0.5)
    med = float(s.median())
    ax.axvline(med, color="#0b0b0b", ls="--", lw=1.2, label=f"median = {med:.0f}")
    if marker is not None:
        ax.axvline(marker, color=SECONDARY, ls=":", lw=1.6, label=marker_label)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("papers"); ax.legend(frameon=False)
    _save(fig, name)
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": round(float(s.mean()), 1), "std": round(float(s.std()), 1),
            "min": int(s.min()), "max": int(s.max()),
            "quantiles": {f"p{int(k * 100)}": float(v) for k, v in q.items()}}


def plot_abstract_length(df: pd.DataFrame) -> dict:
    words = df["abstract"].fillna("").str.split().str.len()
    return _length_hist(words, "Abstract length distribution", "words (clipped at 400)", 400,
                        "02_abstract_length", marker=config.MIN_ABSTRACT_WORDS,
                        marker_label=f"min-length filter ({config.MIN_ABSTRACT_WORDS})")


def plot_title_length(df: pd.DataFrame) -> dict:
    words = df["title"].fillna("").str.split().str.len()
    return _length_hist(words, "Title length distribution", "words (clipped at 30)", 30,
                        "03_title_length")


# 4 ---------------------------------------------------------------------------
def plot_missing_values(schema: dict, source: str) -> dict:
    pct = pd.Series(schema["missing_pct"]).drop(labels=["source"], errors="ignore").sort_values()
    absent = sorted(pct[pct >= 100].index)          # fields this source does not provide at all
    pct = pct[pct < 100]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(pct.index, pct.values, color=PRIMARY, height=0.6)
    for i, v in enumerate(pct.values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 105); ax.set_xlabel("% of records missing or empty")
    ax.set_title(f"Missing values per field (source: {source})")
    if absent:
        ax.text(0, -0.18, "Not provided by this dataset: " + ", ".join(absent), transform=ax.transAxes,
                fontsize=8, color=MUTED)
    ax.grid(axis="y", visible=False)
    _save(fig, "04_missing_values")
    return {"missing_pct": pct.to_dict(), "fields_not_in_source": absent}


# 5 ---------------------------------------------------------------------------
def _ngram_counts(texts: pd.Series, n: int) -> Counter:
    c: Counter = Counter()
    for t in texts:
        toks = [w for w in t.split() if w not in {"mathexpr", "urltoken"}]
        c.update(zip(*[toks[i:] for i in range(n)]) if n > 1 else toks)
    return c


def plot_most_frequent_words(sample: pd.DataFrame, top: int = 20) -> dict:
    """Most frequent unigrams and bigrams after TF-IDF cleaning (stop words removed)."""
    uni = _ngram_counts(sample["text_tfidf"], 1).most_common(top)
    bi = [(" ".join(k), v) for k, v in _ngram_counts(sample["text_tfidf"], 2).most_common(top)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, title in [(axes[0], uni, "unigrams"), (axes[1], bi, "bigrams")]:
        words, vals = zip(*data)
        ax.barh(list(words)[::-1], list(vals)[::-1], color=PRIMARY, height=0.7)
        ax.set_title(f"Top {top} {title} (stop words removed)"); ax.set_xlabel("occurrences")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Most frequent terms in the modelling sample (n = {len(sample):,} abstracts)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_most_frequent_words")
    return {"top_unigrams": dict(uni), "top_bigrams": dict(bi)}


def submission_year(arxiv_id: str) -> int | None:
    """Year a paper was first submitted, from its arXiv ID: '2511.15658' -> 2025, 'cs/9308101' -> 1993."""
    import re

    s = str(arxiv_id)
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", s) or re.match(r"^[a-z.\-]+/(\d{2})(\d{2})\d{3}$", s)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 91 else 2000 + yy


# 6 ---------------------------------------------------------------------------
def plot_category_imbalance(df: pd.DataFrame, sample: pd.DataFrame, labels: list[str]) -> dict:
    """Class counts for the selected labels before and after per-class capping, and cross-listing."""
    before = df["primary_category"].value_counts().reindex(labels)
    after = sample["label"].value_counts().reindex(labels)
    n_cats = df["categories"].map(lambda c: len(split_categories(c))).clip(upper=6)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(labels)); w = 0.4
    axes[0].bar(x - w / 2 - 0.01, before.values, w, color=SECONDARY, label="all loaded papers")
    axes[0].bar(x + w / 2 + 0.01, after.values, w, color=PRIMARY,
                label=f"modelling sample (cap {config.MAX_PER_CLASS:,}/class)")
    axes[0].set_xticks(x, labels, rotation=40, ha="right")
    axes[0].set_ylabel("papers"); axes[0].legend(frameon=False)
    axes[0].set_title(f"Class sizes: imbalance ratio {before.max() / before.min():.1f}:1 "
                      f"-> {after.max() / after.min():.1f}:1 after capping")
    if (n_cats > 1).any():
        vc = n_cats.value_counts().sort_index()
        axes[1].bar([str(i) if i < 6 else "6+" for i in vc.index], vc.values, color=PRIMARY, width=0.7)
        axes[1].set_title("Categories per paper (cross-listing)"); axes[1].set_xlabel("number of categories")
    else:
        # Single-label source: show when the papers were submitted instead (year from the arXiv ID).
        years = df["id"].map(submission_year).dropna().astype(int)
        vc = years.value_counts().sort_index()
        axes[1].bar(vc.index.astype(str), vc.values, color=PRIMARY, width=0.8)
        step = max(1, len(vc) // 8)
        axes[1].set_xticks(range(0, len(vc), step), vc.index.astype(str)[::step], rotation=45, ha="right")
        axes[1].set_title("Papers per submission year (one category per paper)")
        axes[1].set_xlabel("year")
    axes[1].set_ylabel("papers")
    fig.tight_layout()
    _save(fig, "06_category_imbalance")
    years = df["id"].map(submission_year).dropna()
    return {"before": before.to_dict(), "after": after.to_dict(),
            "year_counts": {int(k): int(v) for k, v in years.astype(int).value_counts().sort_index().items()},
            "imbalance_before": round(float(before.max() / before.min()), 2),
            "imbalance_after": round(float(after.max() / after.min()), 2),
            "cross_listed_share": round(float((n_cats > 1).mean()), 4)}


# 7 ---------------------------------------------------------------------------
def plot_word_frequency_analysis(sample: pd.DataFrame, top_per_class: int = 8) -> dict:
    """
    Zipf rank-frequency plot and cumulative vocabulary coverage (which motivates the
    TF-IDF `max_features` grid), plus the most class-distinctive words per category.
    """
    counts = _ngram_counts(sample["text_tfidf"], 1)
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    coverage = np.cumsum(freqs) / freqs.sum()
    cov_at = {k: round(float(coverage[min(k, len(coverage)) - 1]), 4) for k in (1_000, 5_000, 20_000, 50_000)}
    hapax = int((freqs == 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].loglog(ranks, freqs, color=PRIMARY, lw=2)
    axes[0].set_title("Zipf plot: term frequency vs rank (log-log)")
    axes[0].set_xlabel("rank"); axes[0].set_ylabel("frequency")
    axes[1].semilogx(ranks, coverage * 100, color=PRIMARY, lw=2)
    for k, y, ha in ((20_000, 20, "right"), (50_000, 8, "left")):
        if k <= len(ranks):
            axes[1].axvline(k, color=SECONDARY, ls=":", lw=1.4)
            axes[1].text(k, y, f" {k // 1000}k terms: {cov_at[k] * 100:.1f}% ", color=MUTED,
                         fontsize=8, ha=ha)
    axes[1].set_title("Cumulative token coverage by vocabulary size")
    axes[1].set_xlabel("vocabulary size (most frequent terms)"); axes[1].set_ylabel("% of tokens covered")
    fig.tight_layout()
    _save(fig, "07_word_frequency_analysis")

    # Distinctive words: relative frequency in class / relative frequency overall,
    # restricted to words with >= 50 occurrences to avoid rare-word noise.
    total = sum(counts.values())
    rows = []
    for label, g in sample.groupby("label", sort=True):
        cc = _ngram_counts(g["text_tfidf"], 1)
        ctotal = sum(cc.values())
        scores = {w: (cc[w] / ctotal) / (counts[w] / total) for w in cc if counts[w] >= 50}
        best = sorted(scores, key=scores.get, reverse=True)[:top_per_class]
        rows.append({"label": label, "distinctive_words": ", ".join(best)})
    distinctive = pd.DataFrame(rows)
    distinctive.to_csv(config.RESULTS_DIR / "eda_distinctive_words.csv", index=False)
    return {"vocabulary_size": int(len(freqs)), "total_tokens": int(freqs.sum()),
            "hapax_legomena": hapax, "hapax_share_of_vocab": round(hapax / len(freqs), 4),
            "coverage_at": cov_at}


def run_eda(raw: pd.DataFrame, clean: pd.DataFrame, sample: pd.DataFrame, labels: list[str],
            schema: dict, source: str) -> dict:
    """Produce all seven figures and return the measured summary values."""
    return {
        "top_categories": plot_top_categories(clean),
        "abstract_length": plot_abstract_length(clean),
        "title_length": plot_title_length(clean),
        "missing_values": plot_missing_values(schema, source),
        "frequent_words": plot_most_frequent_words(sample),
        "category_imbalance": plot_category_imbalance(clean, sample, labels),
        "word_frequency": plot_word_frequency_analysis(sample),
    }
