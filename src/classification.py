"""
Traditional machine-learning classification: TF-IDF features + linear classifiers.

TF-IDF (term frequency - inverse document frequency) represents each abstract as a
sparse vector. For term t in document d:

    tfidf(t, d) = tf(t, d) * idf(t),     idf(t) = ln((1 + N) / (1 + df(t))) + 1

where N is the number of training documents and df(t) the number containing t
(scikit-learn's smoothed form). With `sublinear_tf=True`, tf is replaced by
1 + ln(tf), damping very repeated words. Each vector is then L2-normalised so long
and short abstracts are comparable. Words frequent in one paper but rare overall get
high weight; words that appear everywhere ("model", "data") get low weight.

Three classifiers are compared:
  * Multinomial Naive Bayes - generative, assumes words are conditionally independent
    given the class; very fast, a classic text baseline (McCallum & Nigam, 1998).
  * Logistic Regression - discriminative linear model giving class probabilities.
  * Linear SVM - finds a maximum-margin linear boundary; strong in high-dimensional
    sparse spaces such as TF-IDF (Joachims, 1998).

Leakage control: the vectoriser is always fit inside a Pipeline on training data only;
hyperparameters are chosen on the validation set; the test set is evaluated once.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from src.preprocessing import clean_for_tfidf
from src.utils import Timer, get_logger

logger = get_logger()

MODEL_DISPLAY_NAMES = {"naive_bayes": "Multinomial Naive Bayes",
                       "logistic_regression": "Logistic Regression",
                       "linear_svm": "Linear SVM",
                       # deep-learning models (python train.py deep); not joblib pipelines,
                       # so available_models() never offers them to the app's classifier page
                       "minilm_mlp": "MiniLM embeddings + MLP",
                       "distilbert": "Fine-tuned DistilBERT"}
CLASSICAL_MODELS = ["naive_bayes", "logistic_regression", "linear_svm"]

# Classifier hyperparameters searched after the TF-IDF settings are fixed.
CLASSIFIER_GRID = {
    "naive_bayes": {"alpha": [0.01, 0.05, 0.1, 0.5]},       # additive (Laplace/Lidstone) smoothing
    "logistic_regression": {"C": [1.0, 5.0, 20.0]},         # inverse L2 regularisation strength
    "linear_svm": {"C": [0.1, 0.3, 1.0]},                   # inverse regularisation (margin softness)
}
DEFAULT_PARAMS = {"naive_bayes": {"alpha": 0.1}, "logistic_regression": {"C": 5.0},
                  "linear_svm": {"C": 0.3}}


class TextCleaner(BaseEstimator, TransformerMixin):
    """Pipeline step that applies `clean_for_tfidf`, so saved models accept raw abstracts."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_for_tfidf(x) for x in X]


def make_classifier(model_name: str, **params):
    if model_name == "naive_bayes":
        return MultinomialNB(**params)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=config.SEED, **params)
    if model_name == "linear_svm":
        return LinearSVC(random_state=config.SEED, **params)
    raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_DISPLAY_NAMES)}")


def make_vectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2, max_df=0.9) -> TfidfVectorizer:
    """
    Args:
        ngram_range: (1,1) unigrams only; (1,2) adds bigrams such as 'dark matter'.
        max_features: keep only the most frequent terms (vocabulary size).
        min_df: ignore terms in fewer than this many documents (removes typos/rare noise).
        max_df: ignore terms in more than this fraction of documents (corpus stop words).
    Input is pre-cleaned, whitespace-separated text, so tokens are split on whitespace.
    """
    return TfidfVectorizer(ngram_range=tuple(ngram_range), max_features=max_features,
                           min_df=min_df, max_df=max_df, sublinear_tf=True,
                           lowercase=False, token_pattern=r"\S+", dtype=np.float32)


def build_pipeline(model_name: str, tfidf_params: dict | None = None,
                   clf_params: dict | None = None, include_cleaner: bool = True) -> Pipeline:
    """Cleaner (optional) -> TF-IDF -> classifier."""
    steps = [("clean", TextCleaner())] if include_cleaner else []
    steps += [("tfidf", make_vectorizer(**(tfidf_params or {}))),
              ("clf", make_classifier(model_name, **(clf_params or DEFAULT_PARAMS[model_name])))]
    return Pipeline(steps)


def _grid(d: dict) -> list[dict]:
    keys = list(d)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(d[k] for k in keys))]


def tune_on_validation(train_text, y_train, val_text, y_val,
                       models=config.CLASSIFIERS, tfidf_grid=config.TFIDF_GRID) -> pd.DataFrame:
    """
    Two-step validation search (keeps CPU cost manageable):
      1. For every TF-IDF configuration, fit the vectoriser on TRAIN once, then fit each
         classifier with default hyperparameters and score macro-F1 on VALIDATION.
      2. For each model, keep its best TF-IDF configuration and search its own grid.
    Returns one row per fitted configuration.
    """
    rows = []
    for tp in _grid(tfidf_grid):
        with Timer() as tv:
            vec = make_vectorizer(**tp)
            Xtr = vec.fit_transform(train_text); Xva = vec.transform(val_text)
        for m in models:
            with Timer() as t:
                clf = make_classifier(m, **DEFAULT_PARAMS[m]).fit(Xtr, y_train)
            f1 = f1_score(y_val, clf.predict(Xva), average="macro")
            rows.append({"model": m, "step": "tfidf", **{f"tfidf_{k}": str(v) for k, v in tp.items()},
                         "clf_params": str(DEFAULT_PARAMS[m]), "val_macro_f1": f1,
                         "n_features": Xtr.shape[1], "fit_s": round(t.elapsed + tv.elapsed, 2),
                         "_tfidf": tp, "_clf": DEFAULT_PARAMS[m]})
            logger.info("%-20s %s  val macro-F1=%.4f", m, tp, f1)
    res = pd.DataFrame(rows)
    for m in models:
        best_tp = res[res["model"] == m].sort_values("val_macro_f1", ascending=False).iloc[0]["_tfidf"]
        vec = make_vectorizer(**best_tp)
        Xtr = vec.fit_transform(train_text); Xva = vec.transform(val_text)
        for cp in _grid(CLASSIFIER_GRID[m]):
            if cp == DEFAULT_PARAMS[m]:
                continue
            with Timer() as t:
                clf = make_classifier(m, **cp).fit(Xtr, y_train)
            f1 = f1_score(y_val, clf.predict(Xva), average="macro")
            rows.append({"model": m, "step": "classifier", **{f"tfidf_{k}": str(v) for k, v in best_tp.items()},
                         "clf_params": str(cp), "val_macro_f1": f1, "n_features": Xtr.shape[1],
                         "fit_s": round(t.elapsed, 2), "_tfidf": best_tp, "_clf": cp})
            logger.info("%-20s %s  val macro-F1=%.4f", m, cp, f1)
    return pd.DataFrame(rows)


def best_configs(results: pd.DataFrame) -> dict[str, dict]:
    """Best (tfidf_params, clf_params) per model by validation macro-F1."""
    out = {}
    for m, g in results.groupby("model"):
        row = g.sort_values("val_macro_f1", ascending=False).iloc[0]
        out[m] = {"tfidf": row["_tfidf"], "clf": row["_clf"], "val_macro_f1": float(row["val_macro_f1"])}
    return out


def train_final(model_name: str, texts, labels, tfidf_params: dict, clf_params: dict) -> Pipeline:
    """Fit the full raw-text pipeline (cleaner included) with the selected settings."""
    pipe = build_pipeline(model_name, tfidf_params, clf_params, include_cleaner=True)
    return pipe.fit(list(texts), list(labels))


def save_model(pipe: Pipeline, model_name: str, out_dir: Path = config.CLASSIFICATION_MODELS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.joblib"
    joblib.dump(pipe, path)
    return path


def load_model(model_name: str, model_dir: Path = config.CLASSIFICATION_MODELS_DIR) -> Pipeline:
    path = model_dir / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python train.py classify` first.")
    return joblib.load(path)


def available_models(model_dir: Path = config.CLASSIFICATION_MODELS_DIR) -> list[str]:
    return [m for m in MODEL_DISPLAY_NAMES if (model_dir / f"{m}.joblib").exists()]


def predict(pipe: Pipeline, texts: list[str]) -> list[str]:
    return list(pipe.predict(list(texts)))


def predict_with_scores(pipe: Pipeline, text: str, top_k: int = 3) -> pd.DataFrame:
    """
    Top-k categories with a score for one abstract.

    NB and LR give class probabilities (predict_proba). LinearSVC only gives decision
    values (signed distances to each class boundary), which are NOT probabilities;
    the column name says which kind of score is shown.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        scores, kind = pipe.predict_proba([text])[0], "probability"
    else:
        scores, kind = pipe.decision_function([text])[0], "decision value (not a probability)"
    order = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame({"category": clf.classes_[order], kind: np.round(scores[order], 4)})


def top_features(pipe: Pipeline, n: int = 10) -> pd.DataFrame:
    """Highest-weighted TF-IDF terms per class (interpretability for linear models / NB)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    weights = clf.feature_log_prob_ if hasattr(clf, "feature_log_prob_") else clf.coef_
    return pd.DataFrame({c: names[np.argsort(w)[::-1][:n]] for c, w in zip(clf.classes_, weights)})
