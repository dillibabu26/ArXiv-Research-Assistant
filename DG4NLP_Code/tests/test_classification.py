"""TF-IDF pipelines, predictions, persistence and metrics."""

import pytest

from src import classification as C
from src.evaluation import classification_metrics


@pytest.fixture
def splits(labelled_df):
    from src.preprocessing import split_dataset

    return split_dataset(labelled_df, test_size=0.2, val_size=0.2, seed=0)


@pytest.mark.parametrize("model", ["naive_bayes", "logistic_regression", "linear_svm"])
def test_each_model_learns_synthetic_topics(model, splits):
    train, _, test = splits
    pipe = C.build_pipeline(model, {"ngram_range": (1, 1), "max_features": 500, "min_df": 1, "max_df": 1.0})
    pipe.fit(train["abstract"].tolist(), train["label"].tolist())
    preds = C.predict(pipe, test["abstract"].tolist())
    assert len(preds) == len(test)
    assert classification_metrics(test["label"], preds)["accuracy"] >= 0.9


def test_pipeline_accepts_raw_text_and_returns_scores(splits):
    train, _, _ = splits
    pipe = C.build_pipeline("logistic_regression", {"min_df": 1, "max_df": 1.0}).fit(
        train["abstract"].tolist(), train["label"].tolist())
    top = C.predict_with_scores(pipe, "Entanglement of a QUBIT and a photon in a quantum circuit!", top_k=2)
    assert top.iloc[0]["category"] == "quant-ph"
    assert "probability" in top.columns and len(top) == 2


def test_svm_scores_are_labelled_as_not_probabilities(splits):
    train, _, _ = splits
    pipe = C.build_pipeline("linear_svm", {"min_df": 1, "max_df": 1.0}).fit(
        train["abstract"].tolist(), train["label"].tolist())
    cols = C.predict_with_scores(pipe, "image segmentation with convolutional camera video").columns
    assert any("not a probability" in c for c in cols)


def test_validation_search_and_best_config(splits):
    train, val, _ = splits
    grid = {"ngram_range": [(1, 1)], "max_features": [200], "min_df": [1], "max_df": [1.0]}
    res = C.tune_on_validation(train["text_tfidf"], train["label"], val["text_tfidf"], val["label"],
                               models=["naive_bayes"], tfidf_grid=grid)
    best = C.best_configs(res)
    assert "naive_bayes" in best and 0 <= best["naive_bayes"]["val_macro_f1"] <= 1
    assert len(res) == 1 + len(C.CLASSIFIER_GRID["naive_bayes"]["alpha"]) - 1


def test_save_and_load_roundtrip(tmp_path, splits):
    train, _, test = splits
    pipe = C.build_pipeline("naive_bayes", {"min_df": 1, "max_df": 1.0}).fit(
        train["abstract"].tolist(), train["label"].tolist())
    C.save_model(pipe, "naive_bayes", tmp_path)
    loaded = C.load_model("naive_bayes", tmp_path)
    assert C.predict(loaded, test["abstract"].tolist()) == C.predict(pipe, test["abstract"].tolist())
    assert C.available_models(tmp_path) == ["naive_bayes"]


def test_load_missing_model_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py classify"):
        C.load_model("linear_svm", tmp_path)


def test_unknown_model_name():
    with pytest.raises(ValueError):
        C.make_classifier("random_forest")


def test_metrics_count_invalid_predictions_as_errors():
    m = classification_metrics(["a", "b", "a", "b"], ["a", "b", "INVALID", "b"], labels=["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["macro_recall"] == pytest.approx(0.75)


# --- statistical significance -------------------------------------------------
def test_mcnemar_known_values():
    import numpy as np

    from src.evaluation import mcnemar_test

    y = np.array(["a"] * 200)
    a, b = y.copy(), y.copy()
    a[:30] = "x"      # A wrong on 0-29
    b[25:70] = "x"    # B wrong on 25-69
    r = mcnemar_test(y, a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (40, 25)
    assert r["p_value"] == pytest.approx(0.0824779, rel=1e-5)      # value checked against SciPy
    a, b = y.copy(), y.copy()
    a[:3] = "x"
    b[3:12] = "x"
    r = mcnemar_test(y, a, b)
    assert r["method"] == "exact binomial" and r["p_value"] == pytest.approx(0.1459961, rel=1e-5)


def test_mcnemar_identical_models():
    from src.evaluation import mcnemar_test

    assert mcnemar_test(["a", "b"], ["a", "a"], ["a", "a"])["p_value"] == 1.0


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    from src.evaluation import bootstrap_ci

    y = ["a", "b", "c"] * 50
    p = ["a", "b", "c"] * 40 + ["a", "a", "a"] * 10
    est, lo, hi = bootstrap_ci(y, p, ["a", "b", "c"], "macro_f1", n_boot=300)
    assert lo <= est <= hi and 0 <= lo and hi <= 1
    assert bootstrap_ci(y, p, ["a", "b", "c"], "macro_f1", n_boot=300) == (est, lo, hi)
    perfect = bootstrap_ci(y, y, ["a", "b", "c"], "accuracy", n_boot=50)
    assert perfect == (1.0, 1.0, 1.0)


def test_bootstrap_fast_path_matches_sklearn():
    import random

    import numpy as np

    from src.evaluation import _metrics_from_codes, classification_metrics

    rnd = random.Random(0)
    labels = ["a", "b", "c", "d"]
    t = [rnd.choice(labels) for _ in range(500)]
    p = [x if rnd.random() < 0.7 else rnd.choice(labels + ["INVALID"]) for x in t]
    code = {l: i for i, l in enumerate(labels)}
    acc, f1 = _metrics_from_codes(np.array([code[x] for x in t]),
                                  np.array([code.get(x, 4) for x in p]), 4)
    ref = classification_metrics(t, p, labels)
    assert acc == pytest.approx(ref["accuracy"]) and f1 == pytest.approx(ref["macro_f1"])
