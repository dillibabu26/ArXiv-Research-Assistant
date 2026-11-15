"""
Experiment runner. Each stage reads the artefacts of the previous stage and writes
its own results, so stages can be re-run independently.

    python train.py download   # optional: fetch the public GitHub archive (if no Kaggle file)
    python train.py prepare    # load, clean, EDA, label selection, sampling, splits
    python train.py classify   # TF-IDF + NB / LR / SVM: validation grid, test evaluation
    python train.py titles     # FLAN-T5 + extractive baseline titles, ROUGE (+ similarity)
    python train.py llm        # LLM classification + 3 prompt styles (needs LLM in .env)
    python train.py index      # retrieval index for the explorer / chatbot
    python train.py compare    # merge results into comparison tables and charts
    python train.py stats      # 95% bootstrap confidence intervals + McNemar significance tests
    python train.py deep       # deep learning: MiniLM+MLP and fine-tuned DistilBERT (not in `all`)
    python train.py all        # prepare -> classify -> titles -> llm -> index -> compare -> stats

Options: --quick (small sample, for a fast smoke test), --limit N (titles/llm subset size).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

import config
from src.utils import get_library_versions, get_logger, load_json, log_duration, save_json, set_seed

logger = get_logger()


def _update_run_config(stage: str, info: dict) -> None:
    path = config.RUN_CONFIG_FILE
    data = load_json(path) if path.exists() else {}
    data.setdefault("settings", config.public_settings())
    data["library_versions"] = get_library_versions()
    data.setdefault("stages", {})[stage] = {"finished": datetime.now().isoformat(timespec="seconds"), **info}
    save_json(data, path)


def _require(path, stage: str) -> None:
    if not path.exists():
        sys.exit(f"Missing {path.name}. Run `python train.py {stage}` first.")


def _merge_rows(path, new: pd.DataFrame, key: str = "key") -> pd.DataFrame:
    """Write `new` rows into a results CSV, replacing rows with the same key and keeping others
    (so re-running `classify` does not delete the deep-learning results, and vice versa)."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    order = {k: i for i, k in enumerate(["naive_bayes", "logistic_regression", "linear_svm", "minilm_mlp", "distilbert"])}
    new = new.sort_values(key, key=lambda s: s.map(order).fillna(99)).reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


# ---------------------------------------------------------------------------
def stage_download(args) -> None:
    from src.data_loader import download_archive

    path = download_archive()
    logger.info("Saved %s", path)


def stage_prepare(args) -> None:
    from src.data_loader import inspect_schema, load_raw
    from src.exploratory_analysis import run_eda
    from src.preprocessing import (assert_no_leakage, build_sample, class_distribution,
                                   clean_dataframe, make_eval_subset, select_top_classes,
                                   split_dataset)

    max_per_class = 500 if args.quick else config.MAX_PER_CLASS
    with log_duration("load raw data"):
        raw, report = load_raw()
    print(f"\nShape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print("\nFirst five records:")
    print(raw.head(5)[["id", "title", "categories"]].to_string())
    schema = inspect_schema(raw)
    print("\nMissing values (%):", schema["missing_pct"])
    print("Duplicate IDs:", schema["duplicate_ids"], "| duplicate abstracts:", schema["duplicate_abstracts"])

    with log_duration("clean records"):
        clean, steps = clean_dataframe(raw)
    print("\nCleaning steps:\n", steps.to_string(index=False))
    steps.to_csv(config.RESULTS_DIR / "cleaning_steps.csv", index=False)

    labels = select_top_classes(clean)
    with log_duration("build sample"):
        sample = build_sample(clean, labels, max_per_class=max_per_class)
    train, val, test = split_dataset(sample)
    assert_no_leakage(train, val, test)
    eval_subset = make_eval_subset(test)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["id", "title", "abstract", "categories", "primary_category", "label",
            "text_tfidf", "authors", "doi", "update_date", "abstract_words", "title_words"]
    sample[keep].to_parquet(config.SAMPLE_FILE, index=False)
    train[keep].to_parquet(config.TRAIN_FILE, index=False)
    val[keep].to_parquet(config.VAL_FILE, index=False)
    test[keep].to_parquet(config.TEST_FILE, index=False)
    eval_subset[keep].to_parquet(config.EVAL_SUBSET_FILE, index=False)
    save_json(labels, config.LABELS_FILE)

    dist = pd.concat({"train": class_distribution(train)["count"],
                      "val": class_distribution(val)["count"],
                      "test": class_distribution(test)["count"],
                      "eval_subset": class_distribution(eval_subset)["count"]}, axis=1)
    dist.to_csv(config.RESULTS_DIR / "split_distribution.csv")
    print("\nSplit sizes per class:\n", dist.to_string())

    with log_duration("EDA figures"):
        eda = run_eda(raw, clean, sample, labels, schema, report.source)
    summary = {"load_report": report.as_dict(), "schema": schema,
               "clean_records": int(len(clean)), "labels": labels,
               "sample_size": int(len(sample)),
               "split_sizes": {"train": len(train), "val": len(val), "test": len(test),
                               "eval_subset": len(eval_subset)},
               "eda": eda}
    save_json(summary, config.RESULTS_DIR / "eda_summary.json")
    _update_run_config("prepare", {"source": report.source, "quick": args.quick,
                                   "max_per_class": max_per_class, **summary["split_sizes"]})
    logger.info("Figures in %s; processed data in %s", config.FIGURES_DIR, config.PROCESSED_DATA_DIR)


def stage_classify(args) -> None:
    from src import classification as C
    from src.evaluation import (classification_metrics, classification_report_df,
                                plot_confusion_matrix, plot_metric_comparison, time_inference)

    for p in (config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE):
        _require(p, "prepare")
    train, val, test = (pd.read_parquet(p) for p in (config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE))
    eval_subset = pd.read_parquet(config.EVAL_SUBSET_FILE)
    labels = load_json(config.LABELS_FILE)

    grid = config.TFIDF_GRID
    if args.quick:
        grid = {"ngram_range": [(1, 2)], "max_features": [20_000], "min_df": [2], "max_df": [0.9]}
    with log_duration("validation search"):
        search = C.tune_on_validation(train["text_tfidf"], train["label"],
                                      val["text_tfidf"], val["label"], tfidf_grid=grid)
    search.drop(columns=["_tfidf", "_clf"]).to_csv(config.RESULTS_DIR / "classification_validation_search.csv", index=False)
    best = C.best_configs(search)

    # Refit the chosen configuration on train + validation; the test set is used once.
    train_val = pd.concat([train, val], ignore_index=True)
    rows, sub_rows, reports = [], [], []
    for m, cfg in best.items():
        with log_duration(f"final fit {m}"):
            pipe = C.train_final(m, train_val["abstract"], train_val["label"], cfg["tfidf"], cfg["clf"])
        C.save_model(pipe, m)
        name = C.MODEL_DISPLAY_NAMES[m]
        y_pred = C.predict(pipe, test["abstract"])
        timing = time_inference(lambda xs: pipe.predict(xs), test["abstract"].tolist())
        metrics = classification_metrics(test["label"], y_pred, labels)
        rows.append({"model": name, "key": m, "split": "test", **metrics, **timing,
                     "tfidf_params": str(cfg["tfidf"]), "clf_params": str(cfg["clf"]),
                     "val_macro_f1": round(cfg["val_macro_f1"], 4)})
        sub_pred = C.predict(pipe, eval_subset["abstract"])
        sub_rows.append({"model": name, "key": m, "split": "eval_subset",
                         **classification_metrics(eval_subset["label"], sub_pred, labels), **timing})
        rep = classification_report_df(test["label"], y_pred, labels); rep["model"] = name
        reports.append(rep)
        plot_confusion_matrix(test["label"], y_pred, labels,
                              f"{name} — normalised confusion matrix (test, n={len(test):,})",
                              config.FIGURES_DIR / f"confusion_matrix_{m}.png")
        pd.DataFrame({"id": test["id"], "true": test["label"], "pred": y_pred}).to_csv(
            config.RESULTS_DIR / f"predictions_test_{m}.csv", index=False)
        C.top_features(pipe).to_csv(config.RESULTS_DIR / f"top_features_{m}.csv", index=False)
        logger.info("%s: test macro-F1 %.4f, accuracy %.4f", name, metrics["macro_f1"], metrics["accuracy"])

    results = _merge_rows(config.RESULTS_DIR / "classification_metrics_test.csv", pd.DataFrame(rows))
    pd.DataFrame(sub_rows).to_csv(config.RESULTS_DIR / "classification_metrics_eval_subset.csv", index=False)
    rep_path, rep_new = config.RESULTS_DIR / "classification_reports.csv", pd.concat(reports)
    if rep_path.exists():   # keep deep-learning reports written by `train.py deep`
        old = pd.read_csv(rep_path, index_col=0)
        rep_new = pd.concat([old[~old["model"].isin(rep_new["model"])], rep_new])
    rep_new.to_csv(rep_path)
    plot_metric_comparison(results, ["accuracy", "macro_f1", "weighted_f1"],
                           f"Traditional ML on the test set (n={len(test):,})",
                           config.FIGURES_DIR / "classification_comparison.png")
    print("\n", results[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                         "weighted_f1", "batch_ms_per_doc", "single_ms_median"]].round(4).to_string(index=False))
    _update_run_config("classify", {"best_configs": {m: {k: str(v) for k, v in c.items()} for m, c in best.items()},
                                    "quick": args.quick})


def _score_titles(df: pd.DataFrame) -> pd.DataFrame:
    from src.evaluation import rouge_scores, semantic_similarity, unsupported_term_rate

    df = df.reset_index(drop=True)
    r = rouge_scores(df["original"], df["generated"])
    sim, sim_method = semantic_similarity(df["original"], df["generated"])
    df = pd.concat([df, r], axis=1)
    df["similarity"] = sim
    df["similarity_method"] = sim_method
    df["unsupported_rate"] = [unsupported_term_rate(a, t) for a, t in zip(df["abstract"], df["generated"])]
    df["words"] = df["generated"].fillna("").str.split().str.len()
    return df


def stage_titles(args) -> None:
    import time

    from src.evaluation import summarise_title_scores
    from src.title_generation import BASELINES, generate_titles, load_title_model, transformers_available

    _require(config.EVAL_SUBSET_FILE, "prepare")
    subset = pd.read_parquet(config.EVAL_SUBSET_FILE)
    if args.limit:
        subset = subset.head(args.limit)
    rows, status = [], {}
    for name, fn in BASELINES.items():
        t0 = time.perf_counter(); gen = [fn(a) for a in subset["abstract"]]
        secs = (time.perf_counter() - t0) / len(subset)
        rows.append(pd.DataFrame({"id": subset["id"], "label": subset["label"], "abstract": subset["abstract"],
                                  "original": subset["title"], "generated": gen, "method": name,
                                  "seconds_per_title": secs}))
        status[name] = "run"

    ok, reason = transformers_available()
    if ok:
        try:
            with log_duration(f"FLAN-T5 titles for {len(subset)} abstracts"):
                model, tok, device = load_title_model()
                t0 = time.perf_counter()
                gen = generate_titles(subset["abstract"].tolist(), model, tok, device)
                secs = (time.perf_counter() - t0) / len(subset)
            rows.append(pd.DataFrame({"id": subset["id"], "label": subset["label"], "abstract": subset["abstract"],
                                      "original": subset["title"], "generated": gen, "method": "flan_t5",
                                      "seconds_per_title": secs}))
            status["flan_t5"] = f"run ({config.TITLE_MODEL_NAME} on {device})"
        except Exception as e:
            status["flan_t5"] = f"not run: {type(e).__name__}: {str(e)[:200]}"
    else:
        status["flan_t5"] = f"not run: transformers/torch unavailable ({reason[:150]})"
    logger.info("FLAN-T5 status: %s", status["flan_t5"])

    scored = _score_titles(pd.concat(rows, ignore_index=True))
    # Keep any LLM rows from a previous `llm` stage run.
    out_path = config.RESULTS_DIR / "titles_generated.csv"
    if out_path.exists():
        prev = pd.read_csv(out_path)
        scored = pd.concat([prev[prev["method"].str.startswith("llm_")], scored], ignore_index=True)
    scored.to_csv(out_path, index=False)
    summary = summarise_title_scores(scored)
    summary.to_csv(config.RESULTS_DIR / "title_metrics.csv")
    print("\n", summary.to_string())
    _update_run_config("titles", {"status": status, "n_abstracts": len(subset),
                                  "similarity_method": scored["similarity_method"].iloc[0]})


def stage_llm(args) -> None:
    import time

    from src.evaluation import classification_metrics, summarise_title_scores
    from src.llm_classification import classify_with_llm
    from src.llm_client import LLMClient
    from src.llm_title_generation import PROMPT_STYLES, generate_title_llm, select_few_shot_examples

    client = LLMClient()
    status_path = config.RESULTS_DIR / "llm_status.json"
    if not client.available:
        save_json({"status": "not run", "reason": client.describe()}, status_path)
        logger.warning("LLM stage skipped: %s. Edit .env to enable it.", client.describe())
        _update_run_config("llm", {"status": "not run", "reason": client.describe()})
        return
    ok, msg = client.check_connection()
    if not ok:
        save_json({"status": "not run", "reason": f"connection failed: {msg}"}, status_path)
        logger.error("Cannot reach the LLM (%s). Is Ollama running / is the key valid?", msg)
        return

    labels = load_json(config.LABELS_FILE)
    subset = pd.read_parquet(config.EVAL_SUBSET_FILE)
    if args.limit:
        subset = subset.head(args.limit)
    from tqdm import tqdm

    preds = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="LLM classification"):
        res = classify_with_llm(row["abstract"], client, labels)
        preds.append({"id": row["id"], "true": row["label"], "pred": res["category"] or "INVALID",
                      "valid": res["valid"], "parse_method": res["parse_method"],
                      "confidence": res["confidence"], "explanation": res["explanation"],
                      "latency_s": res["latency_s"], "error": res["error"]})
    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(config.RESULTS_DIR / "llm_classification_predictions.csv", index=False)
    metrics = classification_metrics(pred_df["true"], pred_df["pred"], labels)
    metrics.update(model=f"LLM zero-shot ({client.model})", key="llm", split="eval_subset",
                   invalid_rate=float((~pred_df["valid"]).mean()),
                   single_ms_median=float(pred_df["latency_s"].dropna().median() * 1000)
                   if pred_df["latency_s"].notna().any() else None)
    pd.DataFrame([metrics]).to_csv(config.RESULTS_DIR / "llm_classification_metrics.csv", index=False)
    logger.info("LLM classification macro-F1 %.4f (invalid %.1f%%)", metrics["macro_f1"], 100 * metrics["invalid_rate"])

    # Title generation: three prompt styles on the same abstracts.
    n_titles = min(len(subset), args.limit or 60)
    tsub = subset.head(n_titles)
    examples = select_few_shot_examples(pd.read_parquet(config.TRAIN_FILE))
    rows = []
    for style in PROMPT_STYLES:
        for _, row in tqdm(tsub.iterrows(), total=len(tsub), desc=f"LLM titles ({style})"):
            t0 = time.perf_counter()
            gen = generate_title_llm(row["abstract"], client, style, examples)
            rows.append({"id": row["id"], "label": row["label"], "abstract": row["abstract"],
                         "original": row["title"], "generated": gen or "", "method": f"llm_{style}",
                         "seconds_per_title": time.perf_counter() - t0})
    scored = _score_titles(pd.DataFrame(rows))
    out_path = config.RESULTS_DIR / "titles_generated.csv"
    if out_path.exists():
        prev = pd.read_csv(out_path)
        scored = pd.concat([prev[~prev["method"].str.startswith("llm_")], scored], ignore_index=True)
    scored.to_csv(out_path, index=False)
    summarise_title_scores(scored).to_csv(config.RESULTS_DIR / "title_metrics.csv")
    save_json({"status": "run", "model": client.model, "base_url": client.base_url,
               "n_classification": len(subset), "n_titles": n_titles,
               "few_shot_example_ids": [e["id"] for e in examples]}, status_path)
    _update_run_config("llm", {"status": "run", "model": client.model})


def stage_index(args) -> None:
    from src.rag import build_index, calibrate_threshold, evaluate_retrieval

    _require(config.TRAIN_FILE, "prepare")
    # The corpus is train + validation only: test papers stay unseen by every component.
    corpus = pd.concat([pd.read_parquet(config.TRAIN_FILE), pd.read_parquet(config.VAL_FILE)],
                       ignore_index=True)
    test = pd.read_parquet(config.TEST_FILE)
    with log_duration(f"index {len(corpus):,} papers"):
        retriever = build_index(corpus, backend=args.backend)
    known = corpus.sample(n=min(300, len(corpus)), random_state=config.SEED)
    metrics = evaluate_retrieval(retriever, known)
    calib = calibrate_threshold(retriever, test.sample(n=min(200, len(test)), random_state=config.SEED)["title"].tolist())
    result = {"backend": retriever.backend, **metrics, "evidence_threshold": calib}
    save_json(result, config.RESULTS_DIR / "retrieval_metrics.json")
    print(result)
    _update_run_config("index", {"backend": retriever.backend, "n_documents": len(corpus)})


def stage_compare(args) -> None:
    from src.classification import MODEL_DISPLAY_NAMES
    from src.evaluation import classification_metrics, plot_metric_comparison

    R = config.RESULTS_DIR
    labels = load_json(config.LABELS_FILE)
    subset_ids = pd.read_parquet(config.EVAL_SUBSET_FILE)["id"].astype(str)
    llm_path = R / "llm_classification_predictions.csv"
    llm_pred = pd.read_csv(llm_path, dtype={"id": str}) if llm_path.exists() else None
    # Every model is scored on exactly the same papers: the LLM's papers if it was run
    # (possibly a --limit prefix of the evaluation subset), otherwise the full subset.
    ids = set(llm_pred["id"]) if llm_pred is not None else set(subset_ids)
    timing = pd.read_csv(R / "classification_metrics_test.csv") if (R / "classification_metrics_test.csv").exists() else None
    rows = []
    for key, name in MODEL_DISPLAY_NAMES.items():
        p = R / f"predictions_test_{key}.csv"
        if not p.exists():
            continue
        pr = pd.read_csv(p, dtype={"id": str})
        pr = pr[pr["id"].isin(ids)]
        trow = timing[timing["key"] == key] if timing is not None else None
        # A model can have predictions without a metrics row if its stage was interrupted.
        t = trow["single_ms_median"].iloc[0] if trow is not None and len(trow) else None
        rows.append({"model": name, **classification_metrics(pr["true"], pr["pred"], labels),
                     "single_ms_median": t, "invalid_rate": 0.0})
    if llm_pred is not None:
        llm_m = pd.read_csv(R / "llm_classification_metrics.csv")
        rows.append({"model": llm_m["model"].iloc[0],
                     **classification_metrics(llm_pred["true"], llm_pred["pred"], labels),
                     "single_ms_median": llm_m["single_ms_median"].iloc[0],
                     "invalid_rate": float((~llm_pred["valid"].astype(bool)).mean())})
    if not rows:
        sys.exit("No classification results yet. Run `python train.py classify`.")
    comp = pd.DataFrame(rows)
    cols = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "single_ms_median", "invalid_rate", "n"]
    comp = comp[cols]
    comp.to_csv(R / "comparison_classification.csv", index=False)
    plot_metric_comparison(comp, ["accuracy", "macro_f1"],
                           f"Classification on the shared evaluation subset (n={int(comp['n'].iloc[0])})",
                           config.FIGURES_DIR / "comparison_classification.png")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.5 * len(comp)))
    ax.barh(comp["model"], comp["single_ms_median"], color="#2a78d6", height=0.6)
    vals = comp["single_ms_median"].astype(float)
    log = vals.max() / max(vals.min(), 1e-9) > 20   # log scale only when an LLM makes the range huge
    if log:
        ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("median latency per abstract (ms" + (", log scale)" if log else ")"))
    ax.set_title("Inference time per abstract", fontweight="bold", fontsize=10)
    for i, v in enumerate(comp["single_ms_median"]):
        ax.text(v, i, f" {v:.2f} ms", va="center", fontsize=8, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "comparison_inference_time.png", dpi=150); plt.close(fig)

    if (R / "title_metrics.csv").exists():
        from src.evaluation import method_label

        tm = pd.read_csv(R / "title_metrics.csv").rename(columns={"method": "model"})
        tm = tm.sort_values("rougeL", ascending=False)            # best method first
        tm["model"] = tm["model"].map(method_label)
        plot_metric_comparison(tm, ["rouge1", "rouge2", "rougeL"],
                               f"Title generation: ROUGE F1 against original titles (n={int(tm['n'].iloc[0])} test papers)",
                               config.FIGURES_DIR / "comparison_titles.png")
    print(comp.round(4).to_string(index=False))


def stage_stats(args) -> None:
    """95% bootstrap confidence intervals and pairwise McNemar tests on the test set."""
    import itertools

    from src.classification import MODEL_DISPLAY_NAMES
    from src.evaluation import bootstrap_ci, mcnemar_test

    R = config.RESULTS_DIR
    labels = load_json(config.LABELS_FILE)
    preds = {}
    for key, name in MODEL_DISPLAY_NAMES.items():
        p = R / f"predictions_test_{key}.csv"
        if p.exists():
            preds[name] = pd.read_csv(p, dtype=str)
    if not preds:
        sys.exit("No predictions yet. Run `python train.py classify`.")
    n_boot = 200 if args.quick else 1000
    rows = []
    for name, df in preds.items():
        for metric in ("accuracy", "macro_f1"):
            point, lo, hi = bootstrap_ci(df["true"], df["pred"], labels, metric, n_boot=n_boot)
            rows.append({"model": name, "split": "test", "metric": metric, "estimate": point,
                         "ci_low": lo, "ci_high": hi, "n": len(df), "n_boot": n_boot})
    # The LLM (if run) is evaluated on the shared subset: compare it with every model there.
    llm_path = R / "llm_classification_predictions.csv"
    subset_pairs = {}
    if llm_path.exists():
        llm = pd.read_csv(llm_path, dtype=str)
        ids = set(llm["id"])
        name = "LLM (zero-shot)"
        for metric in ("accuracy", "macro_f1"):
            point, lo, hi = bootstrap_ci(llm["true"], llm["pred"], labels, metric, n_boot=n_boot)
            rows.append({"model": name, "split": "eval_subset", "metric": metric, "estimate": point,
                         "ci_low": lo, "ci_high": hi, "n": len(llm), "n_boot": n_boot})
        for m, df in preds.items():
            sub = df[df["id"].isin(ids)].set_index("id").loc[llm["id"]]
            subset_pairs[(m, name)] = (llm["true"].values, sub["pred"].values, llm["pred"].values)
    ci = pd.DataFrame(rows)
    ci.to_csv(R / "significance_ci.csv", index=False)

    tests = []
    for (a, da), (b, db) in itertools.combinations(preds.items(), 2):
        # Pair the two models paper by paper on the papers both have predictions for
        # (normally the whole test set; fewer only after a --quick run).
        common = da.merge(db[["id", "pred"]], on="id", suffixes=("_a", "_b"))
        tests.append({"model_a": a, "model_b": b, "split": "test", "n": len(common),
                      **mcnemar_test(common["true"], common["pred_a"], common["pred_b"])})
    for (a, b), (y, pa, pb) in subset_pairs.items():
        tests.append({"model_a": a, "model_b": b, "split": "eval_subset", **mcnemar_test(y, pa, pb)})
    mc = pd.DataFrame(tests)
    mc["significant_at_0.05"] = mc["p_value"] < 0.05
    # Several tests are run at once, so also apply a Bonferroni correction (alpha / number of tests).
    mc["bonferroni_alpha"] = 0.05 / max(1, len(mc))
    mc["significant_bonferroni"] = mc["p_value"] < mc["bonferroni_alpha"]
    mc.to_csv(R / "significance_mcnemar.csv", index=False)
    print("\n95% bootstrap confidence intervals:\n", ci.round(4).to_string(index=False))
    print("\nMcNemar tests:\n", mc.round(4).to_string(index=False))
    _update_run_config("stats", {"n_boot": n_boot})


def stage_deep(args) -> None:
    """Deep-learning classifiers: MiniLM embeddings + MLP, and fine-tuned DistilBERT."""
    from src import deep_learning as D
    from src.evaluation import (classification_metrics, classification_report_df,
                                plot_confusion_matrix, plot_metric_comparison, time_inference)

    for p in (config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE):
        _require(p, "prepare")
    train, val, test = (pd.read_parquet(p) for p in (config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE))
    labels = load_json(config.LABELS_FILE)
    if args.quick:
        train, val, test = (D.balanced_subset(d, n) for d, n in ((train, 100), (val, 20), (test, 30)))
    R, rows, reports, info = config.RESULTS_DIR, [], [], {}

    def record(key, preds, timing, extra):
        name = C_NAMES[key]
        metrics = classification_metrics(test["label"], preds, labels)
        rows.append({"model": name, "key": key, "split": "test", **metrics, **timing, **extra})
        pd.DataFrame({"id": test["id"], "true": test["label"], "pred": preds}).to_csv(
            R / f"predictions_test_{key}.csv", index=False)
        rep = classification_report_df(test["label"], preds, labels); rep["model"] = name
        reports.append(rep)
        plot_confusion_matrix(test["label"], preds, labels,
                              f"{name} — normalised confusion matrix (test, n={len(test):,})",
                              config.FIGURES_DIR / f"confusion_matrix_{key}.png")
        logger.info("%s: test macro-F1 %.4f, accuracy %.4f", name, metrics["macro_f1"], metrics["accuracy"])

    from src.classification import MODEL_DISPLAY_NAMES as C_NAMES

    if not args.skip_mlp:
        with log_duration("MiniLM embeddings + MLP"):
            Xtr, Xva, Xte = D.get_split_embeddings(train, val, test)
            mlp, search = D.train_mlp(Xtr, train["label"], Xva, val["label"], labels)
            search.to_csv(R / "deep_mlp_validation_search.csv", index=False)
            preds = list(mlp.predict(Xte))
            # Inference time includes encoding with MiniLM, i.e. the full cost for a new abstract.
            from src.rag import get_encoder

            enc = get_encoder()
            timing = time_inference(lambda xs: mlp.predict(D.encode_abstracts(xs, enc)),
                                    test["abstract"].tolist()[:500], single_calls=50)
            best = search.sort_values("val_macro_f1", ascending=False).iloc[0]
            record("minilm_mlp", preds, timing, {"tfidf_params": "-", "clf_params": best["params"],
                                                 "val_macro_f1": round(float(best["val_macro_f1"]), 4)})
            import joblib

            D.DL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(mlp, D.DL_MODEL_DIR / "minilm_mlp.joblib")
            info["minilm_mlp"] = {"params": best["params"]}

    if not args.skip_bert:
        with log_duration("fine-tune DistilBERT"):
            kw = dict(per_class=60, val_size=100, epochs=1) if args.quick else {}
            model, tok, history = D.finetune_transformer(train, val, labels, **kw)
            save_json(history, R / "deep_distilbert_history.json")
            t0 = time.perf_counter()
            preds = D.predict_transformer(model, tok, test["abstract"].tolist(), labels)
            batch_ms = (time.perf_counter() - t0) / len(test) * 1000
            single = time_inference(lambda xs: D.predict_transformer(model, tok, xs, labels, batch_size=1),
                                    test["abstract"].tolist()[:50], single_calls=50)
            best_epoch = max(history, key=lambda h: h["val_macro_f1"])
            record("distilbert", preds, {"batch_ms_per_doc": round(batch_ms, 3),
                                         "single_ms_median": single["single_ms_median"]},
                   {"tfidf_params": "-", "clf_params": f"lr={config.DL_LR}, epochs={config.DL_EPOCHS}, "
                    f"batch={config.DL_BATCH_SIZE}, max_len={config.DL_MAX_LEN}, train={config.DL_PER_CLASS}/class",
                    "val_macro_f1": round(best_epoch["val_macro_f1"], 4)})
            D.save_transformer(model, tok)
            info["distilbert"] = {"history": history, "train_per_class": config.DL_PER_CLASS}

    if not rows:
        sys.exit("Nothing to run (both --skip-mlp and --skip-bert given).")
    new = pd.DataFrame(rows)
    allm = _merge_rows(R / "classification_metrics_test.csv", new)
    rep_path = R / "classification_reports.csv"
    rep_new = pd.concat(reports)
    if rep_path.exists():
        old = pd.read_csv(rep_path, index_col=0)
        rep_new = pd.concat([old[~old["model"].isin(new["model"])], rep_new])
    rep_new.to_csv(rep_path)
    plot_metric_comparison(allm, ["accuracy", "macro_f1", "weighted_f1"],
                           f"All classifiers on the test set (n={len(test):,})",
                           config.FIGURES_DIR / "classification_comparison.png")
    print("\n", allm[["model", "accuracy", "macro_f1", "weighted_f1", "batch_ms_per_doc",
                      "single_ms_median"]].round(4).to_string(index=False))
    _update_run_config("deep", {"quick": args.quick, **info})


# ---------------------------------------------------------------------------
STAGES = {"download": stage_download, "prepare": stage_prepare, "classify": stage_classify,
          "titles": stage_titles, "llm": stage_llm, "index": stage_index, "compare": stage_compare,
          "stats": stage_stats, "deep": stage_deep}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="DG4NLP experiment runner")
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    parser.add_argument("--quick", action="store_true", help="small sample for a smoke test")
    parser.add_argument("--limit", type=int, default=None, help="subset size for titles/llm stages")
    parser.add_argument("--skip-mlp", action="store_true", help="deep stage: skip the MiniLM + MLP model")
    parser.add_argument("--skip-bert", action="store_true", help="deep stage: skip DistilBERT fine-tuning")
    parser.add_argument("--backend", choices=["auto", "dense", "tfidf"], default="auto",
                        help="retrieval index type for the index stage")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    set_seed(config.SEED)
    order = ["prepare", "classify", "titles", "llm", "index", "compare", "stats"]
    for stage in (order if args.stage == "all" else [args.stage]):
        if stage in STAGES:
            with log_duration(f"stage '{stage}'"):
                STAGES[stage](args)


if __name__ == "__main__":
    main()
