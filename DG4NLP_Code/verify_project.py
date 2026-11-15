"""
Full project verification - one command, works from any folder:

    python verify_project.py            # everything (about 1-2 minutes)
    python verify_project.py --fast     # skip the pytest run and the app check

Prints PASS / WARN / FAIL for each check, plus the command that fixes it.
  PASS = correct   WARN = optional or not yet run   FAIL = must be fixed
Exit code is 0 only when there are no FAILs.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                      # run as if started from the project folder
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str, str]] = []   # (section, status, check, detail)


def record(section: str, status: str, check: str, detail: str = "") -> None:
    RESULTS.append((section, status, check, detail))
    colour = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}.get(status, "")
    print(f"  {colour}[{status}]\033[0m {check}" + (f"  ->  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
def check_environment() -> bool:
    section("1. Environment")
    v = sys.version_info
    record("env", "PASS" if v >= (3, 10) else "FAIL", f"Python {v.major}.{v.minor}.{v.micro}",
           "" if v >= (3, 10) else "need 3.10+; create the venv with `py -3.11 -m venv .venv`")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    record("env", "PASS" if in_venv else "WARN", "virtual environment active",
           "" if in_venv else r"run .\.venv\Scripts\Activate.ps1")
    try:
        from src.utils import CORE_PACKAGES, get_library_versions
    except Exception as e:
        record("env", "FAIL", "import src.utils", f"{type(e).__name__}: {e}")
        return False
    versions = get_library_versions()
    missing = [n for n, _ in CORE_PACKAGES if versions.get(n) == "NOT INSTALLED"]
    record("env", "FAIL" if missing else "PASS", f"{len(CORE_PACKAGES)} core packages installed",
           f"missing {missing}: run `pip install -r requirements.txt`" if missing else "")
    record("env", "PASS" if versions.get("faiss-cpu") != "NOT INSTALLED" else "WARN",
           "faiss-cpu (optional)", "" if versions.get("faiss-cpu") != "NOT INSTALLED"
           else "optional: pip install -r requirements-optional.txt")
    record("env", "PASS" if (ROOT / ".env").exists() else "WARN", ".env file present",
           "" if (ROOT / ".env").exists() else "copy .env.example .env")
    basic = {"pandas", "numpy", "scikit-learn", "pyarrow", "matplotlib", "streamlit", "python-dotenv"}
    return not (basic & set(missing))   # continue unless the essentials are missing


def check_code() -> None:
    section("2. Code: every module imports")
    try:  # silence Streamlit's "no runtime" notices when importing pages outside `streamlit run`
        import logging

        import streamlit.logger

        streamlit.logger.set_log_level("error")
        logging.getLogger("streamlit").setLevel(logging.ERROR)
    except Exception:
        pass
    modules = ["config", "src.utils", "src.data_loader", "src.preprocessing", "src.exploratory_analysis",
               "src.evaluation", "src.classification", "src.title_generation", "src.llm_client",
               "src.llm_classification", "src.llm_title_generation", "src.rag", "train",
               "ui.components", "ui.dashboard", "ui.classification_page", "ui.generation_page",
               "ui.comparison_page", "ui.explorer_page", "ui.chatbot_page"]
    bad = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(f"{m} ({type(e).__name__}: {str(e)[:80]})")
    record("code", "FAIL" if bad else "PASS", f"{len(modules)} modules import", "; ".join(bad))
    import config

    key_in_settings = "LLM_API_KEY" in str(config.public_settings())
    record("code", "FAIL" if key_in_settings else "PASS", "API key never written to results")
    gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
    record("code", "PASS" if ".env" in gi else "FAIL", ".env excluded by .gitignore")


def check_tests() -> None:
    section("3. Unit tests (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    summary = [l for l in proc.stdout.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    tail = summary[-1].strip("= ") if summary else "(no summary - see `python -m pytest`)"
    passed = re.search(r"(\d+) passed", tail)
    failed = re.search(r"(\d+) (failed|error)", tail)
    status = "PASS" if proc.returncode == 0 and passed else "FAIL"
    record("tests", status, f"pytest: {tail}",
           "" if status == "PASS" else "run `python -m pytest -x -vv` to see the first failure")
    if failed:
        for line in proc.stdout.splitlines():
            if line.startswith(("FAILED", "ERROR")):
                print("        " + line[:150])


def check_data() -> None:
    section("4. Dataset and processed data")
    import config
    from src.data_loader import discover_raw_files

    raw = discover_raw_files()
    record("data", "PASS" if raw else "FAIL", "raw dataset in data/raw",
           ", ".join(p.name for p in raw) if raw else "run `python train.py download` (or add the Kaggle JSON)")
    files = [config.SAMPLE_FILE, config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE,
             config.EVAL_SUBSET_FILE, config.LABELS_FILE]
    missing = [f.name for f in files if not f.exists()]
    record("data", "FAIL" if missing else "PASS", "processed splits exist",
           f"missing {missing}: run `python train.py prepare`" if missing else "")
    if missing:
        return
    import pandas as pd

    from src.preprocessing import assert_no_leakage
    from src.utils import load_json

    tr, va, te = (pd.read_parquet(f) for f in (config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE))
    sub = pd.read_parquet(config.EVAL_SUBSET_FILE)
    try:
        assert_no_leakage(tr, va, te)
        record("data", "PASS", "no paper or abstract shared between train/val/test",
               f"{len(tr):,} / {len(va):,} / {len(te):,}")
    except AssertionError as e:
        record("data", "FAIL", "data leakage between splits", str(e))
    record("data", "PASS" if set(sub["id"]) <= set(te["id"]) else "FAIL",
           "evaluation subset is drawn from the test set", f"{len(sub)} papers")
    labels = load_json(config.LABELS_FILE)
    props = te["label"].value_counts(normalize=True)
    ok = set(props.index) == set(labels)
    record("data", "PASS" if ok else "FAIL", f"all {len(labels)} classes present in test set")
    figs = [f"0{i}_" for i in range(1, 8)]
    have = [p.name for p in config.FIGURES_DIR.glob("0*.png")]
    nfig = sum(any(h.startswith(f) for h in have) for f in figs)
    record("data", "PASS" if nfig == 7 else "FAIL", f"EDA figures: {nfig}/7",
           "" if nfig == 7 else "run `python train.py prepare`")


def check_experiments() -> None:
    section("5. Experiments and result integrity")
    import config
    import pandas as pd

    from src.classification import MODEL_DISPLAY_NAMES, available_models
    from src.evaluation import classification_metrics
    from src.utils import load_json

    R = config.RESULTS_DIR
    models = available_models()
    record("exp", "PASS" if len(models) == 3 else "FAIL", f"trained classifiers: {len(models)}/3",
           "" if len(models) == 3 else "run `python train.py classify`")
    mpath = R / "classification_metrics_test.csv"
    if mpath.exists() and config.LABELS_FILE.exists():
        metrics = pd.read_csv(mpath)
        labels = load_json(config.LABELS_FILE)
        for _, row in metrics.iterrows():
            p = R / f"predictions_test_{row['key']}.csv"
            if not p.exists():
                record("exp", "FAIL", f"{row['model']}: predictions file missing")
                continue
            pr = pd.read_csv(p)
            recomputed = classification_metrics(pr["true"], pr["pred"], labels)["macro_f1"]
            same = abs(recomputed - row["macro_f1"]) < 1e-9
            record("exp", "PASS" if same else "FAIL",
                   f"{row['model']}: reported macro-F1 {row['macro_f1']:.4f} recomputed from predictions",
                   "" if same else f"recomputed {recomputed:.4f}")
        if models:
            from src.classification import load_model

            m = models[0]
            te = pd.read_parquet(config.TEST_FILE).head(200)
            pr = pd.read_csv(R / f"predictions_test_{m}.csv").head(200)
            live = list(load_model(m).predict(te["abstract"].tolist()))
            match = (pd.Series(live) == pr["pred"].values).mean()
            record("exp", "PASS" if match == 1.0 else "FAIL",
                   f"saved {MODEL_DISPLAY_NAMES[m]} model reproduces stored predictions",
                   f"{match:.1%} identical on 200 test papers" + ("" if match == 1 else
                   " - retrain with `python train.py classify`"))

    rc = load_json(config.RUN_CONFIG_FILE) if config.RUN_CONFIG_FILE.exists() else {}
    dl = [k for k in ("minilm_mlp", "distilbert") if (R / f"predictions_test_{k}.csv").exists()]
    record("exp", "PASS" if len(dl) == 2 else "WARN", f"deep-learning classifiers: {len(dl)}/2",
           "" if len(dl) == 2 else "run `python train.py deep` (about 1 hour on a CPU)")
    titles = R / "title_metrics.csv"
    if titles.exists():
        tm = pd.read_csv(titles)
        methods = set(tm["method"])
        record("exp", "PASS", f"title generation evaluated: {', '.join(sorted(methods))}")
        flan = rc.get("stages", {}).get("titles", {}).get("status", {}).get("flan_t5", "unknown")
        record("exp", "PASS" if "flan_t5" in methods else "WARN", "FLAN-T5 titles",
               "" if "flan_t5" in methods else f"{flan[:100]} - needs torch+transformers and internet, "
                                               "then `python train.py titles`")
        sim = pd.read_csv(R / "titles_generated.csv", usecols=["similarity_method"])["similarity_method"].iloc[0]
        record("exp", "PASS" if "sentence" in sim else "WARN", f"title similarity method: {sim}",
               "" if "sentence" in sim else "install sentence-transformers and rerun `python train.py titles`")
    else:
        record("exp", "FAIL", "title generation results", "run `python train.py titles`")

    llm = load_json(R / "llm_status.json") if (R / "llm_status.json").exists() else {}
    record("exp", "PASS" if llm.get("status") == "run" else "WARN", "LLM experiments (required by the brief)",
           f"run with {llm.get('model')}" if llm.get("status") == "run" else
           f"{llm.get('reason', 'not run')} - configure .env, then `python train.py llm --limit 100`")

    rm = R / "retrieval_metrics.json"
    if rm.exists():
        r = load_json(rm)
        record("exp", "PASS" if r["backend"] == "dense" else "WARN", f"retrieval index: {r['backend']}",
               f"Recall@1 {r['recall@1']:.3f}, MRR@10 {r['mrr@10']:.3f}" + (
                   "" if r["backend"] == "dense" else " - dense search needs sentence-transformers; "
                                                    "rerun `python train.py index`"))
    else:
        record("exp", "FAIL", "retrieval index", "run `python train.py index`")
    sig = R / "significance_mcnemar.csv"
    if sig.exists():
        mc = pd.read_csv(sig)
        record("exp", "PASS", f"significance tests: {len(mc)} McNemar comparisons, "
               f"{int(mc['significant_at_0.05'].sum())} significant at p<0.05")
    else:
        record("exp", "WARN", "significance tests (bootstrap CIs, McNemar)", "run `python train.py stats`")
    comp = R / "comparison_classification.csv"
    record("exp", "PASS" if comp.exists() else "FAIL", "model comparison table",
           "" if comp.exists() else "run `python train.py compare`")
    if rc:
        record("exp", "PASS", "run_config.json records settings + library versions",
               "stages: " + ", ".join(rc.get("stages", {})))


def check_app() -> None:
    section("6. Streamlit application (headless)")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        record("app", "FAIL", "streamlit testing module", "pip install -r requirements.txt")
        return
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if at.exception:
        record("app", "FAIL", "app starts", str(at.exception[0].value)[:150])
        return
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        if at.exception:
            record("app", "FAIL", f"page '{page}'", str(at.exception[0].value)[:150])
        elif at.error:
            record("app", "WARN", f"page '{page}' shows an error message", at.error[0].value[:120])
        elif at.warning:
            record("app", "WARN", f"page '{page}' asks for a missing artefact", at.warning[0].value[:100])
        else:
            record("app", "PASS", f"page '{page}' renders")


def check_docs() -> None:
    section("7. Documentation and report")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    import train

    stages = [s for s in train.STAGES]
    undocumented = [s for s in stages if f"train.py {s}" not in readme]
    record("docs", "PASS" if not undocumented else "FAIL", "README documents every train.py stage",
           f"missing {undocumented}" if undocumented else "")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    todos = report.count("[[")
    record("docs", "WARN" if todos else "PASS", f"REPORT.md placeholders remaining: {todos}",
           "fill with your measured results and your own interpretation" if todos else "")
    tex = ROOT / "report_latex" / "main.tex"
    if tex.exists():
        n = tex.read_text(encoding="utf-8").count("\\todo{")
        record("docs", "WARN" if n else "PASS", f"report_latex/main.tex TODO markers remaining: {n}",
               "search the PDF for red TODO text" if n else "")
    body = report.split("## 1. Introduction")[-1].split("## References")[0]
    words = len(re.sub(r"\|.*\|\n", "", body).split())
    record("docs", "PASS", f"report main text about {words:,} words", "check the limit in your brief")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip pytest and the app check")
    args = ap.parse_args()
    print(f"Verifying project at {ROOT}")
    if not check_environment():
        print("\nFix the environment first (see FAIL above).")
        return 1
    check_code()
    if not args.fast:
        check_tests()
    check_data()
    check_experiments()
    if not args.fast:
        check_app()
    check_docs()

    counts = {s: sum(r[1] == s for r in RESULTS) for s in ("PASS", "WARN", "FAIL")}
    print(f"\nSUMMARY: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    if counts["FAIL"]:
        print("Fix the FAIL items (the command after '->' tells you how), then rerun this script.")
    elif counts["WARN"]:
        print("Core project complete. WARN items are optional extensions or results still to add.")
    else:
        print("Everything checked out.")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
