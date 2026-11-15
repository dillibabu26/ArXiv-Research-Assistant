"""
Environment check for the DG4NLP project (Phase 3).

Run from the project root:
    python check_env.py

It verifies the Python version, reports installed library versions, creates
the output folders, downloads NLTK data, detects CPU/GPU, reports the LLM
configuration (without revealing the key) and checks for the raw dataset.
It never downloads the arXiv dataset itself.
"""

from __future__ import annotations

import sys

import config
from src.utils import ensure_nltk_data, get_device, get_library_versions


def main() -> int:
    problems = 0
    print("=" * 60)
    print("DG4NLP environment check")
    print("=" * 60)

    # 1. Python version
    v = sys.version_info
    ok = v >= (3, 10)
    print(f"[{'OK' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro} "
          f"(recommended 3.11)")
    print(f"       interpreter: {sys.executable}")
    if ".venv" not in sys.executable and "venv" not in sys.executable:
        print("[WARN] This does not look like the project virtual environment.")
    problems += not ok

    # 2. Libraries
    print("\nLibraries:")
    versions = get_library_versions()
    optional = {"faiss-cpu"}
    for name, ver in versions.items():
        missing = ver == "NOT INSTALLED"
        tag = "OPT " if (missing and name in optional) else ("FAIL" if missing else "OK  ")
        print(f"  [{tag}] {name:<24} {ver}")
        if missing and name not in optional:
            problems += 1

    # 3. Folders
    config.ensure_dirs()
    print("\n[OK] Output folders created/verified.")

    # 4. NLTK data
    if versions.get("nltk") != "NOT INSTALLED":
        status = ensure_nltk_data()
        # Only one of punkt / punkt_tab is needed, depending on the NLTK version.
        failed = [k for k in ("stopwords", "wordnet") if not status.get(k)]
        if not (status.get("punkt") or status.get("punkt_tab")):
            failed.append("punkt/punkt_tab")
        if failed:
            print(f"[WARN] NLTK resources not available: {failed}")
        else:
            print("[OK] NLTK resources available.")

    # 5. Device
    device = get_device(config.FORCE_CPU)
    print(f"[OK] Compute device: {device}")

    # 6. LLM configuration (never print the key)
    print(f"\nLLM backend: {config.LLM_BACKEND}")
    if config.LLM_BACKEND != "none":
        print(f"  base URL : {config.LLM_BASE_URL}")
        print(f"  model    : {config.LLM_MODEL or '(not set)'}")
        print(f"  API key  : {'set' if config.LLM_API_KEY else 'not set'}")
    else:
        print("  LLM features disabled (this is fine; the core project still runs).")

    # 7. Raw dataset
    raw_files = [p for p in config.RAW_DATA_DIR.iterdir()
                 if p.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".zip"}]
    if raw_files:
        for p in raw_files:
            print(f"[OK] Raw data found: {p.name} ({p.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"[TODO] No dataset in {config.RAW_DATA_DIR}. "
              "Download it from Kaggle or run `python train.py download` (see README).")

    print("\n" + ("All required checks passed." if problems == 0
                  else f"{problems} problem(s) found — see FAIL lines above."))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
