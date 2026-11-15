# ArXiv Research Assistant — DG4NLP Project

Scientific paper **classification** and **title generation** on arXiv metadata, comparing
traditional machine learning (TF-IDF + Naive Bayes / Logistic Regression / Linear SVM) with
FLAN-T5 and optional LLMs, plus paper retrieval and a grounded research-assistant chatbot,
delivered as a local **Streamlit** application.

Everything runs locally on a CPU laptop. No API key is required; LLM features are optional.

---

## 1. Quick start (Windows 11, VS Code, PowerShell)

```powershell
cd path\to\DG4NLP_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements-lock.txt     # records the exact versions you used
copy .env.example .env
python check_env.py                    # verifies the environment
python -m pytest                       # 109 tests, no dataset or internet needed
# put Mendeley_arXiv_Scientific_Research_Paper_Dataset.zip into data\\raw\\ (see §2)
python train.py all                    # runs every experiment stage (~10-25 min on CPU)
python verify_project.py               # full PASS/WARN/FAIL check of the whole project
streamlit run app.py                   # opens http://localhost:8501
```

In VS Code choose **Ctrl+Shift+P → Python: Select Interpreter → .venv**.
If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
Optional GPU: install the CUDA build of PyTorch from https://pytorch.org before the requirements.
Optional FAISS vector store: `pip install -r requirements-optional.txt`.

## 2. Dataset

The project uses the **Mendeley "arXiv Scientific Research Paper Dataset"**
(`Mendeley_arXiv_Scientific_Research_Paper_Dataset.zip`, 287,421 papers, mostly computer
science, 1993–2026 with 62.5% submitted in 2023 or later). Put the zip (or the extracted
`arXiv_scientific_dataset.csv`) in `data/raw/`; it does not need to be unzipped.

Other supported sources (used only if the Mendeley file is absent, in this priority order):
the Kaggle snapshot `arxiv-metadata-oai-snapshot.json`
(https://www.kaggle.com/datasets/Cornell-University/arxiv) and the ArXiV Archive TSV
(`python train.py download`). `config.YEAR_FILTER` (default `None` = all years) can restrict the
data to one year. The loader discovers formats and columns from the files.

## 3. Experiment stages (`train.py`)

| Command | What it does | Main outputs |
|---|---|---|
| `python train.py prepare` | load → schema/missing/duplicates → clean → top-10 primary categories → cap 5,000/class → stratified 70/15/15 split → 300-paper shared evaluation subset → 7 EDA figures | `data/processed/*.parquet`, `outputs/figures/01-07_*.png`, `eda_summary.json` |
| `python train.py classify` | TF-IDF grid (n-grams, vocabulary, min_df) + classifier grid on the **validation** set; refit on train+val; evaluate **once** on test | `models/classification/*.joblib`, `classification_metrics_test.csv`, confusion matrices |
| `python train.py titles` | FLAN-T5 (zero-shot) + two extractive baselines on the evaluation subset; ROUGE-1/2/L, semantic similarity, unsupported-word rate | `titles_generated.csv`, `title_metrics.csv` |
| `python train.py llm` | LLM classification (JSON, validated) + titles with zero-shot / instructed / few-shot prompts. Skipped cleanly if no LLM is configured | `llm_classification_*.csv`, `llm_status.json` |
| `python train.py index` | Retrieval index over train+val papers (dense embeddings, TF-IDF fallback); Recall@k, MRR, evidence threshold | `models/embeddings/`, `retrieval_metrics.json` |
| `python train.py compare` | Same-paper comparison of all classifiers + charts | `comparison_classification.csv`, `comparison_*.png` |
| `python train.py deep` | Deep learning: MiniLM sentence embeddings + MLP (about 10 min), and DistilBERT fine-tuned on 600 papers/class for 2 epochs (about 45-60 min on CPU). Not part of `all`. Options: `--skip-bert`, `--skip-mlp`, `--quick` | `predictions_test_minilm_mlp.csv`, `predictions_test_distilbert.csv`, `deep_distilbert_history.json`, `models/deep/` |
| `python train.py stats` | 95% bootstrap confidence intervals (accuracy, macro-F1) and pairwise McNemar tests on the same test papers | `significance_ci.csv`, `significance_mcnemar.csv` |
| `python train.py all` | all of the above in order | |

Options: `--quick` (small sample, smoke test), `--limit N` (fewer abstracts for `titles` / `llm`;
the subset stays class-balanced), `--backend auto|dense|tfidf` (for `index`).
Every stage records its settings and library versions in `outputs/results/run_config.json`.

## 4. LLM experiments (required by the brief)

**Easiest (Option C, no install, no key):** in `.env` set

```ini
LLM_BACKEND=local
```
then run `python train.py llm`, `python train.py compare` and `python train.py stats`.
This uses `google/flan-t5-large` through Hugging Face Transformers (about 3 GB downloaded once;
roughly 20-30 minutes on a laptop CPU for 300 classifications and 3 x 60 titles).
For a quicker run set `LLM_LOCAL_MODEL=google/flan-t5-base` or add `--limit 100`.

Alternatively, use Ollama or a hosted API. Edit `.env` (never commit or submit it):

```ini
# Option B - local and free: install Ollama (https://ollama.com), pull a small instruct model
LLM_BACKEND=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=<model name you pulled>
LLM_API_KEY=ollama

# Option A - hosted, OpenAI-compatible API
# LLM_BASE_URL=https://<provider>/v1
# LLM_MODEL=<provider model>
# LLM_API_KEY=<secret key>
```

Then run `python train.py llm --limit 100` and `python train.py compare`. All LLM requests
and responses are cached in `outputs/results/llm_cache.jsonl` so results are auditable and
re-runs cost nothing. The key is never cached or logged.

## 5. The application (`streamlit run app.py`)

| Page | Function |
|---|---|
| Home dashboard | dataset statistics, component status, the 7 EDA figures |
| Paper classification | abstract → category with NB / LR / SVM / LLM, top-3 scores, all-model comparison |
| Title generation | FLAN-T5, LLM prompt styles or baselines; ROUGE against an optional original title |
| Model comparison | only measured results: metrics tables, confusion matrices, latency, ROUGE, retrieval |
| Paper explorer | ranked search, category filter, abstracts with arXiv links, similar papers |
| Research assistant chatbot | answers only from retrieved papers, cites arXiv IDs, refuses without evidence |

"Load example abstract" uses real papers from the **held-out test set**. The app never trains
models and shows a run command when an artefact is missing. It works with no LLM configured.

## 6. Project structure

```text
DG4NLP_Project/
├── app.py  train.py  config.py  check_env.py  verify_project.py  pytest.ini
├── requirements.txt  requirements-optional.txt  .env.example  .gitignore
├── README.md  REPORT.md  REPORT.pdf
├── docs/       architecture diagram + PDF export scripts
├── data/raw/ data/processed/          # dataset (not included), processed splits
├── models/classification/ models/embeddings/
├── outputs/figures/ outputs/results/  # every reported number lives here
├── src/  utils, data_loader, preprocessing, exploratory_analysis, classification,
│         evaluation, title_generation, llm_client, llm_classification,
│         llm_title_generation, rag
├── ui/   components, dashboard, classification_page, generation_page,
│         comparison_page, explorer_page, chatbot_page
└── tests/ conftest + 7 test modules (synthetic data only)
```

Changes from the original plan: `app/` was renamed to `ui/` (a package called `app` would clash with
`app.py`); `src/utils.py`, `src/llm_client.py` and `check_env.py` were added; notebooks are optional
and not required to run anything.

## 7. Tests

```powershell
python -m pytest              # all tests
python -m pytest -v tests\test_preprocessing.py
python -m pytest -k llm       # tests whose name contains "llm"
```

The tests use small synthetic data and mocked models/HTTP, so they need neither the dataset nor
internet. `test_app_validation.py` also starts the Streamlit app headless and opens every page.

### Report PDF and architecture diagram

```powershell
python docs/make_architecture_diagram.py   # redraws docs/architecture.png (Figure 1 of the report)
pip install markdown weasyprint            # one-off, for the PDF export
python docs/export_report_pdf.py           # REPORT.md -> REPORT.pdf (cover page, numbered sections)
```
On Windows WeasyPrint needs the GTK runtime. If that is a problem, open `REPORT.md` in VS Code with the
**Markdown PDF** extension and choose *Markdown PDF: Export (pdf)*.

### Whole-project check

`python verify_project.py` (works from any folder) checks the environment, imports every module,
runs pytest, checks the data splits for leakage, recomputes every reported macro-F1 from the saved
predictions, confirms the saved models reproduce them, opens all six app pages headless, and counts
the report placeholders. Each FAIL line ends with the command that fixes it.
`python verify_project.py --fast` skips pytest and the app check.

## 8. Common errors

| Symptom | Fix |
|---|---|
| `No dataset found in data\raw` | Put the Kaggle JSON in `data\raw\` or run `python train.py download` |
| `Missing train.parquet. Run python train.py prepare first.` | Run the stages in order (or `python train.py all`) |
| FLAN-T5 "not run" in results | `pip install torch transformers sentencepiece`, need internet on first run (model download ~1 GB) |
| Retrieval backend shows `tfidf` | `sentence-transformers` missing or offline; install it and rerun `python train.py index` |
| LLM stage "connection failed" | Is Ollama running (`ollama serve`)? Is `LLM_MODEL` pulled? Is the API key valid? |
| `ModuleNotFoundError: config` / `src` | Run commands from the project root with the venv active |
| Streamlit shows old results | Press **R** in the browser or restart `streamlit run app.py` (results are cached) |
| Memory error while loading | Keep `YEAR_FILTER` set, or set `MAX_RAW_RECORDS` in `config.py` |

## 9. Reproducibility

Fixed seed (42) for sampling, splitting and models; vectorisers fitted on training data only;
hyperparameters chosen on validation data; test set evaluated once; all settings, library versions and
stage timestamps in `run_config.json`; exact package versions in `requirements-lock.txt`.
Classification metrics are bit-identical across runs on the same data; latency figures vary slightly
with machine load.
