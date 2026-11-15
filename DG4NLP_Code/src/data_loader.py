"""
Dataset discovery and loading for arXiv metadata.

Three real sources are supported, all derived from arXiv metadata:

0. **Mendeley "arXiv Scientific Research Paper Dataset"** (CSV, optionally zipped; the
   project's default). ~287k papers, mostly computer science, 1993-2026 (most from 2025-26),
   one category per paper. Columns: id, title, category, category_code, published_date,
   updated_date, authors, first_author, summary (= abstract), summary_word_count.

1. **Kaggle snapshot** (`arxiv-metadata-oai-snapshot.json`, Cornell University).
   JSON-lines: one paper per line, ~4.5 GB. It is *streamed* line by line so memory
   use depends on how many records are kept, not on the file size.
2. **ArXiV Archive per-year TSV** (Geiger, 2019; Zenodo DOI 10.5281/zenodo.1463242),
   hosted on GitHub. Used when Kaggle is not available. Its columns differ
   (e.g. `arxiv_id`, comma-separated `categories`, explicit `primary_cat`), so it is
   mapped onto the Kaggle schema.

Whatever the source, `load_raw()` returns a DataFrame with the *standard schema*:

    id, title, abstract, categories (space-separated, primary first),
    authors, comments, doi, update_date, source

Filenames and columns are discovered from the files rather than assumed.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

import config
from src.utils import get_logger

logger = get_logger()

STANDARD_COLUMNS = ["id", "title", "abstract", "categories", "authors",
                    "comments", "doi", "update_date", "source"]
KAGGLE_REQUIRED = {"id", "title", "abstract", "categories"}
ARCHIVE_REQUIRED = {"arxiv_id", "title", "abstract", "categories"}
MENDELEY_REQUIRED = {"id", "title", "summary", "category_code"}
# When several datasets are in data/raw/, the first available format in this order is used.
SOURCE_PRIORITY = ["mendeley_csv", "kaggle_jsonl", "archive_tsv", "csv_generic"]
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".zip"}

ARCHIVE_URL_TEMPLATE = ("https://raw.githubusercontent.com/staeiou/arxiv_archive/HEAD/"
                        "processed_data/20200101/per_year/{year}.tsv.zip")

NEW_STYLE_ID = re.compile(r"^\d{4}\.\d{4,5}$")


@dataclass
class LoadReport:
    """Statistics collected while loading, reported in EDA and the report."""
    source: str = ""
    path: str = ""
    lines_read: int = 0
    malformed: int = 0
    missing_required: int = 0
    filtered_out: int = 0
    kept: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_raw_files(raw_dir: Path = config.RAW_DATA_DIR) -> list[Path]:
    """Return candidate data files in `raw_dir`, largest first."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    files = [p for p in raw_dir.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=lambda p: p.stat().st_size, reverse=True)


def detect_format(path: Path) -> str:
    """
    Inspect the first record/header of a file and return
    'mendeley_csv', 'kaggle_jsonl', 'archive_tsv' or 'csv_generic'. Raises ValueError otherwise.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    keys = set(json.loads(line).keys())
                    if KAGGLE_REQUIRED <= keys:
                        return "kaggle_jsonl"
                    raise ValueError(f"JSON keys {sorted(keys)} lack {KAGGLE_REQUIRED}")
        raise ValueError(f"{path.name} is empty")
    if suffix in {".tsv", ".csv", ".zip"}:
        header = _read_header(path)
        if MENDELEY_REQUIRED <= header:
            return "mendeley_csv"
        if ARCHIVE_REQUIRED <= header:
            return "archive_tsv"
        if KAGGLE_REQUIRED <= header:
            return "csv_generic"
        raise ValueError(f"Columns {sorted(header)} not recognised as arXiv metadata")
    raise ValueError(f"Unsupported file type: {path.name}")


def _open_text(path: Path) -> io.TextIOBase:
    """Open a plain or zipped text file (first member of a zip)."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        member = next(n for n in zf.namelist() if not n.endswith("/"))
        return io.TextIOWrapper(zf.open(member), encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _separator(path: Path) -> str:
    name = path.name.lower()
    return "\t" if (".tsv" in name) else ","


def _read_header(path: Path) -> set[str]:
    with _open_text(path) as f:
        first = f.readline().rstrip("\n")
    sep = "\t" if "\t" in first else ","
    return {c.strip() for c in first.split(sep)}


# ---------------------------------------------------------------------------
# Kaggle JSON-lines (streamed)
# ---------------------------------------------------------------------------
def iter_records(path: Path, report: LoadReport | None = None) -> Iterator[dict]:
    """
    Stream records from a JSON-lines file, skipping malformed lines safely.
    Each malformed or incomplete line is counted in `report` rather than crashing.
    """
    report = report or LoadReport()
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            report.lines_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                report.malformed += 1
                continue
            if not isinstance(rec, dict) or not KAGGLE_REQUIRED <= rec.keys():
                report.missing_required += 1
                continue
            yield rec


def _year_matches(arxiv_id: str, year: str | None) -> bool:
    """New-style arXiv IDs start with YYMM, so '19' selects papers first submitted in 2019."""
    if not year:
        return True
    return bool(NEW_STYLE_ID.match(arxiv_id)) and arxiv_id.startswith(str(year)[2:])


def load_kaggle_jsonl(path: Path, year: str | None = config.YEAR_FILTER,
                      max_records: int | None = config.MAX_RAW_RECORDS) -> tuple[pd.DataFrame, LoadReport]:
    """Stream the Kaggle snapshot into the standard schema."""
    report = LoadReport(source="kaggle_jsonl", path=str(path))
    rows = []
    for rec in iter_records(path, report):
        arxiv_id = str(rec.get("id", "")).strip()
        if not _year_matches(arxiv_id, year):
            report.filtered_out += 1
            continue
        rows.append({
            "id": arxiv_id,
            "title": rec.get("title"),
            "abstract": rec.get("abstract"),
            "categories": rec.get("categories"),
            "authors": rec.get("authors"),
            "comments": rec.get("comments"),
            "doi": rec.get("doi"),
            "update_date": rec.get("update_date"),
            "source": "kaggle",
        })
        if max_records and len(rows) >= max_records:
            report.notes.append(f"stopped at max_records={max_records}")
            break
    report.kept = len(rows)
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS), report


# ---------------------------------------------------------------------------
# ArXiV Archive TSV
# ---------------------------------------------------------------------------
def repair_arxiv_id(raw_id: str) -> str:
    """
    Some archive rows store new-style IDs as floats (e.g. '1910.1113699999999'
    or '1906.0375'). Since 2015 every new-style ID is YYMM.NNNNN (5 digits), so
    rounding to 5 decimals restores the canonical form. Old-style IDs
    (e.g. 'math/0101001') are returned unchanged.
    """
    s = str(raw_id).strip()
    if not re.match(r"^\d{4}\.\d+$", s):
        return s
    return f"{float(s):.5f}"


def load_archive_tsv(path: Path, year: str | None = config.YEAR_FILTER,
                     max_records: int | None = config.MAX_RAW_RECORDS) -> tuple[pd.DataFrame, LoadReport]:
    """Read the archive TSV/CSV (optionally zipped) and map it to the standard schema."""
    report = LoadReport(source="archive_tsv", path=str(path))
    with _open_text(Path(path)) as f:
        df = pd.read_csv(f, sep=_separator(Path(path)), dtype=str,
                         keep_default_na=False, on_bad_lines="skip")
    report.lines_read = len(df)

    df["id"] = df["arxiv_id"].map(repair_arxiv_id)
    cats = df["categories"].fillna("").str.replace(",", " ").str.split()
    # Put the explicit primary category first so the 'first listed = primary'
    # convention of the Kaggle file also holds for this source.
    if "primary_cat" in df.columns:
        prim = df["primary_cat"].str.strip()
        agree = (cats.str[0] == prim).mean()
        report.notes.append(f"first-listed category equals explicit primary for {agree:.2%} of rows")
        cats = [([p] + [c for c in cl if c != p]) if p else cl for p, cl in zip(prim, cats)]
    out = pd.DataFrame({
        "id": df["id"],
        "title": df["title"],
        "abstract": df["abstract"],
        "categories": [" ".join(c) for c in cats],
        "authors": df.get("author_text", ""),
        "comments": df.get("comments", ""),
        "doi": df.get("doi", ""),
        # Kaggle's update_date is always filled (date of the latest version). The archive
        # only fills `updated` for revised papers, so fall back to `created` otherwise.
        "update_date": (df["updated"].where(df["updated"].str.strip() != "", df["created"])
                        if {"updated", "created"} <= set(df.columns) else df.get("updated", "")),
        "source": "arxiv_archive",
    })
    out = out.replace({"": None})
    mask = out["id"].map(lambda x: _year_matches(str(x), year))
    report.filtered_out = int((~mask).sum())
    out = out[mask]
    if max_records:
        out = out.head(max_records)
    report.kept = len(out)
    return out.reset_index(drop=True)[STANDARD_COLUMNS], report


def normalise_mendeley_id(raw_id: str) -> str:
    """
    Mendeley IDs encode the arXiv ID with a prefix and version:
      'abs-1906.03691v1' -> '1906.03691'      (new-style ID)
      'cs-9308101v1'     -> 'cs/9308101'      (old-style archive/number ID)
    """
    s = str(raw_id).strip()
    m = re.match(r"^abs-(\d{4}\.\d{4,5})(?:v\d+)?$", s)
    if m:
        return m.group(1)
    m = re.match(r"^([a-z][a-z.\-]*?)-(\d{7})(?:v\d+)?$", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return re.sub(r"v\d+$", "", s)


def _us_date_to_iso(series: pd.Series) -> pd.Series:
    """'6/9/19' (month/day/two-digit year) -> '2019-06-09'; unparseable -> None."""
    d = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    return d.dt.strftime("%Y-%m-%d").where(d.notna(), None)


def load_mendeley_csv(path: Path, year: str | None = config.YEAR_FILTER,
                      max_records: int | None = config.MAX_RAW_RECORDS) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load the Mendeley arXiv CSV (plain or zipped). Fields are quoted and may contain line
    breaks, which pandas' CSV parser handles. `year` filters on the publication date.
    """
    report = LoadReport(source="mendeley_csv", path=str(path))
    with _open_text(Path(path)) as f:
        df = pd.read_csv(f, dtype=str, keep_default_na=False, on_bad_lines="skip")
    report.lines_read = len(df)
    published = _us_date_to_iso(df["published_date"]) if "published_date" in df else None
    out = pd.DataFrame({
        "id": df["id"].map(normalise_mendeley_id),
        "title": df["title"],
        "abstract": df["summary"],
        "categories": df["category_code"].str.strip(),
        "authors": df.get("authors", ""),
        "comments": None,
        "doi": None,
        "update_date": _us_date_to_iso(df["updated_date"]) if "updated_date" in df else published,
        "source": "mendeley",
    })
    out = out.replace({"": None})
    if published is not None:
        report.notes.append(f"publication dates {published.min()} to {published.max()}")
    if year and published is not None:
        mask = published.fillna("").str.startswith(str(year))
        report.filtered_out = int((~mask).sum())
        out = out[mask.values]
    if max_records:
        out = out.head(max_records)
    report.kept = len(out)
    return out.reset_index(drop=True)[STANDARD_COLUMNS], report


def load_generic_csv(path: Path, year: str | None = config.YEAR_FILTER,
                     max_records: int | None = config.MAX_RAW_RECORDS) -> tuple[pd.DataFrame, LoadReport]:
    """Load a CSV that already uses Kaggle column names."""
    report = LoadReport(source="csv_generic", path=str(path))
    df = pd.read_csv(path, sep=_separator(Path(path)), dtype=str, on_bad_lines="skip")
    report.lines_read = len(df)
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["source"] = "csv"
    mask = df["id"].astype(str).map(lambda x: _year_matches(x, year))
    report.filtered_out = int((~mask).sum())
    df = df[mask]
    if max_records:
        df = df.head(max_records)
    report.kept = len(df)
    return df[STANDARD_COLUMNS].reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def load_raw(path: Path | None = None, year: str | None = config.YEAR_FILTER,
             max_records: int | None = config.MAX_RAW_RECORDS) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load arXiv metadata from `path`, or from the first recognised file in
    data/raw/ (the Kaggle snapshot is preferred when both are present).
    """
    if path is None:
        candidates = discover_raw_files()
        if not candidates:
            raise FileNotFoundError(
                f"No dataset found in {config.RAW_DATA_DIR}. Put the Mendeley CSV/zip or the "
                "Kaggle file there (see README) or run `python train.py download`.")
        formats = {}
        for p in candidates:
            try:
                formats[p] = detect_format(p)
            except (ValueError, json.JSONDecodeError, zipfile.BadZipFile, UnicodeDecodeError):
                continue
        if not formats:
            raise ValueError(f"No recognised arXiv metadata file in {config.RAW_DATA_DIR}")
        path = min(formats, key=lambda p: SOURCE_PRIORITY.index(formats[p]))
    path = Path(path)
    fmt = detect_format(path)
    logger.info("Loading %s (format: %s, year filter: %s)", path.name, fmt, year)
    loader = {"mendeley_csv": load_mendeley_csv, "kaggle_jsonl": load_kaggle_jsonl,
              "archive_tsv": load_archive_tsv, "csv_generic": load_generic_csv}[fmt]
    df, report = loader(path, year=year, max_records=max_records)
    logger.info("Kept %s records (%s malformed lines skipped)", f"{report.kept:,}", report.malformed)
    return df, report


def download_archive(year: str = "2019",
                     dest_dir: Path = config.RAW_DATA_DIR) -> Path:
    """
    Download the public ArXiV Archive TSV for one year from GitHub.
    Used only when the Kaggle snapshot is unavailable.
    """
    import requests

    dest = Path(dest_dir) / f"arxiv_archive_{year}.tsv.zip"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Already downloaded: %s", dest)
        return dest
    url = ARCHIVE_URL_TEMPLATE.format(year=year)
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def inspect_schema(df: pd.DataFrame) -> dict:
    """Return shape, dtypes, missing values (empty strings count as missing) and duplicates."""
    blank = df.apply(lambda s: s.isna() | (s.astype(str).str.strip() == ""))
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in blank.sum().items()},
        "missing_pct": {c: round(float(v) * 100, 2) for c, v in blank.mean().items()},
        "duplicate_ids": int(df["id"].duplicated().sum()) if "id" in df else None,
        "duplicate_abstracts": int(df["abstract"].duplicated().sum()) if "abstract" in df else None,
    }
