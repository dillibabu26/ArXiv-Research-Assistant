"""Dataset discovery, loading of both formats, malformed records, schema inspection."""

import io
import json
import zipfile

import pandas as pd
import pytest

from src.data_loader import (detect_format, discover_raw_files, inspect_schema, load_raw,
                             repair_arxiv_id)


def _write_jsonl(path, records, extra_lines=()):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        for line in extra_lines:
            f.write(line + "\n")


def _rec(i, year="19"):
    return {"id": f"{year}01.{i:05d}", "title": f"Title {i}", "abstract": f"Abstract {i}",
            "categories": "cs.CL cs.LG", "authors": "X", "update_date": "2019-01-02"}


def test_kaggle_jsonl_skips_malformed_and_incomplete(tmp_path):
    p = tmp_path / "arxiv.json"
    _write_jsonl(p, [_rec(1), _rec(2), _rec(3, year="18")],
                 extra_lines=["{not valid json", json.dumps({"id": "1901.99999"})])
    df, report = load_raw(p, year="2019")
    assert list(df["id"]) == ["1901.00001", "1901.00002"]
    assert report.malformed == 1
    assert report.missing_required == 1
    assert report.filtered_out == 1
    assert set(["id", "title", "abstract", "categories", "source"]) <= set(df.columns)


def test_year_filter_none_keeps_all(tmp_path):
    p = tmp_path / "arxiv.json"
    _write_jsonl(p, [_rec(1), _rec(2, year="18")])
    df, _ = load_raw(p, year=None)
    assert len(df) == 2


def test_archive_tsv_zip_is_mapped_to_standard_schema(tmp_path):
    tsv = ("\tabstract\tarxiv_id\tauthor_text\tcategories\tcomments\tcreated\tdoi\tprimary_cat\ttitle\tupdated\n"
           "0\tSome abstract\t1910.1113699999999\tA B\tcs.LG,cs.CL\t\t2019-10-01\t\tcs.CL\tT1\t2019-10-02\n"
           "1\tOther abstract\t1906.0375\tC D\tquant-ph\t5 pages\t2019-06-01\t\tquant-ph\tT2\t\n")
    p = tmp_path / "arxiv_archive_2019.tsv.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("2019.tsv", tsv)
    assert detect_format(p) == "archive_tsv"
    df, report = load_raw(p, year="2019")
    assert list(df["id"]) == ["1910.11137", "1906.03750"]
    # explicit primary category is moved to the front
    assert df.loc[0, "categories"] == "cs.CL cs.LG"
    assert df.loc[1, "comments"] == "5 pages"
    assert pd.isna(df.loc[0, "comments"])
    # update_date: latest update if present, otherwise the creation date (never silently empty)
    assert df.loc[0, "update_date"] == "2019-10-02"
    assert df.loc[1, "update_date"] == "2019-06-01"


def test_repair_arxiv_id():
    assert repair_arxiv_id("1906.0375") == "1906.03750"
    assert repair_arxiv_id("1910.1113699999999") == "1910.11137"
    assert repair_arxiv_id("math/0101001") == "math/0101001"


def test_detect_format_rejects_unrelated_files(tmp_path):
    p = tmp_path / "other.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        detect_format(p)


def test_discover_and_missing_dataset_message(tmp_path, monkeypatch):
    assert discover_raw_files(tmp_path) == []
    import src.data_loader as dl
    monkeypatch.setattr(dl, "discover_raw_files", lambda: [])
    with pytest.raises(FileNotFoundError, match="No dataset found"):
        dl.load_raw()


def test_inspect_schema_counts_empty_strings_as_missing():
    df = pd.DataFrame({"id": ["1", "2", "2"], "abstract": ["a", "", "a"], "doi": [None, "x", None]})
    s = inspect_schema(df)
    assert s["missing"]["abstract"] == 1
    assert s["missing"]["doi"] == 2
    assert s["duplicate_ids"] == 1
    assert s["n_rows"] == 3


MENDELEY_CSV = (
    "id,title,category,category_code,published_date,updated_date,authors,first_author,summary,summary_word_count\n"
    'abs-1906.03691v1,"Interpreting Age Effects\n  using 3D CNNs",Image and Video Processing,eess.IV,6/9/19,6/10/19,'
    '"[\'A B\']",\'A B\',"Understanding fetal\nneurodevelopment matters.",4\n'
    'cs-9308101v1,Old Style Paper,Artificial Intelligence,cs.AI,8/1/93,8/1/93,"[\'C D\']",\'C D\',"An old abstract.",3\n'
    'abs-2511.15658v2,New Paper,Machine Learning,cs.LG,11/19/25,11/20/25,"[\'E F\']",\'E F\',"A new abstract.",3\n'
)


def test_mendeley_csv_and_zip(tmp_path):
    from src.data_loader import normalise_mendeley_id

    p = tmp_path / "arXiv_scientific_dataset.csv"
    p.write_text(MENDELEY_CSV, encoding="utf-8")
    z = tmp_path / "Mendeley_arXiv.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("arXiv Scientific Research Paper Dataset/arXiv_scientific_dataset.csv", MENDELEY_CSV)
    for path in (p, z):
        assert detect_format(path) == "mendeley_csv"
        df, report = load_raw(path, year=None)
        assert list(df["id"]) == ["1906.03691", "cs/9308101", "2511.15658"]
        assert list(df["categories"]) == ["eess.IV", "cs.AI", "cs.LG"]
        assert "fetal\nneurodevelopment" in df.loc[0, "abstract"]          # multi-line field kept
        assert df.loc[0, "update_date"] == "2019-06-10"
        assert report.source == "mendeley_csv"
    df, report = load_raw(p, year="2025")
    assert list(df["id"]) == ["2511.15658"] and report.filtered_out == 2
    assert normalise_mendeley_id("abs-0704.0001v3") == "0704.0001"


def test_mendeley_preferred_over_archive(tmp_path, monkeypatch):
    import config
    import src.data_loader as dl

    (tmp_path / "a.csv").write_text(MENDELEY_CSV, encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "arxiv_archive_2019.tsv.zip", "w") as zf:
        zf.writestr("2019.tsv", "\tabstract\tarxiv_id\tauthor_text\tcategories\tcomments\tcreated\tdoi\tprimary_cat\ttitle\tupdated\n"
                    + "0\t" + "x " * 500 + "\t1901.00001\tA\tcs.LG\t\t2019-01-01\t\tcs.LG\tT\t\n")
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "discover_raw_files", lambda: dl.discover_raw_files.__wrapped__(tmp_path)
                        if hasattr(dl.discover_raw_files, "__wrapped__") else sorted(tmp_path.iterdir()))
    df, report = dl.load_raw(year=None)
    assert report.source == "mendeley_csv"
