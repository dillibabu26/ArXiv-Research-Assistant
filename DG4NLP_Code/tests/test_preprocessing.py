"""Text cleaning, category extraction, cleaning steps and leakage-free splitting."""

import pandas as pd
import pytest

from src.preprocessing import (assert_no_leakage, clean_dataframe, clean_for_tfidf,
                               clean_for_transformer, extract_primary_category, get_archive,
                               make_eval_subset, split_categories, split_dataset)


class TestTfidfCleaning:
    def test_lowercases_and_removes_stopwords(self):
        assert clean_for_tfidf("The Transformer AND the Data") == "transformer data"

    def test_keeps_hyphenated_and_alphanumeric_terms(self):
        out = clean_for_tfidf("Self-supervised GPT-2 models for 3D CO2 sensing").split()
        for term in ["self-supervised", "gpt-2", "3d", "co2", "sensing"]:
            assert term in out

    def test_math_url_and_latex_handling(self):
        out = clean_for_tfidf(r"Energy $E=mc^2$ see https://x.org and \emph{robust} \alpha bounds")
        assert "mathexpr" in out and "urltoken" in out
        assert "robust" in out and "alpha" not in out and "emph" not in out

    def test_drops_pure_numbers(self):
        assert "2019" not in clean_for_tfidf("Results from 2019 experiments").split()

    def test_optional_stemming(self):
        assert clean_for_tfidf("networks learning", stem=True) == "network learn"

    def test_keep_stopwords_option(self):
        assert "the" in clean_for_tfidf("the network", remove_stopwords=False).split()

    @pytest.mark.parametrize("bad", [None, "", "   ", 123])
    def test_non_text_returns_empty(self, bad):
        assert clean_for_tfidf(bad) == ""


def test_transformer_cleaning_keeps_case_and_punctuation():
    out = clean_for_transformer("We use \\emph{BERT}\n  for NER, achieving   $F_1$.")
    assert out == "We use BERT for NER, achieving $F_1$."


@pytest.mark.parametrize("cats,expected", [
    ("cs.CL cs.AI cs.LG", "cs.CL"), ("hep-ph,hep-th", "hep-ph"),
    ("  cond-mat.mes-hall  ", "cond-mat.mes-hall"), ("", None), (None, None)])
def test_extract_primary_category(cats, expected):
    assert extract_primary_category(cats) == expected


def test_split_categories_and_archive():
    assert split_categories("a.B c.D,e") == ["a.B", "c.D", "e"]
    assert get_archive("cs.LG") == "cs" and get_archive("hep-ph") == "hep-ph" and get_archive(None) is None


def test_clean_dataframe_removes_bad_records(synthetic_df):
    df = synthetic_df.copy()
    extra = pd.DataFrame([
        {**df.iloc[0].to_dict(), "id": "1901.90000"},                         # duplicate abstract
        {**df.iloc[1].to_dict(), "id": "1901.90001", "abstract": "Too short."},
        {**df.iloc[2].to_dict(), "id": "1901.90002",
         "abstract": "This paper has been withdrawn by the author due to an error " * 3},
        {**df.iloc[3].to_dict(), "id": "1901.90003", "abstract": None},
        {**df.iloc[4].to_dict()},                                             # duplicate id
    ])
    clean, steps = clean_dataframe(pd.concat([df, extra], ignore_index=True))
    assert len(clean) == len(df)
    assert steps["removed"].sum() == len(extra)
    assert {"primary_category", "abstract_words", "title_words"} <= set(clean.columns)


def test_split_is_stratified_and_leak_free(labelled_df):
    train, val, test = split_dataset(labelled_df, test_size=0.2, val_size=0.2, seed=1)
    assert len(train) + len(val) + len(test) == len(labelled_df)
    assert_no_leakage(train, val, test)
    for part in (train, val, test):
        props = part["label"].value_counts(normalize=True)
        assert props.max() - props.min() < 0.1


def test_split_is_reproducible(labelled_df):
    a = split_dataset(labelled_df, seed=7)[2]["id"].tolist()
    b = split_dataset(labelled_df, seed=7)[2]["id"].tolist()
    assert a == b


def test_leakage_detection(labelled_df):
    with pytest.raises(AssertionError):
        assert_no_leakage(labelled_df, labelled_df.head(2), labelled_df.tail(2))


def test_eval_subset_is_balanced(labelled_df):
    sub = make_eval_subset(labelled_df, size=15)
    assert sub["label"].value_counts().tolist() == [5, 5, 5]
