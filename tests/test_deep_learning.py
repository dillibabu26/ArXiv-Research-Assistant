"""Deep-learning classifiers: subset sampling, MLP on embeddings, embedding reuse, fine-tuning loop.
The fine-tuning test builds a TINY random DistilBERT and tokenizer locally (no downloads)."""

import numpy as np
import pandas as pd
import pytest

import config
from src import deep_learning as D


def test_balanced_subset(labelled_df):
    sub = D.balanced_subset(labelled_df, per_class=5)
    assert sub["label"].value_counts().tolist() == [5, 5, 5]
    assert D.balanced_subset(labelled_df, 5)["id"].tolist() == sub["id"].tolist()   # reproducible


def test_mlp_learns_separable_embeddings():
    rng = np.random.default_rng(0)
    labels = ["a", "b", "c"]
    centres = rng.normal(size=(3, 16)) * 3
    y = np.repeat(labels, 60)
    X = np.vstack([centres[i] + rng.normal(scale=0.5, size=(60, 16)) for i in range(3)]).astype(np.float32)
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    clf, search = D.train_mlp(X[:120], y[:120], X[120:], y[120:], labels,
                              grid=[{"hidden_layer_sizes": (16,), "alpha": 1e-4}])
    assert (clf.predict(X[120:]) == y[120:]).mean() > 0.9 and len(search) == 1


class _FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True):
        self.calls += 1
        return np.array([[len(t), t.count("q")] for t in texts], dtype=np.float32)


def test_split_embeddings_reuse_index_and_cache(labelled_df, tmp_path, monkeypatch):
    from src.preprocessing import split_dataset
    from src.utils import save_json

    monkeypatch.setattr(config, "EMBEDDINGS_DIR", tmp_path)
    train, val, test = split_dataset(labelled_df, seed=0)
    enc = _FakeEncoder()
    # matching dense index -> train/val vectors reused, only the test set is encoded
    tv = pd.concat([train, val], ignore_index=True)
    np.save(tmp_path / "embeddings.npy", enc.encode(tv["abstract"].tolist()))
    tv[["id"]].to_parquet(tmp_path / "documents.parquet")
    save_json({"backend": "dense"}, tmp_path / "index_meta.json")
    enc.calls = 0
    Xtr, Xva, Xte = D.get_split_embeddings(train, val, test, encoder=enc)
    assert (len(Xtr), len(Xva), len(Xte)) == (len(train), len(val), len(test)) and enc.calls == 1
    D.get_split_embeddings(train, val, test, encoder=enc)          # test vectors now cached
    assert enc.calls == 1


def _tiny_model_and_tokenizer(texts, n_labels):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    words = sorted({w.lower().strip(".") for t in texts for w in t.split()})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64,
                           max_position_embeddings=512, num_labels=n_labels)
    torch.manual_seed(0)
    return DistilBertForSequenceClassification(cfg), tok


def test_finetune_loop_end_to_end(labelled_df, monkeypatch):
    from src.preprocessing import split_dataset

    train, val, test = split_dataset(labelled_df, seed=0)
    labels = sorted(labelled_df["label"].unique())
    model, tok = _tiny_model_and_tokenizer(labelled_df["abstract"].tolist(), len(labels))
    monkeypatch.setattr(config, "FORCE_CPU", True)
    # The tiny model starts from RANDOM weights (unlike pre-trained DistilBERT), so it needs
    # more epochs and a higher learning rate to learn the synthetic topics (still < 2 seconds).
    model, tok, history = D.finetune_transformer(train, val, labels, per_class=20, val_size=12, epochs=12,
                                                 batch_size=8, lr=5e-3, max_len=48, model=model, tokenizer=tok)
    assert [h["epoch"] for h in history] == list(range(1, 13))
    assert all(np.isfinite(h["train_loss"]) for h in history)
    assert history[-1]["train_loss"] < history[0]["train_loss"] - 0.3      # the loss really falls
    preds = D.predict_transformer(model, tok, test["abstract"].tolist(), labels, max_len=48)
    assert len(preds) == len(test) and set(preds) <= set(labels)
    assert (np.array(preds) == test["label"].values).mean() > 0.8
