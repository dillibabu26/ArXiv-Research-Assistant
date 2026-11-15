"""
Deep-learning classifiers, compared with the TF-IDF models on the SAME test set.

1. MiniLM embeddings + MLP ("feature extraction" transfer learning)
   Each abstract is encoded by the pre-trained sentence-transformer all-MiniLM-L6-v2 into a
   384-dimensional vector (the encoder is frozen), and a small feed-forward neural network
   (multi-layer perceptron) is trained on these vectors. Cheap: the embeddings of the
   training/validation papers are reused from the retrieval index when available.

2. Fine-tuned DistilBERT ("fine-tuning" transfer learning)
   DistilBERT (Sanh et al., 2019) is a 66M-parameter distilled version of BERT. A
   classification layer is added on top of the [CLS] representation and ALL weights are
   updated with cross-entropy loss. To fit a CPU budget it is trained on a class-balanced
   subset (DL_PER_CLASS papers per category) for DL_EPOCHS epochs; the epoch with the best
   validation macro-F1 is kept, and the model is evaluated once on the full test set.

Both write predictions_test_<key>.csv in the same format as the classical models, so the
comparison, significance tests and verification treat all models identically.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.evaluation import classification_metrics
from src.utils import get_device, get_logger, load_json, set_seed

logger = get_logger()

DL_MODEL_DIR = config.MODELS_DIR / "deep"


# ---------------------------------------------------------------------------
# 1. Sentence embeddings + MLP
# ---------------------------------------------------------------------------
def encode_abstracts(texts: list[str], encoder=None, batch_size: int = 64) -> np.ndarray:
    """L2-normalised MiniLM embeddings (n, 384)."""
    if encoder is None:
        from src.rag import get_encoder

        encoder = get_encoder()
    return np.asarray(encoder.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                                     normalize_embeddings=True), dtype=np.float32)


def get_split_embeddings(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                         encoder=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embeddings for the three splits. Training and validation vectors are reused from the
    dense retrieval index (built over train+val abstracts in that order) when it matches
    exactly; test vectors are cached in models/embeddings/test_embeddings.npz.
    """
    emb_dir = config.EMBEDDINGS_DIR
    meta = load_json(emb_dir / "index_meta.json") if (emb_dir / "index_meta.json").exists() else {}
    tv_ids = pd.concat([train["id"], val["id"]]).astype(str).tolist()
    Xtv = None
    if meta.get("backend") == "dense" and (emb_dir / "embeddings.npy").exists():
        docs = pd.read_parquet(emb_dir / "documents.parquet", columns=["id"])
        if docs["id"].astype(str).tolist() == tv_ids:
            Xtv = np.load(emb_dir / "embeddings.npy")
            logger.info("Reusing %s train+validation embeddings from the retrieval index", f"{len(Xtv):,}")
    if Xtv is None:
        logger.info("Encoding %s train+validation abstracts (no matching dense index found)", f"{len(tv_ids):,}")
        Xtv = encode_abstracts(pd.concat([train["abstract"], val["abstract"]]).tolist(), encoder)
    cache = emb_dir / "test_embeddings.npz"
    Xte = None
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if z["ids"].tolist() == test["id"].astype(str).tolist():
            Xte = z["X"]
            logger.info("Reusing cached test embeddings")
    if Xte is None:
        logger.info("Encoding %s test abstracts", f"{len(test):,}")
        Xte = encode_abstracts(test["abstract"].tolist(), encoder)
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=Xte, ids=np.array(test["id"].astype(str).tolist()))
    return Xtv[:len(train)], Xtv[len(train):], Xte


MLP_GRID = [{"hidden_layer_sizes": (256,), "alpha": 1e-4},
            {"hidden_layer_sizes": (512,), "alpha": 1e-4},
            {"hidden_layer_sizes": (512,), "alpha": 1e-3},
            {"hidden_layer_sizes": (512, 256), "alpha": 1e-4}]


def train_mlp(Xtr, ytr, Xva, yva, labels, grid=MLP_GRID, seed: int = config.SEED):
    """
    Feed-forward network on frozen embeddings (scikit-learn MLPClassifier: ReLU hidden
    layers, Adam optimiser, softmax output, early stopping on an internal 10% split).
    The architecture/regularisation is chosen by validation macro-F1.
    Returns (best_model, search_table).
    """
    from sklearn.neural_network import MLPClassifier

    rows, best, best_f1 = [], None, -1.0
    for params in grid:
        # Early stopping holds out 10% of the training data; only meaningful with enough of it.
        clf = MLPClassifier(max_iter=200, early_stopping=len(ytr) >= 1000, n_iter_no_change=8,
                            random_state=seed, **params).fit(Xtr, ytr)
        f1 = classification_metrics(yva, clf.predict(Xva), labels)["macro_f1"]
        rows.append({"params": str(params), "val_macro_f1": f1, "epochs": clf.n_iter_})
        logger.info("MLP %s  val macro-F1=%.4f", params, f1)
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Fine-tuned transformer
# ---------------------------------------------------------------------------
def balanced_subset(df: pd.DataFrame, per_class: int, seed: int = config.SEED) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), per_class), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _batches(n: int, size: int, rng: np.random.Generator | None = None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def predict_transformer(model, tokenizer, texts: list[str], labels: list[str],
                        max_len: int = config.DL_MAX_LEN, batch_size: int = 64) -> list[str]:
    """Predict labels with a fine-tuned sequence classifier (no gradient computation)."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    out: list[str] = []
    with torch.no_grad():
        for b in _batches(len(texts), batch_size):
            enc = tokenizer([texts[i] for i in b], truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt").to(device)
            out += [labels[j] for j in model(**enc).logits.argmax(-1).tolist()]
    return out


def finetune_transformer(train: pd.DataFrame, val: pd.DataFrame, labels: list[str],
                         model_name: str = config.DL_TRANSFORMER, per_class: int = config.DL_PER_CLASS,
                         val_size: int = config.DL_VAL_SIZE, epochs: int = config.DL_EPOCHS,
                         batch_size: int = config.DL_BATCH_SIZE, lr: float = config.DL_LR,
                         max_len: int = config.DL_MAX_LEN, seed: int = config.SEED,
                         model=None, tokenizer=None):
    """
    Fine-tune a pre-trained encoder for classification.

    Hyperparameters (standard BERT fine-tuning settings, Devlin et al., 2019):
      lr=5e-5 with AdamW, weight decay 0.01, linear warm-up over 10% of steps then linear
      decay, batch size 16, 2 epochs, inputs truncated to 256 sub-word tokens (covers the
      median abstract of ~190 tokens). Returns (model, tokenizer, history).
    """
    import torch
    from torch.optim import AdamW

    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = get_device(config.FORCE_CPU)
    lab2id = {l: i for i, l in enumerate(labels)}
    if model is None or tokenizer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=lab2id)
    model.to(device)

    tr = balanced_subset(train, per_class, seed)
    va = balanced_subset(val, max(1, val_size // len(labels)), seed)
    texts, y = tr["abstract"].tolist(), torch.tensor([lab2id[l] for l in tr["label"]])
    steps_per_epoch = math.ceil(len(tr) / batch_size)
    total = steps_per_epoch * epochs
    warmup = max(1, int(0.1 * total))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = AdamW(groups, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    rng = np.random.default_rng(seed)
    logger.info("Fine-tuning %s on %s papers (%s per class), %d epochs x %d steps, device=%s",
                model_name, f"{len(tr):,}", per_class, epochs, steps_per_epoch, device)

    history, best_state, best_f1, step, t0 = [], None, -1.0, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for b in _batches(len(tr), batch_size, rng):
            enc = tokenizer([texts[i] for i in b], truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=y[b].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item(); step += 1
            if step == 10 or step % 50 == 0:
                rate = (time.perf_counter() - t0) / step
                logger.info("step %d/%d  loss %.3f  (~%.0f min remaining for training)",
                            step, total, running / ((step - 1) % steps_per_epoch + 1), rate * (total - step) / 60)
        val_pred = predict_transformer(model, tokenizer, va["abstract"].tolist(), labels, max_len)
        f1 = classification_metrics(va["label"], val_pred, labels)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_macro_f1": f1,
                        "minutes": round((time.perf_counter() - t0) / 60, 1)})
        logger.info("epoch %d: mean train loss %.3f, validation macro-F1 %.4f", epoch, history[-1]["train_loss"], f1)
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model, tokenizer, history


def save_transformer(model, tokenizer, out_dir: Path = DL_MODEL_DIR / "distilbert") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
