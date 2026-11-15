"""Draws docs/architecture.png (system architecture figure used in the report).
Run from the project root:  python docs/make_architecture_diagram.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#c3c2bd"
FILL = {"data": "#e8f0fb", "model": "#fdeee8", "eval": "#e6f6ef", "app": "#efedfb"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.55, "\n".join(lines), ha="center", va="top", fontsize=8, color=MUTED,
            linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))


fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, 0.2, 3.3, 2.5, 2.6, "1. Data", ["arXiv metadata", "(Kaggle JSON or", "ArXiV Archive TSV)", "streamed, validated", "data_loader.py"], "data")
box(ax, 3.2, 3.3, 2.5, 2.6, "2. Preprocessing", ["cleaning, de-duplication", "primary-category labels", "top-10 classes, cap 5k", "stratified 70/15/15 split", "preprocessing.py + EDA"], "data")
box(ax, 6.2, 4.65, 2.6, 1.25, "3a. Classical ML", ["TF-IDF + NB / LR / SVM", "classification.py"], "model")
box(ax, 6.2, 3.3, 2.6, 1.2, "3b. Title generation", ["FLAN-T5 + 2 baselines", "title_generation.py"], "model")
box(ax, 6.2, 1.95, 2.6, 1.2, "3c. LLM prompting", ["Ollama / API, 3 prompt styles", "llm_*.py"], "model")
box(ax, 6.2, 0.2, 2.6, 1.55, "3d. Retrieval (RAG)", ["MiniLM embeddings", "FAISS / TF-IDF fallback", "rag.py"], "model")
box(ax, 9.3, 3.3, 2.5, 2.6, "4. Evaluation", ["macro-F1, confusion", "ROUGE, similarity", "bootstrap CIs, McNemar", "Recall@k, MRR", "evaluation.py"], "eval")
box(ax, 9.3, 0.2, 2.5, 2.7, "5. Streamlit app", ["6 pages: dashboard,", "classify, titles,", "comparison, explorer,", "grounded chatbot", "app.py + ui/"], "app")
arrow(ax, 2.7, 4.6, 3.2, 4.6)
for y in (5.25, 3.9, 2.55, 0.95):
    arrow(ax, 5.7, 4.6, 6.2, y)
for y in (5.25, 3.9, 2.55):
    arrow(ax, 8.8, y, 9.3, 4.6)
arrow(ax, 10.55, 3.3, 10.55, 2.9)
arrow(ax, 8.8, 0.95, 9.3, 1.3)
ax.text(6, 6.1, "train.py runs stages 1-4 and saves models + results;  the app (5) only reads saved artefacts",
        ha="center", fontsize=9, color=MUTED, style="italic")
out = Path(__file__).resolve().parent / "architecture.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
