"""Matplotlib plots for the media-classifier benchmark.

Consumes the report dict produced by ``benchmarks.run.run()`` and saves PNGs
into ``docs/benchmarks/``:

  * confusion_matrix_<backend>.png  — true vs predicted mediavocab type
  * per_type_f1.png                 — per-type F1 bars (keyword backend, or first available)
  * accuracy_vs_latency.png         — accuracy vs median latency scatter across backends
  * content_filter_recall.png       — adult-slice block recall per backend

Use a non-interactive backend so it works headless.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "benchmarks")

DPI = 140
plt.rcParams.update({
    "figure.autolayout": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "font.size": 10,
})


def _ensure_out() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def _primary_backend(report: Dict) -> Optional[str]:
    """Pick the backend to feature in single-backend plots (prefer keyword)."""
    avail = report.get("available_backends", [])
    if "keyword" in avail:
        return "keyword"
    return avail[0] if avail else None


def plot_confusion(report: Dict, backend: str) -> Optional[str]:
    b = report["backends"].get(backend, {})
    if b.get("status") != "available":
        return None
    labels: List[str] = b["labels"]
    pred_labels: List[str] = b["pred_labels"]
    conf = np.array(b["confusion"], dtype=float)

    # row-normalize for readability (fraction of each true class)
    row_sums = conf.sum(axis=1, keepdims=True)
    norm = np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(pred_labels) + 3),
                                    max(6, 0.5 * len(labels) + 2)))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of true class")

    ax.set_xticks(range(len(pred_labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted media type")
    ax.set_ylabel("true media type")
    ax.set_title(f"Confusion matrix — {backend} backend\n"
                 f"(row-normalized; acc={b['accuracy']:.3f}, macro-F1={b['macro_f1']:.3f})")

    # annotate counts where non-zero
    for i in range(len(labels)):
        for j in range(len(pred_labels)):
            c = int(conf[i, j])
            if c:
                ax.text(j, i, str(c), ha="center", va="center",
                        color="white" if norm[i, j] < 0.6 else "black", fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"confusion_matrix_{backend}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_per_type_f1(report: Dict, backend: str) -> Optional[str]:
    b = report["backends"].get(backend, {})
    if b.get("status") != "available":
        return None
    per_type = b["per_type"]
    types = sorted(per_type, key=lambda t: per_type[t]["f1"])
    f1s = [per_type[t]["f1"] for t in types]
    supports = [per_type[t]["support"] for t in types]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(types) + 3), 6))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in f1s]
    bars = ax.bar(range(len(types)), f1s, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_xlabel("mediavocab media type")
    ax.set_title(f"Per-type F1 — {backend} backend (macro-F1={b['macro_f1']:.3f})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for rect, f1, sup in zip(bars, f1s, supports):
        ax.text(rect.get_x() + rect.get_width() / 2, f1 + 0.015,
                f"{f1:.2f}\nn={sup}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "per_type_f1.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_accuracy_vs_latency(report: Dict) -> Optional[str]:
    avail = report.get("available_backends", [])
    pts = []
    for name in avail:
        b = report["backends"][name]
        pts.append((name, b["latency_ms_median"], b["accuracy"]))
    if not pts:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("plasma")
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    colors = [cmap(i / max(1, len(pts) - 1)) for i in range(len(pts))]
    ax.scatter(xs, ys, s=160, c=colors, edgecolor="black", zorder=3)
    for (name, x, y), c in zip(pts, colors):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("median inference latency (ms/utterance)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    title = "Accuracy vs latency across backends"
    if len(pts) == 1:
        title += "\n(only one backend available in this environment)"
    ax.set_title(title)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "accuracy_vs_latency.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_content_filter_recall(report: Dict) -> Optional[str]:
    avail = report.get("available_backends", [])
    names, recalls, fbr = [], [], []
    for name in avail:
        cf = report["backends"][name].get("content_filter", {})
        names.append(name)
        recalls.append(cf.get("recall", 0.0))
        fbr.append(cf.get("false_block_rate", 0.0))
    if not names:
        return None

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names) + 2), 6))
    x = np.arange(len(names))
    w = 0.38
    b1 = ax.bar(x - w / 2, recalls, w, label="adult-block recall",
                color="#2a9d8f", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, fbr, w, label="false-block rate (non-adult)",
                color="#e76f51", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate")
    ax.set_title("Content-filter behaviour on the adult slice\n(higher recall is better; lower false-block is better)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "content_filter_recall.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def generate_all(report: Dict) -> List[str]:
    _ensure_out()
    made: List[str] = []
    primary = _primary_backend(report)

    # confusion matrix for every available backend
    for name in report.get("available_backends", []):
        p = plot_confusion(report, name)
        if p:
            made.append(p)

    if primary:
        p = plot_per_type_f1(report, primary)
        if p:
            made.append(p)

    for fn in (plot_accuracy_vs_latency, plot_content_filter_recall):
        p = fn(report)
        if p:
            made.append(p)
    return made


if __name__ == "__main__":
    import json
    from benchmarks.run import RESULTS_JSON, run, write_json
    if os.path.isfile(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as fh:
            report = json.load(fh)
    else:
        report = run()
        write_json(report)
    for p in generate_all(report):
        print("wrote", p)
