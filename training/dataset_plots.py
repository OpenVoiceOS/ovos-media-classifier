#!/usr/bin/env python3
"""Regenerate the dataset characterization plots in ``docs/plots/dataset/``.

Four reproducible figures over the built ``data/release`` set:

* ``rows_per_media_type.png``    — class balance across mediavocab leaves.
* ``slot_x_mediatype_heatmap.png`` — which slots fill which media type.
* ``entity_pool_sizes.png``      — entity-pool sizes (log scale).
* ``axis_distributions.png``     — the coarse axes + the ``tags`` namespace mix.

Run it (needs ``[train]`` + matplotlib)::

    python -m training.dataset_plots
    python -m training.dataset_plots --data-dir data/release --out docs/plots/dataset
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)


def _read(data_dir: str) -> pd.DataFrame:
    pq = os.path.join(data_dir, "full.parquet")
    if os.path.isfile(pq):
        return pd.read_parquet(pq)
    return pd.read_csv(os.path.join(data_dir, "full.csv"))


def plot_media_type(df, out):
    vc = df["media_type"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(vc.index[::-1], vc.values[::-1], color="#4c72b0")
    ax.set_title("rows per media_type")
    ax.set_xlabel("rows")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rows_per_media_type.png"), dpi=110)
    plt.close(fig)


def plot_slot_heatmap(df, out):
    ner_cols = [c for c in df.columns if c.startswith("ner_")]
    types = df["media_type"].value_counts().index.tolist()
    mat = []
    for t in types:
        sub = df[df["media_type"] == t]
        mat.append([sub[c].mean() for c in ner_cols])
    fig, ax = plt.subplots(figsize=(max(10, len(ner_cols) * 0.22), 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=7)
    ax.set_xticks(range(len(ner_cols)))
    ax.set_xticklabels([c[4:] for c in ner_cols], rotation=90, fontsize=5)
    ax.set_title("slot (ner_*) fill rate × media_type")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slot_x_mediatype_heatmap.png"), dpi=110)
    plt.close(fig)


def plot_entity_pools(out, entities_dir):
    sizes = {}
    if os.path.isdir(entities_dir):
        for fn in os.listdir(entities_dir):
            if fn.endswith(".csv") and not fn.startswith("_"):
                with open(os.path.join(entities_dir, fn), encoding="utf-8") as fh:
                    sizes[fn[:-4]] = max(sum(1 for _ in fh) - 1, 0)
    if not sizes:
        return
    items = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:40]
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(labels[::-1], vals[::-1], color="#55a868")
    ax.set_xscale("log")
    ax.set_title("entity pool sizes (top 40, log scale)")
    ax.set_xlabel("entities")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "entity_pool_sizes.png"), dpi=110)
    plt.close(fig)


def plot_axes(df, out):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col, title in [
        (axs[0][0], "playback_type", "playback_type"),
        (axs[0][1], "structure", "structure"),
        (axs[1][0], "explicitness", "explicitness"),
    ]:
        vc = df[col].value_counts()
        ax.bar(vc.index, vc.values, color="#c44e52")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    # the tags namespace mix (genre:/mood:/era:)
    ns = Counter()
    for v in df["tags"]:
        for t in json.loads(v) if isinstance(v, str) else (v or []):
            ns[t.split(":")[0]] += 1
    ax = axs[1][1]
    if ns:
        ax.bar(list(ns.keys()), list(ns.values()), color="#8172b3")
    ax.set_title("tags namespace mix (genre / mood / era)")
    fig.suptitle("coarse axis distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "axis_distributions.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "release"))
    ap.add_argument("--entities-dir",
                    default=os.path.join(REPO_ROOT, "data", "entities"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "plots",
                                                  "dataset"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = _read(args.data_dir)
    print(f"loaded {len(df):,} rows")
    plot_media_type(df, args.out)
    plot_slot_heatmap(df, args.out)
    plot_entity_pools(args.out, args.entities_dir)
    plot_axes(df, args.out)
    print(f"wrote 4 plots → {args.out}")


if __name__ == "__main__":
    main()
