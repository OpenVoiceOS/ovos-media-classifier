#!/usr/bin/env python3
"""Master dataset build script for ovos-media-classifier.

Orchestrates all dataset generation steps in order:

  1. download   — pre-download all CSV + HuggingFace sources to local cache
  2. gather     — normalise downloaded CSVs → ocp_dataset.csv
  3. templates  — fill OCP Wikidata templates → ocp_templates.csv
  4. keyword    — offline keyword-based utterances → ocp_keyword.csv
  5. synthetic  — template + HF entity generation → ocp_synthetic.csv
  6. merge      — concatenate all CSVs, dedup → ocp_final.csv
  7. metrics    — compute per-intent / per-lang counts + save plots

Each step writes to the configured output directory.  Steps can be skipped
individually with --skip-* flags or selected with --only.

Usage::

    # Full build (downloads everything)
    uv run python build_dataset.py

    # Quick offline build — no network after initial download
    uv run python build_dataset.py --skip-download

    # Produce only metrics/plots for an existing ocp_final.csv
    uv run python build_dataset.py --only metrics

    # Override cache location
    OVOS_MEDIA_CLASSIFIER_CACHE=/data/ocp uv run python build_dataset.py

    # Also pull from a local media server (env vars)
    RADARR_URL=http://... RADARR_API_KEY=... uv run python build_dataset.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

from ovos_media_classifier.train import get_output_dir, get_hf_cache_dir
from ovos_media_classifier.train.sources import SCHEMA_COLUMNS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = get_output_dir()
PLOTS_DIR = os.path.join(OUTPUT_DIR, "dataset_plots")

_STEP_ORDER = ["download", "gather", "templates", "keyword", "synthetic", "media", "merge", "metrics"]


def _out(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def step_download(args: argparse.Namespace) -> None:
    print("\n[1/8] Downloading datasets …")
    from ovos_media_classifier.train import get_csv_cache_dir, get_hf_cache_dir as _hf
    from ovos_media_classifier.train.sources import ALL_CSV_SOURCES, HF_DATASETS
    from ovos_media_classifier.train.download_datasets import (
        download_csv_sources, download_hf_datasets,
    )
    csv_cache = get_csv_cache_dir()
    hf_cache = _hf()
    download_csv_sources(ALL_CSV_SOURCES, csv_cache, dry_run=args.dry_run)
    download_hf_datasets(HF_DATASETS, hf_cache, dry_run=args.dry_run)
    print("  Download complete.")


def step_gather(args: argparse.Namespace) -> str:
    """Gather + normalise CSV sources → ocp_gathered.csv."""
    print("\n[2/8] Gathering and normalising CSVs …")
    from ovos_media_classifier.train.gather_dataset import build_dataset
    df = build_dataset()
    out = _out("ocp_gathered.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df):,} rows → {out}")
    return out


def step_templates(args: argparse.Namespace, dedup_csv: Optional[str] = None) -> str:
    """Fill OCP Wikidata templates → ocp_templates.csv."""
    print("\n[3/8] Generating from OCP templates …")
    from ovos_media_classifier.train.generate_from_ocp_templates import generate_all
    df = generate_all(
        n_per_template=args.templates_n,
        dedup_against=dedup_csv,
    )
    out = _out("ocp_templates.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df):,} rows → {out}")
    return out


def step_keyword(args: argparse.Namespace, dedup_csv: Optional[str] = None) -> str:
    """Offline keyword-based utterances → ocp_keyword.csv."""
    print("\n[4/8] Generating keyword utterances …")
    from ovos_media_classifier.train.generate_keyword_csv import generate_all
    df = generate_all(n=args.keyword_n, dedup_against=dedup_csv)
    out = _out("ocp_keyword.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df):,} rows → {out}")
    return out


def step_synthetic(args: argparse.Namespace, dedup_csv: Optional[str] = None) -> str:
    """Template + HF entity generation → ocp_synthetic.csv (multilingual)."""
    print("\n[5/8] Generating synthetic utterances …")
    from ovos_media_classifier.train.generate_synthetic import generate_all

    # Determine the templates root directory (one level up from scripts/)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(scripts_dir)
    templates_root = os.path.join(repo_root, "templates")

    # Parse languages
    langs = [l.strip() for l in args.langs.split(",")]
    all_dfs = []

    for lang in langs:
        templates_dir = os.path.join(templates_root, lang)
        print(f"  {lang} … ", end="", flush=True)
        df = generate_all(
            max_per_intent=args.synthetic_n,
            skip_hf=args.skip_hf,
            dedup_against=dedup_csv,
            lang=lang,
            templates_dir=templates_dir,
        )
        print(f"{len(df):,} rows")
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    out = _out("ocp_synthetic.csv")
    result.to_csv(out, index=False)
    print(f"  Total: {len(result):,} rows → {out}")
    return out


def step_media(args: argparse.Namespace) -> Optional[str]:
    """Pull from local media servers if configured → ocp_media.csv."""
    env_vars = {
        "RADARR_URL": ("--radarr-url", "RADARR_API_KEY", "--radarr-api-key"),
        "SONARR_URL": ("--sonarr-url", "SONARR_API_KEY", "--sonarr-api-key"),
        "LIDARR_URL": ("--lidarr-url", "LIDARR_API_KEY", "--lidarr-api-key"),
        "JELLYFIN_URL": ("--jellyfin-url", "JELLYFIN_API_KEY", "--jellyfin-api-key"),
        "MUSIC_ASSISTANT_URL": ("--music-assistant-url", None, None),
        "AUDIOBOOKSHELF_URL": ("--audiobookshelf-url", "AUDIOBOOKSHELF_API_KEY", "--audiobookshelf-api-key"),
        "PODGRAB_URL": ("--podgrab-url", None, None),
        "KAPOWARR_URL": ("--kapowarr-url", "KAPOWARR_API_KEY", "--kapowarr-api-key"),
    }
    cli_args = []
    for url_var, (url_flag, key_var, key_flag) in env_vars.items():
        url = os.environ.get(url_var, "")
        if url:
            cli_args += [url_flag, url]
            if key_var:
                key = os.environ.get(key_var, "")
                if key:
                    cli_args += [key_flag, key]

    if not cli_args:
        print("\n[6/8] Skipping media server step (no *_URL env vars set).")
        print("  Set RADARR_URL, SONARR_URL, LIDARR_URL, JELLYFIN_URL, etc. to enable.")
        return None

    print("\n[6/8] Pulling from local media servers …")
    out = _out("ocp_media.csv")
    import subprocess
    cmd = [
        sys.executable, "-m", "ovos_media_classifier.train.generate_dataset_from_media",
        "--output", out,
    ] + cli_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  Warning: media server step failed; continuing without it.")
        return None
    if os.path.exists(out):
        n = len(pd.read_csv(out))
        print(f"  {n:,} rows → {out}")
        return out
    return None


def step_merge(source_csvs: list[str]) -> str:
    """Concatenate all partial CSVs, dedup on sentence → ocp_final.csv."""
    print("\n[7/8] Merging all sources …")
    frames: list[pd.DataFrame] = []
    for path in source_csvs:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure all schema columns present
            for col in SCHEMA_COLUMNS:
                if col not in df.columns:
                    df[col] = "undefined" if "label" in col else ""
            frames.append(df[SCHEMA_COLUMNS])
            print(f"  + {len(df):>8,}  {os.path.basename(path)}")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged.drop_duplicates(subset=["sentence"], inplace=True)
    after = len(merged)
    print(f"  Deduped {before - after:,} duplicates → {after:,} unique rows")

    out = _out("ocp_final.csv")
    merged.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return out


def step_metrics(final_csv: str) -> None:
    """Compute and display counts; save plots to dataset_plots/."""
    print("\n[8/8] Computing metrics and plots …")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(final_csv)
    total = len(df)

    print(f"\n  Total rows: {total:,}")
    print(f"  Unique sentences: {df['sentence'].nunique():,}")
    print(f"  Languages: {sorted(df['lang'].unique())}")

    # ── Domain distribution ──────────────────────────────────────────────────
    print("\n  Domain distribution:")
    for domain, cnt in df["domain"].value_counts().items():
        print(f"    {domain:<20}  {cnt:>8,}  ({100*cnt/total:.1f}%)")

    # ── Media intent distribution (ocp_play only) ────────────────────────────
    play_df = df[df["domain"] == "ocp_play"]
    print(f"\n  Play intents ({len(play_df):,} rows):")
    intent_counts = play_df["media_label"].value_counts()
    for intent, cnt in intent_counts.items():
        bar = "█" * int(40 * cnt / intent_counts.max())
        print(f"    {intent:<25}  {cnt:>6,}  {bar}")

    # ── Language coverage ────────────────────────────────────────────────────
    print(f"\n  Language coverage:")
    for lang, cnt in df["lang"].value_counts().items():
        print(f"    {lang:<8}  {cnt:>8,}  ({100*cnt/total:.1f}%)")

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_intent_distribution(play_df, PLOTS_DIR)
    _plot_domain_pie(df, PLOTS_DIR)
    _plot_lang_heatmap(df, PLOTS_DIR)
    _plot_playback_balance(play_df, PLOTS_DIR)
    print(f"\n  Plots saved to {PLOTS_DIR}/")


def _plot_intent_distribution(play_df: pd.DataFrame, plots_dir: str) -> None:
    counts = play_df["media_label"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.35)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of utterances")
    ax.set_title("OCP Play Intent Distribution")
    for i, (_, v) in enumerate(counts.items()):
        ax.text(v + counts.max() * 0.005, i, f"{v:,}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "intent_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_domain_pie(df: pd.DataFrame, plots_dir: str) -> None:
    counts = df["domain"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=["steelblue", "darkorange", "grey"])
    ax.set_title("Domain Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "domain_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_lang_heatmap(df: pd.DataFrame, plots_dir: str) -> None:
    langs = df["lang"].unique()
    intents = df["media_label"].unique()
    matrix = pd.crosstab(df["lang"], df["media_label"]).reindex(
        index=sorted(langs), columns=sorted(intents), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(max(12, len(intents) * 0.6), max(6, len(langs) * 0.5)))
    sns.heatmap(
        matrix, ax=ax, cmap="YlOrRd", fmt="d", annot=True if len(intents) * len(langs) < 300 else False,
        linewidths=0.5, cbar_kws={"label": "count"},
    )
    ax.set_title("Utterances per Language × Intent")
    ax.set_xlabel("Intent")
    ax.set_ylabel("Language")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_intent_heatmap.png"), dpi=150)
    plt.close(fig)


def _plot_playback_balance(play_df: pd.DataFrame, plots_dir: str) -> None:
    counts = play_df["playback_label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["steelblue", "darkorange", "grey"])
    for i, (_, v) in enumerate(counts.items()):
        ax.text(i, v + counts.max() * 0.01, f"{v:,}", ha="center")
    ax.set_title("Playback Type Balance (audio / video / undefined)")
    ax.set_ylabel("Utterances")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "playback_balance.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Master OCP dataset build pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--only", choices=_STEP_ORDER, metavar="STEP",
                   help="Run only this one step (skip all others)")
    p.add_argument("--skip-download",   action="store_true", help="Skip step 1 (download)")
    p.add_argument("--skip-gather",     action="store_true", help="Skip step 2 (gather)")
    p.add_argument("--skip-templates",  action="store_true", help="Skip step 3 (templates)")
    p.add_argument("--skip-keyword",    action="store_true", help="Skip step 4 (keyword)")
    p.add_argument("--skip-synthetic",  action="store_true", help="Skip step 5 (synthetic)")
    p.add_argument("--skip-media",      action="store_true", help="Skip step 6 (media servers)")
    p.add_argument("--skip-hf",         action="store_true",
                   help="In synthetic step: skip HuggingFace datasets, use curated lists only")
    p.add_argument("--dry-run",         action="store_true", help="In download step: list files only")
    p.add_argument("--templates-n",     type=int, default=20,
                   help="Samples per Wikidata template (default: 20)")
    p.add_argument("--keyword-n",       type=int, default=3000,
                   help="Utterances per keyword intent (default: 3000)")
    p.add_argument("--synthetic-n",     type=int, default=5000,
                   help="Utterances per synthetic intent (default: 5000)")
    p.add_argument("--langs",           default="en-us",
                   help="Comma-separated BCP-47 lang codes for synthetic generation (default: en-us)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Cache: {os.path.dirname(OUTPUT_DIR)}")
    print(f"Output: {OUTPUT_DIR}")

    only = args.only
    t0 = time.time()
    source_csvs: list[str] = []

    # ── Step 1: Download ─────────────────────────────────────────────────────
    if not only or only == "download":
        if not args.skip_download:
            step_download(args)

    # ── Step 2: Gather ───────────────────────────────────────────────────────
    if not only or only == "gather":
        if not args.skip_gather:
            gathered = step_gather(args)
            source_csvs.append(gathered)
        elif os.path.exists(_out("ocp_gathered.csv")):
            source_csvs.append(_out("ocp_gathered.csv"))

    # ── Step 3: Templates ────────────────────────────────────────────────────
    if not only or only == "templates":
        if not args.skip_templates:
            dedup = source_csvs[-1] if source_csvs else None
            tmpl = step_templates(args, dedup_csv=dedup)
            source_csvs.append(tmpl)
        elif os.path.exists(_out("ocp_templates.csv")):
            source_csvs.append(_out("ocp_templates.csv"))

    # ── Step 4: Keyword ──────────────────────────────────────────────────────
    if not only or only == "keyword":
        if not args.skip_keyword:
            dedup = source_csvs[-1] if source_csvs else None
            kw = step_keyword(args, dedup_csv=dedup)
            source_csvs.append(kw)
        elif os.path.exists(_out("ocp_keyword.csv")):
            source_csvs.append(_out("ocp_keyword.csv"))

    # ── Step 5: Synthetic ────────────────────────────────────────────────────
    if not only or only == "synthetic":
        if not args.skip_synthetic:
            dedup = source_csvs[-1] if source_csvs else None
            syn = step_synthetic(args, dedup_csv=dedup)
            source_csvs.append(syn)
        elif os.path.exists(_out("ocp_synthetic.csv")):
            source_csvs.append(_out("ocp_synthetic.csv"))

    # ── Step 6: Media servers ────────────────────────────────────────────────
    if not only or only == "media":
        if not args.skip_media:
            media = step_media(args)
            if media:
                source_csvs.append(media)
        elif os.path.exists(_out("ocp_media.csv")):
            source_csvs.append(_out("ocp_media.csv"))

    # ── Step 7: Merge ────────────────────────────────────────────────────────
    final_csv = _out("ocp_final.csv")
    if not only or only == "merge":
        if source_csvs:
            final_csv = step_merge(source_csvs)
        else:
            print("\n[7/8] No sources to merge — using existing ocp_final.csv")

    # ── Step 8: Metrics ──────────────────────────────────────────────────────
    if not only or only == "metrics":
        if os.path.exists(final_csv):
            step_metrics(final_csv)
        else:
            print(f"\n[8/8] Skipping metrics — {final_csv} not found")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
