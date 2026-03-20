#!/usr/bin/env python3
"""Unified OCP Media Classifier dataset pipeline.

Consolidates all dataset generation, processing, and analysis into a single tool.

Subcommands:
  download    Download all external data sources (CSV, HuggingFace)
  gather      Normalize downloaded CSVs → ocp_dataset.csv
  templates   Fill OCP Wikidata templates → ocp_templates.csv
  keyword     Generate keyword-based utterances → ocp_keyword.csv
  synthesize  Generate synthetic utterances via templates + entities → ocp_synthetic.csv
  media       Extract from local media servers (Radarr, Sonarr, etc.) → ocp_media.csv
  merge       Concatenate and deduplicate all CSVs → ocp_final.csv
  metrics     Compute dataset statistics and generate plots
  explore     Analyze dataset composition and generate visualizations
  build       Run complete pipeline (download through metrics)

Usage::

    # Full pipeline (all steps)
    python dataset.py build

    # Individual steps
    python dataset.py download
    python dataset.py gather
    python dataset.py synthesize --langs en-us,de-de,fr-fr --synthetic-n 3000
    python dataset.py merge
    python dataset.py metrics
    python dataset.py explore --input ~/.cache/ovos-media-classifier/output/ocp_final.csv --plots-dir plots/

    # Skip certain steps
    python dataset.py build --skip download --skip gather

    # Multilingual synthesis
    python dataset.py synthesize --langs en-us,de-de,fr-fr,es-es,it-it,pt-br,pt-pt,nl-nl,pl-pl,ca-es,eu,gl-es,da-dk

    # Override defaults
    OVOS_MEDIA_CLASSIFIER_CACHE=/data/ocp python dataset.py build
    RADARR_URL=http://localhost:7878 RADARR_API_KEY=xxx python dataset.py build
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
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
sns.set_theme(style="whitegrid")

from ovos_media_classifier.train import get_output_dir, get_hf_cache_dir
from ovos_media_classifier.train.sources import SCHEMA_COLUMNS

# ============================================================================
# Configuration & Paths
# ============================================================================

OUTPUT_DIR = get_output_dir()
PLOTS_DIR = os.path.join(OUTPUT_DIR, "dataset_plots")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_ROOT = os.path.join(REPO_ROOT, "templates")

_STEP_ORDER = ["download", "gather", "templates", "keyword", "synthesize", "media", "merge", "metrics"]


def _out(name: str) -> str:
    """Get output file path."""
    return os.path.join(OUTPUT_DIR, name)


# ============================================================================
# Step: Download
# ============================================================================

def step_download(args: argparse.Namespace) -> None:
    """Download all external data sources."""
    print("\n[1/8] Downloading datasets …")
    from ovos_media_classifier.train import get_csv_cache_dir
    from ovos_media_classifier.train.sources import ALL_CSV_SOURCES, HF_DATASETS
    from ovos_media_classifier.train.download_datasets import (
        download_csv_sources, download_hf_datasets,
    )

    csv_cache = get_csv_cache_dir()
    hf_cache = get_hf_cache_dir()

    if args.dry_run:
        print(f"  Dry-run mode: would download to {csv_cache} and {hf_cache}")
        print(f"  CSV sources: {len(ALL_CSV_SOURCES)}")
        print(f"  HF datasets: {len(HF_DATASETS)}")
        return

    try:
        download_csv_sources(csv_cache)
        download_hf_datasets(hf_cache)
        print(f"  ✓ Downloaded to {csv_cache} and {hf_cache}")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        raise


# ============================================================================
# Step: Gather
# ============================================================================

def step_gather(args: argparse.Namespace) -> str:
    """Normalize downloaded CSVs."""
    print("\n[2/8] Gathering datasets …")
    from ovos_media_classifier.train.gather_dataset import build_dataset

    df = build_dataset()
    out = _out("ocp_gathered.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df):,} rows → {out}")
    return out


# ============================================================================
# Step: Templates (OCP Wikidata)
# ============================================================================

def step_templates(args: argparse.Namespace, dedup_csv: Optional[str] = None) -> str:
    """Fill OCP Wikidata templates."""
    print("\n[3/8] Filling OCP templates …")
    from ovos_media_classifier.train.generate_from_ocp_templates import generate_all

    df = generate_all(
        n_per_template=args.templates_n,
        dedup_against=dedup_csv,
    )
    out = _out("ocp_templates.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df):,} rows → {out}")
    return out


# ============================================================================
# Step: Keyword
# ============================================================================

def step_keyword(args: argparse.Namespace, dedup_csv: Optional[str] = None) -> str:
    """Generate keyword-based utterances."""
    print("\n[4/8] Generating keyword utterances …")
    from ovos_media_classifier.train.generate_keyword_csv import generate_all

    df = generate_all(
        n=args.keyword_n,
        dedup_against=dedup_csv,
    )
    out = _out("ocp_keyword.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df):,} rows → {out}")
    return out


# ============================================================================
# Step: Synthesize (Multilingual)
# ============================================================================

def load_templates_from_csv(templates_dir: str) -> dict[str, list[tuple[str, list[str]]]]:
    """Load templates from CSV files in templates_dir/{lang}/*.csv."""
    result = {}
    for csv_path in glob.glob(os.path.join(templates_dir, "*.csv")):
        intent = os.path.splitext(os.path.basename(csv_path))[0]
        templates = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tmpl = row.get("template", "").strip() if row else ""
                    if not tmpl:
                        continue
                    slots = re.findall(r'\{(\w+)\}', tmpl.replace('{{', '{').replace('}}', '}'))
                    seen = set()
                    unique_slots = []
                    for s in slots:
                        if s not in seen:
                            seen.add(s)
                            unique_slots.append(s)
                    templates.append((tmpl, unique_slots))
        except Exception as e:
            print(f"  Warning: Failed to load {csv_path}: {e}")
        if templates:
            result[intent] = templates
    return result


def step_synthesize(args: argparse.Namespace, dedup_csv: Optional[str] = None) -> str:
    """Generate synthetic utterances (multilingual)."""
    print("\n[5/8] Generating synthetic utterances …")
    from ovos_media_classifier.train.generate_synthetic import generate_all

    langs = [l.strip() for l in args.langs.split(",")]
    all_dfs = []

    for lang in langs:
        templates_dir = os.path.join(TEMPLATES_ROOT, lang)
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


# ============================================================================
# Step: Media (Local servers)
# ============================================================================

def step_media(args: argparse.Namespace) -> Optional[str]:
    """Extract from local media servers."""
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
        sys.executable,
        os.path.join(os.path.dirname(__file__), "scripts", "generate_dataset_from_media.py"),
        "--output", out,
    ] + cli_args
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode == 0:
        return out
    return None


# ============================================================================
# Step: Merge
# ============================================================================

def step_merge(args: argparse.Namespace, source_csvs: list[str]) -> str:
    """Concatenate and deduplicate CSVs."""
    print("\n[7/8] Merging datasets …")

    dfs = [pd.read_csv(csv) for csv in source_csvs if os.path.exists(csv)]
    if not dfs:
        print("  No datasets to merge!")
        return ""

    df = pd.concat(dfs, ignore_index=True)

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["sentence"], keep="first")
    after = len(df)
    print(f"  Merged {len(source_csvs)} sources: {before:,} → {after:,} rows (deduped {before-after:,})")

    out = _out("ocp_final.csv")
    df.to_csv(out, index=False)
    print(f"  Saved to {out}")
    return out


# ============================================================================
# Step: Metrics
# ============================================================================

def step_metrics(args: argparse.Namespace, merged_csv: Optional[str] = None) -> None:
    """Compute dataset statistics and plots."""
    print("\n[8/8] Computing metrics …")

    if not merged_csv or not os.path.exists(merged_csv):
        merged_csv = _out("ocp_final.csv")

    if not os.path.exists(merged_csv):
        print(f"  Merged CSV not found: {merged_csv}")
        return

    df = pd.read_csv(merged_csv)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Summary stats
    print(f"  Dataset: {len(df):,} rows")
    print(f"  Intents: {df['media_label'].nunique()}")
    if "lang" in df.columns:
        print(f"  Languages: {df['lang'].nunique()}")

    # Basic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Intent distribution (top 15)
    df['media_label'].value_counts().head(15).plot(kind='barh', ax=axes[0, 0])
    axes[0, 0].set_title("Top 15 Intents")
    axes[0, 0].set_xlabel("Count")

    # Domain distribution
    df['domain'].value_counts().plot(kind='bar', ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_title("Domain Distribution")
    axes[0, 1].set_ylabel("Count")

    # Utterance length
    df["sentence"].str.len().hist(bins=50, ax=axes[1, 0], color='coral')
    axes[1, 0].set_title("Utterance Length Distribution")
    axes[1, 0].set_xlabel("Characters")
    axes[1, 0].set_ylabel("Count")

    # Language distribution (if present)
    if "lang" in df.columns:
        df['lang'].value_counts().plot(kind='bar', ax=axes[1, 1], color='green')
        axes[1, 1].set_title("Language Distribution")
        axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "dataset_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plots to {PLOTS_DIR}")
    plt.close()


# ============================================================================
# Step: Explore (Dataset Analysis)
# ============================================================================

def step_explore(args: argparse.Namespace) -> None:
    """Analyze dataset composition and generate visualizations."""
    print("\n[EXPLORE] Analyzing dataset …")

    # Load dataset
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows")

    # Filter by language if requested
    if args.lang and "lang" in df.columns:
        df = df[df["lang"] == args.lang]
        print(f"  Filtered to {len(df):,} rows for language: {args.lang}")

    os.makedirs(args.plots_dir, exist_ok=True)

    # Utterance length analysis
    df["utterance_length"] = df["sentence"].str.len()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Utterance Length Distribution", fontsize=14, fontweight="bold")

    axes[0, 0].hist(df["utterance_length"], bins=50, color="steelblue", edgecolor="black")
    axes[0, 0].set_xlabel("Character Length")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Overall")

    for domain in df["domain"].unique()[:3]:
        data = df[df["domain"] == domain]["utterance_length"]
        axes[0, 1].hist(data, bins=40, alpha=0.6, label=domain)
    axes[0, 1].set_title("By Domain")
    axes[0, 1].legend()

    top_intents = df["media_label"].value_counts().head(10).index
    df_top = df[df["media_label"].isin(top_intents)]
    sns.boxplot(data=df_top, x="media_label", y="utterance_length", ax=axes[1, 0])
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
    axes[1, 0].set_title("By Media Label (Top 10)")

    stats_text = f"""Min: {df['utterance_length'].min()}
Max: {df['utterance_length'].max()}
Mean: {df['utterance_length'].mean():.1f}
Median: {df['utterance_length'].median():.1f}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace")
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.plots_dir, "utterance_length_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()

    # Language coverage heatmap
    if "lang" in df.columns:
        coverage = pd.crosstab(df["lang"], df["media_label"])
        top_intents = df["media_label"].value_counts().head(20).index
        coverage = coverage[top_intents]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(coverage, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Language × Intent Coverage (Top 20)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(args.plots_dir, "lang_intent_coverage.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close()

    # Print text metrics
    print("\n  Dataset Metrics:")
    print(f"    Total rows: {len(df):,}")
    print(f"    Intents: {df['media_label'].nunique()}")
    print(f"    Domains: {', '.join(df['domain'].unique())}")
    if "lang" in df.columns:
        print(f"    Languages: {', '.join(sorted(df['lang'].unique()))}")

    print(f"\n  Top 10 intents:")
    for label, count in df["media_label"].value_counts().head(10).items():
        pct = count / len(df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    # Optional: create train/val/test split
    if args.split_output:
        print(f"\n  Creating stratified split…")
        try:
            from sklearn.model_selection import train_test_split

            os.makedirs(args.split_output, exist_ok=True)
            train, temp = train_test_split(df, test_size=0.2, stratify=df["media_label"], random_state=42)
            val, test = train_test_split(temp, test_size=0.5, stratify=temp["media_label"], random_state=42)

            train.to_csv(os.path.join(args.split_output, "train.csv"), index=False)
            val.to_csv(os.path.join(args.split_output, "val.csv"), index=False)
            test.to_csv(os.path.join(args.split_output, "test.csv"), index=False)

            print(f"    train.csv: {len(train):,}")
            print(f"    val.csv:   {len(val):,}")
            print(f"    test.csv:  {len(test):,}")
        except ImportError:
            print("    sklearn required for splitting; skipped")


# ============================================================================
# Main Command: Build (Run all steps)
# ============================================================================

def step_build(args: argparse.Namespace) -> None:
    """Run complete pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Cache: {os.path.dirname(OUTPUT_DIR)}")
    print(f"Output: {OUTPUT_DIR}")

    only = args.only
    t0 = time.time()
    source_csvs: list[str] = []

    # Step 1: Download
    if not only or only == "download":
        if not args.skip_download:
            step_download(args)

    # Step 2: Gather
    if not only or only == "gather":
        if not args.skip_gather:
            gathered = step_gather(args)
            source_csvs.append(gathered)
        elif os.path.exists(_out("ocp_gathered.csv")):
            source_csvs.append(_out("ocp_gathered.csv"))

    # Step 3: Templates
    if not only or only == "templates":
        if not args.skip_templates:
            dedup = source_csvs[-1] if source_csvs else None
            tmpl = step_templates(args, dedup_csv=dedup)
            source_csvs.append(tmpl)
        elif os.path.exists(_out("ocp_templates.csv")):
            source_csvs.append(_out("ocp_templates.csv"))

    # Step 4: Keyword
    if not only or only == "keyword":
        if not args.skip_keyword:
            dedup = source_csvs[-1] if source_csvs else None
            kw = step_keyword(args, dedup_csv=dedup)
            source_csvs.append(kw)
        elif os.path.exists(_out("ocp_keyword.csv")):
            source_csvs.append(_out("ocp_keyword.csv"))

    # Step 5: Synthesize
    if not only or only == "synthesize":
        if not args.skip_synthetic:
            dedup = source_csvs[-1] if source_csvs else None
            syn = step_synthesize(args, dedup_csv=dedup)
            source_csvs.append(syn)
        elif os.path.exists(_out("ocp_synthetic.csv")):
            source_csvs.append(_out("ocp_synthetic.csv"))

    # Step 6: Media
    if not only or only == "media":
        if not args.skip_media:
            media = step_media(args)
            if media:
                source_csvs.append(media)

    # Step 7: Merge
    if not only or only == "merge":
        merged = step_merge(args, source_csvs)
        source_csvs = [merged] if merged else []

    # Step 8: Metrics
    if not only or only == "metrics":
        step_metrics(args, source_csvs[0] if source_csvs else None)

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")


# ============================================================================
# CLI & Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified OCP dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Common arguments for all subcommands
    def add_common_args(p):
        p.add_argument("--dry-run", action="store_true", help="Dry-run mode")
        p.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace datasets")

    # download
    p_download = subparsers.add_parser("download", help="Download datasets")
    add_common_args(p_download)
    p_download.set_defaults(func=lambda args: step_download(args))

    # gather
    p_gather = subparsers.add_parser("gather", help="Gather & normalize CSVs")
    add_common_args(p_gather)
    p_gather.set_defaults(func=lambda args: (step_gather(args), None)[1])

    # templates
    p_templates = subparsers.add_parser("templates", help="Fill OCP templates")
    add_common_args(p_templates)
    p_templates.add_argument("--templates-n", type=int, default=20, help="Samples per template (default: 20)")
    p_templates.set_defaults(func=lambda args: (step_templates(args), None)[1])

    # keyword
    p_keyword = subparsers.add_parser("keyword", help="Generate keyword utterances")
    add_common_args(p_keyword)
    p_keyword.add_argument("--keyword-n", type=int, default=3000, help="Utterances per intent (default: 3000)")
    p_keyword.set_defaults(func=lambda args: (step_keyword(args), None)[1])

    # synthesize
    p_syn = subparsers.add_parser("synthesize", help="Synthesize utterances (multilingual)")
    add_common_args(p_syn)
    p_syn.add_argument("--langs", default="en-us", help="Comma-separated lang codes (default: en-us)")
    p_syn.add_argument("--synthetic-n", type=int, default=5000, help="Utterances per intent (default: 5000)")
    p_syn.set_defaults(func=lambda args: (step_synthesize(args), None)[1])

    # media
    p_media = subparsers.add_parser("media", help="Extract from media servers")
    add_common_args(p_media)
    p_media.set_defaults(func=lambda args: step_media(args))

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge & deduplicate CSVs")
    add_common_args(p_merge)
    p_merge.set_defaults(func=lambda args: (step_merge(args, []), None)[1])

    # metrics
    p_metrics = subparsers.add_parser("metrics", help="Compute metrics")
    add_common_args(p_metrics)
    p_metrics.set_defaults(func=lambda args: step_metrics(args))

    # explore
    p_explore = subparsers.add_parser("explore", help="Analyze dataset")
    add_common_args(p_explore)
    p_explore.add_argument("--input", required=True, help="Input CSV path")
    p_explore.add_argument("--plots-dir", default="dataset_plots/explore/", help="Output plots directory")
    p_explore.add_argument("--split-output", default=None, help="Train/val/test split output directory")
    p_explore.add_argument("--lang", default=None, help="Filter to single language")
    p_explore.set_defaults(func=lambda args: step_explore(args))

    # build (full pipeline)
    p_build = subparsers.add_parser("build", help="Full pipeline (all steps)")
    add_common_args(p_build)
    p_build.add_argument("--only", choices=_STEP_ORDER, metavar="STEP", help="Run only this step")
    p_build.add_argument("--skip-download", action="store_true")
    p_build.add_argument("--skip-gather", action="store_true")
    p_build.add_argument("--skip-templates", action="store_true")
    p_build.add_argument("--skip-keyword", action="store_true")
    p_build.add_argument("--skip-synthetic", action="store_true")
    p_build.add_argument("--skip-media", action="store_true")
    p_build.add_argument("--templates-n", type=int, default=20)
    p_build.add_argument("--keyword-n", type=int, default=3000)
    p_build.add_argument("--synthetic-n", type=int, default=5000)
    p_build.add_argument("--langs", default="en-us", help="Comma-separated lang codes")
    p_build.set_defaults(func=step_build)

    args = parser.parse_args()

    # Set defaults if no command given
    if not args.command:
        parser.print_help()
        sys.exit(1)

    return args


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.command == "build":
        step_build(args)
    elif hasattr(args, 'func'):
        args.func(args)


if __name__ == "__main__":
    main()
