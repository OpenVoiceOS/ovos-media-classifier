#!/usr/bin/env python3
"""Exploratory data analysis and visualization for OCP dataset.

Generates plots and metrics to understand dataset composition, balance, and quality.

Usage::

    python scripts/explore_dataset.py --input ~/.cache/ovos-media-classifier/output/ocp_final.csv
    python scripts/explore_dataset.py --input ocp_final.csv --plots-dir plots/ --split-output splits/
    python scripts/explore_dataset.py --input ocp_final.csv --lang de-de  # single language
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and validate dataset CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def analyze_utterance_length(df: pd.DataFrame, plots_dir: str) -> None:
    """Analyze and plot utterance character length distribution."""
    df["utterance_length"] = df["sentence"].str.len()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Utterance Length Distribution by Domain", fontsize=14, fontweight="bold")

    # Overall histogram
    axes[0, 0].hist(df["utterance_length"], bins=50, color="steelblue", edgecolor="black")
    axes[0, 0].set_xlabel("Character Length")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Overall Distribution")

    # By domain
    for domain in df["domain"].unique():
        data = df[df["domain"] == domain]["utterance_length"]
        axes[0, 1].hist(data, bins=40, alpha=0.6, label=domain)
    axes[0, 1].set_xlabel("Character Length")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("By Domain")
    axes[0, 1].legend()

    # By media_label (top 10)
    top_intents = df["media_label"].value_counts().head(10).index
    df_top = df[df["media_label"].isin(top_intents)]
    sns.boxplot(data=df_top, x="media_label", y="utterance_length", ax=axes[1, 0])
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
    axes[1, 0].set_title("By Media Label (Top 10)")
    axes[1, 0].set_ylabel("Character Length")

    # Statistics text
    stats_text = f"""Min: {df['utterance_length'].min()}
Max: {df['utterance_length'].max()}
Mean: {df['utterance_length'].mean():.1f}
Median: {df['utterance_length'].median():.1f}
Std: {df['utterance_length'].std():.1f}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
                    verticalalignment="center")
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "utterance_length_hist.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_token_count(df: pd.DataFrame, plots_dir: str) -> None:
    """Analyze and plot word token count per intent."""
    df["token_count"] = df["sentence"].str.split().str.len()

    top_intents = df["media_label"].value_counts().head(15).index
    df_top = df[df["media_label"].isin(top_intents)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_top, x="media_label", y="token_count", ax=ax, palette="Set2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Token Count Distribution by Media Label (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Media Label")
    ax.set_ylabel("Word Token Count")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "token_count_per_intent.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_lang_intent_coverage(df: pd.DataFrame, plots_dir: str) -> None:
    """Analyze language x intent coverage as a heatmap."""
    if "lang" not in df.columns:
        print("  Warning: No 'lang' column; skipping language coverage analysis")
        return

    # Create pivot table: rows=lang, cols=intent, values=count
    coverage = pd.crosstab(df["lang"], df["media_label"])

    # Filter to top intents by total count
    top_intents = df["media_label"].value_counts().head(20).index
    coverage = coverage[top_intents]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(coverage, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Language × Intent Coverage (Top 20 Intents)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Media Label")
    ax.set_ylabel("Language")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "lang_intent_coverage.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_label_imbalance(df: pd.DataFrame, plots_dir: str) -> None:
    """Analyze per-language intent imbalance (coefficient of variation)."""
    if "lang" not in df.columns:
        print("  Warning: No 'lang' column; skipping imbalance analysis")
        return

    langs = df["lang"].unique()
    cv_per_lang = {}

    for lang in sorted(langs):
        df_lang = df[df["lang"] == lang]
        counts = df_lang["media_label"].value_counts().values
        if len(counts) > 1:
            cv = np.std(counts) / np.mean(counts)
        else:
            cv = 0
        cv_per_lang[lang] = cv

    fig, ax = plt.subplots(figsize=(10, 6))
    langs_sorted = sorted(cv_per_lang.keys())
    cvs = [cv_per_lang[l] for l in langs_sorted]
    bars = ax.bar(langs_sorted, cvs, color="coral", edgecolor="black")
    ax.set_xlabel("Language")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Per-Language Intent Imbalance\n(lower = more balanced)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(langs_sorted, rotation=45, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "label_imbalance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_slot_fill_rate(df: pd.DataFrame, plots_dir: str) -> None:
    """Estimate slot fill rate for synthetic rows."""
    # Heuristic: if sentence contains '{' or '{{', it likely has unfilled slots
    df["has_slots"] = df["sentence"].str.contains(r'\{', regex=True, na=False)

    if df["has_slots"].sum() == 0:
        print("  Info: No unfilled slots detected in dataset")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if "lang" in df.columns:
        slot_status = pd.crosstab(df["lang"], df["has_slots"])
        slot_status_pct = slot_status.div(slot_status.sum(axis=1), axis=0) * 100
        slot_status_pct.plot(kind="bar", ax=ax, color=["green", "red"])
        ax.set_title("Slot Fill Rate by Language\n(red = unfilled slots detected)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Language")
    else:
        has_filled = (~df["has_slots"]).sum()
        has_unfilled = df["has_slots"].sum()
        ax.bar(["Filled Slots", "Unfilled Slots"], [has_filled, has_unfilled], color=["green", "red"])
        ax.set_title("Slot Fill Rate\n(red = unfilled slots detected)", fontsize=14, fontweight="bold")

    ax.set_ylabel("Count")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "slot_fill_rate.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def print_text_metrics(df: pd.DataFrame, filter_lang: Optional[str] = None) -> None:
    """Print text-based metrics to stdout."""
    if filter_lang:
        if "lang" not in df.columns:
            print(f"Warning: Cannot filter by language '{filter_lang}' (no 'lang' column)")
            return
        df = df[df["lang"] == filter_lang]
        print(f"\n=== Metrics for language: {filter_lang} ===")
    else:
        print("\n=== Dataset Metrics ===")

    # Per-language intent coverage
    if "lang" in df.columns:
        print("\nPer-language intent coverage:")
        for lang in sorted(df["lang"].unique()):
            df_lang = df[df["lang"] == lang]
            n_intents = df_lang["media_label"].nunique()
            n_with_samples = (df_lang["media_label"].value_counts() >= 100).sum()
            pct = (n_with_samples / n_intents * 100) if n_intents > 0 else 0
            print(f"  {lang}: {n_intents} intents ({n_with_samples}/{n_intents} have ≥100 samples, {pct:.0f}%)")

    # Duplicate rate
    exact_dupes = df.duplicated(subset=["sentence"]).sum()
    case_norm_dupes = df.duplicated(subset=["sentence"], keep=False).copy()
    case_norm_dupes = case_norm_dupes[df["sentence"].str.lower().duplicated(keep=False)].shape[0]
    print(f"\nDuplicate rate:")
    print(f"  Exact: {exact_dupes:,} ({exact_dupes/len(df)*100:.2f}%)")
    print(f"  Case-normalized: {case_norm_dupes:,} ({case_norm_dupes/len(df)*100:.2f}%)")

    # Utterance length stats
    print(f"\nUtterance length statistics (characters):")
    lengths = df["sentence"].str.len()
    print(f"  Min: {lengths.min()}, Max: {lengths.max()}, Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")

    # Per-intent balance
    print(f"\nTop 10 media labels by count:")
    for label, count in df["media_label"].value_counts().head(10).items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")


def create_train_val_test_split(df: pd.DataFrame, output_dir: str) -> None:
    """Stratified split into train/val/test (80/10/10)."""
    print(f"\nCreating stratified train/val/test split (80/10/10):")
    os.makedirs(output_dir, exist_ok=True)

    # Stratify by media_label
    from sklearn.model_selection import train_test_split

    # First split: 80/20
    train, temp = train_test_split(df, test_size=0.2, stratify=df["media_label"],
                                   random_state=42)
    # Second split: 50/50 of remaining (10/10)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["media_label"],
                                 random_state=42)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"  train.csv: {len(train):,} rows ({len(train)/len(df)*100:.1f}%)")
    print(f"  val.csv:   {len(val):,} rows ({len(val)/len(df)*100:.1f}%)")
    print(f"  test.csv:  {len(test):,} rows ({len(test)/len(df)*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for OCP dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Path to merged CSV dataset")
    parser.add_argument("--plots-dir", default="dataset_plots/explore/",
                        help="Output directory for PNG plots (default: dataset_plots/explore/)")
    parser.add_argument("--split-output", default=None,
                        help="If provided, create stratified train/val/test split in this directory")
    parser.add_argument("--lang", default=None,
                        help="Optional filter for single-language analysis")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.input)

    # Filter by language if requested
    if args.lang:
        if "lang" not in df.columns:
            print(f"Warning: Cannot filter by language (no 'lang' column)")
        else:
            df = df[df["lang"] == args.lang]
            print(f"Filtered to {len(df):,} rows for language: {args.lang}")

    # Create plots directory
    os.makedirs(args.plots_dir, exist_ok=True)
    print(f"\nGenerating plots to {args.plots_dir}:")

    # Generate plots
    analyze_utterance_length(df, args.plots_dir)
    analyze_token_count(df, args.plots_dir)
    analyze_lang_intent_coverage(df, args.plots_dir)
    analyze_label_imbalance(df, args.plots_dir)
    analyze_slot_fill_rate(df, args.plots_dir)

    # Print text metrics
    print_text_metrics(df, filter_lang=args.lang)

    # Create train/val/test split if requested
    if args.split_output:
        create_train_val_test_split(df, args.split_output)

    print("\n✓ Analysis complete")


if __name__ == "__main__":
    main()
