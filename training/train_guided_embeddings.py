#!/usr/bin/env python3
"""Train guided-categorical-embeddings models for OCP media classification.

Reads ``categorical_features.parquet`` (produced by
``generate_categorical_features.py``) and trains two ONNX models:

  output/domain/ — OCPDomain classifier (ocp_play / ocp_control / not_ocp)
  output/play/   — media-label classifier (music / movie / podcast / …)

Usage::

    python scripts/train_guided_embeddings.py \\
        --input ~/.cache/ovos-media-classifier/output/categorical_features.parquet \\
        --output ~/.cache/ovos-media-classifier/models/guided \\
        --lang en \\
        --music-cap 10000 \\
        --max-domain-rows 200000

Requires: ``pip install ovos-media-classifier[train]``
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def _build_domain_labels(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[str]]:
    """Map parquet rows to (X_dicts, y_domain) for domain model training.

    binary_label values:
      "ocp"  + media rows → "ocp_play"
      "not_ocp"           → "not_ocp"
      "ocp_control"       → "ocp_control"

    Args:
        df: DataFrame with binary_label column and 84 binary feature columns.

    Returns:
        (X_dicts, y) where X_dicts are sparse feature dicts.
    """
    feat_cols = [c for c in df.columns if c not in (
        "binary_label", "media_label", "utterance", "lang"
    )]

    def _map(label: str) -> str:
        if label in ("ocp", "media"):
            return "ocp_play"
        return label  # "not_ocp", "ocp_control"

    y = [_map(v) for v in df["binary_label"]]
    X = [
        {col: "1" for col in feat_cols if row[col] == 1}
        for _, row in df.iterrows()
    ]
    return X, y


def _build_play_labels(
    df: pd.DataFrame,
    music_cap: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Map parquet rows to (X_dicts, y_play) for play-intent model training.

    Filters to ocp_play domain rows, drops classes with <10 samples,
    caps music at *music_cap* to reduce class imbalance.

    Args:
        df: Full DataFrame.
        music_cap: Maximum number of music rows to include.

    Returns:
        (X_dicts, y) where X_dicts are sparse feature dicts.
    """
    ocp_df = df[df["binary_label"].isin(("ocp", "media"))].copy()
    feat_cols = [c for c in df.columns if c not in (
        "binary_label", "media_label", "utterance", "lang"
    )]

    # Drop classes with too few samples
    counts = ocp_df["media_label"].value_counts()
    valid_labels = counts[counts >= 10].index.tolist()
    ocp_df = ocp_df[ocp_df["media_label"].isin(valid_labels)]

    # Cap music
    music_rows = ocp_df[ocp_df["media_label"] == "music"]
    other_rows = ocp_df[ocp_df["media_label"] != "music"]
    if len(music_rows) > music_cap:
        music_rows = music_rows.sample(n=music_cap, random_state=42)
    ocp_df = pd.concat([music_rows, other_rows]).sample(frac=1, random_state=42)

    y = list(ocp_df["media_label"])
    X = [
        {col: "1" for col in feat_cols if row[col] == 1}
        for _, row in ocp_df.iterrows()
    ]
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    input_path: str,
    output_dir: str,
    lang: str,
    music_cap: int,
    max_domain_rows: int,
    hidden_sizes: List[int],
    embedding_size: int,
    seed: int,
) -> None:
    """Train domain and play ONNX models from a parquet feature file.

    Args:
        input_path: Path to categorical_features.parquet.
        output_dir: Root directory for ONNX exports (domain/ and play/).
        lang: Language filter; use "all" to skip filtering.
        music_cap: Max music rows in play-head training.
        max_domain_rows: Max total rows for domain-head training.
        hidden_sizes: MLP hidden layer sizes.
        embedding_size: PCA embedding dimension.
        seed: Random seed for reproducibility.
    """
    try:
        from guided_categorical_embeddings.training.trainer import LabelGuidedTrainer
    except ImportError:
        _LOG.error(
            "guided-categorical-embeddings[train] is required. "
            "Install with: pip install ovos-media-classifier[train]"
        )
        sys.exit(1)

    _LOG.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)

    if lang != "all" and "lang" in df.columns:
        df = df[df["lang"].str.startswith(lang)]
        _LOG.info(f"Filtered to lang={lang!r}: {len(df)} rows")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Domain model ──────────────────────────────────────────────────
    _LOG.info("Building domain training set")
    domain_df = df.copy()
    ocp_play_count = (domain_df["binary_label"].isin(("ocp", "media"))).sum()
    # Cap ocp_play at half of max_domain_rows; keep minority classes
    cap = max_domain_rows // 2
    if ocp_play_count > cap:
        play_mask = domain_df["binary_label"].isin(("ocp", "media"))
        domain_df = pd.concat([
            domain_df[play_mask].sample(n=cap, random_state=seed),
            domain_df[~play_mask],
        ])
    if len(domain_df) > max_domain_rows:
        domain_df = domain_df.sample(n=max_domain_rows, random_state=seed)

    X_domain, y_domain = _build_domain_labels(domain_df)
    _LOG.info(f"Domain training: {len(X_domain)} samples, "
              f"classes={set(y_domain)}")

    domain_trainer = LabelGuidedTrainer(
        hidden_sizes=hidden_sizes,
        embedding_size=embedding_size,
        seed=seed,
    )
    domain_trainer.fit(X_domain, y_domain)
    domain_out = output_path / "domain"
    domain_trainer.export(str(domain_out))
    _LOG.info(f"Domain model exported to {domain_out}")

    # ── Play model ────────────────────────────────────────────────────
    _LOG.info("Building play training set")
    X_play, y_play = _build_play_labels(df, music_cap=music_cap)
    _LOG.info(f"Play training: {len(X_play)} samples, "
              f"classes={set(y_play)}")

    play_trainer = LabelGuidedTrainer(
        hidden_sizes=hidden_sizes,
        embedding_size=embedding_size,
        seed=seed,
    )
    play_trainer.fit(X_play, y_play)
    play_out = output_path / "play"
    play_trainer.export(str(play_out))
    _LOG.info(f"Play model exported to {play_out}")

    _LOG.info(f"Done. Models in {output_path}/domain/ and {output_path}/play/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for training CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train guided-categorical-embeddings OCP models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to categorical_features.parquet",
    )
    parser.add_argument(
        "--output", required=True,
        help="Root output directory (domain/ and play/ created inside)",
    )
    parser.add_argument("--lang", default="en",
                        help="Language filter prefix (or 'all' to skip)")
    parser.add_argument("--music-cap", type=int, default=10_000,
                        help="Max music rows in play-head training data")
    parser.add_argument("--max-domain-rows", type=int, default=200_000,
                        help="Max total rows for domain-head training")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[256, 128, 64],
                        help="MLP hidden layer sizes")
    parser.add_argument("--embedding-size", type=int, default=48,
                        help="PCA embedding dimension")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    train(
        input_path=args.input,
        output_dir=args.output,
        lang=args.lang,
        music_cap=args.music_cap,
        max_domain_rows=args.max_domain_rows,
        hidden_sizes=args.hidden_sizes,
        embedding_size=args.embedding_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
