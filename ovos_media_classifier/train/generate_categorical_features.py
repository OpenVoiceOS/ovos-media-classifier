#!/usr/bin/env python3
"""Generate categorical NER features using the REAL OCP inference code.

Feature columns come from two sources that mirror what happens at runtime:

1. **Keyword features** — one binary column per ``*Keyword.voc`` file, using
   the same ``_VocMatcher.match()`` logic as ``KeywordMediaClassifier``.
   Column names: ``kw_music``, ``kw_movie``, ``kw_podcast``, …

2. **NER entity features** — one binary column per ``OCPEntityLabel``, using
   the same ``EntitiesContainer`` + ``AhocorasickNER.tag()`` pipeline as
   ``AhocorasickMediaClassifier``.
   Column names: ``artist_name``, ``movie_title``, ``tv_show_title``, …

Entity data is loaded from ``ocp_entities.csv`` (and optionally from the
HuggingFace entity datasets listed in ``sources.py``).  The same CSV is
used at inference time when running ``AhocorasickMediaClassifier.from_csv()``.

Usage::

    python -m ovos_media_classifier.train.generate_categorical_features \\
        --input  ~/.cache/ovos-media-classifier/output/ocp_final.csv \\
        --output categorical_features.parquet \\
        --entities scripts/ocp_entities.csv \\
        --format parquet \\
        --workers 8

    # Resume from checkpoint if interrupted
    python -m ovos_media_classifier.train.generate_categorical_features ... --resume

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── suppress noisy OVOS boot logs ──────────────────────────────────────────
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from ovos_media_classifier.features import (
    _ENTITY_LABEL_VALUES,
    _KEYWORD_VOCABS,
)
from ovos_media_classifier.keyword import _VocMatcher, _LOCALE_DIR

_REPO_ROOT = Path(__file__).parent.parent.parent

_KW_FEATURE_NAMES = [col for _, col in _KEYWORD_VOCABS]
_ALL_FEATURE_NAMES = _KW_FEATURE_NAMES + _ENTITY_LABEL_VALUES


# ── entity data loading ─────────────────────────────────────────────────────

def load_entity_wordlists(entities_csv: str) -> Dict[str, List[str]]:
    """Parse ocp_entities.csv into {label: [entity_string, …]}.

    Handles both the rich multi-column format produced by the gather step
    (title, ocp_label, media_type, genre, actor, director, …) and the
    simple two-column format (entity, label) accepted by EntitiesContainer.

    Returns:
        Mapping from OCPEntityLabel string to deduplicated list of entity
        strings — ready to pass to EntitiesContainer.from_wordlists().
    """
    df = pd.read_csv(entities_csv, low_memory=False)
    wordlists: Dict[str, set] = {}

    def _add(label: str, value: str) -> None:
        value = str(value).strip()
        if value and value.lower() != "nan":
            wordlists.setdefault(label, set()).add(value.lower())

    if "entity" in df.columns and "label" in df.columns:
        # Simple two-column format
        for _, row in df.iterrows():
            if pd.notna(row["entity"]) and pd.notna(row["label"]):
                _add(str(row["label"]), str(row["entity"]))
    else:
        # Rich multi-column format from ocp_entities.csv
        col_map = {
            "actor":    "movie_actor",
            "director": "movie_director",
            "producer": "movie_producer",
            "writer":   "movie_writer",
            "composer": "movie_composer",
            "artist":   "artist_name",
            "album":    "album_name",
            "author":   "audiobook_author",
            "narrator": "audiobook_narrator",
            "studio":   "movie_studio",
            "genre":    "music_genre",
        }
        for _, row in df.iterrows():
            ocp_label = row.get("ocp_label")
            title     = row.get("title")
            if pd.notna(title) and pd.notna(ocp_label):
                _add(str(ocp_label), str(title))
            for col, label in col_map.items():
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    for part in str(val).split("|"):
                        _add(label, part.strip())

    return {label: list(values) for label, values in wordlists.items()}


# ── worker initialiser (runs once per process in the Pool) ──────────────────

_worker_ner = None       # AhocorasickNER instance (per-process)
_worker_matcher = None   # _VocMatcher instance (per-process)


def _init_worker(wordlists_json: str) -> None:
    """Initialise NER and VocMatcher once per worker process."""
    global _worker_ner, _worker_matcher

    import logging as _logging
    _logging.disable(_logging.WARNING)

    from ovos_media_classifier.entities import EntitiesContainer

    wordlists: Dict[str, List[str]] = json.loads(wordlists_json)
    container = EntitiesContainer()
    for label, words in wordlists.items():
        container.add_many((label, w) for w in words)
    _worker_ner = container.ner  # AhocorasickNER, shared by reference

    _worker_matcher = _VocMatcher(_LOCALE_DIR)


def _extract_batch(rows: List[Tuple[str, str]]) -> List[Dict[str, int]]:
    """Extract features from a batch of (sentence, lang) tuples.

    Uses the worker-local NER and VocMatcher — identical to inference.
    """
    results = []
    for sentence, lang in rows:
        feat: Dict[str, int] = {}

        # ── keyword features (real _VocMatcher, real .voc files) ───────────
        for vocab_name, col_name in _KEYWORD_VOCABS:
            if _worker_matcher.match(sentence, vocab_name, lang):
                feat[col_name] = 1

        # ── NER entity features (real AhocorasickNER.tag) ──────────────────
        try:
            for hit in _worker_ner.tag(sentence):
                label = hit.get("label")
                if label and label in _ENTITY_LABEL_VALUES:
                    feat[label] = 1
        except Exception:
            pass

        results.append(feat)
    return results


# ── checkpoint helpers ──────────────────────────────────────────────────────

def _checkpoint_path(output: str) -> str:
    """Return path for the checkpoint file alongside *output*."""
    p = Path(output)
    return str(p.parent / f".{p.name}.checkpoint.json")


def _load_checkpoint(path: str) -> Optional[dict]:
    """Load a checkpoint file, returning None if absent or corrupt."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            cp = json.load(fh)
        print(f"  Resuming from row {cp['processed_rows']:,} …")
        return cp
    except Exception as exc:
        print(f"  Warning: ignoring corrupt checkpoint ({exc})")
        return None


def _save_checkpoint(path: str, processed: int, features: list, total: int) -> None:
    """Persist checkpoint to *path*."""
    with open(path, "w") as fh:
        json.dump({"processed_rows": processed, "total_rows": total,
                   "features": features}, fh)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: parse args and run feature extraction."""
    parser = argparse.ArgumentParser(
        description="Generate categorical NER features using the real OCP inference stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",    required=True,  help="Input CSV (ocp_final.csv)")
    parser.add_argument("--output",   required=True,  help="Output file path")
    parser.add_argument("--entities", default=None,
                        help="Path to ocp_entities.csv (default: scripts/ocp_entities.csv)")
    parser.add_argument("--format",   default="parquet", choices=["csv", "parquet"])
    parser.add_argument("--sample",   type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--workers",  type=int, default=None, help="Worker count (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--resume",   action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # ── locate entity CSV ───────────────────────────────────────────────────
    entities_csv = args.entities or str(_REPO_ROOT / "scripts" / "ocp_entities.csv")
    if not os.path.exists(entities_csv):
        print(f"ERROR: entity CSV not found: {entities_csv}")
        sys.exit(1)

    # ── load entity pools ───────────────────────────────────────────────────
    print(f"Loading entity pools from {entities_csv} …")
    wordlists = load_entity_wordlists(entities_csv)
    total_entities = sum(len(v) for v in wordlists.values())
    print(f"  {len(wordlists)} labels, {total_entities:,} entities")
    for label, vals in sorted(wordlists.items(), key=lambda kv: -len(kv[1])):
        print(f"    {label:30s}: {len(vals):,}")
    wordlists_json = json.dumps({k: list(v) for k, v in wordlists.items()})

    # ── load input dataset ──────────────────────────────────────────────────
    print(f"\nLoading dataset {args.input} …")
    df = pd.read_csv(args.input, low_memory=False)
    df = df.dropna(subset=["sentence", "intent"])
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)
    print(f"  {len(df):,} rows")

    # ── detect columns to keep ─────────────────────────────────────────────
    keep_cols = [c for c in
                 ["sentence", "intent", "lang", "domain",
                  "binary_label", "media_label", "playback_label"]
                 if c in df.columns]

    # ── resolve lang: most rows use 2-char code (e.g. "en"), VocMatcher
    #    wants "en-us" — normalise to the closest available locale folder ────
    def _normalise_lang(lang: str) -> str:
        lang = str(lang).lower().strip()
        # Try exact first, then 2-char, then fallback to en
        import os as _os
        for candidate in [lang, lang.split("-")[0], "en-us", "en"]:
            if _os.path.isdir(os.path.join(_LOCALE_DIR, candidate)):
                return candidate
        return "en"

    if "lang" in df.columns:
        df["_lang_norm"] = df["lang"].apply(_normalise_lang)
    else:
        df["_lang_norm"] = "en"

    rows = list(zip(df["sentence"].astype(str), df["_lang_norm"].astype(str)))
    total = len(rows)

    # ── checkpoint / resume ─────────────────────────────────────────────────
    checkpoint_path = _checkpoint_path(args.output)
    all_features: List[dict] = []
    skip = 0

    if args.resume:
        cp = _load_checkpoint(checkpoint_path)
        if cp:
            all_features = cp["features"]
            skip = cp["processed_rows"]
            rows = rows[skip:]

    # ── parallel feature extraction ─────────────────────────────────────────
    num_workers = args.workers or cpu_count()
    batch_size  = args.batch_size
    print(f"\nExtracting features: {len(rows):,} rows, {num_workers} workers, "
          f"batch_size={batch_size}")
    print(f"  NER entity labels : {len(_ENTITY_LABEL_VALUES)}")
    print(f"  Keyword voc labels: {len(_KEYWORD_VOCABS)}")
    print(f"  Total features    : {len(_ALL_FEATURE_NAMES)}")

    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(wordlists_json,)) as pool:

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_feats = pool.apply(_extract_batch, (batch,))
            all_features.extend(batch_feats)

            processed = skip + i + len(batch)
            if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(rows):
                pct = processed / total * 100
                print(f"  {processed:,}/{total:,}  ({pct:.1f}%)")
                _save_checkpoint(checkpoint_path, processed, all_features, total)

    print(f"Done — {len(all_features):,} rows processed")

    # ── assemble result DataFrame ───────────────────────────────────────────
    feat_df = pd.DataFrame(all_features, columns=_ALL_FEATURE_NAMES).fillna(0).astype("int8")
    result  = pd.concat(
        [df[keep_cols].reset_index(drop=True), feat_df],
        axis=1,
    )

    # ── save ────────────────────────────────────────────────────────────────
    print(f"\nSaving to {args.output} …")
    if args.format == "parquet":
        result.to_parquet(args.output, compression="snappy", index=False)
    else:
        result.to_csv(args.output, index=False)

    # clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"✓  {len(result):,} rows × {len(result.columns)} columns")

    # ── coverage report ─────────────────────────────────────────────────────
    print("\nKeyword feature coverage:")
    for col in _KW_FEATURE_NAMES:
        if col in result.columns:
            n   = int(result[col].sum())
            pct = n / len(result) * 100
            print(f"  {col:30s}: {n:8,d}  ({pct:5.1f}%)")

    print("\nNER entity feature coverage (non-zero only):")
    for col in _ENTITY_LABEL_VALUES:
        if col in result.columns:
            n = int(result[col].sum())
            if n:
                pct = n / len(result) * 100
                print(f"  {col:30s}: {n:8,d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
