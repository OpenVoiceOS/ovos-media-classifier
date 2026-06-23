"""Merge all dataset layers, enforce the mediavocab taxonomy, split, and publish.

This is the one-shot wrapper that turns the partial CSVs produced by the various
generators (templates / slot-fill / agentpipe / gathered intent CSVs) into the
**canonical** ``ocp-media-intents`` dataset that future classifiers train and
benchmark on.

Pipeline:
  1. Concatenate every input CSV (schema = ``training.sources.SCHEMA_COLUMNS``),
     dedup on ``sentence``.
  2. **Enforce taxonomy** — add two columns derived from the fine-grained
     ``media_label`` (an ``OCPPlayIntent`` value):
       * ``mediavocab_type``  via ``PLAY_INTENT_TO_MEDIA_TYPE`` (``not_ocp`` →
         ``not_media``);
       * ``genres``           via ``PLAY_INTENT_TO_GENRES`` (carries the ``adult``
         content-filter signal), serialized as ``;``-joined tags.
  3. Stratified **train/validation/test** split (80/10/10, ``random_state=42``,
     stratified on ``mediavocab_type``).
  4. Optionally ``push_to_hub`` as a ``DatasetDict`` (private) with a dataset card.

Usage::

    python -m training.build_and_publish \
        --inputs /tmp/omc_ds/output/ocp_slot_filled.csv \
                 /tmp/omc_ds/output/ocp_agentpipe.csv \
                 /tmp/omc_ds/output/ocp_gathered.csv \
        --out-dir /tmp/omc_ds/release \
        --push --repo TigreGotico/ocp-media-intents --private
"""
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

from training.sources import SCHEMA_COLUMNS
from ovos_media_classifier.intents import (
    MediaType,
    OCPPlayIntent,
    PLAY_INTENT_TO_MEDIA_TYPE,
    PLAY_INTENT_TO_GENRES,
)

OUT_COLUMNS = SCHEMA_COLUMNS + ["mediavocab_type", "genres"]

# normalize bare language codes to BCP-47 region forms for a consistent dataset
_LANG_NORM = {
    "en": "en-us", "pt": "pt-pt", "es": "es-es", "fr": "fr-fr", "de": "de-de",
    "it": "it-it", "nl": "nl-nl", "ca": "ca-es", "gl": "gl-es", "da": "da-dk",
    "eu": "eu-es", "pl": "pl-pl",
}


def _intent_to_mvtype(media_label: str) -> str:
    if media_label in ("not_ocp", "", None):
        return MediaType.NOT_MEDIA.value
    try:
        intent = OCPPlayIntent(media_label)
    except ValueError:
        return MediaType.GENERIC.value
    return PLAY_INTENT_TO_MEDIA_TYPE.get(intent, MediaType.GENERIC).value


def _intent_to_genres(media_label: str) -> str:
    try:
        intent = OCPPlayIntent(media_label)
    except ValueError:
        return ""
    return ";".join(PLAY_INTENT_TO_GENRES.get(intent, []))


def load_and_merge(inputs: List[str]) -> pd.DataFrame:
    frames = []
    for path in inputs:
        if not os.path.isfile(path):
            print(f"  skip (missing): {path}")
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
        if missing:
            print(f"  skip (cols {missing} missing): {path}")
            continue
        frames.append(df[SCHEMA_COLUMNS])
        print(f"  + {len(df):>8,} rows  {path}")
    if not frames:
        raise SystemExit("no valid input CSVs")
    merged = pd.concat(frames, ignore_index=True)
    merged["sentence"] = merged["sentence"].astype(str).str.strip()
    merged = merged[merged["sentence"].str.len() > 0]
    before = len(merged)
    merged.drop_duplicates(subset=["sentence"], inplace=True)
    print(f"  merged {before:,} → {len(merged):,} unique sentences")
    return merged


def enforce_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize casing that varies between generators (e.g. binary_label OCP/ocp)
    for col in ("domain", "binary_label", "playback_label", "media_label", "intent"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    # normalize bare language codes to BCP-47 region forms
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    df["lang"] = df["lang"].map(lambda l: _LANG_NORM.get(l, l))
    df["mediavocab_type"] = df["media_label"].map(_intent_to_mvtype)
    df["genres"] = df["media_label"].map(_intent_to_genres)
    return df[OUT_COLUMNS]


def cap_per_type(df: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """Downsample each ``mediavocab_type`` to at most *cap* rows (balances the
    heavily music-dominant raw mix into a usable classifier dataset)."""
    parts = []
    for mt, grp in df.groupby("mediavocab_type"):
        if len(grp) > cap:
            grp = grp.sample(n=cap, random_state=seed)
        parts.append(grp)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42):
    from sklearn.model_selection import train_test_split
    # stratify on mediavocab_type; fold singleton classes into train to avoid errors
    counts = df["mediavocab_type"].value_counts()
    rare = counts[counts < 10].index
    strat = df["mediavocab_type"].where(~df["mediavocab_type"].isin(rare), "._rare_")
    train, temp = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)
    strat_temp = temp["mediavocab_type"].where(
        ~temp["mediavocab_type"].isin(rare), "._rare_")
    val, test = train_test_split(temp, test_size=0.5, random_state=seed,
                                 stratify=strat_temp)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def _card(repo: str, n_total: int, splits, type_counts, lang_counts) -> str:
    tc = "\n".join(f"| `{k}` | {v:,} |" for k, v in type_counts.items())
    lc = ", ".join(f"{k} ({v:,})" for k, v in lang_counts.items())
    return f"""---
license: apache-2.0
task_categories:
- text-classification
language:
{os.linesep.join('- ' + l.split('-')[0] for l in lang_counts)}
tags:
- ovos
- ocp
- media
- intent-classification
- mediavocab
pretty_name: OCP Media Intents
---

# {repo.split('/')[-1]}

Canonical training/benchmark dataset for **OVOS Common Playback (OCP)** media-type
*command/intent* classification. Each row is a natural-language voice command labeled
with a media type. Future `ovos-media-classifier` backends train and benchmark on this.

**Total rows:** {n_total:,} — train {len(splits[0]):,} / validation {len(splits[1]):,} / test {len(splits[2]):,}
(stratified 80/10/10, `random_state=42`).

## Schema

| column | meaning |
|---|---|
| `lang` | BCP-47 language code |
| `domain` | `ocp_play` / `ocp_control` / `not_ocp` |
| `intent` / `media_label` | fine-grained `OCPPlayIntent` label |
| `binary_label` | `ocp` / `not_ocp` |
| `playback_label` | `audio` / `video` / `undefined` |
| `mediavocab_type` | canonical [`mediavocab.MediaType`](https://github.com/TigreGotico/mediavocab) (enforced taxonomy) |
| `genres` | `;`-joined mediavocab genre tags (carries the `adult` content-filter signal) |
| `sentence` | the utterance |

## Sources

Real entity pools (HuggingFace `Jarbas/WikidataMediaEntities` + metal/jazz/prog/movie-role
sets, public catalogue APIs, and `metadatarr`) feed template slot-fill; a naturalistic
layer is synthesized with `agentpipe`. Negatives come from OVOS intent datasets.

## Languages

{lc}

## mediavocab_type distribution

| type | rows |
|---|---|
{tc}

## Content-filter note

The `adult` genre slice exists **so a default-on content filter can BLOCK such requests**
(detect-to-block / parental control) — it is not for adult content provision.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap-per-type", type=int, default=0,
                    help="max rows per mediavocab_type (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default="TigreGotico/ocp-media-intents")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Merging inputs …")
    merged = load_and_merge(args.inputs)
    print("Enforcing mediavocab taxonomy …")
    full = enforce_taxonomy(merged)

    if args.cap_per_type and args.cap_per_type > 0:
        before = len(full)
        full = cap_per_type(full, args.cap_per_type, seed=args.seed)
        print(f"  capped per-type at {args.cap_per_type:,}: {before:,} → {len(full):,}")

    type_counts = full["mediavocab_type"].value_counts().to_dict()
    lang_counts = full["lang"].value_counts().to_dict()
    print("Splitting …")
    train, val, test = split(full, seed=args.seed)

    for name, d in [("full", full), ("train", train),
                    ("validation", val), ("test", test)]:
        p = os.path.join(args.out_dir, f"{name}.csv")
        d.to_csv(p, index=False)
        print(f"  wrote {p}  ({len(d):,})")

    card = _card(args.repo, len(full), (train, val, test), type_counts, lang_counts)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)

    if args.push:
        print(f"Pushing to hub: {args.repo} (private={args.private}) …")
        from datasets import Dataset, DatasetDict
        dd = DatasetDict({
            "train": Dataset.from_pandas(train, preserve_index=False),
            "validation": Dataset.from_pandas(val, preserve_index=False),
            "test": Dataset.from_pandas(test, preserve_index=False),
        })
        dd.push_to_hub(args.repo, private=args.private)
        # upload the card
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=os.path.join(args.out_dir, "README.md"),
            path_in_repo="README.md", repo_id=args.repo, repo_type="dataset",
        )
        print("  pushed.")
    print("Done.")


if __name__ == "__main__":
    main()
