#!/usr/bin/env python3
"""Build the canonical ``ocp-media-intents`` dataset — the single entry point.

One reproducible command rebuilds the whole training/benchmark set on demand:

    .intent templates  ──expand()──►  slot-free samples
                                          │  slot-fill {slot} from entity pools
                                          ▼
                                    labelled rows
                                          │  rich columns (keyword + NER-by-
                                          ▼  construction + axes + provenance)
                                    balance per media_type  ──►  80/10/10 split
                                          │
                                          ▼   CSV + parquet + dataset card

Run it::

    python -m training.build_dataset                      # default: data/release
    python -m training.build_dataset --langs en-us pt-pt es-es
    python -m training.build_dataset --target-per-type 20000 --adult-cap 7000
    python -m training.build_dataset --push --repo TigreGotico/ocp-media-intents

Everything downstream is deterministic for a fixed ``--seed`` (default 42).

How to extend the set
---------------------
* **More/translated templates** — edit ``training/templates/<lang>/<intent>.intent``
  (and the shared ``vocab/<lang>/<Lead*>.voc`` lead-ins).  The user manages and
  translates these through ovos-localize; ``build_dataset`` picks them up with no
  code change.
* **More entities** — re-run ``python -m training.ingest_entities`` to refresh
  ``data/entities/<label>.csv``, or drop curated values into
  ``training/seed_entities/<label>.csv``.

Columns are documented in ``docs/dataset.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ovos_spec_tools import expand

from mediavocab import MediaType, infer_playback_type
from ovos_media_classifier.axes import infer_structure
from ovos_media_classifier.intents import (
    LABEL_TO_MEDIA_TYPE,
    LABEL_TO_GENRES,
    OCPEntityLabel,
)
from ovos_media_classifier.features import _KEYWORD_VOCABS, CategoricalFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(_HERE, "templates")
SEED_ENTITIES_DIR = os.path.join(_HERE, "seed_entities")
DEFAULT_ENTITIES_DIR = os.path.join(os.path.dirname(_HERE), "data", "entities")

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# Languages with authored lead-in vocabularies (see author_templates.LEADINS).
CORE_LANGS = ["en-us", "pt-pt", "es-es", "fr-fr", "de-de", "it-it", "nl-nl"]

# The full set of OCPEntityLabel string values (for the NER-by-construction
# columns + slot validation).
ENTITY_LABELS: List[str] = [e.value for e in OCPEntityLabel]

# Slot aliases: a few template slots have no dedicated metadata pool; fill them
# from a closely-related real pool so the surface text stays realistic.
SLOT_ALIASES: Dict[str, str] = {
    "movie_genre":        "video_genre",
    "trailer_title":      "movie_title",
    "bts_title":          "movie_title",
    "music_video_title":  "tv_show_title",
    "silent_movie_title": "movie_title",
    "bw_movie_title":     "movie_title",
    "track_name":         "album_name",
}

# Every media-template row is a play request.
PLAY_DOMAIN = "ocp_play"


# ---------------------------------------------------------------------------
# Entity pools + lead-in vocabularies + templates
# ---------------------------------------------------------------------------

def load_entity_pools(entities_dir: str) -> Dict[str, List[str]]:
    """Load ``<label>.csv`` pools (data/entities + seed_entities), with aliases."""
    pools: Dict[str, List[str]] = {}
    for base in (entities_dir, SEED_ENTITIES_DIR):
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if not fn.endswith(".csv"):
                continue
            label = fn[:-4]
            try:
                df = pd.read_csv(os.path.join(base, fn), usecols=["value"],
                                 dtype=str, keep_default_na=False)
            except Exception:
                continue
            vals = [v.strip() for v in df["value"].tolist() if v and v.strip()]
            if not vals:
                continue
            pool = pools.setdefault(label, [])
            seen = {x.lower() for x in pool}
            for v in vals:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    pool.append(v)
    for slot, target in SLOT_ALIASES.items():
        if slot not in pools and target in pools:
            pools[slot] = pools[target]
    return pools


def load_leadin_vocabs(lang: str) -> Dict[str, Sequence[str]]:
    """Load the shared lead-in ``.voc`` members for ``expand()``."""
    vocs: Dict[str, Sequence[str]] = {}
    voc_dir = os.path.join(TEMPLATES_DIR, "vocab", lang)
    if not os.path.isdir(voc_dir):
        return vocs
    for fn in os.listdir(voc_dir):
        if not fn.endswith(".voc"):
            continue
        with open(os.path.join(voc_dir, fn), encoding="utf-8") as fh:
            members = [ln.strip() for ln in fh if ln.strip()]
        if members:
            vocs[fn[:-4]] = members
    return vocs


def load_intent_templates(lang: str) -> Dict[str, List[str]]:
    """Return ``{intent: [template lines]}`` for a language."""
    lang_dir = os.path.join(TEMPLATES_DIR, lang)
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(lang_dir):
        return out
    for fn in sorted(os.listdir(lang_dir)):
        if not fn.endswith(".intent"):
            continue
        with open(os.path.join(lang_dir, fn), encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if lines:
            out[fn[:-len(".intent")]] = lines
    return out


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _label_axes(media_label: str) -> Tuple[str, List[str], str, str, str]:
    """Return (media_type, genres, playback_type, structure, binary_label)."""
    mt = LABEL_TO_MEDIA_TYPE.get(media_label, MediaType.GENERIC)
    genres = list(LABEL_TO_GENRES.get(media_label, []))
    return (mt.value, genres, infer_playback_type(mt).value,
            infer_structure(mt).value, "ocp")


# ---------------------------------------------------------------------------
# Derived multi-task axes — all free (ground-truth by construction) from the
# template ``intent`` + the ``slot_values`` that filled the row.  Each becomes a
# dataset column and a trained head.  See docs/model.md.
# ---------------------------------------------------------------------------

# slot names whose VALUE is a real (non-form) genre → the ``content_genre`` axis
_GENRE_SLOTS = (
    "music_genre", "movie_genre", "video_genre", "tv_genre", "game_genre",
    "radio_genre", "podcast_genre", "book_genre", "comic_genre",
    "audiobook_genre",
)
# slot names whose value is a mood / activity → the ``mood`` axis
_MOOD_SLOTS = ("playlist_mood", "playlist_activity")
# slot names carrying a release year / decade → the ``era`` axis
_YEAR_SLOTS = ("release_year",)
_DECADE_SLOTS = ("release_decade",)

# ``intent`` aliases that are really a base media type + a result-narrowing
# qualifier (the qualifier is the real signal; the type is the base).
_INTENT_QUALIFIERS: Dict[str, List[str]] = {
    "bw_movie":     ["black_and_white"],
    "silent_movie": ["silent"],
    "audio_description": ["audio_described"],
    "trailer":      ["trailer"],
    "behind_the_scenes": ["behind_the_scenes"],
}
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _decade_of(year: str) -> Optional[str]:
    m = _YEAR_RE.search(year or "")
    if not m:
        return None
    y = int(m.group(0))
    return f"{(y // 10) * 10}s"


def _derive_axes(intent: str, genres: List[str],
                 slot_values: Dict[str, str]) -> Dict[str, object]:
    """Compute the free multi-task axis labels for one row.

    All labels are ground-truth: ``content_genre`` / ``mood`` / ``era`` come
    straight from the slot value that filled the template, ``qualifiers`` from
    the intent alias, ``explicitness`` from the ``adult`` form-genre.
    """
    content_genre = sorted({
        slot_values[s].lower() for s in _GENRE_SLOTS if s in slot_values
    })
    mood = next((slot_values[s] for s in _MOOD_SLOTS if s in slot_values), "")
    era = next((slot_values[s] for s in _YEAR_SLOTS if s in slot_values), "")
    decade = next((slot_values[s] for s in _DECADE_SLOTS if s in slot_values), "")
    if not decade and era:
        decade = _decade_of(era) or ""

    qualifiers = list(_INTENT_QUALIFIERS.get(intent, []))

    is_adult = "adult" in genres
    explicitness = "adult" if is_adult else "clean"

    return {
        # multi-label sensitive/form genres — what the content filter reads
        "content_form_genres": json.dumps(list(genres)),
        # multi-label real genre (rock/jazz/action/…)
        "content_genre": json.dumps(content_genre),
        "mood": mood,
        "era": era,
        "decade": decade,
        "explicitness": explicitness,
        # multi-label result-narrowing qualifiers (bw / silent / …)
        "qualifiers": json.dumps(qualifiers),
        # control axis — play rows carry no control intent
        "control_intent": "",
    }


def fill_slots(
    sample: str, pools: Dict[str, List[str]], rng: random.Random,
) -> Optional[Tuple[str, Dict[str, str]]]:
    """Replace each ``{slot}`` with a sampled real entity.

    Returns ``(filled_sentence, {slot: value})`` or ``None`` if a required slot
    pool is empty (the sample is skipped — never emit a literal ``{slot}``).
    """
    chosen: Dict[str, str] = {}
    filled = sample
    for slot in _SLOT_RE.findall(sample):
        pool = pools.get(slot)
        if not pool:
            return None
        if slot not in chosen:
            chosen[slot] = rng.choice(pool)
        filled = filled.replace("{" + slot + "}", chosen[slot], 1)
    return " ".join(filled.split()), chosen


def build_rows_for_lang(
    lang: str, pools: Dict[str, List[str]],
    fills_per_template: int, rng: random.Random,
) -> List[dict]:
    """Expand + slot-fill all templates for one language into labelled rows."""
    vocs = load_leadin_vocabs(lang)
    templates = load_intent_templates(lang)
    rows: List[dict] = []
    tid = 0
    for intent, lines in templates.items():
        media_type, genres, pb, struct, binary = _label_axes(intent)
        genres_json = json.dumps(genres)
        for line in lines:
            tid += 1
            template_id = f"{lang}:{intent}:{tid}"
            try:
                samples = expand(line, vocs)
            except Exception:
                continue
            for sample in samples:
                n_fills = fills_per_template if _SLOT_RE.search(sample) else 1
                seen_local: set = set()
                for _ in range(n_fills):
                    res = fill_slots(sample, pools, rng)
                    if res is None:
                        break
                    sentence, slot_values = res
                    key = sentence.lower()
                    if key in seen_local:
                        continue
                    seen_local.add(key)
                    row = {
                        "sentence": sentence,
                        "lang": lang,
                        "domain": PLAY_DOMAIN,
                        "intent": intent,
                        "media_type": media_type,
                        "genres": genres_json,
                        "playback_type": pb,
                        "structure": struct,
                        "binary_label": binary,
                    }
                    row.update(_derive_axes(intent, genres, slot_values))
                    row.update({
                        "template_id": template_id,
                        "template": line,
                        "n_slots": len(slot_values),
                        "entity_labels": json.dumps(sorted(slot_values)),
                        "slot_values": json.dumps(slot_values, ensure_ascii=False),
                    })
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Rich feature columns
# ---------------------------------------------------------------------------

_KEYWORD_COLS = [col for _voc, col in _KEYWORD_VOCABS]
_NER_COLS = [f"ner_{label}" for label in ENTITY_LABELS]


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add keyword (CategoricalFeatureExtractor) + NER-by-construction columns.

    * ``kw_*`` / ``verb_*`` / ``mod_*`` / ``fmt_*`` — computed on the realised
      sentence by the same extractor the runtime uses, so a model trains on the
      exact features it will see at inference (no extraction step needed).
    * ``ner_<label>`` — 1 where ``{label}`` was filled (ground truth, not
      predicted), so the set doubles as NER / slot-filling training data.
    """
    extractor = CategoricalFeatureExtractor.from_locale_dir()

    kw_cache: Dict[Tuple[str, str], set] = {}
    kw_data = {col: [] for col in _KEYWORD_COLS}
    for sent, lang in zip(df["sentence"], df["lang"]):
        ckey = (sent, lang)
        fired = kw_cache.get(ckey)
        if fired is None:
            fired = set(extractor.extract(sent, lang=lang))
            kw_cache[ckey] = fired
        for col in _KEYWORD_COLS:
            kw_data[col].append(1 if col in fired else 0)

    ner_data = {f"ner_{label}": [] for label in ENTITY_LABELS}
    for raw in df["entity_labels"]:
        present = set(json.loads(raw)) if raw else set()
        for label in ENTITY_LABELS:
            ner_data[f"ner_{label}"].append(1 if label in present else 0)

    # assemble all feature columns at once (avoids fragmentation)
    feat_df = pd.DataFrame({**kw_data, **ner_data}, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ---------------------------------------------------------------------------
# Balance + split
# ---------------------------------------------------------------------------

ADULT_GENRE = "adult"


def _is_adult(genres_json: str) -> bool:
    try:
        return ADULT_GENRE in json.loads(genres_json)
    except Exception:
        return False


def balance(df: pd.DataFrame, target_per_type: int, adult_cap: int,
            seed: int) -> pd.DataFrame:
    """Cap (mediavocab) type classes toward an even band; keep adult a minority.

    Non-adult classes are sampled toward ``target_per_type`` each.  Adult rows
    (any class carrying the ``adult`` genre) are sampled to at most ``adult_cap``
    TOTAL so the content filter has enough diverse examples to learn while the
    slice stays well below a normal class.
    """
    adult_mask = df["genres"].map(_is_adult)
    adult_df = df[adult_mask]
    main_df = df[~adult_mask]

    parts: List[pd.DataFrame] = []
    for _mt, grp in main_df.groupby("media_type"):
        if len(grp) > target_per_type:
            grp = grp.sample(n=target_per_type, random_state=seed)
        parts.append(grp)
    if len(adult_df) > adult_cap:
        adult_df = adult_df.sample(n=adult_cap, random_state=seed)
    parts.append(adult_df)

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, seed: int):
    from sklearn.model_selection import train_test_split
    counts = df["media_type"].value_counts()
    rare = counts[counts < 10].index
    strat = df["media_type"].where(~df["media_type"].isin(rare), "._rare_")
    train, temp = train_test_split(df, test_size=0.2, random_state=seed,
                                   stratify=strat)
    strat_t = temp["media_type"].where(~temp["media_type"].isin(rare), "._rare_")
    val, test = train_test_split(temp, test_size=0.5, random_state=seed,
                                 stratify=strat_t)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def dataset_card(repo: str, n_total: int, splits, type_counts, lang_counts,
                 adult_n: int) -> str:
    tc = "\n".join(f"| `{k}` | {v:,} |" for k, v in type_counts.items())
    lc = ", ".join(f"{k} ({v:,})" for k, v in lang_counts.items())
    langs_yaml = os.linesep.join(
        "- " + l.split("-")[0] for l in dict.fromkeys(lang_counts))
    return f"""---
license: apache-2.0
task_categories:
- text-classification
- token-classification
language:
{langs_yaml}
tags:
- ovos
- ocp
- media
- intent-classification
- slot-filling
- mediavocab
pretty_name: OCP Media Intents
---

# {repo.split('/')[-1]}

Canonical training/benchmark dataset for **OVOS Common Playback (OCP)** media
command classification. Each row is a natural-language voice command labelled
with a media type, the orthogonal coarse axes, content-filter genres, and a full
set of precomputed features — so it trains a classifier (or a slot-filler / NER /
entity-linker) directly, with no feature-extraction step.

**Total rows:** {n_total:,} — train {len(splits[0]):,} / validation {len(splits[1]):,} / test {len(splits[2]):,}
(stratified 80/10/10 on `media_type`, `random_state=42`).

## How it is built

Reproducible from source with one command (`python -m training.build_dataset`):
translatable OVOS-INTENT-1 `.intent` templates are expanded with
`ovos_spec_tools.expand`, their `{{slot}}` placeholders are filled with **real
entities** from the
[TigreGotico media-metadata collection](https://huggingface.co/TigreGotico),
rich feature columns are computed, the type classes are balanced, and the set is
split. See `docs/dataset.md` and `docs/data-sources.md`.

## Columns

| group | columns |
|---|---|
| core | `sentence`, `lang`, `domain`, `intent`, `media_type`, `genres`, `playback_type`, `structure`, `binary_label` |
| keyword features | `kw_*`, `verb_*`, `mod_*`, `fmt_*` — 0/1, computed on `sentence` by the runtime `CategoricalFeatureExtractor` |
| NER-by-construction | `ner_<entity_label>` — 0/1 ground-truth flag set where `{{entity_label}}` was filled; `slot_values` maps each filled slot to its entity |
| provenance | `template_id`, `template`, `n_slots`, `entity_labels` |

`genres`, `entity_labels` and `slot_values` are JSON strings.

## media_type distribution

| type | rows |
|---|---|
{tc}

## Languages

{lc}

## Content-filter slice (adult, detect-to-block)

This set includes a **deliberate minority of {adult_n:,} adult rows** built from
real adult performer / title entities. They exist **only so a default-on content
filter can BLOCK such requests** (parental control / detect-to-block) — never for
adult-content provision. Every adult row carries the `adult` genre (via
`LABEL_TO_GENRES`); the filter blocks on that genre. The slice is kept far below
a normal class so the model learns to detect it without it dominating training.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# column order: core first, then the multi-task axes, then features, provenance
_CORE = ["sentence", "lang", "domain", "intent", "media_type", "genres",
         "playback_type", "structure", "binary_label"]
# the free, ground-truth-by-construction multi-task axis columns (one head each)
_AXES = ["content_form_genres", "content_genre", "mood", "era", "decade",
         "explicitness", "qualifiers", "control_intent"]
_PROV = ["template_id", "template", "n_slots", "entity_labels", "slot_values"]


def build(out_dir: str, entities_dir: str, langs: List[str],
          fills_per_template: int, target_per_type: int, adult_cap: int,
          seed: int, push: bool = False,
          repo: str = "TigreGotico/ocp-media-intents",
          private: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    print(f"Loading entity pools from {entities_dir} (+ seed) …")
    pools = load_entity_pools(entities_dir)
    print(f"  {len(pools)} pools; {sum(len(v) for v in pools.values()):,} entities")

    all_rows: List[dict] = []
    for lang in langs:
        rows = build_rows_for_lang(lang, pools, fills_per_template, rng)
        print(f"  {lang}: {len(rows):,} filled rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    before = len(df)
    df.drop_duplicates(subset=["sentence", "lang"], inplace=True)
    print(f"Generated {before:,} → {len(df):,} unique (sentence, lang) rows")

    print("Computing feature columns …")
    df = add_feature_columns(df)

    print("Balancing …")
    adult_total = int(df["genres"].map(_is_adult).sum())
    df = balance(df, target_per_type, adult_cap, seed)
    adult_kept = int(df["genres"].map(_is_adult).sum())
    print(f"  adult rows {adult_total:,} → kept {adult_kept:,} (minority)")

    type_counts = df["media_type"].value_counts().to_dict()
    lang_counts = df["lang"].value_counts().to_dict()

    print("Splitting 80/10/10 …")
    train, val, test = split(df, seed=seed)

    ordered = _CORE + _AXES + _KEYWORD_COLS + _NER_COLS + _PROV
    df = df[[c for c in ordered if c in df.columns]]
    train = train[[c for c in ordered if c in train.columns]]
    val = val[[c for c in ordered if c in val.columns]]
    test = test[[c for c in ordered if c in test.columns]]

    for name, d in [("full", df), ("train", train),
                    ("validation", val), ("test", test)]:
        d.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        d.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
        print(f"  wrote {name}: {len(d):,}")

    card = dataset_card(repo, len(df), (train, val, test),
                        type_counts, lang_counts, adult_kept)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    print(f"  wrote dataset card → {out_dir}/README.md")

    if push:
        print(f"Pushing to {repo} (private={private}) …")
        from datasets import Dataset, DatasetDict
        dd = DatasetDict({
            "train": Dataset.from_pandas(train, preserve_index=False),
            "validation": Dataset.from_pandas(val, preserve_index=False),
            "test": Dataset.from_pandas(test, preserve_index=False),
        })
        dd.push_to_hub(repo, private=private)
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=os.path.join(out_dir, "README.md"),
            path_in_repo="README.md", repo_id=repo, repo_type="dataset")
        print("  pushed.")
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the ocp-media-intents dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out-dir", default="data/release")
    ap.add_argument("--entities-dir", default=DEFAULT_ENTITIES_DIR)
    ap.add_argument("--langs", nargs="*", default=CORE_LANGS)
    ap.add_argument("--fills-per-template", type=int, default=6,
                    help="entity fills per expanded slotted sample (default: 6)")
    ap.add_argument("--target-per-type", type=int, default=20000,
                    help="cap per (non-adult) media_type (default: 20000)")
    ap.add_argument("--adult-cap", type=int, default=7000,
                    help="max adult rows total — the learnable minority (default: 7000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default="TigreGotico/ocp-media-intents")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    build(out_dir=args.out_dir, entities_dir=args.entities_dir, langs=args.langs,
          fills_per_template=args.fills_per_template,
          target_per_type=args.target_per_type, adult_cap=args.adult_cap,
          seed=args.seed, push=args.push, repo=args.repo, private=args.private)


if __name__ == "__main__":
    main()
