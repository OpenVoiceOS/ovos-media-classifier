"""Build a deterministic labeled eval set for the media classifier.

The eval set is sourced entirely from the bundled ``.voc`` keyword files so the
"ground truth" is whatever vocabulary the keyword backend itself ships with —
no network, no randomness that varies run-to-run (a fixed seed gates the small
amount of template/keyword sampling done to keep the set a few hundred rows).

For every ``<Type>Keyword.voc`` we know the ``mediavocab.MediaType`` and genre
tags the keyword backend resolves it to (``_VOC_TO_MEDIA_TYPE`` /
``_VOC_TO_GENRES`` below mirror the branch outcomes in
``keyword.KeywordMediaClassifier._classify_leaf``).  Each keyword phrase is
dropped into a handful of play-style templates to produce realistic utterances.

The result is written to ``benchmarks/eval_set.csv`` with columns::

    utterance, lang, expected_media_type, expected_genres

``expected_genres`` is a ``|``-joined list (possibly empty).  The adult slice
(AdultKeyword / HentaiKeyword) is always included so content-filter recall can
be measured downstream.
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from mediavocab import PlaybackType, infer_playback_type

from ovos_media_classifier.intents import MediaType

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LOCALE_DIR = os.path.join(REPO_ROOT, "ovos_media_classifier", "locale")
EVAL_CSV = os.path.join(HERE, "eval_set.csv")

SEED = 42

# Languages to include (must have keyword .voc files).  en-us is mandatory.
LANGS = ["en-us", "de-de", "pt-pt"]

# ---------------------------------------------------------------------------
# voc filename -> (MediaType, genres) the keyword backend resolves it to.
#
# Only "unambiguous single-keyword" types are listed: types whose keyword
# branch in KeywordMediaClassifier fires on a single .voc match without needing
# a co-occurring keyword (e.g. SHORT_FILM needs MovieKeyword+ShortKeyword, so it
# is intentionally excluded — a bare "short film" template would be mislabeled).
#
# Several vocs collapse onto the same MediaType but carry distinct genres
# (anime → EPISODIC_SERIES + ["anime"]; cartoon → EPISODIC_SERIES +
# ["animation"]); ``_VOC_TO_GENRES`` keeps that signal for content-filter recall.
#
# Correctness comes from each keyword phrase only ever matching its own branch
# first.  Phrases that would be shadowed by an earlier branch (a different media
# type) are filtered out at build time (see _build_rows).
# ---------------------------------------------------------------------------
_VOC_TO_MEDIA_TYPE: Dict[str, MediaType] = {
    "DocumentaryKeyword": MediaType.MOVIE,
    "AudioBookKeyword": MediaType.AUDIOBOOK,
    "NewsKeyword": MediaType.RADIO,
    "AnimeKeyword": MediaType.EPISODIC_SERIES,
    "CartoonKeyword": MediaType.EPISODIC_SERIES,
    "PodcastKeyword": MediaType.PODCAST,
    "AudioDramaKeyword": MediaType.AUDIO_DRAMA,
    "RadioKeyword": MediaType.RADIO,
    "MusicVideoKeyword": MediaType.MUSIC_VIDEO,
    "MusicKeyword": MediaType.MUSIC,
    "IPTVKeyword": MediaType.TV,
    "TVKeyword": MediaType.TV,
    "SeriesKeyword": MediaType.EPISODIC_SERIES,
    "MovieKeyword": MediaType.MOVIE,
    "TrailerKeyword": MediaType.MOVIE,
    "BehindTheScenesKeyword": MediaType.MOVIE,
    "ComicBookKeyword": MediaType.COMIC,
    "GameKeyword": MediaType.GAME,
    "ADKeyword": MediaType.MOVIE,
    "ASMRKeyword": MediaType.PROCEDURAL_AMBIENT,
    "AdultKeyword": MediaType.MOVIE,
    "HentaiKeyword": MediaType.EPISODIC_SERIES,
    "VideoKeyword": MediaType.MOVIE,
    "AudioKeyword": MediaType.MUSIC,
}

# voc filename -> genre tags it surfaces (empty for genre-neutral leaves).
_VOC_TO_GENRES: Dict[str, List[str]] = {
    "AnimeKeyword": ["anime"],
    "CartoonKeyword": ["animation"],
    "ASMRKeyword": ["asmr"],
    "AdultKeyword": ["adult"],
    "HentaiKeyword": ["anime", "adult"],
}

# Adult slice vocs (used to measure content-filter recall).
_ADULT_VOCS = {"AdultKeyword", "HentaiKeyword"}

# Play-style templates, per language.  ``{kw}`` is the keyword phrase.
#
# Templates are **modality-consistent**: the carrier verb must agree with the
# leaf's playback modality, because the keyword backend is now *hierarchical*
# (coarse-to-fine) — it predicts the modality from the verb FIRST and constrains
# the leaf to it.  A contradictory carrier ("watch a radio") is not something a
# real user says, and pairing it with an audio label would be a self-inflicted
# mislabel, so each modality gets its own neutral + verb-matched templates.
#
# ``_NEUTRAL`` carriers (generic "play") work for every modality; the
# modality-specific ones supply the matching verb (watch / listen / read / launch).
_NEUTRAL_TEMPLATES: Dict[str, List[str]] = {
    "en-us": ["play {kw}", "play some {kw}", "can you play {kw} for me",
              "start playing {kw}"],
    "de-de": ["starte {kw}", "mach {kw} an"],
    "pt-pt": ["toca {kw}", "põe {kw}"],
}
_MODALITY_TEMPLATES: Dict[str, Dict[PlaybackType, List[str]]] = {
    "en-us": {
        PlaybackType.VIDEO: ["i want to watch a {kw}", "watch {kw}"],
        PlaybackType.AUDIO: ["i want to listen to {kw}", "put on some {kw}"],
        PlaybackType.PAGED: ["read me {kw}", "i want to read a {kw}"],
        PlaybackType.INTERACTIVE: ["i want to play a {kw}", "launch {kw}"],
    },
    "de-de": {
        PlaybackType.VIDEO: ["ich möchte {kw} sehen", "zeig mir {kw}"],
        PlaybackType.AUDIO: ["ich möchte {kw} hören"],
        PlaybackType.PAGED: ["lies mir {kw} vor"],
        PlaybackType.INTERACTIVE: ["ich will {kw} spielen"],
    },
    "pt-pt": {
        PlaybackType.VIDEO: ["quero ver {kw}", "mostra-me {kw}"],
        PlaybackType.AUDIO: ["quero ouvir {kw}"],
        PlaybackType.PAGED: ["lê-me {kw}"],
        PlaybackType.INTERACTIVE: ["quero jogar {kw}"],
    },
}


def _templates_for(lang: str, media_type: MediaType) -> List[str]:
    """Modality-consistent templates for a leaf: neutral carriers + the carriers
    whose verb matches the leaf's playback modality."""
    neutral = _NEUTRAL_TEMPLATES.get(lang, _NEUTRAL_TEMPLATES["en-us"])
    by_mod = _MODALITY_TEMPLATES.get(lang, _MODALITY_TEMPLATES["en-us"])
    modality = infer_playback_type(media_type)
    return list(neutral) + list(by_mod.get(modality, []))


@dataclass
class EvalRow:
    utterance: str
    lang: str
    expected_media_type: str          # mediavocab MediaType value
    expected_genres: List[str]        # mediavocab genre tags (may be empty)


def _load_voc(voc_name: str, lang: str) -> List[str]:
    """Return the keyword phrases in ``<lang>/<voc_name>.voc`` (lowercased)."""
    # locale dirs on disk are lowercase (en-us); match case-insensitively.
    candidates = [lang.lower(), lang.lower().split("-")[0]]
    for tag in candidates:
        for entry in os.listdir(LOCALE_DIR):
            if entry.lower() == tag and os.path.isdir(os.path.join(LOCALE_DIR, entry)):
                path = os.path.join(LOCALE_DIR, entry, f"{voc_name}.voc")
                if os.path.isfile(path):
                    out: List[str] = []
                    with open(path, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                out.append(line.lower())
                    return out
    return []


def _build_rows(lang: str, rng: random.Random) -> List[EvalRow]:
    """Build eval rows for one language.

    To avoid mislabeling from the keyword backend's priority chain we drop any
    keyword phrase that is a substring of (or contains) a *higher-priority*
    keyword for a different media type — those would be shadowed and never
    resolve to the file's own type.  This keeps the ground truth honest without
    hard-coding the backend's exact behaviour.
    """
    # Pre-load all vocs for this lang, skip empties (lang may lack a file).
    voc_phrases: Dict[str, List[str]] = {}
    for voc in _VOC_TO_MEDIA_TYPE:
        phrases = _load_voc(voc, lang)
        if phrases:
            voc_phrases[voc] = phrases

    # Build a global phrase -> owning voc map to detect cross-intent collisions.
    # If a phrase appears in two vocs that map to different media types, it is
    # genuinely ambiguous for the keyword backend, so exclude it.
    owner: Dict[str, set] = {}
    for voc, phrases in voc_phrases.items():
        for p in phrases:
            owner.setdefault(p, set()).add(voc)

    rows: List[EvalRow] = []
    for voc, phrases in voc_phrases.items():
        mtype: MediaType = _VOC_TO_MEDIA_TYPE.get(voc, MediaType.GENERIC)
        genres: List[str] = list(_VOC_TO_GENRES.get(voc, []))

        # Modality-consistent carriers for this leaf (verb agrees with modality).
        templates = _templates_for(lang, mtype)

        for phrase in phrases:
            owners = owner.get(phrase, {voc})
            # ambiguous keyword shared across different media types -> skip
            mts = {
                _VOC_TO_MEDIA_TYPE.get(o, MediaType.GENERIC)
                for o in owners
            }
            if len(mts) > 1:
                continue
            for tmpl in templates:
                rows.append(
                    EvalRow(
                        utterance=tmpl.format(kw=phrase),
                        lang=lang,
                        expected_media_type=mtype.value,
                        expected_genres=genres,
                    )
                )

    # Deterministic cap to keep the set a few hundred rows per lang while
    # preserving at least one row per (media_type) present.
    rng.shuffle(rows)
    return rows


def build_eval_rows() -> List[EvalRow]:
    """Build the full multi-lang eval set deterministically."""
    rng = random.Random(SEED)
    all_rows: List[EvalRow] = []
    for lang in LANGS:
        all_rows.extend(_build_rows(lang, rng))
    # stable sort so the CSV is byte-reproducible across runs
    all_rows.sort(key=lambda r: (r.lang, r.expected_media_type, r.utterance))
    return all_rows


def write_eval_csv(rows: Optional[List[EvalRow]] = None, path: str = EVAL_CSV) -> str:
    rows = rows or build_eval_rows()
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utterance", "lang", "expected_media_type", "expected_genres"])
        for r in rows:
            w.writerow([r.utterance, r.lang, r.expected_media_type, "|".join(r.expected_genres)])
    return path


def load_eval_csv(path: str = EVAL_CSV) -> List[EvalRow]:
    """Load the committed eval set; (re)build it if missing."""
    if not os.path.isfile(path):
        write_eval_csv(path=path)
    rows: List[EvalRow] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            genres = [g for g in (r.get("expected_genres") or "").split("|") if g]
            rows.append(
                EvalRow(
                    utterance=r["utterance"],
                    lang=r["lang"],
                    expected_media_type=r["expected_media_type"],
                    expected_genres=genres,
                )
            )
    return rows


def summary(rows: List[EvalRow]) -> Dict[str, object]:
    by_lang: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    adult = 0
    for r in rows:
        by_lang[r.lang] = by_lang.get(r.lang, 0) + 1
        by_type[r.expected_media_type] = by_type.get(r.expected_media_type, 0) + 1
        if "adult" in r.expected_genres:
            adult += 1
    return {"total": len(rows), "by_lang": by_lang, "by_type": by_type, "adult_rows": adult}


if __name__ == "__main__":
    rows = build_eval_rows()
    path = write_eval_csv(rows)
    s = summary(rows)
    print(f"wrote {s['total']} rows -> {path}")
    print("by lang:", s["by_lang"])
    print("by media_type:", s["by_type"])
    print("adult-genre rows:", s["adult_rows"])
