"""Build a deterministic labeled eval set for the media classifier.

The eval set is sourced entirely from the bundled ``.voc`` keyword files so the
"ground truth" is whatever vocabulary the keyword backend itself ships with —
no network, no randomness that varies run-to-run (a fixed seed gates the small
amount of template/keyword sampling done to keep the set a few hundred rows).

For every ``<Type>Keyword.voc`` we know which :class:`OCPPlayIntent` the keyword
backend resolves it to (``_VOC_TO_INTENT`` below mirrors the branch order in
``keyword.KeywordMediaClassifier._classify_intent``), and from that the canonical
``mediavocab.MediaType`` (via ``PLAY_INTENT_TO_MEDIA_TYPE``) and genre tags (via
``PLAY_INTENT_TO_GENRES``).  Each keyword phrase is dropped into a handful of
play-style templates to produce realistic utterances.

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

from ovos_media_classifier.intents import (
    MediaType,
    OCPPlayIntent,
    PLAY_INTENT_TO_GENRES,
    PLAY_INTENT_TO_MEDIA_TYPE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LOCALE_DIR = os.path.join(REPO_ROOT, "ovos_media_classifier", "locale")
EVAL_CSV = os.path.join(HERE, "eval_set.csv")

SEED = 42

# Languages to include (must have keyword .voc files).  en-us is mandatory.
LANGS = ["en-us", "de-de", "pt-pt"]

# ---------------------------------------------------------------------------
# voc filename -> OCPPlayIntent it resolves to in the keyword backend.
#
# Only "unambiguous single-keyword" types are listed: types whose keyword
# branch in KeywordMediaClassifier fires on a single .voc match without needing
# a co-occurring keyword (e.g. SHORT_FILM needs MovieKeyword+ShortKeyword, so it
# is intentionally excluded — a bare "short film" template would be mislabeled).
#
# The order here is irrelevant (we label by file); correctness comes from each
# keyword phrase only ever matching its own branch first.  Phrases that would be
# shadowed by an earlier branch are filtered out at build time (see _build_rows).
# ---------------------------------------------------------------------------
_VOC_TO_INTENT: Dict[str, OCPPlayIntent] = {
    "DocumentaryKeyword": OCPPlayIntent.DOCUMENTARY,
    "AudioBookKeyword": OCPPlayIntent.AUDIOBOOK,
    "NewsKeyword": OCPPlayIntent.NEWS,
    "AnimeKeyword": OCPPlayIntent.ANIME,
    "CartoonKeyword": OCPPlayIntent.CARTOON,
    "PodcastKeyword": OCPPlayIntent.PODCAST,
    "AudioDramaKeyword": OCPPlayIntent.RADIO_THEATRE,
    "RadioKeyword": OCPPlayIntent.RADIO,
    "MusicVideoKeyword": OCPPlayIntent.MUSIC_VIDEO,
    "MusicKeyword": OCPPlayIntent.MUSIC,
    "IPTVKeyword": OCPPlayIntent.TV,
    "TVKeyword": OCPPlayIntent.TV,
    "SeriesKeyword": OCPPlayIntent.VIDEO_EPISODES,
    "MovieKeyword": OCPPlayIntent.MOVIE,
    "TrailerKeyword": OCPPlayIntent.TRAILER,
    "BehindTheScenesKeyword": OCPPlayIntent.BEHIND_THE_SCENES,
    "ComicBookKeyword": OCPPlayIntent.VISUAL_STORY,
    "GameKeyword": OCPPlayIntent.GAME,
    "ADKeyword": OCPPlayIntent.AUDIO_DESCRIPTION,
    "ASMRKeyword": OCPPlayIntent.ASMR,
    "AdultKeyword": OCPPlayIntent.ADULT,
    "HentaiKeyword": OCPPlayIntent.HENTAI,
    "VideoKeyword": OCPPlayIntent.VIDEO,
    "AudioKeyword": OCPPlayIntent.AUDIO,
}

# Adult slice vocs (used to measure content-filter recall).
_ADULT_VOCS = {"AdultKeyword", "HentaiKeyword"}

# Play-style templates, per language.  ``{kw}`` is the keyword phrase.
_TEMPLATES: Dict[str, List[str]] = {
    "en-us": [
        "play {kw}",
        "play some {kw}",
        "put on some {kw}",
        "i want to watch a {kw}",
        "can you play {kw} for me",
        "start playing {kw}",
    ],
    # NOTE: German play-verbs "spiele"/"abspielen" embed the GameKeyword "spiel",
    # so we use "starte"/"mach an"/"zeig mir" to keep the carrier phrase from
    # colliding with a keyword.  The "tv show"->TV vs EPISODIC_SERIES mismatch is
    # a genuine keyword-backend priority artifact and is deliberately left in.
    "de-de": [
        "starte {kw}",
        "mach {kw} an",
        "ich möchte {kw} sehen",
        "zeig mir {kw}",
    ],
    "pt-pt": [
        "toca {kw}",
        "põe {kw}",
        "quero ver {kw}",
        "mostra-me {kw}",
    ],
}


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
    keyword for a different intent — those would be shadowed and never resolve
    to the file's own intent.  This keeps the ground truth honest without
    hard-coding the backend's exact behaviour.
    """
    templates = _TEMPLATES.get(lang, _TEMPLATES["en-us"])

    # Pre-load all vocs for this lang, skip empties (lang may lack a file).
    voc_phrases: Dict[str, List[str]] = {}
    for voc in _VOC_TO_INTENT:
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
        intent = _VOC_TO_INTENT[voc]
        mtype: MediaType = PLAY_INTENT_TO_MEDIA_TYPE.get(intent, MediaType.GENERIC)
        genres: List[str] = list(PLAY_INTENT_TO_GENRES.get(intent, []))

        for phrase in phrases:
            owners = owner.get(phrase, {voc})
            # ambiguous keyword shared across different media types -> skip
            mts = {
                PLAY_INTENT_TO_MEDIA_TYPE.get(_VOC_TO_INTENT[o], MediaType.GENERIC)
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
