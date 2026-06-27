"""Layer A — **offline popularity gazetteer** of real media titles.

The keyword (``.voc``) backend abstains on a *bare real title* it has no cue for
("attack on titan", "play interstellar", "stream the lakers game") — the
open-vocab gap.  The embedding router can route such a title **once a matching
entity slot fires**, but the router's entity stream starts EMPTY: it only fires
slots for titles the *user* injected (their personal library).  A user who never
saved "Attack on Titan" therefore gets no offline route for it.

This module closes that gap with a **default offline library**: a
popularity-ranked gazetteer of the most common real titles per media type,
derived from the metadatarr-sourced entity pools (``data/entities/*``).  It is
injected into the router's entity matcher *in addition to* the user's personal
library, so the hybrid recognises common real titles offline — no network call,
no user setup.

Why a ranked, capped head — not the whole catalogue
---------------------------------------------------
The metadatarr catalogues hold millions of rows.  Feeding all of them would
(a) over-match arbitrary words at inference (every common word is *some* obscure
title) and (b) blow up live latency — the matcher is O(titles) per query.  So we
keep only the **head of the popularity distribution**, capped top-N per type.

Two builders produce the JSON (build-time only, never shipped):

* ``data/build_gazetteer_hf.py`` (authoritative) — the **TigreGotico HF metadata
  collections**: per-source, type-correct datasets with real popularity signals
  and built-in adult flags (imdb num_votes + is_adult, anilist popularity +
  is_adult, tvmaze rating, steam reviews, podcastindex explicit, …).  No
  cross-contamination, adult excluded at source.
* ``data/build_gazetteer.py`` (fallback) — the local ``data/entities/*.csv``
  pools, ranked by ``_imdb_votes.csv`` with movie-contamination subtraction.

LIVE size MUST stay bounded: see :data:`DEFAULT_TOP_N` and the latency sweep in
``benchmarks/routing_eval.py``.  The long tail is the ONLINE metadatarr layer's
job, not a giant local set.

Safety
------
* **No adult labels.**  ``adult_title`` / ``pornstar`` / ``hentai_title`` are
  NEVER added to the offline gazetteer — an offline adult title would route into
  MOVIE / EPISODIC_SERIES and bypass the keyword adult gate.  Adult routing
  stays exclusively on the keyword content-policy axis (0.0 leak floor).
* **Minimum length.**  Single short common-word titles ("it", "up", "her") are
  dropped (``MIN_TITLE_LEN``) so the gazetteer never hijacks ordinary speech
  that merely contains a common word.  The hybrid's gate stays on keyword
  regardless, but this keeps the *leaf* honest too.

Format & persistence
--------------------
A gazetteer is a plain ``{ner_label: [title, ...]}`` dict (the same shape
:meth:`EmbeddingMediaClassifier.register_user_library` consumes).  Two sources:

* **bundled default** — :data:`BUNDLED_GAZETTEER_PATH`
  (``ovos_media_classifier/data/gazetteer.json``), a small top-N-per-type file
  shipped in the wheel so the offline route works out of the box.
* **generated full** — ``data/gazetteer.json`` (gitignored), the larger pool the
  build script writes from the local ``data/entities/*`` catalogues.

:func:`load_default_gazetteer` prefers the generated file when present, else the
bundled one, and applies the per-type size cap from config.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from ovos_utils.log import LOG

HERE = os.path.dirname(os.path.abspath(__file__))
#: small top-N-per-type gazetteer shipped in the wheel (always available offline)
BUNDLED_GAZETTEER_PATH = os.path.join(HERE, "data", "gazetteer.json")
#: larger generated gazetteer (gitignored); preferred when present
GENERATED_GAZETTEER_PATH = os.path.join(
    os.path.dirname(HERE), "data", "gazetteer.json"
)

#: titles shorter than this are dropped (common single words → hijack risk)
MIN_TITLE_LEN = 4
#: titles longer than this are dropped (junk / descriptions, not real titles)
MAX_TITLE_LEN = 60

# A **single-word** title that is also a common English word is dropped: the
# entity matcher is word-boundary, so a one-word common title ("Legend", "Seven",
# "Serial", "Wednesday") fires on any utterance containing that word and hijacks
# the leaf — e.g. a movie titled "Legend" matching "play the legend of zelda".
# Multi-word titles are unambiguous enough to keep; genuine one-word titles that
# are NOT common words ("Naruto", "Radiohead", "Interstellar") survive.  This is
# a focused stoplist of the highest-collision common words seen in the pools, not
# a full dictionary — it only needs to cover words a user is likely to say.
_COMMON_WORD_STOPLIST = frozenset("""
a an and the of to in on at by for with from into over under up down out off
i you he she it we they me him her us them my your his its our their this that
these those who what when where why how which whom whose
play watch listen read put throw on stream show open start stop pause next
some something anything everything nothing thing things one two three
legend seven serial wednesday monday tuesday thursday friday saturday sunday
heat lion up her him spotlight challengers it train sing dog key agent
love war world worlds time life death day night man woman boy girl
king queen prince god heat fire water earth air light dark home house
yes no maybe okay now then here there again forever never always
hello goodbye please thanks sorry help stop go come back away
music movie game book show video audio song album film series episode
red blue green black white gold silver
""".split())

# Default per-type cap for the LIVE gazetteer.  The entity matcher is a
# word-boundary regex whose per-query cost grows with the injected phrase count,
# so the live gazetteer MUST stay bounded.  Measured on the routing eval
# (``benchmarks/routing_eval.py --latency-sweep``) the median/p95 per-query
# latency is the knee around 500–1000 entities/type:
#
#     cap/type   titles   median   p95
#            0        0   0.48 ms  0.84 ms   (no gazetteer)
#          500     4500   0.87 ms  2.6 ms    <- knee
#         1000     9000   1.3 ms   4.4 ms    <- DEFAULT
#         5000    45000   4.8 ms   23 ms     (p95 climbs sharply)
#
# 1000/type keeps p95 well under ~5 ms while covering the common-titles head;
# the long tail is the ONLINE metadatarr layer's job, not a giant local set.
# Tune via ``media_classifier_gazetteer_size``; ``0``/``None`` = no cap (only
# sane for OFFLINE tagging, NOT live routing — see ``_LIVE_SIZE_WARN``).
DEFAULT_TOP_N = 1000

# Past this many total injected titles the matcher adds noticeable per-query
# latency; a live deployment configuring more than this gets a one-time warning.
_LIVE_SIZE_WARN = 50_000

# Entity-pool CSV → NER label.  ONLY non-adult title/name labels that
# deterministically imply a confident MediaType via NER_LABEL_TO_MEDIA_TYPE.
# Adult labels (adult_title / pornstar / hentai_title) are intentionally ABSENT.
POOL_TO_LABEL: Dict[str, str] = {
    "movie_title": "movie_title",
    "tv_show_title": "tv_show_title",
    "anime_title": "anime_title",
    "cartoon_title": "cartoon_title",
    "artist_name": "artist_name",
    "album_name": "album_name",
    "game_title": "game_title",
    "audiobook_title": "audiobook_title",
    "book_title": "book_title",
    "podcast_title": "podcast_title",
    "radio_station": "radio_station",
    "tv_channel": "tv_channel",
    "documentary_title": "documentary_title",
    "comic_title": "comic_title",
}

# Pools ranked by ``_imdb_votes.csv``.  The vote table is an IMDb *movie*-biased
# popularity table, so it is trustworthy ONLY for the movie pool: applied to the
# (noisy) tv / anime pools it surfaces whatever films leaked into them rather
# than genuine TV / anime.  Every other pool keeps source order — the upstream
# build emits those popular-first (Bach, Counter-Strike, …) and, for the noisy
# video pools, movie-contamination is subtracted first (below) so the source-
# ordered head is at least *plausibly* the right media type.
_VOTE_RANKED_POOLS = {"movie_title"}

# Non-movie video pools that get **movie-contamination subtracted**: the source
# catalogues are noisy and leak famous movie titles into the tv / anime / cartoon
# pools.  Dropping every title that is ALSO a member of ``movie_title.csv``
# removes the worst film leaks (Seven, Iron Man, Fight Club, Spotlight, …) while
# keeping genuine non-film titles (Attack on Titan, Cowboy Bebop, Stranger
# Things — none of which are in the movie pool).
_SUBTRACT_MOVIE_POOLS = {
    "tv_show_title", "anime_title", "cartoon_title", "documentary_title",
}

#: labels that must never appear in the offline gazetteer (adult content policy)
_FORBIDDEN_LABELS = {
    "adult_title", "pornstar", "hentai_title", "porn_genre",
    "adult_streaming_service", "adult_audio_keyword", "adult_keyword",
}


def _clean_title(value: str) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if not (MIN_TITLE_LEN <= len(v) <= MAX_TITLE_LEN):
        return None
    low = v.lower()
    if low in {"unknown", "n/a", "none", "untitled", "various artists",
               "various", "title", "value"}:
        return None
    # MusicBrainz / catalogue placeholders: "[unknown]", "[no artist]",
    # "[language instruction]", "[data]" — bracket-wrapped non-titles.
    if v.startswith("[") and v.endswith("]"):
        return None
    # drop single-word common-English-word titles (word-boundary hijack risk)
    if " " not in v and low in _COMMON_WORD_STOPLIST:
        return None
    return v


def _read_imdb_votes(entities_dir: str) -> Dict[str, int]:
    """``{title_lower: num_votes}`` from ``_imdb_votes.csv`` (popularity ranking)."""
    path = os.path.join(entities_dir, "_imdb_votes.csv")
    votes: Dict[str, int] = {}
    if not os.path.isfile(path):
        return votes
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = (row.get("value") or "").strip()
            if not title:
                continue
            try:
                n = int(row.get("num_votes") or 0)
            except (TypeError, ValueError):
                n = 0
            key = title.lower()
            if n > votes.get(key, -1):
                votes[key] = n
    return votes


def _rank_pool(titles: List[str], votes: Optional[Dict[str, int]],
               top_n: int) -> List[str]:
    """Keep the top-N cleaned titles.

    When *votes* is given (an IMDb-ranked pool), titles WITH a vote count sort
    first (numVotes desc); titles without one follow in source order.  When
    ``None`` (music / game / book pools), keep pure source order — the upstream
    catalogue is already emitted popular-first.
    """
    seen = set()
    ranked: List[tuple] = []
    for order, raw in enumerate(titles):
        v = _clean_title(raw)
        if v is None:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        if votes is not None and key in votes:
            # voted titles: bucket 0, by votes desc
            ranked.append((0, -votes[key], order, v))
        else:
            # unvoted: bucket 1, by source order
            ranked.append((1, 0, order, v))
    ranked.sort()
    return [v for *_, v in ranked[:top_n]]


def build_gazetteer(entities_dir: str, top_n: int = DEFAULT_TOP_N
                    ) -> Dict[str, List[str]]:
    """Build a ``{ner_label: [title, ...]}`` gazetteer from the entity pools.

    Reads each ``data/entities/<pool>.csv`` in :data:`POOL_TO_LABEL`, ranks by
    ``_imdb_votes.csv`` popularity where available, caps at *top_n* per type, and
    skips every adult label.  Returns the in-memory gazetteer (the build script
    persists it to JSON).
    """
    votes = _read_imdb_votes(entities_dir)

    def _pool_titles(pool: str) -> List[str]:
        path = os.path.join(entities_dir, f"{pool}.csv")
        if not os.path.isfile(path):
            return []
        rows: List[str] = []
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row.get("value") or "")
        return rows

    movie_set = {t.lower().strip() for t in _pool_titles("movie_title") if t}

    out: Dict[str, List[str]] = {}
    for pool, label in POOL_TO_LABEL.items():
        if label in _FORBIDDEN_LABELS:
            continue
        if not os.path.isfile(os.path.join(entities_dir, f"{pool}.csv")):
            LOG.debug(f"gazetteer: pool {pool}.csv missing, skipping")
            continue
        titles = _pool_titles(pool)
        if pool in _SUBTRACT_MOVIE_POOLS:
            titles = [t for t in titles if t.lower().strip() not in movie_set]
        pool_votes = votes if pool in _VOTE_RANKED_POOLS else None
        ranked = _rank_pool(titles, pool_votes, top_n)
        if ranked:
            out[label] = ranked
    return out


def _apply_cap(gaz: Dict[str, List[str]], top_n: Optional[int]
               ) -> Dict[str, List[str]]:
    if top_n is None or top_n <= 0:
        return {k: list(v) for k, v in gaz.items() if k not in _FORBIDDEN_LABELS}
    return {k: list(v)[:top_n] for k, v in gaz.items()
            if k not in _FORBIDDEN_LABELS}


def load_default_gazetteer(top_n: Optional[int] = DEFAULT_TOP_N,
                           path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load the offline gazetteer ``{ner_label: [title, ...]}``.

    Resolution order: explicit *path* → generated ``data/gazetteer.json`` →
    bundled ``ovos_media_classifier/data/gazetteer.json``.  Adult labels are
    stripped defensively even if present on disk, and each type is capped at
    *top_n* (``None``/``<=0`` → no cap).  Returns ``{}`` (and logs) on any
    failure so a bad/absent file never breaks the hybrid.
    """
    candidates = [path] if path else [GENERATED_GAZETTEER_PATH,
                                      BUNDLED_GAZETTEER_PATH]
    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as fh:
                gaz = json.load(fh)
            if not isinstance(gaz, dict):
                continue
            capped = _apply_cap(gaz, top_n)
            total = sum(len(v) for v in capped.values())
            if total > _LIVE_SIZE_WARN:
                LOG.warning(
                    f"gazetteer has {total} titles (>{_LIVE_SIZE_WARN}); the "
                    "entity matcher is O(titles) per query, so this adds tens-to-"
                    "hundreds of ms of LIVE routing latency. A bounded set "
                    "(media_classifier_gazetteer_size ~1000/type) is recommended "
                    "for live OCP; large sets are only productive for OFFLINE "
                    "dataset tagging. The online metadatarr layer handles the "
                    "long tail without a giant local set.")
            return capped
        except Exception as e:  # noqa: BLE001 - never break the hybrid
            LOG.warning(f"failed to load gazetteer {cand}: {e}")
    return {}
