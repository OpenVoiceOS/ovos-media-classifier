#!/usr/bin/env python3
"""Ingest the full TigreGotico media-metadata collection into entity pools.

Every source in the collection is read into one or more ``OCPEntityLabel``
pools.  Each pool is written to ``data/entities/<label>.csv`` (single column
``value``), deduplicated case-insensitively and capped at ``--cap`` rows.

Two source backends are used, in priority order:

1. **Local ``metadatarr`` scraper cache** (``~/.cache/metadatarr/scrapers/
   <name>.jsonl``) when present — this is the freshest, complete dump and avoids
   re-downloading hundreds of MB from the Hub.
2. **HuggingFace** (``TigreGotico/<id>``) for everything not cached locally
   (the genre-specific music archives, the movie crew sets, the wikidata
   entities split, and the adult content-filter sets).

The mapping from each source to its slot label(s) lives in ``SOURCE_SPECS``
below — it is the single source of truth documented in ``docs/data-sources.md``.

Usage::

    python -m training.ingest_entities                  # all sources → data/entities/
    python -m training.ingest_entities --only adult     # adult content-filter sets only
    python -m training.ingest_entities --cap 200000 --output data/entities
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

# the local metadatarr scraper cache (jsonl dumps, one object per line)
METADATARR_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "metadatarr", "scrapers"
)

# bump the csv field-size limit: some descriptions/aliases are long
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


# ---------------------------------------------------------------------------
# Row → pool emitters
#
# Each emitter takes one source record (a dict) and yields ``(label, value)``
# pairs.  Emitters are written once and shared by the local-jsonl and HF
# backends (both deliver dict rows with the same field names).
# ---------------------------------------------------------------------------

def _clean(v) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in ("", "nan", "none", "null", "[]", "{}"):
        return ""
    return s


def _maybe_list(v) -> List[str]:
    """Coerce a field that may be a python/json list-string or a scalar.

    Tolerates ``numpy.ndarray`` (HF parquet often loads multi-valued columns —
    e.g. IMDb ``genres`` — as ``ndarray`` of ``object``), python/json
    list-strings, and ``sep``-joined scalars.
    """
    if v is None:
        return []
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            v = v.tolist()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v if _clean(x)]
    # a scalar NaN (float) is not a list
    if isinstance(v, float):
        return []
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return []
    if s[0] in "[(":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [_clean(x) for x in parsed if _clean(x)]
        except (ValueError, SyntaxError):
            pass
    # pipe / comma separated fallbacks
    for sep in ("|", ";"):
        if sep in s:
            return [_clean(p) for p in s.split(sep) if _clean(p)]
    return [_clean(s)] if _clean(s) else []


def _decade_of(year_val) -> str:
    s = _clean(year_val)
    m = re.search(r"(\d{4})", s)
    if not m:
        return ""
    return f"{int(m.group(1)) // 10 * 10}s"


def emit_unified_artists(row):
    """``TigreGotico/media-metadata-artists`` — the canonical UNIFIED artist set.

    Supersedes the per-source musicbrainz / audiodb / jazz / prog / metal /
    classical merges: this dataset is already cross-deduplicated across all those
    sources (the ``sources`` array records provenance).  ``name`` + ``aliases`` →
    ``artist_name``; ``style`` / ``tags`` → ``music_genre`` (when usable).
    """
    v = _clean(row.get("name"))
    if v:
        yield ("artist_name", v)
    for a in _maybe_list(row.get("aliases")):
        yield ("artist_name", a)
    g = _clean(row.get("style"))
    if g:
        # ``style`` may be a slash/comma compound ("Rock/Pop") — split it
        for part in re.split(r"[/,;|]", g):
            part = _clean(part)
            if part:
                yield ("music_genre", part)
    # a couple of free-text genre tags (skip the long-tail noise)
    for t in _maybe_list(row.get("tags"))[:2]:
        t = _clean(t)
        if t and len(t) > 2:
            yield ("music_genre", t)


def emit_musicbrainz_releases(row):
    title = _clean(row.get("title"))
    if title:
        # release groups are albums; tracks are not in this dump
        yield ("album_name", title)
    for a in _maybe_list(row.get("artist_names")):
        yield ("artist_name", a)


def emit_tvmaze_shows(row):
    v = _clean(row.get("name")) or _clean(row.get("tv_show_title"))
    if v:
        yield ("tv_show_title", v)
    for g in _maybe_list(row.get("genres")):
        yield ("tv_genre", g)
    net = _clean(row.get("network_name"))
    if net:
        yield ("tv_network", net)
    dec = _decade_of(row.get("premiered"))
    if dec:
        yield ("release_decade", dec)
    yr = re.search(r"(\d{4})", _clean(row.get("premiered")))
    if yr:
        yield ("release_year", yr.group(1))


# IMDb ``title.basics``-style ``titleType`` → slot label.  ``movie``/``tvMovie``
# → movie_title, the series types → tv_show_title, shorts → short_film_title,
# games → game_title.  Other types (tvEpisode, video, …) are ignored.
_IMDB_TYPE_TO_SLOT: Dict[str, str] = {
    "movie": "movie_title",
    "tvmovie": "movie_title",
    "tvseries": "tv_show_title",
    "tvminiseries": "tv_show_title",
    "short": "short_film_title",
    "tvshort": "short_film_title",
    "videogame": "game_title",
}
# slot → the genre pool a title's IMDb ``genres`` feed (besides content_genre)
_IMDB_SLOT_TO_GENRE_POOL: Dict[str, str] = {
    "movie_title": "movie_genre",
    "tv_show_title": "tv_genre",
    "game_title": "game_genre",
}


def emit_imdb_titles(row):
    """IMDb title.basics-style rows → the right title slot by ``titleType``.

    The authoritative, type-split title source (far larger than the wikidata
    fallback).  ``isAdult == 1`` titles are routed to the adult pool (kept OUT of
    the clean movie/tv pools); ``genres`` feed ``content_genre`` (+ the per-type
    genre pool) and ``startYear`` feeds ``release_year`` / ``release_decade``.

    Tolerant of column-name variants (``titleType``/``title_type``,
    ``primaryTitle``/``title``) so it keeps working as the upload firms up.
    """
    ttype = str(row.get("titleType") or row.get("title_type") or "").strip().lower()
    slot = _IMDB_TYPE_TO_SLOT.get(ttype)
    if not slot:
        return
    title = (_clean(row.get("primaryTitle")) or _clean(row.get("primary_title"))
             or _clean(row.get("title")) or _clean(row.get("originalTitle")))
    if not title:
        return

    is_adult = str(row.get("isAdult") or row.get("is_adult") or "").strip() in (
        "1", "true", "True")
    if is_adult:
        # never leak adult titles into the clean pools
        yield ("adult_title", title)
    else:
        yield (slot, title)
        genre_pool = _IMDB_SLOT_TO_GENRE_POOL.get(slot)
        # IMDb ``genres`` is a comma-joined string ("Sci-Fi,Thriller"); split it.
        raw_genres = row.get("genres")
        parts = []
        for chunk in _maybe_list(raw_genres):
            parts.extend(str(chunk).split(","))
        for g in parts:
            g = _clean(g)
            if not g or g == "\\N":
                continue
            yield ("content_genre", g)
            if genre_pool:
                yield (genre_pool, g)
        year = _clean(row.get("startYear") or row.get("start_year"))
        m = re.search(r"(\d{4})", year)
        if m and m.group(1) != "0000":
            yield ("release_year", m.group(1))
            yield ("release_decade", f"{(int(m.group(1)) // 10) * 10}s")


def _is_hentai(row) -> bool:
    """True for adult anime/manga — splits the adult subset out of anime_title.

    Uses the explicit ``is_adult`` boolean when present, else a ``Hentai`` tag in
    the genres / tags / themes / demographics (case-insensitive).  Keeping these
    out of the clean ``anime_title`` pool prevents a normal "watch an anime"
    template from filling in an adult title and being mislabelled.
    """
    if str(row.get("is_adult")).strip().lower() in ("true", "1"):
        return True
    blob = " ".join(str(row.get(f, "")) for f in
                    ("genres", "tags", "themes", "demographics"))
    return "hentai" in blob.lower()


def emit_anilist_anime(row):
    hentai = _is_hentai(row)
    title_label = "hentai_title" if hentai else "anime_title"
    studio_label = "hentai_studio" if hentai else "anime_studio"
    for f in ("title_romaji", "title_english", "title_native", "anime_title"):
        v = _clean(row.get(f))
        if v:
            yield (title_label, v)
    for s in _maybe_list(row.get("studios")):
        yield (studio_label, s)
    yr = _clean(row.get("season_year"))
    m = re.search(r"(\d{4})", yr)
    if m:
        yield ("release_year", m.group(1))
        yield ("release_decade", f"{int(m.group(1)) // 10 * 10}s")


def emit_jikan_manga(row):
    # Manga is READ (paged) → comic_title, distinct from anime (watched).
    # Adult manga still routes to the hentai pool.
    hentai = _is_hentai(row)
    title_label = "hentai_title" if hentai else "comic_title"
    for f in ("title", "title_english", "anime_title"):
        v = _clean(row.get(f))
        if v:
            yield (title_label, v)
    for a in _maybe_list(row.get("aliases")):
        yield (title_label, a)
    if not hentai:
        for g in _maybe_list(row.get("genres"))[:3]:
            yield ("comic_genre", g)


def emit_librivox(row):
    """LibriVox = narrated audio → AUDIOBOOK (title + author + narrator)."""
    title = _clean(row.get("title")) or _clean(row.get("audiobook_title"))
    if title:
        yield ("audiobook_title", title)
    for a in _maybe_list(row.get("authors") or row.get("author")):
        yield ("audiobook_author", a)
    for r in _maybe_list(row.get("readers")):
        yield ("audiobook_narrator", r)
    for g in _maybe_list(row.get("genres")):
        yield ("audiobook_genre", g)


def emit_books(row):
    """Gutenberg / OpenLibrary = readable text → BOOK (title + author + genre)."""
    title = _clean(row.get("title")) or _clean(row.get("book_title"))
    if title:
        yield ("book_title", title)
    for a in _maybe_list(row.get("authors") or row.get("author")):
        yield ("book_author", a)
    for s in _maybe_list(row.get("subjects"))[:3]:
        yield ("book_genre", s)
    m = re.search(r"(\d{4})", _clean(row.get("first_publish_year")))
    if m:
        yield ("release_year", m.group(1))
    pub = _clean(row.get("publisher"))
    if pub:
        yield ("record_label", pub)


def emit_steam_games(row):
    v = _clean(row.get("name")) or _clean(row.get("game_title"))
    if v:
        yield ("game_title", v)
    for g in _maybe_list(row.get("genres")):
        yield ("game_genre", g)
    m = re.search(r"(\d{4})", _clean(row.get("release_date")))
    if m:
        yield ("release_year", m.group(1))


def emit_radiobrowser(row):
    v = _clean(row.get("name")) or _clean(row.get("radio_station"))
    if v:
        yield ("radio_station", v)
    for t in _maybe_list(row.get("tags"))[:3]:
        yield ("radio_genre", t)


def emit_podcasts(row):
    v = _clean(row.get("title")) or _clean(row.get("podcast_title"))
    if v:
        yield ("podcast_title", v)
    host = _clean(row.get("author")) or _clean(row.get("podcast_host"))
    if host:
        yield ("podcast_host", host)
    for g in _maybe_list(row.get("genres"))[:3]:
        yield ("podcast_genre", g)


def _emit_movie_crew(label: str) -> Callable:
    def emit(row):
        v = _clean(row.get("name")) or _clean(row.get(label))
        if v:
            yield (label, v)
    return emit


# wikidata entity_type → slot label.  Films become real ``movie_title`` (this is
# what replaces fabricated movie titles); the rest route to their natural label.
WIKIDATA_TYPE_TO_LABEL: Dict[str, str] = {
    "film": "movie_title",
    "animated_film": "movie_title",
    "short_film": "short_film_title",
    "documentary": "documentary_title",
    "silent_film": "silent_movie_title",
    "tv_series": "tv_show_title",
    "television_series": "tv_show_title",
    "miniseries": "tv_show_title",
    "web_series": "tv_show_title",
    "animated_series": "cartoon_title",
    "anime_series": "anime_title",
    "anime_studio": "anime_studio",
    "manga": "anime_title",
    "comic_book_series": "visual_story_title",
    "comic_book": "visual_story_title",
    "tv_channel": "tv_channel",
    "youtube_channel": "youtube_channel",
    "radio_station": "radio_station",
    "radio_program": "radio_drama_title",
    "radio_drama": "radio_drama_title",
    "audio_drama": "radio_drama_title",
    "stage_play": "radio_drama_title",
    "podcast": "podcast_title",
    "podcaster": "podcast_host",
    "audiobook": "audiobook_title",
    "novelist": "audiobook_author",
    "narrator": "audiobook_narrator",
    "book_publisher": "record_label",
    "record_label": "record_label",
    "music_band": "artist_name",
    "performing_arts_ensemble": "artist_name",
    "conductor": "artist_name",
    "dj": "artist_name",
    "lyricist": "artist_name",
    "music_genre": "music_genre",
    "film_genre": "video_genre",
    "tv_genre": "tv_genre",
    "literary_genre": "audiobook_genre",
    "anime_genre": "video_genre",
    "video_game_genre": "game_genre",
    "board_game": "game_title",
    "video_game_series": "game_title",
    "visual_novel": "game_title",
    "voice_actor": "movie_actor",
    "comedian": "movie_actor",
    "television_host": "movie_actor",
    "stunt_performer": "movie_actor",
    "radio_actor": "movie_actor",
}


def emit_wikidata(row):
    et = _clean(row.get("entity_type"))
    label = WIKIDATA_TYPE_TO_LABEL.get(et)
    name = _clean(row.get("label_en")) or _clean(row.get("name")) or _clean(row.get("title"))
    if label and name:
        yield (label, name)


# ---- adult content-filter sets (detect-to-block ONLY) ---------------------
# Beyond the performer name, the physical-attribute columns become slot pools so
# the content filter learns to fire on a DESCRIPTION ("porn with red hair") and
# not only on a named performer (which would be trivially evaded).  These are
# detect-to-block training signals, never content provision.

def _enum_clean(v) -> str:
    """Tidy an ALL-CAPS / enum-ish attribute value into natural lowercase."""
    s = _clean(v)
    if not s or s.lower() in ("unknown", "na", "n/a", "other"):
        return ""
    return s.replace("_", " ").lower()


def emit_unified_performers(row):
    """``TigreGotico/media-metadata-adult-performers`` — canonical UNIFIED set.

    Supersedes the per-source stashdb / iafd / freeones / boobpedia / thenude
    performer merges: this dataset is already cross-deduplicated across all those
    sources (the ``sources`` array records provenance).  ``name`` + ``aliases`` →
    ``pornstar``; the physical-attribute columns feed the detect-to-block
    description pools so the content filter fires on a DESCRIPTION
    ("porn with red hair"), not only on a named performer.  Detect-to-block
    training signals only — never content provision.
    """
    v = _clean(row.get("name"))
    if v:
        yield ("pornstar", v)
    for a in _maybe_list(row.get("aliases")):
        yield ("pornstar", a)
    for col, label in (("eye_color", "adult_eye_color"),
                       ("hair_color", "adult_hair_color"),
                       ("ethnicity", "adult_ethnicity"),
                       ("country", "adult_country"),
                       ("breast_type", "adult_body_type")):
        val = _enum_clean(row.get(col))
        if val:
            yield (label, val)
    # cup size is a body descriptor too (e.g. "DD") — emit lower-cased
    cup = _enum_clean(row.get("cup_size"))
    if cup and len(cup) <= 4:
        yield ("adult_body_type", cup)


def emit_iafd_titles(row):
    """Real adult film titles + director/studio → adult_title / adult_studio."""
    t = _clean(row.get("title"))
    if t:
        yield ("adult_title", t)
    st = _clean(row.get("studio")) or _clean(row.get("distributor"))
    if st:
        yield ("adult_studio", st)


def emit_hanime(row):
    """hanime.tv hentai → hentai_title (+ brand → hentai_studio, tags)."""
    t = _clean(row.get("name")) or _clean(row.get("title"))
    if t:
        yield ("hentai_title", t)
    brand = _clean(row.get("brand"))
    if brand:
        yield ("hentai_studio", brand)


def emit_mal_hentai(row):
    """MyAnimeList hentai → hentai_title (+ studios → hentai_studio)."""
    for f in ("title", "title_english"):
        v = _clean(row.get(f))
        if v:
            yield ("hentai_title", v)
    for s in _maybe_list(row.get("studios")):
        yield ("hentai_studio", s)


def emit_hentaisea(row):
    """hentaisea → hentai_title."""
    t = _clean(row.get("title"))
    if t:
        yield ("hentai_title", t)


def emit_iafd_distributors(row):
    v = _clean(row.get("name"))
    if v:
        yield ("adult_studio", v)


# ---------------------------------------------------------------------------
# Source registry
#
# Each spec: local jsonl basename (in metadatarr cache) and/or a HF dataset id,
# plus the emitter.  ``local`` is tried first; ``hf`` is the fallback / only
# source when not cached locally.  ``adult`` flags the content-filter sets.
# ---------------------------------------------------------------------------

class Spec:
    def __init__(self, name, emit, local=None, hf=None, adult=False):
        self.name = name
        self.emit = emit
        self.local = local       # metadatarr jsonl basename
        self.hf = hf             # TigreGotico/<id>
        self.adult = adult


SOURCE_SPECS: List[Spec] = [
    # ---- music artists / releases ----
    # The canonical UNIFIED artist set replaces the per-source musicbrainz /
    # audiodb / jazz / prog / metal / classical merge: it is already
    # cross-deduplicated across all those sources, so we ingest it directly into
    # ``artist_name`` (+ ``music_genre``) instead of re-merging the raw sources.
    Spec("unified-artists", emit_unified_artists, hf="media-metadata-artists"),
    # MusicBrainz *releases* are kept (albums + the album↔artist relation feed) —
    # they are NOT in the artist set above.
    Spec("musicbrainz-releases", emit_musicbrainz_releases,
         local="musicbrainz_releases", hf="musicbrainz-releases"),
    # ---- video / shows / anime ----
    # IMDb is the authoritative, type-split title source — listed FIRST so its
    # movie_title / tv_show_title / short_film_title / game_title fill the
    # (capped) pools before the smaller wikidata fallback.  The dataset may be
    # empty while it is still uploading; ``ingest`` tolerates that (it yields no
    # rows and the existing pools are used unchanged).
    Spec("imdb-titles", emit_imdb_titles, hf="media-metadata-imdb-titles"),
    Spec("tvmaze-shows", emit_tvmaze_shows,
         local="tvmaze_shows", hf="media-metadata-tvmaze-shows"),
    Spec("anilist-anime", emit_anilist_anime,
         local="anilist_anime", hf="media-metadata-anilist-anime"),
    Spec("jikan-manga", emit_jikan_manga,
         local="jikan_manga", hf="media-metadata-jikan-manga"),
    # ---- books / audiobooks ----
    Spec("gutenberg-books", emit_books,
         local="gutenberg_books", hf="media-metadata-gutenberg-books"),
    Spec("librivox-audiobooks", emit_librivox,
         local="librivox_audiobooks", hf="media-metadata-librivox-audiobooks"),
    Spec("openlibrary-books", emit_books,
         local="openlibrary_books", hf="media-metadata-openlibrary-books"),
    # ---- games / radio / podcasts ----
    Spec("steam-games", emit_steam_games,
         local="steam_games", hf="media-metadata-steam-games"),
    Spec("radiobrowser-stations", emit_radiobrowser,
         local="radiobrowser_stations", hf="media-metadata-radiobrowser-stations"),
    Spec("podcastindex-podcasts", emit_podcasts,
         local="podcastindex_podcasts", hf="media-metadata-podcastindex-podcasts"),
    Spec("listennotes-podcasts", emit_podcasts,
         local="listennotes_podcasts", hf="media-metadata-listennotes-podcasts"),
    # ---- wikidata (split by entity_type; films → real movie_title) ----
    Spec("wikidata-entities", emit_wikidata,
         local="wikidata_entities", hf="media-metadata-wikidata-entities"),
    # ---- movie crew ----
    Spec("movie_actors", _emit_movie_crew("movie_actor"), hf="movie_actors"),
    Spec("movie_directors", _emit_movie_crew("movie_director"), hf="movie_directors"),
    Spec("movie_producers", _emit_movie_crew("movie_producer"), hf="movie_producers"),
    Spec("movie_writers", _emit_movie_crew("movie_writer"), hf="movie_writers"),
    Spec("movie_composers", _emit_movie_crew("movie_composer"), hf="movie_composers"),
    # ---- dedicated hentai sets → hentai_title / hentai_studio ----
    Spec("hanime", emit_hanime,
         hf="adult-metadata-hanime", adult=True),
    Spec("mal-hentai", emit_mal_hentai,
         hf="adult-metadata-mal-hentai", adult=True),
    Spec("hentaisea", emit_hentaisea,
         hf="adult-metadata-hentaisea", adult=True),
    # ---- adult (content-filter detect-to-block ONLY) ----
    # The canonical UNIFIED performer set replaces the per-source stashdb / iafd /
    # freeones / boobpedia / thenude performer merge: it is already
    # cross-deduplicated across all those rosters, so we ingest it directly into
    # the ``pornstar`` + physical-attribute pools.  iafd TITLES / DISTRIBUTORS
    # (adult_title / adult_studio) are NOT in the performer set and are kept.
    Spec("unified-performers", emit_unified_performers,
         hf="media-metadata-adult-performers", adult=True),
    Spec("iafd-titles", emit_iafd_titles,
         hf="adult-metadata-iafd-titles", adult=True),
    Spec("iafd-distributors", emit_iafd_distributors,
         hf="adult-metadata-iafd-distributors", adult=True),
]


# ---------------------------------------------------------------------------
# Relational record emitters
#
# Besides the flat per-label pools, the multi-field sources below also emit ONE
# coherent record per row so a template that fills several slots of a domain
# ("{album_name} by {artist_name}") can draw them from the SAME real entity.
# Each emitter yields ``(group_name, record_dict)`` where the record's keys are
# the *template field names* the matching ``RelationalGroup`` in build_dataset
# binds to slots.  Sources with no usable multi-field record yield nothing.
# ---------------------------------------------------------------------------

def rel_musicbrainz_releases(row):
    title = _clean(row.get("title"))
    artists = _maybe_list(row.get("artist_names"))
    if title and artists:
        rec = {"album": title, "artist": artists[0]}
        yr = re.search(r"(\d{4})", _clean(row.get("date") or row.get("first_release_date")))
        if yr:
            rec["year"] = yr.group(1)
        yield ("music", rec)


def rel_anilist(row):
    if _is_hentai(row):
        return
    title = (_clean(row.get("title_english")) or _clean(row.get("title_romaji"))
             or _clean(row.get("anime_title")))
    studios = _maybe_list(row.get("studios"))
    if not title:
        return
    rec = {"anime_title": title}
    if studios:
        rec["anime_studio"] = studios[0]
    yr = re.search(r"(\d{4})", _clean(row.get("season_year")))
    if yr:
        rec["year"] = yr.group(1)
    for g in _maybe_list(row.get("genres"))[:1]:
        rec["anime_genre"] = g
    if len(rec) > 1:
        yield ("anime", rec)


def rel_tvmaze(row):
    show = _clean(row.get("name")) or _clean(row.get("tv_show_title"))
    net = _clean(row.get("network_name"))
    if not show:
        return
    rec = {"tv_show": show}
    if net:
        rec["tv_network"] = net
    for g in _maybe_list(row.get("genres"))[:1]:
        rec["tv_genre"] = g
    yr = re.search(r"(\d{4})", _clean(row.get("premiered")))
    if yr:
        rec["year"] = yr.group(1)
    if len(rec) > 1:
        yield ("tv", rec)


def _person_name(v) -> str:
    """Format a person field that may be a ``{first_name,last_name}`` dict."""
    if isinstance(v, dict):
        nm = " ".join(p for p in (_clean(v.get("first_name")),
                                  _clean(v.get("last_name"))) if p)
        return nm or _clean(v.get("name"))
    return _clean(v)


def _first_person(v) -> str:
    """First person name from a list/ndarray/json of names-or-name-dicts."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            v = v.tolist()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        for x in v:
            nm = _person_name(x)
            if nm:
                return nm
        return ""
    s = str(v).strip()
    if s.startswith("[") or s.startswith("{"):
        try:
            parsed = ast.literal_eval(s)
            return _first_person(parsed if isinstance(parsed, list) else [parsed])
        except (ValueError, SyntaxError):
            pass
    return _clean(v)


def rel_librivox(row):
    title = _clean(row.get("title")) or _clean(row.get("audiobook_title"))
    author = _first_person(row.get("authors") or row.get("author"))
    reader = _first_person(row.get("readers"))
    if not title:
        return
    rec = {"audiobook_title": title}
    if author:
        rec["audiobook_author"] = author
    if reader:
        rec["audiobook_narrator"] = reader
    for g in _maybe_list(row.get("genres"))[:1]:
        rec["audiobook_genre"] = g
    if len(rec) > 1:
        yield ("audiobook", rec)


def rel_books(row):
    title = _clean(row.get("title")) or _clean(row.get("book_title"))
    author = _first_person(row.get("authors") or row.get("author"))
    if not (title and author):
        return
    rec = {"book_title": title, "book_author": author}
    for s in _maybe_list(row.get("subjects"))[:1]:
        rec["book_genre"] = s
    m = re.search(r"(\d{4})", _clean(row.get("first_publish_year")))
    if m:
        rec["year"] = m.group(1)
    yield ("book", rec)


def rel_podcasts(row):
    title = _clean(row.get("title")) or _clean(row.get("podcast_title"))
    host = _clean(row.get("author")) or _clean(row.get("podcast_host"))
    if not (title and host):
        return
    rec = {"podcast_title": title, "podcast_host": host}
    for g in _maybe_list(row.get("genres"))[:1]:
        rec["podcast_genre"] = g
    yield ("podcast", rec)


# spec-name -> relational emitter (only the multi-field sources have one)
RELATIONAL_EMITTERS: Dict[str, Callable] = {
    "musicbrainz-releases": rel_musicbrainz_releases,
    "anilist-anime": rel_anilist,
    "tvmaze-shows": rel_tvmaze,
    "librivox-audiobooks": rel_librivox,
    "openlibrary-books": rel_books,
    "gutenberg-books": rel_books,
    "podcastindex-podcasts": rel_podcasts,
    "listennotes-podcasts": rel_podcasts,
}


def build_relations(relational_dir: str, cap: int = 300_000,
                    prefer_local: bool = True) -> Dict[str, int]:
    """Write ``data/relational/<group>.jsonl`` coherent records per domain.

    Iterates only the multi-field sources (RELATIONAL_EMITTERS), de-duplicating
    each group's records on their title-ish first field, capped.  These power the
    coherent multi-slot fills in build_dataset (music album↔artist, tv show↔
    network, audiobook title↔author↔narrator, …).  IMDb movie/episode relations
    are built separately by ``training/imdb_relations.py``.
    """
    os.makedirs(relational_dir, exist_ok=True)
    groups: Dict[str, Dict[str, dict]] = defaultdict(dict)
    by_name = {s.name: s for s in SOURCE_SPECS}
    for name, emit in RELATIONAL_EMITTERS.items():
        spec = by_name.get(name)
        if spec is None:
            continue
        n = 0
        try:
            for row in iter_source(spec, prefer_local=prefer_local):
                for group, rec in emit(row):
                    g = groups[group]
                    if len(g) >= cap:
                        continue
                    key = next(iter(rec.values())).lower()
                    g.setdefault(key, rec)
                    n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[rel:{name}] ERROR {type(exc).__name__}: {exc}", flush=True)
        print(f"[rel:{name}] {n:,} records", flush=True)
    counts: Dict[str, int] = {}
    for group, recs in sorted(groups.items()):
        path = os.path.join(relational_dir, f"{group}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for rec in recs.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[group] = len(recs)
        print(f"  {group:<12}: {len(recs):>9,} → {path}", flush=True)
    return counts


# ---------------------------------------------------------------------------
# Source iterators
# ---------------------------------------------------------------------------

def _iter_local(basename: str) -> Iterable[dict]:
    path = os.path.join(METADATARR_CACHE, f"{basename}.jsonl")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_hf(repo: str) -> Iterable[dict]:
    """Yield rows from a HF dataset, downloading the data file directly.

    Avoids ``load_dataset`` streaming hangs by pulling the parquet/csv file via
    ``hf_hub_download`` and iterating it with pandas.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    full = f"TigreGotico/{repo}"
    files = [f for f in HfApi().list_repo_files(full, repo_type="dataset")
             if f.endswith((".parquet", ".csv", ".jsonl"))]
    files.sort()
    for f in files:
        local = hf_hub_download(full, f, repo_type="dataset")
        if f.endswith(".parquet"):
            df = pd.read_parquet(local)
            for rec in df.to_dict("records"):
                yield rec
        elif f.endswith(".csv"):
            for chunk in pd.read_csv(local, dtype=str, keep_default_na=False,
                                     chunksize=50000):
                for rec in chunk.to_dict("records"):
                    yield rec
        else:  # jsonl
            yield from _iter_local_path(local)


def _iter_local_path(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def iter_source(spec: Spec, prefer_local: bool = True) -> Iterable[dict]:
    if prefer_local and spec.local:
        path = os.path.join(METADATARR_CACHE, f"{spec.local}.jsonl")
        if os.path.isfile(path):
            yield from _iter_local(spec.local)
            return
    if spec.hf:
        yield from _iter_hf(spec.hf)


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

def ingest(
    output_dir: str,
    only: Optional[List[str]] = None,
    cap: int = 200_000,
    prefer_local: bool = True,
) -> Dict[str, int]:
    """Run every (or selected) source and write per-label deduped CSVs.

    Returns ``{label: row_count}``.
    """
    os.makedirs(output_dir, exist_ok=True)
    # label → {lower-cased value: original-cased value}  (dedup, cap)
    pools: Dict[str, Dict[str, str]] = defaultdict(dict)
    capped: set = set()

    specs = SOURCE_SPECS
    if only:
        wanted = set(only)
        specs = [s for s in specs
                 if s.name in wanted
                 or (s.adult and "adult" in wanted)
                 or (not s.adult and "media" in wanted)]

    for spec in specs:
        n_in = n_out = 0
        src_kind = "local" if (prefer_local and spec.local and os.path.isfile(
            os.path.join(METADATARR_CACHE, f"{spec.local}.jsonl"))) else "hf"
        print(f"[{spec.name}] reading ({src_kind}) …", flush=True)
        try:
            for row in iter_source(spec, prefer_local=prefer_local):
                n_in += 1
                for label, value in spec.emit(row):
                    if not value or label in capped:
                        continue
                    key = value.lower()
                    pool = pools[label]
                    if key not in pool:
                        pool[key] = value
                        n_out += 1
                        if len(pool) >= cap:
                            capped.add(label)
        except Exception as exc:  # noqa: BLE001 - keep ingesting other sources
            print(f"[{spec.name}] ERROR: {type(exc).__name__}: {exc}", flush=True)
        print(f"[{spec.name}] {n_in:,} rows → {n_out:,} new entity values", flush=True)

    counts: Dict[str, int] = {}
    for label, pool in sorted(pools.items()):
        path = os.path.join(output_dir, f"{label}.csv")
        # merge with any pre-existing file (idempotent re-runs / partial --only)
        existing: Dict[str, str] = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                for rec in r:
                    v = (rec.get("value") or "").strip()
                    if v:
                        existing.setdefault(v.lower(), v)
        merged = dict(existing)
        for k, v in pool.items():
            if len(merged) >= cap:
                break
            merged.setdefault(k, v)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["value"])
            for v in merged.values():
                w.writerow([v])
        counts[label] = len(merged)
        print(f"  {label:<22}: {len(merged):>9,} → {path}", flush=True)

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest the TigreGotico media-metadata collection into entity pools",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--output", default="data/entities",
                    help="output dir for <label>.csv files (default: data/entities)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to source names, or 'adult' / 'media' groups")
    ap.add_argument("--cap", type=int, default=200_000,
                    help="max entities per label (default: 200000)")
    ap.add_argument("--no-local", action="store_true",
                    help="ignore the metadatarr cache; always read from HuggingFace")
    ap.add_argument("--relations", action="store_true",
                    help="also write data/relational/<group>.jsonl coherent records")
    ap.add_argument("--relational-dir",
                    default=os.path.join(REPO_ROOT, "data", "relational"),
                    help="output dir for relational jsonl (default: data/relational)")
    args = ap.parse_args()

    counts = ingest(args.output, only=args.only, cap=args.cap,
                    prefer_local=not args.no_local)
    total = sum(counts.values())
    print(f"\nDone: {total:,} entities across {len(counts)} labels → {args.output}")

    if args.relations:
        print("\nBuilding relational records …")
        rc = build_relations(args.relational_dir, cap=args.cap,
                             prefer_local=not args.no_local)
        print(f"Relational groups: {sum(rc.values()):,} records across "
              f"{len(rc)} groups → {args.relational_dir}")


if __name__ == "__main__":
    main()
