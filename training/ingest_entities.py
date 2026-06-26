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
    """Coerce a field that may be a python/json list-string or a scalar."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v if _clean(x)]
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


def _iter_dicts(v) -> List[dict]:
    """Coerce a field that may be a list / ndarray / json-string of dicts."""
    if v is None:
        return []
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            v = v.tolist()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return [x for x in v if isinstance(x, dict)]
    s = str(v).strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except (ValueError, SyntaxError):
            pass
    return []


def _emit_artist(name_field: str) -> Callable:
    def emit(row):
        v = _clean(row.get(name_field))
        if v:
            yield ("artist_name", v)
    return emit


def _decade_of(year_val) -> str:
    s = _clean(year_val)
    m = re.search(r"(\d{4})", s)
    if not m:
        return ""
    return f"{int(m.group(1)) // 10 * 10}s"


def emit_musicbrainz_artists(row):
    v = _clean(row.get("name"))
    if v:
        yield ("artist_name", v)
    for a in _maybe_list(row.get("aliases")):
        yield ("artist_name", a)


def emit_musicbrainz_releases(row):
    title = _clean(row.get("title"))
    if title:
        # release groups are albums; tracks are not in this dump
        yield ("album_name", title)
    for a in _maybe_list(row.get("artist_names")):
        yield ("artist_name", a)


def emit_audiodb_artists(row):
    v = _clean(row.get("name"))
    if v:
        yield ("artist_name", v)
    alt = _clean(row.get("alternate_name"))
    if alt:
        yield ("artist_name", alt)
    g = _clean(row.get("genre")) or _clean(row.get("style"))
    if g:
        yield ("music_genre", g)
    lbl = _clean(row.get("label"))
    if lbl:
        yield ("record_label", lbl)


def emit_jazz_artists(row):
    v = _clean(row.get("artist")) or _clean(row.get("name"))
    if v:
        yield ("artist_name", v)
    g = _clean(row.get("genre"))
    if g:
        yield ("music_genre", g)


def emit_prog_artists(row):
    v = _clean(row.get("artist")) or _clean(row.get("name"))
    if v:
        yield ("artist_name", v)
    g = _clean(row.get("genre"))
    if g:
        yield ("music_genre", g)


def emit_metal_archives(row):
    v = (_clean(row.get("name")) or _clean(row.get("band_name"))
         or _clean(row.get("artist")) or _clean(row.get("band")))
    if v:
        yield ("artist_name", v)
    g = _clean(row.get("genre"))
    if g:
        yield ("music_genre", g)
    lbl = _clean(row.get("label"))
    if lbl:
        yield ("record_label", lbl)


def emit_classical_composers(row):
    v = (_clean(row.get("name")) or _clean(row.get("composer"))
         or _clean(row.get("artist")))
    if v:
        yield ("artist_name", v)


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


def emit_stashdb_performers(row):
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
    # tattoos / piercings are list-ish; emit a generic descriptor when present
    if _maybe_list(row.get("tattoos")):
        yield ("adult_marking", "tattoos")
    if _maybe_list(row.get("piercings")):
        yield ("adult_marking", "piercings")


def emit_iafd_performers(row):
    v = _clean(row.get("name"))
    if v:
        yield ("pornstar", v)
    for a in _maybe_list(row.get("aliases")):
        yield ("pornstar", a)
    # filmography titles → adult_title (detect-to-block titles)
    for it in _iter_dicts(row.get("filmography"))[:20]:
        t = _clean(it.get("title"))
        if t:
            yield ("adult_title", t)


# Ethnicity / hair / eye hints found inside free-text category / description
# blobs (boobpedia ``categories``, freeones ``professions``, etc.).
_ETHNICITY_HINTS = ("caucasian", "asian", "black", "latina", "latin", "ebony",
                    "hispanic", "indian", "arab", "interracial", "white")
_HAIR_HINTS = ("blonde", "brunette", "redhead", "black hair", "brown hair",
               "raven", "ginger")


def _emit_performer(row):
    """Shared emitter for the freeones / boobpedia / thenude performer sets.

    Mines name + aliases → ``pornstar`` and every usable physical / descriptive
    attribute into its pool (detect-to-block training signals only). Robust to
    column-name differences across the three sources.
    """
    for col in ("name",):
        v = _clean(row.get(col))
        if v:
            yield ("pornstar", v)
    for col in ("aliases", "all_names"):
        for a in _maybe_list(row.get(col)):
            yield ("pornstar", a)
    nat = _enum_clean(row.get("nationality")) or _enum_clean(row.get("ethnicity"))
    if nat and len(nat) > 2:
        yield ("adult_country", nat)
    for col in ("eye_color", "hair_color", "ethnicity", "build", "body_type"):
        val = _enum_clean(row.get(col))
        if not val:
            continue
        if "eye" in col:
            yield ("adult_eye_color", val)
        elif "hair" in col:
            yield ("adult_hair_color", val)
        elif col == "ethnicity":
            yield ("adult_ethnicity", val)
        else:
            yield ("adult_body_type", val)
    # mine free-text blobs (categories / description / professions / tags)
    blob = " ".join(str(row.get(c, "")) for c in
                    ("categories", "description", "professions", "tags",
                     "performances")).lower()
    for hint in _ETHNICITY_HINTS:
        if hint in blob:
            yield ("adult_ethnicity", "latina" if hint == "latin" else hint)
            break
    for hint in _HAIR_HINTS:
        if hint in blob:
            yield ("adult_hair_color", hint.replace(" hair", ""))
            break


def emit_freeones_performers(row):
    yield from _emit_performer(row)


def emit_boobpedia_performers(row):
    yield from _emit_performer(row)


def emit_thenude_performers(row):
    yield from _emit_performer(row)


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
    # The curated genre-specific archives are listed BEFORE the bulk MusicBrainz
    # dump so they always make it into the (capped) ``artist_name`` pool — the
    # 1.5 M-row MusicBrainz set would otherwise saturate the cap on its own and
    # crowd out jazz / prog / metal / classical diversity.
    Spec("jazz-artists", emit_jazz_artists, hf="media-metadata-jazz-artists"),
    Spec("progarchives-artists", emit_prog_artists,
         hf="media-metadata-progarchives-artists"),
    Spec("metal-archives", emit_metal_archives,
         hf="media-metadata-metal-archives"),
    Spec("classical-composers", emit_classical_composers,
         hf="media-metadata-classical-composers"),
    Spec("audiodb-artists", emit_audiodb_artists,
         local="audiodb_artists", hf="audiodb-artists"),
    Spec("musicbrainz-artists", emit_musicbrainz_artists,
         local="musicbrainz_artists", hf="musicbrainz-artists"),
    Spec("musicbrainz-releases", emit_musicbrainz_releases,
         local="musicbrainz_releases", hf="musicbrainz-releases"),
    # ---- video / shows / anime ----
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
    # Performer sets are deduplicated into one ``pornstar`` pool (case-insensitive),
    # so overlapping rosters across stashdb / iafd / freeones / boobpedia / thenude
    # do not inflate the pool.
    Spec("stashdb-performers", emit_stashdb_performers,
         hf="adult-metadata-stashdb-performers", adult=True),
    Spec("iafd-performers", emit_iafd_performers,
         hf="adult-metadata-iafd-performers", adult=True),
    Spec("iafd-titles", emit_iafd_titles,
         hf="adult-metadata-iafd-titles", adult=True),
    Spec("iafd-distributors", emit_iafd_distributors,
         hf="adult-metadata-iafd-distributors", adult=True),
    Spec("freeones-performers", emit_freeones_performers,
         hf="adult-metadata-freeones-performers", adult=True),
    Spec("boobpedia-performers", emit_boobpedia_performers,
         hf="adult-metadata-boobpedia-performers", adult=True),
    Spec("thenude-performers", emit_thenude_performers,
         hf="adult-metadata-thenude-performers", adult=True),
]


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
    args = ap.parse_args()

    counts = ingest(args.output, only=args.only, cap=args.cap,
                    prefer_local=not args.no_local)
    total = sum(counts.values())
    print(f"\nDone: {total:,} entities across {len(counts)} labels → {args.output}")


if __name__ == "__main__":
    main()
