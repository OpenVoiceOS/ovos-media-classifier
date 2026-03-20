#!/usr/bin/env python3
"""Gather entity pools from all available sources into per-OCPEntityLabel CSV files.

Each source function returns ``{label: [values]}``.  ``gather_all()`` merges and
deduplicates these dicts, then writes one CSV per label to the output directory
plus a combined ``entities_combined.csv``.

Output schema for per-label files: ``value,source``
Output schema for combined file:   ``value,source,label``

Usage::

    python -m ovos_media_classifier.train.gather_entities
    python -m ovos_media_classifier.train.gather_entities \\
        --sources wikidata steam gutendex \\
        --output /tmp/entities

    # List available sources without running
    python -m ovos_media_classifier.train.gather_entities --list-sources
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from ovos_media_classifier.train import get_cache_dir, get_hf_cache_dir

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def get_entities_dir(output_dir: Optional[str] = None) -> str:
    """Return the default entities output directory."""
    return output_dir or os.path.join(get_cache_dir(), "entities")


# ---------------------------------------------------------------------------
# HuggingFace-backed source functions
# ---------------------------------------------------------------------------

def _gather_wikidata(hf_cache: str) -> Dict[str, List[str]]:
    """Load Jarbas/WikidataMediaEntities → all applicable OCPEntityLabel pools."""
    try:
        import datasets as _ds
        ds = _ds.load_dataset("Jarbas/WikidataMediaEntities", cache_dir=hf_cache,
                               split="train", trust_remote_code=True)
    except Exception as exc:
        LOG.warning("wikidata: failed to load — %s", exc)
        return {}

    result: Dict[str, List[str]] = {}

    # Column → label mapping used in this dataset
    col_map: Dict[str, str] = {
        "title":    "ocp_label",   # ocp_label column names the label
    }

    for row in ds:
        ocp_label = str(row.get("ocp_label", "") or "").strip()
        title = str(row.get("title", "") or "").strip()
        if ocp_label and title:
            result.setdefault(ocp_label, []).append(title)

        for col, lbl_col in [
            ("actor",    "movie_actor"),
            ("director", "movie_director"),
            ("producer", "movie_producer"),
            ("writer",   "movie_writer"),
            ("composer", "movie_composer"),
            ("artist",   "artist_name"),
            ("album",    "album_name"),
            ("author",   "audiobook_author"),
            ("narrator", "audiobook_narrator"),
            ("studio",   "movie_studio"),
            ("genre",    "music_genre"),
        ]:
            val = str(row.get(col, "") or "").strip()
            if val:
                for part in val.split("|"):
                    part = part.strip()
                    if part:
                        result.setdefault(lbl_col, []).append(part)

    return result


def _gather_metal_bands(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/metal-archives-bands → artist_name."""
    return _load_hf_column("TigreGotico/metal-archives-bands", hf_cache,
                            columns={"band_name": "artist_name",
                                     "name": "artist_name"})


def _gather_metal_tracks(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/metal-archives-tracks → track_name."""
    return _load_hf_column("TigreGotico/metal-archives-tracks", hf_cache,
                            columns={"title": "track_name",
                                     "song": "track_name",
                                     "name": "track_name"})


def _gather_jazz(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/jazz-music-archives → artist_name, track_name."""
    return _load_hf_column("TigreGotico/jazz-music-archives", hf_cache,
                            columns={"artist": "artist_name",
                                     "title":  "track_name",
                                     "song":   "track_name"})


def _gather_prog(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/prog-archives → artist_name, album_name."""
    return _load_hf_column("TigreGotico/prog-archives", hf_cache,
                            columns={"artist": "artist_name",
                                     "album":  "album_name",
                                     "title":  "album_name"})


def _gather_classic_composers(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/classic-composers → artist_name."""
    return _load_hf_column("TigreGotico/classic-composers", hf_cache,
                            columns={"name":     "artist_name",
                                     "composer": "artist_name"})


def _gather_trance(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/trance_tracks → track_name, artist_name."""
    return _load_hf_column("TigreGotico/trance_tracks", hf_cache,
                            columns={"title":  "track_name",
                                     "artist": "artist_name"})


def _gather_movie_actors(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/movie_actors → movie_actor."""
    return _load_hf_column("TigreGotico/movie_actors", hf_cache,
                            columns={"name":  "movie_actor",
                                     "actor": "movie_actor"})


def _gather_movie_directors(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/movie_directors → movie_director."""
    return _load_hf_column("TigreGotico/movie_directors", hf_cache,
                            columns={"name":     "movie_director",
                                     "director": "movie_director"})


def _gather_movie_writers(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/movie_writers → movie_writer."""
    return _load_hf_column("TigreGotico/movie_writers", hf_cache,
                            columns={"name":   "movie_writer",
                                     "writer": "movie_writer"})


def _gather_movie_producers(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/movie_producers → movie_producer."""
    return _load_hf_column("TigreGotico/movie_producers", hf_cache,
                            columns={"name":     "movie_producer",
                                     "producer": "movie_producer"})


def _gather_movie_composers(hf_cache: str) -> Dict[str, List[str]]:
    """Load TigreGotico/movie_composers → movie_composer."""
    return _load_hf_column("TigreGotico/movie_composers", hf_cache,
                            columns={"name":     "movie_composer",
                                     "composer": "movie_composer"})


def _load_hf_column(dataset_name: str, hf_cache: str,
                    columns: Dict[str, str]) -> Dict[str, List[str]]:
    """Generic HuggingFace loader: map dataset columns to OCPEntityLabel pools.

    Args:
        dataset_name: HuggingFace dataset identifier.
        hf_cache: Local cache directory.
        columns: ``{dataset_column: ocp_label}`` mapping (first matching column wins per row).

    Returns:
        ``{ocp_label: [values]}`` dict.
    """
    try:
        import datasets as _ds
        ds = _ds.load_dataset(dataset_name, cache_dir=hf_cache,
                               split="train", trust_remote_code=True)
    except Exception as exc:
        LOG.warning("%s: failed to load — %s", dataset_name, exc)
        return {}

    result: Dict[str, List[str]] = {}
    for row in ds:
        for col, label in columns.items():
            val = str(row.get(col, "") or "").strip()
            if val and val.lower() != "nan":
                result.setdefault(label, []).append(val)
    return result


# ---------------------------------------------------------------------------
# REST / HTTP source functions
# ---------------------------------------------------------------------------

def _gather_gutendex(max_pages: int = 50) -> Dict[str, List[str]]:
    """Paginate Gutendex API → audiobook_title, audiobook_author.

    Args:
        max_pages: Maximum pages to fetch (100 books/page).
    """
    result: Dict[str, List[str]] = {"audiobook_title": [], "audiobook_author": []}
    url: Optional[str] = "https://gutendex.com/books/?page=1"
    pages = 0
    while url and pages < max_pages:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            LOG.warning("gutendex: page %d failed — %s", pages + 1, exc)
            break
        for book in data.get("results", []):
            title = str(book.get("title", "") or "").strip()
            if title:
                result["audiobook_title"].append(title)
            for author in book.get("authors", []):
                name = str(author.get("name", "") or "").strip()
                if name:
                    result["audiobook_author"].append(name)
        url = data.get("next")
        pages += 1
        time.sleep(0.1)
    return result


def _gather_librivox(max_pages: int = 100) -> Dict[str, List[str]]:
    """Paginate LibriVox API → audiobook_title, audiobook_author, audiobook_narrator.

    Args:
        max_pages: Maximum pages to fetch (100 books/page).
    """
    result: Dict[str, List[str]] = {
        "audiobook_title": [], "audiobook_author": [], "audiobook_narrator": []
    }
    base = "https://librivox.org/api/feed/audiobooks"
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(base, params={"format": "json", "limit": 100,
                                               "offset": (page - 1) * 100},
                                 timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            LOG.warning("librivox: page %d failed — %s", page, exc)
            break
        books = data.get("books") or []
        if not books:
            break
        for book in books:
            title = str(book.get("title", "") or "").strip()
            if title:
                result["audiobook_title"].append(title)
            for author in book.get("authors") or []:
                first = str(author.get("first_name", "") or "").strip()
                last = str(author.get("last_name", "") or "").strip()
                name = f"{first} {last}".strip()
                if name:
                    result["audiobook_author"].append(name)
        time.sleep(0.1)
    return result


def _gather_radio_garden() -> Dict[str, List[str]]:
    """Fetch Radio Garden API → radio_station."""
    result: Dict[str, List[str]] = {"radio_station": []}
    try:
        resp = requests.get("https://radio.garden/api/ara/content/places",
                             timeout=15)
        resp.raise_for_status()
        places = resp.json().get("data", {}).get("list", [])
        for place in places:
            for channel in place.get("channels") or []:
                name = str(channel.get("title", "") or "").strip()
                if name:
                    result["radio_station"].append(name)
    except Exception as exc:
        LOG.warning("radio_garden: failed — %s", exc)
    return result


def _gather_anime_offline_db() -> Dict[str, List[str]]:
    """Fetch manami-project Anime Offline Database → anime_title, anime_studio."""
    result: Dict[str, List[str]] = {"anime_title": [], "anime_studio": []}
    url = ("https://raw.githubusercontent.com/manami-project/anime-offline-database"
           "/master/anime-offline-database-minified.json")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        LOG.warning("anime_offline_db: failed — %s", exc)
        return result
    for entry in data.get("data", []):
        title = str(entry.get("title", "") or "").strip()
        if title:
            result["anime_title"].append(title)
        for syn in entry.get("synonyms", []):
            syn = str(syn or "").strip()
            if syn:
                result["anime_title"].append(syn)
        for studio in entry.get("animationStudios") or entry.get("studios") or []:
            studio = str(studio or "").strip()
            if studio:
                result["anime_studio"].append(studio)
    return result


def _gather_steam() -> Dict[str, List[str]]:
    """Fetch Steam app list → game_title (no auth required)."""
    result: Dict[str, List[str]] = {"game_title": []}
    try:
        resp = requests.get(
            "https://api.steampowered.com/ISteamApps/GetAppList/v2/",
            timeout=20,
        )
        resp.raise_for_status()
        apps = resp.json().get("applist", {}).get("apps", [])
        for app in apps:
            name = str(app.get("name", "") or "").strip()
            if name and len(name) > 1:
                result["game_title"].append(name)
    except Exception as exc:
        LOG.warning("steam: failed — %s", exc)
    return result


def _gather_anilist(max_pages: int = 50) -> Dict[str, List[str]]:
    """Query AniList GraphQL API → anime_title, anime_studio.

    Args:
        max_pages: Maximum GraphQL pages to fetch (50 entries/page).
    """
    result: Dict[str, List[str]] = {"anime_title": [], "anime_studio": []}
    query = """
    query ($page: Int) {
      Page(page: $page, perPage: 50) {
        pageInfo { hasNextPage }
        media(type: ANIME) {
          title { romaji english native }
          studios(isMain: true) { nodes { name } }
        }
      }
    }
    """
    for page in range(1, max_pages + 1):
        try:
            resp = requests.post(
                "https://graphql.anilist.co",
                json={"query": query, "variables": {"page": page}},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {}).get("Page", {})
        except Exception as exc:
            LOG.warning("anilist: page %d failed — %s", page, exc)
            break
        for media in data.get("media", []):
            for title in (media.get("title") or {}).values():
                title = str(title or "").strip()
                if title:
                    result["anime_title"].append(title)
            for studio_node in (media.get("studios") or {}).get("nodes", []):
                name = str((studio_node or {}).get("name", "") or "").strip()
                if name:
                    result["anime_studio"].append(name)
        if not data.get("pageInfo", {}).get("hasNextPage"):
            break
        time.sleep(0.5)
    return result


def _gather_open_library(subjects: Optional[List[str]] = None,
                          max_pages: int = 10) -> Dict[str, List[str]]:
    """Fetch Open Library subject pages → audiobook_title, audiobook_author.

    Args:
        subjects: Subject slugs to fetch (default: common fiction subjects).
        max_pages: Pages per subject.
    """
    if subjects is None:
        subjects = ["fiction", "science_fiction", "fantasy", "mystery",
                    "biography", "history", "philosophy", "poetry"]
    result: Dict[str, List[str]] = {"audiobook_title": [], "audiobook_author": []}
    for subject in subjects:
        for page in range(1, max_pages + 1):
            try:
                resp = requests.get(
                    f"https://openlibrary.org/subjects/{subject}.json",
                    params={"limit": 100, "offset": (page - 1) * 100},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                LOG.warning("open_library subject=%s page=%d: %s", subject, page, exc)
                break
            works = data.get("works", [])
            if not works:
                break
            for work in works:
                title = str(work.get("title", "") or "").strip()
                if title:
                    result["audiobook_title"].append(title)
                for author in work.get("authors", []):
                    name = str(author.get("name", "") or "").strip()
                    if name:
                        result["audiobook_author"].append(name)
            time.sleep(0.1)
    return result


# ---------------------------------------------------------------------------
# Homeserver / Arr-stack source functions
#
# These thin wrappers delegate entirely to the existing loaders in
# generate_dataset_from_media.py and convert their row dicts to entity pools.
# Connection parameters are read from environment variables — the same vars
# used by build_dataset.py's media step — so no extra config is needed.
# ---------------------------------------------------------------------------

def _rows_to_pools(rows: List[dict], source_name: str) -> Dict[str, List[str]]:
    """Convert generate_dataset_from_media row dicts to ``{label: [values]}``.

    Reads the rich multi-column format (title, ocp_label, actor, director,
    producer, writer, composer, artist, album, author, narrator, studio,
    genre) and maps each populated column to the corresponding OCPEntityLabel.

    Args:
        rows: Row dicts as returned by ``load_radarr()``, ``load_lidarr()``, etc.
        source_name: Human-readable source tag (for logging only).

    Returns:
        ``{ocp_label: [values]}`` pools.
    """
    col_to_label: Dict[str, str] = {
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
    }
    result: Dict[str, List[str]] = {}

    for row in rows:
        # Primary title → ocp_label from the row
        ocp_label = str(row.get("ocp_label", "") or "").strip()
        title = str(row.get("title", "") or "").strip()
        if title and ocp_label:
            result.setdefault(ocp_label, []).append(title)

        # Secondary columns → derived entity labels
        for col, label in col_to_label.items():
            val = str(row.get(col, "") or "").strip()
            if not val:
                continue
            # Values may be pipe-separated (multiple people in one field)
            for part in val.split("|"):
                part = part.strip()
                if part:
                    result.setdefault(label, []).append(part)

        # Genre → video_genre / music_genre heuristic
        genre_val = str(row.get("genre", "") or "").strip()
        if genre_val:
            # Use music_genre for music-labelled rows, video_genre otherwise
            genre_label = "music_genre" if ocp_label in (
                "artist_name", "track_name", "album_name", "radio_station"
            ) else "video_genre"
            for g in genre_val.split("|"):
                g = g.strip()
                if g:
                    result.setdefault(genre_label, []).append(g)

    total = sum(len(v) for v in result.values())
    LOG.info("%s: converted %d rows → %d entity values across %d labels",
             source_name, len(rows), total, len(result))
    return result


def _env(key: str) -> str:
    """Return environment variable value, empty string if unset."""
    return os.environ.get(key, "")


def _gather_radarr() -> Dict[str, List[str]]:
    """Fetch movies + cast/crew from Radarr (RADARR_URL + RADARR_API_KEY env vars)."""
    url, key = _env("RADARR_URL"), _env("RADARR_API_KEY")
    if not url or not key:
        LOG.info("radarr: RADARR_URL / RADARR_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_radarr
    return _rows_to_pools(load_radarr(url, key), "radarr")


def _gather_sonarr() -> Dict[str, List[str]]:
    """Fetch TV shows + anime from Sonarr (SONARR_URL + SONARR_API_KEY env vars)."""
    url, key = _env("SONARR_URL"), _env("SONARR_API_KEY")
    if not url or not key:
        LOG.info("sonarr: SONARR_URL / SONARR_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_sonarr
    return _rows_to_pools(load_sonarr(url, key), "sonarr")


def _gather_lidarr() -> Dict[str, List[str]]:
    """Fetch artists, albums, tracks from Lidarr (LIDARR_URL + LIDARR_API_KEY env vars)."""
    url, key = _env("LIDARR_URL"), _env("LIDARR_API_KEY")
    if not url or not key:
        LOG.info("lidarr: LIDARR_URL / LIDARR_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_lidarr
    return _rows_to_pools(load_lidarr(url, key), "lidarr")


def _gather_readarr() -> Dict[str, List[str]]:
    """Fetch audiobooks/books from Readarr (READARR_URL + READARR_API_KEY env vars)."""
    url, key = _env("READARR_URL"), _env("READARR_API_KEY")
    if not url or not key:
        LOG.info("readarr: READARR_URL / READARR_API_KEY not set — skipping")
        return {}
    try:
        from ovos_media_classifier.train.generate_dataset_from_media import load_readarr
        return _rows_to_pools(load_readarr(url, key), "readarr")
    except ImportError:
        LOG.warning("readarr: load_readarr not available in this version")
        return {}


def _gather_jellyfin() -> Dict[str, List[str]]:
    """Fetch all media types from Jellyfin (JELLYFIN_URL + JELLYFIN_API_KEY env vars)."""
    url, key = _env("JELLYFIN_URL"), _env("JELLYFIN_API_KEY")
    if not url or not key:
        LOG.info("jellyfin: JELLYFIN_URL / JELLYFIN_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_jellyfin, JELLYFIN_ALL_TYPES
    user_id = _env("JELLYFIN_USER_ID") or None
    return _rows_to_pools(load_jellyfin(url, key, user_id=user_id,
                                         item_types=JELLYFIN_ALL_TYPES), "jellyfin")


def _gather_music_assistant() -> Dict[str, List[str]]:
    """Fetch artists/albums/tracks/radio from Music Assistant (MUSIC_ASSISTANT_URL env var)."""
    url = _env("MUSIC_ASSISTANT_URL")
    if not url:
        LOG.info("music_assistant: MUSIC_ASSISTANT_URL not set — skipping")
        return {}
    token = _env("MUSIC_ASSISTANT_TOKEN") or None
    from ovos_media_classifier.train.generate_dataset_from_media import load_music_assistant
    return _rows_to_pools(load_music_assistant(url, token=token), "music_assistant")


def _gather_audiobookshelf() -> Dict[str, List[str]]:
    """Fetch audiobooks/podcasts from Audiobookshelf (AUDIOBOOKSHELF_URL + AUDIOBOOKSHELF_API_KEY)."""
    url, key = _env("AUDIOBOOKSHELF_URL"), _env("AUDIOBOOKSHELF_API_KEY")
    if not url or not key:
        LOG.info("audiobookshelf: AUDIOBOOKSHELF_URL / AUDIOBOOKSHELF_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_audiobookshelf
    return _rows_to_pools(load_audiobookshelf(url, key), "audiobookshelf")


def _gather_podgrab() -> Dict[str, List[str]]:
    """Fetch podcasts from Podgrab (PODGRAB_URL env var; optional PODGRAB_USERNAME/PASSWORD)."""
    url = _env("PODGRAB_URL")
    if not url:
        LOG.info("podgrab: PODGRAB_URL not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_podgrab
    return _rows_to_pools(load_podgrab(url, _env("PODGRAB_USERNAME") or None,
                                        _env("PODGRAB_PASSWORD") or None), "podgrab")


def _gather_kapowarr() -> Dict[str, List[str]]:
    """Fetch comic volumes from Kapowarr (KAPOWARR_URL + KAPOWARR_API_KEY env vars)."""
    url, key = _env("KAPOWARR_URL"), _env("KAPOWARR_API_KEY")
    if not url or not key:
        LOG.info("kapowarr: KAPOWARR_URL / KAPOWARR_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_kapowarr
    return _rows_to_pools(load_kapowarr(url, key), "kapowarr")


def _gather_mylar3() -> Dict[str, List[str]]:
    """Fetch comic series from Mylar3 (MYLAR3_URL + MYLAR3_API_KEY env vars)."""
    url, key = _env("MYLAR3_URL"), _env("MYLAR3_API_KEY")
    if not url or not key:
        LOG.info("mylar3: MYLAR3_URL / MYLAR3_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_mylar3
    return _rows_to_pools(load_mylar3(url, key), "mylar3")


def _gather_whisparr() -> Dict[str, List[str]]:
    """Fetch adult titles/performers from Whisparr (WHISPARR_URL + WHISPARR_API_KEY env vars)."""
    url, key = _env("WHISPARR_URL"), _env("WHISPARR_API_KEY")
    if not url or not key:
        LOG.info("whisparr: WHISPARR_URL / WHISPARR_API_KEY not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_whisparr
    return _rows_to_pools(load_whisparr(url, key), "whisparr")


def _gather_stash() -> Dict[str, List[str]]:
    """Fetch adult scenes/performers from Stash (STASH_URL env var; optional STASH_API_KEY)."""
    url = _env("STASH_URL")
    if not url:
        LOG.info("stash: STASH_URL not set — skipping")
        return {}
    from ovos_media_classifier.train.generate_dataset_from_media import load_stash
    return _rows_to_pools(load_stash(url, api_key=_env("STASH_API_KEY") or None), "stash")


def _gather_existing_media_csv() -> Dict[str, List[str]]:
    """Load entities from the existing ocp_media.csv if present (no network needed).

    Reads ``~/.cache/ovos-media-classifier/output/ocp_media.csv`` (produced by
    the ``media`` pipeline step) and converts it to entity pools via
    ``_rows_to_pools()``.  Use this when you have already run the media step and
    want to rebuild entity pools without re-querying your homeserver.
    """
    from ovos_media_classifier.train import get_output_dir
    path = os.path.join(get_output_dir(), "ocp_media.csv")
    if not os.path.exists(path):
        LOG.info("existing_media_csv: %s not found — skipping", path)
        return {}
    try:
        df = pd.read_csv(path, low_memory=False)
        rows = df.to_dict("records")
        return _rows_to_pools(rows, "existing_media_csv")
    except Exception as exc:
        LOG.warning("existing_media_csv: failed to read %s — %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_HOMESERVER_SOURCES: Dict[str, object] = {
    "radarr":              _gather_radarr,
    "sonarr":              _gather_sonarr,
    "lidarr":              _gather_lidarr,
    "readarr":             _gather_readarr,
    "jellyfin":            _gather_jellyfin,
    "music_assistant":     _gather_music_assistant,
    "audiobookshelf":      _gather_audiobookshelf,
    "podgrab":             _gather_podgrab,
    "kapowarr":            _gather_kapowarr,
    "mylar3":              _gather_mylar3,
    "whisparr":            _gather_whisparr,
    "stash":               _gather_stash,
    "existing_media_csv":  _gather_existing_media_csv,
}

# Maps source name → callable that returns {label: [values]}
_HF_SOURCES: Dict[str, object] = {
    "wikidata":          _gather_wikidata,
    "metal_bands":       _gather_metal_bands,
    "metal_tracks":      _gather_metal_tracks,
    "jazz":              _gather_jazz,
    "prog":              _gather_prog,
    "classic_composers": _gather_classic_composers,
    "trance":            _gather_trance,
    "movie_actors":      _gather_movie_actors,
    "movie_directors":   _gather_movie_directors,
    "movie_writers":     _gather_movie_writers,
    "movie_producers":   _gather_movie_producers,
    "movie_composers":   _gather_movie_composers,
}

_REST_SOURCES: Dict[str, object] = {
    "gutendex":        _gather_gutendex,
    "librivox":        _gather_librivox,
    "radio_garden":    _gather_radio_garden,
    "anime_offline_db": _gather_anime_offline_db,
    "steam":           _gather_steam,
    "anilist":         _gather_anilist,
    "open_library":    _gather_open_library,
}

ALL_SOURCES: Dict[str, object] = {**_HF_SOURCES, **_REST_SOURCES, **_HOMESERVER_SOURCES}


# ---------------------------------------------------------------------------
# Main gather logic
# ---------------------------------------------------------------------------

def gather_all(
    sources: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    hf_cache: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run all (or selected) source gatherers, merge, dedup, and save CSVs.

    Args:
        sources: Source names to run (default: all).
        output_dir: Directory for output CSVs (default: ``~/.cache/ovos-media-classifier/entities/``).
        hf_cache: HuggingFace cache dir (default: ``get_hf_cache_dir()``).

    Returns:
        ``{label: DataFrame}`` — one DataFrame per label with columns ``value, source``.
    """
    if sources is None:
        sources = list(ALL_SOURCES.keys())
    hf_cache = hf_cache or get_hf_cache_dir()
    out_dir = get_entities_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    merged: Dict[str, Dict[str, List[str]]] = {}  # {label: {source: [values]}}

    for name in sources:
        fn = ALL_SOURCES.get(name)
        if fn is None:
            LOG.warning("Unknown source: %s", name)
            continue
        print(f"  Gathering {name} …", flush=True)
        try:
            if name in _HF_SOURCES:
                result = fn(hf_cache)  # type: ignore[call-arg]
            else:
                result = fn()  # type: ignore[call-arg]
        except Exception as exc:
            LOG.warning("%s: unexpected error — %s", name, exc)
            continue
        for label, values in result.items():
            merged.setdefault(label, {}).setdefault(name, []).extend(values)

    # Build per-label DataFrames
    label_dfs: Dict[str, pd.DataFrame] = {}
    for label, source_map in merged.items():
        rows = []
        seen: set = set()
        for source, values in source_map.items():
            for v in values:
                v_norm = v.strip().lower()
                if v_norm and v_norm not in seen:
                    seen.add(v_norm)
                    rows.append({"value": v.strip(), "source": source})
        df = pd.DataFrame(rows, columns=["value", "source"])
        label_path = os.path.join(out_dir, f"{label}.csv")
        df.to_csv(label_path, index=False)
        label_dfs[label] = df
        print(f"    {label}: {len(df):,} entities → {label_path}")

    # Combined CSV
    combined_rows = []
    for label, df in label_dfs.items():
        tmp = df.copy()
        tmp["label"] = label
        combined_rows.append(tmp)
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)[["value", "source", "label"]]
        combined_path = os.path.join(out_dir, "entities_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  Combined: {len(combined):,} entities → {combined_path}")

    return label_dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for gather_entities pipeline step."""
    parser = argparse.ArgumentParser(
        description="Gather entity pools from all available sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sources", nargs="*", metavar="SOURCE",
                        help=f"Sources to run (default: all). Available: {sorted(ALL_SOURCES)}")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: ~/.cache/ovos-media-classifier/entities/)")
    parser.add_argument("--list-sources", action="store_true",
                        help="Print available source names and exit")
    args = parser.parse_args()

    if args.list_sources:
        for name in sorted(ALL_SOURCES):
            if name in _HF_SOURCES:
                kind = "HF    "
            elif name in _HOMESERVER_SOURCES:
                kind = "HOMESERVER (env var)"
            else:
                kind = "REST  "
            print(f"  {name:<25} [{kind}]")
        return

    logging.basicConfig(level=logging.WARNING)
    print("Gathering entities …")
    gather_all(sources=args.sources, output_dir=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
