#!/usr/bin/env python3
"""
generate_dataset_from_media.py

Generates a wide-format OCP media dataset CSV where every row is a single
piece of primary media (movie, song, TV show, album, audiobook, …) and every
column is a metadata attribute (actor, director, artist, album, genre, …).

Multi-value fields (e.g. multiple actors) are stored as pipe-separated strings.

Output CSV columns:
  title        -- primary title of the media item
  ocp_label    -- OCPEntityLabel string (e.g. "movie_title", "track_name")
  media_type   -- OCP MediaType name (e.g. "MOVIE", "MUSIC", "TV_SHOW")
  genre        -- pipe-separated genres
  actor        -- pipe-separated actor/performer names
  director     -- pipe-separated director names
  producer     -- pipe-separated producer names
  writer       -- pipe-separated writer names
  composer     -- pipe-separated composer names
  artist       -- pipe-separated artist names (music)
  album        -- album title (tracks) or blank
  author       -- author name (audiobooks) or blank
  narrator     -- narrator name (audiobooks) or blank
  studio       -- studio / network / label name
  source       -- which service provided this row

Sources:
  - Radarr          (movies + cast/crew metadata)
  - Sonarr          (TV shows, anime, documentaries)
  - Lidarr          (artists, albums, tracks, genres)
  - Readarr         (books / audiobooks + authors)
  - Whisparr        (adult movies + performers)
  - Stash           (adult scenes, performers, studios — GraphQL)
  - Jellyfin        (movies, TV, music; audiobooks/podcasts opt-in)
  - Music Assistant (artists, albums, tracks, radio stations)

Jellyfin item types are configurable via --jellyfin-types.  The default set
excludes Book, AudioBook, and Podcast because Jellyfin's metadata for those
types is often incomplete or unreliable.  Use Readarr for books/audiobooks
instead, and pass --jellyfin-types to re-include them if needed.

Usage:
  python generate_dataset_from_media.py --output media_dataset.csv \\
      --jellyfin-url   http://localhost:8096 --jellyfin-api-key KEY \\
      --radarr-url     http://localhost:7878 --radarr-api-key   KEY \\
      --sonarr-url     http://localhost:8989 --sonarr-api-key   KEY \\
      --lidarr-url     http://localhost:8686 --lidarr-api-key   KEY \\
      --readarr-url    http://localhost:8787 --readarr-api-key  KEY \\
      --whisparr-url   http://localhost:6969 --whisparr-api-key KEY \\
      --stash-url      http://localhost:9999 --stash-api-key    KEY \\
      --music-assistant-url http://localhost:8095

  # Include Jellyfin audiobooks/podcasts (off by default):
  python generate_dataset_from_media.py \\
      --jellyfin-url http://localhost:8096 --jellyfin-api-key KEY \\
      --jellyfin-types Movie,Series,MusicAlbum,MusicArtist,Audio,Book,AudioBook,Podcast

Any source can be omitted — the script skips it gracefully.
Sources can also be configured via environment variables:
  JELLYFIN_URL,          JELLYFIN_API_KEY,   JELLYFIN_USER_ID,   JELLYFIN_TYPES
  RADARR_URL,            RADARR_API_KEY
  SONARR_URL,            SONARR_API_KEY
  LIDARR_URL,            LIDARR_API_KEY
  READARR_URL,           READARR_API_KEY
  WHISPARR_URL,          WHISPARR_API_KEY
  STASH_URL,             STASH_API_KEY
  MUSIC_ASSISTANT_URL
  AUDIOBOOKSHELF_URL,    AUDIOBOOKSHELF_API_KEY
  LISTENARR_URL,         LISTENARR_API_KEY
  KAPOWARR_URL,          KAPOWARR_API_KEY
  MYLAR3_URL,            MYLAR3_API_KEY
  PODGRAB_URL,           PODGRAB_USERNAME,   PODGRAB_PASSWORD
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional

LOG = logging.getLogger("ocp-dataset-gen")

# ---------------------------------------------------------------------------
# CSV column order
# ---------------------------------------------------------------------------

COLUMNS = [
    "title",
    "ocp_label",
    "media_type",
    "genre",
    "actor",
    "director",
    "producer",
    "writer",
    "composer",
    "artist",
    "album",
    "author",
    "narrator",   # audiobooks
    "studio",
    "source",
]

# ---------------------------------------------------------------------------
# OCP label → MediaType name mapping (subset used by the loaders below)
# ---------------------------------------------------------------------------

_LABEL_TO_MEDIA_TYPE: Dict[str, str] = {
    "movie_title":       "MOVIE",
    "anime_title":       "ANIME",
    "cartoon_title":     "CARTOON",
    "documentary_title": "DOCUMENTARY",
    "tv_show_title":     "TV_SHOW",
    "track_name":        "MUSIC",
    "album_name":        "MUSIC",
    "artist_name":       "MUSIC",
    "radio_station":     "RADIO",
    "audiobook_title":   "AUDIOBOOK",
    "audiobook_author":  "AUDIOBOOK",
    "podcast_title":     "PODCAST",
    "comic_title":       "VISUAL_STORY",
    "adult":             "ADULT",
}


def _media_type(ocp_label: str) -> str:
    """Return the OCP MediaType name for an entity label, falling back to GENERIC."""
    try:
        from ovos_media_classifier.intents import NER_LABEL_TO_MEDIA_TYPE
        mt = NER_LABEL_TO_MEDIA_TYPE.get(ocp_label)
        if mt is not None:
            return mt.name
    except Exception:
        pass
    return _LABEL_TO_MEDIA_TYPE.get(ocp_label, "GENERIC")


# ---------------------------------------------------------------------------
# Genre helpers
# ---------------------------------------------------------------------------

_ANIME_GENRES   = {"anime"}
_CARTOON_GENRES = {"animation", "animated", "cartoon"}
_DOC_GENRES     = {"documentary"}


def _video_label_from_genres(genres: List[str], default: str) -> str:
    gl = {g.lower() for g in genres}
    if gl & _ANIME_GENRES:
        return "anime_title"
    if gl & _DOC_GENRES:
        return "documentary_title"
    if gl & _CARTOON_GENRES:
        return "cartoon_title"
    return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join(values: List[str]) -> str:
    """Join a list of non-empty strings with '|'."""
    return "|".join(v for v in values if v)


def _http_get(session, url: str, **kwargs):
    try:
        resp = session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        LOG.warning("HTTP GET %s failed: %s", url, exc)
        return None


def _make_row(**kwargs) -> Dict[str, str]:
    """Return a blank row dict, populated from kwargs."""
    row = {col: "" for col in COLUMNS}
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# Per-source loaders  (each returns List[dict])
# ---------------------------------------------------------------------------

def load_radarr(url: str, api_key: str) -> List[dict]:
    import requests
    session = requests.Session()
    session.headers["X-Api-Key"] = api_key
    base = url.rstrip("/")

    LOG.info("[radarr] Fetching movie library from %s …", base)
    movies = _http_get(session, f"{base}/api/v3/movie") or []
    LOG.info("[radarr] Found %d movies", len(movies))

    rows = []
    for movie in movies:
        title = movie.get("title", "")
        if not title:
            continue
        genres = movie.get("genres") or []
        ocp_label = _video_label_from_genres(genres, "movie_title")
        credits = movie.get("credits") or {}

        actors, directors, producers, writers, composers = [], [], [], [], []
        for person in credits.get("castMembers") or []:
            name = person.get("name", "")
            if name:
                actors.append(name)
        for person in credits.get("crewMembers") or []:
            name = person.get("name", "")
            job  = (person.get("job") or person.get("department") or "").lower()
            if not name:
                continue
            if "direct" in job:
                directors.append(name)
            elif "produc" in job:
                producers.append(name)
            elif "writ" in job or "screenplay" in job:
                writers.append(name)
            elif "compos" in job or "original score" in job:
                composers.append(name)

        rows.append(_make_row(
            title=title,
            ocp_label=ocp_label,
            media_type=_media_type(ocp_label),
            genre=_join(genres),
            actor=_join(actors),
            director=_join(directors),
            producer=_join(producers),
            writer=_join(writers),
            composer=_join(composers),
            studio=movie.get("studio", ""),
            source="radarr",
        ))

        # Alternate titles as extra rows (title only, same metadata)
        for alt in movie.get("alternateTitles") or []:
            alt_title = alt.get("title", "")
            if alt_title and alt_title != title:
                rows.append(_make_row(
                    title=alt_title,
                    ocp_label=ocp_label,
                    media_type=_media_type(ocp_label),
                    genre=_join(genres),
                    actor=_join(actors),
                    director=_join(directors),
                    producer=_join(producers),
                    writer=_join(writers),
                    composer=_join(composers),
                    studio=movie.get("studio", ""),
                    source="radarr",
                ))

    LOG.info("[radarr] Produced %d rows", len(rows))
    return rows


def load_sonarr(url: str, api_key: str) -> List[dict]:
    import requests
    session = requests.Session()
    session.headers["X-Api-Key"] = api_key
    base = url.rstrip("/")

    LOG.info("[sonarr] Fetching series library from %s …", base)
    series = _http_get(session, f"{base}/api/v3/series") or []
    LOG.info("[sonarr] Found %d series", len(series))

    rows = []
    for show in series:
        title = show.get("title", "")
        if not title:
            continue
        genres = show.get("genres") or []
        series_type = (show.get("seriesType") or "").lower()
        if series_type == "anime" or "anime" in {g.lower() for g in genres}:
            ocp_label = "anime_title"
        else:
            ocp_label = _video_label_from_genres(genres, "tv_show_title")

        rows.append(_make_row(
            title=title,
            ocp_label=ocp_label,
            media_type=_media_type(ocp_label),
            genre=_join(genres),
            studio=show.get("network", ""),
            source="sonarr",
        ))

        for alt in show.get("alternateTitles") or []:
            alt_title = alt.get("title", "")
            if alt_title and alt_title != title:
                rows.append(_make_row(
                    title=alt_title,
                    ocp_label=ocp_label,
                    media_type=_media_type(ocp_label),
                    genre=_join(genres),
                    studio=show.get("network", ""),
                    source="sonarr",
                ))

    LOG.info("[sonarr] Produced %d rows", len(rows))
    return rows


def load_lidarr(url: str, api_key: str) -> List[dict]:
    import requests
    session = requests.Session()
    session.headers["X-Api-Key"] = api_key
    base = url.rstrip("/")

    LOG.info("[lidarr] Fetching artists from %s …", base)
    artists_data = _http_get(session, f"{base}/api/v1/artist") or []
    LOG.info("[lidarr] Found %d artists", len(artists_data))

    # Build artist-id → name+genres lookup
    artist_map: Dict[str, dict] = {}
    rows = []
    for artist in artists_data:
        name = artist.get("artistName", "")
        if not name:
            continue
        aid = str(artist.get("id", ""))
        genres = artist.get("genres") or []
        artist_map[aid] = {"name": name, "genres": genres}
        rows.append(_make_row(
            title=name,
            ocp_label="artist_name",
            media_type=_media_type("artist_name"),
            genre=_join(genres),
            artist=name,
            source="lidarr",
        ))

    LOG.info("[lidarr] Fetching albums …")
    albums = _http_get(session, f"{base}/api/v1/album") or []
    LOG.info("[lidarr] Found %d albums", len(albums))

    for album in albums:
        album_title = album.get("title", "")
        if not album_title:
            continue
        artist_name = (album.get("artist") or {}).get("artistName", "")
        genres = (album.get("artist") or {}).get("genres") or album.get("genres") or []
        tracks = []
        for medium in album.get("media") or []:
            for track in medium.get("tracks") or []:
                t = track.get("title", "")
                if t:
                    tracks.append(t)

        rows.append(_make_row(
            title=album_title,
            ocp_label="album_name",
            media_type=_media_type("album_name"),
            genre=_join(genres),
            artist=artist_name,
            source="lidarr",
        ))

        # Each track as its own row
        for track_title in tracks:
            rows.append(_make_row(
                title=track_title,
                ocp_label="track_name",
                media_type=_media_type("track_name"),
                genre=_join(genres),
                artist=artist_name,
                album=album_title,
                source="lidarr",
            ))

    LOG.info("[lidarr] Produced %d rows", len(rows))
    return rows


def load_whisparr(url: str, api_key: str) -> List[dict]:
    import requests
    session = requests.Session()
    session.headers["X-Api-Key"] = api_key
    base = url.rstrip("/")

    LOG.info("[whisparr] Fetching movie library from %s …", base)
    movies = _http_get(session, f"{base}/api/v3/movie") or []
    LOG.info("[whisparr] Found %d titles", len(movies))

    rows = []
    for movie in movies:
        title = movie.get("title", "")
        if not title:
            continue
        rows.append(_make_row(
            title=title,
            ocp_label="adult",
            media_type=_media_type("adult"),
            actor=_join(movie.get("performerNames", [])),
            studio=movie.get("studioTitle", ""),
            source="whisparr",
        ))

        for alt in movie.get("alternateTitles") or []:
            alt_title = alt.get("title", "")
            if alt_title and alt_title != title:
                rows.append(_make_row(
                    title=alt_title,
                    ocp_label="adult",
                    media_type=_media_type("adult"),
                    actor=_join(movie.get("performerNames", [])),
                    studio=movie.get("studioTitle", ""),
                    source="whisparr",
                ))

    LOG.info("[whisparr] Produced %d rows", len(rows))
    return rows


def load_stash(url: str, api_key: Optional[str] = None, page_size: int = 500) -> List[dict]:
    import requests
    session = requests.Session()
    if api_key:
        session.headers["ApiKey"] = api_key
    gql_url = url.rstrip("/") + "/graphql"

    def _gql(query: str, variables: dict) -> dict:
        try:
            resp = session.post(gql_url, json={"query": query, "variables": variables}, timeout=30)
            resp.raise_for_status()
            return resp.json().get("data") or {}
        except Exception as exc:
            LOG.warning("[stash] GraphQL request failed: %s", exc)
            return {}

    LOG.info("[stash] Fetching scenes from %s …", url)

    # Build performer → name map
    performers_map: Dict[str, str] = {}
    page = 1
    while True:
        data = _gql("""
            query FindPerformers($filter: FindFilterType) {
              findPerformers(filter: $filter) {
                count
                performers { id name aliases }
              }
            }
        """, {"filter": {"page": page, "per_page": page_size}})
        pdata = (data.get("findPerformers") or {})
        items = pdata.get("performers") or []
        if not items:
            break
        for p in items:
            performers_map[p.get("id", "")] = p.get("name", "")
        if page * page_size >= pdata.get("count", 0):
            break
        page += 1

    # Build studio → name map
    studios_map: Dict[str, str] = {}
    page = 1
    while True:
        data = _gql("""
            query FindStudios($filter: FindFilterType) {
              findStudios(filter: $filter) {
                count
                studios { id name }
              }
            }
        """, {"filter": {"page": page, "per_page": page_size}})
        sdata = (data.get("findStudios") or {})
        items = sdata.get("studios") or []
        if not items:
            break
        for s in items:
            studios_map[s.get("id", "")] = s.get("name", "")
        if page * page_size >= sdata.get("count", 0):
            break
        page += 1

    # Scenes
    rows = []
    page = 1
    while True:
        data = _gql("""
            query FindScenes($filter: FindFilterType) {
              findScenes(filter: $filter) {
                count
                scenes {
                  title
                  tags { name }
                  performers { id name }
                  studio { id name }
                }
              }
            }
        """, {"filter": {"page": page, "per_page": page_size}})
        scenes_data = (data.get("findScenes") or {})
        items = scenes_data.get("scenes") or []
        if not items:
            break
        for s in items:
            title = s.get("title", "")
            if not title:
                continue
            actors = [p.get("name", "") for p in (s.get("performers") or []) if p.get("name")]
            studio = (s.get("studio") or {}).get("name", "")
            genre = _join([t.get("name", "") for t in (s.get("tags") or []) if t.get("name")])
            rows.append(_make_row(
                title=title,
                ocp_label="adult",
                media_type=_media_type("adult"),
                genre=genre,
                actor=_join(actors),
                studio=studio,
                source="stash",
            ))
        if page * page_size >= scenes_data.get("count", 0):
            break
        page += 1

    LOG.info("[stash] Produced %d rows", len(rows))
    return rows


# All Jellyfin item types the loader understands, in fetch order.
# Book/AudioBook/Podcast are excluded from the default because Jellyfin's
# metadata for those types is often sparse or incorrect.
JELLYFIN_ALL_TYPES = ["Movie", "Series", "MusicAlbum", "MusicArtist", "Audio",
                      "Book", "AudioBook", "Podcast"]
JELLYFIN_DEFAULT_TYPES = ["Movie", "Series", "MusicAlbum", "MusicArtist", "Audio"]

_JELLYFIN_TYPE_LABEL = {
    "Movie":       "movie_title",
    "Series":      "tv_show_title",
    "MusicAlbum":  "album_name",
    "MusicArtist": "artist_name",
    "Audio":       "track_name",
    "Book":        "audiobook_title",
    "AudioBook":   "audiobook_title",
    "Podcast":     "podcast_title",
}




def load_jellyfin(
    url: str,
    api_key: str,
    user_id: Optional[str] = None,
    item_types: Optional[List[str]] = None,
) -> List[dict]:
    """Load media items from a Jellyfin instance.

    Args:
        item_types: Which Jellyfin item types to fetch.  Defaults to
                    ``JELLYFIN_DEFAULT_TYPES`` (Movie, Series, MusicAlbum,
                    MusicArtist, Audio).  Pass a custom list to include or
                    exclude types — e.g. add ``"Book"`` / ``"AudioBook"`` /
                    ``"Podcast"`` if your Jellyfin library has good metadata
                    for those (many do not).
    """
    import requests
    session = requests.Session()
    base = url.rstrip("/")

    # Resolve user ID
    if not user_id:
        users = _http_get(session, f"{base}/Users", params={"api_key": api_key}) or []
        user_id = users[0].get("Id", "") if users else ""
    items_url = f"{base}/Users/{user_id}/Items" if user_id else f"{base}/Items"

    types_to_fetch = item_types if item_types is not None else JELLYFIN_DEFAULT_TYPES
    unknown = [t for t in types_to_fetch if t not in _JELLYFIN_TYPE_LABEL]
    if unknown:
        LOG.warning("[jellyfin] Unknown item types (will be skipped): %s", unknown)

    rows = []
    for item_type in types_to_fetch:
        default_label = _JELLYFIN_TYPE_LABEL.get(item_type)
        if default_label is None:
            continue
        LOG.info("[jellyfin] Fetching %s items …", item_type)
        start = 0
        limit = 500
        while True:
            page = _http_get(session, items_url, params={
                "api_key":          api_key,
                "IncludeItemTypes": item_type,
                "Recursive":        "true",
                "Fields":           "Genres,Studios,People,Artists,Album,Overview",
                "StartIndex":       start,
                "Limit":            limit,
            }) or {}
            items = page.get("Items") or []
            if not items:
                break

            for item in items:
                name = item.get("Name", "")
                if not name:
                    continue
                genres = item.get("Genres") or []
                if default_label == "movie_title":
                    ocp_label = _video_label_from_genres(genres, "movie_title")
                elif default_label == "tv_show_title":
                    ocp_label = _video_label_from_genres(genres, "tv_show_title")
                else:
                    ocp_label = default_label

                actors, directors, producers, writers, composers, authors = [], [], [], [], [], []
                for person in item.get("People") or []:
                    pname = person.get("Name", "")
                    prole = (person.get("Type") or "").lower()
                    if not pname:
                        continue
                    if prole == "actor":
                        actors.append(pname)
                    elif prole == "director":
                        directors.append(pname)
                    elif prole == "producer":
                        producers.append(pname)
                    elif prole == "writer":
                        writers.append(pname)
                    elif prole in ("composer", "music"):
                        composers.append(pname)
                    elif prole in ("author",):
                        authors.append(pname)

                studios = [s.get("Name", "") for s in (item.get("Studios") or []) if s.get("Name")]

                # Music-specific fields
                track_artists = item.get("Artists") or []
                album_name = item.get("Album", "") if item_type == "Audio" else ""

                rows.append(_make_row(
                    title=name,
                    ocp_label=ocp_label,
                    media_type=_media_type(ocp_label),
                    genre=_join(genres),
                    actor=_join(actors),
                    director=_join(directors),
                    producer=_join(producers),
                    writer=_join(writers),
                    composer=_join(composers),
                    artist=_join(track_artists),
                    album=album_name,
                    author=_join(authors),
                    studio=_join(studios),
                    source="jellyfin",
                ))

            total = page.get("TotalRecordCount", 0)
            start += limit
            if start >= total:
                break

        LOG.info("[jellyfin] %s: done", item_type)

    LOG.info("[jellyfin] Produced %d rows", len(rows))
    return rows


def load_music_assistant(url: str, token: str = None) -> List[dict]:
    import uuid
    import requests
    session = requests.Session()
    base = url.rstrip("/")
    api_url = f"{base}/api"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _fetch_all(media_type: str) -> list:
        """Fetch all library items for a given media_type via command-based POST API."""
        results = []
        offset = 0
        limit = 50
        while True:
            # For Music Assistant 2.0+ REST API, searching with an empty query 
            # and library_only=True is a reliable way to list all library items.
            payload = {
                "command": "music/search",
                "message_id": uuid.uuid4().hex,
                "args": {
                    "search_query": "",
                    "media_types": [media_type],
                    "library_only": True,
                    "limit": limit,
                    "offset": offset
                },
            }
            try:
                resp = session.post(api_url, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # Handle JSON-RPC result wrapper if present
                if isinstance(data, dict) and "result" in data:
                    data = data["result"]
            except Exception as exc:
                LOG.warning("[music_assistant] POST %s media_type=%s failed: %s", api_url, media_type, exc)
                break

            if isinstance(data, dict):
                # Search results are grouped by type
                key = f"{media_type}s" if media_type != "radio" else "radio"
                items = data.get(key) or []
            else:
                items = data if isinstance(data, list) else []
            
            results.extend(items)
            offset += limit
            # Pagination ends when we get fewer items than requested
            if not items or len(items) < limit:
                break
        return results

    LOG.info("[music_assistant] Fetching library from %s …", base)
    rows = []

    for artist in _fetch_all("artist"):
        name = artist.get("name", "")
        if not name:
            continue
        genres = artist.get("metadata", {}).get("genres") or []
        rows.append(_make_row(
            title=name,
            ocp_label="artist_name",
            media_type=_media_type("artist_name"),
            genre=_join(genres),
            artist=name,
            source="music_assistant",
        ))

    for album in _fetch_all("album"):
        title = album.get("name", "")
        if not title:
            continue
        artist_names = [a.get("name", "") for a in (album.get("artists") or []) if a.get("name")]
        rows.append(_make_row(
            title=title,
            ocp_label="album_name",
            media_type=_media_type("album_name"),
            artist=_join(artist_names),
            source="music_assistant",
        ))

    for track in _fetch_all("track"):
        title = track.get("name", "")
        if not title:
            continue
        artist_names = [a.get("name", "") for a in (track.get("artists") or []) if a.get("name")]
        album_name = (track.get("album") or {}).get("name", "")
        rows.append(_make_row(
            title=title,
            ocp_label="track_name",
            media_type=_media_type("track_name"),
            artist=_join(artist_names),
            album=album_name,
            source="music_assistant",
        ))

    for station in _fetch_all("radio"):
        name = station.get("name", "")
        if not name:
            continue
        rows.append(_make_row(
            title=name,
            ocp_label="radio_station",
            media_type=_media_type("radio_station"),
            source="music_assistant",
        ))

    LOG.info("[music_assistant] Produced %d rows", len(rows))
    return rows


def load_audiobookshelf(url: str, api_key: str) -> List[dict]:
    """Load audiobooks and podcasts from an Audiobookshelf instance.

    Auth: ``Authorization: Bearer <api_key>``

    Fetches all libraries, then paginates each library's items:
    - ``mediaType = "book"``    → ``audiobook_title`` rows
    - ``mediaType = "podcast"`` → ``podcast_title`` rows

    Book metadata fields used: ``media.metadata.title``,
    ``media.metadata.authorName``, ``media.metadata.narratorName``,
    ``media.metadata.genres``, ``media.metadata.seriesName``,
    ``media.metadata.publisherName``.
    """
    import requests
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"
    base = url.rstrip("/")

    LOG.info("[audiobookshelf] Fetching libraries from %s …", base)
    libs_data = _http_get(session, f"{base}/api/libraries") or {}
    libraries = libs_data.get("libraries") or []
    LOG.info("[audiobookshelf] Found %d libraries", len(libraries))

    rows = []
    for lib in libraries:
        lib_id = lib.get("id", "")
        lib_name = lib.get("name", "")
        media_type = (lib.get("mediaType") or "").lower()
        if media_type not in ("book", "podcast"):
            LOG.debug("[audiobookshelf] Skipping library '%s' (mediaType=%s)", lib_name, media_type)
            continue

        LOG.info("[audiobookshelf] Fetching items from library '%s' (type=%s) …", lib_name, media_type)
        page = 0
        limit = 100
        while True:
            data = _http_get(session, f"{base}/api/libraries/{lib_id}/items",
                             params={"limit": limit, "page": page}) or {}
            items = data.get("results") or []
            if not items:
                break
            for item in items:
                meta = (item.get("media") or {}).get("metadata") or {}
                title = meta.get("title", "") or item.get("media", {}).get("metadata", {}).get("title", "")
                if not title:
                    continue

                if media_type == "book":
                    author    = meta.get("authorName", "")
                    narrator  = meta.get("narratorName", "")
                    genres    = meta.get("genres") or []
                    series    = meta.get("seriesName", "")
                    publisher = meta.get("publisherName", "")

                    rows.append(_make_row(
                        title=title,
                        ocp_label="audiobook_title",
                        media_type=_media_type("audiobook_title"),
                        genre=_join(genres),
                        author=author,
                        narrator=narrator,
                        studio=publisher,
                        source="audiobookshelf",
                    ))
                    if series and series != title:
                        rows.append(_make_row(
                            title=series,
                            ocp_label="audiobook_title",
                            media_type=_media_type("audiobook_title"),
                            author=author,
                            narrator=narrator,
                            source="audiobookshelf",
                        ))
                else:  # podcast
                    author = meta.get("author", "")
                    genres = meta.get("genres") or []
                    rows.append(_make_row(
                        title=title,
                        ocp_label="podcast_title",
                        media_type=_media_type("podcast_title"),
                        genre=_join(genres),
                        author=author,
                        source="audiobookshelf",
                    ))

            total = data.get("total", 0)
            page += 1
            if page * limit >= total:
                break

    LOG.info("[audiobookshelf] Produced %d rows", len(rows))
    return rows



def load_kapowarr(url: str, api_key: str) -> List[dict]:
    """Load comic volumes from a Kapowarr instance.

    Auth: ``X-Api-Key`` header.  Endpoint: ``GET /api/volumes``.

    Fields used: ``title``, ``alt_title``, ``year``, ``publisher``,
    ``volume_number``.  Comics map to ``comic_title`` / ``VISUAL_STORY``
    (closest OCP MediaType — no dedicated comic type exists yet).
    """
    import requests
    session = requests.Session()
    session.headers["X-Api-Key"] = api_key
    base = url.rstrip("/")

    LOG.info("[kapowarr] Fetching volumes from %s …", base)
    data = _http_get(session, f"{base}/api/volumes") or {}
    volumes = data.get("result") or data if isinstance(data, list) else []
    LOG.info("[kapowarr] Found %d volumes", len(volumes))

    rows = []
    for vol in volumes:
        title = vol.get("title", "")
        if not title:
            continue
        publisher  = vol.get("publisher", "")
        alt_title  = vol.get("alt_title", "")

        rows.append(_make_row(
            title=title,
            ocp_label="comic_title",
            media_type=_media_type("comic_title"),
            studio=publisher,
            source="kapowarr",
        ))
        if alt_title and alt_title != title:
            rows.append(_make_row(
                title=alt_title,
                ocp_label="comic_title",
                media_type=_media_type("comic_title"),
                studio=publisher,
                source="kapowarr",
            ))

    LOG.info("[kapowarr] Produced %d rows", len(rows))
    return rows


def load_mylar3(url: str, api_key: str) -> List[dict]:
    """Load comic series from a Mylar3 instance.

    Auth: ``?apikey=KEY`` query param.  Endpoint: ``GET /api?cmd=getComicList``.

    Response format: ``{"data": {"comics": [{id, name, publisher, status, ...}]}}``.
    Comics map to ``comic_title`` / ``VISUAL_STORY``.
    """
    import requests
    session = requests.Session()
    base = url.rstrip("/")

    LOG.info("[mylar3] Fetching comic list from %s …", base)
    data = _http_get(session, f"{base}/api",
                     params={"apikey": api_key, "cmd": "getComicList"}) or {}
    comics = (data.get("data") or {}).get("comics") or []
    if not comics and isinstance(data.get("data"), list):
        comics = data["data"]
    LOG.info("[mylar3] Found %d comics", len(comics))

    rows = []
    for comic in comics:
        title = comic.get("name") or comic.get("ComicName", "")
        if not title:
            continue
        publisher = comic.get("publisher") or comic.get("ComicPublisher", "")
        rows.append(_make_row(
            title=title,
            ocp_label="comic_title",
            media_type=_media_type("comic_title"),
            studio=publisher,
            source="mylar3",
        ))

    LOG.info("[mylar3] Produced %d rows", len(rows))
    return rows


def load_podgrab(url: str, username: Optional[str] = None, password: Optional[str] = None) -> List[dict]:
    """Load podcasts from a Podgrab instance.

    Auth: HTTP basic auth (username + password), optional.
    Endpoint: ``GET /podcasts``.

    Fields used: ``title``, ``categories`` (→ genre), ``author`` (from
    gpodder subscription data if present).
    """
    import requests
    session = requests.Session()
    if username and password:
        session.auth = (username, password)
    base = url.rstrip("/")

    LOG.info("[podgrab] Fetching podcasts from %s …", base)
    podcasts = _http_get(session, f"{base}/podcasts") or []
    if isinstance(podcasts, dict):
        podcasts = podcasts.get("data") or podcasts.get("podcasts") or []
    LOG.info("[podgrab] Found %d podcasts", len(podcasts))

    rows = []
    for podcast in podcasts:
        # Podgrab's Go structs have no json tags → serialised as CamelCase
        title  = podcast.get("Title") or podcast.get("title", "")
        if not title:
            continue
        author = podcast.get("Author") or podcast.get("author", "")
        # Tags is a list of tag objects with a "Name" field; no genre/category field
        tags   = podcast.get("Tags") or podcast.get("tags") or []
        genre  = _join([t.get("Name") or t.get("name", "") for t in tags if isinstance(t, dict)])
        rows.append(_make_row(
            title=title,
            ocp_label="podcast_title",
            media_type=_media_type("podcast_title"),
            genre=genre,
            author=author,
            source="podgrab",
        ))

    LOG.info("[podgrab] Produced %d rows", len(rows))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env(key: str) -> str:
    return os.environ.get(key, "")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate wide-format OCP media dataset CSV from media servers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", "-o", default="ocp_media_dataset.csv",
                   help="Output CSV file path")
    p.add_argument("--verbose", "-v", action="store_true")

    jf = p.add_argument_group("Jellyfin")
    jf.add_argument("--jellyfin-url", default=_env("JELLYFIN_URL"))
    jf.add_argument("--jellyfin-api-key", default=_env("JELLYFIN_API_KEY"))
    jf.add_argument("--jellyfin-user-id", default=_env("JELLYFIN_USER_ID"))
    jf.add_argument(
        "--jellyfin-types",
        default=_env("JELLYFIN_TYPES") or ",".join(JELLYFIN_DEFAULT_TYPES),
        help=(
            "Comma-separated Jellyfin item types to fetch. "
            f"All known types: {','.join(JELLYFIN_ALL_TYPES)}. "
            "Book/AudioBook/Podcast are excluded by default due to poor "
            "Jellyfin metadata quality for those types."
        ),
    )

    rr = p.add_argument_group("Radarr")
    rr.add_argument("--radarr-url", default=_env("RADARR_URL"))
    rr.add_argument("--radarr-api-key", default=_env("RADARR_API_KEY"))

    sr = p.add_argument_group("Sonarr")
    sr.add_argument("--sonarr-url", default=_env("SONARR_URL"))
    sr.add_argument("--sonarr-api-key", default=_env("SONARR_API_KEY"))

    lr = p.add_argument_group("Lidarr")
    lr.add_argument("--lidarr-url", default=_env("LIDARR_URL"))
    lr.add_argument("--lidarr-api-key", default=_env("LIDARR_API_KEY"))

    wr = p.add_argument_group("Whisparr")
    wr.add_argument("--whisparr-url", default=_env("WHISPARR_URL"))
    wr.add_argument("--whisparr-api-key", default=_env("WHISPARR_API_KEY"))

    st = p.add_argument_group("Stash")
    st.add_argument("--stash-url", default=_env("STASH_URL"))
    st.add_argument("--stash-api-key", default=_env("STASH_API_KEY"))

    ma = p.add_argument_group("Music Assistant")
    ma.add_argument("--music-assistant-url", default=_env("MUSIC_ASSISTANT_URL"))
    ma.add_argument("--music-assistant-token", default=_env("MUSIC_ASSISTANT_TOKEN"),
                    help="Bearer token (Settings → Advanced → Authentication Tokens); required for MA >= 2.7.2")

    abs_ = p.add_argument_group("Audiobookshelf")
    abs_.add_argument("--audiobookshelf-url", default=_env("AUDIOBOOKSHELF_URL"))
    abs_.add_argument("--audiobookshelf-api-key", default=_env("AUDIOBOOKSHELF_API_KEY"),
                      help="API token (Settings → Users → your user → API Token)")

    kp = p.add_argument_group("Kapowarr")
    kp.add_argument("--kapowarr-url", default=_env("KAPOWARR_URL"))
    kp.add_argument("--kapowarr-api-key", default=_env("KAPOWARR_API_KEY"))

    my = p.add_argument_group("Mylar3")
    my.add_argument("--mylar3-url", default=_env("MYLAR3_URL"))
    my.add_argument("--mylar3-api-key", default=_env("MYLAR3_API_KEY"))

    pg = p.add_argument_group("Podgrab")
    pg.add_argument("--podgrab-url", default=_env("PODGRAB_URL"))
    pg.add_argument("--podgrab-username", default=_env("PODGRAB_USERNAME"))
    pg.add_argument("--podgrab-password", default=_env("PODGRAB_PASSWORD"))

    return p


def _write_csv(rows: List[dict], path: str) -> None:
    LOG.info("Writing %d rows to %s", len(rows), path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    all_rows: List[dict] = []
    any_source = False

    # Whisparr
    if args.whisparr_url and args.whisparr_api_key:
        any_source = True
        try:
            all_rows.extend(load_whisparr(args.whisparr_url, args.whisparr_api_key))
        except Exception as exc:
            LOG.error("[whisparr] Failed: %s", exc)
    else:
        LOG.info("[whisparr] Skipped (no URL/key configured)")

    # Radarr
    if args.radarr_url and args.radarr_api_key:
        any_source = True
        try:
            all_rows.extend(load_radarr(args.radarr_url, args.radarr_api_key))
        except Exception as exc:
            LOG.error("[radarr] Failed: %s", exc)
    else:
        LOG.info("[radarr] Skipped (no URL/key configured)")

    # Sonarr
    if args.sonarr_url and args.sonarr_api_key:
        any_source = True
        try:
            all_rows.extend(load_sonarr(args.sonarr_url, args.sonarr_api_key))
        except Exception as exc:
            LOG.error("[sonarr] Failed: %s", exc)
    else:
        LOG.info("[sonarr] Skipped (no URL/key configured)")

    # Lidarr
    if args.lidarr_url and args.lidarr_api_key:
        any_source = True
        try:
            all_rows.extend(load_lidarr(args.lidarr_url, args.lidarr_api_key))
        except Exception as exc:
            LOG.error("[lidarr] Failed: %s", exc)
    else:
        LOG.info("[lidarr] Skipped (no URL/key configured)")


    # Jellyfin
    if args.jellyfin_url and args.jellyfin_api_key:
        any_source = True
        jellyfin_types = [t.strip() for t in args.jellyfin_types.split(",") if t.strip()]
        LOG.info("[jellyfin] Item types: %s", jellyfin_types)
        try:
            all_rows.extend(load_jellyfin(
                args.jellyfin_url,
                args.jellyfin_api_key,
                user_id=args.jellyfin_user_id or None,
                item_types=jellyfin_types,
            ))
        except Exception as exc:
            LOG.error("[jellyfin] Failed: %s", exc)
    else:
        LOG.info("[jellyfin] Skipped (no URL/key configured)")

    # Stash (api_key optional)
    if args.stash_url:
        any_source = True
        try:
            all_rows.extend(load_stash(args.stash_url, api_key=args.stash_api_key or None))
        except Exception as exc:
            LOG.error("[stash] Failed: %s", exc)
    else:
        LOG.info("[stash] Skipped (no URL configured)")

    # Music Assistant
    if args.music_assistant_url:
        any_source = True
        try:
            all_rows.extend(load_music_assistant(args.music_assistant_url, token=args.music_assistant_token or None))
        except Exception as exc:
            LOG.error("[music_assistant] Failed: %s", exc)
    else:
        LOG.info("[music_assistant] Skipped (no URL configured)")

    # Audiobookshelf
    if args.audiobookshelf_url and args.audiobookshelf_api_key:
        any_source = True
        try:
            all_rows.extend(load_audiobookshelf(args.audiobookshelf_url, args.audiobookshelf_api_key))
        except Exception as exc:
            LOG.error("[audiobookshelf] Failed: %s", exc)
    else:
        LOG.info("[audiobookshelf] Skipped (no URL/key configured)")

    # Kapowarr
    if args.kapowarr_url and args.kapowarr_api_key:
        any_source = True
        try:
            all_rows.extend(load_kapowarr(args.kapowarr_url, args.kapowarr_api_key))
        except Exception as exc:
            LOG.error("[kapowarr] Failed: %s", exc)
    else:
        LOG.info("[kapowarr] Skipped (no URL/key configured)")

    # Mylar3
    if args.mylar3_url and args.mylar3_api_key:
        any_source = True
        try:
            all_rows.extend(load_mylar3(args.mylar3_url, args.mylar3_api_key))
        except Exception as exc:
            LOG.error("[mylar3] Failed: %s", exc)
    else:
        LOG.info("[mylar3] Skipped (no URL/key configured)")

    # Podgrab (no auth required by default)
    if args.podgrab_url:
        any_source = True
        try:
            all_rows.extend(load_podgrab(
                args.podgrab_url,
                username=args.podgrab_username or None,
                password=args.podgrab_password or None,
            ))
        except Exception as exc:
            LOG.error("[podgrab] Failed: %s", exc)
    else:
        LOG.info("[podgrab] Skipped (no URL configured)")

    if not any_source:
        LOG.error(
            "No sources configured. Provide at least one --*-url (+ --*-api-key) pair."
        )
        sys.exit(1)

    _write_csv(all_rows, args.output)

    # Summary
    from collections import Counter
    label_counts = Counter(r["ocp_label"] for r in all_rows)
    mt_counts    = Counter(r["media_type"] for r in all_rows)

    print(f"\nTotal rows: {len(all_rows)}")
    print(f"\n{'OCP Label':<35} {'Count':>6}")
    print("-" * 43)
    for label, count in sorted(label_counts.items()):
        print(f"{label:<35} {count:>6}")
    print(f"\n{'MediaType':<35} {'Count':>6}")
    print("-" * 43)
    for mt, count in sorted(mt_counts.items()):
        print(f"{mt:<35} {count:>6}")
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()