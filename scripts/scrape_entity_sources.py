#!/usr/bin/env python3
"""Download entity pools from free public APIs to expand ocp_entities.csv.

Sources (all free, no auth required by default):
  - Gutendex   : https://gutendex.com/books/       — audiobook_title, audiobook_author
  - LibriVox   : https://librivox.org/api/          — audiobook_title, audiobook_author, audiobook_narrator
  - Radio Garden: https://radio.garden/api/ara/     — radio_station, radio_genre
  - Anime Offline DB: GitHub JSON dump              — anime_title, anime_studio
  - Steam      : ISteamApps/GetAppList/v2/          — game_title
  - AniList    : GraphQL API (anilist.co)           — anime_title, anime_studio, music_genre (OSTs)
  - Open Library: openlibrary.org                  — audiobook_title, audiobook_author

Output:
  One CSV per source in the cache dir, then merged/appended to ocp_entities.csv.
  Format: title,ocp_label,media_type,genre,actor,director,producer,writer,
           composer,artist,album,author,narrator,studio,source
  (Same schema as ocp_entities.csv — safe to pd.concat and deduplicate.)

Usage::

    python scripts/scrape_entity_sources.py \\
        --output ~/.cache/ovos-media-classifier/entities/ \\
        --sources gutendex librivox radio_garden anime steam anilist \\
        --merge scripts/ocp_entities.csv

    # Single source
    python scripts/scrape_entity_sources.py --sources gutendex

    # Dry run (show counts only, don't write)
    python scripts/scrape_entity_sources.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ── repo root on sys.path so we can import ovos_media_classifier ────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_CACHE_DIR = Path(os.environ.get(
    "OVOS_MEDIA_CLASSIFIER_CACHE",
    Path.home() / ".cache" / "ovos-media-classifier"
)) / "entities"

_DEFAULT_HEADERS = {
    "User-Agent": "ovos-media-classifier/1.0 (https://github.com/OpenVoiceOS/ovos-media-classifier)",
    "Accept": "application/json",
}

_ENTITY_COLS = [
    "title", "ocp_label", "media_type", "genre",
    "actor", "director", "producer", "writer", "composer",
    "artist", "album", "author", "narrator", "studio", "source",
]


# ── helpers ─────────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[Dict] = None, retries: int = 4,
         delay: float = 1.0) -> dict | list:
    """GET JSON with retry / exponential backoff (handles 429 Too Many Requests)."""
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    for attempt in range(retries):
        try:
            req = Request(url, headers=_DEFAULT_HEADERS)
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code == 429:
                wait = delay * (3 ** attempt)  # longer backoff for rate limits
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
        except URLError as exc:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    raise RuntimeError("unreachable")


def _post_json(url: str, payload: dict, retries: int = 4,
               delay: float = 1.0) -> dict:
    """POST JSON body, return parsed JSON (handles 429 Too Many Requests)."""
    data = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = Request(url, data=data, headers={
                **_DEFAULT_HEADERS,
                "Content-Type": "application/json",
            })
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code == 429:
                wait = delay * (3 ** attempt)
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to POST {url}: {exc}") from exc
        except URLError as exc:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to POST {url}: {exc}") from exc
    raise RuntimeError("unreachable")


def _row(**kwargs) -> Dict[str, str]:
    """Build an entity row with all _ENTITY_COLS, defaulting to empty string."""
    return {col: str(kwargs.get(col, "") or "") for col in _ENTITY_COLS}


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ENTITY_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {path.name}: {len(rows):,} rows")


def _clean(s: str) -> str:
    """Strip whitespace / quotes."""
    return str(s).strip().strip('"').strip("'").strip()


# ── Gutendex ─────────────────────────────────────────────────────────────────

def fetch_gutendex(max_pages: int = 200) -> List[Dict[str, str]]:
    """Download book metadata from Gutendex (Project Gutenberg).

    Returns rows with ocp_label=audiobook_title (title) and audiobook_author.
    """
    rows: List[Dict[str, str]] = []
    url = "https://gutendex.com/books/"
    page = 0
    while url and page < max_pages:
        page += 1
        try:
            data = _get(url)
        except RuntimeError as exc:
            print(f"  [gutendex] error on page {page}: {exc}")
            break

        for book in data.get("results", []):
            title = _clean(book.get("title", ""))
            if not title:
                continue
            authors = [a.get("name", "") for a in book.get("authors", [])]
            author_str = " | ".join(_clean(a) for a in authors if a)
            rows.append(_row(
                title=title,
                ocp_label="audiobook_title",
                media_type="AUDIOBOOK",
                author=author_str,
                source="gutendex",
            ))
            # Also emit author rows
            for author in authors:
                author = _clean(author)
                if author:
                    rows.append(_row(
                        title=author,
                        ocp_label="audiobook_author",
                        media_type="AUDIOBOOK",
                        source="gutendex",
                    ))

        url = data.get("next")  # paginated
        if url:
            time.sleep(0.3)
        sys.stdout.write(f"\r  [gutendex] page {page:3d} — {len(rows):,} rows")
        sys.stdout.flush()
    print()
    return rows


# ── LibriVox ─────────────────────────────────────────────────────────────────

def fetch_librivox() -> List[Dict[str, str]]:
    """Download audiobook catalog from LibriVox public API.

    Paginated JSON at https://librivox.org/api/feed/audiobooks
    """
    rows: List[Dict[str, str]] = []
    base = "https://librivox.org/api/feed/audiobooks"
    offset = 0
    limit = 100
    while True:
        try:
            data = _get(base, params={
                "format": "json",
                "fields": "{id,title,authors,genres,language,sections}",
                "offset": offset,
                "limit": limit,
            })
        except RuntimeError as exc:
            print(f"  [librivox] error at offset {offset}: {exc}")
            break

        books = data.get("books")
        if not books:
            break

        for book in books:
            title = _clean(book.get("title", ""))
            if not title:
                continue

            # Authors
            authors_raw = book.get("authors") or []
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            author_names = [
                _clean(f"{a.get('first_name', '')} {a.get('last_name', '')}").strip()
                for a in authors_raw
            ]
            author_str = " | ".join(a for a in author_names if a)

            # Narrators come from sections[*].readers
            narrator_names: List[str] = []
            for section in (book.get("sections") or []):
                for reader in (section.get("readers") or []):
                    name = _clean(f"{reader.get('first_name','')} {reader.get('last_name','')}").strip()
                    if name and name not in narrator_names:
                        narrator_names.append(name)
            narrator_str = " | ".join(narrator_names[:5])  # cap at 5

            rows.append(_row(
                title=title,
                ocp_label="audiobook_title",
                media_type="AUDIOBOOK",
                author=author_str,
                narrator=narrator_str,
                source="librivox",
            ))
            for a in author_names:
                if a:
                    rows.append(_row(
                        title=a, ocp_label="audiobook_author",
                        media_type="AUDIOBOOK", source="librivox",
                    ))
            for n in narrator_names:
                if n:
                    rows.append(_row(
                        title=n, ocp_label="audiobook_narrator",
                        media_type="AUDIOBOOK", source="librivox",
                    ))

        offset += limit
        sys.stdout.write(f"\r  [librivox] offset {offset:5d} — {len(rows):,} rows")
        sys.stdout.flush()
        time.sleep(0.5)

    print()
    return rows


# ── Radio Garden ─────────────────────────────────────────────────────────────

_RADIO_GARDEN_SEARCH_TERMS = [
    "music", "news", "talk", "sports", "rock", "pop", "jazz", "classical",
    "country", "hip hop", "r&b", "reggae", "electronic", "dance", "folk",
    "blues", "soul", "gospel", "comedy", "drama", "culture", "public radio",
    "community", "adult contemporary", "oldies", "top 40", "hits",
    "christian", "religious", "weather", "traffic",
]


def fetch_radio_garden() -> List[Dict[str, str]]:
    """Download radio station names from Radio Garden search API.

    The bulk channel list endpoint is no longer public, but the search API
    returns up to 20 stations per query.  We run a curated set of genre
    search terms to harvest a broad station + genre list.
    """
    rows: List[Dict[str, str]] = []
    seen_stations: set = set()
    seen_genres: set = set()
    base = "https://radio.garden/api/search"

    for term in _RADIO_GARDEN_SEARCH_TERMS:
        try:
            req = Request(
                f"{base}?q={term.replace(' ', '%20')}",
                headers={**_DEFAULT_HEADERS,
                         "Referer": "https://radio.garden/",
                         "Accept": "application/json"},
            )
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except (URLError, HTTPError, json.JSONDecodeError) as exc:
            print(f"  [radio_garden] {term!r} error: {exc}")
            time.sleep(1.0)
            continue

        hits = (data.get("hits") or {}).get("hits") or []
        for hit in hits:
            src = hit.get("_source") or {}
            page = src.get("page") or {}
            if page.get("type") != "channel":
                continue
            name = _clean(page.get("title", ""))
            if not name or name in seen_stations:
                continue
            seen_stations.add(name)
            rows.append(_row(
                title=name, ocp_label="radio_station",
                media_type="RADIO", genre=term,
                source="radio_garden",
            ))

        # Emit the search term as a genre entity too
        if term not in seen_genres:
            seen_genres.add(term)
            rows.append(_row(
                title=term, ocp_label="radio_genre",
                media_type="RADIO", source="radio_garden",
            ))

        sys.stdout.write(f"\r  [radio_garden] term={term!r:20s} — {len(rows):,} rows")
        sys.stdout.flush()
        time.sleep(0.5)

    print()
    return rows


# ── Anime Offline Database (GitHub) ─────────────────────────────────────────

def fetch_anime_offline_db() -> List[Dict[str, str]]:
    """Download anime metadata from the Anime Offline Database GitHub project.

    Tries several known URL patterns for the minified JSON dump.
    Falls back gracefully if the repository structure has changed.
    """
    _CANDIDATE_URLS = [
        "https://raw.githubusercontent.com/manami-project/anime-offline-database/master/anime-offline-database-minified.json",
        "https://raw.githubusercontent.com/manami-project/anime-offline-database/main/anime-offline-database-minified.json",
        "https://raw.githubusercontent.com/manami-project/anime-offline-database/master/anime-offline-database.json",
        "https://raw.githubusercontent.com/manami-project/anime-offline-database/main/anime-offline-database.json",
    ]
    rows: List[Dict[str, str]] = []
    raw = None
    for url in _CANDIDATE_URLS:
        print(f"  [anime_offline_db] trying {url.split('/')[-1]} …", end="", flush=True)
        try:
            req = Request(url, headers=_DEFAULT_HEADERS)
            with urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode())
            print(" OK")
            break
        except (URLError, HTTPError, json.JSONDecodeError) as exc:
            print(f" {exc}")

    if not raw:
        print("  [anime_offline_db] all URLs failed — skipping")
        return rows

    for anime in raw.get("data", []):
        title = _clean(anime.get("title", ""))
        if not title:
            continue
        # synonyms (alternative titles, sometimes in English)
        synonyms = [_clean(s) for s in (anime.get("synonyms") or []) if s]
        # tags are loose genre labels
        tags = anime.get("tags") or []

        rows.append(_row(
            title=title,
            ocp_label="anime_title",
            media_type="ANIME",
            genre=" | ".join(tags[:5]),
            source="anime_offline_db",
        ))
        for syn in synonyms[:3]:
            if syn and syn != title:
                rows.append(_row(
                    title=syn, ocp_label="anime_title",
                    media_type="ANIME", source="anime_offline_db",
                ))

    print(f"  [anime_offline_db] {len(rows):,} rows")
    return rows


# ── AniList GraphQL ──────────────────────────────────────────────────────────

_ANILIST_URL = "https://graphql.anilist.co"
_ANILIST_QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: POPULARITY_DESC) {
      title { romaji english }
      studios(isMain: true) { nodes { name } }
      genres
    }
  }
}
"""


def fetch_anilist(max_pages: int = 100) -> List[Dict[str, str]]:
    """Download anime titles + studios from AniList GraphQL API."""
    rows: List[Dict[str, str]] = []
    for page in range(1, max_pages + 1):
        try:
            resp = _post_json(_ANILIST_URL, {
                "query": _ANILIST_QUERY,
                "variables": {"page": page},
            })
        except RuntimeError as exc:
            print(f"  [anilist] error on page {page}: {exc}")
            break

        page_data = (resp.get("data") or {}).get("Page") or {}
        media_list = page_data.get("media") or []

        for m in media_list:
            titles = m.get("title") or {}
            title = _clean(titles.get("english") or titles.get("romaji") or "")
            if not title:
                continue
            studios = [n.get("name", "") for n in ((m.get("studios") or {}).get("nodes") or [])]
            studio_str = " | ".join(_clean(s) for s in studios if s)
            genres = m.get("genres") or []

            rows.append(_row(
                title=title,
                ocp_label="anime_title",
                media_type="ANIME",
                genre=" | ".join(genres[:5]),
                studio=studio_str,
                source="anilist",
            ))
            for studio in studios:
                studio = _clean(studio)
                if studio:
                    rows.append(_row(
                        title=studio, ocp_label="anime_studio",
                        media_type="ANIME", source="anilist",
                    ))

        has_next = page_data.get("pageInfo", {}).get("hasNextPage", False)
        sys.stdout.write(f"\r  [anilist] page {page:3d} — {len(rows):,} rows")
        sys.stdout.flush()
        if not has_next:
            break
        time.sleep(1.2)  # AniList rate limit: ~90 req/min; stay well under

    print()
    return rows


# ── Steam ────────────────────────────────────────────────────────────────────

_STEAM_APP_LIST_URLS = [
    # Steam removed the public v2 endpoint; try known mirrors / alternates
    "https://api.steampowered.com/ISteamApps/GetAppList/v2/",
    "https://api.steampowered.com/ISteamApps/GetAppList/v1/",
    # SteamSpy has a public all-games endpoint (includes ~75k games)
    "https://steamspy.com/api.php?request=all",
]

_STEAM_NOISE = (
    " - soundtrack", " ost", " dlc", " demo", " server", " tool",
    "dedicated server", "beta test", "test server", "playtest",
    "trailer", "making of",
)


def fetch_steam() -> List[Dict[str, str]]:
    """Download game names from Steam / SteamSpy public API.

    Tries the official Steam ISteamApps endpoint first, then falls back to
    SteamSpy which also has a free all-apps list.
    """
    rows: List[Dict[str, str]] = []
    data = None
    for url in _STEAM_APP_LIST_URLS:
        print(f"  [steam] trying {url.split('?')[0].split('/')[-2:]} …", end="", flush=True)
        try:
            req = Request(url, headers=_DEFAULT_HEADERS)
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            print(" OK")
            break
        except (URLError, HTTPError, json.JSONDecodeError) as exc:
            print(f" {exc}")

    if not data:
        print("  [steam] all endpoints failed — skipping")
        return rows

    # Steam format: {"applist": {"apps": [{"appid":..., "name":...}]}}
    # SteamSpy format: {"appid": {"name":..., "genre":...}, ...}
    seen: set = set()

    if "applist" in data:
        apps = (data.get("applist") or {}).get("apps") or []
        for app in apps:
            name = _clean(app.get("name", ""))
            if not name or name in seen:
                continue
            low = name.lower()
            if any(noise in low for noise in _STEAM_NOISE):
                continue
            seen.add(name)
            rows.append(_row(title=name, ocp_label="game_title",
                             media_type="GAME", source="steam"))
    else:
        # SteamSpy format
        for appid, info in data.items():
            if not isinstance(info, dict):
                continue
            name = _clean(info.get("name", ""))
            if not name or name in seen:
                continue
            low = name.lower()
            if any(noise in low for noise in _STEAM_NOISE):
                continue
            seen.add(name)
            genre = _clean(info.get("genre", ""))
            rows.append(_row(title=name, ocp_label="game_title",
                             media_type="GAME", genre=genre, source="steamspy"))

    print(f"  [steam] {len(rows):,} rows")
    return rows


# ── Open Library ─────────────────────────────────────────────────────────────

def fetch_open_library(subjects: Optional[List[str]] = None,
                       max_per_subject: int = 1000) -> List[Dict[str, str]]:
    """Download book metadata from Open Library subject search."""
    if subjects is None:
        subjects = [
            "audiobooks", "fiction", "science_fiction", "mystery",
            "biography", "history", "fantasy", "romance",
        ]
    rows: List[Dict[str, str]] = []
    base = "https://openlibrary.org/subjects/"
    seen_titles: set = set()
    for subject in subjects:
        offset = 0
        limit = 100
        fetched = 0
        while fetched < max_per_subject:
            try:
                data = _get(f"{base}{subject}.json",
                            params={"limit": limit, "offset": offset})
            except RuntimeError as exc:
                print(f"  [open_library/{subject}] error: {exc}")
                break
            works = data.get("works") or []
            if not works:
                break
            for work in works:
                title = _clean(work.get("title", ""))
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                authors = [_clean(a.get("name", "")) for a in (work.get("authors") or [])]
                author_str = " | ".join(a for a in authors if a)
                rows.append(_row(
                    title=title,
                    ocp_label="audiobook_title",
                    media_type="AUDIOBOOK",
                    author=author_str,
                    source="open_library",
                ))
                for a in authors:
                    if a:
                        rows.append(_row(
                            title=a, ocp_label="audiobook_author",
                            media_type="AUDIOBOOK", source="open_library",
                        ))
            fetched += len(works)
            offset += limit
            if len(works) < limit:
                break
            time.sleep(0.5)

        sys.stdout.write(f"\r  [open_library] subject={subject} — {len(rows):,} rows")
        sys.stdout.flush()

    print()
    return rows


# ── merge helpers ─────────────────────────────────────────────────────────────

def _deduplicate(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove rows with duplicate (title, ocp_label) pairs (case-insensitive)."""
    seen: set = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        key = (row["title"].lower().strip(), row["ocp_label"].lower().strip())
        if key not in seen and key[0] and key[1]:
            seen.add(key)
            out.append(row)
    return out


def _load_existing(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({col: row.get(col, "") for col in _ENTITY_COLS})
    return rows


def merge_into_csv(new_rows: List[Dict[str, str]], target_csv: str,
                   dry_run: bool = False) -> None:
    """Append deduplicated new_rows to target_csv (preserving existing entries)."""
    existing = _load_existing(target_csv)
    combined = _deduplicate(existing + new_rows)
    added = len(combined) - len(existing)
    print(f"\nMerge summary: {len(existing):,} existing + {len(new_rows):,} new → "
          f"{len(combined):,} total ({added:+,} net new after dedup)")

    if dry_run:
        print("  [dry-run] not writing.")
        return

    target = Path(target_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ENTITY_COLS)
        writer.writeheader()
        writer.writerows(combined)
    print(f"  Written: {target}")


# ── label coverage report ────────────────────────────────────────────────────

def print_coverage(rows: List[Dict[str, str]]) -> None:
    counts: Dict[str, int] = {}
    for row in rows:
        lbl = row.get("ocp_label", "")
        if lbl:
            counts[lbl] = counts.get(lbl, 0) + 1
    print("\nEntity label coverage:")
    for lbl, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:35s}: {n:7,}")


# ── Radio Browser (radio-browser.info) ──────────────────────────────────────

_RADIO_BROWSER_SERVERS = [
    "https://de1.api.radio-browser.info",
    "https://nl1.api.radio-browser.info",
    "https://at1.api.radio-browser.info",
]


def fetch_radio_browser(limit: int = 50000) -> List[Dict[str, str]]:
    """Download radio station list from radio-browser.info (community database).

    No auth required.  Tries multiple mirror servers.  Returns radio_station
    and radio_genre rows.  Coverage: 30,000+ worldwide stations.
    """
    rows: List[Dict[str, str]] = []
    data = None
    for server in _RADIO_BROWSER_SERVERS:
        url = f"{server}/json/stations?limit={limit}&hidebroken=true&order=votes&reverse=true"
        print(f"  [radio_browser] trying {server} …", end="", flush=True)
        try:
            req = Request(url, headers=_DEFAULT_HEADERS)
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            print(f" OK ({len(data):,} stations)")
            break
        except (URLError, HTTPError, json.JSONDecodeError) as exc:
            print(f" {exc}")

    if not data:
        print("  [radio_browser] all servers failed — skipping")
        return rows

    seen_stations: set = set()
    seen_genres: set = set()
    for station in data:
        name = _clean(station.get("name", ""))
        if not name or name in seen_stations:
            continue
        seen_stations.add(name)
        tags_raw = station.get("tags", "") or ""
        genres = [_clean(t) for t in tags_raw.split(",") if _clean(t)]
        rows.append(_row(
            title=name, ocp_label="radio_station",
            media_type="RADIO",
            genre=" | ".join(genres[:3]),
            source="radio_browser",
        ))
        for genre in genres:
            if genre and genre not in seen_genres and len(genre) > 2:
                seen_genres.add(genre)
                rows.append(_row(
                    title=genre, ocp_label="radio_genre",
                    media_type="RADIO", source="radio_browser",
                ))

    print(f"  [radio_browser] {len(rows):,} rows "
          f"({len(seen_stations):,} stations, {len(seen_genres):,} genres)")
    return rows


# ── PornHub Webmasters API ───────────────────────────────────────────────────

_PH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.pornhub.com/",
}
_PH_BASE = "https://www.pornhub.com/webmasters"

# Category slugs to skip (non-entity tags, technical quality labels, etc.)
_PH_SKIP_CATS = frozenset({
    "180-1", "360-1", "60fps-1", "2d", "3d", "hd-porn", "4k",
    "vr", "interactive", "sbs-vr", "psvr",
})


def _ph_fetch_videos_page(url: str) -> List[dict]:
    """Fetch one page of PornHub search results; return raw video dicts."""
    try:
        req = Request(url, headers=_PH_HEADERS)
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode()).get("videos") or []
    except (URLError, HTTPError, json.JSONDecodeError):
        return []


def fetch_pornhub(min_videos: int = 1,
                  video_pages: int = 50,
                  cat_video_pages: int = 3) -> List[Dict[str, str]]:
    """Download pornstar names, category tags, and video titles from PornHub.

    The Webmasters API is public and requires no authentication.

    Endpoints used:
      - /webmasters/categories         — genre/category list (164 tags)
      - /webmasters/stars_detailed     — full performer list (27k+, single page)
      - /webmasters/search             — video search (30 results/page, paginated)

    Entity labels produced:
      - ``adult_title`` — video titles
      - ``pornstar``    — performer names
      - ``porn_genre``  — category/tag strings

    Args:
        min_videos:      Minimum video count to include a performer (default 1).
        video_pages:     Pages of top-viewed global videos to fetch (default 50
                         → up to 1,500 titles).
        cat_video_pages: Pages of per-category videos to fetch (default 3 × 164
                         categories → up to ~15k additional titles).
    """
    rows: List[Dict[str, str]] = []
    seen_titles: set = set()

    # ── Categories (used both as porn_genre rows and as search keys) ─────────
    print("  [pornhub] fetching categories …", end="", flush=True)
    cat_slugs: List[str] = []
    try:
        req = Request(f"{_PH_BASE}/categories", headers=_PH_HEADERS)
        with urlopen(req, timeout=20) as resp:
            cats = json.loads(resp.read().decode()).get("categories") or []
        cat_count = 0
        for cat in cats:
            slug = str(cat.get("category", "")).strip()
            if not slug or slug in _PH_SKIP_CATS:
                continue
            cat_slugs.append(slug)
            label = slug.rstrip("-1").replace("-", " ").strip()
            if label:
                rows.append(_row(title=label, ocp_label="porn_genre",
                                 media_type="ADULT", source="pornhub"))
                cat_count += 1
        print(f" {cat_count} genres")
    except (URLError, HTTPError, json.JSONDecodeError) as exc:
        print(f" error: {exc}")

    # ── Performers ──────────────────────────────────────────────────────────
    print("  [pornhub] fetching performers …", end="", flush=True)
    try:
        req = Request(f"{_PH_BASE}/stars_detailed?page=1", headers=_PH_HEADERS)
        with urlopen(req, timeout=30) as resp:
            stars = json.loads(resp.read().decode()).get("stars") or []
        star_count = 0
        for entry in stars:
            star = entry.get("star") or {}
            name = _clean(star.get("star_name", ""))
            if not name:
                continue
            try:
                n_videos = int(star.get("videos_count_all") or 0)
            except (ValueError, TypeError):
                n_videos = 0
            if n_videos < min_videos:
                continue
            rows.append(_row(title=name, ocp_label="pornstar",
                             media_type="ADULT", source="pornhub"))
            star_count += 1
        print(f" {star_count:,} performers (of {len(stars):,} listed)")
    except (URLError, HTTPError, json.JSONDecodeError) as exc:
        print(f" error: {exc}")

    # ── Video titles: global top-viewed ─────────────────────────────────────
    print(f"  [pornhub] fetching top videos ({video_pages} pages) …")
    for page in range(1, video_pages + 1):
        url = (f"{_PH_BASE}/search?search=&ordering=mostviewed"
               f"&page={page}&thumbsize=small")
        videos = _ph_fetch_videos_page(url)
        if not videos:
            break
        for v in videos:
            title = _clean(v.get("title", ""))
            if title and title not in seen_titles:
                seen_titles.add(title)
                rows.append(_row(title=title, ocp_label="adult_title",
                                 media_type="ADULT", source="pornhub"))
        sys.stdout.write(f"\r    global page {page:3d}/{video_pages} "
                         f"— {len(seen_titles):,} titles so far")
        sys.stdout.flush()
        time.sleep(0.4)
    print()

    # ── Video titles: per category sweep ────────────────────────────────────
    if cat_video_pages > 0 and cat_slugs:
        print(f"  [pornhub] sweeping {len(cat_slugs)} categories "
              f"× {cat_video_pages} pages …")
        for ci, slug in enumerate(cat_slugs):
            for page in range(1, cat_video_pages + 1):
                url = (f"{_PH_BASE}/search?search=&ordering=mostviewed"
                       f"&category={slug}&page={page}&thumbsize=small")
                videos = _ph_fetch_videos_page(url)
                if not videos:
                    break
                for v in videos:
                    title = _clean(v.get("title", ""))
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        rows.append(_row(title=title, ocp_label="adult_title",
                                         media_type="ADULT", source="pornhub"))
                time.sleep(0.3)
            sys.stdout.write(f"\r    category {ci+1:3d}/{len(cat_slugs)} "
                             f"({slug:25s}) — {len(seen_titles):,} titles")
            sys.stdout.flush()
        print()

    print(f"  [pornhub] total adult_title: {len(seen_titles):,}")
    return rows


# ── IMDB / Wikidata ──────────────────────────────────────────────────────────

_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_WIKIDATA_HEADERS = {
    "User-Agent": "ovos-media-classifier/1.0 (https://github.com/OpenVoiceOS/ovos-media-classifier)",
    "Accept": "application/sparql-results+json",
}

# Wikidata Q-IDs for media types we care about
_WD_TYPES = {
    "movie_title": [
        "wd:Q11424",   # film
        "wd:Q24869",   # animated film
        "wd:Q93204",   # documentary film
        "wd:Q506240",  # short film
    ],
    "tv_show_title": [
        "wd:Q5398426",  # television series
        "wd:Q21191270", # TV miniseries
    ],
    "cartoon_title": [
        "wd:Q202866",  # animated series
    ],
}

_WD_PERSON_TYPES = {
    "movie_actor":    "wd:Q33999",   # actor
    "movie_director": "wd:Q2526255", # film director
}


def _wd_query(sparql: str) -> List[dict]:
    """Run a Wikidata SPARQL query; return list of binding dicts."""
    from urllib.parse import urlencode
    url = f"{_WIKIDATA_SPARQL}?{urlencode({'query': sparql, 'format': 'json'})}"
    try:
        req = Request(url, headers=_WIKIDATA_HEADERS)
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())["results"]["bindings"]
    except Exception as exc:
        print(f"\n  [imdb/wikidata] query error: {exc}")
        return []


def fetch_imdb(titles_per_type: int = 5000,
               persons_per_role: int = 5000) -> List[Dict[str, str]]:
    """Fetch movie titles, TV show titles, actors and directors from Wikidata.

    Wikidata is the largest open knowledge base and is structurally equivalent
    to IMDB data (it links to IMDB IDs) — with no auth or rate limits for
    SPARQL queries up to 10,000 rows.

    Entity labels produced:
      - ``movie_title``    — film titles
      - ``tv_show_title``  — TV series titles
      - ``cartoon_title``  — animated series
      - ``movie_actor``    — actor/actress names
      - ``movie_director`` — film director names

    Args:
        titles_per_type:   Max titles to fetch per media type (default 5,000).
        persons_per_role:  Max person names per occupation (default 5,000).
    """
    rows: List[Dict[str, str]] = []
    seen: set = set()

    def _add(title: str, ocp_label: str, media_type: str) -> None:
        key = (title.lower(), ocp_label)
        if key not in seen and title:
            seen.add(key)
            rows.append(_row(title=title, ocp_label=ocp_label,
                             media_type=media_type, source="wikidata"))

    # ── Titles ───────────────────────────────────────────────────────────────
    _MT_MAP = {
        "movie_title":   ("MOVIE", _WD_TYPES["movie_title"]),
        "tv_show_title": ("TV",    _WD_TYPES["tv_show_title"]),
        "cartoon_title": ("CARTOON", _WD_TYPES["cartoon_title"]),
    }

    for ocp_label, (media_type, qids) in _MT_MAP.items():
        type_filter = " ".join(f"{{ ?item wdt:P31 {q} }}" for q in qids)
        union = " UNION ".join(f"{{ ?item wdt:P31 {q} }}" for q in qids)
        sparql = f"""
SELECT DISTINCT ?title WHERE {{
  {{ {union} }}
  ?item rdfs:label ?title FILTER(lang(?title) = "en")
}} LIMIT {titles_per_type}
"""
        bindings = _wd_query(sparql)
        before = len(seen)
        for b in bindings:
            _add(_clean(b.get("title", {}).get("value", "")), ocp_label, media_type)
        added = len(seen) - before
        print(f"  [imdb/wikidata] {ocp_label:20s}: {added:,} titles")
        time.sleep(1.0)  # Wikidata requests polite delay

    # ── People ───────────────────────────────────────────────────────────────
    for ocp_label, qid in _WD_PERSON_TYPES.items():
        sparql = f"""
SELECT DISTINCT ?name WHERE {{
  ?person wdt:P106 {qid};
          rdfs:label ?name FILTER(lang(?name) = "en")
}} LIMIT {persons_per_role}
"""
        bindings = _wd_query(sparql)
        before = len(seen)
        for b in bindings:
            _add(_clean(b.get("name", {}).get("value", "")), ocp_label, "MOVIE")
        added = len(seen) - before
        print(f"  [imdb/wikidata] {ocp_label:20s}: {added:,} people")
        time.sleep(1.0)

    print(f"  [imdb/wikidata] total: {len(rows):,} rows")
    return rows


# ── Adult / Hentai entity extraction ────────────────────────────────────────

# Curated porn genre list — standard industry tags (public knowledge)
_PORN_GENRES: List[str] = [
    "amateur", "anal", "asian", "BBW", "BDSM", "big ass", "big tits",
    "bisexual", "blonde", "blowjob", "bondage", "brunette", "casting",
    "compilation", "cosplay", "creampie", "cumshot", "ebony", "european",
    "facial", "feet", "femdom", "fetish", "gangbang", "gay", "german",
    "girl on girl", "gonzo", "granny", "group sex", "handjob", "hardcore",
    "interracial", "japanese", "latina", "lesbian", "MILF", "massage",
    "masturbation", "mature", "natural tits", "orgy", "outdoor", "parody",
    "pov", "petite", "public", "redhead", "roleplay", "romantic", "rough sex",
    "schoolgirl", "softcore", "solo", "squirting", "striptease", "swinger",
    "teen", "threesome", "toys", "trans", "uniform", "vintage",
    "voyeur", "wife",
]

# Well-known adult studios (public companies)
_ADULT_STUDIOS: List[str] = [
    "Bang Bros", "Brazzers", "Digital Playground", "Evil Angel",
    "Elegant Angel", "Falcon Entertainment", "Girlfriends Films",
    "Girlsway", "Harmony Films", "Hustler", "Jules Jordan Video",
    "Kink", "Lethal Hardcore", "Manual Films", "Marc Dorcel",
    "Mofos", "New Sensations", "Nubiles", "Penthouse",
    "Perv Moms", "Playboy", "Private Media Group",
    "Pure Taboo", "Reality Kings", "Rotten Cotton",
    "Score", "Scoreland", "Skow", "Slayed", "Spizoo",
    "Stoney Curtis Films", "Subway Entertainment",
    "TeamSkeet", "Tiny4K", "TransAngels", "Tushy",
    "Twistys", "Vivid Entertainment", "Wicked Pictures",
    "Women Who Love Women", "X-Art", "Zero Tolerance",
]

# Hentai-specific genres
_HENTAI_GENRES: List[str] = [
    "ahegao", "dark skin", "defloration", "demon girl", "elf",
    "futanari", "gynoid", "harem", "impregnation", "incest",
    "kemonomimi", "loli", "maid", "mecha musume", "mind control",
    "monster girl", "netorare", "netorase", "ntr", "nun",
    "oppai", "orc", "overpowered", "rape", "reverse rape",
    "romance", "schoolgirl", "succubus", "swimsuit",
    "tentacles", "trap", "tsundere", "vampire", "vanilla",
    "witch", "yaoi", "yuri", "zombie",
]

# Adult streaming services
_ADULT_SERVICES: List[str] = [
    "Brazzers", "MindGeek", "OnlyFans", "Pornhub", "xVideos",
    "xHamster", "xNxx", "RedTube", "YouPorn",
    "AdultTime", "Aebn", "SexArt", "Vixen Media",
    "ManyVids", "clips4sale", "Chaturbate",
]

_ANILIST_HENTAI_QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: ANIME, isAdult: true, sort: POPULARITY_DESC) {
      title { romaji english }
      studios(isMain: true) { nodes { name } }
      genres
    }
  }
}
"""


def fetch_adult(entities_csv: Optional[str] = None) -> List[Dict[str, str]]:
    """Build adult entity rows from three sources:

    1. Re-emit pornstar names and studio names already in ocp_entities.csv
       with proper OCPEntityLabel values (``pornstar``, ``adult_streaming_service``)
    2. Curated genre, studio, and streaming service lists
    3. AniList hentai titles and studios (isAdult=true filter)
    """
    rows: List[Dict[str, str]] = []

    # ── 1. Re-emit actors/studios already embedded in existing adult rows ────
    if entities_csv is None:
        entities_csv = str(_REPO_ROOT / "scripts" / "ocp_entities.csv")

    seen_stars: set = set()
    seen_studios: set = set()
    seen_titles: set = set()

    if os.path.exists(entities_csv):
        with open(entities_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mt = row.get("media_type", "").strip().upper()
                lbl = row.get("ocp_label", "").strip()
                if mt not in ("ADULT", "HENTAI") and lbl not in (
                    "adult", "adult_title", "hentai_title", "pornstar",
                    "porn_genre", "adult_streaming_service",
                ):
                    continue

                title = _clean(row.get("title", ""))
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    out_lbl = "hentai_title" if mt == "HENTAI" else "adult_title"
                    rows.append(_row(title=title, ocp_label=out_lbl,
                                     media_type=mt or "ADULT", source="whisparr"))

                for actor in (row.get("actor") or "").split("|"):
                    actor = _clean(actor)
                    if actor and actor not in seen_stars:
                        seen_stars.add(actor)
                        rows.append(_row(title=actor, ocp_label="pornstar",
                                         media_type="ADULT", source="whisparr"))

                for studio in (row.get("studio") or "").split("|"):
                    studio = _clean(studio)
                    if studio and studio not in seen_studios:
                        seen_studios.add(studio)
                        rows.append(_row(title=studio, ocp_label="adult_streaming_service",
                                         media_type="ADULT", source="whisparr"))

        print(f"  [adult] from existing CSV: {len(seen_titles):,} titles, "
              f"{len(seen_stars):,} performers, {len(seen_studios):,} studios")

    # ── 2. Curated lists ────────────────────────────────────────────────────
    for genre in _PORN_GENRES:
        rows.append(_row(title=genre, ocp_label="porn_genre",
                         media_type="ADULT", source="curated"))
    for genre in _HENTAI_GENRES:
        rows.append(_row(title=genre, ocp_label="porn_genre",
                         media_type="HENTAI", source="curated"))
    for studio in _ADULT_STUDIOS:
        if studio not in seen_studios:
            rows.append(_row(title=studio, ocp_label="adult_streaming_service",
                             media_type="ADULT", source="curated"))
    for service in _ADULT_SERVICES:
        rows.append(_row(title=service, ocp_label="adult_streaming_service",
                         media_type="ADULT", source="curated"))

    print(f"  [adult] curated: {len(_PORN_GENRES)} porn_genre, "
          f"{len(_HENTAI_GENRES)} hentai_genre, "
          f"{len(_ADULT_STUDIOS) + len(_ADULT_SERVICES)} services/studios")

    # ── 3. AniList hentai ───────────────────────────────────────────────────
    hentai_count = 0
    studio_count = 0
    for page in range(1, 51):
        try:
            resp = _post_json(_ANILIST_URL, {
                "query": _ANILIST_HENTAI_QUERY,
                "variables": {"page": page},
            })
        except RuntimeError as exc:
            print(f"\n  [adult/anilist] error on page {page}: {exc}")
            break

        page_data = (resp.get("data") or {}).get("Page") or {}
        for m in (page_data.get("media") or []):
            titles = m.get("title") or {}
            title = _clean(titles.get("english") or titles.get("romaji") or "")
            if title:
                rows.append(_row(title=title, ocp_label="hentai_title",
                                 media_type="HENTAI", source="anilist_adult"))
                hentai_count += 1
            for node in ((m.get("studios") or {}).get("nodes") or []):
                studio = _clean(node.get("name", ""))
                if studio:
                    rows.append(_row(title=studio, ocp_label="adult_streaming_service",
                                     media_type="HENTAI", source="anilist_adult"))
                    studio_count += 1

        has_next = page_data.get("pageInfo", {}).get("hasNextPage", False)
        sys.stdout.write(f"\r  [adult/anilist] page {page:2d} — "
                         f"{hentai_count:,} hentai, {studio_count:,} studios")
        sys.stdout.flush()
        if not has_next:
            break
        time.sleep(1.2)

    print()
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

_ALL_SOURCES = [
    "gutendex", "librivox", "radio_garden", "radio_browser",
    "anime_offline_db", "anilist",
    "steam", "open_library",
    "imdb",
    "adult", "pornhub",
]

_FETCHERS = {
    "gutendex":        fetch_gutendex,
    "librivox":        fetch_librivox,
    "radio_garden":    fetch_radio_garden,
    "radio_browser":   fetch_radio_browser,
    "anime_offline_db": fetch_anime_offline_db,
    "anilist":         fetch_anilist,
    "steam":           fetch_steam,
    "open_library":    fetch_open_library,
    "imdb":            fetch_imdb,
    "adult":           fetch_adult,
    "pornhub":         fetch_pornhub,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download entity pools from free public APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sources", default=",".join(_ALL_SOURCES),
                        help="Comma-separated list of sources to fetch "
                             f"(default: all). Available: {', '.join(_ALL_SOURCES)}")
    parser.add_argument("--output", default=str(_CACHE_DIR),
                        help="Directory for per-source CSV files (default: %(default)s)")
    parser.add_argument("--merge", default=None,
                        help="Path to ocp_entities.csv to append results to "
                             "(default: scripts/ocp_entities.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report counts but do not write files")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging into ocp_entities.csv")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    invalid = [s for s in sources if s not in _FETCHERS]
    if invalid:
        print(f"ERROR: Unknown sources: {invalid}")
        print(f"Available: {', '.join(_ALL_SOURCES)}")
        sys.exit(1)

    output_dir = Path(args.output)
    merge_target = args.merge or str(_REPO_ROOT / "scripts" / "ocp_entities.csv")

    all_new_rows: List[Dict[str, str]] = []

    for source in sources:
        print(f"\n[{source}]")
        try:
            fetcher = _FETCHERS[source]
            rows = fetcher()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        rows = _deduplicate(rows)
        print(f"  Deduplicated: {len(rows):,} rows")

        out_path = output_dir / f"{source}_entities.csv"
        if not args.dry_run:
            _write_csv(rows, out_path)
        else:
            print(f"  [dry-run] would write to {out_path}")

        all_new_rows.extend(rows)

    print_coverage(all_new_rows)

    if not args.no_merge:
        merge_into_csv(all_new_rows, merge_target, dry_run=args.dry_run)
    else:
        print(f"\nSkipping merge (--no-merge). "
              f"Re-run without --no-merge to append to {merge_target}")


if __name__ == "__main__":
    main()
