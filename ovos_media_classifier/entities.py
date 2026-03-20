"""OCP EntitiesContainer — unified entity registry for AhocorasickMediaClassifier.

Aggregates entity strings (artist names, movie titles, TV show titles, …) from
multiple sources and keeps them in sync with an underlying AhocorasickNER so
that any entity added at runtime is immediately reflected in classification
results without rebuilding the automaton.

Supported sources
-----------------
- In-memory word lists / dicts
- CSV files (the format produced by ``generate_dataset_from_media.py``)
- Jellyfin  (movies, TV, music, audiobooks, podcasts)
- Radarr    (movies + cast/crew)
- Sonarr    (TV shows, anime, documentaries)
- Lidarr    (artists, albums, tracks, genres)
- HuggingFace ``datasets`` (any dataset with entity/label columns)

Dependencies
------------
- ``ahocorasick-ner``  — required to materialise the NER (``pip install ovos-media-classifier[ner]``)
- ``requests``         — required for media-server loaders  (``pip install ovos-media-classifier[media_servers]``)
- ``datasets``         — required for HuggingFace loader   (``pip install ovos-media-classifier[huggingface]``)

All three are optional — the container works as a pure data store without them.

Runtime awareness
-----------------
The same ``AhocorasickNER`` instance is shared between the container and
``AhocorasickMediaClassifier``.  Every call to :meth:`add` propagates to the
NER via ``ner.add_word()`` immediately, so newly registered entities
(skills announcing their content, media-server libraries updated mid-session)
are reflected in classification results with no rebuild step.

Example::

    container = EntitiesContainer()
    container.load_radarr("http://localhost:7878", api_key="...")
    container.load_sonarr("http://localhost:8989", api_key="...")
    container.load_lidarr("http://localhost:8686", api_key="...")
    container.load_jellyfin("http://localhost:8096", api_key="...")

    clf = AhocorasickMediaClassifier.from_container(container)
    clf.classify("play the dark knight", "en-us")
    # → (MediaType.MOVIE, 0.6)

    # New entity registered at runtime (e.g. from a skill announcement):
    container.add("artist_name", "Radiohead")
    clf.classify("play radiohead", "en-us")
    # → (MediaType.MUSIC, 0.6)  — no rebuild needed
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ovos_utils.log import LOG

# ---------------------------------------------------------------------------
# Genre helpers (shared with the standalone dataset-generation script)
# ---------------------------------------------------------------------------

_ANIME_GENRES = {"anime"}
_CARTOON_GENRES = {"animation", "animated", "cartoon"}
_DOC_GENRES = {"documentary"}


def _video_label_from_genres(genres: List[str], default: str) -> str:
    """Refine a generic video label based on genre tags.

    Args:
        genres: List of genre strings from the media server.
        default: Label to return if no genre matches (e.g., "movie_title").

    Returns:
        A more specific entity label based on genres, or the default.
    """
    gl = {g.lower() for g in genres}
    if gl & _ANIME_GENRES:
        return "anime_title"
    if gl & _DOC_GENRES:
        return "documentary_title"
    if gl & _CARTOON_GENRES:
        return "cartoon_title"
    return default


# ---------------------------------------------------------------------------
# HTTP helper (lazy import of requests)
# ---------------------------------------------------------------------------


def _http_get(session: Any, url: str, **kwargs: Any) -> Optional[Any]:
    """Perform an HTTP GET request with error handling.

    Args:
        session: A requests.Session object with authentication configured.
        url: The URL to GET.
        **kwargs: Additional arguments passed to session.get().

    Returns:
        Parsed JSON response (dict or list), or None on failure.
    """
    try:
        resp = session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        LOG.warning("HTTP GET %s failed: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# EntitiesContainer
# ---------------------------------------------------------------------------


class EntitiesContainer:
    """Unified entity store for ``AhocorasickMediaClassifier``.

    Holds entity strings grouped by :class:`~ovos_media_classifier.intents.OCPEntityLabel`
    value strings and optionally backs an ``AhocorasickNER`` instance for
    fast substring matching.

    Args:
        ner: An existing ``AhocorasickNER`` to attach.  When provided the
             container will use it directly (sharing by reference) so any
             :meth:`add` call immediately updates the automaton.  When
             ``None`` (default) a new NER is created the first time
             :attr:`ner` is accessed.
    """

    def __init__(self, ner=None) -> None:
        self._ner = ner  # AhocorasickNER | None
        self._by_label: Dict[str, Set[str]] = defaultdict(set)  # dedup + stats

        # Back-fill from an existing NER (best-effort — only if it exposes
        # the internal word dict; not critical if unavailable)
        if ner is not None:
            self._sync_from_ner(ner)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_from_ner(self, ner) -> None:
        """Populate ``_by_label`` from an existing NER's word store."""
        try:
            for label, words in ner.get_all_words().items():
                self._by_label[label].update(words)
        except Exception:
            pass  # NER may not expose this; non-fatal

    def _get_or_create_ner(self):
        if self._ner is None:
            try:
                from ahocorasick_ner import AhocorasickNER
            except ImportError:
                raise ImportError(
                    "ahocorasick-ner is required to materialise the NER. "
                    "Install it with: pip install ovos-media-classifier[ner]"
                )
            self._ner = AhocorasickNER()
            # Populate from accumulated data
            for label, words in self._by_label.items():
                for word in words:
                    self._ner.add_word(label, word)
        return self._ner

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def ner(self):
        """The underlying ``AhocorasickNER``, created on first access."""
        return self._get_or_create_ner()

    def add(self, label: str, entity: str) -> None:
        """Register a single entity string under *label*.

        Immediately propagates to the NER if one is already attached.
        Safe to call from any thread — ``AhocorasickNER.add_word`` is
        designed for incremental updates.
        """
        entity = entity.strip()
        if not entity:
            return
        if entity in self._by_label[label]:
            return  # already registered
        self._by_label[label].add(entity)
        if self._ner is not None:
            self._ner.add_word(label, entity)

    def add_many(self, items: Iterable[Tuple[str, str]]) -> None:
        """Register multiple ``(label, entity)`` pairs."""
        for label, entity in items:
            self.add(label, entity)

    @property
    def wordlists(self) -> Dict[str, List[str]]:
        """Return ``{label: [entity, …]}`` dict (a snapshot, not live)."""
        return {label: sorted(words) for label, words in self._by_label.items()}

    @property
    def stats(self) -> Dict[str, int]:
        """Entity count per label."""
        return {label: len(words) for label, words in sorted(self._by_label.items())}

    def __len__(self) -> int:
        return sum(len(w) for w in self._by_label.values())

    def __repr__(self) -> str:
        total = len(self)
        labels = len(self._by_label)
        return f"EntitiesContainer({total} entities across {labels} labels)"

    # ------------------------------------------------------------------
    # CSV loader (format from generate_dataset_from_media.py)
    # ------------------------------------------------------------------

    def load_csv(
        self,
        path: str,
        entity_col: str = "entity",
        label_col: str = "label",
    ) -> int:
        """Load entities from a CSV file.

        Accepts both named-column CSVs (``entity``, ``label``, optional
        ``source``) and plain two-column CSVs (``label``, ``value``).

        Returns the number of new entities added.
        """
        before = len(self)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            # Handle the older (label, value) format used by from_csv()
            if "value" in fieldnames and "entity" not in fieldnames:
                entity_col = "value"
            for row in reader:
                entity = (row.get(entity_col) or "").strip()
                label = (row.get(label_col) or "").strip()
                if entity and label:
                    self.add(label, entity)
        added = len(self) - before
        LOG.info("[entities] Loaded %d new entities from CSV: %s", added, path)
        return added

    # ------------------------------------------------------------------
    # Radarr loader
    # ------------------------------------------------------------------

    def load_radarr(self, url: str, api_key: str) -> int:
        """Load movie titles and cast/crew from a Radarr instance."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        session.headers["X-Api-Key"] = api_key
        base = url.rstrip("/")

        before = len(self)
        LOG.info("[entities/radarr] Fetching movie library from %s …", base)
        movies = _http_get(session, f"{base}/api/v3/movie") or []
        LOG.info("[entities/radarr] Found %d movies", len(movies))

        for movie in movies:
            title = movie.get("title", "")
            if not title:
                continue
            genres = movie.get("genres") or []
            label = _video_label_from_genres(genres, "movie_title")
            self.add(label, title)
            for alt in movie.get("alternateTitles") or []:
                alt_title = alt.get("title", "")
                if alt_title and alt_title != title:
                    self.add(label, alt_title)
            credits = movie.get("credits") or {}
            for person in credits.get("castMembers") or []:
                name = person.get("name", "")
                if name:
                    self.add("movie_actor", name)
            for person in credits.get("crewMembers") or []:
                name = person.get("name", "")
                job = (person.get("job") or person.get("department") or "").lower()
                if not name:
                    continue
                if "direct" in job:
                    self.add("movie_director", name)
                elif "produc" in job:
                    self.add("movie_producer", name)
                elif "writ" in job or "screenplay" in job:
                    self.add("movie_writer", name)
                elif "compos" in job or "original score" in job:
                    self.add("movie_composer", name)
            studio = movie.get("studio", "")
            if studio:
                self.add("movie_streaming_service", studio)

        added = len(self) - before
        LOG.info("[entities/radarr] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # Sonarr loader
    # ------------------------------------------------------------------

    def load_sonarr(self, url: str, api_key: str) -> int:
        """Load TV show / anime titles from a Sonarr instance."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        session.headers["X-Api-Key"] = api_key
        base = url.rstrip("/")

        before = len(self)
        LOG.info("[entities/sonarr] Fetching series library from %s …", base)
        series = _http_get(session, f"{base}/api/v3/series") or []
        LOG.info("[entities/sonarr] Found %d series", len(series))

        for show in series:
            title = show.get("title", "")
            if not title:
                continue
            genres = show.get("genres") or []
            series_type = (show.get("seriesType") or "").lower()
            if series_type == "anime" or "anime" in {g.lower() for g in genres}:
                label = "anime_title"
            else:
                label = _video_label_from_genres(genres, "tv_show_title")
            self.add(label, title)
            for alt in show.get("alternateTitles") or []:
                alt_title = alt.get("title", "")
                if alt_title and alt_title != title:
                    self.add(label, alt_title)
            network = show.get("network", "")
            if network:
                self.add("tv_streaming_service", network)

        added = len(self) - before
        LOG.info("[entities/sonarr] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # Lidarr loader
    # ------------------------------------------------------------------

    def load_lidarr(self, url: str, api_key: str) -> int:
        """Load artist, album, track and genre entities from a Lidarr instance."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        session.headers["X-Api-Key"] = api_key
        base = url.rstrip("/")

        before = len(self)
        LOG.info("[entities/lidarr] Fetching artists from %s …", base)
        artists = _http_get(session, f"{base}/api/v1/artist") or []
        LOG.info("[entities/lidarr] Found %d artists", len(artists))

        seen_artists: Set[str] = set()
        for artist in artists:
            name = artist.get("artistName", "")
            if name:
                self.add("artist_name", name)
                seen_artists.add(name.lower())
            for genre in artist.get("genres") or []:
                if genre:
                    self.add("music_genre", genre)

        LOG.info("[entities/lidarr] Fetching albums …")
        albums = _http_get(session, f"{base}/api/v1/album") or []
        LOG.info("[entities/lidarr] Found %d albums", len(albums))

        for album in albums:
            title = album.get("title", "")
            if title:
                self.add("album_name", title)
            artist_name = (album.get("artist") or {}).get("artistName", "")
            if artist_name and artist_name.lower() not in seen_artists:
                self.add("artist_name", artist_name)
                seen_artists.add(artist_name.lower())
            for medium in album.get("media") or []:
                for track in medium.get("tracks") or []:
                    track_title = track.get("title", "")
                    if track_title:
                        self.add("track_name", track_title)

        added = len(self) - before
        LOG.info("[entities/lidarr] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # Jellyfin loader
    # ------------------------------------------------------------------

    def load_jellyfin(
        self,
        url: str,
        api_key: str,
        user_id: Optional[str] = None,
    ) -> int:
        """Load all media types from a Jellyfin instance."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        base = url.rstrip("/")

        # Resolve user ID
        if not user_id:
            users = (
                _http_get(session, f"{base}/Users", params={"api_key": api_key}) or []
            )
            user_id = users[0].get("Id", "") if users else ""
        if user_id:
            items_url = f"{base}/Users/{user_id}/Items"
        else:
            items_url = f"{base}/Items"

        # Item types → default OCP entity label
        _type_label = {
            "Movie": "movie_title",
            "Series": "tv_show_title",
            "MusicAlbum": "album_name",
            "MusicArtist": "artist_name",
            "Audio": "track_name",
            "Book": "audiobook_title",
            "AudioBook": "audiobook_title",
            "Podcast": "podcast_title",
        }

        before = len(self)
        for item_type, default_label in _type_label.items():
            LOG.info("[entities/jellyfin] Fetching %s items …", item_type)
            count = 0
            start = 0
            limit = 500
            while True:
                page = (
                    _http_get(
                        session,
                        items_url,
                        params={
                            "api_key": api_key,
                            "IncludeItemTypes": item_type,
                            "Recursive": "true",
                            "Fields": "Genres,Studios,People",
                            "StartIndex": start,
                            "Limit": limit,
                        },
                    )
                    or {}
                )
                items = page.get("Items") or []
                if not items:
                    break
                for item in items:
                    count += 1
                    name = item.get("Name", "")
                    if not name:
                        continue
                    genres = item.get("Genres") or []
                    if default_label == "movie_title":
                        label = _video_label_from_genres(genres, "movie_title")
                    elif default_label == "tv_show_title":
                        label = _video_label_from_genres(genres, "tv_show_title")
                    else:
                        label = default_label
                    self.add(label, name)

                    is_movie = label in (
                        "movie_title",
                        "anime_title",
                        "cartoon_title",
                        "documentary_title",
                    )
                    is_tv = label == "tv_show_title"
                    for person in item.get("People") or []:
                        pname = person.get("Name", "")
                        prole = (person.get("Type") or "").lower()
                        if not pname:
                            continue
                        if is_movie or is_tv:
                            if prole == "actor":
                                self.add("movie_actor", pname)
                            elif prole == "director":
                                self.add("movie_director", pname)
                            elif prole == "producer":
                                self.add("movie_producer", pname)
                            elif prole == "writer":
                                self.add("movie_writer", pname)
                            elif prole in ("composer", "music"):
                                self.add("movie_composer", pname)
                        elif label in ("audiobook_title",):
                            if prole in ("author", "writer"):
                                self.add("audiobook_author", pname)

                    for studio in item.get("Studios") or []:
                        sname = studio.get("Name", "")
                        if not sname:
                            continue
                        if is_movie:
                            self.add("movie_streaming_service", sname)
                        elif is_tv:
                            self.add("tv_streaming_service", sname)

                    if item_type == "Audio":
                        for artist in item.get("Artists") or []:
                            if artist:
                                self.add("artist_name", artist)
                        album = item.get("Album", "")
                        if album:
                            self.add("album_name", album)

                total = page.get("TotalRecordCount", 0)
                start += limit
                if start >= total:
                    break
            LOG.info("[entities/jellyfin] %s: processed %d items", item_type, count)

        added = len(self) - before
        LOG.info("[entities/jellyfin] Added %d new entities total", added)
        return added

    # ------------------------------------------------------------------
    # Whisparr loader (Radarr fork for adult content)
    # ------------------------------------------------------------------

    def load_whisparr(self, url: str, api_key: str) -> int:
        """Load adult movie titles and performers from a Whisparr instance.

        Whisparr uses the same API shape as Radarr v3.  Titles are stored
        under ``"adult"`` and performer names under ``"movie_actor"``.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        session.headers["X-Api-Key"] = api_key
        base = url.rstrip("/")

        before = len(self)
        LOG.info("[entities/whisparr] Fetching movie library from %s …", base)
        movies = _http_get(session, f"{base}/api/v3/movie") or []
        LOG.info("[entities/whisparr] Found %d titles", len(movies))

        for movie in movies:
            title = movie.get("title", "")
            if title:
                self.add("adult", title)
            for alt in movie.get("alternateTitles") or []:
                alt_title = alt.get("title", "")
                if alt_title and alt_title != title:
                    self.add("adult", alt_title)
            credits = movie.get("credits") or {}
            for person in credits.get("castMembers") or []:
                name = person.get("name", "")
                if name:
                    self.add("movie_actor", name)
            studio = movie.get("studio", "")
            if studio:
                self.add("adult_streaming_service", studio)

        added = len(self) - before
        LOG.info("[entities/whisparr] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # Stash loader (GraphQL — adult content manager)
    # ------------------------------------------------------------------

    def load_stash(
        self,
        url: str,
        api_key: Optional[str] = None,
        page_size: int = 500,
    ) -> int:
        """Load scenes, performers, studios and tags from a Stash instance.

        Stash exposes a GraphQL API at ``<url>/graphql``.  Performer names
        are stored as ``"movie_actor"``, studio names as
        ``"adult_streaming_service"``, scene titles as ``"adult"``, and tags
        as ``"adult"``.

        Args:
            url:       Base URL of the Stash server (e.g. ``http://localhost:9999``).
            api_key:   Optional API key (set in Stash → Settings → Security).
            page_size: Number of records to fetch per request.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        if api_key:
            session.headers["ApiKey"] = api_key
        gql_url = url.rstrip("/") + "/graphql"

        def _gql(query: str, variables: dict) -> dict:
            try:
                resp = session.post(
                    gql_url, json={"query": query, "variables": variables}, timeout=30
                )
                resp.raise_for_status()
                return resp.json().get("data") or {}
            except Exception as exc:
                LOG.warning("[entities/stash] GraphQL request failed: %s", exc)
                return {}

        before = len(self)

        # --- Performers ---
        LOG.info("[entities/stash] Fetching performers from %s …", url)
        page = 1
        while True:
            data = _gql(
                """
                query FindPerformers($filter: FindFilterType) {
                  findPerformers(filter: $filter) {
                    count
                    performers { name aliases }
                  }
                }
            """,
                {"filter": {"page": page, "per_page": page_size}},
            )
            performers = data.get("findPerformers") or {}
            items = performers.get("performers") or []
            if not items:
                break
            for p in items:
                name = p.get("name", "")
                if name:
                    self.add("movie_actor", name)
                for alias in p.get("aliases") or []:
                    if alias and alias != name:
                        self.add("movie_actor", alias)
            total = performers.get("count", 0)
            if page * page_size >= total:
                break
            page += 1

        # --- Studios ---
        LOG.info("[entities/stash] Fetching studios …")
        page = 1
        while True:
            data = _gql(
                """
                query FindStudios($filter: FindFilterType) {
                  findStudios(filter: $filter) {
                    count
                    studios { name aliases }
                  }
                }
            """,
                {"filter": {"page": page, "per_page": page_size}},
            )
            studios_data = data.get("findStudios") or {}
            items = studios_data.get("studios") or []
            if not items:
                break
            for s in items:
                name = s.get("name", "")
                if name:
                    self.add("adult_streaming_service", name)
                for alias in s.get("aliases") or []:
                    if alias and alias != name:
                        self.add("adult_streaming_service", alias)
            total = studios_data.get("count", 0)
            if page * page_size >= total:
                break
            page += 1

        # --- Scenes (titles only — skip stream URLs) ---
        LOG.info("[entities/stash] Fetching scene titles …")
        page = 1
        while True:
            data = _gql(
                """
                query FindScenes($filter: FindFilterType) {
                  findScenes(filter: $filter) {
                    count
                    scenes { title }
                  }
                }
            """,
                {"filter": {"page": page, "per_page": page_size}},
            )
            scenes_data = data.get("findScenes") or {}
            items = scenes_data.get("scenes") or []
            if not items:
                break
            for s in items:
                title = s.get("title", "")
                if title:
                    self.add("adult", title)
            total = scenes_data.get("count", 0)
            if page * page_size >= total:
                break
            page += 1

        added = len(self) - before
        LOG.info("[entities/stash] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # Music Assistant loader
    # ------------------------------------------------------------------

    def load_music_assistant(self, url: str) -> int:
        """Load artists, albums, tracks and radio stations from a Music Assistant instance.

        Music Assistant exposes a REST API under ``<url>/api``.  Endpoints
        used: ``/api/music/artists``, ``/api/music/albums``,
        ``/api/music/tracks``, ``/api/music/radio``.

        Args:
            url: Base URL of Music Assistant (e.g. ``http://localhost:8095``).
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for media-server loaders. "
                "Install it with: pip install ovos-media-classifier[media_servers]"
            )
        session = requests.Session()
        base = url.rstrip("/")

        before = len(self)
        LOG.info("[entities/music_assistant] Fetching library from %s …", base)

        # Helper: paginated list endpoint
        def _fetch_all(path: str) -> list:
            results = []
            offset = 0
            limit = 500
            while True:
                data = (
                    _http_get(
                        session,
                        f"{base}{path}",
                        params={"limit": limit, "offset": offset},
                    )
                    or []
                )
                # MA may return a list directly or {"items": [...], "total": N}
                if isinstance(data, dict):
                    items = data.get("items") or data.get("results") or []
                    total = data.get("total", len(items))
                else:
                    items = data
                    total = len(items)
                results.extend(items)
                offset += limit
                if offset >= total or not items:
                    break
            return results

        # Artists
        for artist in _fetch_all("/api/music/artists"):
            name = artist.get("name", "")
            if name:
                self.add("artist_name", name)
            for genre in artist.get("metadata", {}).get("genres") or []:
                if genre:
                    self.add("music_genre", genre)

        # Albums
        for album in _fetch_all("/api/music/albums"):
            title = album.get("name", "")
            if title:
                self.add("album_name", title)
            # Artist(s) embedded in album object
            for artist in album.get("artists") or []:
                name = artist.get("name", "")
                if name:
                    self.add("artist_name", name)

        # Tracks
        for track in _fetch_all("/api/music/tracks"):
            title = track.get("name", "")
            if title:
                self.add("track_name", title)
            for artist in track.get("artists") or []:
                name = artist.get("name", "")
                if name:
                    self.add("artist_name", name)

        # Radio stations
        for station in _fetch_all("/api/music/radio"):
            name = station.get("name", "")
            if name:
                self.add("radio_station", name)

        added = len(self) - before
        LOG.info("[entities/music_assistant] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # HuggingFace datasets loader
    # ------------------------------------------------------------------

    def load_huggingface(
        self,
        dataset_name: str,
        config: Optional[str] = None,
        split: str = "train",
        entity_col: str = "entity",
        label_col: str = "label",
        trust_remote_code: bool = False,
    ) -> int:
        """Load entities from a HuggingFace dataset.

        The dataset must have at minimum an entity column and a label column
        whose values are valid ``OCPEntityLabel`` strings.

        Args:
            dataset_name: HuggingFace dataset repo name (e.g.
                ``"TigreGotico/ocp-entities"``).
            config:       Dataset configuration name (subset), if any.
            split:        Dataset split to load (default ``"train"``).
            entity_col:   Column name containing entity strings.
            label_col:    Column name containing OCP entity label strings.
            trust_remote_code: Passed through to ``datasets.load_dataset``.

        Returns the number of new entities added.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets is required for the HuggingFace loader. "
                "Install it with: pip install ovos-media-classifier[huggingface]"
            )
        LOG.info(
            "[entities/huggingface] Loading dataset: %s (split=%s) …",
            dataset_name,
            split,
        )
        ds = load_dataset(
            dataset_name,
            config,
            split=split,
            trust_remote_code=trust_remote_code,
        )
        before = len(self)
        for row in ds:
            entity = (row.get(entity_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if entity and label:
                self.add(label, entity)
        added = len(self) - before
        LOG.info("[entities/huggingface] Added %d new entities", added)
        return added

    # ------------------------------------------------------------------
    # Config-driven factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "EntitiesContainer":
        """Build an ``EntitiesContainer`` from a config sub-dict.

        Expected keys (all optional)::

            {
                "csv": ["/path/to/entities.csv"],

                "jellyfin": {"url": "http://localhost:8096", "api_key": "…"},
                "radarr":   {"url": "http://localhost:7878", "api_key": "…"},
                "sonarr":   {"url": "http://localhost:8989", "api_key": "…"},
                "lidarr":   {"url": "http://localhost:8686", "api_key": "…"},

                "huggingface": [
                    {
                        "dataset":    "TigreGotico/ocp-entities",
                        "config":     null,
                        "split":      "train",
                        "entity_col": "entity",
                        "label_col":  "label"
                    }
                ],

                "wordlists": {
                    "artist_name": ["Radiohead", "Pink Floyd"],
                    "movie_title": ["The Matrix"]
                }
            }
        """
        container = cls()

        # Inline word lists
        for label, words in (config.get("wordlists") or {}).items():
            for word in words:
                container.add(label, word)

        # CSV files
        for path in config.get("csv") or []:
            try:
                container.load_csv(path)
            except Exception as exc:
                LOG.warning("[entities] CSV load failed (%s): %s", path, exc)

        # HuggingFace datasets
        for hf in config.get("huggingface") or []:
            try:
                container.load_huggingface(
                    dataset_name=hf["dataset"],
                    config=hf.get("config"),
                    split=hf.get("split", "train"),
                    entity_col=hf.get("entity_col", "entity"),
                    label_col=hf.get("label_col", "label"),
                    trust_remote_code=hf.get("trust_remote_code", False),
                )
            except Exception as exc:
                LOG.warning(
                    "[entities] HuggingFace load failed (%s): %s",
                    hf.get("dataset"),
                    exc,
                )

        # Media servers — all optional; api_key-based (url + api_key required)
        for server, loader in (
            ("radarr", container.load_radarr),
            ("sonarr", container.load_sonarr),
            ("lidarr", container.load_lidarr),
            ("whisparr", container.load_whisparr),
        ):
            srv_cfg = config.get(server) or {}
            if srv_cfg.get("url") and srv_cfg.get("api_key"):
                try:
                    loader(srv_cfg["url"], srv_cfg["api_key"])
                except Exception as exc:
                    LOG.warning("[entities] %s load failed: %s", server, exc)

        jf_cfg = config.get("jellyfin") or {}
        if jf_cfg.get("url") and jf_cfg.get("api_key"):
            try:
                container.load_jellyfin(
                    jf_cfg["url"],
                    jf_cfg["api_key"],
                    user_id=jf_cfg.get("user_id"),
                )
            except Exception as exc:
                LOG.warning("[entities] jellyfin load failed: %s", exc)

        # Stash (GraphQL, api_key optional)
        stash_cfg = config.get("stash") or {}
        if stash_cfg.get("url"):
            try:
                container.load_stash(
                    stash_cfg["url"],
                    api_key=stash_cfg.get("api_key"),
                    page_size=stash_cfg.get("page_size", 500),
                )
            except Exception as exc:
                LOG.warning("[entities] stash load failed: %s", exc)

        # Music Assistant (no api_key required)
        ma_cfg = config.get("music_assistant") or {}
        if ma_cfg.get("url"):
            try:
                container.load_music_assistant(ma_cfg["url"])
            except Exception as exc:
                LOG.warning("[entities] music_assistant load failed: %s", exc)

        LOG.info("[entities] Container ready: %r", container)
        return container
