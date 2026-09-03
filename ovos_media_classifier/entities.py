"""Entity lists — the source-agnostic store that powers OCP entity matching.

An :class:`EntitiesContainer` is, fundamentally, a set of **entity lists**: a
mapping of ``label → list of strings`` (``artist_name → ["Radiohead", …]``,
``movie_title → ["Inception", …]``, …).  It is shared infrastructure, not tied
to any one classifier:

- the **NER backend** (:class:`~ovos_media_classifier.ahocorasick.AhocorasickMediaClassifier`)
  feeds the lists into an Aho-Corasick automaton for fast exact substring
  matching;
- the (future) **guided-embeddings** classifier consumes the *same* lists as
  categorical NER features.

Build the same lists once, use them with either strategy.

Where entity lists come from
----------------------------
1. **Provided at runtime** — the ``add`` / ``add_many`` path.  The OCP pipeline
   registers the user's media as it discovers it (a skill announcing its
   content, a background media-server sync); each call is reflected in
   classification results immediately, with no rebuild step.
2. **Provided via config as source specs** — dispatched by :meth:`load_source`:

   - a **file path** ending in ``.csv`` / ``.tsv`` / ``.jsonl``
     (:meth:`load_csv` / :meth:`load_tsv` / :meth:`load_jsonl`);
   - a **HuggingFace dataset** reference — a dict with a ``"dataset"`` key
     (:meth:`load_huggingface`);
   - an **inline** ``{label: [values]}`` dict (added directly);
   - a **media-server** dict (``jellyfin`` / ``radarr`` / ``sonarr`` /
     ``lidarr`` / ``music_assistant``), keyed by a ``"type"`` hint.

Performance / memory tradeoff
-----------------------------
Entity lists are a **deliberate, bounded choice**.  The Aho-Corasick automaton
holds every entity string in memory and every added entity widens the matcher:
the more entities loaded, the slower the per-utterance tagging and the larger
the memory footprint.  Load the user's *actual* library (a few thousand titles),
not an open-ended public catalogue.  The same caution applies to the
guided-embeddings features — a bloated categorical vocabulary dilutes the
signal.  Prefer a handful of focused lists over one giant dump.

Dependencies
------------
- ``ahocorasick-ner``  — required to materialise the NER (``pip install ovos-media-classifier[ner]``)
- ``requests``         — required for media-server loaders  (``pip install ovos-media-classifier[media_servers]``)
- ``datasets``         — required for HuggingFace loader   (``pip install ovos-media-classifier[huggingface]``)

All three are optional.  Loading entity lists from files / inline dicts needs
**none** of them — list loading is independent of the matcher.  The container is
a pure data store until its :attr:`ner` is materialised (lazily, on first
access).

Runtime awareness
-----------------
The same ``AhocorasickNER`` instance is shared between the container and
``AhocorasickMediaClassifier``.  Every call to :meth:`add` propagates to the
NER via ``ner.add_word()`` immediately, so newly registered entities are
reflected in classification results with no rebuild step.

Example::

    # Build entity lists from mixed sources (no optional deps needed for files):
    container = EntitiesContainer.from_sources([
        "/data/my_library.csv",                 # .csv path
        "/data/extra.tsv",                       # .tsv path
        "/data/aliases.jsonl",                   # .jsonl path
        {"artist_name": ["Radiohead", "Bjork"]}, # inline {label: [values]}
        {"dataset": "TigreGotico/ocp-entities"}, # HuggingFace dataset
    ])

    clf = AhocorasickMediaClassifier.from_container(container)
    clf.classify("play the dark knight", "en-us")
    # → (MediaType.MOVIE, 0.6)

    # New entity registered at runtime (e.g. from a skill announcement):
    container.add("artist_name", "Aphex Twin")
    clf.classify("play aphex twin", "en-us")
    # → (MediaType.MUSIC, 0.6)  — no rebuild needed
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

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
    """A set of **entity lists** (``label → list of strings``).

    This is the source-agnostic store behind OCP entity matching.  It holds
    entity strings grouped by :class:`~ovos_media_classifier.intents.OCPEntityLabel`
    value strings and optionally backs an ``AhocorasickNER`` instance for fast
    substring matching.  The *same* lists are shared by the NER backend
    (Aho-Corasick exact match) and the future guided-embeddings classifier (as
    categorical NER features) — the entity-list machinery is shared
    infrastructure, not NER-specific.

    Entity lists arrive two ways:

    - **at runtime**, via :meth:`add` / :meth:`add_many` (the OCP pipeline
      registering the user's media as it discovers it);
    - **from config**, via source specs dispatched by :meth:`load_source` —
      ``.csv`` / ``.tsv`` / ``.jsonl`` file paths, HuggingFace dataset dicts,
      inline ``{label: [values]}`` dicts, or media-server dicts.  See
      :meth:`from_sources` / :meth:`load_lists` / :meth:`from_config`.

    Performance note: every entity widens the Aho-Corasick automaton and grows
    the memory footprint, so larger lists mean slower, hungrier matching.
    Entity lists are a deliberate, bounded choice — load the user's real
    library, not an open-ended catalogue.

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
        except Exception as exc:
            LOG.warning("EntitiesContainer: could not back-fill from NER (%s); non-fatal", exc)

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
    # Delimited-file loaders (CSV / TSV) — one entity list per file
    # ------------------------------------------------------------------

    def _load_delimited(
        self,
        path: str,
        delimiter: str,
        entity_col: str,
        label_col: str,
        fmt: str,
    ) -> int:
        before = len(self)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
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
        LOG.info("[entities] Loaded %d new entities from %s: %s", added, fmt, path)
        return added

    def load_csv(
        self,
        path: str,
        entity_col: str = "entity",
        label_col: str = "label",
    ) -> int:
        """Load an entity list from a CSV file (one ``label,entity`` per row).

        Accepts both named-column CSVs (``entity``, ``label``, optional
        ``source``) and plain two-column CSVs (``label``, ``value``).

        Returns the number of new entities added.
        """
        return self._load_delimited(path, ",", entity_col, label_col, "CSV")

    def load_tsv(
        self,
        path: str,
        entity_col: str = "entity",
        label_col: str = "label",
    ) -> int:
        """Load an entity list from a tab-separated (``.tsv``) file.

        Identical to :meth:`load_csv` but tab-delimited — useful when entity
        strings themselves contain commas.  Accepts the named-column
        (``entity``/``label``) and the two-column (``label``/``value``) shapes.

        Returns the number of new entities added.
        """
        return self._load_delimited(path, "\t", entity_col, label_col, "TSV")

    # ------------------------------------------------------------------
    # JSON Lines loader — one entity list per file
    # ------------------------------------------------------------------

    def load_jsonl(
        self,
        path: str,
        entity_col: str = "entity",
        label_col: str = "label",
    ) -> int:
        """Load an entity list from a JSON Lines (``.jsonl``) file.

        One JSON object per line.  Two object shapes are accepted (and may be
        mixed within the same file):

        - **per-entity rows** — ``{"label": "movie_title", "entity": "Inception"}``
          (the column names are configurable via *entity_col* / *label_col*;
          ``"value"`` is also accepted as an entity key for convenience);
        - **list rows** — ``{"artist_name": ["Radiohead", "Bjork"]}`` — a dict
          mapping one or more labels to a list (or single string) of values.
          Rows carrying the reserved ``label`` / ``entity`` keys are treated as
          per-entity rows, never as list rows.

        Blank lines are ignored.  Returns the number of new entities added.
        """
        before = len(self)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOG.warning("[entities] skipping malformed JSONL line in %s: %s", path, exc)
                    continue
                if not isinstance(obj, dict):
                    LOG.warning("[entities] skipping non-object JSONL line in %s", path)
                    continue
                if label_col in obj or entity_col in obj or "value" in obj:
                    # per-entity row
                    entity = obj.get(entity_col)
                    if entity is None:
                        entity = obj.get("value")
                    label = obj.get(label_col)
                    if entity and label:
                        self.add(str(label).strip(), str(entity).strip())
                else:
                    # list row: {label: [values]} (or {label: "value"})
                    self._add_inline(obj)
        added = len(self) - before
        LOG.info("[entities] Loaded %d new entities from JSONL: %s", added, path)
        return added

    # ------------------------------------------------------------------
    # Inline entity lists + source dispatch
    # ------------------------------------------------------------------

    def _add_inline(self, mapping: Dict[str, Any]) -> int:
        """Add an inline ``{label: [values]}`` (or ``{label: "value"}``) dict."""
        before = len(self)
        for label, values in mapping.items():
            if isinstance(values, str):
                values = [values]
            for value in values or []:
                if value:
                    self.add(str(label).strip(), str(value).strip())
        return len(self) - before

    # Keys that mark a dict source spec as a media-server loader.
    _MEDIA_SERVER_LOADERS = (
        "radarr",
        "sonarr",
        "lidarr",
        "jellyfin",
        "music_assistant",
    )

    def load_source(self, spec: Union[str, Dict[str, Any]]) -> int:
        """Load a single **entity-list source spec**, dispatching by shape.

        Dispatch rules:

        - ``str`` path → by file extension: ``.csv`` → :meth:`load_csv`,
          ``.tsv`` → :meth:`load_tsv`, ``.jsonl`` → :meth:`load_jsonl`;
        - ``dict`` with a ``"dataset"`` key → :meth:`load_huggingface`;
        - ``dict`` with a media-server ``"type"`` hint (or a single known
          media-server key, e.g. ``{"radarr": {...}}``) → the matching
          ``load_<server>`` loader;
        - any other ``dict`` → treated as an inline ``{label: [values]}`` list.

        Returns the number of new entities added.  Unknown extensions / specs
        raise ``ValueError``.
        """
        if isinstance(spec, str):
            ext = os.path.splitext(spec)[1].lower()
            if ext == ".csv":
                return self.load_csv(spec)
            if ext == ".tsv":
                return self.load_tsv(spec)
            if ext == ".jsonl":
                return self.load_jsonl(spec)
            raise ValueError(
                f"Unsupported entity-list file extension {ext!r} for {spec!r}; "
                "expected .csv, .tsv or .jsonl"
            )

        if isinstance(spec, dict):
            # HuggingFace dataset reference
            if "dataset" in spec:
                return self.load_huggingface(
                    dataset_name=spec["dataset"],
                    config=spec.get("config"),
                    split=spec.get("split", "train"),
                    entity_col=spec.get("entity_col", "entity"),
                    label_col=spec.get("label_col", "label"),
                    trust_remote_code=spec.get("trust_remote_code", False),
                )

            # Media-server spec via explicit {"type": "radarr", ...}
            srv_type = (spec.get("type") or "").lower()
            if srv_type in self._MEDIA_SERVER_LOADERS:
                return self._load_media_server(srv_type, spec)

            # Media-server spec via single known key: {"radarr": {...}}
            if len(spec) == 1:
                (only_key,) = spec.keys()
                if only_key in self._MEDIA_SERVER_LOADERS:
                    cfg = spec[only_key]
                    if isinstance(cfg, dict):
                        return self._load_media_server(only_key, cfg)

            # Otherwise: inline {label: [values]}
            return self._add_inline(spec)

        raise ValueError(f"Unsupported entity-list source spec: {spec!r}")

    def _load_media_server(self, server: str, cfg: Dict[str, Any]) -> int:
        """Invoke a media-server loader from a ``{"url": …, "api_key": …}`` cfg."""
        before = len(self)
        if server == "radarr":
            self.load_radarr(cfg["url"], cfg["api_key"])
        elif server == "sonarr":
            self.load_sonarr(cfg["url"], cfg["api_key"])
        elif server == "lidarr":
            self.load_lidarr(cfg["url"], cfg["api_key"])
        elif server == "jellyfin":
            self.load_jellyfin(cfg["url"], cfg["api_key"], user_id=cfg.get("user_id"))
        elif server == "music_assistant":
            self.load_music_assistant(cfg["url"])
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unknown media server: {server!r}")
        return len(self) - before

    def load_lists(self, specs: Iterable[Union[str, Dict[str, Any]]]) -> int:
        """Load a list of entity-list source specs (each via :meth:`load_source`).

        Individual spec failures are logged and skipped so one bad path / dead
        media server doesn't abort the rest.  Returns the total number of new
        entities added across all specs.
        """
        before = len(self)
        for spec in specs or []:
            try:
                self.load_source(spec)
            except Exception as exc:
                LOG.warning("[entities] entity-list source failed (%r): %s", spec, exc)
        return len(self) - before

    @classmethod
    def from_sources(
        cls,
        specs: Iterable[Union[str, Dict[str, Any]]],
        ner=None,
    ) -> "EntitiesContainer":
        """Build a container from a list of entity-list source specs.

        Each spec is dispatched by :meth:`load_source` — file paths
        (``.csv`` / ``.tsv`` / ``.jsonl``), HuggingFace dataset dicts, inline
        ``{label: [values]}`` dicts, or media-server dicts.  See
        :meth:`load_source` for the dispatch rules.

        Example::

            container = EntitiesContainer.from_sources([
                "/data/library.csv",
                {"artist_name": ["Radiohead"]},
                {"dataset": "TigreGotico/ocp-entities"},
            ])
        """
        container = cls(ner=ner)
        container.load_lists(specs)
        return container

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

        The preferred, source-agnostic form is a single ``entity_lists`` key
        holding a **list of source specs** dispatched via :meth:`load_source`
        (file paths, HuggingFace dicts, inline ``{label: [values]}`` dicts,
        media-server dicts)::

            {
                "entity_lists": [
                    "/path/to/entities.csv",
                    "/path/to/extra.tsv",
                    "/path/to/aliases.jsonl",
                    {"artist_name": ["Radiohead", "Pink Floyd"]},
                    {"dataset": "TigreGotico/ocp-entities"},
                    {"radarr": {"url": "http://localhost:7878", "api_key": "…"}}
                ]
            }

        The original structured keys remain supported (and are processed *after*
        ``entity_lists``) for back-compatibility::

            {
                "csv": ["/path/to/entities.csv"],

                "jellyfin": {"url": "http://localhost:8096", "api_key": "…"},
                "radarr":   {"url": "http://localhost:7878", "api_key": "…"},
                "sonarr":   {"url": "http://localhost:8989", "api_key": "…"},
                "lidarr":   {"url": "http://localhost:8686", "api_key": "…"},
                "music_assistant": {"url": "http://localhost:8095"},

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

        All keys are optional and additive — entities from every source are
        merged (and de-duplicated) into one set of entity lists.
        """
        container = cls()

        # Source-agnostic entity lists (preferred form)
        container.load_lists(config.get("entity_lists") or [])

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

        # Music Assistant (no api_key required)
        ma_cfg = config.get("music_assistant") or {}
        if ma_cfg.get("url"):
            try:
                container.load_music_assistant(ma_cfg["url"])
            except Exception as exc:
                LOG.warning("[entities] music_assistant load failed: %s", exc)

        LOG.info("[entities] Container ready: %r", container)
        return container
