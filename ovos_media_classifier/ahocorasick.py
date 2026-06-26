"""AhocorasickNER-based media classifier.

Uses exact substring matching via a pre-populated AhocorasickNER instance.
This is the same NER engine that the OCP pipeline plugin uses for entity
extraction (streaming service names, artist/album keywords, etc.).

Unlike the keyword classifier (which uses locale voc files + fuzzy match),
this backend performs fast, language-agnostic substring matching over large
custom dictionaries — ideal for:

  - Matching known service/artist/genre names registered by OCP skills
  - Classifying utterances that mention explicit media keywords (e.g. "jazz",
    "BBC Radio", "Netflix")
  - Supplementing the keyword classifier with runtime-registered vocabulary

Dependency: ``ahocorasick-ner`` (optional)

    pip install ovos-media-classifier[ner]

If the package is not available, instantiation raises ``ImportError``.

Usage::

    from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

    clf = AhocorasickMediaClassifier.from_wordlists({
        "music": ["jazz", "blues", "rock", "spotify", "tidal"],
        "movie": ["netflix", "film", "cinema", "hollywood"],
        "podcast": ["podcast", "episode", "series"],
    })
    media_type, conf = clf.classify("play some jazz", "en-us")
    # → (MediaType.MUSIC, 0.6)

    # Or share a NER instance with the pipeline:
    from ahocorasick_ner import AhocorasickNER
    ner = AhocorasickNER()
    ner.add_word("music_streaming_service", "Spotify")
    clf = AhocorasickMediaClassifier(ner)

    # Or use an EntitiesContainer (recommended — enables runtime updates):
    from ovos_media_classifier.entities import EntitiesContainer
    container = EntitiesContainer()
    container.load_radarr("http://localhost:7878", api_key="…")
    container.load_lidarr("http://localhost:8686", api_key="…")
    clf = AhocorasickMediaClassifier.from_container(container)

    # New entity added at runtime → immediately reflected in classify():
    container.add("artist_name", "Radiohead")
"""
import csv
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import (
    MediaType,
    OCPDomain,
    NER_LABEL_TO_MEDIA_TYPE,
    NER_LABEL_TO_GENRES,
)

if TYPE_CHECKING:
    from ovos_media_classifier.entities import EntitiesContainer

from ovos_media_classifier.constants import DEFAULT_NER_HIT_CONFIDENCE

# Priority order for resolving multiple entity hits in one utterance.
# More specific / higher-confidence types rank first (lower index wins).
# Distinctions that are *not* media types (documentary, anime, adult, …) are
# carried as genre tags via ``NER_LABEL_TO_GENRES``; the type priority below
# reflects the representative ordering (e.g. a music hit beats a movie hit).
_MEDIA_TYPE_PRIORITY: List[MediaType] = [
    MediaType.AUDIOBOOK,
    MediaType.RADIO,
    MediaType.PODCAST,
    MediaType.AUDIO_DRAMA,
    MediaType.TV,
    MediaType.MUSIC,
    MediaType.MUSIC_VIDEO,
    MediaType.EPISODIC_SERIES,
    MediaType.SHORT_FILM,
    MediaType.MOVIE,
    MediaType.COMIC,
    MediaType.GAME,
    MediaType.PROCEDURAL_AMBIENT,
    MediaType.GENERIC,
]

_MEDIA_TYPE_RANK: Dict[MediaType, int] = {
    mt: rank for rank, mt in enumerate(_MEDIA_TYPE_PRIORITY)
}

# Confidence returned when an entity hit yields a match
_HIT_CONFIDENCE = DEFAULT_NER_HIT_CONFIDENCE


class AhocorasickMediaClassifier(AbstractMediaClassifier):
    """Media classifier backed by an AhocorasickNER instance.

    The NER labels are mapped to ``mediavocab.MediaType`` values via
    ``NER_LABEL_TO_MEDIA_TYPE`` (and to genre tags via ``NER_LABEL_TO_GENRES``).
    Custom label names are supported by passing a *label_map* override.

    When constructed from an :class:`~ovos_media_classifier.entities.EntitiesContainer`
    (via :meth:`from_container`) the classifier is *runtime-aware*: any
    subsequent call to ``container.add(label, entity)`` — from skills
    announcing their content, from background media-server syncs, etc. —
    is immediately reflected in :meth:`classify` results because the NER
    is shared by reference between the container and this classifier.

    Args:
        ner_or_container: Either an ``AhocorasickNER`` instance or an
            :class:`~ovos_media_classifier.entities.EntitiesContainer`.
            When a container is passed its ``ner`` property is used and
            the container is retained so callers can continue registering
            entities after construction.
        label_map: Override mapping from NER label strings to
                   ``mediavocab.MediaType``.  Merged on top of the default
                   ``NER_LABEL_TO_MEDIA_TYPE``.
    """

    def __init__(
        self,
        ner_or_container,
        label_map: Optional[Dict[str, MediaType]] = None,
    ) -> None:
        try:
            from ahocorasick_ner import AhocorasickNER  # noqa: F401
        except ImportError:
            raise ImportError(
                "ahocorasick-ner is required for AhocorasickMediaClassifier. "
                "Install it with: pip install ovos-media-classifier[ner]"
            )
        from ovos_media_classifier.entities import EntitiesContainer
        if isinstance(ner_or_container, EntitiesContainer):
            self._container: Optional[EntitiesContainer] = ner_or_container
            self._ner = ner_or_container.ner  # shared reference
        else:
            self._container = None
            self._ner = ner_or_container
        self._label_map: Dict[str, MediaType] = {**NER_LABEL_TO_MEDIA_TYPE}
        if label_map:
            self._label_map.update(label_map)

    @property
    def container(self) -> "Optional[EntitiesContainer]":
        """The :class:`~ovos_media_classifier.entities.EntitiesContainer` backing
        this classifier, or ``None`` when constructed from a raw NER."""
        return self._container

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_container(
        cls,
        container: "EntitiesContainer",
        label_map: Optional[Dict[str, MediaType]] = None,
    ) -> "AhocorasickMediaClassifier":
        """Build a runtime-aware classifier from an :class:`~ovos_media_classifier.entities.EntitiesContainer`.

        The container's NER is shared by reference — any entity added to the
        container after this call is immediately visible to :meth:`classify`.

        Example::

            container = EntitiesContainer()
            container.load_radarr("http://localhost:7878", api_key="…")
            clf = AhocorasickMediaClassifier.from_container(container)

            container.add("artist_name", "Radiohead")   # live update
            clf.classify("play radiohead", "en-us")
            # → (MediaType.MUSIC, 0.6)
        """
        return cls(container, label_map=label_map)

    @classmethod
    def from_wordlists(
        cls,
        wordlists: Dict[str, List[str]],
        label_map: Optional[Dict[str, MediaType]] = None,
    ) -> "AhocorasickMediaClassifier":
        """Build a classifier from a dict mapping label → list of keywords.

        Keys may be either raw media labels ("music", "movie", …) or any of the
        pipeline NER label names ("music_streaming_service", …).

        Example::

            clf = AhocorasickMediaClassifier.from_wordlists({
                "music": ["jazz", "blues", "rock"],
                "music_streaming_service": ["Spotify", "Tidal", "Deezer"],
                "movie": ["cinema", "film"],
            })
        """
        from ovos_media_classifier.entities import EntitiesContainer
        container = EntitiesContainer()
        for label, words in wordlists.items():
            for word in words:
                container.add(label, word)
        return cls.from_container(container, label_map=label_map)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        label_col: int = 0,
        value_col: int = 1,
        skip_header: bool = True,
        label_map: Optional[Dict[str, MediaType]] = None,
    ) -> "AhocorasickMediaClassifier":
        """Build from a CSV file with columns (label, keyword).

        This is the same format used by the pipeline's skill keyword CSV API::

            label,value
            music_streaming_service,Spotify
            music_streaming_service,Tidal
            movie_streaming_service,Netflix

        Also accepts the three-column format from ``generate_dataset_from_media.py``::

            entity,label,source
            The Dark Knight,movie_title,radarr
        """
        from ovos_media_classifier.entities import EntitiesContainer
        container = EntitiesContainer()
        container.load_csv(csv_path)
        return cls.from_container(container, label_map=label_map)

    # ------------------------------------------------------------------
    # Runtime word registration (mirrors the pipeline API)
    # ------------------------------------------------------------------

    def add_word(self, label: str, word: str) -> None:
        """Register a single keyword under *label* at runtime.

        When backed by an :class:`~ovos_media_classifier.entities.EntitiesContainer`
        the entity is also tracked there (dedup, stats).  Otherwise it is
        added directly to the underlying NER.
        """
        if self._container is not None:
            self._container.add(label, word)
        else:
            self._ner.add_word(label, word)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _entities_to_media_type(
        self, entities: Dict[str, str]
    ) -> Optional[MediaType]:
        """Pick the highest-priority MediaType from a set of NER entity labels."""
        hits: List[MediaType] = []
        for label in entities:
            media_type = self._label_map.get(label)
            if media_type is not None:
                hits.append(media_type)
        if not hits:
            return None
        # Return the highest-priority hit (lowest rank index)
        return min(hits, key=lambda mt: _MEDIA_TYPE_RANK.get(mt, 999))

    def _winning_labels(self, entities: Dict[str, str], media_type: MediaType) -> List[str]:
        """The entity labels that map to the winning MediaType (for genre lookup)."""
        return [label for label in entities
                if self._label_map.get(label) == media_type]

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        try:
            raw = self._ner.tag(query)
            entities = {e["label"]: e["word"] for e in raw}
        except Exception as e:
            LOG.error(f"AhocorasickMediaClassifier NER failed: {e}")
            return MediaType.GENERIC, 0.0

        media_type = self._entities_to_media_type(entities)
        if media_type is None:
            return MediaType.GENERIC, 0.0

        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0

        return media_type, _HIT_CONFIDENCE

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Genre tags implied by the matched NER entities (e.g. adult/anime/asmr).

        Without this override the content filter would have no genre signal from
        the NER backend and could not block adult entities (pornstar, etc.).
        Genres are gathered from every label that maps to the winning MediaType,
        so an ``adult_title`` (MOVIE) still surfaces ``adult`` even though MOVIE
        itself is genre-neutral.
        """
        try:
            raw = self._ner.tag(query)
            entities = {e["label"]: e["word"] for e in raw}
        except Exception:
            return []
        media_type = self._entities_to_media_type(entities)
        if media_type is None:
            return []
        genres: List[str] = []
        for label in self._winning_labels(entities, media_type):
            for g in NER_LABEL_TO_GENRES.get(label, []):
                if g not in genres:
                    genres.append(g)
        return genres

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Domain is OCP_PLAY when any known entity is found, NOT_OCP otherwise."""
        try:
            raw = self._ner.tag(query)
            entities = {e["label"]: e["word"] for e in raw}
        except Exception as e:
            LOG.error(f"AhocorasickMediaClassifier NER failed: {e}")
            return OCPDomain.NOT_OCP, 0.0

        media_type = self._entities_to_media_type(entities)
        if media_type is None:
            return OCPDomain.NOT_OCP, 0.0
        return OCPDomain.OCP_PLAY, _HIT_CONFIDENCE
