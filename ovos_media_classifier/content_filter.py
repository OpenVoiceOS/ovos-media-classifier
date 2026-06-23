"""Content filtering for OCP media requests.

The *primary driver* for recognising sensitive media (adult, etc.) is to let
OVOS **block such requests by default** — this is a content-moderation /
parental-control capability, not a content provider.  The classifier surfaces a
``mediavocab`` genre signal (``adult`` for adult/hentai/porn queries) and this
filter decides whether the request is allowed.

Default policy
--------------
* ``adult`` is **blocked by default** (``allow_adult_content: false``).
* The administrator can extend the blocklist with any ``mediavocab`` genre or
  ``MediaType`` value, or disable filtering entirely.

Config (under ``mycroft.conf``)::

    {
      // master opt-in for adult content (default false => adult blocked)
      "allow_adult_content": false,

      "media_content_filter": {
        "enabled": true,
        "blocked_genres": ["adult"],          // mediavocab genre tags
        "blocked_media_types": []             // mediavocab MediaType values, e.g. "game"
      }
    }
"""
from typing import Iterable, List, Optional, Tuple

from mediavocab import MediaType

# adult is blocked unless the operator explicitly opts in
DEFAULT_BLOCKED_GENRES = ("adult",)


class ContentFilter:
    """Decides whether a classified media request is allowed.

    Args:
        config: the OCP / mycroft config block.  Reads ``allow_adult_content``
            (top-level convenience flag) and the ``media_content_filter`` sub-dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        cfg = config.get("media_content_filter", {}) or {}

        self.enabled: bool = cfg.get("enabled", True)

        blocked = set(cfg.get("blocked_genres", list(DEFAULT_BLOCKED_GENRES)))
        # top-level allow_adult_content (or nested) lifts the adult block
        allow_adult = config.get(
            "allow_adult_content", cfg.get("allow_adult_content", False)
        )
        if allow_adult:
            blocked.discard("adult")
        self.blocked_genres = {g.lower() for g in blocked}

        # accept either MediaType members or their string values
        self.blocked_media_types = {
            mt.value if isinstance(mt, MediaType) else str(mt)
            for mt in cfg.get("blocked_media_types", [])
        }

    def is_blocked(
        self,
        media_type: MediaType,
        genres: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, str]:
        """Return ``(blocked, reason)`` for a classification.

        A request is blocked if any of its genres is in ``blocked_genres`` or
        its media type is in ``blocked_media_types``.
        """
        if not self.enabled:
            return False, ""

        g = {x.lower() for x in (genres or [])}
        hit = g & self.blocked_genres
        if hit:
            return True, f"blocked genre: {', '.join(sorted(hit))}"

        mt_value = media_type.value if isinstance(media_type, MediaType) else str(media_type)
        if mt_value in self.blocked_media_types:
            return True, f"blocked media type: {mt_value}"

        return False, ""

    def check(self, classifier, query: str, lang: str = "en-us") -> Tuple[bool, str]:
        """Classify *query* with *classifier* and apply the filter.

        Convenience wrapper: pulls both the media type and the genre tags from an
        :class:`~ovos_media_classifier.base.AbstractMediaClassifier`.
        """
        media_type, _ = classifier.classify(query, lang)
        genres: List[str] = []
        try:
            genres = classifier.classify_genres(query, lang)
        except Exception:
            pass
        return self.is_blocked(media_type, genres)
