"""Keyword feature slots — the contextual-classification contract.

A **keyword feature slot** is a named entity category (``artist_name``,
``movie_title``, ``tv_show_title``, …) that maps to a point on the
[taxonomy](../docs/classification-model.md): a ``mediavocab.MediaType`` (+ its
playback type and any genre tags). A slot is a *definition*; it carries no
entities of its own. Entities are filled in at **runtime** from whatever media
is actually available — a Jellyfin library, the *arr stack (Radarr/Sonarr/
Lidarr), Music Assistant, a CSV/HF export (see
[entity lists](../docs/entity-lists.md) and ``entities.EntitiesContainer``).

The same slots drive **two** classifier mechanisms from one contract:

* **NER (exact match)** — a query containing a slot's entity yields that slot's
  ``media_type`` directly. Because the slot is filled with *your* media, the
  signal is grounded and high-confidence: "play Inception" resolves to ``MOVIE``
  *because Inception is in your Radarr*, not because of a guessed cue.
* **Guided embeddings (learned features)** — each slot is a categorical feature
  ("does the query contain a known ``artist_name``?"). A model learns weights
  over the slots; at runtime the slots are filled with the available entities,
  so the features — and therefore the prediction — are **biased toward the media
  you actually own**. Defining the labels statically and filling them at runtime
  is the same pattern as the ``guided-categorical-embeddings`` library.

This is what makes classification **contextual**: available media *influences*
the prediction. Ambiguous requests lean toward what is present; a slot with no
entities contributes nothing. It is the principled form of "constrain the leaf"
— grounded in real inventory rather than a noisy modality guess.

``KEYWORD_FEATURE_SLOTS`` is derived from the canonical
``NER_LABEL_TO_PLAY_INTENT`` + ``PLAY_INTENT_TO_MEDIA_TYPE`` +
``PLAY_INTENT_TO_GENRES`` maps, so it is a formalized *view* of the existing
taxonomy with no second source of truth to drift.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from mediavocab import MediaType, PlaybackType, infer_playback_type

from ovos_media_classifier.intents import (
    OCPEntityLabel,
    NER_LABEL_TO_PLAY_INTENT,
    PLAY_INTENT_TO_MEDIA_TYPE,
    PLAY_INTENT_TO_GENRES,
)

__all__ = [
    "KeywordFeatureSlot",
    "KEYWORD_FEATURE_SLOTS",
    "SLOTS_BY_LABEL",
    "slot_for_label",
    "slots_for_media_type",
]


@dataclass(frozen=True)
class KeywordFeatureSlot:
    """A named entity category usable as a classifier feature.

    Attributes:
        label: the slot name (an ``OCPEntityLabel`` value, e.g. ``artist_name``).
        media_type: the ``mediavocab.MediaType`` the slot implies when matched.
        playback_type: the modality of that media type (derived).
        genres: orthogonal genre tags carried by the slot (e.g. ``adult`` for
            ``hentai_title``) — the content-filter signal survives the slot.
    """

    label: str
    media_type: MediaType
    playback_type: PlaybackType
    genres: Tuple[str, ...] = field(default_factory=tuple)


def _build() -> List[KeywordFeatureSlot]:
    slots: List[KeywordFeatureSlot] = []
    for label, intent in NER_LABEL_TO_PLAY_INTENT.items():
        mt = PLAY_INTENT_TO_MEDIA_TYPE.get(intent, MediaType.GENERIC)
        genres = tuple(PLAY_INTENT_TO_GENRES.get(intent, ()))
        slots.append(KeywordFeatureSlot(
            label=getattr(label, "value", label),
            media_type=mt,
            playback_type=infer_playback_type(mt),
            genres=genres,
        ))
    return slots


#: The canonical slot definitions — one per entity label, single source of truth.
KEYWORD_FEATURE_SLOTS: List[KeywordFeatureSlot] = _build()

#: Fast lookup: slot label -> KeywordFeatureSlot.
SLOTS_BY_LABEL: Dict[str, KeywordFeatureSlot] = {s.label: s for s in KEYWORD_FEATURE_SLOTS}


def slot_for_label(label) -> KeywordFeatureSlot:
    """Return the :class:`KeywordFeatureSlot` for a label (str or OCPEntityLabel).

    Raises ``KeyError`` if the label is not a known slot.
    """
    return SLOTS_BY_LABEL[getattr(label, "value", label)]


def slots_for_media_type(media_type: MediaType) -> List[KeywordFeatureSlot]:
    """All slots that resolve to ``media_type`` (e.g. every music slot:
    ``artist_name``, ``track_name``, ``album_name``, …)."""
    return [s for s in KEYWORD_FEATURE_SLOTS if s.media_type == media_type]
