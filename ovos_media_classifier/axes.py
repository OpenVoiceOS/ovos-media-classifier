"""Multi-axis (orthogonal) media classification.

Media classification is naturally coarse-to-fine: a few high-signal axes each
prune the leaf label space.  Rather than a brittle hard hierarchy, the axes are
**orthogonal** and combined into one :class:`MediaClassification`:

* **domain**        — is this a media request at all (play / control / not_media)
* **playback_type** — modality / "physical type": audio / video / paged /
  interactive (``mediavocab.PlaybackType``) — the coarse, high-confidence axis.
* **structure**     — single / episodic / continuous / collection (this module).
* **media_type**    — the concrete ``mediavocab.MediaType`` leaf.
* **genres**        — orthogonal tags (the ``adult`` genre drives content filtering).

The keyword classifier predicts the coarse axes **coarse-to-fine** from their
own ``.voc`` evidence (modality → structure) and *constrains* the leaf to them
(see ``keyword.py``).  The ``infer_*`` helpers here provide the leaf-derived
*defaults* — the fallback for locales without axis vocab and for plugins that
model only the leaf.  Trained classifier plugins MAY predict each axis with its
own head and soft-gate the leaf — see OVOS-MEDIA-CLASSIFY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from mediavocab import MediaType, PlaybackType, infer_playback_type

from ovos_media_classifier.intents import OCPDomain


class Structure(str, Enum):
    """How a work is structured in time — orthogonal to modality."""
    SINGLE = "single"          # one self-contained work: a movie, a track, a book
    EPISODIC = "episodic"      # a series of discrete instalments: tv series, podcast
    CONTINUOUS = "continuous"  # an unbounded live/looping stream: radio, live tv, ambient
    COLLECTION = "collection"  # an ordered set of works: a playlist
    UNKNOWN = "unknown"


# MediaType → default Structure.  Largely intrinsic to the type; a trained model
# MAY override per-utterance (e.g. "play the *album*" → COLLECTION).
MEDIA_TYPE_TO_STRUCTURE = {
    MediaType.MOVIE: Structure.SINGLE,
    MediaType.SHORT_FILM: Structure.SINGLE,
    MediaType.MUSIC: Structure.SINGLE,
    MediaType.MUSIC_VIDEO: Structure.SINGLE,
    MediaType.AUDIOBOOK: Structure.SINGLE,
    MediaType.BOOK: Structure.SINGLE,
    MediaType.COMIC: Structure.SINGLE,
    MediaType.GAME: Structure.SINGLE,
    MediaType.INTERACTIVE_FICTION: Structure.SINGLE,
    MediaType.SOUND_EFFECT: Structure.SINGLE,
    MediaType.EPISODIC_SERIES: Structure.EPISODIC,
    MediaType.PODCAST: Structure.EPISODIC,
    MediaType.AUDIO_DRAMA: Structure.EPISODIC,
    MediaType.TV: Structure.CONTINUOUS,        # live channel
    MediaType.RADIO: Structure.CONTINUOUS,
    MediaType.PROCEDURAL_AMBIENT: Structure.CONTINUOUS,
    MediaType.PLAYLIST: Structure.COLLECTION,
    MediaType.GENERIC: Structure.UNKNOWN,
    MediaType.NOT_MEDIA: Structure.UNKNOWN,
    MediaType.CONTROL: Structure.UNKNOWN,
}


def infer_structure(media_type: MediaType) -> Structure:
    """Default :class:`Structure` for a ``MediaType``."""
    return MEDIA_TYPE_TO_STRUCTURE.get(media_type, Structure.UNKNOWN)


@dataclass
class MediaClassification:
    """The full multi-axis result for one query."""
    media_type: MediaType
    playback_type: PlaybackType
    structure: Structure
    domain: OCPDomain
    genres: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {
            "media_type": self.media_type.value,
            "playback_type": self.playback_type.value,
            "structure": self.structure.value,
            "domain": self.domain.value,
            "genres": list(self.genres),
            "confidence": self.confidence,
        }


def classification_from_media_type(
    media_type: MediaType,
    domain: OCPDomain,
    genres: List[str],
    confidence: float,
) -> MediaClassification:
    """Build a :class:`MediaClassification`, deriving the coarse axes from the
    leaf ``MediaType`` (the keyword-classifier path)."""
    return MediaClassification(
        media_type=media_type,
        playback_type=infer_playback_type(media_type),
        structure=infer_structure(media_type),
        domain=domain,
        genres=list(genres or []),
        confidence=confidence,
    )
