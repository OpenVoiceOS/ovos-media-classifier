"""Multi-axis (orthogonal) media classification.

Media classification is naturally coarse-to-fine: a few high-signal axes each
prune the leaf label space.  Rather than a brittle hard hierarchy, the axes are
**orthogonal** and combined into one :class:`MediaClassification`:

* **domain**        — is this a media request at all (play / control / not_media)
* **playback_type** — modality / "physical type": audio / video / paged /
  interactive (``mediavocab.PlaybackType``) — the coarse, high-confidence axis.
* **structure**     — single / episodic / continuous / collection
  (``mediavocab.Structure``, re-exported here).
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
from typing import List, Optional

from mediavocab import (
    MediaType,
    PlaybackType,
    infer_playback_type,
    Structure,
    MEDIA_TYPE_TO_STRUCTURE,
    infer_structure,
)

from ovos_media_classifier.intents import OCPDomain, OCPControlIntent


@dataclass
class MediaClassification:
    """The full multi-axis result for one query."""
    media_type: MediaType
    playback_type: PlaybackType
    structure: Structure
    domain: OCPDomain
    genres: List[str] = field(default_factory=list)
    confidence: float = 0.0
    # The control action when ``domain == OCP_CONTROL`` (else ``None``).
    control_intent: Optional[OCPControlIntent] = None

    def as_dict(self) -> dict:
        return {
            "media_type": self.media_type.value,
            "playback_type": self.playback_type.value,
            "structure": self.structure.value,
            "domain": self.domain.value,
            "genres": list(self.genres),
            "confidence": self.confidence,
            "control_intent": (self.control_intent.value
                               if self.control_intent is not None else None),
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
