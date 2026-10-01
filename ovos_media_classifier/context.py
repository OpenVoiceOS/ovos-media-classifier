"""Context-aware classification — the classifier's OWN standalone contract.

``classify_full`` accepts, beyond the utterance + lang, exactly **two** optional
context inputs.  They are the classifier's own contract (mediavocab-aligned where
natural), defaulting to ``None`` so the no-context call is today's behaviour:

* :class:`PlayerStatus` — what is *now playing* and the transport state
  (play / pause / stop).  This lets the classifier resolve **relative / control**
  intents that are ambiguous without it:

    * "play something else" → a re-query biased to the current ``media_type``;
    * "next" / "previous" / "pause" / "resume" → ``OCP_CONTROL`` even when no
      media keyword is present;
    * a light bias toward the current ``media_type`` on an ambiguous follow-up.

  It is **conservative**: it never overrides a confident explicit route (an
  explicit "play some jazz" wins regardless of what is playing).

* ``ner_list`` (``{ner_label: [entity, ...]}``) — the **available entities**: the
  skill-registered keywords + the user's library, i.e. *what the user actually
  has*.  It is the entity context for NER matching and the embedding router's
  runtime entity injection, threaded per-query so a caller can pass the live
  entity list without retraining.  Same shape as
  :meth:`EmbeddingMediaClassifier.register_user_library`.

What is intentionally NOT here
------------------------------
Device capabilities, blocked genres and session/QueryContext are the *pipeline's*
provider-filtering concerns, not classifier input — they stay in
``ovos-ocp-pipeline-plugin``'s ``QueryContext``.  The classifier's context is
exactly the two inputs above.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from mediavocab import MediaType


class PlayerState(str, Enum):
    """Transport state of the now-playing session."""
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


# A bare relative/control utterance ("next", "pause", "something else") is only
# actionable when there IS an active session.  These two states mean a session
# exists (playing or paused); STOPPED / no status means there is nothing to
# control or re-query against, so the context adds no signal.
ACTIVE_STATES = (PlayerState.PLAYING, PlayerState.PAUSED)


@dataclass
class PlayerStatus:
    """Minimal now-playing context: the current media type + transport state.

    Args:
        now_playing: the ``mediavocab.MediaType`` currently loaded (``None`` when
            nothing is). Drives the light follow-up bias and the
            "play something else" re-query type.
        state: the :class:`PlayerState` (playing / paused / stopped). A relative
            control ("next" / "pause") only fires when a session is *active*
            (see :data:`ACTIVE_STATES`).
    """
    now_playing: Optional[MediaType] = None
    state: PlayerState = PlayerState.STOPPED

    @property
    def is_active(self) -> bool:
        """True when there is a session to control / re-query (playing|paused)."""
        return self.state in ACTIVE_STATES

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["PlayerStatus"]:
        """Build from a plain ``{"now_playing": <type>, "state": <state>}`` dict.

        Tolerant: unknown / missing values degrade to ``None`` / ``STOPPED`` so a
        malformed status never raises (the no-context path is the floor).
        Returns ``None`` for a falsy input.
        """
        if not data:
            return None
        mt = data.get("now_playing")
        media_type: Optional[MediaType] = None
        if mt is not None:
            try:
                media_type = mt if isinstance(mt, MediaType) else MediaType(mt)
            except (ValueError, KeyError):
                media_type = None
        st = data.get("state")
        state = PlayerState.STOPPED
        if st is not None:
            try:
                state = st if isinstance(st, PlayerState) else PlayerState(str(st))
            except (ValueError, KeyError):
                state = PlayerState.STOPPED
        return cls(now_playing=media_type, state=state)
