"""Keyword-based media classifier — hierarchical coarse-to-fine.

This is the ``.voc`` keyword backend, restructured from the original
leaf-first chain (which matched the most-specific ``*Keyword`` first and
*derived* the coarse axes from it) into a **coarse-to-fine** model:

    domain  →  modality (PlaybackType)  →  structure  →  constrained leaf

The high-signal coarse axes are predicted *first*, each from its own ``.voc``
evidence, and they **constrain** the leaf-candidate set so a stray specific
keyword cannot win (e.g. a strong "listen" audio cue prevents a video leaf).
The concrete ``mediavocab.MediaType`` leaf is then chosen *within* the
constrained set; if no specific leaf voc matches inside the set we fall back to
a sensible default leaf for that (modality, structure) cell.

It can work in two modes:

1. **Pipeline mode** – pass ``voc_match_func`` (typically
   ``OVOSAbstractApplication.voc_match`` bound to the pipeline plugin).
   The pipeline plugin owns the locale files; the classifier just calls
   the function.

2. **Standalone mode** – omit ``voc_match_func`` and the classifier loads
   the bundled ``.voc`` files from
   ``ovos_media_classifier/locale/<lang>/<VocabName>.voc`` via
   ``ovos-spec-tools`` (the OVOS-INTENT-2 loader) and matches them on
   **word boundaries**.
   Use ``KeywordMediaClassifier.from_locale_dir(locale_dir, lang)`` or
   simply ``KeywordMediaClassifier()`` to use the built-in locale files.

Graceful degradation: where a locale lacks the axis vocab (``VerbAudio`` /
``ModEpisode`` / …) the axis simply gathers no evidence and the model degrades
to leaf-only matching — the en-us path is the reference.

Usage::

    # Standalone (uses bundled locale)
    from ovos_media_classifier.keyword import KeywordMediaClassifier
    clf = KeywordMediaClassifier()
    media_type, conf = clf.classify("play some jazz", "en-us")

    # Pipeline (delegates to skill voc_match)
    clf = KeywordMediaClassifier(voc_match_func=self.voc_match)
    media_type, conf = clf.classify("play some jazz", "en-us")
"""
import os
import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple

from ovos_spec_tools import LocaleResources

from mediavocab import (
    MediaType,
    PlaybackType,
    MEDIA_TYPE_TO_PLAYBACK_TYPE,
    infer_playback_type,
)

from ovos_media_classifier.axes import (
    Structure,
    MEDIA_TYPE_TO_STRUCTURE,
    infer_structure,
    MediaClassification,
)
from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.constants import (
    DEFAULT_KEYWORD_CONFIDENCE,
    DEFAULT_KEYWORD_HIGH_CONFIDENCE,
    DEFAULT_KEYWORD_LOW_CONFIDENCE,
)
from ovos_media_classifier.intents import (
    OCPDomain,
    OCPControlIntent,
)

# Bundled locale directory (ported from ovos-ocp-pipeline-plugin)
_LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")


class _VocMatcher:
    """Vocabulary matcher backed by ``.voc`` files via ``ovos-spec-tools``.

    Wraps :class:`ovos_spec_tools.LocaleResources` (the OVOS-INTENT-2 spec
    loader) so a phrase matches the query on **word boundaries** rather than as
    a naive substring — German ``"sexfilm"`` no longer matches ``"film"`` and
    ``"audiobeschreibung"`` no longer matches ``"audio"``.

    ``LocaleResources`` owns the locale-dir resolution: it indexes the
    per-language subdirs case-insensitively (``en-us`` vs ``en-US``) and applies
    the spec §2.2 smart fallback (exact tag → language family → ``en-us``), so
    the prior hand-rolled case folding + ``lang``→language-only fallback are
    preserved. Compiled per-``(vocab, lang)`` regexes are cached.
    """

    def __init__(self, locale_dir: str) -> None:
        self._locale_dir = locale_dir
        self._resources = LocaleResources(skill_locale=locale_dir)

    @lru_cache(maxsize=1024)
    def _voc_phrases(self, vocab_name: str, lang: str) -> Tuple[str, ...]:
        # spec-tools resolves the lang subdir (case-insensitive) and the
        # language-family fallback chain internally.
        phrases = self._resources.vocabularies(lang).get(vocab_name)
        if phrases:
            return tuple(phrases)
        # Final safety net: the canonical en-us source.
        if lang.lower() != "en-us":
            phrases = self._resources.vocabularies("en-us").get(vocab_name)
            if phrases:
                return tuple(phrases)
        return ()

    @lru_cache(maxsize=1024)
    def _voc_regex(self, vocab_name: str, lang: str) -> Optional[re.Pattern]:
        phrases = self._voc_phrases(vocab_name, lang)
        if not phrases:
            return None
        # Longest-first so the alternation prefers the most specific phrase.
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        alternation = "|".join(re.escape(p) for p in sorted_phrases)
        return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)

    def match(self, phrase: str, vocab_name: str, lang: str) -> bool:
        rx = self._voc_regex(vocab_name, lang.lower())
        return bool(rx and rx.search(phrase))


# ---------------------------------------------------------------------------
# Precomputed axis inversions: MediaType ←→ (PlaybackType, Structure)
# ---------------------------------------------------------------------------

def _invert(mapping: Dict[MediaType, object]) -> Dict[object, Set[MediaType]]:
    out: Dict[object, Set[MediaType]] = {}
    for mt, axis in mapping.items():
        out.setdefault(axis, set()).add(mt)
    return out


# PlaybackType -> {MediaType...}, Structure -> {MediaType...}
_PLAYBACK_TO_MEDIA_TYPES: Dict[PlaybackType, Set[MediaType]] = _invert(
    MEDIA_TYPE_TO_PLAYBACK_TYPE
)
_STRUCTURE_TO_MEDIA_TYPES: Dict[Structure, Set[MediaType]] = _invert(
    MEDIA_TYPE_TO_STRUCTURE
)

# Default leaf MediaType per (modality, structure) cell — used when the
# constrained leaf chain matches no specific ``*Keyword`` voc but the coarse
# axes are confident.  This is the "sensible fallback" the maintainer asked for.
_DEFAULT_LEAF: Dict[Tuple[PlaybackType, Structure], MediaType] = {
    (PlaybackType.AUDIO, Structure.SINGLE): MediaType.MUSIC,
    (PlaybackType.AUDIO, Structure.EPISODIC): MediaType.PODCAST,
    (PlaybackType.AUDIO, Structure.CONTINUOUS): MediaType.RADIO,
    (PlaybackType.AUDIO, Structure.COLLECTION): MediaType.MUSIC,
    (PlaybackType.VIDEO, Structure.SINGLE): MediaType.MOVIE,
    (PlaybackType.VIDEO, Structure.EPISODIC): MediaType.EPISODIC_SERIES,
    (PlaybackType.VIDEO, Structure.CONTINUOUS): MediaType.TV,
    (PlaybackType.VIDEO, Structure.COLLECTION): MediaType.MOVIE,
    (PlaybackType.INTERACTIVE, Structure.SINGLE): MediaType.GAME,
    (PlaybackType.INTERACTIVE, Structure.EPISODIC): MediaType.GAME,
    (PlaybackType.INTERACTIVE, Structure.CONTINUOUS): MediaType.GAME,
    (PlaybackType.INTERACTIVE, Structure.COLLECTION): MediaType.GAME,
    (PlaybackType.PAGED, Structure.SINGLE): MediaType.BOOK,
    (PlaybackType.PAGED, Structure.EPISODIC): MediaType.COMIC,
    (PlaybackType.PAGED, Structure.CONTINUOUS): MediaType.BOOK,
    (PlaybackType.PAGED, Structure.COLLECTION): MediaType.BOOK,
}

# Per-modality fallback when structure is UNKNOWN — the single-structure default.
_MODALITY_DEFAULT_LEAF: Dict[PlaybackType, MediaType] = {
    PlaybackType.AUDIO: MediaType.MUSIC,
    PlaybackType.VIDEO: MediaType.MOVIE,
    PlaybackType.INTERACTIVE: MediaType.GAME,
    PlaybackType.PAGED: MediaType.BOOK,
}


# ---------------------------------------------------------------------------
# Transport-control voc resolution
#
# ``classify_control`` matches these ``Ctrl*.voc`` files **in order** (first hit
# wins) so the more-specific / less-ambiguous action is preferred when several
# could fire.  Notes on the deliberate ordering:
#   * SeekBackward before Prev, and SeekForward before Next: "go back 30
#     seconds" / "skip ahead a minute" is a seek, not a track change — a trailing
#     duration/number is the disambiguator (see ``_DURATION_RE``).
#   * Shuffle / Repeat are unambiguous keywords and come early.
#   * Pause/Stop/Resume are high-signal and ordered before the directional ones.
#   * Open / SaveGame / LoadGame / Like are domain-specific.
# Each entry: (voc_name, OCPControlIntent).
# ---------------------------------------------------------------------------
_CONTROL_VOC_ORDER: List[Tuple[str, "OCPControlIntent"]] = [
    ("CtrlShuffle", OCPControlIntent.SHUFFLE),
    ("CtrlRepeat", OCPControlIntent.REPEAT),
    ("CtrlPause", OCPControlIntent.PAUSE),
    ("CtrlStop", OCPControlIntent.STOP),
    ("CtrlResume", OCPControlIntent.RESUME),
    ("CtrlSeekBackward", OCPControlIntent.SEEK_BACKWARD),
    ("CtrlSeekForward", OCPControlIntent.SEEK_FORWARD),
    ("CtrlNext", OCPControlIntent.NEXT),
    ("CtrlPrev", OCPControlIntent.PREVIOUS),
    ("CtrlLoadGame", OCPControlIntent.LOAD_GAME),
    ("CtrlSaveGame", OCPControlIntent.SAVE_GAME),
    ("CtrlOpen", OCPControlIntent.OPEN),
    ("CtrlLike", OCPControlIntent.LIKE_SONG),
]

# A trailing duration ("30 seconds", "a minute", "2 mins") turns an ambiguous
# directional cue ("go back", "forward") into a *seek* rather than a track skip.
_DURATION_RE = re.compile(
    r"\b(?:\d+|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|sixty)\b"
    r"[^.]*?\b(?:sec(?:ond)?s?|min(?:ute)?s?|hours?|hrs?)\b",
    re.IGNORECASE,
)


class KeywordMediaClassifier(AbstractMediaClassifier):
    """Classifies media type using hierarchical voc keyword matching.

    The coarse axes (modality / structure) are predicted first from their own
    ``.voc`` evidence and constrain the leaf candidate set; the concrete
    ``mediavocab.MediaType`` is matched *within* that set.

    Args:
        voc_match_func: Optional callable with signature
            ``(phrase: str, vocab_name: str, *, lang: str) -> bool``.
            When omitted the bundled locale files are used directly.
        locale_dir: Override the locale directory used in standalone mode.
            Defaults to the bundled ``ovos_media_classifier/locale/``.
    """

    def __init__(
        self,
        voc_match_func: Optional[Callable] = None,
        locale_dir: Optional[str] = None,
    ) -> None:
        if voc_match_func is not None:
            self._voc_match = voc_match_func
        else:
            _matcher = _VocMatcher(locale_dir or _LOCALE_DIR)
            self._voc_match = _matcher.match

    @classmethod
    def from_locale_dir(cls, locale_dir: str) -> "KeywordMediaClassifier":
        """Create a standalone classifier reading ``.voc`` files from *locale_dir*.

        Args:
            locale_dir: Root directory containing per-language subdirs
                (e.g. ``locale/en-us/MusicKeyword.voc``).
        """
        return cls(locale_dir=locale_dir)

    def _match(self, phrase: str, vocab: str, lang: str) -> bool:
        return bool(self._voc_match(phrase, vocab, lang=lang))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Hierarchical coarse-to-fine classification.

        Predicts modality then structure, constrains the leaf candidates to the
        MediaTypes compatible with those axes, matches the leaf voc *within* the
        candidate set, and returns the constrained ``mediavocab.MediaType``.
        Falls back to (GENERIC, 0.0) when nothing media-ish matches.
        """
        media_type, _genres, conf = self._classify_leaf(query, lang, valid_labels)
        return media_type, conf

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Domain axis: ocp_control / ocp_play / not_ocp.

        Precedence — this is where the "play" ambiguity is resolved:

        * A concrete **media leaf** (music/movie/podcast/...) ⇒ ``OCP_PLAY``,
          *unless* it is merely the object of a transport control and no
          play-initiation verb is present ("stop the music", "save the game" →
          CONTROL).
        * A **play-initiation verb** (``Play.voc``: play/start/...) followed by
          content ⇒ ``OCP_PLAY`` even when no specific leaf voc matched
          ("play some jazz").
        * Otherwise a **transport-control** voc (pause/stop/next/seek/...) ⇒
          ``OCP_CONTROL``; bare "play"/"resume"/"continue" with no media also
          resolve here (→ CONTROL PLAY / RESUME) via ``classify_control``.
        * Else ⇒ ``NOT_OCP``.

        The control vocs deliberately exclude bare "play", so "play <media>"
        never trips a control match.
        """
        control = self.classify_control(query, lang)
        media_type, conf = self.classify(query, lang)
        has_leaf = media_type != MediaType.GENERIC
        play_verb = self._is_play_request(query, lang)

        # A new-play request (explicit play verb + content, or a concrete leaf)
        # beats a transport control unless the control is the dominant verb with
        # no play verb present (e.g. "stop the music").
        if (has_leaf or play_verb) and control is None:
            return OCPDomain.OCP_PLAY, (conf if has_leaf else DEFAULT_KEYWORD_CONFIDENCE)
        if (has_leaf or play_verb) and control is not None:
            # Both fired: a play verb means a *new* play request wins; otherwise
            # the control owns the (media-noun) object.
            if play_verb and control not in (OCPControlIntent.PLAY,
                                             OCPControlIntent.RESUME):
                return OCPDomain.OCP_PLAY, DEFAULT_KEYWORD_CONFIDENCE
            return OCPDomain.OCP_CONTROL, DEFAULT_KEYWORD_CONFIDENCE

        if control is not None:
            return OCPDomain.OCP_CONTROL, DEFAULT_KEYWORD_CONFIDENCE
        return OCPDomain.NOT_OCP, 0.0

    def _is_play_request(self, query: str, lang: str) -> bool:
        """True when a play-initiation verb (``Play.voc``) is followed by content.

        "play some jazz" / "start the movie" are new-play requests; a *bare*
        play verb ("play", "start") with nothing after it is not (it is handled
        as a CONTROL PLAY resume-current).
        """
        if not self._match(query, "Play", lang):
            return False
        # require something beyond the bare verb (a query / media noun)
        words = [w for w in re.split(r"\s+", query.strip()) if w]
        return len(words) > 1

    def classify_control(
        self, query: str, lang: str
    ) -> Optional[OCPControlIntent]:
        """Match a transport-control action from the ``Ctrl*.voc`` files.

        Matches the per-action control vocs in ``_CONTROL_VOC_ORDER`` (most
        specific / least ambiguous first) on word boundaries and returns the
        first :class:`~ovos_media_classifier.intents.OCPControlIntent` that
        fires; ``None`` when no control voc matches.

        Disambiguations:

        * A trailing duration ("go back 30 seconds", "skip ahead a minute")
          forces an ambiguous directional cue to a *seek* (SEEK_BACKWARD /
          SEEK_FORWARD) rather than a track change (PREVIOUS / NEXT).
        * "play" is never a control cue here; a *bare* play/resume verb with no
          media falls through to ``OCPControlIntent.PLAY`` / ``RESUME`` so
          "play"/"resume"/"continue" (resume-current) are still control.
        """
        m = self._match
        q = query
        has_duration = bool(_DURATION_RE.search(q))

        for voc_name, action in _CONTROL_VOC_ORDER:
            if not m(q, voc_name, lang):
                continue
            if action is OCPControlIntent.SEEK_BACKWARD and not has_duration:
                if not self._explicit_seek_back(q, lang):
                    continue
            if action is OCPControlIntent.SEEK_FORWARD and not has_duration:
                if not self._explicit_seek_fwd(q, lang):
                    continue
            return action

        # Bare play/resume verb with no media leaf ⇒ resume-current control.
        # ``Resume.voc`` (resume/unpause) is unambiguous → RESUME.
        if m(q, "Resume", lang):
            return OCPControlIntent.RESUME
        # A *bare* play verb ("play", "start") with no media is CONTROL PLAY
        # (resume the current track).  "play <media>" is filtered out by the
        # word-count guard so it stays a play request, not a control.
        if m(q, "Play", lang) and not self._is_play_request(q, lang):
            return OCPControlIntent.PLAY
        return None

    def _explicit_seek_back(self, q: str, lang: str) -> bool:
        """True when an *unambiguous* rewind cue (not just "go back") fired."""
        for kw in ("rewind", "skip back", "jump back", "seek backward",
                   "seek back"):
            if re.search(rf"\b{re.escape(kw)}\b", q, re.IGNORECASE):
                return True
        return False

    def _explicit_seek_fwd(self, q: str, lang: str) -> bool:
        """True when an *unambiguous* fast-forward cue (not just "forward") fired."""
        for kw in ("fast forward", "skip ahead", "jump ahead", "skip forward",
                   "seek forward"):
            if re.search(rf"\b{re.escape(kw)}\b", q, re.IGNORECASE):
                return True
        return False

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Return mediavocab genre tags implied by the winning keyword intent.

        This is what the content filter blocks on (e.g. an adult or anime
        query surfaces ``["adult"]`` / ``["anime"]`` even though the public
        ``MediaType`` is a generic ``MOVIE`` / ``EPISODIC_SERIES``).  Adult
        precedence is preserved so adult/hentai/porn cues are never lost.
        """
        _media_type, genres, _ = self._classify_leaf(query, lang, None)
        return list(genres)

    # ------------------------------------------------------------------
    # Multi-axis output — PREDICT each axis top-down (do not derive from leaf).
    # ------------------------------------------------------------------

    def classify_playback_type(self, query: str, lang: str) -> PlaybackType:
        """Return the PREDICTED ``mediavocab.PlaybackType`` (the modality axis)."""
        modality, _ = self._predict_modality(query, lang)
        if modality is not PlaybackType.UNKNOWN:
            return modality
        # No coarse evidence — fall back to the leaf's intrinsic modality.
        media_type, _ = self.classify(query, lang)
        return infer_playback_type(media_type)

    def classify_structure(self, query: str, lang: str) -> Structure:
        """Return the PREDICTED :class:`Structure`.

        Predicted from structure cues, with the constrained leaf as a tie-break
        so an unambiguous leaf (radio/podcast/series) still carries its
        intrinsic structure when no modifier voc fired.
        """
        modality, _ = self._predict_modality(query, lang)
        structure = self._predict_structure(query, lang, modality)
        if structure is not Structure.UNKNOWN:
            return structure
        media_type, _ = self.classify(query, lang)
        return infer_structure(media_type)

    def classify_full(self, query: str, lang: str) -> MediaClassification:
        """Full multi-axis result, predicting each axis top-down.

        Unlike the base implementation (which derives the coarse axes from the
        leaf), this predicts modality and structure from their own voc evidence,
        constrains the leaf to them, and reports all four axes consistently.

        When the query is a **transport-control** request (``OCP_CONTROL``) the
        media axes are left UNKNOWN/GENERIC (a pure control has no media leaf)
        and ``control_intent`` carries the action.
        """
        modality, _ = self._predict_modality(query, lang)
        structure = self._predict_structure(query, lang, modality)
        media_type, leaf_genres, conf = self._classify_leaf(query, lang, None)
        genres = list(leaf_genres)

        # The domain head owns the play-vs-control precedence (see
        # ``classify_domain``); keep ``classify_full`` consistent with it.
        domain, dconf = self.classify_domain(query, lang)
        control_intent: Optional[OCPControlIntent] = None

        if domain is OCPDomain.OCP_CONTROL:
            # Pure transport control — media axes are unknown.
            control_intent = self.classify_control(query, lang)
            media_type = MediaType.GENERIC
            playback = PlaybackType.UNKNOWN
            structure = Structure.UNKNOWN
            genres = []
            conf = dconf
        elif domain is OCPDomain.OCP_PLAY:
            if media_type == MediaType.GENERIC:
                # play-verb request with no specific leaf ("play some jazz")
                conf = dconf
                playback = (modality if modality is not PlaybackType.UNKNOWN
                            else PlaybackType.UNKNOWN)
                if structure is Structure.UNKNOWN and modality is not PlaybackType.UNKNOWN:
                    structure = Structure.SINGLE
            else:
                # Prefer the predicted coarse axes; degrade to the leaf's
                # intrinsic axis where no coarse evidence was found.
                playback = (modality if modality is not PlaybackType.UNKNOWN
                            else infer_playback_type(media_type))
                if structure is Structure.UNKNOWN:
                    structure = infer_structure(media_type)
        else:
            media_type = MediaType.GENERIC
            genres = []
            playback = PlaybackType.UNKNOWN
            structure = Structure.UNKNOWN

        return MediaClassification(
            media_type=media_type,
            playback_type=playback,
            structure=structure,
            domain=domain,
            genres=genres,
            confidence=conf,
            control_intent=control_intent,
        )

    # ------------------------------------------------------------------
    # Coarse axis prediction (each axis from its own voc evidence)
    # ------------------------------------------------------------------

    def _predict_modality(
        self, query: str, lang: str
    ) -> Tuple[PlaybackType, int]:
        """Predict the modality (PlaybackType) from per-axis voc evidence.

        Each modality is scored by how many of its cues match: explicit
        modality verbs (``VerbAudio``/``VerbVideo``/``VerbGame``/``VerbRead``/
        ``VerbTune``), structure cues that are modality-specific (``ModLive`` →
        video), and leaf-family keywords (Music/Movie/… count as modality
        evidence for their modality).  The strongest score wins; ties break by a
        fixed modality preference.  Returns ``(UNKNOWN, 0)`` with no evidence.
        """
        m = self._match
        q = query
        scores: Dict[PlaybackType, int] = {
            PlaybackType.AUDIO: 0,
            PlaybackType.VIDEO: 0,
            PlaybackType.INTERACTIVE: 0,
            PlaybackType.PAGED: 0,
        }

        # --- explicit modality verbs (high-signal) ---
        if m(q, "VerbAudio", lang):
            scores[PlaybackType.AUDIO] += 2
        if m(q, "VerbTune", lang):
            scores[PlaybackType.AUDIO] += 2
        if m(q, "VerbVideo", lang):
            scores[PlaybackType.VIDEO] += 2
        if m(q, "VerbGame", lang):
            scores[PlaybackType.INTERACTIVE] += 2
        if m(q, "VerbRead", lang):
            scores[PlaybackType.PAGED] += 2

        # --- structure cues that imply a modality ---
        if m(q, "ModLive", lang):
            scores[PlaybackType.VIDEO] += 1

        # --- leaf-family keywords as modality evidence ---
        # audio families
        for voc in ("MusicKeyword", "AudioKeyword", "PodcastKeyword",
                    "RadioKeyword", "AudioBookKeyword", "AudioDramaKeyword",
                    "ASMRKeyword", "NewsKeyword"):
            if m(q, voc, lang):
                scores[PlaybackType.AUDIO] += 1
        # video families
        for voc in ("MovieKeyword", "VideoKeyword", "TVKeyword", "IPTVKeyword",
                    "SeriesKeyword", "AnimeKeyword", "CartoonKeyword",
                    "DocumentaryKeyword", "MusicVideoKeyword", "TrailerKeyword",
                    "BehindTheScenesKeyword", "ADKeyword"):
            if m(q, voc, lang):
                scores[PlaybackType.VIDEO] += 1
        # interactive
        if m(q, "GameKeyword", lang):
            scores[PlaybackType.INTERACTIVE] += 1
        # paged
        if m(q, "ComicBookKeyword", lang):
            scores[PlaybackType.PAGED] += 1

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return PlaybackType.UNKNOWN, 0
        # tie-break by a stable preference (audio < video < interactive < paged)
        top = scores[best]
        order = [PlaybackType.AUDIO, PlaybackType.VIDEO,
                 PlaybackType.INTERACTIVE, PlaybackType.PAGED]
        for pb in order:
            if scores[pb] == top:
                return pb, top
        return best, top

    def _predict_structure(
        self, query: str, lang: str, modality: PlaybackType
    ) -> Structure:
        """Predict the temporal Structure from cue vocs.

        ``ModEpisode``/``ModSeason``/``SeriesKeyword``/``PodcastKeyword`` →
        episodic; ``ModLive``/``VerbTune``/``RadioKeyword``/``IPTVKeyword`` →
        continuous; album/playlist cues → collection; else single (or UNKNOWN
        when there is no media evidence at all).
        """
        m = self._match
        q = query

        # Structure-modifier cues + leaf families that are intrinsically episodic
        # (a series / podcast / anime / cartoon is a run of discrete instalments).
        episodic = (m(q, "ModEpisode", lang) or m(q, "ModSeason", lang) or
                    m(q, "SeriesKeyword", lang) or m(q, "PodcastKeyword", lang) or
                    m(q, "AudioDramaKeyword", lang) or m(q, "AnimeKeyword", lang) or
                    m(q, "CartoonKeyword", lang))
        # ...and leaf families that are intrinsically continuous (live channel,
        # radio, ambient stream, news bulletin loop).
        continuous = (m(q, "ModLive", lang) or m(q, "VerbTune", lang) or
                      m(q, "RadioKeyword", lang) or m(q, "IPTVKeyword", lang) or
                      m(q, "TVKeyword", lang) or m(q, "ASMRKeyword", lang) or
                      m(q, "NewsKeyword", lang))
        collection = m(q, "PlaylistKeyword", lang) or m(q, "ModCollection", lang)

        # Episodic is the most specific signal (a season/episode of something).
        if episodic and not continuous:
            return Structure.EPISODIC
        if continuous and not episodic:
            return Structure.CONTINUOUS
        if episodic and continuous:
            # both fired (e.g. "live podcast") — episodic wins (discrete instalments)
            return Structure.EPISODIC
        if collection:
            return Structure.COLLECTION

        # No structure modifier: SINGLE if there is any coarse/leaf evidence,
        # else UNKNOWN (nothing media-ish matched).
        if modality is not PlaybackType.UNKNOWN:
            return Structure.SINGLE
        return Structure.UNKNOWN

    # ------------------------------------------------------------------
    # Constrained leaf resolution
    # ------------------------------------------------------------------

    def _candidate_media_types(
        self, modality: PlaybackType, structure: Structure
    ) -> Optional[Set[MediaType]]:
        """Build the leaf candidate set from the predicted coarse axes.

        Candidates are the MediaTypes whose intrinsic ``infer_playback_type``
        equals the predicted modality AND whose ``infer_structure`` is
        compatible with the predicted structure.  Returns ``None`` when modality
        is UNKNOWN (no constraint — leaf-only fallback).
        """
        if modality is PlaybackType.UNKNOWN:
            return None
        by_modality = _PLAYBACK_TO_MEDIA_TYPES.get(modality, set())
        if structure is Structure.UNKNOWN:
            return set(by_modality)
        by_structure = _STRUCTURE_TO_MEDIA_TYPES.get(structure, set())
        constrained = by_modality & by_structure
        # If the modality has no type for that structure (e.g. interactive +
        # continuous), relax the structure constraint rather than emptying out.
        return constrained or set(by_modality)

    def _classify_leaf(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]],
    ) -> Tuple[MediaType, List[str], float]:
        """Resolve the leaf ``(MediaType, genres, confidence)`` coarse-to-fine.

        1. Adult family FIRST. Word-boundary matching (via ovos-spec-tools)
           already prevents the old substring collisions (German "sexfilm" no
           longer leaks "film"), so this gate is no longer a collision guard —
           it stays because it owns the adult/hentai/audio routing: it maps the
           adult cues to the correct type + ``adult`` (and, for hentai,
           ``anime``) genre tag so the content filter always sees the signal.
        2. Predict modality, then structure.
        3. Constrain the leaf candidates to those axes and run the specific-leaf
           voc chain; the first match wins.
        4. If no specific leaf matched but the coarse axes are confident, emit
           the DEFAULT leaf for that (modality, structure) cell.

        ``valid_labels`` (mediavocab types) gate every branch by MediaType.
        """
        valid = set(valid_labels) if valid_labels is not None else None
        m = self._match
        q = query

        def _ok(media_type: MediaType) -> bool:
            return valid is None or media_type in valid

        # --- 1. Adult family (owns adult/hentai genre routing) ----------------
        # adult → MOVIE + ["adult"]; adult_audio → MUSIC + ["adult"];
        # hentai → EPISODIC_SERIES + ["anime", "adult"].
        if (m(q, "AdultKeyword", lang) or m(q, "HentaiKeyword", lang)):
            if (_ok(MediaType.EPISODIC_SERIES) and
                    (m(q, "HentaiKeyword", lang) or
                     m(q, "CartoonKeyword", lang) or
                     m(q, "AnimeKeyword", lang))):
                return (MediaType.EPISODIC_SERIES, ["anime", "adult"],
                        DEFAULT_KEYWORD_LOW_CONFIDENCE)
            if (_ok(MediaType.MUSIC) and
                    (m(q, "AudioKeyword", lang) or m(q, "ASMRKeyword", lang))):
                return MediaType.MUSIC, ["adult"], DEFAULT_KEYWORD_LOW_CONFIDENCE
            if _ok(MediaType.MOVIE):
                return MediaType.MOVIE, ["adult"], DEFAULT_KEYWORD_LOW_CONFIDENCE

        # --- 2. Predict the coarse axes ---------------------------------------
        modality, _strength = self._predict_modality(q, lang)
        structure = self._predict_structure(q, lang, modality)

        # --- 3. Specific-leaf chain: a direct voc match is strong evidence -----
        # A direct specific-leaf voc match must override a *mispredicted* coarse
        # axis, so the chain runs leaf-first gated only by ``valid_labels``. This
        # keeps real-query accuracy at least on par with flat leaf-first matching
        # (a hard axis gate regressed macro-F1 when modality prediction was noisy).
        hit = self._match_leaf_chain(q, lang, _ok)
        if hit is not None:
            return hit

        # --- 4. Default leaf for the (modality, structure) cell ---------------
        # No specific leaf voc matched, but the coarse axes carry evidence — emit
        # the sensible default leaf so a strong modality verb ("listen", "watch")
        # still resolves to *something*.
        if modality is not PlaybackType.UNKNOWN:
            default = _DEFAULT_LEAF.get(
                (modality, structure),
                _MODALITY_DEFAULT_LEAF.get(modality),
            )
            if default is not None:
                leaf = self._default_leaf_type(default)
                if leaf is not None and _ok(leaf):
                    return leaf, [], DEFAULT_KEYWORD_LOW_CONFIDENCE

        return MediaType.GENERIC, [], 0.0

    def _match_leaf_chain(
        self, q: str, lang: str, allow: Callable[[MediaType], bool]
    ) -> Optional[Tuple[MediaType, List[str], float]]:
        """Specific-leaf voc priority chain (most-specific first).

        ``allow(media_type)`` decides whether a branch may fire (the
        ``valid_labels`` gate). Returns ``(MediaType, genres, confidence)`` on
        the first match, else ``None``.

        Several leaves collapse onto one ``MediaType`` but carry distinct genre
        tags (anime → EPISODIC_SERIES + ["anime"]; cartoon → EPISODIC_SERIES +
        ["animation"]) or distinct confidences (documentary / trailer / bts all
        → MOVIE) — the chain emits the right ``(type, genres)`` pair directly.
        """
        m = self._match
        T = MediaType
        if allow(T.MOVIE) and m(q, "DocumentaryKeyword", lang):
            return T.MOVIE, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.AUDIOBOOK) and m(q, "AudioBookKeyword", lang):
            return T.AUDIOBOOK, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.RADIO) and m(q, "NewsKeyword", lang):
            return T.RADIO, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.EPISODIC_SERIES) and m(q, "AnimeKeyword", lang):
            return T.EPISODIC_SERIES, ["anime"], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.EPISODIC_SERIES) and m(q, "CartoonKeyword", lang):
            return T.EPISODIC_SERIES, ["animation"], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.PODCAST) and m(q, "PodcastKeyword", lang):
            return T.PODCAST, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.AUDIO_DRAMA) and m(q, "AudioDramaKeyword", lang):
            return T.AUDIO_DRAMA, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.RADIO) and m(q, "RadioKeyword", lang):
            return T.RADIO, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.MUSIC_VIDEO) and m(q, "MusicVideoKeyword", lang):
            return T.MUSIC_VIDEO, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.MUSIC) and m(q, "MusicKeyword", lang):
            return T.MUSIC, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.TV) and m(q, "IPTVKeyword", lang):
            return T.TV, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.EPISODIC_SERIES) and m(q, "SeriesKeyword", lang):
            return T.EPISODIC_SERIES, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.TV) and m(q, "TVKeyword", lang):
            return T.TV, [], DEFAULT_KEYWORD_CONFIDENCE

        # Movie family: MOVIE and SHORT_FILM are distinct types; silent / b&w are
        # MOVIE with no extra genre but a higher (more specific) confidence.
        if (allow(T.MOVIE) or allow(T.SHORT_FILM)) and m(q, "MovieKeyword", lang):
            if allow(T.SHORT_FILM) and m(q, "ShortKeyword", lang):
                return T.SHORT_FILM, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if allow(T.MOVIE) and m(q, "SilentKeyword", lang):
                return T.MOVIE, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if allow(T.MOVIE) and m(q, "BWKeyword", lang):
                return T.MOVIE, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if allow(T.MOVIE):
                return T.MOVIE, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.MOVIE) and m(q, "TrailerKeyword", lang):
            return T.MOVIE, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.MOVIE) and m(q, "BehindTheScenesKeyword", lang):
            return T.MOVIE, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.COMIC) and m(q, "ComicBookKeyword", lang):
            return T.COMIC, [], DEFAULT_KEYWORD_LOW_CONFIDENCE
        if allow(T.GAME) and m(q, "GameKeyword", lang):
            return T.GAME, [], DEFAULT_KEYWORD_LOW_CONFIDENCE
        if allow(T.MOVIE) and m(q, "ADKeyword", lang):
            return T.MOVIE, [], DEFAULT_KEYWORD_LOW_CONFIDENCE
        if allow(T.PROCEDURAL_AMBIENT) and m(q, "ASMRKeyword", lang):
            return T.PROCEDURAL_AMBIENT, ["asmr"], DEFAULT_KEYWORD_LOW_CONFIDENCE
        if allow(T.MOVIE) and m(q, "VideoKeyword", lang):
            return T.MOVIE, [], DEFAULT_KEYWORD_LOW_CONFIDENCE
        if allow(T.MUSIC) and m(q, "AudioKeyword", lang):
            return T.MUSIC, [], DEFAULT_KEYWORD_LOW_CONFIDENCE
        return None

    @staticmethod
    def _default_leaf_type(media_type: MediaType) -> Optional[MediaType]:
        """Resolve a default-cell leaf to a concrete public ``MediaType``.

        ``BOOK`` has no dedicated keyword leaf — the closest paged leaf the
        keyword backend models is the comic (visual story); audiobook would
        change the modality, so map BOOK → COMIC.
        """
        if media_type is MediaType.BOOK:
            return MediaType.COMIC
        return media_type
