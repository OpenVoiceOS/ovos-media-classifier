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
from typing import Callable, Dict, List, Optional, Tuple

from ovos_spec_tools import LocaleResources

from mediavocab import (
    MediaType,
    PlaybackType,
    infer_playback_type,
    Structure,
    infer_structure,
)
from mediavocab.taxonomy import (
    ContentForm, AccessibilityKind, ProgrammeFormat, PictureFormat,
)

from ovos_media_classifier.axes import (
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
# Default leaf MediaType per (modality, structure) cell — used when the
# constrained leaf chain matches no specific ``*Keyword`` voc but the coarse
# axes are confident.  This is the sensible per-cell fallback leaf.
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

# ---------------------------------------------------------------------------
# Supplementary-content voc → mediavocab.ContentForm
#
# Cues that are NOT a media type but an experiential *kind* of a parent title:
# the parent ``MediaType`` stays MOVIE / EPISODIC but the "trailer / BTS, not
# the full title" signal rides ``mediavocab.ContentForm``.  ``classify_content_form``
# returns the FIRST match in this order (the experiential kind is single-valued
# on a work), so the more-specific cue is preferred.  Each entry:
# (voc_name, ContentForm).
# ---------------------------------------------------------------------------
_CONTENT_FORM_VOC_ORDER: List[Tuple[str, ContentForm]] = [
    ("TeaserKeyword",          ContentForm.TEASER),
    ("TrailerKeyword",         ContentForm.TRAILER),
    ("MakingOfKeyword",        ContentForm.BEHIND_SCENES),
    ("BehindTheScenesKeyword", ContentForm.BEHIND_SCENES),
    ("BloopersKeyword",        ContentForm.BEHIND_SCENES),
    ("DeletedScenesKeyword",   ContentForm.BEHIND_SCENES),
    ("FeaturetteKeyword",      ContentForm.BEHIND_SCENES),
    ("InterviewKeyword",       ContentForm.SUPPLEMENT),
    ("ClipKeyword",            ContentForm.EXCERPT),
]

# Programme-format voc → mediavocab.ProgrammeFormat (single-valued).
# Only the cues with bundled vocabulary are wired; the rest of the enum
# (concert / stand_up / talk_show / reality / sports / quiz) has no .voc yet.
# documentary / news keep their carrier ``MediaType`` (MOVIE / RADIO) — the
# structural format rides this orthogonal axis (un-collapsed from the leaf).
_PROGRAMME_FORMAT_VOC_ORDER: List[Tuple[str, ProgrammeFormat]] = [
    ("DocumentaryKeyword",     ProgrammeFormat.DOCUMENTARY),
    ("NewsKeyword",            ProgrammeFormat.NEWS),
]

# Accessibility-asset voc → mediavocab.AccessibilityKind (multi-label).
_ACCESSIBILITY_VOC_ORDER: List[Tuple[str, AccessibilityKind]] = [
    ("ADKeyword",              AccessibilityKind.AUDIO_DESCRIPTION),
]

# Picture-presentation voc → mediavocab.PictureFormat (multi-label).
# ``classify_picture_format`` collects every match.
_PICTURE_FORMAT_VOC_ORDER: List[Tuple[str, PictureFormat]] = [
    ("SilentKeyword",          PictureFormat.SILENT),
    ("BWKeyword",              PictureFormat.BLACK_AND_WHITE),
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

    def _game_evidence(self, query: str, lang: str) -> bool:
        """True when there is genuine GAME evidence, not a play-verb collision.

        In several languages the play-initiation imperative is spelled exactly
        like the GAME noun ("spiel(e)" de, "spil" da). A bare "spiel <music>"
        is a transport request, not a game request, so a *leading* play verb
        must not, by itself, trigger the GAME leaf. We therefore strip a leading
        play verb before testing ``GameKeyword``; an explicit ``VerbGame`` cue
        (launch / boot / "spiel das spiel") always counts as game evidence.
        """
        if self._match(query, "VerbGame", lang):
            return True
        if not self._match(query, "GameKeyword", lang):
            return False
        # Strip a leading play verb (Play.voc) and re-test: if the only game
        # cue was the leading "play" verb itself, no game evidence remains.
        words = [w for w in re.split(r"\s+", query.strip()) if w]
        if len(words) > 1 and self._match(words[0], "Play", lang):
            remainder = " ".join(words[1:])
            return self._match(remainder, "GameKeyword", lang)
        return True

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

    def classify_content_form(self, query: str, lang: str) -> Optional[ContentForm]:
        """Return the :class:`mediavocab.ContentForm` implied by the query.

        Supplementary-content cues (``trailer`` / ``teaser`` / ``behind_scenes``
        / ``excerpt`` / ``supplement``) are NOT media types — the parent
        ``MediaType`` stays MOVIE / EPISODIC; the experiential kind rides this
        axis.  The first match in ``_CONTENT_FORM_VOC_ORDER`` wins (single-valued
        on a work), so "show me bloopers" → ``ContentForm.BEHIND_SCENES`` and
        "the Dune teaser" → ``ContentForm.TEASER`` fire even with no title.
        """
        m = self._match
        for voc_name, form in _CONTENT_FORM_VOC_ORDER:
            if m(query, voc_name, lang):
                return form
        return None

    def classify_programme_format(self, query: str, lang: str):
        """Return the :class:`mediavocab.ProgrammeFormat` implied by the query.

        A documentary / news broadcast is a structural programme format, not a
        media type — the ``MediaType`` stays the carrier; the format rides this
        orthogonal axis.  The first match in ``_PROGRAMME_FORMAT_VOC_ORDER``
        wins (single-valued).
        """
        m = self._match
        for voc_name, fmt in _PROGRAMME_FORMAT_VOC_ORDER:
            if m(query, voc_name, lang):
                return fmt
        return None

    def classify_accessibility(
        self, query: str, lang: str
    ) -> List[AccessibilityKind]:
        """Return the :class:`mediavocab.AccessibilityKind` assets requested.

        Matched directly from the ``.voc`` evidence (word boundaries), multi-label.
        """
        m = self._match
        kinds: List[AccessibilityKind] = []
        for voc_name, kind in _ACCESSIBILITY_VOC_ORDER:
            if m(query, voc_name, lang) and kind not in kinds:
                kinds.append(kind)
        return kinds

    def classify_picture_format(self, query: str, lang: str) -> List[PictureFormat]:
        """Return the :class:`mediavocab.PictureFormat` presentation attributes
        (``silent`` / ``black_and_white``), multi-label.

        Matched directly from the ``.voc`` evidence (word boundaries), so
        "a silent film" → ``[PictureFormat.SILENT]`` fires even with no title.
        """
        m = self._match
        formats: List[PictureFormat] = []
        for voc_name, fmt in _PICTURE_FORMAT_VOC_ORDER:
            if m(query, voc_name, lang) and fmt not in formats:
                formats.append(fmt)
        return formats

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

    def classify_full(self, query: str, lang: str,
                      player_status=None, ner_list=None) -> MediaClassification:
        """Full multi-axis result, predicting each axis top-down.

        Unlike the base implementation (which derives the coarse axes from the
        leaf), this predicts modality and structure from their own voc evidence,
        constrains the leaf to them, and reports all four axes consistently.

        When the query is a **transport-control** request (``OCP_CONTROL``) the
        media axes are left UNKNOWN/GENERIC (a pure control has no media leaf)
        and ``control_intent`` carries the action.

        *player_status* / *ner_list* are the standalone context inputs (see
        :meth:`AbstractMediaClassifier.classify_full`).  The keyword backend has
        no entity stream so *ner_list* is inert here; *player_status* is layered
        on conservatively by the shared base helper after the context-free axes
        are predicted.
        """
        result = self._classify_full_nocontext(query, lang)
        return self._apply_player_status(self, query, lang, result, player_status)

    def _classify_full_nocontext(self, query: str, lang: str) -> MediaClassification:
        """The context-free multi-axis result (see :meth:`classify_full`)."""
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
                    "TeaserKeyword", "BehindTheScenesKeyword", "MakingOfKeyword",
                    "BloopersKeyword", "DeletedScenesKeyword", "FeaturetteKeyword",
                    "InterviewKeyword", "ClipKeyword", "ADKeyword"):
            if m(q, voc, lang):
                scores[PlaybackType.VIDEO] += 1
        # interactive (a leading play verb that is spelled like the GAME noun,
        # e.g. de "spiel"/da "spil", must not count as game evidence)
        if self._game_evidence(q, lang):
            scores[PlaybackType.INTERACTIVE] += 1
        # paged
        if m(q, "ComicBookKeyword", lang):
            scores[PlaybackType.PAGED] += 1
        if m(q, "BookKeyword", lang):
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
                    (m(q, "AudioKeyword", lang) or m(q, "ASMRKeyword", lang) or
                     m(q, "VerbAudio", lang))):
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
            if default is not None and _ok(default):
                return default, [], DEFAULT_KEYWORD_LOW_CONFIDENCE

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
        # High-specificity leaves first (unambiguous keyword cues).  Ambient is
        # tested before SoundEffect because phrases like "white noise" / "rain
        # sounds" are ambient streams, not one-shot effects.
        if allow(T.PROCEDURAL_AMBIENT) and m(q, "AmbientKeyword", lang):
            return T.PROCEDURAL_AMBIENT, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.SOUND_EFFECT) and m(q, "SoundEffectKeyword", lang):
            return T.SOUND_EFFECT, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.INTERACTIVE_FICTION) and m(q, "InteractiveFictionKeyword", lang):
            return T.INTERACTIVE_FICTION, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.PLAYLIST) and m(q, "PlaylistKeyword", lang):
            return T.PLAYLIST, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.MOVIE) and m(q, "DocumentaryKeyword", lang):
            return T.MOVIE, [], DEFAULT_KEYWORD_CONFIDENCE
        if allow(T.AUDIOBOOK) and m(q, "AudioBookKeyword", lang):
            return T.AUDIOBOOK, [], DEFAULT_KEYWORD_CONFIDENCE
        # BOOK (TTS-read text) vs AUDIOBOOK (play a narration): a book cue
        # ("book"/"novel") gated by a read verb routes to BOOK, *after* the more
        # specific audiobook cue above so "play the audiobook" stays AUDIOBOOK.
        if allow(T.BOOK) and m(q, "BookKeyword", lang) and m(q, "VerbRead", lang):
            return T.BOOK, [], DEFAULT_KEYWORD_CONFIDENCE
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
        # Supplementary / promotional content — all collapse onto the parent
        # MOVIE type; the distinguishing form (trailer / teaser / BTS / making_of
        # / bloopers / …) rides the orthogonal mediavocab.ContentForm axis
        # (see ``classify_content_form`` / ``LABEL_TO_CONTENT_FORM``), so even a
        # title-less "show me bloopers" resolves to MOVIE + ContentForm.BEHIND_SCENES.
        for _supp_voc in ("TrailerKeyword", "TeaserKeyword",
                          "BehindTheScenesKeyword", "MakingOfKeyword",
                          "BloopersKeyword", "DeletedScenesKeyword",
                          "FeaturetteKeyword", "InterviewKeyword",
                          "ClipKeyword"):
            if allow(T.MOVIE) and m(q, _supp_voc, lang):
                return T.MOVIE, [], DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if allow(T.COMIC) and m(q, "ComicBookKeyword", lang):
            return T.COMIC, [], DEFAULT_KEYWORD_LOW_CONFIDENCE
        if allow(T.GAME) and self._game_evidence(q, lang):
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
