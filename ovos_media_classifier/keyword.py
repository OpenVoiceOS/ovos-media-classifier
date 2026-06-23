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

2. **Standalone mode** – omit ``voc_match_func`` and the classifier reads
   the bundled ``.voc`` files from
   ``ovos_media_classifier/locale/<lang>/<VocabName>.voc`` directly.
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
from typing import Callable, Dict, List, Optional, Set, Tuple

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
    OCPPlayIntent,
    PLAY_INTENT_TO_MEDIA_TYPE,
    PLAY_INTENT_TO_GENRES,
)

# Bundled locale directory (ported from ovos-ocp-pipeline-plugin)
_LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")


class _VocMatcher:
    """Simple vocabulary matcher backed by ``.voc`` files on disk.

    Each ``.voc`` file contains one keyword or phrase per line.  A phrase
    matches if any entry appears as a substring of the (lowercased) query.
    Vocabulary files are loaded lazily and cached per (lang, vocab_name).
    """

    def __init__(self, locale_dir: str) -> None:
        self._locale_dir = locale_dir
        self._cache: Dict[Tuple[str, str], Set[str]] = {}
        # case-insensitive index of available locale subdirs: "en-us" -> "en-US"
        self._dirs: Dict[str, str] = {}
        try:
            for name in os.listdir(locale_dir):
                if os.path.isdir(os.path.join(locale_dir, name)):
                    self._dirs[name.lower()] = name
        except OSError:
            pass

    def _resolve_dir(self, lang_tag: str) -> Optional[str]:
        actual = self._dirs.get(lang_tag.lower())
        return os.path.join(self._locale_dir, actual) if actual else None

    def _load(self, vocab_name: str, lang: str) -> Set[str]:
        key = (lang.lower(), vocab_name)
        if key not in self._cache:
            words: Set[str] = set()
            # Try exact lang tag first, then language-only fallback (e.g. "en"),
            # resolving the locale subdir case-insensitively (en-us vs en-US).
            lang_candidates = [lang.lower(), lang.lower().split("-")[0]]
            for lang_tag in lang_candidates:
                lang_dir = self._resolve_dir(lang_tag)
                if not lang_dir:
                    continue
                voc_path = os.path.join(lang_dir, f"{vocab_name}.voc")
                if os.path.isfile(voc_path):
                    with open(voc_path, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                words.add(line.lower())
                    break
            self._cache[key] = words
        return self._cache[key]

    def match(self, phrase: str, vocab_name: str, lang: str) -> bool:
        phrase_lower = phrase.lower()
        for word in self._load(vocab_name, lang):
            if word in phrase_lower:
                return True
        return False


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
        intent, conf = self._classify_intent(query, lang, valid_labels)
        return PLAY_INTENT_TO_MEDIA_TYPE.get(intent, MediaType.GENERIC), conf

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Domain axis: ocp_play when any media evidence matches, else not_ocp."""
        media_type, conf = self.classify(query, lang)
        if media_type != MediaType.GENERIC:
            return OCPDomain.OCP_PLAY, conf
        return OCPDomain.NOT_OCP, 0.0

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Return mediavocab genre tags implied by the winning keyword intent.

        This is what the content filter blocks on (e.g. an adult or anime
        query surfaces ``["adult"]`` / ``["anime"]`` even though the public
        ``MediaType`` is a generic ``MOVIE`` / ``EPISODIC_SERIES``).  Adult
        precedence is preserved so adult/hentai/porn cues are never lost.
        """
        intent, _ = self._classify_intent(query, lang, None)
        return list(PLAY_INTENT_TO_GENRES.get(intent, []))

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
        """
        modality, _ = self._predict_modality(query, lang)
        structure = self._predict_structure(query, lang, modality)
        intent, conf = self._classify_intent(query, lang, None)
        media_type = PLAY_INTENT_TO_MEDIA_TYPE.get(intent, MediaType.GENERIC)
        genres = list(PLAY_INTENT_TO_GENRES.get(intent, []))

        if media_type == MediaType.GENERIC:
            domain = OCPDomain.NOT_OCP
            playback = PlaybackType.UNKNOWN
            structure = Structure.UNKNOWN
        else:
            domain = OCPDomain.OCP_PLAY
            # Prefer the predicted coarse axes; degrade to the leaf's intrinsic
            # axis where no coarse evidence was found (graceful degradation).
            playback = (modality if modality is not PlaybackType.UNKNOWN
                        else infer_playback_type(media_type))
            if structure is Structure.UNKNOWN:
                structure = infer_structure(media_type)

        return MediaClassification(
            media_type=media_type,
            playback_type=playback,
            structure=structure,
            domain=domain,
            genres=genres,
            confidence=conf,
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

    def _classify_intent(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]],
    ) -> Tuple[OCPPlayIntent, float]:
        """Resolve the play intent coarse-to-fine.

        1. Adult family FIRST (safety: content filtering must never be bypassed
           by a substring collision such as German "sexfilm" ⊃ "film").
        2. Predict modality, then structure.
        3. Constrain the leaf candidates to those axes and run the specific-leaf
           voc chain *gated to the candidate set*; the first match wins.
        4. If no specific leaf matched but the coarse axes are confident, emit
           the DEFAULT leaf for that (modality, structure) cell.

        ``valid_labels`` (mediavocab types) gate every branch via the
        intent→type map, exactly as before.
        """
        valid = set(valid_labels) if valid_labels is not None else None
        I = OCPPlayIntent
        m = self._match
        q = query

        def _ok(intent: OCPPlayIntent) -> bool:
            return valid is None or PLAY_INTENT_TO_MEDIA_TYPE.get(intent) in valid

        # --- 1. Adult family (precedence preserved) ---------------------------
        adult_intents = {I.ADULT, I.HENTAI, I.ADULT_AUDIO}
        if any(_ok(t) for t in adult_intents) and (
                m(q, "AdultKeyword", lang) or m(q, "HentaiKeyword", lang)):
            if (_ok(I.HENTAI) and
                    (m(q, "HentaiKeyword", lang) or
                     m(q, "CartoonKeyword", lang) or
                     m(q, "AnimeKeyword", lang))):
                return I.HENTAI, DEFAULT_KEYWORD_LOW_CONFIDENCE
            if (_ok(I.ADULT_AUDIO) and
                    (m(q, "AudioKeyword", lang) or m(q, "ASMRKeyword", lang))):
                return I.ADULT_AUDIO, DEFAULT_KEYWORD_LOW_CONFIDENCE
            if _ok(I.ADULT):
                return I.ADULT, DEFAULT_KEYWORD_LOW_CONFIDENCE

        # --- 2. Predict the coarse axes ---------------------------------------
        modality, _strength = self._predict_modality(q, lang)
        structure = self._predict_structure(q, lang, modality)
        candidates = self._candidate_media_types(modality, structure)

        def _in_set(intent: OCPPlayIntent) -> bool:
            """The intent's leaf MediaType is within the constrained candidate set."""
            if candidates is None:
                return True
            return PLAY_INTENT_TO_MEDIA_TYPE.get(intent) in candidates

        def _allow(intent: OCPPlayIntent) -> bool:
            return _ok(intent) and _in_set(intent)

        # --- 3. Constrained specific-leaf chain -------------------------------
        # Ordering mirrors the original priority chain (most-specific first) but
        # every branch is now additionally gated by ``_in_set`` so a leaf whose
        # modality/structure disagrees with the predicted axes cannot fire.
        if _allow(I.DOCUMENTARY) and m(q, "DocumentaryKeyword", lang):
            return I.DOCUMENTARY, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.AUDIOBOOK) and m(q, "AudioBookKeyword", lang):
            return I.AUDIOBOOK, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.NEWS) and m(q, "NewsKeyword", lang):
            return I.NEWS, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.ANIME) and m(q, "AnimeKeyword", lang):
            return I.ANIME, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.CARTOON) and m(q, "CartoonKeyword", lang):
            return I.CARTOON, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.PODCAST) and m(q, "PodcastKeyword", lang):
            return I.PODCAST, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.RADIO_THEATRE) and m(q, "AudioDramaKeyword", lang):
            # must come before plain RADIO so "radio theatre" wins
            return I.RADIO_THEATRE, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.RADIO) and m(q, "RadioKeyword", lang):
            return I.RADIO, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.MUSIC_VIDEO) and m(q, "MusicVideoKeyword", lang):
            # must come before MusicKeyword (music video is more specific)
            return I.MUSIC_VIDEO, DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if _allow(I.MUSIC) and m(q, "MusicKeyword", lang):
            # must come before MOVIE to handle "{movie} soundtrack"
            return I.MUSIC, DEFAULT_KEYWORD_CONFIDENCE
        # IPTVKeyword (live channel/stream) takes priority over generic TVKeyword
        if _allow(I.TV) and m(q, "IPTVKeyword", lang):
            return I.TV, DEFAULT_KEYWORD_CONFIDENCE
        # SeriesKeyword (e.g. "tv show", "episode") before the generic TVKeyword,
        # else "tv show" would match "tv" and be classified as a live channel.
        if _allow(I.VIDEO_EPISODES) and m(q, "SeriesKeyword", lang):
            return I.VIDEO_EPISODES, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.TV) and m(q, "TVKeyword", lang):
            return I.TV, DEFAULT_KEYWORD_CONFIDENCE

        # Movie family
        movie_intents = {I.MOVIE, I.SHORT_FILM, I.SILENT_MOVIE, I.BW_MOVIE}
        if any(_allow(t) for t in movie_intents) and m(q, "MovieKeyword", lang):
            if _allow(I.SHORT_FILM) and m(q, "ShortKeyword", lang):
                return I.SHORT_FILM, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _allow(I.SILENT_MOVIE) and m(q, "SilentKeyword", lang):
                return I.SILENT_MOVIE, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _allow(I.BW_MOVIE) and m(q, "BWKeyword", lang):
                return I.BW_MOVIE, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _allow(I.MOVIE):
                return I.MOVIE, DEFAULT_KEYWORD_CONFIDENCE
        if _allow(I.TRAILER) and m(q, "TrailerKeyword", lang):
            return I.TRAILER, DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if _allow(I.BEHIND_THE_SCENES) and m(q, "BehindTheScenesKeyword", lang):
            return I.BEHIND_THE_SCENES, DEFAULT_KEYWORD_HIGH_CONFIDENCE

        if _allow(I.VISUAL_STORY) and m(q, "ComicBookKeyword", lang):
            return I.VISUAL_STORY, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _allow(I.GAME) and m(q, "GameKeyword", lang):
            return I.GAME, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _allow(I.AUDIO_DESCRIPTION) and m(q, "ADKeyword", lang):
            return I.AUDIO_DESCRIPTION, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _allow(I.ASMR) and m(q, "ASMRKeyword", lang):
            return I.ASMR, DEFAULT_KEYWORD_LOW_CONFIDENCE

        if _allow(I.VIDEO) and m(q, "VideoKeyword", lang):
            return I.VIDEO, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _allow(I.AUDIO) and m(q, "AudioKeyword", lang):
            return I.AUDIO, DEFAULT_KEYWORD_LOW_CONFIDENCE

        # --- 4. Default leaf for the (modality, structure) cell ---------------
        # No specific leaf voc matched inside the constrained set, but the coarse
        # axes carry evidence — emit the sensible default leaf so a strong
        # modality verb ("listen", "watch") still resolves to *something*.
        if modality is not PlaybackType.UNKNOWN:
            default = _DEFAULT_LEAF.get(
                (modality, structure),
                _MODALITY_DEFAULT_LEAF.get(modality),
            )
            if default is not None:
                intent = self._intent_for_default(default)
                if intent is not None and _ok(intent):
                    return intent, DEFAULT_KEYWORD_LOW_CONFIDENCE

        return I.GENERIC, 0.0

    @staticmethod
    def _intent_for_default(media_type: MediaType) -> Optional[OCPPlayIntent]:
        """Pick a representative play intent for a default leaf MediaType."""
        from ovos_media_classifier.intents import MEDIA_TYPE_TO_PLAY_INTENT
        intent = MEDIA_TYPE_TO_PLAY_INTENT.get(media_type)
        if intent is not None:
            return intent
        # BOOK has no dedicated OCPPlayIntent — closest paged intent is the comic
        # (visual story); audiobook would change the modality, so use VISUAL_STORY.
        if media_type in (MediaType.BOOK, MediaType.COMIC):
            return OCPPlayIntent.VISUAL_STORY
        return None
