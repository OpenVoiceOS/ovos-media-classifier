"""Keyword-based media classifier.

This is a direct extraction of the voc_match_media() logic that lived inside
OCPPipelineMatcher.  It can work in two modes:

1. **Pipeline mode** – pass ``voc_match_func`` (typically
   ``OVOSAbstractApplication.voc_match`` bound to the pipeline plugin).
   The pipeline plugin owns the locale files; the classifier just calls
   the function.

2. **Standalone mode** – omit ``voc_match_func`` and the classifier reads
   the bundled ``.voc`` files from
   ``ovos_media_classifier/locale/<lang>/<VocabName>.voc`` directly.
   Use ``KeywordMediaClassifier.from_locale_dir(locale_dir, lang)`` or
   simply ``KeywordMediaClassifier()`` to use the built-in locale files.

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

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.constants import (
    DEFAULT_KEYWORD_CONFIDENCE,
    DEFAULT_KEYWORD_HIGH_CONFIDENCE,
    DEFAULT_KEYWORD_LOW_CONFIDENCE,
)
from ovos_media_classifier.intents import (
    MediaType,
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


class KeywordMediaClassifier(AbstractMediaClassifier):
    """Classifies media type using vocabulary keyword matching.

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

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Keyword-based classification mirroring the original voc_match_media.

        Checks vocabulary files in priority order.  Returns the first match
        that maps to a ``mediavocab.MediaType`` in *valid_labels* (or any match
        when valid_labels is None).  Falls back to (GENERIC, 0.0).
        """
        intent, conf = self._classify_intent(query, lang, valid_labels)
        return PLAY_INTENT_TO_MEDIA_TYPE.get(intent, MediaType.GENERIC), conf

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Return mediavocab genre tags implied by the winning keyword intent.

        This is what the content filter blocks on (e.g. an adult or anime
        query surfaces ``["adult"]`` / ``["anime"]`` even though the public
        ``MediaType`` is a generic ``MOVIE`` / ``EPISODIC_SERIES``).
        """
        intent, _ = self._classify_intent(query, lang, None)
        return list(PLAY_INTENT_TO_GENRES.get(intent, []))

    def _classify_intent(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]],
    ) -> Tuple[OCPPlayIntent, float]:
        """Resolve the fine-grained play intent in priority order.

        Mirrors the original voc_match_media chain but yields an
        ``OCPPlayIntent`` so both the public ``MediaType`` and the genre tags
        can be derived from one pass.  ``valid_labels`` (mediavocab types) gate
        each branch via the intent→type map.
        """
        valid = set(valid_labels) if valid_labels is not None else None
        I = OCPPlayIntent

        def _ok(intent: OCPPlayIntent) -> bool:
            return valid is None or PLAY_INTENT_TO_MEDIA_TYPE.get(intent) in valid

        m = self._match
        q = query

        # Adult family FIRST — content filtering is a safety feature, so an adult
        # keyword must take precedence over substring collisions (e.g. German
        # "sexfilm" contains "film", which would otherwise classify as a movie).
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

        if _ok(I.DOCUMENTARY) and m(q, "DocumentaryKeyword", lang):
            return I.DOCUMENTARY, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.AUDIOBOOK) and m(q, "AudioBookKeyword", lang):
            return I.AUDIOBOOK, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.NEWS) and m(q, "NewsKeyword", lang):
            return I.NEWS, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.ANIME) and m(q, "AnimeKeyword", lang):
            return I.ANIME, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.CARTOON) and m(q, "CartoonKeyword", lang):
            return I.CARTOON, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.PODCAST) and m(q, "PodcastKeyword", lang):
            return I.PODCAST, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.RADIO_THEATRE) and m(q, "AudioDramaKeyword", lang):
            # NOTE: must come before plain RADIO so "radio theatre" wins
            return I.RADIO_THEATRE, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.RADIO) and m(q, "RadioKeyword", lang):
            return I.RADIO, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.MUSIC_VIDEO) and m(q, "MusicVideoKeyword", lang):
            # NOTE: must come before MusicKeyword (music video is more specific)
            return I.MUSIC_VIDEO, DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if _ok(I.MUSIC) and m(q, "MusicKeyword", lang):
            # NOTE: must come before MOVIE to handle "{movie} soundtrack"
            return I.MUSIC, DEFAULT_KEYWORD_CONFIDENCE
        # IPTVKeyword (live channel/stream) takes priority over generic TVKeyword
        if _ok(I.TV) and m(q, "IPTVKeyword", lang):
            return I.TV, DEFAULT_KEYWORD_CONFIDENCE
        # SeriesKeyword (e.g. "tv show", "episode") before the generic TVKeyword,
        # else "tv show" would match "tv" and be classified as a live channel.
        if _ok(I.VIDEO_EPISODES) and m(q, "SeriesKeyword", lang):
            return I.VIDEO_EPISODES, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.TV) and m(q, "TVKeyword", lang):
            return I.TV, DEFAULT_KEYWORD_CONFIDENCE

        # Movie family
        movie_intents = {I.MOVIE, I.SHORT_FILM, I.SILENT_MOVIE, I.BW_MOVIE}
        if any(_ok(t) for t in movie_intents) and m(q, "MovieKeyword", lang):
            if _ok(I.SHORT_FILM) and m(q, "ShortKeyword", lang):
                return I.SHORT_FILM, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _ok(I.SILENT_MOVIE) and m(q, "SilentKeyword", lang):
                return I.SILENT_MOVIE, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _ok(I.BW_MOVIE) and m(q, "BWKeyword", lang):
                return I.BW_MOVIE, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _ok(I.MOVIE):
                return I.MOVIE, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(I.TRAILER) and m(q, "TrailerKeyword", lang):
            return I.TRAILER, DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if _ok(I.BEHIND_THE_SCENES) and m(q, "BehindTheScenesKeyword", lang):
            return I.BEHIND_THE_SCENES, DEFAULT_KEYWORD_HIGH_CONFIDENCE

        if _ok(I.VISUAL_STORY) and m(q, "ComicBookKeyword", lang):
            return I.VISUAL_STORY, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(I.GAME) and m(q, "GameKeyword", lang):
            return I.GAME, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(I.AUDIO_DESCRIPTION) and m(q, "ADKeyword", lang):
            return I.AUDIO_DESCRIPTION, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(I.ASMR) and m(q, "ASMRKeyword", lang):
            return I.ASMR, DEFAULT_KEYWORD_LOW_CONFIDENCE

        if _ok(I.VIDEO) and m(q, "VideoKeyword", lang):
            return I.VIDEO, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(I.AUDIO) and m(q, "AudioKeyword", lang):
            return I.AUDIO, DEFAULT_KEYWORD_LOW_CONFIDENCE

        return I.GENERIC, 0.0
