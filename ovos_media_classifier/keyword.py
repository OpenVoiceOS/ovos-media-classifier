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
from ovos_media_classifier.intents import MediaType

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

    def _load(self, vocab_name: str, lang: str) -> Set[str]:
        key = (lang.lower(), vocab_name)
        if key not in self._cache:
            words: Set[str] = set()
            # Try exact lang tag first, then language-only fallback (e.g. "en")
            lang_candidates = [lang.lower(), lang.lower().split("-")[0]]
            for lang_tag in lang_candidates:
                voc_path = os.path.join(
                    self._locale_dir, lang_tag, f"{vocab_name}.voc"
                )
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
        that is also in *valid_labels* (or any match when valid_labels is None).
        Falls back to (GENERIC, 0.0) when nothing matches.
        """
        # When no restriction is given, all types are valid.
        valid = set(valid_labels) if valid_labels is not None else None

        def _ok(mt: MediaType) -> bool:
            return valid is None or mt in valid

        m = self._match
        q = query

        if _ok(MediaType.DOCUMENTARY) and m(q, "DocumentaryKeyword", lang):
            return MediaType.DOCUMENTARY, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.AUDIOBOOK) and m(q, "AudioBookKeyword", lang):
            return MediaType.AUDIOBOOK, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.NEWS) and m(q, "NewsKeyword", lang):
            return MediaType.NEWS, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.ANIME) and m(q, "AnimeKeyword", lang):
            return MediaType.ANIME, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.CARTOON) and m(q, "CartoonKeyword", lang):
            return MediaType.CARTOON, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.PODCAST) and m(q, "PodcastKeyword", lang):
            return MediaType.PODCAST, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.RADIO_THEATRE) and m(q, "AudioDramaKeyword", lang):
            # NOTE: must come before plain RADIO so "radio theatre" wins
            return MediaType.RADIO_THEATRE, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.RADIO) and m(q, "RadioKeyword", lang):
            return MediaType.RADIO, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.MUSIC_VIDEO) and m(q, "MusicVideoKeyword", lang):
            # NOTE: must come before MusicKeyword (music video is more specific)
            return MediaType.MUSIC_VIDEO, DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if _ok(MediaType.MUSIC) and m(q, "MusicKeyword", lang):
            # NOTE: must come before MOVIE to handle "{movie} soundtrack"
            return MediaType.MUSIC, DEFAULT_KEYWORD_CONFIDENCE
        # IPTVKeyword (live channel/stream) takes priority over generic TVKeyword
        if _ok(MediaType.TV) and m(q, "IPTVKeyword", lang):
            return MediaType.TV, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.TV) and m(q, "TVKeyword", lang):
            return MediaType.TV, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.VIDEO_EPISODES) and m(q, "SeriesKeyword", lang):
            return MediaType.VIDEO_EPISODES, DEFAULT_KEYWORD_CONFIDENCE

        # Movie family
        movie_types = {MediaType.MOVIE, MediaType.SHORT_FILM,
                       MediaType.SILENT_MOVIE, MediaType.BLACK_WHITE_MOVIE}
        if any(_ok(t) for t in movie_types) and m(q, "MovieKeyword", lang):
            if _ok(MediaType.SHORT_FILM) and m(q, "ShortKeyword", lang):
                return MediaType.SHORT_FILM, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _ok(MediaType.SILENT_MOVIE) and m(q, "SilentKeyword", lang):
                return MediaType.SILENT_MOVIE, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _ok(MediaType.BLACK_WHITE_MOVIE) and m(q, "BWKeyword", lang):
                return MediaType.BLACK_WHITE_MOVIE, DEFAULT_KEYWORD_HIGH_CONFIDENCE
            if _ok(MediaType.MOVIE):
                return MediaType.MOVIE, DEFAULT_KEYWORD_CONFIDENCE
        if _ok(MediaType.TRAILER) and m(q, "TrailerKeyword", lang):
            return MediaType.TRAILER, DEFAULT_KEYWORD_HIGH_CONFIDENCE
        if _ok(MediaType.BEHIND_THE_SCENES) and m(q, "BehindTheScenesKeyword", lang):
            return MediaType.BEHIND_THE_SCENES, DEFAULT_KEYWORD_HIGH_CONFIDENCE

        if _ok(MediaType.VISUAL_STORY) and m(q, "ComicBookKeyword", lang):
            return MediaType.VISUAL_STORY, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(MediaType.GAME) and m(q, "GameKeyword", lang):
            return MediaType.GAME, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(MediaType.AUDIO_DESCRIPTION) and m(q, "ADKeyword", lang):
            return MediaType.AUDIO_DESCRIPTION, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(MediaType.ASMR) and m(q, "ASMRKeyword", lang):
            return MediaType.ASMR, DEFAULT_KEYWORD_LOW_CONFIDENCE

        # Adult family
        adult_types = {MediaType.ADULT, MediaType.HENTAI, MediaType.ADULT_AUDIO}
        if any(_ok(t) for t in adult_types) and m(q, "AdultKeyword", lang):
            if (_ok(MediaType.HENTAI) and
                    (m(q, "CartoonKeyword", lang) or
                     m(q, "AnimeKeyword", lang) or
                     m(q, "HentaiKeyword", lang))):
                return MediaType.HENTAI, DEFAULT_KEYWORD_LOW_CONFIDENCE
            if (_ok(MediaType.ADULT_AUDIO) and
                    (m(q, "AudioKeyword", lang) or m(q, "ASMRKeyword", lang))):
                return MediaType.ADULT_AUDIO, DEFAULT_KEYWORD_LOW_CONFIDENCE
            if _ok(MediaType.ADULT):
                return MediaType.ADULT, DEFAULT_KEYWORD_LOW_CONFIDENCE

        if _ok(MediaType.HENTAI) and m(q, "HentaiKeyword", lang):
            return MediaType.HENTAI, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(MediaType.VIDEO) and m(q, "VideoKeyword", lang):
            return MediaType.VIDEO, DEFAULT_KEYWORD_LOW_CONFIDENCE
        if _ok(MediaType.AUDIO) and m(q, "AudioKeyword", lang):
            return MediaType.AUDIO, DEFAULT_KEYWORD_LOW_CONFIDENCE

        return MediaType.GENERIC, 0.0
