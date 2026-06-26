"""Categorical feature extraction for the ONNX trained backend.

Produces a **sparse categorical feature dict** at runtime from one utterance.
Absent features are omitted (sparse); present features have value ``"1"``
(string).  The :class:`~ovos_media_classifier.onnx.OnnxMediaClassifier`
vectorizes this dict (in the model's ``feature_names`` order) before running
the ONNX heads.

This release ships the **pure-python keyword path only** — features come from
the bundled ``.voc`` files via :class:`~ovos_media_classifier.keyword._VocMatcher`
(word-boundary matching backed by ``ovos-spec-tools``).
A richer NER feature path (Aho-Corasick over an ``EntitiesContainer``) is *not*
part of this PR (``entities.py`` is not shipped); the extractor is written so it
simply contributes no NER features when no entities backend is supplied.

``_KEYWORD_VOCABS`` here is the single source of truth for the keyword feature
columns and their stable order; a trained model bundle records the column order
it was trained on in ``meta.json`` (``feature_names``), so the runtime vectorizer
never has to assume an order — it reads it from the bundle.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ovos_media_classifier.keyword import _LOCALE_DIR, _VocMatcher

# ---------------------------------------------------------------------------
# Keyword vocab files → feature column names (stable order).
# A trained bundle records its own ``feature_names`` order in meta.json; this
# table is the menu of keyword columns the pure-python extractor can produce.
# ---------------------------------------------------------------------------
_KEYWORD_VOCABS: List[Tuple[str, str]] = [
    ("DocumentaryKeyword",     "kw_documentary"),
    ("AudioBookKeyword",       "kw_audiobook"),
    ("BookKeyword",            "kw_book"),
    ("NewsKeyword",            "kw_news"),
    ("AnimeKeyword",           "kw_anime"),
    ("CartoonKeyword",         "kw_cartoon"),
    ("PodcastKeyword",         "kw_podcast"),
    ("AudioDramaKeyword",      "kw_audio_drama"),
    ("RadioKeyword",           "kw_radio"),
    ("MusicVideoKeyword",      "kw_music_video"),
    ("SoundtrackKeyword",      "kw_soundtrack"),
    ("MusicKeyword",           "kw_music"),
    ("IPTVKeyword",            "kw_iptv"),
    ("TVKeyword",              "kw_tv"),
    ("SeriesKeyword",          "kw_series"),
    ("MovieKeyword",           "kw_movie"),
    ("ShortKeyword",           "kw_short"),
    ("SilentKeyword",          "kw_silent"),
    ("BWKeyword",              "kw_bw"),
    ("TrailerKeyword",         "kw_trailer"),
    ("BehindTheScenesKeyword", "kw_bts"),
    ("ComicBookKeyword",       "kw_comic"),
    ("GameKeyword",            "kw_game"),
    ("ADKeyword",              "kw_ad"),
    ("ASMRKeyword",            "kw_asmr"),
    ("AdultKeyword",           "kw_adult"),
    ("HentaiKeyword",          "kw_hentai"),
    ("PlaylistKeyword",        "kw_playlist"),
    ("SoundEffectKeyword",     "kw_sound_effect"),
    ("InteractiveFictionKeyword", "kw_interactive_fiction"),
    ("AmbientKeyword",         "kw_ambient"),
    ("VideoKeyword",           "kw_video"),
    ("AudioKeyword",           "kw_audio"),
    # Linguistically-motivated verb and discourse features
    ("VerbAudio",              "verb_audio"),
    ("VerbVideo",              "verb_video"),
    ("VerbGame",               "verb_game"),
    ("VerbRead",               "verb_read"),
    ("VerbTune",               "verb_tune"),
    ("AttrTopic",              "attr_topic"),
    ("AttrStarring",           "attr_starring"),
    ("ModEpisode",             "mod_episode"),
    ("ModSeason",              "mod_season"),
    ("ModLive",                "mod_live"),
    ("ModContinue",            "mod_continue"),
    ("ModLatest",              "mod_latest"),
    ("audio_only",             "fmt_audio_only"),
    ("video_only",             "fmt_video_only"),
]


class CategoricalFeatureExtractor:
    """Extract a sparse categorical feature dict from one utterance.

    Keys are feature names; the value is always ``"1"`` (absent features are
    omitted).  Keyword features come from the bundled ``.voc`` files; the table
    of available columns is :data:`_KEYWORD_VOCABS`.

    Args:
        voc_matcher: a :class:`~ovos_media_classifier.keyword._VocMatcher`
            instance, or ``None`` to skip keyword features entirely.
        lang: default BCP-47 language tag.
    """

    def __init__(
        self,
        voc_matcher: Optional[_VocMatcher] = None,
        lang: str = "en",
    ) -> None:
        self._matcher = voc_matcher
        self._lang = lang

    def extract(
        self,
        utterance: str,
        lang: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return the sparse dict ``{feature_name: "1"}`` of fired features.

        Args:
            utterance: raw user utterance.
            lang: BCP-47 language tag; falls back to ``self._lang``.

        Returns:
            Sparse dict containing only the features that fired.
        """
        effective_lang = lang or self._lang
        feat: Dict[str, str] = {}

        if self._matcher is not None:
            for vocab_name, col_name in _KEYWORD_VOCABS:
                try:
                    if self._matcher.match(utterance, vocab_name, effective_lang):
                        feat[col_name] = "1"
                except Exception:
                    # a missing/unreadable .voc file simply yields no feature
                    continue

        return feat

    @classmethod
    def from_locale_dir(
        cls,
        locale_dir: Optional[str] = None,
        lang: str = "en",
    ) -> "CategoricalFeatureExtractor":
        """Build a keyword-only extractor reading ``.voc`` files from disk.

        Args:
            locale_dir: path to a ``locale/`` dir with per-language ``.voc``
                subdirs.  Defaults to the bundled locale.
            lang: default language tag.
        """
        matcher = _VocMatcher(locale_dir or _LOCALE_DIR)
        return cls(voc_matcher=matcher, lang=lang)
