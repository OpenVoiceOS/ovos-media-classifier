"""Categorical feature extraction for guided-embeddings inference.

Produces the same 98-column sparse feature dict at runtime that
``training.generate_categorical_features`` produces during
dataset generation.  Absent features are omitted (sparse); present features
have value ``"1"`` (string, matching ``CategoricalVectorizer`` convention).

Feature sources (mirrors generate_categorical_features.py):
  - 41 keyword features  — columns via ``_VocMatcher.match()``
  - 57+ NER entity features — ``OCPEntityLabel`` values via AhocorasickNER

Constants ``_KEYWORD_VOCABS`` and ``_ENTITY_LABEL_VALUES`` defined here are
the single source of truth; ``train/generate_categorical_features.py`` imports
them from this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ovos_media_classifier.intents import OCPEntityLabel
from ovos_media_classifier.keyword import _LOCALE_DIR, _VocMatcher

if TYPE_CHECKING:
    from ovos_media_classifier.entities import EntitiesContainer

# ---------------------------------------------------------------------------
# Module-level constants — single source of truth; imported by
# training/generate_categorical_features.py
# ---------------------------------------------------------------------------

_KEYWORD_LABEL_VALUES: frozenset = frozenset(
    e.value for e in OCPEntityLabel
    if e.value in {
        "music", "podcast", "radio", "audiobook", "news", "movie", "tv",
        "tv_show", "video", "video_episodes", "audio", "game", "anime",
        "cartoon", "documentary", "short_film", "silent_movie", "bw_movie",
        "radio_theatre", "visual_story", "asmr", "audio_description",
        "music_video", "trailer", "behind_the_scenes", "adult",
        "adult_audio", "hentai",
    }
)

# NER labels that correspond to actual named entities (not keyword categories)
_ENTITY_LABEL_VALUES: List[str] = [
    e.value for e in OCPEntityLabel if e.value not in _KEYWORD_LABEL_VALUES
]

# Keyword vocab files → column names (order matches training)
_KEYWORD_VOCABS: List[Tuple[str, str]] = [
    ("DocumentaryKeyword",     "kw_documentary"),
    ("AudioBookKeyword",       "kw_audiobook"),
    ("NewsKeyword",            "kw_news"),
    ("AnimeKeyword",           "kw_anime"),
    ("CartoonKeyword",         "kw_cartoon"),
    ("PodcastKeyword",         "kw_podcast"),
    ("AudioDramaKeyword",      "kw_audio_drama"),
    ("RadioKeyword",           "kw_radio"),
    ("MusicVideoKeyword",      "kw_music_video"),
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
    """Extracts the sparse categorical feature dict from one utterance.

    Mirrors ``generate_categorical_features.py`` for runtime use.
    Keys are feature names; value is always ``"1"`` (absent features omitted).

    Args:
        voc_matcher: ``_VocMatcher`` instance, or None (keyword features skipped).
        entities_container: ``EntitiesContainer`` for NER, or None (NER skipped).
        lang: Default BCP-47 language tag.
    """

    def __init__(
        self,
        voc_matcher: Optional[_VocMatcher] = None,
        entities_container: Optional["EntitiesContainer"] = None,
        lang: str = "en",
    ) -> None:
        self._matcher = voc_matcher
        self._entities = entities_container
        self._lang = lang

    def extract(
        self,
        utterance: str,
        lang: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return sparse dict ``{feature_name: "1"}`` for each fired feature.

        Args:
            utterance: Raw user utterance.
            lang: BCP-47 language tag; falls back to ``self._lang``.

        Returns:
            Sparse dict with only the features that fired.
        """
        effective_lang = lang or self._lang
        feat: Dict[str, str] = {}

        # Keyword features
        if self._matcher is not None:
            for vocab_name, col_name in _KEYWORD_VOCABS:
                if self._matcher.match(utterance, vocab_name, effective_lang):
                    feat[col_name] = "1"

        # NER entity features
        if self._entities is not None:
            try:
                for hit in self._entities.ner.tag(utterance):
                    label = hit.get("label")
                    if label and label in _ENTITY_LABEL_VALUES:
                        feat[label] = "1"
            except Exception:
                pass

        return feat

    @classmethod
    def from_container(
        cls,
        container: "EntitiesContainer",
        lang: str = "en",
    ) -> "CategoricalFeatureExtractor":
        """Build extractor from a loaded ``EntitiesContainer``.

        Args:
            container: Populated ``EntitiesContainer`` with NER data.
            lang: Default language tag.
        """
        matcher = _VocMatcher(_LOCALE_DIR)
        return cls(voc_matcher=matcher, entities_container=container, lang=lang)

    @classmethod
    def from_locale_dir(
        cls,
        locale_dir: Optional[str] = None,
        lang: str = "en",
    ) -> "CategoricalFeatureExtractor":
        """Build keyword-only extractor (no NER) from locale directory.

        Args:
            locale_dir: Path to ``locale/`` dir with ``.voc`` files.
                        Defaults to bundled locale.
            lang: Default language tag.
        """
        matcher = _VocMatcher(locale_dir or _LOCALE_DIR)
        return cls(voc_matcher=matcher, entities_container=None, lang=lang)
