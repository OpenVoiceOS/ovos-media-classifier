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

import re
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
    ("TeaserKeyword",          "kw_teaser"),
    ("BehindTheScenesKeyword", "kw_bts"),
    ("MakingOfKeyword",        "kw_making_of"),
    ("BloopersKeyword",        "kw_bloopers"),
    ("DeletedScenesKeyword",   "kw_deleted_scenes"),
    ("FeaturetteKeyword",      "kw_featurette"),
    ("InterviewKeyword",       "kw_interview"),
    ("ClipKeyword",            "kw_clip"),
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


# ---------------------------------------------------------------------------
# Per-VALUE keyword features for the bounded categorical axes.
#
# The plain keyword vocabs above fire ``ner_music_genre=1`` / ``kw_music=1`` —
# they signal *that* a genre was named but not *which* one, so the genre head
# cannot tell metal from jazz.  This table adds ONE feature **per canonical
# value** (``kw_genre_horror``) that fires when any of its surface words appears
# in the utterance (word-boundary).  The value space is **bounded and curated**
# to mediavocab ``KNOWN_GENRES`` so the feature set stays small and the
# classifier never emits a tag the taxonomy does not recognise — see
# docs/model.md.  Open / unseen genres are out of scope for this path (the
# semantic backend handles those — the next rung).
#
# The table maps ``canonical_value -> (surface words / synonyms)``.  Words are
# matched case-insensitively on word boundaries; the canonical value is what the
# genre head learns to predict, so it MUST be a ``KNOWN_GENRES`` value.
# ---------------------------------------------------------------------------

GENRE_VALUE_VOCAB: Dict[str, Tuple[str, ...]] = {
    # --- film / tv / general ---
    "action": ("action",),
    "adventure": ("adventure",),
    "comedy": ("comedy", "comedic", "funny"),
    "drama": ("drama", "dramatic"),
    "horror": ("horror", "scary", "slasher"),
    "thriller": ("thriller", "suspense"),
    "sci_fi": ("sci fi", "sci-fi", "scifi", "science fiction"),
    "fantasy": ("fantasy",),
    "romance": ("romance", "romantic", "rom com", "romcom"),
    "mystery": ("mystery",),
    "crime": ("crime", "gangster"),
    "western": ("western",),
    "war": ("war",),
    "biography": ("biography", "biopic", "biographical"),
    "historical": ("history", "historical"),
    "musical": ("musical",),
    "family": ("family",),
    "animation": ("animation", "animated"),
    "anime": ("anime",),
    "noir": ("noir", "film noir"),
    # --- music ---
    "rock": ("rock",),
    "metal": ("metal", "heavy metal", "death metal"),
    "jazz": ("jazz",),
    "pop": ("pop",),
    "hip_hop": ("hip hop", "hip-hop", "hiphop", "rap"),
    "classical": ("classical",),
    "electronic": ("electronic", "edm", "electronica"),
    "techno": ("techno",),
    "house": ("house music",),
    "trance": ("trance",),
    "dubstep": ("dubstep",),
    "drum_and_bass": ("drum and bass", "dnb", "drum n bass"),
    "blues": ("blues",),
    "country": ("country music",),
    "folk": ("folk",),
    "reggae": ("reggae",),
    "punk": ("punk",),
    "funk": ("funk",),
    "soul": ("soul",),
    "rnb": ("rnb", "r and b", "r&b", "rhythm and blues"),
    "disco": ("disco",),
    "indie": ("indie",),
    "gospel": ("gospel",),
    "latin": ("latin",),
    # --- spoken / other ---
    "true_crime": ("true crime",),
    "educational": ("educational", "education"),
    "cooking": ("cooking",),
    "nature": ("nature",),
    "travel": ("travel",),
}

# (axis prefix, value-vocab table) — the menu of per-value feature columns.
# Only the genre value-vocab remains: mood / era are no longer modelled axes
# (dropped from the taxonomy in the mediavocab-axes alignment).
_VALUE_VOCABS: List[Tuple[str, Dict[str, Tuple[str, ...]]]] = [
    ("kw_genre", GENRE_VALUE_VOCAB),
]


def _value_feature_columns() -> List[str]:
    """Ordered ``kw_genre_*`` feature column names."""
    cols: List[str] = []
    for prefix, table in _VALUE_VOCABS:
        for value in table:
            cols.append(f"{prefix}_{value}")
    return cols


# the full stable list of per-value feature columns (genre + mood + era)
VALUE_FEATURE_COLS: List[str] = _value_feature_columns()


def _compile_value_matchers() -> List[Tuple[str, "re.Pattern"]]:
    """Compile ``(feature_name, word-boundary regex)`` for every curated value.

    Longer surface words are matched first within each value's alternation so a
    multi-word form ("science fiction") is preferred over a fragment.
    """
    out: List[Tuple[str, re.Pattern]] = []
    for prefix, table in _VALUE_VOCABS:
        for value, words in table.items():
            ordered = sorted(words, key=len, reverse=True)
            alt = "|".join(re.escape(w) for w in ordered)
            rx = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
            out.append((f"{prefix}_{value}", rx))
    return out


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

    # compiled once per process — the per-value vocabs are language-agnostic
    # (genre surface words are matched the same across locales).
    _VALUE_MATCHERS: List[Tuple[str, "re.Pattern"]] = _compile_value_matchers()

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

        # Per-VALUE genre features (curated bounded vocab ⊆ KNOWN_GENRES) — these
        # tell the genre head *which* value was named, not just that some genre
        # word occurred.  Independent of the .voc matcher so they fire even when
        # ``voc_matcher`` is None.
        for col_name, rx in self._VALUE_MATCHERS:
            if rx.search(utterance):
                feat[col_name] = "1"

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
