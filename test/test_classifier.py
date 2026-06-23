"""Unit tests for ovos-media-classifier.

Tests cover:
  - KeywordMediaClassifier.classify() — each branch of the if/elif chain
  - KeywordMediaClassifier.is_ocp_query() — inherited default implementation
  - Model2VecMediaClassifier.classify() — with a mocked model
  - Model2VecMediaClassifier.is_ocp_query() — fast domain-head path
  - AhocorasickMediaClassifier — NER-based classification with runtime updates
  - SklearnMediaClassifier — sklearn pipeline-based classification
  - PadatiousMediaClassifier — padatious/padacioso-based classification
  - EntitiesContainer loaders — Radarr, Sonarr, Lidarr, Jellyfin, etc.
  - load_media_classifier() factory — fallback behaviour
"""
import csv
import tempfile
import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import enum

import mediavocab
import numpy as np

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import (
    MediaType,
    OCPControlIntent,
    OCPEntityLabel,
    OCPPlayIntent,
    PLAY_INTENT_TO_MEDIA_TYPE,
)
from ovos_media_classifier.keyword import KeywordMediaClassifier
from ovos_media_classifier.m2v import (
    LABEL_TO_MEDIA_TYPE as INTENT_TO_MEDIA_TYPE,
    Model2VecMediaClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kw_clf(*match_vocabs: str) -> KeywordMediaClassifier:
    """Return a KeywordMediaClassifier whose voc_match returns True only for
    vocab names listed in *match_vocabs*."""

    def _voc_match(phrase: str, vocab: str, **kw) -> bool:
        return vocab in match_vocabs

    return KeywordMediaClassifier(_voc_match)


def _mock_m2v_model(
    domain: str = "ocp_play",
    intent: str = "music",
    domain_prob: float = 0.9,
    intent_prob: float = 0.8,
    domain_classes: Optional[List[str]] = None,
):
    """Build a MagicMock mimicking StaticModelForHierarchicalClassification."""
    model = MagicMock()
    model.predict.return_value = (
        np.array([domain]),
        np.array([intent]),
    )
    d_probs = np.zeros(3)
    d_idx = {"ocp_play": 0, "ocp_control": 1, "other": 2}.get(domain, 2)
    d_probs[d_idx] = domain_prob

    i_probs = np.zeros(5)
    i_probs[0] = intent_prob

    model.predict_proba.return_value = (
        np.array([d_probs]),
        np.array([i_probs]),
    )
    model.domain_classes_ = np.array(
        domain_classes or ["ocp_play", "ocp_control", "other"]
    )
    return model


# ---------------------------------------------------------------------------
# AbstractMediaClassifier contract
# ---------------------------------------------------------------------------

class TestAbstractMediaClassifier(unittest.TestCase):
    """Verify the ABC contract and default is_ocp_query implementation."""

    def _make_concrete(self, fixed_type: MediaType, fixed_conf: float):
        class _Concrete(AbstractMediaClassifier):
            def classify(self, query, lang, valid_labels=None):
                return fixed_type, fixed_conf

        return _Concrete()

    def test_is_ocp_query_true_for_non_generic(self):
        clf = self._make_concrete(MediaType.MUSIC, 0.9)
        result, conf = clf.is_ocp_query("play music", "en-us")
        self.assertTrue(result)
        self.assertAlmostEqual(conf, 0.9)

    def test_is_ocp_query_false_for_generic(self):
        clf = self._make_concrete(MediaType.GENERIC, 0.0)
        result, conf = clf.is_ocp_query("hello", "en-us")
        self.assertFalse(result)
        self.assertAlmostEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# KeywordMediaClassifier
# ---------------------------------------------------------------------------

class TestKeywordMediaClassifierNoMatch(unittest.TestCase):
    def test_no_match_returns_generic(self):
        clf = _kw_clf()  # nothing matches
        mt, conf = clf.classify("something random", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)


class TestKeywordMediaClassifierBasicTypes(unittest.TestCase):
    """Each first-tier vocab should produce the expected mediavocab MediaType.

    Several fine-grained vocabs now collapse onto a shared mediavocab type
    (e.g. DocumentaryKeyword/VideoKeyword/ADKeyword → MOVIE); the lost nuance
    is carried as genre tags and asserted separately below.
    """

    _CASES = [
        ("DocumentaryKeyword", MediaType.MOVIE),
        ("AudioBookKeyword",   MediaType.AUDIOBOOK),
        ("NewsKeyword",        MediaType.RADIO),
        ("AnimeKeyword",       MediaType.EPISODIC_SERIES),
        ("CartoonKeyword",     MediaType.EPISODIC_SERIES),
        ("PodcastKeyword",     MediaType.PODCAST),
        ("RadioKeyword",       MediaType.RADIO),
        ("MusicKeyword",       MediaType.MUSIC),
        ("TVKeyword",          MediaType.TV),
        ("SeriesKeyword",      MediaType.EPISODIC_SERIES),
        ("ComicBookKeyword",   MediaType.COMIC),
        ("GameKeyword",        MediaType.GAME),
        ("ADKeyword",          MediaType.MOVIE),
        ("ASMRKeyword",        MediaType.PROCEDURAL_AMBIENT),
        ("VideoKeyword",       MediaType.MOVIE),
        ("AudioKeyword",       MediaType.MUSIC),
    ]

    def test_each_vocab_maps_to_correct_type(self):
        for vocab, expected_type in self._CASES:
            with self.subTest(vocab=vocab):
                clf = _kw_clf(vocab)
                mt, conf = clf.classify("irrelevant query", "en-us")
                self.assertEqual(mt, expected_type, f"Failed for {vocab}")
                self.assertGreater(conf, 0.0)

    def test_genre_tags_preserve_collapsed_distinctions(self):
        """Where a vocab collapses to a shared type, classify_genres keeps
        the distinguishing tag (anime / animation / asmr)."""
        cases = [
            ("AnimeKeyword",   "anime"),
            ("CartoonKeyword", "animation"),
            ("ASMRKeyword",    "asmr"),
        ]
        for vocab, genre in cases:
            with self.subTest(vocab=vocab):
                clf = _kw_clf(vocab)
                genres = clf.classify_genres("irrelevant query", "en-us")
                self.assertIn(genre, genres, f"Failed for {vocab}")


class TestKeywordMediaClassifierAudioDrama(unittest.TestCase):
    """AudioDramaKeyword must beat plain RadioKeyword (ordering check)."""

    def test_audio_drama_wins_over_radio(self):
        clf = _kw_clf("AudioDramaKeyword", "RadioKeyword")
        mt, _ = clf.classify("radio theatre", "en-us")
        self.assertEqual(mt, MediaType.AUDIO_DRAMA)

    def test_radio_when_no_audio_drama(self):
        clf = _kw_clf("RadioKeyword")
        mt, _ = clf.classify("radio", "en-us")
        self.assertEqual(mt, MediaType.RADIO)


class TestKeywordMediaClassifierMovieFamily(unittest.TestCase):
    """Movie sub-types (short/silent/b&w) refine a generic MovieKeyword hit."""

    def test_short_film_when_short_keyword(self):
        clf = _kw_clf("MovieKeyword", "ShortKeyword")
        mt, conf = clf.classify("short movie", "en-us")
        self.assertEqual(mt, MediaType.SHORT_FILM)
        self.assertAlmostEqual(conf, 0.7)

    def test_silent_movie_when_silent_keyword(self):
        # SILENT_MOVIE intent now collapses onto the MOVIE mediavocab type,
        # but the more specific keyword still wins a higher confidence.
        clf = _kw_clf("MovieKeyword", "SilentKeyword")
        mt, conf = clf.classify("silent movie", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertAlmostEqual(conf, 0.7)

    def test_bw_movie_when_bw_keyword(self):
        # BW_MOVIE intent now collapses onto the MOVIE mediavocab type.
        clf = _kw_clf("MovieKeyword", "BWKeyword")
        mt, conf = clf.classify("black and white movie", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertAlmostEqual(conf, 0.7)

    def test_plain_movie_fallback(self):
        clf = _kw_clf("MovieKeyword")
        mt, conf = clf.classify("movie", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertAlmostEqual(conf, 0.6)


class TestKeywordMediaClassifierAdultFamily(unittest.TestCase):
    def test_hentai_via_hentai_keyword_alone(self):
        # HENTAI intent collapses onto EPISODIC_SERIES; the adult/anime
        # nuance survives as genre tags.
        clf = _kw_clf("HentaiKeyword")
        mt, _ = clf.classify("hentai", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        genres = clf.classify_genres("hentai", "en-us")
        self.assertIn("anime", genres)
        self.assertIn("adult", genres)

    def test_adult_audio_via_audio_keyword(self):
        # ADULT_AUDIO intent collapses onto MUSIC; adult signal via genre.
        clf = _kw_clf("AdultKeyword", "AudioKeyword")
        mt, _ = clf.classify("adult audio", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertIn("adult", clf.classify_genres("adult audio", "en-us"))

    def test_plain_adult_fallback(self):
        # ADULT intent collapses onto MOVIE; adult signal via genre.
        clf = _kw_clf("AdultKeyword")
        mt, _ = clf.classify("adult content", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertIn("adult", clf.classify_genres("adult content", "en-us"))

    def test_adult_anime_is_hentai_and_blockable(self):
        """Adult detection takes precedence (content filtering is a safety
        feature): 'adult anime' resolves to the hentai intent — still
        EPISODIC_SERIES, but tagged BOTH anime and adult so the filter blocks it."""
        clf = _kw_clf("AdultKeyword", "AnimeKeyword")
        mt, _ = clf.classify("adult anime", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        genres = clf.classify_genres("adult anime", "en-us")
        self.assertIn("anime", genres)
        self.assertIn("adult", genres)  # must be blockable

    def test_adult_asmr_is_adult_audio_and_blockable(self):
        """'adult asmr' resolves to the adult_audio intent (MUSIC + adult genre)
        so the content filter blocks it, rather than escaping as plain ASMR."""
        clf = _kw_clf("AdultKeyword", "ASMRKeyword")
        mt, _ = clf.classify("adult asmr", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertIn("adult", clf.classify_genres("adult asmr", "en-us"))


class TestKeywordMediaClassifierValidLabels(unittest.TestCase):
    """valid_labels filtering: excluded types must fall through to GENERIC."""

    def test_excluded_type_returns_generic(self):
        clf = _kw_clf("MusicKeyword")
        # Music matches, but valid_labels excludes it
        mt, conf = clf.classify("music", "en-us",
                                valid_labels=[MediaType.MOVIE])
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_included_type_returned_normally(self):
        clf = _kw_clf("MusicKeyword")
        mt, conf = clf.classify("music", "en-us",
                                valid_labels=[MediaType.MUSIC])
        self.assertEqual(mt, MediaType.MUSIC)

    def test_movie_family_blocked_when_no_movie_type_in_valid(self):
        """MovieKeyword matched but no movie variant is in valid_labels."""
        clf = _kw_clf("MovieKeyword")
        mt, _ = clf.classify("movie", "en-us",
                             valid_labels=[MediaType.MUSIC])
        self.assertEqual(mt, MediaType.GENERIC)

    def test_short_film_blocked_but_plain_movie_allowed(self):
        """ShortKeyword hit, but SHORT_FILM excluded; MOVIE is allowed."""
        clf = _kw_clf("MovieKeyword", "ShortKeyword")
        mt, _ = clf.classify("short movie", "en-us",
                             valid_labels=[MediaType.MOVIE])
        self.assertEqual(mt, MediaType.MOVIE)


class TestKeywordMediaClassifierIsOcpQuery(unittest.TestCase):
    def test_is_ocp_for_music(self):
        clf = _kw_clf("MusicKeyword")
        is_ocp, conf = clf.is_ocp_query("play jazz", "en-us")
        self.assertTrue(is_ocp)
        self.assertGreater(conf, 0.0)

    def test_not_ocp_for_no_match(self):
        clf = _kw_clf()
        is_ocp, conf = clf.is_ocp_query("set a timer", "en-us")
        self.assertFalse(is_ocp)
        self.assertEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# Model2VecMediaClassifier
# ---------------------------------------------------------------------------

class TestModel2VecClassify(unittest.TestCase):
    def test_music_classification(self):
        model = _mock_m2v_model(domain="ocp_play", intent="music",
                                domain_prob=0.9, intent_prob=0.8)
        clf = Model2VecMediaClassifier(model)
        mt, conf = clf.classify("play jazz", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertAlmostEqual(conf, 0.8)

    def test_domain_not_ocp_play_returns_generic(self):
        model = _mock_m2v_model(domain="other", intent="music",
                                domain_prob=0.9, intent_prob=0.8)
        clf = Model2VecMediaClassifier(model)
        mt, conf = clf.classify("set an alarm", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_domain_below_threshold_returns_generic(self):
        model = _mock_m2v_model(domain="ocp_play", intent="music",
                                domain_prob=0.3, intent_prob=0.8)
        clf = Model2VecMediaClassifier(model, domain_threshold=0.5)
        mt, conf = clf.classify("play jazz", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_intent_below_threshold_returns_generic(self):
        model = _mock_m2v_model(domain="ocp_play", intent="music",
                                domain_prob=0.9, intent_prob=0.1)
        clf = Model2VecMediaClassifier(model, intent_threshold=0.3)
        mt, conf = clf.classify("play jazz", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_unknown_intent_label_returns_generic(self):
        model = _mock_m2v_model(domain="ocp_play", intent="unknown_future_label",
                                domain_prob=0.9, intent_prob=0.8)
        clf = Model2VecMediaClassifier(model)
        mt, conf = clf.classify("play something", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        # Unknown intent still returns the model's confidence
        self.assertAlmostEqual(conf, 0.8)

    def test_valid_labels_filter(self):
        model = _mock_m2v_model(domain="ocp_play", intent="music",
                                domain_prob=0.9, intent_prob=0.8)
        clf = Model2VecMediaClassifier(model)
        mt, conf = clf.classify("play jazz", "en-us",
                                valid_labels=[MediaType.MOVIE])
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_valid_labels_passes_when_included(self):
        model = _mock_m2v_model(domain="ocp_play", intent="music",
                                domain_prob=0.9, intent_prob=0.8)
        clf = Model2VecMediaClassifier(model)
        mt, conf = clf.classify("play jazz", "en-us",
                                valid_labels=[MediaType.MUSIC])
        self.assertEqual(mt, MediaType.MUSIC)

    def test_model_exception_returns_generic(self):
        model = MagicMock()
        model.predict.side_effect = RuntimeError("model broken")
        clf = Model2VecMediaClassifier(model)
        mt, conf = clf.classify("play something", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_all_intent_labels_have_mapping(self):
        """Every label the model may produce should map to a MediaType."""
        for label, mt in INTENT_TO_MEDIA_TYPE.items():
            with self.subTest(label=label):
                self.assertIsInstance(mt, MediaType)

    def test_all_known_media_types_covered(self):
        """Spot-check that the most common mediavocab types appear in the map."""
        mapped = set(INTENT_TO_MEDIA_TYPE.values())
        for mt in (MediaType.MUSIC, MediaType.MOVIE, MediaType.PODCAST,
                   MediaType.RADIO, MediaType.AUDIOBOOK,
                   MediaType.TV, MediaType.EPISODIC_SERIES):
            self.assertIn(mt, mapped)


class TestModel2VecIsOcpQuery(unittest.TestCase):
    def test_ocp_play_domain_is_ocp(self):
        model = _mock_m2v_model(domain="ocp_play", domain_prob=0.9)
        clf = Model2VecMediaClassifier(model)
        is_ocp, conf = clf.is_ocp_query("play jazz", "en-us")
        self.assertTrue(is_ocp)
        self.assertAlmostEqual(conf, 0.9)

    def test_ocp_control_domain_is_ocp(self):
        model = _mock_m2v_model(domain="ocp_control", domain_prob=0.85)
        clf = Model2VecMediaClassifier(model)
        is_ocp, conf = clf.is_ocp_query("pause the music", "en-us")
        self.assertTrue(is_ocp)

    def test_other_domain_not_ocp(self):
        model = _mock_m2v_model(domain="other", domain_prob=0.95)
        clf = Model2VecMediaClassifier(model)
        is_ocp, conf = clf.is_ocp_query("what is the weather", "en-us")
        self.assertFalse(is_ocp)

    def test_low_confidence_ocp_play_not_ocp(self):
        model = _mock_m2v_model(domain="ocp_play", domain_prob=0.2)
        clf = Model2VecMediaClassifier(model, domain_threshold=0.5)
        is_ocp, conf = clf.is_ocp_query("play", "en-us")
        self.assertFalse(is_ocp)

    def test_exception_returns_false(self):
        model = MagicMock()
        model.predict_proba.side_effect = RuntimeError("broken")
        clf = Model2VecMediaClassifier(model)
        is_ocp, conf = clf.is_ocp_query("play jazz", "en-us")
        self.assertFalse(is_ocp)
        self.assertEqual(conf, 0.0)

    def test_only_uses_domain_head(self):
        """is_ocp_query must call predict_proba but NOT predict (no intent)."""
        model = _mock_m2v_model(domain="ocp_play", domain_prob=0.9)
        clf = Model2VecMediaClassifier(model)
        clf.is_ocp_query("play jazz", "en-us")
        model.predict_proba.assert_called_once()
        model.predict.assert_not_called()


class TestModel2VecFromPath(unittest.TestCase):
    def test_missing_torch_raises_runtime_error(self):
        with patch.dict("sys.modules", {"torch": None}):
            with self.assertRaises(RuntimeError):
                Model2VecMediaClassifier.from_path("/nonexistent/path")

    def test_bad_path_raises_runtime_error(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

        mock_cls = MagicMock(side_effect=FileNotFoundError("no such file"))
        with patch(
            "ovos_media_classifier.m2v.Model2VecMediaClassifier.from_path",
            side_effect=RuntimeError("Failed to load"),
        ):
            with self.assertRaises(RuntimeError):
                Model2VecMediaClassifier.from_path("/nonexistent/path")


# ---------------------------------------------------------------------------
# load_media_classifier factory
# ---------------------------------------------------------------------------

class TestLoadMediaClassifierFactory(unittest.TestCase):
    def test_no_config_returns_keyword(self):
        from ovos_media_classifier import load_media_classifier

        clf = load_media_classifier()
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_empty_config_returns_keyword(self):
        from ovos_media_classifier import load_media_classifier

        clf = load_media_classifier(config={})
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_voc_match_func_injected(self):
        from ovos_media_classifier import load_media_classifier

        sentinel = object()
        called_with = []

        def _voc(phrase, vocab, **kw):
            called_with.append((phrase, vocab))
            return False

        clf = load_media_classifier(voc_match_func=_voc)
        clf.classify("play jazz", "en-us")
        # At least one voc_match call was made
        self.assertTrue(len(called_with) > 0)

    def test_model_path_loads_m2v(self):
        from ovos_media_classifier import load_media_classifier

        mock_clf = MagicMock(spec=Model2VecMediaClassifier)
        with patch(
            "ovos_media_classifier.m2v.Model2VecMediaClassifier.from_path",
            return_value=mock_clf,
        ):
            clf = load_media_classifier(
                config={"media_classifier_model": "/some/path"}
            )
        self.assertIs(clf, mock_clf)

    def test_m2v_load_failure_falls_back_to_keyword(self):
        from ovos_media_classifier import load_media_classifier

        with patch(
            "ovos_media_classifier.m2v.Model2VecMediaClassifier.from_path",
            side_effect=RuntimeError("load failed"),
        ):
            clf = load_media_classifier(
                config={"media_classifier_model": "/some/path"}
            )
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_m2v_import_error_falls_back_to_keyword(self):
        from ovos_media_classifier import load_media_classifier

        with patch.dict(
            "sys.modules",
            {"ovos_media_classifier.m2v": None},
        ):
            clf = load_media_classifier(
                config={"media_classifier_model": "/some/path"}
            )
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_custom_thresholds_passed_to_m2v(self):
        from ovos_media_classifier import load_media_classifier

        with patch(
            "ovos_media_classifier.m2v.Model2VecMediaClassifier.from_path"
        ) as mock_from_path:
            mock_from_path.return_value = MagicMock(spec=Model2VecMediaClassifier)
            load_media_classifier(
                config={
                    "media_classifier_model": "/some/path",
                    "media_classifier_domain_threshold": 0.7,
                    "media_classifier_intent_threshold": 0.4,
                }
            )
        mock_from_path.assert_called_once_with(
            "/some/path",
            domain_threshold=0.7,
            intent_threshold=0.4,
        )

    def test_no_voc_match_func_classify_returns_generic(self):
        from ovos_media_classifier import load_media_classifier

        clf = load_media_classifier(voc_match_func=None)
        mt, conf = clf.classify("play jazz", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# New taxonomy tests (TV/TV_SHOW split, TRAILER, BEHIND_THE_SCENES, etc.)
# ---------------------------------------------------------------------------

class TestMediaTypeIsMediavocab(unittest.TestCase):
    """The public MediaType is no longer locally defined — it is re-exported
    from ``mediavocab`` (a str-Enum), enforcing the shared taxonomy."""

    def test_media_type_is_mediavocab(self):
        self.assertIs(MediaType, mediavocab.MediaType)

    def test_media_type_is_str_enum(self):
        self.assertTrue(issubclass(MediaType, str))
        self.assertTrue(issubclass(MediaType, enum.Enum))

    def test_members_are_canonical_strings(self):
        # str-Enum: members compare equal to their string value, not ints.
        self.assertEqual(MediaType.MOVIE, "movie")
        self.assertEqual(MediaType.EPISODIC_SERIES, "episodic_series")
        self.assertNotEqual(MediaType.MOVIE, MediaType.TV)

    def test_removed_legacy_members_absent(self):
        for name in ("DOCUMENTARY", "ANIME", "CARTOON", "NEWS", "VIDEO",
                     "TV_SHOW", "TRAILER", "BEHIND_THE_SCENES", "ADULT",
                     "HENTAI", "ASMR"):
            self.assertFalse(hasattr(MediaType, name),
                             f"{name} should no longer be a MediaType member")

    def test_episodic_series_present(self):
        self.assertEqual(MediaType.EPISODIC_SERIES.value, "episodic_series")


class TestOCPEntityLabelStrings(unittest.TestCase):
    """Verify fixed string values in OCPEntityLabel."""

    def test_adult_streaming_service_value(self):
        self.assertEqual(OCPEntityLabel.ADULT_STREAMING_SERVICE.value,
                         "adult_streaming_service")

    def test_no_porn_streaming_service_string(self):
        values = {label.value for label in OCPEntityLabel}
        self.assertNotIn("porn_streaming_service", values)

    def test_no_news_topic_label(self):
        names = {label.name for label in OCPEntityLabel}
        self.assertNotIn("NEWS_TOPIC", names)

    def test_news_category_present(self):
        self.assertEqual(OCPEntityLabel.NEWS_CATEGORY.value, "news_category")

    def test_tv_channel_present(self):
        self.assertEqual(OCPEntityLabel.TV_CHANNEL.value, "tv_channel")

    def test_trailer_title_present(self):
        self.assertEqual(OCPEntityLabel.TRAILER_TITLE.value, "trailer_title")

    def test_bts_title_present(self):
        self.assertEqual(OCPEntityLabel.BTS_TITLE.value, "bts_title")


class TestPlayIntentToMediaTypeMapping(unittest.TestCase):
    """Verify the intent → mediavocab.MediaType mapping is correct/complete."""

    # Expected collapse of the fine-grained intent space onto mediavocab types.
    _EXPECTED = {
        OCPPlayIntent.MUSIC:             MediaType.MUSIC,
        OCPPlayIntent.PODCAST:           MediaType.PODCAST,
        OCPPlayIntent.RADIO:             MediaType.RADIO,
        OCPPlayIntent.AUDIOBOOK:         MediaType.AUDIOBOOK,
        OCPPlayIntent.NEWS:              MediaType.RADIO,
        OCPPlayIntent.MOVIE:             MediaType.MOVIE,
        OCPPlayIntent.TV:                MediaType.TV,
        OCPPlayIntent.TV_SHOW:           MediaType.EPISODIC_SERIES,
        OCPPlayIntent.VIDEO:             MediaType.MOVIE,
        OCPPlayIntent.VIDEO_EPISODES:    MediaType.EPISODIC_SERIES,
        OCPPlayIntent.AUDIO:             MediaType.MUSIC,
        OCPPlayIntent.GAME:              MediaType.GAME,
        OCPPlayIntent.ANIME:             MediaType.EPISODIC_SERIES,
        OCPPlayIntent.CARTOON:           MediaType.EPISODIC_SERIES,
        OCPPlayIntent.DOCUMENTARY:       MediaType.MOVIE,
        OCPPlayIntent.SHORT_FILM:        MediaType.SHORT_FILM,
        OCPPlayIntent.SILENT_MOVIE:      MediaType.MOVIE,
        OCPPlayIntent.BW_MOVIE:          MediaType.MOVIE,
        OCPPlayIntent.RADIO_THEATRE:     MediaType.AUDIO_DRAMA,
        OCPPlayIntent.VISUAL_STORY:      MediaType.COMIC,
        OCPPlayIntent.ASMR:              MediaType.PROCEDURAL_AMBIENT,
        OCPPlayIntent.AUDIO_DESCRIPTION: MediaType.MOVIE,
        OCPPlayIntent.MUSIC_VIDEO:       MediaType.MUSIC_VIDEO,
        OCPPlayIntent.TRAILER:           MediaType.MOVIE,
        OCPPlayIntent.BEHIND_THE_SCENES: MediaType.MOVIE,
        OCPPlayIntent.ADULT:             MediaType.MOVIE,
        OCPPlayIntent.ADULT_AUDIO:       MediaType.MUSIC,
        OCPPlayIntent.HENTAI:            MediaType.EPISODIC_SERIES,
        OCPPlayIntent.GENERIC:           MediaType.GENERIC,
    }

    def test_tv_show_maps_to_episodic_series_not_tv(self):
        self.assertEqual(PLAY_INTENT_TO_MEDIA_TYPE[OCPPlayIntent.TV_SHOW],
                         MediaType.EPISODIC_SERIES)
        self.assertNotEqual(PLAY_INTENT_TO_MEDIA_TYPE[OCPPlayIntent.TV_SHOW],
                            MediaType.TV)

    def test_tv_intent_maps_to_tv(self):
        self.assertEqual(PLAY_INTENT_TO_MEDIA_TYPE[OCPPlayIntent.TV],
                         MediaType.TV)

    def test_trailer_intent_collapses_to_movie(self):
        self.assertEqual(PLAY_INTENT_TO_MEDIA_TYPE[OCPPlayIntent.TRAILER],
                         MediaType.MOVIE)

    def test_behind_the_scenes_intent_collapses_to_movie(self):
        self.assertEqual(
            PLAY_INTENT_TO_MEDIA_TYPE[OCPPlayIntent.BEHIND_THE_SCENES],
            MediaType.MOVIE,
        )

    def test_full_mapping_matches_expected(self):
        self.assertEqual(dict(PLAY_INTENT_TO_MEDIA_TYPE), self._EXPECTED)

    def test_all_play_intents_have_mapping(self):
        for intent in OCPPlayIntent:
            with self.subTest(intent=intent):
                self.assertIn(intent, PLAY_INTENT_TO_MEDIA_TYPE)

    def test_every_value_is_a_mediavocab_type(self):
        for intent, mt in PLAY_INTENT_TO_MEDIA_TYPE.items():
            with self.subTest(intent=intent):
                self.assertIsInstance(mt, mediavocab.MediaType)


class TestOCPControlIntentExtensions(unittest.TestCase):
    """Verify new control intents are present."""

    def test_shuffle_present(self):
        self.assertEqual(OCPControlIntent.SHUFFLE.value, "shuffle")

    def test_repeat_present(self):
        self.assertEqual(OCPControlIntent.REPEAT.value, "repeat")

    def test_seek_forward_present(self):
        self.assertEqual(OCPControlIntent.SEEK_FORWARD.value, "seek_forward")

    def test_seek_backward_present(self):
        self.assertEqual(OCPControlIntent.SEEK_BACKWARD.value, "seek_backward")

    def test_total_control_intents(self):
        self.assertEqual(len(OCPControlIntent), 15)


class TestKeywordMediaClassifierNewTypes(unittest.TestCase):
    """Verify IPTV, TRAILER, and BEHIND_THE_SCENES keyword routing."""

    def test_iptv_keyword_returns_tv(self):
        clf = _kw_clf("IPTVKeyword")
        mt, conf = clf.classify("stream BBC One", "en-us")
        self.assertEqual(mt, MediaType.TV)
        self.assertAlmostEqual(conf, 0.6)

    def test_iptv_keyword_beats_tv_keyword(self):
        # IPTVKeyword is checked before TVKeyword in the chain
        clf = _kw_clf("IPTVKeyword", "TVKeyword")
        mt, conf = clf.classify("live channel", "en-us")
        self.assertEqual(mt, MediaType.TV)

    def test_trailer_keyword_returns_movie(self):
        # TRAILER intent collapses onto the MOVIE mediavocab type.
        clf = _kw_clf("TrailerKeyword")
        mt, conf = clf.classify("show the trailer for Top Gun", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertAlmostEqual(conf, 0.7)

    def test_behind_the_scenes_keyword_returns_movie(self):
        # BEHIND_THE_SCENES intent collapses onto the MOVIE mediavocab type.
        clf = _kw_clf("BehindTheScenesKeyword")
        mt, conf = clf.classify("watch the making of Dune", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertAlmostEqual(conf, 0.7)

    def test_trailer_blocked_by_valid_labels(self):
        # TRAILER collapses onto MOVIE, so MOVIE must be excluded to block it.
        clf = _kw_clf("TrailerKeyword")
        mt, conf = clf.classify("trailer", "en-us",
                                valid_labels=[MediaType.MUSIC])
        self.assertEqual(mt, MediaType.GENERIC)

    def test_tv_show_keyword_still_works(self):
        clf = _kw_clf("TVKeyword")
        mt, conf = clf.classify("tv", "en-us")
        self.assertEqual(mt, MediaType.TV)


class TestIntentPriorityList(unittest.TestCase):
    """Verify _INTENT_PRIORITY in ahocorasick covers all OCPPlayIntent values."""

    def test_all_intents_in_priority_list(self):
        from ovos_media_classifier.ahocorasick import _INTENT_PRIORITY
        priority_set = set(_INTENT_PRIORITY)
        for intent in OCPPlayIntent:
            with self.subTest(intent=intent):
                self.assertIn(intent, priority_set)

    def test_no_duplicate_intents_in_priority(self):
        from ovos_media_classifier.ahocorasick import _INTENT_PRIORITY
        self.assertEqual(len(_INTENT_PRIORITY), len(set(_INTENT_PRIORITY)))

    def test_tv_before_tv_show_in_priority(self):
        from ovos_media_classifier.ahocorasick import _INTENT_PRIORITY
        tv_idx = _INTENT_PRIORITY.index(OCPPlayIntent.TV)
        tv_show_idx = _INTENT_PRIORITY.index(OCPPlayIntent.TV_SHOW)
        self.assertLess(tv_idx, tv_show_idx)

    def test_generic_is_last(self):
        from ovos_media_classifier.ahocorasick import _INTENT_PRIORITY
        self.assertEqual(_INTENT_PRIORITY[-1], OCPPlayIntent.GENERIC)


# ---------------------------------------------------------------------------
# AhocorasickMediaClassifier
# ---------------------------------------------------------------------------


class TestAhocorasickMediaClassifierImportError(unittest.TestCase):
    """Verify ahocorasick-ner dependency is enforced."""

    def test_missing_ahocorasick_ner_raises_import_error(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        with patch.dict("sys.modules", {"ahocorasick_ner": None}):
            with self.assertRaises(ImportError) as ctx:
                AhocorasickMediaClassifier(MagicMock())
            self.assertIn("ahocorasick-ner", str(ctx.exception))


class TestAhocorasickMediaClassifierBasic(unittest.TestCase):
    """Test AhocorasickMediaClassifier with mocked NER."""

    def _make_ner(self, entities: Optional[Dict[str, str]] = None):
        """Return a mock AhocorasickNER that returns given entities."""
        ner = MagicMock()
        if entities is None:
            entities = {}

        def _tag(query):
            return [{"label": label, "word": word} for label, word in entities.items()]

        ner.tag.return_value = _tag("")
        return ner

    def test_classify_with_entity_hit(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = self._make_ner({"music_streaming_service": "Spotify"})
        ner.tag.return_value = [{"label": "music_streaming_service", "word": "Spotify"}]

        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("play on Spotify", "en-us")

        self.assertEqual(mt, MediaType.MUSIC)
        self.assertEqual(conf, 0.6)

    def test_classify_no_hit_returns_generic(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = self._make_ner()
        ner.tag.return_value = []

        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("random query", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_with_valid_labels_filter(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = self._make_ner()
        ner.tag.return_value = [{"label": "music_streaming_service", "word": "Spotify"}]

        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("play on Spotify", "en-us",
                               valid_labels=[MediaType.MOVIE])

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_domain_with_hit(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            from ovos_media_classifier.intents import OCPDomain
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = self._make_ner()
        ner.tag.return_value = [{"label": "music_streaming_service", "word": "Spotify"}]

        clf = AhocorasickMediaClassifier(ner)
        domain, conf = clf.classify_domain("play on Spotify", "en-us")

        self.assertEqual(domain, OCPDomain.OCP_PLAY)
        self.assertEqual(conf, 0.6)

    def test_classify_domain_no_hit(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            from ovos_media_classifier.intents import OCPDomain
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = self._make_ner()
        ner.tag.return_value = []

        clf = AhocorasickMediaClassifier(ner)
        domain, conf = clf.classify_domain("random query", "en-us")

        self.assertEqual(domain, OCPDomain.NOT_OCP)
        self.assertEqual(conf, 0.0)

    def test_ner_exception_returns_generic(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = MagicMock()
        ner.tag.side_effect = RuntimeError("NER error")

        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("query", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_add_word_to_container(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            from ovos_media_classifier.entities import EntitiesContainer
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        container = EntitiesContainer()
        container.add("music_streaming_service", "Spotify")

        clf = AhocorasickMediaClassifier.from_container(container)
        self.assertIsNotNone(clf.container)

        # Add a new word and verify it's reflected
        clf.add_word("music_streaming_service", "Tidal")
        # The container's NER should have been updated
        self.assertIsNotNone(clf._ner)

    def test_label_map_override(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = MagicMock()
        ner.tag.return_value = [{"label": "custom_label", "word": "value"}]

        # Map custom_label to TV instead of default
        custom_map = {"custom_label": OCPPlayIntent.TV}
        clf = AhocorasickMediaClassifier(ner, label_map=custom_map)

        mt, conf = clf.classify("query", "en-us")
        self.assertEqual(mt, MediaType.TV)

    def test_priority_resolution_multiple_hits(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        ner = MagicMock()
        # Two hits: movie and music; music comes earlier in _INTENT_PRIORITY
        ner.tag.return_value = [
            {"label": "music_streaming_service", "word": "Spotify"},
            {"label": "movie_streaming_service", "word": "Netflix"},
        ]

        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("query", "en-us")

        # Music should win (comes earlier in _INTENT_PRIORITY at index 10 vs MOVIE at 21)
        self.assertEqual(mt, MediaType.MUSIC)


class TestAhocorasickMediaClassifierFactories(unittest.TestCase):
    """Test AhocorasickMediaClassifier factory methods."""

    def test_from_wordlists(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        clf = AhocorasickMediaClassifier.from_wordlists({
            "music": ["jazz", "blues", "rock"],
            "movie": ["cinema", "film"],
        })
        self.assertIsNotNone(clf._ner)
        self.assertIsNotNone(clf.container)

    def test_from_csv(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "value"])
                writer.writerow(["music", "jazz"])
                writer.writerow(["movie", "cinema"])

            clf = AhocorasickMediaClassifier.from_csv(csv_path)
            self.assertIsNotNone(clf._ner)
            self.assertIsNotNone(clf.container)

    def test_from_container(self):
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            from ovos_media_classifier.entities import EntitiesContainer
        except ImportError:
            self.skipTest("ahocorasick-ner not installed")

        container = EntitiesContainer()
        container.add("music", "jazz")
        container.add("movie", "cinema")

        clf = AhocorasickMediaClassifier.from_container(container)
        self.assertEqual(clf.container, container)


# ---------------------------------------------------------------------------
# SklearnMediaClassifier
# ---------------------------------------------------------------------------


class TestSklearnMediaClassifierBasic(unittest.TestCase):
    """Test SklearnMediaClassifier with mocked pipelines."""

    def _make_play_pipeline(self, label: str = "music", proba: float = 0.8):
        """Return a mock sklearn pipeline for play classification."""
        pipe = MagicMock()
        pipe.predict.return_value = np.array([label])
        pipe.predict_proba.return_value = np.array([[proba]])
        return pipe

    def _make_domain_pipeline(self, label: str = "ocp_play", proba: float = 0.8):
        """Return a mock sklearn pipeline for domain classification."""
        pipe = MagicMock()
        pipe.predict.return_value = np.array([label])
        pipe.predict_proba.return_value = np.array([[proba]])
        return pipe

    def test_classify_basic(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = self._make_play_pipeline("music", 0.85)
        clf = SklearnMediaClassifier(play_pipe)
        mt, conf = clf.classify("play jazz", "en-us")

        self.assertEqual(mt, MediaType.MUSIC)
        self.assertAlmostEqual(conf, 0.85)

    def test_classify_below_threshold(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = self._make_play_pipeline("music", 0.2)
        clf = SklearnMediaClassifier(play_pipe, play_threshold=0.5)
        mt, conf = clf.classify("play jazz", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_unknown_label(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = self._make_play_pipeline("unknown_intent", 0.8)
        clf = SklearnMediaClassifier(play_pipe)
        mt, conf = clf.classify("query", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        # Unknown intent still returns the model's confidence
        self.assertAlmostEqual(conf, 0.8)

    def test_classify_with_valid_labels(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = self._make_play_pipeline("music", 0.8)
        clf = SklearnMediaClassifier(play_pipe)
        mt, conf = clf.classify("play jazz", "en-us",
                               valid_labels=[MediaType.MOVIE])

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_exception_returns_generic(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = MagicMock()
        play_pipe.predict.side_effect = RuntimeError("predict failed")

        clf = SklearnMediaClassifier(play_pipe)
        mt, conf = clf.classify("query", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_domain_with_domain_pipeline(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
            from ovos_media_classifier.intents import OCPDomain
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = self._make_play_pipeline("music", 0.8)
        domain_pipe = self._make_domain_pipeline("ocp_play", 0.9)

        clf = SklearnMediaClassifier(play_pipe, domain_pipeline=domain_pipe)
        domain, conf = clf.classify_domain("play jazz", "en-us")

        self.assertEqual(domain, OCPDomain.OCP_PLAY)
        self.assertAlmostEqual(conf, 0.9)

    def test_classify_domain_below_threshold(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
            from ovos_media_classifier.intents import OCPDomain
        except ImportError:
            self.skipTest("sklearn not installed")

        domain_pipe = self._make_domain_pipeline("ocp_play", 0.2)
        clf = SklearnMediaClassifier(MagicMock(), domain_pipeline=domain_pipe,
                                     domain_threshold=0.5)
        domain, conf = clf.classify_domain("query", "en-us")

        self.assertEqual(domain, OCPDomain.NOT_OCP)

    def test_predict_with_conf_no_proba(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        # Mock a classifier that doesn't have predict_proba (e.g., SVM)
        pipe = MagicMock()
        pipe.predict.return_value = np.array(["music"])
        del pipe.predict_proba  # Remove predict_proba
        pipe.decision_function.return_value = np.array([[0.5]])

        label, conf = SklearnMediaClassifier._predict_with_conf(pipe, "query")

        self.assertEqual(label, "music")
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


class TestSklearnMediaClassifierPersistence(unittest.TestCase):
    """Test SklearnMediaClassifier save/load."""

    def test_from_path_missing_joblib(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        with patch.dict("sys.modules", {"joblib": None}):
            with self.assertRaises(ImportError):
                SklearnMediaClassifier.from_path("/nonexistent/path")

    def test_save_requires_joblib(self):
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
        except ImportError:
            self.skipTest("sklearn not installed")

        play_pipe = MagicMock()
        clf = SklearnMediaClassifier(play_pipe)

        with patch.dict("sys.modules", {"joblib": None}):
            with self.assertRaises(ImportError):
                clf.save("/tmp/model.joblib")


# ---------------------------------------------------------------------------
# PadatiousMediaClassifier
# ---------------------------------------------------------------------------


class TestPadatiousMediaClassifierBasic(unittest.TestCase):
    """Test PadatiousMediaClassifier with mocked containers."""

    def _make_intent_container(self, intent_name: str = "music",
                              intent_conf: float = 0.8):
        """Return a mock IntentContainer."""
        container = MagicMock()

        match = MagicMock()
        match.name = intent_name
        match.conf = intent_conf

        container.calc_intent.return_value = match
        return container

    def test_classify_basic(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = self._make_intent_container("music", 0.85)
        clf = PadatiousMediaClassifier(play_c)

        mt, conf = clf.classify("play some jazz", "en-us")

        self.assertEqual(mt, MediaType.MUSIC)
        self.assertAlmostEqual(conf, 0.85)

    def test_classify_below_threshold(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = self._make_intent_container("music", 0.3)
        clf = PadatiousMediaClassifier(play_c, play_threshold=0.5)

        mt, conf = clf.classify("query", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_no_match(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = MagicMock()
        play_c.calc_intent.return_value = None

        clf = PadatiousMediaClassifier(play_c)
        mt, conf = clf.classify("query", "en-us")

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_with_valid_labels(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = self._make_intent_container("music", 0.8)
        clf = PadatiousMediaClassifier(play_c)

        mt, conf = clf.classify("query", "en-us",
                               valid_labels=[MediaType.MOVIE])

        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_domain_with_domain_container(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
            from ovos_media_classifier.intents import OCPDomain
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = self._make_intent_container("music", 0.8)
        domain_c = self._make_intent_container("ocp_play", 0.9)

        clf = PadatiousMediaClassifier(play_c, domain_container=domain_c)
        domain, conf = clf.classify_domain("play jazz", "en-us")

        self.assertEqual(domain, OCPDomain.OCP_PLAY)
        self.assertAlmostEqual(conf, 0.9)

    def test_calc_intent_exception(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = MagicMock()
        play_c.calc_intent.side_effect = RuntimeError("calc failed")

        clf = PadatiousMediaClassifier(play_c)
        name, conf = clf._calc_intent(play_c, "query")

        self.assertIsNone(name)
        self.assertEqual(conf, 0.0)

    def test_calc_intent_padacioso_dict_result(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        play_c = MagicMock()
        # padacioso returns a dict
        play_c.calc_intent.return_value = {
            "name": "music",
            "conf": 0.75,
        }

        clf = PadatiousMediaClassifier(play_c)
        name, conf = clf._calc_intent(play_c, "query")

        self.assertEqual(name, "music")
        self.assertAlmostEqual(conf, 0.75)


class TestPadatiousMediaClassifierFactories(unittest.TestCase):
    """Test PadatiousMediaClassifier factory methods."""

    def test_from_samples(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        clf = PadatiousMediaClassifier.from_samples({
            "music": ["play {genre}", "put on some {artist}"],
            "movie": ["watch {title}", "play the movie {name}"],
        })
        self.assertIsNotNone(clf._play)

    def test_from_samples_with_domain(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        clf = PadatiousMediaClassifier.from_samples(
            play_samples={
                "music": ["play {genre}"],
                "movie": ["watch {title}"],
            },
            domain_samples={
                "ocp_play": ["play {query}"],
                "ocp_control": ["pause", "resume"],
                "not_ocp": ["set an alarm"],
            }
        )
        self.assertIsNotNone(clf._play)
        self.assertIsNotNone(clf._domain)

    def test_from_locale_dir(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            lang_dir = os.path.join(tmpdir, "en-us")
            os.makedirs(lang_dir)

            # Create intent files
            with open(os.path.join(lang_dir, "music.intent"), "w") as f:
                f.write("play {genre}\nput on some {artist}\n")

            with open(os.path.join(lang_dir, "movie.intent"), "w") as f:
                f.write("watch {title}\nplay the movie {name}\n")

            clf = PadatiousMediaClassifier.from_locale_dir(tmpdir, "en-us")
            self.assertIsNotNone(clf._play)

    def test_from_locale_dir_missing_lang(self):
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
        except ImportError:
            self.skipTest("padatious not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                PadatiousMediaClassifier.from_locale_dir(tmpdir, "en-us")


# ---------------------------------------------------------------------------
# EntitiesContainer loaders
# ---------------------------------------------------------------------------


class TestEntitiesContainerBasic(unittest.TestCase):
    """Test EntitiesContainer basic operations."""

    def test_add_and_get_words(self):
        from ovos_media_classifier.entities import EntitiesContainer

        container = EntitiesContainer()
        container.add("music", "jazz")
        container.add("music", "blues")
        container.add("movie", "cinema")

        # Verify words are stored
        words = container.wordlists
        self.assertIn("music", words)
        self.assertIn("movie", words)

    def test_deduplication(self):
        from ovos_media_classifier.entities import EntitiesContainer

        container = EntitiesContainer()
        container.add("music", "jazz")
        container.add("music", "jazz")  # duplicate

        words = container.wordlists
        music_words = list(words.get("music", []))
        # Count occurrences of 'jazz'
        jazz_count = sum(1 for w in music_words if w == "jazz")
        self.assertEqual(jazz_count, 1)

    def test_load_csv(self):
        import os
        from ovos_media_classifier.entities import EntitiesContainer

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["entity", "label", "source"])
                writer.writerow(["jazz", "music", "manual"])
                writer.writerow(["Netflix", "movie_streaming_service", "api"])

            container = EntitiesContainer()
            count = container.load_csv(csv_path)

            self.assertGreater(count, 0)
            words = container.wordlists
            self.assertIn("music", words)


class TestEntitiesContainerRadarr(unittest.TestCase):
    """Test Radarr loader."""

    @patch("ovos_media_classifier.entities._http_get")
    def test_load_radarr_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        # Mock the HTTP response to return a list of movies
        mock_http_get.return_value = [
            {
                "title": "The Dark Knight",
                "genres": ["action", "crime"],
                "alternateTitles": [],
                "credits": {
                    "castMembers": [
                        {"name": "Christian Bale"},
                    ],
                    "crewMembers": [
                        {"name": "Christopher Nolan", "job": "Director"},
                    ],
                },
                "studio": "Warner Bros",
            },
            {
                "title": "Inception",
                "genres": ["sci-fi", "thriller"],
                "alternateTitles": [],
                "credits": {"castMembers": [], "crewMembers": []},
                "studio": "",
            },
        ]

        container = EntitiesContainer()
        count = container.load_radarr("http://localhost:7878", api_key="test")

        self.assertGreater(count, 0)

    @patch("ovos_media_classifier.entities._http_get")
    def test_load_radarr_http_error(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = None

        container = EntitiesContainer()
        count = container.load_radarr("http://localhost:7878", api_key="test")

        self.assertEqual(count, 0)


class TestEntitiesContainerSonarr(unittest.TestCase):
    """Test Sonarr loader."""

    @patch("ovos_media_classifier.entities._http_get")
    def test_load_sonarr_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = [
            {
                "title": "Breaking Bad",
                "genres": [],
                "alternateTitles": [],
                "year": 2008,
            },
            {
                "title": "Game of Thrones",
                "genres": ["drama"],
                "alternateTitles": [],
                "year": 2011,
            },
        ]

        container = EntitiesContainer()
        count = container.load_sonarr("http://localhost:8989", api_key="test")

        self.assertGreater(count, 0)


class TestEntitiesContainerLidarr(unittest.TestCase):
    """Test Lidarr loader."""

    @patch("ovos_media_classifier.entities._http_get")
    def test_load_lidarr_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        def http_get_side_effect(session, url, **kwargs):
            if "/artist" in url:
                return [
                    {
                        "artistName": "Radiohead",
                        "genres": ["rock", "alternative"],
                    },
                    {
                        "artistName": "The Beatles",
                        "genres": ["rock", "pop"],
                    },
                ]
            elif "/album" in url:
                return [
                    {
                        "title": "OK Computer",
                        "artist": {"artistName": "Radiohead"},
                        "media": [
                            {
                                "tracks": [
                                    {"title": "Paranoid Android"},
                                    {"title": "Karma Police"},
                                ]
                            }
                        ],
                    },
                ]
            return []

        mock_http_get.side_effect = http_get_side_effect

        container = EntitiesContainer()
        count = container.load_lidarr("http://localhost:8686", api_key="test")

        self.assertGreater(count, 0)


class TestEntitiesContainerJellyfin(unittest.TestCase):
    """Test Jellyfin loader."""

    @patch("ovos_media_classifier.entities._http_get")
    def test_load_jellyfin_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        def http_get_side_effect(session, url, **kwargs):
            if "/Users" in url and "Items" not in url:
                # Users endpoint
                return [
                    {"Id": "user123"}
                ]
            else:
                # Items endpoint - return paginated results
                return {
                    "Items": [
                        {
                            "Name": "The Dark Knight",
                            "Type": "Movie",
                            "Genres": ["Action", "Crime"],
                            "People": [
                                {"Name": "Christian Bale", "Type": "Actor"},
                            ],
                        },
                    ],
                    "TotalRecordCount": 1,
                }

        mock_http_get.side_effect = http_get_side_effect

        container = EntitiesContainer()
        count = container.load_jellyfin("http://localhost:8096", api_key="test")

        # Even with only one movie, we should have added at least some entities
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
