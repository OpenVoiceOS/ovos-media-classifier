"""Unit tests for ovos-media-classifier (.voc keyword-only release).

Tests cover:
  - AbstractMediaClassifier — the ABC contract and default is_ocp_query
  - KeywordMediaClassifier.classify() — each branch of the if/elif chain
  - KeywordMediaClassifier.is_ocp_query() — inherited default implementation
  - intents.py taxonomy — MediaType / OCPControlIntent / OCPEntityLabel and the
    raw label → mediavocab.MediaType + genres mapping
  - load_media_classifier() factory — keyword default + external plugin selection
"""
import unittest
from unittest.mock import patch

import enum

import mediavocab

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import (
    MediaType,
    OCPControlIntent,
    OCPEntityLabel,
    LABEL_TO_MEDIA_TYPE,
    LABEL_TO_GENRES,
    genres_for_label,
)
from ovos_media_classifier.keyword import KeywordMediaClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kw_clf(*match_vocabs: str) -> KeywordMediaClassifier:
    """Return a KeywordMediaClassifier whose voc_match returns True only for
    vocab names listed in *match_vocabs*."""

    def _voc_match(phrase: str, vocab: str, **kw) -> bool:
        return vocab in match_vocabs

    return KeywordMediaClassifier(_voc_match)


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
# load_media_classifier factory
# ---------------------------------------------------------------------------

class _DummyClassifier(AbstractMediaClassifier):
    """A trivial external classifier used to exercise plugin selection."""

    def classify(self, query, lang, valid_labels=None):
        return MediaType.MOVIE, 1.0


class TestLoadMediaClassifierFactory(unittest.TestCase):
    """The .voc-only factory: keyword default + external-plugin selection."""

    def test_no_args_returns_keyword(self):
        from ovos_media_classifier import load_media_classifier

        clf = load_media_classifier()
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_empty_config_returns_keyword(self):
        from ovos_media_classifier import load_media_classifier

        clf = load_media_classifier(config={})
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_voc_match_func_is_used(self):
        from ovos_media_classifier import load_media_classifier

        called_with = []

        def _voc(phrase, vocab, **kw):
            called_with.append((phrase, vocab))
            return vocab == "MusicKeyword"

        clf = load_media_classifier(voc_match_func=_voc)
        self.assertIsInstance(clf, KeywordMediaClassifier)
        mt, conf = clf.classify("play jazz", "en-us")
        # the injected matcher drove the classification
        self.assertTrue(len(called_with) > 0)
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertGreater(conf, 0.0)

    def test_missing_plugin_falls_back_to_keyword(self):
        """A configured plugin name that is not installed must log a warning
        and fall back to the bundled keyword classifier (never raise)."""
        from ovos_media_classifier import load_media_classifier

        with patch(
            "ovos_media_classifier.plugins.find_media_classifier_plugins",
            return_value={},
        ):
            with patch("ovos_media_classifier.LOG.warning") as warn:
                clf = load_media_classifier(
                    config={"media_classifier_plugin": "does-not-exist"}
                )
        self.assertIsInstance(clf, KeywordMediaClassifier)
        warn.assert_called_once()
        self.assertIn("does-not-exist", warn.call_args.args[0])

    def test_registered_external_plugin_is_returned(self):
        """When the named plugin IS registered it is instantiated and returned."""
        from ovos_media_classifier import load_media_classifier

        with patch(
            "ovos_media_classifier.plugins.find_media_classifier_plugins",
            return_value={"x": _DummyClassifier},
        ):
            clf = load_media_classifier(
                config={"media_classifier_plugin": "x"}
            )
        self.assertIsInstance(clf, _DummyClassifier)
        self.assertEqual(clf.classify("anything", "en-us")[0], MediaType.MOVIE)


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


class TestLabelToMediaTypeMapping(unittest.TestCase):
    """Verify raw detection label → mediavocab.MediaType + genres is correct.

    There is no per-media-type intent layer: each raw label resolves directly
    to a ``mediavocab.MediaType`` (and any genre tags), with several labels
    deliberately collapsing onto one type (the lost nuance survives as genres).
    """

    # Expected collapse of the raw label space onto mediavocab types.
    _EXPECTED = {
        "music":             MediaType.MUSIC,
        "podcast":           MediaType.PODCAST,
        "radio":             MediaType.RADIO,
        "audiobook":         MediaType.AUDIOBOOK,
        "news":              MediaType.RADIO,
        "movie":             MediaType.MOVIE,
        "tv":                MediaType.TV,
        "tv_show":           MediaType.EPISODIC_SERIES,
        "video":             MediaType.MOVIE,
        "video_episodes":    MediaType.EPISODIC_SERIES,
        "audio":             MediaType.MUSIC,
        "game":              MediaType.GAME,
        "anime":             MediaType.EPISODIC_SERIES,
        "cartoon":           MediaType.EPISODIC_SERIES,
        "documentary":       MediaType.MOVIE,
        "short_film":        MediaType.SHORT_FILM,
        "silent_movie":      MediaType.MOVIE,
        "bw_movie":          MediaType.MOVIE,
        "radio_theatre":     MediaType.AUDIO_DRAMA,
        "visual_story":      MediaType.COMIC,
        "asmr":              MediaType.PROCEDURAL_AMBIENT,
        "audio_description": MediaType.MOVIE,
        "music_video":       MediaType.MUSIC_VIDEO,
        "trailer":           MediaType.MOVIE,
        "teaser":            MediaType.MOVIE,
        "behind_the_scenes": MediaType.MOVIE,
        "making_of":         MediaType.MOVIE,
        "bloopers":          MediaType.MOVIE,
        "deleted_scenes":    MediaType.MOVIE,
        "featurette":        MediaType.MOVIE,
        "interview":         MediaType.MOVIE,
        "clip":              MediaType.MOVIE,
        "adult":             MediaType.MOVIE,
        "adult_audio":       MediaType.MUSIC,
        "hentai":            MediaType.EPISODIC_SERIES,
        "book":              MediaType.BOOK,
        "playlist":          MediaType.PLAYLIST,
        "sound_effect":      MediaType.SOUND_EFFECT,
        "interactive_fiction": MediaType.INTERACTIVE_FICTION,
        "ambient":           MediaType.PROCEDURAL_AMBIENT,
        "comic":             MediaType.COMIC,
        "generic":           MediaType.GENERIC,
    }

    def test_tv_show_maps_to_episodic_series_not_tv(self):
        self.assertEqual(LABEL_TO_MEDIA_TYPE["tv_show"],
                         MediaType.EPISODIC_SERIES)
        self.assertNotEqual(LABEL_TO_MEDIA_TYPE["tv_show"], MediaType.TV)

    def test_tv_label_maps_to_tv(self):
        self.assertEqual(LABEL_TO_MEDIA_TYPE["tv"], MediaType.TV)

    def test_trailer_label_collapses_to_movie(self):
        self.assertEqual(LABEL_TO_MEDIA_TYPE["trailer"], MediaType.MOVIE)

    def test_behind_the_scenes_label_collapses_to_movie(self):
        self.assertEqual(LABEL_TO_MEDIA_TYPE["behind_the_scenes"], MediaType.MOVIE)

    def test_full_mapping_matches_expected(self):
        self.assertEqual(dict(LABEL_TO_MEDIA_TYPE), self._EXPECTED)

    def test_every_value_is_a_mediavocab_type(self):
        for label, mt in LABEL_TO_MEDIA_TYPE.items():
            with self.subTest(label=label):
                self.assertIsInstance(mt, mediavocab.MediaType)

    def test_genre_labels_carry_genres(self):
        self.assertEqual(LABEL_TO_GENRES["anime"], ["anime"])
        self.assertEqual(LABEL_TO_GENRES["cartoon"], ["animation"])
        self.assertEqual(LABEL_TO_GENRES["asmr"], ["asmr"])
        self.assertEqual(LABEL_TO_GENRES["adult"], ["adult"])
        # hentai collapses to EPISODIC_SERIES but carries BOTH anime + adult
        self.assertIn("anime", LABEL_TO_GENRES["hentai"])
        self.assertIn("adult", LABEL_TO_GENRES["hentai"])

    def test_neutral_labels_have_no_genres(self):
        for label in ("music", "movie", "podcast", "game", "tv_show"):
            with self.subTest(label=label):
                self.assertEqual(genres_for_label(label), [])

    def test_every_genre_is_a_known_mediavocab_genre(self):
        from mediavocab.taxonomy.genre import KNOWN_GENRES
        for label, genres in LABEL_TO_GENRES.items():
            for g in genres:
                with self.subTest(label=label, genre=g):
                    self.assertIn(g, KNOWN_GENRES)


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


class TestSupplementaryContentForm(unittest.TestCase):
    """Supplementary content (trailer / BTS / bloopers / …) is MOVIE + a
    mediavocab ``ContentForm``, never a bare MOVIE with the signal lost.

    The finer classifier labels collapse onto mediavocab's set (making_of /
    bloopers / deleted_scenes / featurette → ``behind_scenes``; clip →
    ``excerpt``; interview → ``supplement``)."""

    def setUp(self):
        from mediavocab.taxonomy import ContentForm
        self.ContentForm = ContentForm
        # bundled-locale standalone classifier (real .voc word-boundary match)
        self.clf = KeywordMediaClassifier()

    def _check(self, query, expected_form):
        mt, _conf = self.clf.classify(query, "en-us")
        form = self.clf.classify_content_form(query, "en-us")
        self.assertEqual(mt, MediaType.MOVIE, f"{query!r} should be MOVIE")
        self.assertEqual(form, expected_form,
                         f"{query!r} → {form}, expected {expected_form!r}")

    def test_trailer_with_title(self):
        self._check("the Top Gun trailer", self.ContentForm.TRAILER)

    def test_teaser(self):
        self._check("the Dune teaser", self.ContentForm.TEASER)

    def test_behind_the_scenes(self):
        self._check("behind the scenes of Dune", self.ContentForm.BEHIND_SCENES)

    def test_making_of(self):
        self._check("the making of Inception", self.ContentForm.BEHIND_SCENES)

    def test_bloopers_no_title(self):
        # the regression case: "show me bloopers" must still fire the axis
        self._check("show me bloopers", self.ContentForm.BEHIND_SCENES)

    def test_deleted_scenes(self):
        self._check("the deleted scenes from Avatar",
                    self.ContentForm.BEHIND_SCENES)

    def test_featurette(self):
        self._check("the Dune featurette", self.ContentForm.BEHIND_SCENES)

    def test_interview(self):
        self._check("cast interview for Barbie", self.ContentForm.SUPPLEMENT)

    def test_clip(self):
        self._check("a movie clip", self.ContentForm.EXCERPT)

    def test_silent_and_bw_are_picture_format(self):
        # bw/silent are mediavocab PictureFormat presentation attributes (T6),
        # NOT content_form.
        from mediavocab import PictureFormat
        self.assertIn(PictureFormat.SILENT,
                      self.clf.classify_picture_format("a silent film", "en-us"))
        self.assertIn(PictureFormat.BLACK_AND_WHITE,
                      self.clf.classify_picture_format("a black and white movie",
                                                       "en-us"))

    def test_label_to_content_form_map(self):
        from ovos_media_classifier.intents import (
            LABEL_TO_CONTENT_FORM, content_form_for_label,
        )
        self.assertEqual(LABEL_TO_CONTENT_FORM["trailer"],
                         self.ContentForm.TRAILER)
        self.assertEqual(LABEL_TO_CONTENT_FORM["behind_the_scenes"],
                         self.ContentForm.BEHIND_SCENES)
        self.assertEqual(content_form_for_label("bloopers"),
                         self.ContentForm.BEHIND_SCENES)
        self.assertIsNone(content_form_for_label("music"))


if __name__ == "__main__":
    unittest.main()
