"""Tests for the content filter (detect-to-block) and external plugin discovery."""
import unittest
from unittest.mock import patch

from mediavocab import MediaType

from ovos_media_classifier import (
    ContentFilter,
    KeywordMediaClassifier,
    find_media_classifier_plugins,
    load_media_classifier_plugin,
)
from ovos_media_classifier.base import AbstractMediaClassifier


class TestContentFilterDefaults(unittest.TestCase):
    def test_adult_genre_blocked_by_default(self):
        cf = ContentFilter()
        blocked, reason = cf.is_blocked(MediaType.MOVIE, ["adult"])
        self.assertTrue(blocked)
        self.assertIn("adult", reason)

    def test_non_adult_allowed(self):
        cf = ContentFilter()
        self.assertEqual(cf.is_blocked(MediaType.MUSIC, []), (False, ""))
        self.assertEqual(cf.is_blocked(MediaType.EPISODIC_SERIES, ["anime"]), (False, ""))

    def test_allow_adult_content_lifts_block(self):
        cf = ContentFilter({"allow_adult_content": True})
        self.assertEqual(cf.is_blocked(MediaType.MOVIE, ["adult"]), (False, ""))

    def test_disabled_filter_allows_everything(self):
        cf = ContentFilter({"media_content_filter": {"enabled": False}})
        self.assertEqual(cf.is_blocked(MediaType.MOVIE, ["adult"]), (False, ""))

    def test_block_media_type(self):
        cf = ContentFilter({"media_content_filter": {"blocked_media_types": ["game"]}})
        blocked, reason = cf.is_blocked(MediaType.GAME, [])
        self.assertTrue(blocked)
        self.assertIn("game", reason)

    def test_custom_blocked_genre(self):
        cf = ContentFilter({"media_content_filter": {"blocked_genres": ["asmr"]}})
        # custom list replaces default, so adult is no longer blocked
        self.assertTrue(cf.is_blocked(MediaType.PROCEDURAL_AMBIENT, ["asmr"])[0])
        self.assertFalse(cf.is_blocked(MediaType.MOVIE, ["adult"])[0])


class TestContentFilterWithClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_check_blocks_hentai_query(self):
        cf = ContentFilter()
        blocked, reason = cf.check(self.clf, "play hentai", "en-us")
        self.assertTrue(blocked)

    def test_check_blocks_porn_query(self):
        cf = ContentFilter()
        self.assertTrue(cf.check(self.clf, "play some porn", "en-us")[0])

    def test_check_allows_music_query(self):
        cf = ContentFilter()
        self.assertFalse(cf.check(self.clf, "play a podcast", "en-us")[0])


class _DummyClassifier(AbstractMediaClassifier):
    def classify(self, query, lang, valid_labels=None):
        return MediaType.MUSIC, 1.0


class TestExternalPluginDiscovery(unittest.TestCase):
    def test_find_returns_dict(self):
        # no external classifiers installed in the test env
        self.assertIsInstance(find_media_classifier_plugins(), dict)

    def test_load_unknown_raises(self):
        with self.assertRaises(ValueError):
            load_media_classifier_plugin("does-not-exist")

    def test_load_named_plugin(self):
        with patch(
            "ovos_media_classifier.plugins.find_media_classifier_plugins",
            return_value={"dummy": _DummyClassifier},
        ):
            clf = load_media_classifier_plugin("dummy", {})
            self.assertIsInstance(clf, _DummyClassifier)
            self.assertEqual(clf.classify("x", "en-us")[0], MediaType.MUSIC)

    def test_factory_selects_external_plugin(self):
        from ovos_media_classifier import load_media_classifier
        with patch(
            "ovos_media_classifier.plugins.find_media_classifier_plugins",
            return_value={"dummy": _DummyClassifier},
        ):
            clf = load_media_classifier({"media_classifier_plugin": "dummy"})
            self.assertIsInstance(clf, _DummyClassifier)


if __name__ == "__main__":
    unittest.main()
