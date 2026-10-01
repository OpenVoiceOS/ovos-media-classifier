"""Tests for the content filter (detect-to-block) and external plugin discovery."""
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from mediavocab import MediaType

from ovos_media_classifier import (
    ContentFilter,
    KeywordMediaClassifier,
    find_media_classifier_plugins,
    load_media_classifier_plugin,
)
from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.keyword import _LOCALE_DIR


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



class TestAdultBlockIsLanguageIndependent(unittest.TestCase):
    """The adult block must not weaken because the box speaks another language.

    T-6805. The filter matches ``AdultKeyword`` per language. A locale that
    ships a *partial* adult vocabulary shadowed the ``en-us`` fallback, so a
    loanword that every one of these locales uses in speech ("porn") passed
    unblocked on a box configured for that locale, while the same phrase was
    blocked on ``en-us``. A locale with *no* adult vocabulary was already safe,
    because the resolver fell back to ``en-us``: a half-translated locale was
    less safe than an untranslated one.
    """

    # Read off disk rather than written out, so a locale added to the package
    # is covered the day it lands. A hand-written list cannot cover the next
    # partial translation, which is the shape this defect arrives in.
    LANGS = sorted(
        d.name for d in Path(_LOCALE_DIR).iterdir() if d.is_dir()
    )

    def setUp(self):
        self.clf = KeywordMediaClassifier()
        self.cf = ContentFilter()

    def test_english_loanword_blocked_in_every_bundled_locale(self):
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                blocked, reason = self.cf.check(self.clf, "play some porn", lang)
                self.assertTrue(blocked, f"adult request not blocked on {lang}")
                self.assertIn("adult", reason)

    def test_native_adult_term_still_blocked(self):
        # the locale vocabulary keeps working; the en-us union only adds to it
        for lang, phrase in [
            ("nl-nl", "speel wat pornografie"),
            ("de-de", "spiel einen pornofilm"),
            ("pt-pt", "poe um filme porno"),
        ]:
            with self.subTest(lang=lang):
                self.assertTrue(self.cf.check(self.clf, phrase, lang)[0])

    def test_clean_request_not_blocked_in_any_locale(self):
        # the union must not over-block: no en-us adult phrase may fire on a
        # benign request in another language
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                blocked, reason = self.cf.check(self.clf, "play some jazz", lang)
                self.assertFalse(blocked, f"clean request blocked on {lang}: {reason}")



class TestEachHalfOfTheUnionIsPinned(unittest.TestCase):
    """One test per half, so removing either half fails a test.

    Reviewer finding on #72: the full suite passed with EITHER half reverted,
    because the two are redundant for ``classify`` in standalone mode. They are
    not redundant for the feature extractor, which calls the matcher directly
    and never goes through ``_match``, nor for a pipeline host, which never
    goes through the matcher.
    """

    def test_voc_phrases_half_pins_the_feature_extractor(self):
        # CategoricalFeatureExtractor calls _VocMatcher.match directly, so this
        # path exercises the _voc_phrases union and nothing else.
        from ovos_media_classifier.features import CategoricalFeatureExtractor
        extractor = CategoricalFeatureExtractor.from_locale_dir()
        feats = extractor.extract("speel wat porn", lang="nl-nl")
        self.assertEqual(feats.get("kw_adult"), "1",
                         "the nl-nl feature row lost kw_adult: the "
                         "_voc_phrases union is what supplies it")

    def test_match_half_pins_the_pipeline_path(self):
        # A pipeline host supplies voc_match_func, so _VocMatcher never runs
        # and only the _match union can hold the gate. This host owns en-us,
        # which is the condition that half depends on.
        from ovos_spec_tools import LocaleResources
        from ovos_media_classifier.keyword import _fold
        # Resolved straight from LocaleResources, NOT through _VocMatcher, so
        # the host cannot inherit the _voc_phrases union. This is what a real
        # pipeline host is: its own voc_match over its own locale files.
        resources = LocaleResources(skill_locale=_LOCALE_DIR)

        def host_voc_match(phrase, vocab, lang="en-us"):
            phrases = resources.vocabularies(lang).get(vocab) or ()
            folded = _fold(phrase)
            return any(re.search(rf"\b{re.escape(_fold(p))}\b", folded)
                       for p in phrases if _fold(p))

        clf = KeywordMediaClassifier(voc_match_func=host_voc_match)
        cf = ContentFilter()
        blocked, reason = cf.check(clf, "speel wat porn", "nl-nl")
        self.assertTrue(blocked,
                        "a pipeline host that owns en-us must still block: "
                        "the _match union is what asks en-us")


if __name__ == "__main__":
    unittest.main()
