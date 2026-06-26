"""Public-API contract, full-taxonomy coverage, content-filter robustness,
multi-task per-axis output, keyword↔ONNX parity, and edge cases.

These tests exercise the package as an integrator sees it: every ``__all__``
symbol imports and behaves, every non-sentinel ``MediaType`` is reachable from
the bundled keyword backend, the content filter blocks adult/hentai by default
and lifts on opt-in, the multi-axis output (``classify_full`` / ``to_signals``
/ the per-axis ``classify_*`` methods) has the right shape, and odd inputs do
not crash.

The ONNX-parity tests require a trained bundle under ``data/models/`` (produced
by ``python -m training.train_sklearn``); they SKIP cleanly when it is absent so
the suite is green in a bare checkout.
"""
import importlib
import os
import unittest

import mediavocab

import ovos_media_classifier as omc
from ovos_media_classifier import (
    ContentFilter,
    KeywordMediaClassifier,
    MediaType,
    load_media_classifier,
)
from ovos_media_classifier.axes import Structure
from ovos_media_classifier.intents import OCPDomain

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLE = os.path.join(_REPO_ROOT, "data", "models", "context_ner")


def _bundle_available() -> bool:
    return os.path.isfile(os.path.join(_BUNDLE, "meta.json"))


# ---------------------------------------------------------------------------
# Public API contract
# ---------------------------------------------------------------------------

class TestPublicAPI(unittest.TestCase):
    def test_all_symbols_importable(self):
        """Every name in __all__ resolves on the package."""
        for name in omc.__all__:
            self.assertTrue(hasattr(omc, name), f"missing public symbol: {name}")

    def test_all_symbols_are_real(self):
        for name in omc.__all__:
            self.assertIsNotNone(getattr(omc, name))

    def test_factory_returns_classifier(self):
        clf = load_media_classifier()
        self.assertIsInstance(clf, KeywordMediaClassifier)
        mt, conf = clf.classify("play some music", "en-us")
        self.assertIsInstance(mt, MediaType)
        self.assertIsInstance(conf, float)

    def test_classification_dataclass_shape(self):
        clf = load_media_classifier()
        d = clf.classify_full("play a podcast", "en-us").as_dict()
        for key in ("media_type", "playback_type", "structure", "domain",
                    "genres", "confidence", "control_intent"):
            self.assertIn(key, d)


# ---------------------------------------------------------------------------
# Full taxonomy reachability
# ---------------------------------------------------------------------------

class TestTaxonomyCoverage(unittest.TestCase):
    """Every non-sentinel MediaType is reachable from the keyword backend."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def _classify(self, q):
        mt, _ = self.clf.classify(q, "en-us")
        return mt

    def test_each_non_sentinel_type_reachable(self):
        # one clear utterance per leaf type
        cases = {
            MediaType.MUSIC: "play some music",
            MediaType.MUSIC_VIDEO: "play the music video",
            MediaType.PODCAST: "play a podcast",
            MediaType.RADIO: "tune in to the radio",
            MediaType.AUDIOBOOK: "play the audiobook",
            MediaType.BOOK: "read me the book",
            MediaType.AUDIO_DRAMA: "play a radio drama",
            MediaType.MOVIE: "watch a movie",
            MediaType.SHORT_FILM: "play a short film",
            MediaType.EPISODIC_SERIES: "watch the anime",
            MediaType.TV: "put on the tv channel",
            MediaType.COMIC: "read a comic book",
            MediaType.GAME: "launch the game",
            MediaType.INTERACTIVE_FICTION: "play an interactive fiction story",
            MediaType.PLAYLIST: "play my playlist",
            MediaType.SOUND_EFFECT: "play a sound effect",
            MediaType.PROCEDURAL_AMBIENT: "play some ambient sounds",
        }
        for expected, q in cases.items():
            with self.subTest(media_type=expected, query=q):
                self.assertEqual(self._classify(q), expected)

    def test_book_vs_audiobook_split(self):
        """A read verb routes a book to BOOK; an audiobook cue stays AUDIOBOOK."""
        self.assertEqual(self._classify("read me the book"), MediaType.BOOK)
        self.assertEqual(self._classify("play the audiobook"), MediaType.AUDIOBOOK)

    def test_comic_vs_anime_read_vs_watch(self):
        """A comic (paged) and anime (episodic series) are distinct leaves."""
        self.assertEqual(self._classify("read a comic book"), MediaType.COMIC)
        self.assertEqual(self._classify("watch the anime"), MediaType.EPISODIC_SERIES)

    def test_anime_and_cartoon_genres(self):
        self.assertIn("anime", self.clf.classify_genres("watch the anime", "en-us"))
        self.assertIn("animation",
                      self.clf.classify_genres("watch a cartoon", "en-us"))


# ---------------------------------------------------------------------------
# Content-filter robustness
# ---------------------------------------------------------------------------

class TestContentFilterRobustness(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_adult_blocked_by_default(self):
        cf = ContentFilter()
        for q in ("play porn", "play an adult movie", "play hentai"):
            with self.subTest(query=q):
                blocked, reason = cf.check(self.clf, q, "en-us")
                self.assertTrue(blocked, f"{q!r} should be blocked")
                self.assertIn("adult", reason)

    def test_hentai_carries_anime_and_adult(self):
        genres = self.clf.classify_content_form_genres("play hentai", "en-us")
        self.assertIn("adult", genres)
        self.assertIn("anime", genres)

    def test_allow_adult_content_lifts_block(self):
        cf = ContentFilter({"allow_adult_content": True})
        blocked, _ = cf.check(self.clf, "play porn", "en-us")
        self.assertFalse(blocked)

    def test_clean_request_not_blocked(self):
        cf = ContentFilter()
        blocked, _ = cf.check(self.clf, "play some jazz", "en-us")
        self.assertFalse(blocked)

    def test_filter_reads_content_form_genres_axis(self):
        """check() pulls the content-form genres axis (robust blocking)."""

        class _OnlyFormGenres(KeywordMediaClassifier):
            # leaf says GENERIC, but the form-genre axis flags adult
            def classify(self, query, lang, valid_labels=None):
                return MediaType.GENERIC, 0.0

            def classify_content_form_genres(self, query, lang):
                return ["adult"]

        blocked, reason = ContentFilter().check(_OnlyFormGenres(), "x", "en-us")
        self.assertTrue(blocked)
        self.assertIn("adult", reason)


# ---------------------------------------------------------------------------
# Multi-task per-axis output (base contract + defaults)
# ---------------------------------------------------------------------------

class TestPerAxisContract(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_axis_methods_exist_and_typed(self):
        q, lang = "play a black and white movie", "en-us"
        self.assertIsInstance(self.clf.classify_content_form_genres(q, lang), list)
        self.assertIsInstance(self.clf.classify_content_genres(q, lang), list)
        self.assertIsInstance(self.clf.classify_qualifiers(q, lang), list)
        self.assertIsInstance(self.clf.classify_tags(q, lang), list)
        self.assertIn(self.clf.classify_explicitness(q, lang), ("clean", "adult"))
        # mood/era default to None for the keyword backend
        self.assertIsNone(self.clf.classify_mood(q, lang))
        self.assertIsNone(self.clf.classify_era(q, lang))

    def test_tags_are_namespaced(self):
        # every tag the contract emits is namespaced (genre:/mood:/era:)
        tags = self.clf.classify_tags("play some jazz", "en-us")
        for t in tags:
            self.assertRegex(t, r"^(genre|mood|era):")
        # the genre: slice is the genre-classifier view
        self.assertEqual(
            self.clf.tags_namespace(["genre:rock", "mood:chill", "era:1980s"],
                                    "genre"), ["rock"])

    def test_explicitness_derives_from_form_genres(self):
        self.assertEqual(self.clf.classify_explicitness("play porn", "en-us"),
                         "adult")
        self.assertEqual(self.clf.classify_explicitness("play jazz", "en-us"),
                         "clean")

    def test_content_form_genres_defaults_to_genres(self):
        self.assertEqual(
            self.clf.classify_content_form_genres("watch the anime", "en-us"),
            self.clf.classify_genres("watch the anime", "en-us"))

    def test_to_signals_shape(self):
        sig = self.clf.to_signals("play some jazz", "en-us")
        self.assertEqual(sig.title, "play some jazz")
        self.assertEqual(sig.language, "en")
        self.assertIsInstance(sig.content_genres, list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_empty_and_whitespace(self):
        for q in ("", "   ", "\t\n"):
            with self.subTest(query=repr(q)):
                mt, conf = self.clf.classify(q, "en-us")
                self.assertEqual(mt, MediaType.GENERIC)
                domain, _ = self.clf.classify_domain(q, "en-us")
                self.assertEqual(domain, OCPDomain.NOT_OCP)

    def test_garbage_is_not_ocp(self):
        mt, _ = self.clf.classify("asdf qwer zxcv", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        is_ocp, _ = self.clf.is_ocp_query("asdf qwer zxcv", "en-us")
        self.assertFalse(is_ocp)

    def test_control_intent_routes_to_control_domain(self):
        domain, _ = self.clf.classify_domain("pause the music", "en-us")
        self.assertEqual(domain, OCPDomain.OCP_CONTROL)

    def test_unknown_lang_does_not_crash(self):
        # a language with no bundled locale falls back gracefully
        mt, conf = self.clf.classify("play some music", "xx-xx")
        self.assertIsInstance(mt, MediaType)

    def test_to_signals_empty_query(self):
        sig = self.clf.to_signals("", "en-us")
        self.assertIsNone(sig.title)


# ---------------------------------------------------------------------------
# Keyword ↔ ONNX parity (requires a trained bundle; skips otherwise)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_bundle_available(),
                     "no trained bundle under data/models/context_ner")
class TestOnnxParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ovos_media_classifier.onnx import OnnxMediaClassifier
        cls.kw = KeywordMediaClassifier()
        cls.onnx = OnnxMediaClassifier.from_path(_BUNDLE)

    def test_clear_cases_agree_on_media_type(self):
        # The keyword backend (the floor) must nail each unambiguous leaf.
        clear = {
            "play some music": MediaType.MUSIC,
            "watch a movie": MediaType.MOVIE,
            "play a podcast": MediaType.PODCAST,
            "read me a book": MediaType.BOOK,
            "launch a game": MediaType.GAME,
        }
        for q, expected in clear.items():
            with self.subTest(query=q):
                kw_mt, _ = self.kw.classify(q, "en-us")
                self.assertEqual(kw_mt, expected)

    def test_onnx_separable_cases_agree_with_keyword(self):
        # The ONNX backend is run on the keyword-only runtime feature path (no
        # NER store populated — see docs/model.md §6d), where some leaves share
        # all their cue words and are NOT separable from keywords alone
        # (music ↔ music_video both fire only ``kw_music``; game ↔
        # interactive_fiction both fire ``kw_game``).  On the cases that DO have a
        # distinct cue, the trained head agrees with the keyword floor.
        separable = {
            "watch a movie": MediaType.MOVIE,
            "play a podcast": MediaType.PODCAST,
            "read me a book": MediaType.BOOK,
        }
        for q, expected in separable.items():
            with self.subTest(query=q):
                onnx_mt, _ = self.onnx.classify(q, "en-us")
                self.assertEqual(onnx_mt, expected)

    def test_onnx_content_form_genres_flags_adult(self):
        genres = self.onnx.classify_content_form_genres("play porn", "en-us")
        self.assertIn("adult", genres)

    def test_onnx_qualifiers_head(self):
        quals = self.onnx.classify_qualifiers("play a silent film", "en-us")
        self.assertIn("silent", quals)

    def test_onnx_multi_axis_heads_present(self):
        # the multi-task bundle carries one head per axis
        for axis in ("media_type", "playback_type", "structure",
                     "content_form_genres", "qualifiers"):
            self.assertIn(axis, self.onnx._heads)

    def test_onnx_classify_full_shape(self):
        full = self.onnx.classify_full("watch a movie", "en-us")
        self.assertEqual(full.media_type, MediaType.MOVIE)
        self.assertIsInstance(full.playback_type, mediavocab.PlaybackType)
        self.assertIsInstance(full.structure, Structure)
        self.assertEqual(full.domain, OCPDomain.OCP_PLAY)


if __name__ == "__main__":
    unittest.main()
