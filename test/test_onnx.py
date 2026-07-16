"""Unit tests for the optional ONNX trained backend.

onnxruntime is **mocked** throughout: there is no real model file in the suite,
and the suite must pass whether or not onnxruntime/numpy are installed.  We feed
the backend fake ``InferenceSession`` objects that return controlled logits and
assert it maps them to the right ``mediavocab.MediaType`` + axes + genres, and
that the factory selects it when configured and falls back to keyword when the
backend can't be loaded.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import mediavocab

from ovos_media_classifier.axes import Structure
from ovos_media_classifier.intents import MediaType, OCPDomain
from ovos_media_classifier.keyword import KeywordMediaClassifier
from ovos_media_classifier.onnx import OnnxMediaClassifier
from ovos_media_classifier.features import CategoricalFeatureExtractor


# ---------------------------------------------------------------------------
# Helpers — a fake onnxruntime InferenceSession returning canned logits.
# ---------------------------------------------------------------------------

class _FakeSession:
    """Stand-in for ``onnxruntime.InferenceSession``.

    Returns a fixed ``(1, n_classes)`` logits row regardless of input, and
    advertises a single input named ``features``.
    """

    def __init__(self, logits):
        self._logits = logits

    def get_inputs(self):
        inp = MagicMock()
        inp.name = "features"
        return [inp]

    def run(self, output_names, feed):
        import numpy as np
        return [np.asarray([self._logits], dtype="float32")]


def _clf(domain_logits, play_logits, *, feature_names=None,
         domain_thresh=0.5, play_thresh=0.3):
    """Build an OnnxMediaClassifier wired to fake sessions.

    domain_labels: idx 0 -> ocp_play, 1 -> ocp_control, 2 -> not_ocp
    play_labels:   idx 0 -> music, 1 -> movie, 2 -> hentai, 3 -> generic
    """
    feature_names = feature_names or ["kw_music", "kw_movie", "kw_hentai"]
    extractor = CategoricalFeatureExtractor(voc_matcher=None)  # no keyword fires
    return OnnxMediaClassifier(
        domain_session=_FakeSession(domain_logits),
        play_session=_FakeSession(play_logits),
        feature_names=feature_names,
        domain_labels={0: "ocp_play", 1: "ocp_control", 2: "not_ocp"},
        play_labels={0: "music", 1: "movie", 2: "hentai", 3: "generic"},
        extractor=extractor,
        domain_threshold=domain_thresh,
        play_threshold=play_thresh,
    )


# ---------------------------------------------------------------------------
# Inference / axis mapping
# ---------------------------------------------------------------------------

class TestOnnxInference(unittest.TestCase):
    def test_classify_music(self):
        # domain argmax -> ocp_play (idx 0), play argmax -> music (idx 0)
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        mt, conf = clf.classify("play some jazz", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertGreater(conf, 0.5)

    def test_classify_movie(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[0.0, 5.0, 0.0, 0.0])
        mt, _ = clf.classify("play a movie", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)

    def test_not_ocp_domain_gates_to_generic(self):
        # domain argmax -> not_ocp (idx 2): classify must short-circuit to GENERIC
        clf = _clf(domain_logits=[0.0, 0.0, 5.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        mt, conf = clf.classify("what time is it", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_low_domain_confidence_gates_to_not_ocp(self):
        # near-uniform logits -> softmax conf below the 0.5 domain threshold
        clf = _clf(domain_logits=[0.1, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        domain, _ = clf.classify_domain("ambiguous", "en-us")
        self.assertEqual(domain, OCPDomain.NOT_OCP)

    def test_low_play_confidence_gates_to_generic(self):
        # ocp_play passes, but play head is near-uniform (< 0.3 threshold)
        clf = _clf(domain_logits=[5.0, 0.0, 0.0],
                   play_logits=[0.01, 0.0, 0.0, 0.0])
        mt, _ = clf.classify("play something", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)

    def test_valid_labels_filter(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        # music predicted, but only MOVIE allowed -> GENERIC
        mt, _ = clf.classify("play jazz", "en-us", valid_labels=[MediaType.MOVIE])
        self.assertEqual(mt, MediaType.GENERIC)

    def test_classify_domain_play(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        domain, conf = clf.classify_domain("play music", "en-us")
        self.assertEqual(domain, OCPDomain.OCP_PLAY)
        self.assertGreater(conf, 0.5)

    def test_classify_domain_control(self):
        clf = _clf(domain_logits=[0.0, 5.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        domain, _ = clf.classify_domain("pause", "en-us")
        self.assertEqual(domain, OCPDomain.OCP_CONTROL)

    def test_is_ocp_query_true(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        is_ocp, _ = clf.is_ocp_query("play music", "en-us")
        self.assertTrue(is_ocp)

    def test_is_ocp_query_false(self):
        clf = _clf(domain_logits=[0.0, 0.0, 5.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        is_ocp, _ = clf.is_ocp_query("what's the weather", "en-us")
        self.assertFalse(is_ocp)

    def test_classify_genres_hentai(self):
        # play argmax -> hentai (idx 2) -> genres ["anime", "adult"]
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[0.0, 0.0, 5.0, 0.0])
        genres = clf.classify_genres("play hentai", "en-us")
        self.assertIn("adult", genres)
        self.assertIn("anime", genres)

    def test_classify_genres_empty_when_not_ocp(self):
        clf = _clf(domain_logits=[0.0, 0.0, 5.0], play_logits=[0.0, 0.0, 5.0, 0.0])
        self.assertEqual(clf.classify_genres("hello", "en-us"), [])


# ---------------------------------------------------------------------------
# Multi-axis output (classify_full / playback_type / structure)
# ---------------------------------------------------------------------------

class TestOnnxMultiAxis(unittest.TestCase):
    def test_classify_full_music(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        result = clf.classify_full("play jazz", "en-us")
        self.assertEqual(result.media_type, MediaType.MUSIC)
        self.assertEqual(result.domain, OCPDomain.OCP_PLAY)
        self.assertEqual(result.playback_type,
                         mediavocab.infer_playback_type(MediaType.MUSIC))
        self.assertEqual(result.structure, Structure.SINGLE)
        self.assertGreater(result.confidence, 0.5)

    def test_classify_full_genres_carried(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[0.0, 0.0, 5.0, 0.0])
        result = clf.classify_full("play hentai", "en-us")
        self.assertIn("adult", result.genres)
        # hentai maps to EPISODIC_SERIES -> EPISODIC structure
        self.assertEqual(result.media_type, MediaType.EPISODIC_SERIES)
        self.assertEqual(result.structure, Structure.EPISODIC)

    def test_classify_full_not_ocp(self):
        clf = _clf(domain_logits=[0.0, 0.0, 5.0], play_logits=[5.0, 0.0, 0.0, 0.0])
        result = clf.classify_full("hello", "en-us")
        self.assertEqual(result.media_type, MediaType.GENERIC)
        self.assertEqual(result.domain, OCPDomain.NOT_OCP)

    def test_playback_type_and_structure_axes(self):
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[0.0, 5.0, 0.0, 0.0])
        self.assertEqual(clf.classify_playback_type("play a movie", "en-us"),
                         mediavocab.infer_playback_type(MediaType.MOVIE))
        self.assertEqual(clf.classify_structure("play a movie", "en-us"),
                         Structure.SINGLE)


# ---------------------------------------------------------------------------
# Vectorization order (feature_names is the contract)
# ---------------------------------------------------------------------------

class TestOnnxVectorize(unittest.TestCase):
    def test_vectorize_respects_feature_order(self):
        import numpy as np
        clf = _clf(domain_logits=[5.0, 0.0, 0.0], play_logits=[5.0, 0.0, 0.0, 0.0],
                   feature_names=["a", "b", "c", "d"])
        row = clf._vectorize({"b": "1", "d": "1"})
        np.testing.assert_array_equal(row, np.array([[0.0, 1.0, 0.0, 1.0]],
                                                     dtype="float32"))


# ---------------------------------------------------------------------------
# Factory wiring — selection + graceful fallback
# ---------------------------------------------------------------------------

class TestFactoryOnnxSelection(unittest.TestCase):
    def test_factory_selects_onnx_when_configured(self):
        from ovos_media_classifier import load_media_classifier

        sentinel = _clf(domain_logits=[5.0, 0.0, 0.0],
                        play_logits=[5.0, 0.0, 0.0, 0.0])
        with patch.object(OnnxMediaClassifier, "from_path",
                          return_value=sentinel) as mock_from_path:
            clf = load_media_classifier({"media_classifier_onnx_model": "/some/bundle"})
        mock_from_path.assert_called_once_with("/some/bundle")
        self.assertIs(clf, sentinel)

    def test_factory_falls_back_when_onnxruntime_missing(self):
        from ovos_media_classifier import load_media_classifier

        with patch.object(OnnxMediaClassifier, "from_path",
                          side_effect=ImportError("No module named 'onnxruntime'")):
            clf = load_media_classifier({"media_classifier_onnx_model": "/some/bundle"})
        # graceful fallback to the lean keyword classifier
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_factory_falls_back_on_bad_bundle(self):
        from ovos_media_classifier import load_media_classifier

        with patch.object(OnnxMediaClassifier, "from_path",
                          side_effect=FileNotFoundError("missing domain.onnx")):
            clf = load_media_classifier({"media_classifier_onnx_model": "/bad/bundle"})
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_factory_keyword_default_no_onnx_config(self):
        from ovos_media_classifier import load_media_classifier

        clf = load_media_classifier({})
        self.assertIsInstance(clf, KeywordMediaClassifier)


# ---------------------------------------------------------------------------
# Bundle loading via from_path with a fully mocked onnxruntime module
# ---------------------------------------------------------------------------

class TestFromPath(unittest.TestCase):
    def test_from_path_reads_meta_and_builds_sessions(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            # touch the required bundle files
            open(os.path.join(d, "domain.onnx"), "wb").close()
            open(os.path.join(d, "play.onnx"), "wb").close()
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump({
                    "feature_names": ["kw_music", "kw_movie"],
                    "domain_labels": {"0": "ocp_play", "1": "not_ocp"},
                    "play_labels": {"0": "music", "1": "movie"},
                }, fh)

            fake_ort = MagicMock()
            fake_ort.InferenceSession.side_effect = \
                lambda path: _FakeSession([5.0, 0.0, 0.0, 0.0])
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                clf = OnnxMediaClassifier.from_path(d)

        self.assertEqual(clf._feature_names, ["kw_music", "kw_movie"])
        self.assertEqual(clf._domain_labels, {0: "ocp_play", 1: "not_ocp"})
        self.assertEqual(clf._play_labels, {0: "music", 1: "movie"})

    def test_from_path_missing_file_raises(self):
        import tempfile

        fake_ort = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                with self.assertRaises(FileNotFoundError):
                    OnnxMediaClassifier.from_path(d)


class TestAxisEnumCoercion(unittest.TestCase):
    """The shared head→enum coercion helpers, exercised without a real bundle.

    A bare instance (no onnx sessions) has its primitive head accessors stubbed
    so only the coercion contract is under test: valid labels become enums,
    invalid labels are dropped, and a missing/abstaining head yields ``None``
    so the caller falls back to the inherited default.
    """

    def setUp(self):
        from mediavocab.taxonomy import ContentForm, AccessibilityKind
        self.ContentForm = ContentForm
        self.AccessibilityKind = AccessibilityKind
        # object.__new__ skips __init__ (no onnx sessions needed); we drive the
        # helpers purely through the _has_head/_single_head/_multi_head primitives.
        self.clf = object.__new__(OnnxMediaClassifier)

    def test_single_head_valid_label_coerces_to_enum(self):
        self.clf._single_head = lambda head, q, lang: ("trailer", 0.9)
        self.assertEqual(
            self.clf._enum_from_single_head("content_form", self.ContentForm, "q", "en-us"),
            self.ContentForm.TRAILER)

    def test_single_head_invalid_label_returns_none(self):
        # An out-of-vocabulary label must not raise — the caller falls back.
        self.clf._single_head = lambda head, q, lang: ("not_a_form", 0.9)
        self.assertIsNone(
            self.clf._enum_from_single_head("content_form", self.ContentForm, "q", "en-us"))

    def test_single_head_abstain_returns_none(self):
        for res in (None, ("", 0.0)):
            self.clf._single_head = lambda head, q, lang, _r=res: _r
            self.assertIsNone(
                self.clf._enum_from_single_head("content_form", self.ContentForm, "q", "en-us"))

    def test_multi_head_skips_invalid_keeps_valid(self):
        self.clf._has_head = lambda head: True
        self.clf._multi_head = lambda head, q, lang: ["subtitles", "bogus", "sign_language"]
        self.assertEqual(
            self.clf._enums_from_multi_head("accessibility", self.AccessibilityKind, "q", "en-us"),
            [self.AccessibilityKind.SUBTITLES, self.AccessibilityKind.SIGN_LANGUAGE])

    def test_multi_head_missing_head_returns_none(self):
        self.clf._has_head = lambda head: False
        self.assertIsNone(
            self.clf._enums_from_multi_head("accessibility", self.AccessibilityKind, "q", "en-us"))

    def test_multi_head_abstain_returns_none(self):
        self.clf._has_head = lambda head: True
        self.clf._multi_head = lambda head, q, lang: None
        self.assertIsNone(
            self.clf._enums_from_multi_head("accessibility", self.AccessibilityKind, "q", "en-us"))

    def test_multi_head_all_invalid_returns_empty_list(self):
        # A present head whose labels are all OOV coerces to [] (not None): the
        # head fired, it just yielded nothing valid — distinct from "no head".
        self.clf._has_head = lambda head: True
        self.clf._multi_head = lambda head, q, lang: ["bogus", "nope"]
        self.assertEqual(
            self.clf._enums_from_multi_head("accessibility", self.AccessibilityKind, "q", "en-us"),
            [])


if __name__ == "__main__":
    unittest.main()
