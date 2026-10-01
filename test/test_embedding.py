"""Unit tests for the embedding-router backend.

onnxruntime / numpy / guided-categorical-embeddings are **not** required: the
``_AxisHead`` is built by hand with a fake softmax-emitting session and a stub
vectorizer, exactly like ``test_onnx.py`` feeds the ONNX backend canned logits.
This asserts the routing-aware behaviour (calibrated route, reject → GENERIC,
runtime entity injection) and the hybrid gating (keyword first pass, no
regression of the gate / adult axes) without any model file.
"""
import unittest

import numpy as np

from ovos_media_classifier.embedding import (
    EmbeddingMediaClassifier,
    HybridMediaClassifier,
    GENERIC,
    _AxisHead,
    _NumpyEntityMatcher,
)
from ovos_media_classifier.features import CategoricalFeatureExtractor
from ovos_media_classifier.intents import MediaType, OCPDomain
from ovos_media_classifier.keyword import KeywordMediaClassifier


# ---------------------------------------------------------------------------
# Helpers — a fake calibrated-softmax session + a stub vectorizer.
# ---------------------------------------------------------------------------

class _FakeSession:
    """Returns a fixed ``(1, n_classes)`` probability row (already a softmax)."""

    def __init__(self, probs):
        self._probs = np.asarray([probs], dtype="float32")

    def get_inputs(self):
        inp = type("I", (), {"name": "features"})()
        return [inp]

    def run(self, output_names, feed):
        return [self._probs]


class _StubVectorizer:
    """Minimal CategoricalVectorizer stand-in: maps a feat dict to a zero row."""

    def __init__(self, n=3):
        self._n = n

    def transform(self, dicts):
        return np.zeros((len(dicts), self._n), dtype="float32")


def _axis_head(probs, labels, *, entity_labels=None, threshold=0.5,
               abstain_label=GENERIC):
    """Build an ``_AxisHead`` without touching disk / onnxruntime / GCE."""
    head = _AxisHead.__new__(_AxisHead)
    head.static_dim = 3
    head.entity_labels = list(entity_labels or [])
    head.labels = list(labels)
    head.abstain_label = abstain_label
    head.temperature = 1.0
    head.threshold = float(threshold)
    head.vectorizer = _StubVectorizer(head.static_dim)
    head._session = _FakeSession(probs)
    head._input_name = "features"
    head.matcher = _NumpyEntityMatcher(head.entity_labels)
    return head


def _router(mt_probs, mt_labels, *, pb_probs=None, pb_labels=None,
            entity_labels=None, mt_threshold=0.5):
    mt = _axis_head(mt_probs, mt_labels, entity_labels=entity_labels,
                    threshold=mt_threshold)
    pb = None
    if pb_probs is not None:
        pb = _axis_head(pb_probs, pb_labels, entity_labels=entity_labels)
    extractor = CategoricalFeatureExtractor(voc_matcher=None)
    return EmbeddingMediaClassifier(mt, pb, extractor)


# ---------------------------------------------------------------------------
# Routing-aware inference
# ---------------------------------------------------------------------------

class TestEmbeddingRouting(unittest.TestCase):
    def test_confident_route(self):
        # music prob 0.9 (> 0.5 threshold) -> MediaType.MUSIC
        clf = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC])
        mt, conf = clf.classify("play jazz", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertGreater(conf, 0.5)

    def test_reject_below_threshold_abstains(self):
        # top prob 0.4 < 0.5 -> reject -> GENERIC
        clf = _router([0.4, 0.35, 0.25], ["music", "movie", GENERIC])
        mt, conf = clf.classify("ambiguous thing", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_argmax_abstain_label_abstains(self):
        # argmax is the trained GENERIC class -> abstain even if confident
        clf = _router([0.1, 0.1, 0.8], ["music", "movie", GENERIC])
        mt, _ = clf.classify("who knows", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)

    def test_valid_labels_filter(self):
        clf = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC])
        mt, _ = clf.classify("play jazz", "en-us", valid_labels=[MediaType.MOVIE])
        self.assertEqual(mt, MediaType.GENERIC)

    def test_domain_derives_from_leaf(self):
        play = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC])
        self.assertEqual(play.classify_domain("play jazz", "en-us")[0],
                         OCPDomain.OCP_PLAY)
        abstain = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC])
        self.assertEqual(abstain.classify_domain("nonsense", "en-us")[0],
                         OCPDomain.NOT_OCP)

    def test_playback_head_routes(self):
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      pb_probs=[0.9, 0.05, 0.05], pb_labels=["audio", "video", GENERIC])
        from mediavocab import PlaybackType
        self.assertEqual(clf.classify_playback_type("x", "en-us"),
                         PlaybackType.AUDIO)


# ---------------------------------------------------------------------------
# Runtime entity injection (no retraining)
# ---------------------------------------------------------------------------

class TestEntityInjection(unittest.TestCase):
    def test_matcher_starts_empty(self):
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      entity_labels=["anime_title", "audiobook_title"])
        # nothing injected -> no entity fires, head abstains -> GENERIC
        self.assertEqual(clf.classify("watch attack on titan", "en-us")[0],
                         MediaType.GENERIC)

    def test_injected_entity_routes_bare_title(self):
        # head abstains, but an injected anime library fires -> EPISODIC_SERIES
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      entity_labels=["anime_title", "audiobook_title"])
        clf.register_user_entities("anime_title", ["Attack on Titan"])
        mt, conf = clf.classify("watch attack on titan", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        self.assertGreater(conf, 0.0)

    def test_injection_no_retrain_audiobook(self):
        clf = _router([0.2, 0.2, 0.6], ["music", "movie", GENERIC],
                      entity_labels=["audiobook_title"])
        clf.register_user_library({"audiobook_title": ["Harry Potter"]})
        self.assertEqual(clf.classify("listen to harry potter", "en-us")[0],
                         MediaType.AUDIOBOOK)

    def test_unknown_entity_label_ignored(self):
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      entity_labels=["anime_title"])
        # registering a label not in the bundle is a no-op (does not raise)
        clf.register_user_entities("not_a_label", ["Whatever"])
        self.assertEqual(clf.classify("play whatever", "en-us")[0],
                         MediaType.GENERIC)

    def test_word_boundary_matching(self):
        m = _NumpyEntityMatcher(["artist_name"])
        m.register("artist_name", ["Pink"])
        self.assertEqual(m.fired_labels("play pink"), ["artist_name"])
        # substring inside another word must not fire
        self.assertEqual(m.fired_labels("play pinkfloyd song"), [])


# ---------------------------------------------------------------------------
# Hybrid gating — keyword first pass, no gate / adult regression.
# ---------------------------------------------------------------------------

class TestHybridGating(unittest.TestCase):
    def _hybrid(self, router):
        return HybridMediaClassifier(KeywordMediaClassifier(), router)

    def test_keyword_confident_route_wins(self):
        # router would route MUSIC, but keyword sees "movie" -> keyword wins
        router = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC])
        hy = self._hybrid(router)
        mt, _ = hy.classify("play the movie inception", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)

    def test_router_fills_keywordless(self):
        # keyword abstains on a bare phrase; router confidently routes MUSIC
        router = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC])
        hy = self._hybrid(router)
        # "play jazz" fires no media-type .voc keyword in the bundle here
        mt, _ = hy.classify("aaa bbb ccc", "en-us")
        # router routes music regardless of (absent) keyword cue
        self.assertEqual(mt, MediaType.MUSIC)

    def test_injected_entity_overrides_keyword_gap(self):
        # keyword routes MOVIE on "watch"; injected anime library overrides it
        router = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                         entity_labels=["anime_title"])
        hy = self._hybrid(router)
        hy.register_user_library({"anime_title": ["Attack on Titan"]})
        mt, _ = hy.classify("watch attack on titan", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)

    def test_gate_stays_keyword(self):
        # a non-OCP query must stay NOT_OCP even though the router is present
        router = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC])
        hy = self._hybrid(router)
        domain, _ = hy.classify_domain("what's the weather", "en-us")
        self.assertEqual(domain, OCPDomain.NOT_OCP)

    def test_fired_entity_does_not_hijack_gate(self):
        # a fired injected entity in plainly-non-OCP speech must NOT flip the
        # gate to OCP — the router never moves the gate (keyword owns it), so a
        # common short title cannot hijack ordinary speech.
        router = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC],
                         entity_labels=["podcast_title"])
        hy = self._hybrid(router)
        hy.register_user_library({"podcast_title": ["The Daily"]})
        # "the daily forecast" is not an OCP request; keyword gates it NOT_OCP
        domain, _ = hy.classify_domain("the daily forecast", "en-us")
        self.assertEqual(domain, OCPDomain.NOT_OCP)
        self.assertFalse(hy.is_ocp_query("the daily forecast", "en-us")[0])

    def test_playback_consistent_with_injected_leaf(self):
        # injected anime title routes EPISODIC_SERIES (video); playback must
        # agree even if the router playback head would say AUDIO.
        from mediavocab import PlaybackType, infer_playback_type
        router = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                         pb_probs=[0.9, 0.05, 0.05],
                         pb_labels=["audio", "video", GENERIC],
                         entity_labels=["anime_title"])
        hy = self._hybrid(router)
        hy.register_user_library({"anime_title": ["Attack on Titan"]})
        mt, _ = hy.classify("watch attack on titan", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        self.assertEqual(hy.classify_playback_type("watch attack on titan", "en-us"),
                         infer_playback_type(MediaType.EPISODIC_SERIES))

    def test_adult_axis_stays_keyword(self):
        # content_form_genres comes from the keyword adult lexicon (0.0 leak)
        router = _router([0.9, 0.05, 0.05], ["music", "movie", GENERIC])
        hy = self._hybrid(router)
        genres = hy.classify_content_form_genres("play some porn", "en-us")
        self.assertIn("adult", genres)


if __name__ == "__main__":
    unittest.main()
