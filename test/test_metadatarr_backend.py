"""Tests for Layer B — the online metadatarr backend + hybrid fall-through.

metadatarr is mocked throughout (no network).  Covers: resolve→media_type
mapping, playback/genre/programme_format derivation, the robustness contract
(timeout / failure / empty / low-confidence → abstain, never raise),
lazy-import-when-disabled, and the hybrid's layered fall-through order
(keyword → offline router → online, later layers only fill abstentions).
"""
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from mediavocab import MediaType as MV, PlaybackType, Signals

from ovos_media_classifier.intents import MediaType, OCPDomain
from ovos_media_classifier.metadatarr_backend import MetadatarrMediaClassifier
from ovos_media_classifier.embedding import HybridMediaClassifier, GENERIC

from test.test_embedding import _router


def _fake_result(medium, *, confidence=0.9, year=None,
                 playback_type=None, content_genres=None):
    r = MagicMock()
    r.signals = Signals(title="X", medium=medium, year=year,
                        playback_type=playback_type,
                        content_genres=content_genres or [])
    m = MagicMock()
    m.confidence = confidence
    r.accepted = [m]
    return r


def _patch_resolve(fn):
    """Patch metadatarr.resolve.resolve with *fn* (network never hit).

    metadatarr is an optional (``[online]``-extra) dependency the ``[test]``
    extra deliberately excludes — the suite mocks it rather than requiring the
    real package.  We inject fake ``metadatarr`` / ``metadatarr.resolve``
    modules into ``sys.modules`` so the backend's lazy
    ``from metadatarr.resolve import resolve`` resolves to *fn* whether or not
    the real package is installed.
    """
    resolve_mod = types.ModuleType("metadatarr.resolve")
    resolve_mod.resolve = fn
    metadatarr_mod = types.ModuleType("metadatarr")
    metadatarr_mod.resolve = resolve_mod
    return patch.dict(sys.modules, {"metadatarr": metadatarr_mod,
                                    "metadatarr.resolve": resolve_mod})


class TestResolveMapping(unittest.TestCase):
    def test_resolve_maps_medium_to_media_type(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        with _patch_resolve(lambda s, max_workers=8: _fake_result(MV.MOVIE)):
            mt, conf = clf.classify("the matrix", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertAlmostEqual(conf, 0.9)

    def test_resolve_derives_playback(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        with _patch_resolve(lambda s, max_workers=8: _fake_result(MV.MUSIC)):
            self.assertEqual(clf.classify_playback_type("x", "en-us"),
                             PlaybackType.AUDIO)

    def test_resolve_uses_explicit_playback_when_set(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        res = _fake_result(MV.GENERIC, playback_type=PlaybackType.VIDEO)
        with _patch_resolve(lambda s, max_workers=8: res):
            # GENERIC medium → classify abstains, but playback head still derives
            self.assertEqual(clf.classify_playback_type("x", "en-us"),
                             PlaybackType.VIDEO)

    def test_domain_play_on_confident_leaf(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        with _patch_resolve(lambda s, max_workers=8: _fake_result(MV.MOVIE)):
            self.assertEqual(clf.classify_domain("x", "en-us")[0],
                             OCPDomain.OCP_PLAY)

    def test_to_signals_keeps_spoken_title(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        with _patch_resolve(lambda s, max_workers=8: _fake_result(MV.MOVIE, year=1999)):
            sig = clf.to_signals("the matrix", "en-us")
        self.assertEqual(sig.medium, MV.MOVIE)
        self.assertEqual(sig.title, "the matrix")


class TestRobustness(unittest.TestCase):
    def test_failure_abstains(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)

        def boom(s, max_workers=8):
            raise RuntimeError("network down")

        with _patch_resolve(boom):
            self.assertEqual(clf.classify("x", "en-us"), (MediaType.GENERIC, 0.0))

    def test_timeout_abstains(self):
        clf = MetadatarrMediaClassifier(timeout_s=0.2)

        def slow(s, max_workers=8):
            time.sleep(2.0)
            return None

        with _patch_resolve(slow):
            self.assertEqual(clf.classify("x", "en-us"), (MediaType.GENERIC, 0.0))

    def test_timeout_returns_before_hung_call_finishes(self):
        # adversarial: the resolve hangs far longer than the budget; classify
        # must return within roughly the timeout, not block until the hang ends.
        clf = MetadatarrMediaClassifier(timeout_s=0.2)
        hang = threading.Event()

        def never_returns(s, max_workers=8):
            hang.wait(30)  # abandoned daemon thread; released on teardown
            return _fake_result(MV.MOVIE)

        with _patch_resolve(never_returns):
            start = time.monotonic()
            result = clf.classify("x", "en-us")
            elapsed = time.monotonic() - start
        hang.set()
        self.assertEqual(result, (MediaType.GENERIC, 0.0))
        self.assertLess(elapsed, 2.0)  # would be ~30s with the join bug

    def test_empty_result_abstains(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        with _patch_resolve(lambda s, max_workers=8: None):
            self.assertEqual(clf.classify("x", "en-us")[0], MediaType.GENERIC)

    def test_low_confidence_abstains(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0, min_confidence=0.5)
        with _patch_resolve(lambda s, max_workers=8: _fake_result(MV.MOVIE, confidence=0.3)):
            self.assertEqual(clf.classify("x", "en-us")[0], MediaType.GENERIC)

    def test_no_medium_abstains(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        res = MagicMock()
        res.signals = Signals(title="X")  # no medium
        m = MagicMock(); m.confidence = 0.9
        res.accepted = [m]
        with _patch_resolve(lambda s, max_workers=8: res):
            self.assertEqual(clf.classify("x", "en-us")[0], MediaType.GENERIC)

    def test_empty_query_abstains_without_network(self):
        clf = MetadatarrMediaClassifier(timeout_s=2.0)
        called = MagicMock()
        with _patch_resolve(lambda s, max_workers=8: called()):
            self.assertEqual(clf.classify("", "en-us")[0], MediaType.GENERIC)
        called.assert_not_called()


class TestLazyImport(unittest.TestCase):
    def test_construct_does_not_import_metadatarr(self):
        # constructing the backend must not pull metadatarr into sys.modules
        # (only an actual classify() does the lazy import)
        mods = {k for k in sys.modules if k.startswith("metadatarr")}
        for k in mods:
            sys.modules.pop(k, None)
        clf = MetadatarrMediaClassifier()
        self.assertFalse(any(k.startswith("metadatarr") for k in sys.modules),
                         "metadatarr must not be imported until classify()")
        # touching the API lazy-imports it
        del clf


class TestHybridFallThrough(unittest.TestCase):
    def _hybrid_with_online(self, online):
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      entity_labels=["anime_title"])
        from ovos_media_classifier.keyword import KeywordMediaClassifier
        return HybridMediaClassifier(KeywordMediaClassifier(), clf, online=online)

    def test_online_not_consulted_on_confident_keyword(self):
        online = MagicMock()
        online.classify.return_value = (MediaType.BOOK, 0.9)
        h = self._hybrid_with_online(online)
        mt, _ = h.classify("play some jazz music", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        online.classify.assert_not_called()

    def test_online_fills_when_all_cheaper_abstain(self):
        online = MagicMock()
        online.classify.return_value = (MediaType.BOOK, 0.85)
        h = self._hybrid_with_online(online)
        mt, conf = h.classify("zzz qqq vvv", "en-us")
        self.assertEqual(mt, MediaType.BOOK)
        online.classify.assert_called()

    def test_no_online_layer_stays_offline(self):
        h = self._hybrid_with_online(None)
        # no online wired → abstains safely, never raises
        self.assertEqual(h.classify("zzz qqq vvv", "en-us")[0], MediaType.GENERIC)

    def test_online_never_moves_gate_or_adult(self):
        # the gate + adult policy stay on keyword even with online wired
        online = MagicMock()
        online.classify.return_value = (MediaType.MOVIE, 0.9)
        online.classify_domain.return_value = (OCPDomain.OCP_PLAY, 0.9)
        h = self._hybrid_with_online(online)
        # a non-media utterance: keyword gate must keep it NOT_OCP
        domain, _ = h.classify_domain("what time is it", "en-us")
        self.assertEqual(domain, OCPDomain.NOT_OCP)
        online.classify_domain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
