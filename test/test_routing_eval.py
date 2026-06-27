"""Tests for the harm-weighted out-of-distribution routing eval.

Covers:
  - eval-set integrity: every case is well-formed, labels use the mediavocab
    vocabulary, and the gate/domain/explicit fields are internally consistent;
  - that the set is genuinely out-of-distribution (its utterances are NOT the
    bundled ``locale/*/dataset/*.intent`` template lines);
  - the harm-metric logic: GENERIC/abstain is scored as SAFE (never a mis-route),
    a confident wrong leaf IS a mis-route, false-hijack / false-miss / adult-leak
    are counted on the right slices.

The benchmarks suite is not part of the installed package, so it is imported by
path.
"""
import importlib.util
import json
import os
import sys
import unittest

import mediavocab
from mediavocab import MediaType, PlaybackType

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks")
EVAL_JSONL = os.path.join(BENCH_DIR, "routing_eval.jsonl")


def _load_routing_module():
    spec = importlib.util.spec_from_file_location(
        "routing_eval_mod", os.path.join(BENCH_DIR, "routing_eval.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # register before exec so dataclasses can resolve forward-ref annotations
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


RE = _load_routing_module()


class TestEvalSetIntegrity(unittest.TestCase):
    def setUp(self):
        self.cases = RE.load_cases(EVAL_JSONL)

    def test_nonempty_and_sized(self):
        self.assertGreaterEqual(len(self.cases), 150,
                                "routing eval should be a few hundred cases")

    def test_labels_valid(self):
        media_values = {m.value for m in MediaType}
        pb_values = {p.value for p in PlaybackType}
        domains = {"ocp_play", "ocp_control", "not_ocp"}
        for c in self.cases:
            self.assertTrue(c.utterance.strip(), "empty utterance")
            self.assertIn(c.media_type, media_values, c.utterance)
            self.assertIn(c.playback_type, pb_values, c.utterance)
            self.assertIn(c.domain, domains, c.utterance)

    def test_gate_consistency(self):
        for c in self.cases:
            expect_ocp = c.domain in ("ocp_play", "ocp_control")
            self.assertEqual(c.is_ocp_query, expect_ocp,
                             f"is_ocp_query disagrees with domain: {c.utterance!r}")

    def test_explicit_implies_play(self):
        for c in self.cases:
            if c.explicit:
                self.assertEqual(c.domain, "ocp_play", c.utterance)

    def test_has_all_required_categories(self):
        cats = {c.category for c in self.cases}
        for required in ("media", "keywordless", "gate_negative", "control",
                         "content_policy", "playback_divergent"):
            self.assertIn(required, cats)

    def test_multilingual(self):
        langs = {c.lang for c in self.cases}
        # en-us mandatory plus at least two more languages
        self.assertIn("en-us", langs)
        self.assertGreaterEqual(len(langs), 3)

    def test_out_of_distribution(self):
        """No eval utterance may be a verbatim line from the .intent templates."""
        locale_dir = os.path.join(REPO_ROOT, "ovos_media_classifier", "locale")
        template_lines = set()
        for root, _dirs, files in os.walk(locale_dir):
            if os.path.basename(root) != "dataset":
                continue
            for fn in files:
                if not fn.endswith(".intent"):
                    continue
                with open(os.path.join(root, fn), encoding="utf-8") as fh:
                    for line in fh:
                        s = line.strip().lower()
                        if s:
                            template_lines.add(s)
        for c in self.cases:
            self.assertNotIn(
                c.utterance.strip().lower(), template_lines,
                f"eval case reuses a template line (not OOD): {c.utterance!r}",
            )


class _StubClassifier:
    """Minimal classifier returning fixed routing outputs for metric tests."""

    def __init__(self, domain, media_type, playback="unknown", genres=()):
        self._domain = domain
        self._media_type = media_type
        self._playback = playback
        self._genres = list(genres)

    def classify_domain(self, q, lang):
        from ovos_media_classifier.intents import OCPDomain
        return OCPDomain(self._domain), 1.0

    def is_ocp_query(self, q, lang):
        return self._domain != "not_ocp", 1.0

    def classify(self, q, lang, valid_labels=None):
        return MediaType(self._media_type), 1.0

    def classify_playback_type(self, q, lang):
        return PlaybackType(self._playback)

    def classify_content_form_genres(self, q, lang):
        return list(self._genres)


def _case(**kw):
    base = dict(
        utterance="x", lang="en-us", category="media", is_ocp_query=True,
        domain="ocp_play", media_type="music", playback_type="audio",
        explicit=False, abstain_ok=True, note="",
    )
    base.update(kw)
    return RE.Case(**base)


class TestHarmMetrics(unittest.TestCase):
    def test_abstain_is_safe_not_misroute(self):
        # truth wants music; backend abstains (generic) -> safe, not a mis-route
        clf = _StubClassifier("ocp_play", "generic")
        rep = RE.evaluate(clf, [_case(media_type="music")], "stub")
        self.assertEqual(rep.media_type.confident_wrong, 0)
        self.assertEqual(rep.media_type.abstain, 1)
        self.assertEqual(rep.mis_route_rate, 0.0)

    def test_confident_wrong_is_a_misroute(self):
        # truth wants music; backend confidently says movie -> mis-route
        clf = _StubClassifier("ocp_play", "movie")
        rep = RE.evaluate(clf, [_case(media_type="music")], "stub")
        self.assertEqual(rep.media_type.confident_wrong, 1)
        self.assertEqual(rep.mis_route_rate, 1.0)

    def test_confident_correct(self):
        clf = _StubClassifier("ocp_play", "music")
        rep = RE.evaluate(clf, [_case(media_type="music")], "stub")
        self.assertEqual(rep.media_type.confident_correct, 1)
        self.assertEqual(rep.mis_route_rate, 0.0)

    def test_false_hijack(self):
        # non-media truth; backend routes into OCP -> hijack
        clf = _StubClassifier("ocp_play", "music")
        case = _case(category="gate_negative", is_ocp_query=False,
                     domain="not_ocp", media_type="generic", playback_type="unknown")
        rep = RE.evaluate(clf, [case], "stub")
        self.assertEqual(rep.gate_hijack_n, 1)
        self.assertEqual(rep.false_hijack_rate, 1.0)

    def test_false_miss(self):
        # media truth; backend routes to not_ocp -> miss
        clf = _StubClassifier("not_ocp", "generic")
        rep = RE.evaluate(clf, [_case(media_type="music")], "stub")
        self.assertEqual(rep.gate_miss_n, 1)
        self.assertEqual(rep.false_miss_rate, 1.0)

    def test_adult_leak(self):
        # adult truth, backend does not flag adult -> leak (the worst error)
        clf = _StubClassifier("ocp_play", "movie", genres=[])
        case = _case(category="content_policy", explicit=True,
                     media_type="movie", playback_type="video")
        rep = RE.evaluate(clf, [case], "stub")
        self.assertEqual(rep.adult_leak_n, 1)
        self.assertEqual(rep.adult_leak_rate, 1.0)

    def test_adult_flagged_no_leak(self):
        clf = _StubClassifier("ocp_play", "movie", genres=["adult"])
        case = _case(category="content_policy", explicit=True,
                     media_type="movie", playback_type="video")
        rep = RE.evaluate(clf, [case], "stub")
        self.assertEqual(rep.adult_leak_n, 0)

    def test_clean_overflag(self):
        # clean truth, backend wrongly flags adult -> false block
        clf = _StubClassifier("ocp_play", "music", genres=["adult"])
        rep = RE.evaluate(clf, [_case(explicit=False, media_type="music")], "stub")
        self.assertEqual(rep.adult_overflag_n, 1)

    def test_control_recall(self):
        clf = _StubClassifier("ocp_control", "generic")
        case = _case(category="control", domain="ocp_control",
                     media_type="control", playback_type="unknown")
        rep = RE.evaluate(clf, [case], "stub")
        self.assertEqual(rep.control_hit_n, 1)
        self.assertEqual(rep.control_recall, 1.0)


class TestKeywordBaselineRuns(unittest.TestCase):
    """The keyword backend is always available and must produce a full report."""

    def test_keyword_report_shape(self):
        report = RE.run(only_keyword=True)
        self.assertIn("keyword", report["backends"])
        kw = report["backends"]["keyword"]
        self.assertEqual(kw["status"], "available")
        for key in ("mis_route_rate", "media_type", "playback_type", "gate",
                    "adult_leak", "control"):
            self.assertIn(key, kw)
        # keyword abstains by default -> a low mis-route rate on the OOD set
        self.assertLess(kw["mis_route_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
