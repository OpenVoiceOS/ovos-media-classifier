"""Unit tests for GuidedEmbeddingsMediaClassifier and CategoricalFeatureExtractor.

All tests are fully mocked — no ONNX files or guided-categorical-embeddings
package required.

[CategoricalFeatureExtractor — ovos_media_classifier/features.py]
[GuidedEmbeddingsMediaClassifier — ovos_media_classifier/guided.py]
"""
from __future__ import annotations

import sys
import types
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out guided_categorical_embeddings so the module can be imported
# without the real package installed.
# ---------------------------------------------------------------------------
_gce = types.ModuleType("guided_categorical_embeddings")
_gce_inf = types.ModuleType("guided_categorical_embeddings.inference")
_gce_emb = types.ModuleType("guided_categorical_embeddings.inference.embeddings")


class _FakeLGE:
    """Minimal stub for LabelGuidedEmbeddings."""

    def __init__(self, model_dir: str, providers=None) -> None:
        self.model_dir = model_dir
        self.idx_to_label = {0: "ocp_play", 1: "not_ocp", 2: "ocp_control"}
        self._clf_session = MagicMock()
        self.vectorizer = MagicMock()

    def _to_array(self, X):
        return np.zeros((len(X), 10), dtype=np.float32)


_gce_emb.LabelGuidedEmbeddings = _FakeLGE
_gce_inf.embeddings = _gce_emb
_gce.inference = _gce_inf
sys.modules.setdefault("guided_categorical_embeddings", _gce)
sys.modules.setdefault("guided_categorical_embeddings.inference", _gce_inf)
sys.modules.setdefault(
    "guided_categorical_embeddings.inference.embeddings", _gce_emb
)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from ovos_media_classifier.features import (  # noqa: E402
    CategoricalFeatureExtractor,
    _ENTITY_LABEL_VALUES,
    _KEYWORD_VOCABS,
)
from ovos_media_classifier.guided import GuidedEmbeddingsMediaClassifier  # noqa: E402
from ovos_media_classifier.intents import MediaType, OCPDomain  # noqa: E402


# ===========================================================================
# TestCategoricalFeatureExtractor
# ===========================================================================

class TestCategoricalFeatureExtractor:
    """Tests for CategoricalFeatureExtractor.extract()."""

    def _make_extractor(self, matched_vocabs=(), ner_hits=()):
        """Build an extractor with controlled mock responses."""
        matcher = MagicMock()
        matcher.match.side_effect = (
            lambda utterance, vocab_name, lang: vocab_name in matched_vocabs
        )
        container = MagicMock()
        container.ner.tag.return_value = [
            {"label": lbl} for lbl in ner_hits
        ]
        return CategoricalFeatureExtractor(
            voc_matcher=matcher, entities_container=container, lang="en"
        )

    def test_keyword_feature_fires(self):
        """Keyword vocab match → correct feature key present."""
        extractor = self._make_extractor(matched_vocabs={"MusicKeyword"})
        feat = extractor.extract("play some music", "en")
        assert feat.get("kw_music") == "1"

    def test_absent_keyword_not_in_dict(self):
        """Vocab not matched → key absent from sparse dict."""
        extractor = self._make_extractor(matched_vocabs=set())
        feat = extractor.extract("tell me a joke", "en")
        assert "kw_music" not in feat

    def test_ner_feature_fires(self):
        """NER hit with valid entity label → feature key present."""
        valid_ner_label = _ENTITY_LABEL_VALUES[0]
        extractor = self._make_extractor(ner_hits=[valid_ner_label])
        feat = extractor.extract("play radiohead", "en")
        assert feat.get(valid_ner_label) == "1"

    def test_ner_invalid_label_ignored(self):
        """NER hit with unknown label → not added to feat dict."""
        extractor = self._make_extractor(ner_hits=["unknown_label_xyz"])
        feat = extractor.extract("something", "en")
        assert "unknown_label_xyz" not in feat

    def test_empty_utterance_returns_empty_dict(self):
        """Empty utterance with no matches → empty dict."""
        extractor = self._make_extractor()
        feat = extractor.extract("", "en")
        assert feat == {}

    def test_no_container_skips_ner(self):
        """Extractor without container still runs keywords."""
        matcher = MagicMock()
        matcher.match.side_effect = (
            lambda u, v, l: v == "MovieKeyword"
        )
        extractor = CategoricalFeatureExtractor(
            voc_matcher=matcher, entities_container=None, lang="en"
        )
        feat = extractor.extract("watch a film", "en")
        assert feat.get("kw_movie") == "1"
        # No NER keys
        for lbl in _ENTITY_LABEL_VALUES:
            assert lbl not in feat

    def test_ner_exception_is_swallowed(self):
        """NER raising an exception → feature dict still returned."""
        matcher = MagicMock()
        matcher.match.return_value = False
        container = MagicMock()
        container.ner.tag.side_effect = RuntimeError("NER boom")
        extractor = CategoricalFeatureExtractor(
            voc_matcher=matcher, entities_container=container, lang="en"
        )
        feat = extractor.extract("test", "en")
        assert isinstance(feat, dict)

    def test_from_locale_dir_creates_keyword_only(self):
        """from_locale_dir() → no entities_container."""
        extractor = CategoricalFeatureExtractor.from_locale_dir(lang="en")
        assert extractor._entities is None
        assert extractor._matcher is not None


# ===========================================================================
# TestGuidedEmbeddingsMediaClassifier
# ===========================================================================

def _make_model(label: str, confidence: float) -> _FakeLGE:
    """Create a stub LabelGuidedEmbeddings that always predicts (label, conf)."""
    model = _FakeLGE("/fake/dir")
    probs = np.zeros((1, 3), dtype=np.float32)
    idx = list(model.idx_to_label.values()).index(label)
    probs[0, idx] = confidence
    model._clf_session.run.return_value = [probs]
    return model


class TestGuidedEmbeddingsMediaClassifier:
    """Tests for GuidedEmbeddingsMediaClassifier."""

    def _make_clf(
        self,
        domain_label: str = "ocp_play",
        domain_conf: float = 0.9,
        play_label: str = "music",
        play_conf: float = 0.8,
        domain_threshold: float = 0.5,
        play_threshold: float = 0.3,
    ) -> GuidedEmbeddingsMediaClassifier:
        domain_model = _make_model(domain_label, domain_conf)
        # Play model needs its own idx_to_label for play labels
        play_model = _make_model("ocp_play", 0.5)  # dummy, overridden below
        play_model.idx_to_label = {0: play_label, 1: "movie", 2: "podcast"}
        probs = np.zeros((1, 3), dtype=np.float32)
        probs[0, 0] = play_conf
        play_model._clf_session.run.return_value = [probs]

        extractor = MagicMock(spec=CategoricalFeatureExtractor)
        extractor.extract.return_value = {"kw_music": "1"}

        return GuidedEmbeddingsMediaClassifier(
            domain_model=domain_model,
            play_model=play_model,
            feature_extractor=extractor,
            domain_threshold=domain_threshold,
            play_threshold=play_threshold,
        )

    def test_classify_music(self):
        """High-confidence ocp_play + music → MediaType.MUSIC."""
        clf = self._make_clf(domain_label="ocp_play", play_label="music",
                             play_conf=0.9)
        media, conf = clf.classify("play some jazz", "en")
        assert media == MediaType.MUSIC
        assert conf > 0.5

    def test_classify_not_ocp_returns_generic(self):
        """Domain = not_ocp → GENERIC regardless of play head."""
        clf = self._make_clf(domain_label="not_ocp", domain_conf=0.95)
        media, conf = clf.classify("what is the weather", "en")
        assert media == MediaType.GENERIC

    def test_domain_below_threshold_returns_generic(self):
        """Domain confidence below threshold → GENERIC."""
        clf = self._make_clf(domain_conf=0.1, domain_threshold=0.5)
        media, conf = clf.classify("maybe play something", "en")
        assert media == MediaType.GENERIC

    def test_play_below_threshold_returns_generic(self):
        """Play confidence below threshold → GENERIC."""
        clf = self._make_clf(play_conf=0.05, play_threshold=0.3)
        media, conf = clf.classify("play something", "en")
        assert media == MediaType.GENERIC

    def test_valid_labels_filter(self):
        """valid_labels excludes predicted type → GENERIC."""
        clf = self._make_clf(play_label="music", play_conf=0.9)
        media, conf = clf.classify("play jazz", "en",
                                   valid_labels=[MediaType.MOVIE])
        assert media == MediaType.GENERIC

    def test_classify_domain_ocp_play(self):
        """classify_domain returns OCP_PLAY when domain model fires."""
        clf = self._make_clf(domain_label="ocp_play", domain_conf=0.9)
        domain, conf = clf.classify_domain("play some jazz", "en")
        assert domain == OCPDomain.OCP_PLAY
        assert conf > 0.5

    def test_classify_domain_not_ocp(self):
        """classify_domain returns NOT_OCP when domain conf below threshold."""
        clf = self._make_clf(domain_label="not_ocp", domain_conf=0.95)
        domain, _ = clf.classify_domain("what time is it", "en")
        assert domain == OCPDomain.NOT_OCP

    def test_from_path_uses_correct_subdirs(self, tmp_path):
        """from_path() passes domain/ and play/ subdirs to LabelGuidedEmbeddings."""
        loaded_dirs = []

        class _TrackingLGE(_FakeLGE):
            def __init__(self, model_dir, providers=None):
                super().__init__(model_dir, providers)
                loaded_dirs.append(model_dir)

        import guided_categorical_embeddings.inference.embeddings as _emb_mod
        original = _emb_mod.LabelGuidedEmbeddings
        _emb_mod.LabelGuidedEmbeddings = _TrackingLGE
        try:
            clf = GuidedEmbeddingsMediaClassifier.from_path(str(tmp_path))
        finally:
            _emb_mod.LabelGuidedEmbeddings = original

        assert any(str(tmp_path / "domain") in d for d in loaded_dirs)
        assert any(str(tmp_path / "play") in d for d in loaded_dirs)


# ===========================================================================
# TestLoadMediaClassifierGuidedBackend
# ===========================================================================

class TestLoadMediaClassifierGuidedBackend:
    """Integration tests for the load_media_classifier factory."""

    def test_factory_selects_guided_when_config_key_set(self, tmp_path):
        """load_media_classifier picks GuidedEmbeddingsMediaClassifier."""
        (tmp_path / "domain").mkdir()
        (tmp_path / "play").mkdir()

        fake_clf = MagicMock(spec=GuidedEmbeddingsMediaClassifier)

        with patch(
            "ovos_media_classifier.guided.GuidedEmbeddingsMediaClassifier"
            ".from_path",
            return_value=fake_clf,
        ):
            from ovos_media_classifier import load_media_classifier
            clf = load_media_classifier(
                config={"media_classifier_guided_model": str(tmp_path)}
            )

        assert clf is fake_clf

    def test_factory_falls_back_on_guided_load_error(self):
        """load_media_classifier falls back gracefully when guided load fails."""
        with patch(
            "ovos_media_classifier.guided.GuidedEmbeddingsMediaClassifier"
            ".from_path",
            side_effect=RuntimeError("load failure"),
        ):
            from ovos_media_classifier import load_media_classifier
            clf = load_media_classifier(
                config={"media_classifier_guided_model": "/nonexistent"}
            )

        # Should have fallen back to keyword classifier
        from ovos_media_classifier.keyword import KeywordMediaClassifier
        assert isinstance(clf, KeywordMediaClassifier)
