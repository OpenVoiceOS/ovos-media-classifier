"""ONNX label-guided categorical-embeddings media classifier.

Uses pre-trained ONNX models exported by ``guided-categorical-embeddings``
to classify media queries from sparse categorical feature dicts.

Two ONNX heads (mirrors Model2VecMediaClassifier architecture):
  ``domain/`` — OCPDomain (ocp_play / ocp_control / not_ocp)
  ``play/``   — OCPPlayIntent (music / movie / podcast / …)

Inference pipeline for each utterance:
  1. ``CategoricalFeatureExtractor.extract()`` → Dict[str, str]
  2. domain head → (domain_label, confidence)
  3. if domain == ocp_play: play head → (play_label, confidence)
  4. map play_label via LABEL_TO_MEDIA_TYPE → MediaType

[LabelGuidedEmbeddings — guided_categorical_embeddings/inference/embeddings.py:33]
[CategoricalFeatureExtractor — ovos_media_classifier/features.py]
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.features import CategoricalFeatureExtractor
from ovos_media_classifier.intents import (
    LABEL_TO_MEDIA_TYPE,
    MediaType,
    OCPDomain,
)

if TYPE_CHECKING:
    from guided_categorical_embeddings.inference.embeddings import LabelGuidedEmbeddings
    from ovos_media_classifier.entities import EntitiesContainer

DEFAULT_DOMAIN_THRESHOLD = 0.5
DEFAULT_PLAY_THRESHOLD = 0.3


class GuidedEmbeddingsMediaClassifier(AbstractMediaClassifier):
    """Media classifier using ONNX label-guided categorical embeddings.

    Args:
        domain_model: ``LabelGuidedEmbeddings`` loaded from ``domain/`` subdir.
        play_model: ``LabelGuidedEmbeddings`` loaded from ``play/`` subdir.
        feature_extractor: ``CategoricalFeatureExtractor`` instance.
        domain_threshold: Min softmax confidence for domain prediction.
        play_threshold: Min softmax confidence for play-intent prediction.
    """

    def __init__(
        self,
        domain_model: "LabelGuidedEmbeddings",
        play_model: "LabelGuidedEmbeddings",
        feature_extractor: CategoricalFeatureExtractor,
        domain_threshold: float = DEFAULT_DOMAIN_THRESHOLD,
        play_threshold: float = DEFAULT_PLAY_THRESHOLD,
    ) -> None:
        self._domain_model = domain_model
        self._play_model = play_model
        self._extractor = feature_extractor
        self._domain_thresh = domain_threshold
        self._play_thresh = play_threshold

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        model_dir: str,
        entities_container: Optional["EntitiesContainer"] = None,
        locale_dir: Optional[str] = None,
        domain_threshold: float = DEFAULT_DOMAIN_THRESHOLD,
        play_threshold: float = DEFAULT_PLAY_THRESHOLD,
    ) -> "GuidedEmbeddingsMediaClassifier":
        """Load from a directory containing ``domain/`` and ``play/`` subdirs.

        Args:
            model_dir: Root directory with ONNX export subdirectories.
            entities_container: Optional ``EntitiesContainer`` for NER features.
            locale_dir: Override bundled locale for keyword features.
            domain_threshold: Min confidence for domain head.
            play_threshold: Min confidence for play head.
        """
        try:
            from guided_categorical_embeddings.inference.embeddings import (
                LabelGuidedEmbeddings,
            )
        except ImportError as exc:
            raise ImportError(
                "guided-categorical-embeddings is required. "
                "Install with: pip install ovos-media-classifier[guided]"
            ) from exc

        domain_model = LabelGuidedEmbeddings(os.path.join(model_dir, "domain"))
        play_model = LabelGuidedEmbeddings(os.path.join(model_dir, "play"))

        if entities_container is not None:
            extractor = CategoricalFeatureExtractor.from_container(
                entities_container
            )
        else:
            extractor = CategoricalFeatureExtractor.from_locale_dir(locale_dir)

        return cls(
            domain_model=domain_model,
            play_model=play_model,
            feature_extractor=extractor,
            domain_threshold=domain_threshold,
            play_threshold=play_threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_with_conf(
        self,
        model: "LabelGuidedEmbeddings",
        feat_dict: Dict[str, str],
    ) -> Tuple[str, float]:
        """Run ONNX classifier head on *feat_dict* and return (label, conf).

        Extracts softmax probabilities directly from ``_clf_session`` to get
        per-class confidence (``model.predict()`` returns labels only).

        Args:
            model: Loaded ``LabelGuidedEmbeddings`` instance.
            feat_dict: Sparse feature dict ``{feature_name: "1"}``.

        Returns:
            Tuple of (label_string, softmax_confidence).
        """
        X_arr: np.ndarray = model._to_array([feat_dict])
        result = model._clf_session.run(None, {"features": X_arr})
        probs: np.ndarray = result[0]  # shape (1, n_classes)
        idx = int(np.argmax(probs[0]))
        conf = float(probs[0][idx])
        label = model.idx_to_label[idx]
        return label, conf

    # ------------------------------------------------------------------
    # AbstractMediaClassifier implementation
    # ------------------------------------------------------------------

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Classify the OCP domain (ocp_play / ocp_control / not_ocp).

        Args:
            query: User utterance.
            lang: BCP-47 language tag.
        """
        feat = self._extractor.extract(query, lang)
        try:
            label, conf = self._predict_with_conf(self._domain_model, feat)
        except Exception as exc:
            LOG.error(f"GuidedEmbeddingsMediaClassifier domain prediction failed: {exc}")
            return OCPDomain.NOT_OCP, 0.0

        if conf < self._domain_thresh:
            return OCPDomain.NOT_OCP, conf

        try:
            return OCPDomain(label), conf
        except ValueError:
            return OCPDomain.NOT_OCP, conf

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Return (MediaType, confidence) for an ocp_play utterance.

        Falls back to (GENERIC, 0.0) when:
        - domain is not ocp_play, or
        - play confidence is below threshold, or
        - predicted label is not in valid_labels.

        Args:
            query: User utterance.
            lang: BCP-47 language tag.
            valid_labels: When set, only these MediaTypes are valid results.
        """
        feat = self._extractor.extract(query, lang)

        # Domain gate
        try:
            domain_label, domain_conf = self._predict_with_conf(
                self._domain_model, feat
            )
        except Exception as exc:
            LOG.error(f"GuidedEmbeddingsMediaClassifier domain prediction failed: {exc}")
            return MediaType.GENERIC, 0.0

        if domain_conf < self._domain_thresh:
            return MediaType.GENERIC, 0.0

        try:
            domain = OCPDomain(domain_label)
        except ValueError:
            domain = OCPDomain.NOT_OCP

        if domain != OCPDomain.OCP_PLAY:
            return MediaType.GENERIC, 0.0

        # Play head
        try:
            play_label, play_conf = self._predict_with_conf(self._play_model, feat)
        except Exception as exc:
            LOG.error(f"GuidedEmbeddingsMediaClassifier play prediction failed: {exc}")
            return MediaType.GENERIC, 0.0

        if play_conf < self._play_thresh:
            return MediaType.GENERIC, 0.0

        media_type = LABEL_TO_MEDIA_TYPE.get(play_label, MediaType.GENERIC)

        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0

        return media_type, play_conf
