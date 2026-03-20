"""Model2Vec-based hierarchical media classifier.

Uses a ``StaticModelForHierarchicalClassification`` trained on OCP intent data.
The model has two heads:

  domain_head  → OCPDomain values  ("ocp_play" | "ocp_control" | "not_ocp")
  intent_head  → OCPPlayIntent / OCPControlIntent labels

The intent head is conditioned on the domain prediction and its logits are
masked to the intents that belong to the predicted domain at inference time.

When the model is not installed or fails to load, this class is never returned
by the factory — the caller falls back to KeywordMediaClassifier.

Configuration keys (under the pipeline "OCP" config block)::

    "media_classifier_model": "/path/to/saved/model"
        Path to a directory produced by StaticModelForHierarchicalClassification
        (same format as StaticModel.save_pretrained()).

    "media_classifier_domain_threshold": 0.5   (default)
        Minimum softmax probability for the predicted domain to be trusted.
        If below this the classifier returns (GENERIC, 0.0).

    "media_classifier_intent_threshold": 0.3   (default)
        Minimum softmax probability for the predicted intent label.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import MediaType
from ovos_media_classifier.intents import (
    OCPDomain,
    LABEL_TO_MEDIA_TYPE,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ovos_media_classifier.constants import DEFAULT_DOMAIN_THRESHOLD, DEFAULT_PLAY_THRESHOLD as DEFAULT_INTENT_THRESHOLD


class Model2VecMediaClassifier(AbstractMediaClassifier):
    """Media classifier backed by a hierarchical Model2Vec model.

    This classifier uses a two-stage prediction:
      1. Domain prediction (ocp_play / ocp_control / not_ocp)
      2. Intent prediction (music, movie, podcast, etc.) - only if domain is ocp_play

    Args:
        model: A loaded ``StaticModelForHierarchicalClassification`` instance.
        domain_threshold: Minimum confidence to trust the domain prediction.
        intent_threshold: Minimum confidence to trust the intent prediction.

    Example:
        >>> clf = Model2VecMediaClassifier.from_path("/path/to/model")
        >>> clf.classify("play jazz", "en-us")
        (MediaType.MUSIC, 0.85)
    """

    def __init__(
        self,
        model: Any,
        domain_threshold: float = DEFAULT_DOMAIN_THRESHOLD,
        intent_threshold: float = DEFAULT_INTENT_THRESHOLD,
    ) -> None:
        self._model: Any = model
        self._domain_threshold: float = domain_threshold
        self._intent_threshold: float = intent_threshold

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls, path: str, domain_threshold: float = 0.5, intent_threshold: float = 0.3
    ) -> "Model2VecMediaClassifier":
        """Load a trained model from *path* and return a ready classifier.

        Raises ``RuntimeError`` if the training dependencies are not installed
        or the model files cannot be loaded.
        """
        try:
            import torch  # noqa: F401 — confirms torch is available

            try:
                from ovos_media_classifier.models import (
                    StaticModelForHierarchicalClassification,
                )
            except ImportError:
                from train.train_hierarchical import (
                    StaticModelForHierarchicalClassification,
                )
            model = StaticModelForHierarchicalClassification.from_pretrained(path)
            model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load M2V classifier from {path!r}: {e}"
            ) from e
        return cls(
            model, domain_threshold=domain_threshold, intent_threshold=intent_threshold
        )

    # ------------------------------------------------------------------
    # Domain classification (fast path — domain head only)
    # ------------------------------------------------------------------

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Classify domain using only the domain head (no intent lookup)."""
        try:
            d_probs, _ = self._model.predict_proba([query])
            domain_conf = float(d_probs[0].max())
            domain_idx = int(d_probs[0].argmax())
            domain_label = self._model.domain_classes_[domain_idx]
            if domain_conf < self._domain_threshold:
                return OCPDomain.NOT_OCP, domain_conf
            try:
                domain = OCPDomain(domain_label)
            except ValueError:
                domain = OCPDomain.NOT_OCP
            return domain, domain_conf
        except Exception as e:
            LOG.error(f"M2V classify_domain failed: {e}")
            return OCPDomain.NOT_OCP, 0.0

    def is_ocp_query(self, query: str, lang: str) -> Tuple[bool, float]:
        """Fast domain-level check using only the domain head."""
        domain, conf = self.classify_domain(query, lang)
        return domain != OCPDomain.NOT_OCP, conf

    # ------------------------------------------------------------------
    # Full classification
    # ------------------------------------------------------------------

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Classify *query* into a MediaType using the hierarchical model.

        If the domain head predicts ``ocp_play`` with sufficient confidence,
        the intent label is mapped to a MediaType.  Otherwise returns
        (GENERIC, 0.0).
        """
        try:
            domains, intents = self._model.predict([query])
            d_probs, i_probs = self._model.predict_proba([query])

            domain_label = domains[0]
            intent_label = intents[0]
            domain_conf = float(d_probs[0].max())
            intent_conf = float(i_probs[0].max())
        except Exception as e:
            LOG.error(f"M2V classify failed: {e}")
            return MediaType.GENERIC, 0.0

        # Domain gate: only ocp_play carries media-type information
        if domain_label != OCPDomain.OCP_PLAY:
            return MediaType.GENERIC, 0.0
        if domain_conf < self._domain_threshold:
            return MediaType.GENERIC, 0.0

        media_type = LABEL_TO_MEDIA_TYPE.get(intent_label, MediaType.GENERIC)

        if intent_conf < self._intent_threshold:
            return MediaType.GENERIC, 0.0

        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0

        return media_type, intent_conf
