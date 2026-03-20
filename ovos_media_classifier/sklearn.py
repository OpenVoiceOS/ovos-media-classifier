"""scikit-learn based media classifier.

Wraps a scikit-learn Pipeline (TF-IDF + classifier) trained on OCP utterance
data.  The pipeline outputs OCPPlayIntent label strings.

Two models are supported:

  play_pipeline  — multi-class classifier: predicts OCPPlayIntent from an
                   utterance in the ocp_play domain.
  domain_pipeline — binary/ternary classifier: predicts OCPDomain
                   (ocp_play / ocp_control / not_ocp).  Optional; when
                   absent the base classify_domain() fallback is used.

Dependency: ``scikit-learn`` (optional)

    pip install ovos-media-classifier[sklearn]

Usage::

    # Load from a persisted joblib file:
    clf = SklearnMediaClassifier.from_path("/path/to/model.joblib")

    # Train from scratch:
    clf = SklearnMediaClassifier.from_training_data(
        X=["play jazz", "put on some blues", ...],
        y=["music", "music", ...],
    )
    clf.save("/path/to/model.joblib")

Persisted format (joblib dict)::

    {
        "play_pipeline": sklearn.pipeline.Pipeline,
        "domain_pipeline": sklearn.pipeline.Pipeline | None,
        "play_labels": list[str],   # class label strings for play head
        "domain_labels": list[str], # class label strings for domain head
    }
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.constants import DEFAULT_DOMAIN_THRESHOLD, DEFAULT_PLAY_THRESHOLD
from ovos_media_classifier.intents import MediaType
from ovos_media_classifier.intents import (
    OCPDomain,
    LABEL_TO_MEDIA_TYPE,
)


class SklearnMediaClassifier(AbstractMediaClassifier):
    """Media classifier backed by scikit-learn TF-IDF + classifier pipelines.

    This classifier uses two optional sklearn pipelines:
      - play_pipeline: Multi-class classifier for OCPPlayIntent labels
      - domain_pipeline: Optional binary/ternary classifier for OCPDomain

    Args:
        play_pipeline: sklearn Pipeline that predicts OCPPlayIntent labels.
        domain_pipeline: Optional sklearn Pipeline that predicts OCPDomain
                         labels. When provided, is_ocp_query / classify_domain
                         use this instead of deriving from the play head.
        play_threshold: Minimum probability (from predict_proba) for the play
                        classifier to be trusted. 0 for models without proba.
        domain_threshold: Same for the domain classifier.

    Example:
        >>> clf = SklearnMediaClassifier.from_path("/path/to/model.joblib")
        >>> clf.classify("play jazz", "en-us")
        (MediaType.MUSIC, 0.85)
    """

    def __init__(
        self,
        play_pipeline: Any,
        domain_pipeline: Optional[Any] = None,
        play_threshold: float = DEFAULT_PLAY_THRESHOLD,
        domain_threshold: float = DEFAULT_DOMAIN_THRESHOLD,
    ) -> None:
        self._play: Any = play_pipeline
        self._domain: Optional[Any] = domain_pipeline
        self._play_thresh: float = play_threshold
        self._domain_thresh: float = domain_threshold

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: str, **kwargs) -> "SklearnMediaClassifier":
        """Load a persisted classifier from a joblib file.

        Args:
            path: Path to a ``.joblib`` file previously saved with
                  :meth:`save`.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "joblib is required to load a SklearnMediaClassifier. "
                "Install it with: pip install scikit-learn"
            )
        data = joblib.load(path)
        return cls(
            play_pipeline=data["play_pipeline"],
            domain_pipeline=data.get("domain_pipeline"),
            **kwargs,
        )

    @classmethod
    def from_training_data(
        cls,
        X: List[str],
        y: List[str],
        X_domain: Optional[List[str]] = None,
        y_domain: Optional[List[str]] = None,
        classifier=None,
        domain_classifier=None,
        **kwargs,
    ) -> "SklearnMediaClassifier":
        """Train a TF-IDF + classifier pipeline from raw text samples.

        Args:
            X:  List of utterance strings for play-intent training.
            y:  List of OCPPlayIntent label strings (e.g. "music", "movie").
            X_domain: Optional utterances for domain classifier training.
            y_domain: Optional OCPDomain label strings for domain classifier.
            classifier:        Override the play-head sklearn estimator.
                               Defaults to LogisticRegression.
            domain_classifier: Override the domain-head sklearn estimator.
        """
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError(
                "scikit-learn is required to train a SklearnMediaClassifier. "
                "Install it with: pip install scikit-learn"
            )

        def _build(estimator, X_train, y_train):
            pipe = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            sublinear_tf=True,
                            min_df=1,
                            max_features=50_000,
                        ),
                    ),
                    ("clf", estimator),
                ]
            )
            pipe.fit(X_train, y_train)
            return pipe

        play_pipe = _build(
            classifier or LogisticRegression(max_iter=1000, C=5.0),
            X,
            y,
        )

        domain_pipe = None
        if X_domain and y_domain:
            domain_pipe = _build(
                domain_classifier or LogisticRegression(max_iter=1000, C=5.0),
                X_domain,
                y_domain,
            )

        return cls(play_pipe, domain_pipeline=domain_pipe, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the classifier to a joblib file at *path*."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required. pip install scikit-learn")
        joblib.dump(
            {
                "play_pipeline": self._play,
                "domain_pipeline": self._domain,
            },
            path,
        )
        LOG.info(f"SklearnMediaClassifier saved to {path!r}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_with_conf(pipeline, text: str) -> Tuple[str, float]:
        """Run *pipeline* on *text*, returning (label, confidence).

        Handles both probabilistic classifiers (predict_proba) and
        decision-function classifiers (predict only).
        """
        label = str(pipeline.predict([text])[0])
        try:
            proba = float(pipeline.predict_proba([text])[0].max())
        except AttributeError:
            # SVM / LinearSVC: no predict_proba; use a fixed fallback
            try:
                score = float(pipeline.decision_function([text])[0].max())
                # Squash decision-function score to (0, 1) heuristically
                import math

                proba = 1.0 / (1.0 + math.exp(-score))
            except Exception:
                proba = 0.5
        return label, proba

    # ------------------------------------------------------------------
    # AbstractMediaClassifier implementation
    # ------------------------------------------------------------------

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        if self._domain is None:
            return super().classify_domain(query, lang)
        try:
            label, conf = self._predict_with_conf(self._domain, query)
            if conf < self._domain_thresh:
                return OCPDomain.NOT_OCP, conf
            try:
                return OCPDomain(label), conf
            except ValueError:
                return OCPDomain.NOT_OCP, conf
        except Exception as e:
            LOG.error(f"SklearnMediaClassifier domain prediction failed: {e}")
            return OCPDomain.NOT_OCP, 0.0

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        try:
            label, conf = self._predict_with_conf(self._play, query)
        except Exception as e:
            LOG.error(f"SklearnMediaClassifier play prediction failed: {e}")
            return MediaType.GENERIC, 0.0

        if conf < self._play_thresh:
            return MediaType.GENERIC, 0.0

        media_type = LABEL_TO_MEDIA_TYPE.get(label, MediaType.GENERIC)

        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0

        return media_type, conf
