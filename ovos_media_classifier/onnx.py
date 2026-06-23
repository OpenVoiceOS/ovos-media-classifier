"""Optional ONNX trained-classifier backend for ovos-media-classifier.

This is the **experimental, opt-in trained backend**.  The default install ships
only the lean ``.voc`` keyword classifier; this module is reached only when the
``onnx`` extra is installed *and* a model bundle is configured.  It depends on
**raw ``onnxruntime`` + ``numpy`` only** — no ``guided-categorical-embeddings``,
no heavy ML framework — and both are imported lazily (inside ``from_path`` /
``__init__``), so importing this module never pulls them in by itself.

Self-describing model-bundle format
-----------------------------------
:meth:`OnnxMediaClassifier.from_path` loads a **bundle directory** that fully
describes itself, so a future trained model can target a stable contract::

    <bundle>/
      ├── domain.onnx   # domain head: float input → logits over domain_labels
      ├── play.onnx     # play head:   float input → logits over play_labels
      └── meta.json

``meta.json`` carries everything the runtime needs (no hard-coded assumptions)::

    {
      "feature_names": ["kw_music", "kw_movie", ...],   # ORDERED feature columns
      "domain_labels": {"0": "ocp_play", "1": "ocp_control", "2": "not_ocp"},
      "play_labels":   {"0": "music", "1": "movie", ...},  # idx -> OCPPlayIntent label
      "input_name":    "features",   # optional; defaults to each session's 1st input
      "domain_threshold": 0.5,       # optional; falls back to constants.py defaults
      "play_threshold":   0.3
    }

* ``feature_names`` — the ordered list of categorical feature columns the model
  was trained on.  At inference the sparse feature dict from
  :class:`~ovos_media_classifier.features.CategoricalFeatureExtractor` is
  vectorized into a dense ``float32`` row in exactly this order (present → 1.0,
  absent → 0.0).
* ``domain_labels`` / ``play_labels`` — maps from *output index* (string keys,
  as JSON has no int keys) to the label string.  ``domain_labels`` values are
  :class:`~ovos_media_classifier.intents.OCPDomain` values; ``play_labels``
  values are :class:`~ovos_media_classifier.intents.OCPPlayIntent` labels, mapped
  to :class:`mediavocab.MediaType` / genres via ``LABEL_TO_MEDIA_TYPE`` /
  ``LABEL_TO_GENRES``.

Inference pipeline per utterance:
  1. ``CategoricalFeatureExtractor.extract()`` → sparse ``{feature: "1"}`` dict
  2. vectorize → ``float32`` row in ``feature_names`` order
  3. domain head → softmax → argmax → ``OCPDomain`` (+ confidence)
  4. if ``ocp_play``: play head → softmax → argmax → play label
  5. play label → ``mediavocab.MediaType`` + genres; coarse axes derived.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.constants import (
    DEFAULT_DOMAIN_THRESHOLD,
    DEFAULT_PLAY_THRESHOLD,
)
from ovos_media_classifier.features import CategoricalFeatureExtractor
from ovos_media_classifier.intents import (
    LABEL_TO_GENRES,
    LABEL_TO_MEDIA_TYPE,
    MediaType,
    OCPDomain,
)

# bundle file names
_DOMAIN_ONNX = "domain.onnx"
_PLAY_ONNX = "play.onnx"
_META_JSON = "meta.json"


def _softmax(logits) -> "list":
    """Numerically-stable softmax over a 1-D numpy row → numpy array."""
    import numpy as np

    arr = np.asarray(logits, dtype="float64").reshape(-1)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


class OnnxMediaClassifier(AbstractMediaClassifier):
    """Trained media classifier backed by two ONNX heads + numpy.

    Loaded from a self-describing bundle (see module docstring) via
    :meth:`from_path`.  Use the factory rather than the constructor directly
    unless you are wiring sessions in by hand (e.g. in tests).

    Args:
        domain_session: ``onnxruntime.InferenceSession`` for the domain head.
        play_session: ``onnxruntime.InferenceSession`` for the play head.
        feature_names: ordered feature columns the model was trained on.
        domain_labels: output-index → :class:`OCPDomain` value string.
        play_labels: output-index → ``OCPPlayIntent`` label string.
        extractor: the categorical feature extractor.
        input_name: ONNX graph input name (default: each session's 1st input).
        domain_threshold: min softmax confidence to trust the domain head.
        play_threshold: min softmax confidence to trust the play head.
    """

    def __init__(
        self,
        domain_session,
        play_session,
        feature_names: List[str],
        domain_labels: Dict[int, str],
        play_labels: Dict[int, str],
        extractor: CategoricalFeatureExtractor,
        input_name: Optional[str] = None,
        domain_threshold: float = DEFAULT_DOMAIN_THRESHOLD,
        play_threshold: float = DEFAULT_PLAY_THRESHOLD,
    ) -> None:
        self._domain_session = domain_session
        self._play_session = play_session
        self._feature_names = list(feature_names)
        self._domain_labels = {int(k): v for k, v in domain_labels.items()}
        self._play_labels = {int(k): v for k, v in play_labels.items()}
        self._extractor = extractor
        self._input_name = input_name
        self._domain_thresh = domain_threshold
        self._play_thresh = play_threshold

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        model_dir: str,
        locale_dir: Optional[str] = None,
    ) -> "OnnxMediaClassifier":
        """Load an :class:`OnnxMediaClassifier` from a bundle directory.

        Args:
            model_dir: bundle dir containing ``domain.onnx``, ``play.onnx`` and
                ``meta.json`` (see module docstring for the format).
            locale_dir: override the locale dir used for keyword features;
                defaults to the bundled ``.voc`` files.

        Raises:
            ImportError: when ``onnxruntime`` / ``numpy`` are not installed
                (i.e. the ``onnx`` extra was not selected).
            FileNotFoundError / ValueError: when the bundle is incomplete.
        """
        try:
            import numpy  # noqa: F401  (vectorization happens in classify_*)
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise ImportError(
                "The ONNX backend requires onnxruntime + numpy. "
                "Install with: pip install ovos-media-classifier[onnx]"
            ) from exc

        domain_path = os.path.join(model_dir, _DOMAIN_ONNX)
        play_path = os.path.join(model_dir, _PLAY_ONNX)
        meta_path = os.path.join(model_dir, _META_JSON)
        for p in (domain_path, play_path, meta_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"ONNX model bundle missing file: {p}")

        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        feature_names = meta.get("feature_names")
        domain_labels = meta.get("domain_labels")
        play_labels = meta.get("play_labels")
        if not feature_names or not domain_labels or not play_labels:
            raise ValueError(
                f"{meta_path} must define feature_names, domain_labels and play_labels"
            )

        domain_session = onnxruntime.InferenceSession(domain_path)
        play_session = onnxruntime.InferenceSession(play_path)

        extractor = CategoricalFeatureExtractor.from_locale_dir(locale_dir)

        return cls(
            domain_session=domain_session,
            play_session=play_session,
            feature_names=feature_names,
            domain_labels=domain_labels,
            play_labels=play_labels,
            extractor=extractor,
            input_name=meta.get("input_name"),
            domain_threshold=float(meta.get("domain_threshold", DEFAULT_DOMAIN_THRESHOLD)),
            play_threshold=float(meta.get("play_threshold", DEFAULT_PLAY_THRESHOLD)),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vectorize(self, feat: Dict[str, str]):
        """Dense ``float32`` row of shape ``(1, n_features)`` in feature_names order."""
        import numpy as np

        row = np.zeros((1, len(self._feature_names)), dtype="float32")
        for i, name in enumerate(self._feature_names):
            if name in feat:
                row[0, i] = 1.0
        return row

    def _run(self, session, row) -> "list":
        """Run a head and return its softmax probabilities (1-D numpy array)."""
        name = self._input_name or session.get_inputs()[0].name
        outputs = session.run(None, {name: row})
        logits = outputs[0]
        # outputs[0] is shape (1, n_classes) (or (n_classes,)); take row 0
        import numpy as np

        logits = np.asarray(logits)
        if logits.ndim == 2:
            logits = logits[0]
        return _softmax(logits)

    def _predict_domain(self, feat: Dict[str, str]) -> Tuple[OCPDomain, float]:
        try:
            probs = self._run(self._domain_session, self._vectorize(feat))
        except Exception as exc:
            LOG.error(f"OnnxMediaClassifier domain prediction failed: {exc}")
            return OCPDomain.NOT_OCP, 0.0
        import numpy as np

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self._domain_labels.get(idx)
        if conf < self._domain_thresh or label is None:
            return OCPDomain.NOT_OCP, conf
        try:
            return OCPDomain(label), conf
        except ValueError:
            return OCPDomain.NOT_OCP, conf

    def _predict_play(self, feat: Dict[str, str]) -> Tuple[str, float]:
        """Return (play_label_string, confidence); ("generic", 0.0) on failure."""
        try:
            probs = self._run(self._play_session, self._vectorize(feat))
        except Exception as exc:
            LOG.error(f"OnnxMediaClassifier play prediction failed: {exc}")
            return "generic", 0.0
        import numpy as np

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self._play_labels.get(idx, "generic")
        return label, conf

    def _play_label(self, query: str, lang: str) -> Tuple[str, float, OCPDomain]:
        """Resolve the winning play label (domain-gated). Shared by the axes."""
        feat = self._extractor.extract(query, lang)
        domain, dconf = self._predict_domain(feat)
        if domain != OCPDomain.OCP_PLAY:
            return "generic", dconf, domain
        label, pconf = self._predict_play(feat)
        if pconf < self._play_thresh:
            return "generic", pconf, domain
        return label, pconf, domain

    # ------------------------------------------------------------------
    # AbstractMediaClassifier implementation
    # ------------------------------------------------------------------

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Classify the OCP domain via the dedicated domain head."""
        feat = self._extractor.extract(query, lang)
        return self._predict_domain(feat)

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Return (MediaType, confidence) for an ocp_play utterance.

        Falls back to (GENERIC, 0.0) when the domain is not ocp_play, the play
        confidence is below threshold, or the predicted type is not in
        *valid_labels*.
        """
        label, conf, _ = self._play_label(query, lang)
        media_type = LABEL_TO_MEDIA_TYPE.get(label, MediaType.GENERIC)
        if media_type == MediaType.GENERIC:
            return MediaType.GENERIC, 0.0
        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0
        return media_type, conf

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Return mediavocab genre tags implied by the winning play label."""
        label, _, _ = self._play_label(query, lang)
        return list(LABEL_TO_GENRES.get(label, []))

    def classify_full(self, query: str, lang: str):
        """Full multi-axis result.

        The model predicts ``domain`` and the fine-grained play label (→
        ``MediaType`` + genres) directly; the coarse axes (``playback_type`` and
        ``structure``) are derived from the predicted ``MediaType`` via
        ``mediavocab.infer_playback_type`` / :func:`axes.infer_structure`.
        """
        from mediavocab import infer_playback_type

        from ovos_media_classifier.axes import MediaClassification, infer_structure

        label, conf, domain = self._play_label(query, lang)
        media_type = LABEL_TO_MEDIA_TYPE.get(label, MediaType.GENERIC)
        genres = list(LABEL_TO_GENRES.get(label, []))
        return MediaClassification(
            media_type=media_type,
            playback_type=infer_playback_type(media_type),
            structure=infer_structure(media_type),
            domain=domain,
            genres=genres,
            confidence=conf,
        )
