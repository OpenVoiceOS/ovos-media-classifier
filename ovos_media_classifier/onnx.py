"""Optional ONNX trained-classifier backend for ovos-media-classifier.

This is the **experimental, opt-in trained backend**.  The default install ships
only the lean ``.voc`` keyword classifier; this module is reached only when the
``onnx`` extra is installed *and* a model bundle is configured.  It depends on
**raw ``onnxruntime`` + ``numpy`` only** — no ``guided-categorical-embeddings``,
no heavy ML framework — and both are imported lazily (inside ``from_path`` /
``__init__``), so importing this module never pulls them in by itself.

Self-describing, multi-head model-bundle format
-----------------------------------------------
:meth:`OnnxMediaClassifier.from_path` loads a **bundle directory** that fully
describes itself.  The bundle is **multi-task**: one ONNX head per axis
(``training/train_sklearn.py`` produces it).  ``from_path`` loads whatever heads
are present; an axis with no head is *derived* the same way the keyword default
does (so partial bundles — and old 2-head ``domain``/``play`` bundles — load)::

    <bundle>/
      ├── domain.onnx              # ocp_play / not_ocp
      ├── media_type.onnx          # the leaf mediavocab.MediaType
      ├── playback_type.onnx       # audio / video / paged / interactive
      ├── structure.onnx           # single / episodic / continuous / collection
      ├── content_form_genres.onnx # MULTI-LABEL adult/anime/animation/asmr  ← content filter
      ├── content_form.onnx        # SINGLE  trailer/teaser/behind_scenes/excerpt/…
      ├── programme_format.onnx    # SINGLE  documentary/news/concert/stand_up/…
      ├── accessibility.onnx       # MULTI-LABEL subtitles/audio_description/sign_language
      ├── variant.onnx             # SINGLE  directors/extended/remastered/colorized/…
      ├── picture_format.onnx      # MULTI-LABEL black_and_white/silent/3d
      ├── explicitness.onnx        # clean / adult  (when trained)
      ├── play.onnx                # back-compat alias of the media_type head
      └── meta.json

``meta.json`` carries everything the runtime needs::

    {
      "feature_names": ["kw_music", "kw_movie", ...],   # ORDERED feature columns
      "input_name": "input",
      "heads": {
        "domain":      {"onnx": "domain.onnx", "kind": "single",
                        "labels": {"0": "ocp_play", "1": "not_ocp"}},
        "media_type":  {"onnx": "media_type.onnx", "kind": "single",
                        "labels": {"0": "music", "1": "movie", ...}},
        "content_form_genres": {"onnx": "content_form_genres.onnx", "kind": "multi",
                        "labels": {"0": "adult", "1": "anime", ...}, "threshold": 0.5},
        ...
      },
      # legacy keys (a pre-multihead loader still works):
      "domain_labels": {...}, "play_labels": {...},
      "domain_threshold": 0.5, "play_threshold": 0.3
    }

* ``feature_names`` — the ordered feature columns the model was trained on.  At
  inference the sparse categorical dict from
  :class:`~ovos_media_classifier.features.CategoricalFeatureExtractor` fills the
  categorical columns; when the bundle also declares richer **text feature
  blocks** — ``meta["text_hash"]`` (hashed char n-grams,
  :mod:`~ovos_media_classifier.features_text`) and/or ``meta["wordvec"]`` (pooled
  domain word vectors, :mod:`~ovos_media_classifier.features_wordvec`) — the
  ``txt_*`` / ``wv_*`` columns are filled from the **raw utterance** in numpy
  (no torch / gensim).  A neural bundle trained by ``training/train_torch.py``
  carries these; a categorical-only sklearn bundle omits them (back-compat).
* ``heads`` — one entry per trained axis.  ``kind == "single"`` heads argmax over
  their ``labels``; ``kind == "multi"`` heads keep every label whose probability
  is ≥ ``threshold`` (sigmoid-style multi-label).  ``media_type`` labels are raw
  media labels resolved to :class:`mediavocab.MediaType` + genres via
  ``LABEL_TO_MEDIA_TYPE`` / ``LABEL_TO_GENRES``.

Per-axis methods (``classify`` / ``classify_domain`` / ``classify_genres`` /
``classify_content_form_genres`` / ``classify_content_form`` /
``classify_programme_format`` / ``classify_accessibility`` / ``classify_variant`` /
``classify_playback_type`` / ``classify_structure``) each use their
head when the bundle carries it, else fall back to the inherited derive/empty
default — so a backend
can **soft-gate**: trust an axis head even when the leaf is uncertain (the whole
point of the multi-task design, and what makes content-filter blocking robust —
``content_form_genres`` can flag ``adult`` independently of the leaf).
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
from ovos_media_classifier.features_text import TextHashSpec, hash_vector
from ovos_media_classifier.features_wordvec import WordVecPooler, WordVecSpec
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
    """Trained, multi-head media classifier backed by ONNX + numpy.

    Loaded from a self-describing multi-task bundle (see module docstring) via
    :meth:`from_path`.  The ``domain`` and ``play`` (media-type) heads are
    always present; any further per-axis head (``media_type`` /
    ``playback_type`` / ``structure`` / ``content_form_genres`` /
    ``content_form`` / ``programme_format`` / ``accessibility`` / ``variant`` /
    ``explicitness``) is loaded into ``extra_heads`` when the
    bundle carries it, and the axis is *derived* otherwise.  Use the factory
    rather than the constructor directly unless you are wiring sessions in by
    hand (e.g. in tests).

    Args:
        domain_session: ``onnxruntime.InferenceSession`` for the domain head.
        play_session: ``onnxruntime.InferenceSession`` for the play head.
        feature_names: ordered feature columns the model was trained on.
        domain_labels: output-index → :class:`OCPDomain` value string.
        play_labels: output-index → raw media label string.
        extractor: the categorical feature extractor.
        input_name: ONNX graph input name (default: each session's 1st input).
        domain_threshold: min softmax confidence to trust the domain head.
        play_threshold: min softmax confidence to trust the play head.
        extra_heads: optional ``axis -> head-spec`` map for the extended
            multi-task heads (each ``{"session", "kind", "labels", "threshold"}``).
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
        extra_heads: Optional[Dict[str, dict]] = None,
        text_spec: Optional[TextHashSpec] = None,
        wordvec_spec: Optional[WordVecSpec] = None,
        wordvec_pooler: Optional[WordVecPooler] = None,
        input_kind: str = "features",
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
        # ---- richer text feature blocks (optional; back-compat with bundles
        # that only carry categorical columns — both specs are then None) ----
        # ``feature_names`` lists EVERY column the model expects, in order,
        # including the ``txt_*`` (char-hash) and ``wv_*`` (word-vector) blocks.
        # The categorical extractor only produces the non-text columns, so the
        # text blocks are filled separately from the raw utterance.
        self._text_spec = text_spec
        self._wordvec_spec = wordvec_spec
        self._wordvec_pooler = wordvec_pooler
        # ``"features"`` (default) → heads consume the dense numeric feature row;
        # ``"text"`` → the bundle is a self-contained skl2onnx TfidfVectorizer→clf
        # pipeline whose graph takes the RAW utterance as a (1,1) string tensor,
        # so no python featurization runs at all (the vectorizer is baked in).
        self._input_kind = input_kind
        # offsets of the contiguous text blocks inside the feature row, derived
        # once from the column-name prefixes (the trainer always appends the
        # txt_* block then the wv_* block after the categorical columns).
        self._txt_idx = [i for i, n in enumerate(self._feature_names)
                         if text_spec and n.startswith(f"{text_spec.prefix}_")]
        self._wv_idx = [i for i, n in enumerate(self._feature_names)
                        if wordvec_spec and n.startswith(f"{wordvec_spec.prefix}_")]
        # the categorical column subset (everything that is not a text block)
        self._cat_names = [n for i, n in enumerate(self._feature_names)
                           if i not in set(self._txt_idx) | set(self._wv_idx)]
        # axis -> {"session", "kind", "labels": {idx: label}, "threshold"}
        # for the extended multi-task heads (media_type/playback_type/structure/
        # content_form_genres/content_form/programme_format/accessibility/variant/
        # explicitness), each emitting mediavocab's own vocabulary.
        self._heads: Dict[str, dict] = extra_heads or {}

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

        input_kind = meta.get("input_kind", "features")
        # a text-input (baked-vectorizer) bundle needs no feature_names — its
        # graph tokenizes the raw string itself, so only the labels are required.
        feature_names = meta.get("feature_names") or []
        domain_labels = meta.get("domain_labels")
        play_labels = meta.get("play_labels")
        if not domain_labels or not play_labels or (
                input_kind != "text" and not feature_names):
            raise ValueError(
                f"{meta_path} must define feature_names (unless input_kind=='text'), "
                "domain_labels and play_labels"
            )

        domain_session = onnxruntime.InferenceSession(domain_path)
        play_session = onnxruntime.InferenceSession(play_path)

        extractor = CategoricalFeatureExtractor.from_locale_dir(locale_dir)

        # ---- richer text feature blocks (optional, self-describing) ----
        # ``meta["text_hash"]`` / ``meta["wordvec"]`` declare the char-hash and
        # pooled-word-vector featurizers a neural bundle was trained with; the
        # runtime rebuilds them in numpy only (no torch / gensim). Absent keys →
        # categorical-only bundle (the sklearn ladder), unchanged.
        text_spec = TextHashSpec.from_meta(meta.get("text_hash"))
        wordvec_spec = WordVecSpec.from_meta(meta.get("wordvec"))
        wordvec_pooler = (WordVecPooler.from_bundle(model_dir, wordvec_spec)
                          if wordvec_spec is not None else None)
        if wordvec_spec is not None and wordvec_pooler is None:
            LOG.warning(f"bundle declares a wordvec spec but the matrix/vocab "
                        f"files are missing in {model_dir}; wv_* columns stay 0")

        # ---- load the extended multi-task heads (when present) ----
        # ``meta["heads"]`` is the multi-head manifest; each entry names its own
        # .onnx file, kind (single/multi), labels, and (for multi) threshold. A
        # head whose file is missing is skipped (the axis falls back to derive).
        extra_heads: Dict[str, dict] = {}
        for axis, spec in (meta.get("heads") or {}).items():
            if axis == "domain":  # already loaded as the dedicated domain head
                continue
            onnx_file = spec.get("onnx", f"{axis}.onnx")
            path = os.path.join(model_dir, onnx_file)
            if not os.path.isfile(path):
                LOG.warning(f"bundle head {axis!r} missing file {onnx_file}; "
                            "axis will be derived")
                continue
            extra_heads[axis] = {
                "session": onnxruntime.InferenceSession(path),
                "kind": spec.get("kind", "single"),
                "labels": {int(k): v for k, v in spec.get("labels", {}).items()},
                "threshold": float(spec.get("threshold", 0.5)),
            }

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
            extra_heads=extra_heads,
            text_spec=text_spec,
            wordvec_spec=wordvec_spec,
            wordvec_pooler=wordvec_pooler,
            input_kind=meta.get("input_kind", "features"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vectorize(self, feat: Dict[str, str], query: Optional[str] = None):
        """Dense ``float32`` row ``(1, n_features)`` in feature_names order.

        Fills the categorical columns from the sparse *feat* dict (binary 0/1).
        When the bundle declares text feature blocks (``txt_*`` char-hash and/or
        ``wv_*`` word-vector) **and** *query* is supplied, those contiguous
        blocks are filled from the raw utterance with the bundle's recorded spec
        (numpy only — no torch). Categorical-only bundles ignore *query* and the
        behaviour is unchanged (back-compat).
        """
        import numpy as np

        row = np.zeros((1, len(self._feature_names)), dtype="float32")
        for i, name in enumerate(self._feature_names):
            if name in feat:
                row[0, i] = 1.0

        if query:
            if self._text_spec is not None and self._txt_idx:
                hv = hash_vector(query, self._text_spec)
                row[0, self._txt_idx[0]:self._txt_idx[0] + len(self._txt_idx)] = hv
            if self._wordvec_pooler is not None and self._wv_idx:
                wv = self._wordvec_pooler.pool(query)
                row[0, self._wv_idx[0]:self._wv_idx[0] + len(self._wv_idx)] = wv
        return row

    def _row_for(self, query: str, lang: str):
        """The model input for *query*.

        ``input_kind == "text"`` → a ``(1, 1)`` object array of the raw utterance
        (the baked-in skl2onnx vectorizer tokenizes it in-graph); otherwise the
        dense numeric feature row (categorical ⊕ optional text blocks).
        """
        import numpy as np

        if self._input_kind == "text":
            return np.array([[query or ""]], dtype=object)
        feat = self._extractor.extract(query, lang)
        return self._vectorize(feat, query=query)

    def _run(self, session, row) -> "list":
        """Run a head and return a 1-D probability array over its classes.

        Handles both graph shapes the bundle contract allows:

        * a **single numeric output** of raw logits — softmax is applied
          (a hand-built / guided-embeddings style head); and
        * the scikit-learn → ONNX shape ``[label_tensor, probability_tensor]``
          (``skl2onnx`` with ``zipmap=False``) — the probability tensor is
          already a normalised distribution and is returned as-is.

        The numeric ``(·, n_classes)`` output is located by scanning the outputs
        (an sklearn classifier graph's first output is the predicted *label*,
        often a string), so a bundle trained with ``training/train_sklearn.py``
        loads unchanged.
        """
        import numpy as np

        name = self._input_name or session.get_inputs()[0].name
        outputs = session.run(None, {name: row})

        probs = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.dtype.kind in "fiu" and arr.ndim >= 1 and arr.shape[-1] > 1:
                probs = arr
                break
        if probs is None:  # no numeric vector output — fall back to the first
            probs = np.asarray(outputs[0])
        if probs.ndim == 2:
            probs = probs[0]
        probs = probs.astype("float64").reshape(-1)

        # A genuine probability distribution (sklearn predict_proba) is
        # non-negative and sums to ~1; raw logits are not — softmax those.
        if probs.min() < 0.0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
            return _softmax(probs)
        return probs

    def _predict_domain(self, row) -> Tuple[OCPDomain, float]:
        try:
            probs = self._run(self._domain_session, row)
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

    def _predict_play(self, row) -> Tuple[str, float]:
        """Return (play_label_string, confidence); ("generic", 0.0) on failure."""
        try:
            probs = self._run(self._play_session, row)
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
        row = self._row_for(query, lang)
        domain, dconf = self._predict_domain(row)
        if domain != OCPDomain.OCP_PLAY:
            return "generic", dconf, domain
        label, pconf = self._predict_play(row)
        if pconf < self._play_thresh:
            return "generic", pconf, domain
        return label, pconf, domain

    # ------------------------------------------------------------------
    # Extended multi-task heads — run a named head when the bundle carries it.
    # ------------------------------------------------------------------

    def _has_head(self, axis: str) -> bool:
        return axis in self._heads

    def _single_head(self, axis: str, query: str, lang: str) -> Optional[Tuple[str, float]]:
        """Argmax a single-label head → ``(label, confidence)`` or ``None``."""
        head = self._heads.get(axis)
        if head is None:
            return None
        import numpy as np

        try:
            probs = self._run(head["session"], self._row_for(query, lang))
        except Exception as exc:
            LOG.error(f"OnnxMediaClassifier {axis} head failed: {exc}")
            return None
        idx = int(np.argmax(probs))
        return head["labels"].get(idx, ""), float(probs[idx])

    def _multi_head(self, axis: str, query: str, lang: str) -> Optional[List[str]]:
        """Threshold a multi-label head → list of labels ≥ threshold, or ``None``.

        Multi-label graphs may emit per-label probabilities as either a single
        ``(1, n_labels)`` tensor or a list of ``(1, 2)`` per-label tensors (one
        OvR estimator each); both are handled here.
        """
        head = self._heads.get(axis)
        if head is None:
            return None
        import numpy as np

        row = self._row_for(query, lang)
        thr = head["threshold"]
        labels = head["labels"]
        name = self._input_name or head["session"].get_inputs()[0].name
        try:
            outputs = head["session"].run(None, {name: row})
        except Exception as exc:
            LOG.error(f"OnnxMediaClassifier {axis} multi-head failed: {exc}")
            return None

        n = len(labels)
        arrays = [np.asarray(o) for o in outputs]

        # The OneVsRest → ONNX graph emits ``[label(1,n) int, probabilities(1,n)
        # float]``; prefer the FLOAT (·, n) tensor (per-label positive prob).
        per_label: List[float] = []
        prob = next((a for a in arrays
                     if a.dtype.kind == "f" and a.reshape(1, -1).shape[-1] == n),
                    None)
        if prob is not None:
            per_label = [float(x) for x in prob.reshape(-1)[:n]]
        else:
            # fallback: one (1,2) tensor per label → positive column
            for a in arrays:
                if a.dtype.kind in "fiu":
                    row2 = a.reshape(a.shape[0] if a.ndim > 1 else 1, -1)
                    per_label.append(float(row2[0, -1]))
        return [labels[i] for i in range(n)
                if i < len(per_label) and per_label[i] >= thr]

    # ------------------------------------------------------------------
    # AbstractMediaClassifier implementation
    # ------------------------------------------------------------------

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Classify the OCP domain via the dedicated domain head."""
        return self._predict_domain(self._row_for(query, lang))

    def _media_type_from_head(self, query: str, lang: str):
        """``(MediaType, conf)`` from the dedicated media_type head, or ``None``.

        The media_type head predicts the mediavocab leaf value directly (e.g.
        ``"movie"``); it is domain-gated so a non-OCP query returns GENERIC.
        """
        if not self._has_head("media_type"):
            return None
        domain, _ = self._predict_domain(self._row_for(query, lang))
        if domain != OCPDomain.OCP_PLAY:
            return MediaType.GENERIC, 0.0
        res = self._single_head("media_type", query, lang)
        if res is None:
            return None
        label, conf = res
        try:
            return MediaType(label), conf
        except ValueError:
            return LABEL_TO_MEDIA_TYPE.get(label, MediaType.GENERIC), conf

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Return (MediaType, confidence) for an ocp_play utterance.

        Uses the dedicated ``media_type`` head when the bundle carries it,
        otherwise the legacy play-label head.  Falls back to (GENERIC, 0.0) when
        the domain is not ocp_play, the confidence is below threshold, or the
        predicted type is not in *valid_labels*.
        """
        from_head = self._media_type_from_head(query, lang)
        if from_head is not None:
            media_type, conf = from_head
        else:
            label, conf, _ = self._play_label(query, lang)
            media_type = LABEL_TO_MEDIA_TYPE.get(label, MediaType.GENERIC)
        if media_type == MediaType.GENERIC:
            return MediaType.GENERIC, 0.0
        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0
        return media_type, conf

    # ------------------------------------------------------------------
    # Per-axis output — use a head when the bundle has one, else derive.
    # ------------------------------------------------------------------

    def classify_content_form_genres(self, query: str, lang: str) -> List[str]:
        """Sensitive / content-form genres (the content-filter axis).

        Prefers the dedicated ``content_form_genres`` multi-label head — so it
        can flag ``adult`` independently of the (uncertain) leaf — and falls back
        to the play-label genres otherwise.
        """
        if self._has_head("content_form_genres"):
            out = self._multi_head("content_form_genres", query, lang)
            if out is not None:
                return out
        return self.classify_genres(query, lang)

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Content-form genre tags (``adult`` / ``anime`` / …).

        Prefers the ``content_form_genres`` head; else the play-label genres.
        """
        if self._has_head("content_form_genres"):
            out = self._multi_head("content_form_genres", query, lang)
            if out is not None:
                return out
        label, _, _ = self._play_label(query, lang)
        return list(LABEL_TO_GENRES.get(label, []))

    def _enum_from_single_head(self, head: str, enum_cls, query: str, lang: str):
        """Coerce a single-label head's top prediction to *enum_cls*.

        Returns ``None`` when the bundle has no such head, the head abstains,
        or the predicted label is not a valid enum value — the caller then
        falls back to the inherited default.
        """
        res = self._single_head(head, query, lang)
        if res is not None and res[0]:
            try:
                return enum_cls(res[0])
            except ValueError:
                pass
        return None

    def _enums_from_multi_head(self, head: str, enum_cls, query: str, lang: str):
        """Coerce a multi-label head's predictions to a list of *enum_cls*.

        Invalid labels are skipped (the coerced list may be empty). Returns
        ``None`` when the bundle has no such head or the head abstains — the
        caller then falls back to the inherited default.
        """
        if self._has_head(head):
            out = self._multi_head(head, query, lang)
            if out is not None:
                values = []
                for v in out:
                    try:
                        values.append(enum_cls(v))
                    except ValueError:
                        continue
                return values
        return None

    def classify_content_form(self, query: str, lang: str):
        """The :class:`mediavocab.ContentForm` from the ``content_form`` head,
        else the inherited default."""
        from mediavocab.taxonomy import ContentForm
        out = self._enum_from_single_head("content_form", ContentForm, query, lang)
        return out if out is not None else super().classify_content_form(query, lang)

    def classify_programme_format(self, query: str, lang: str):
        """The :class:`mediavocab.ProgrammeFormat` from the head, else ``None``."""
        from mediavocab.taxonomy import ProgrammeFormat
        out = self._enum_from_single_head(
            "programme_format", ProgrammeFormat, query, lang)
        return out if out is not None else super().classify_programme_format(query, lang)

    def classify_accessibility(self, query: str, lang: str) -> List:
        """The :class:`mediavocab.AccessibilityKind` assets from the head.

        Multi-label; falls back to the inherited empty default.
        """
        from mediavocab.taxonomy import AccessibilityKind
        out = self._enums_from_multi_head(
            "accessibility", AccessibilityKind, query, lang)
        return out if out is not None else super().classify_accessibility(query, lang)

    def classify_variant(self, query: str, lang: str):
        """The :class:`mediavocab.VariantKind` from the ``variant`` head, else ``None``."""
        from mediavocab.taxonomy import VariantKind
        out = self._enum_from_single_head("variant", VariantKind, query, lang)
        return out if out is not None else super().classify_variant(query, lang)

    def classify_picture_format(self, query: str, lang: str) -> List:
        """The :class:`mediavocab.PictureFormat` attributes from the head.

        Multi-label; falls back to the inherited empty default.
        """
        from mediavocab.taxonomy import PictureFormat
        out = self._enums_from_multi_head(
            "picture_format", PictureFormat, query, lang)
        return out if out is not None else super().classify_picture_format(query, lang)

    def classify_playback_type(self, query: str, lang: str):
        """PlaybackType from the head when present, else derived from the leaf."""
        from mediavocab import PlaybackType
        out = self._enum_from_single_head("playback_type", PlaybackType, query, lang)
        return out if out is not None else super().classify_playback_type(query, lang)

    def classify_structure(self, query: str, lang: str):
        """Structure from the head when present, else derived from the leaf."""
        from mediavocab import Structure
        out = self._enum_from_single_head("structure", Structure, query, lang)
        return out if out is not None else super().classify_structure(query, lang)

    def classify_explicitness(self, query: str, lang: str) -> str:
        """Explicitness from the head when present, else derived from the form genres."""
        res = self._single_head("explicitness", query, lang)
        if res is not None and res[0]:
            return res[0]
        return super().classify_explicitness(query, lang)

    def classify_full(self, query: str, lang: str,
                      player_status=None, ner_list=None):
        """Full multi-axis result, predicting each axis from its own head.

        Each axis uses its dedicated head when the bundle carries one and
        otherwise derives (so partial / old bundles still produce a full result).
        The content-form genres come from the ``content_form_genres`` head, so a
        request can be flagged ``adult`` even when the leaf is uncertain
        (soft-gating).

        *player_status* / *ner_list* are the standalone context inputs (see
        :meth:`AbstractMediaClassifier.classify_full`); *player_status* is layered
        on by the shared base helper.
        """
        from ovos_media_classifier.axes import MediaClassification
        ctx = self._with_ner_context(ner_list)
        media_type, conf = ctx.classify(query, lang)
        domain, _ = ctx.classify_domain(query, lang)
        playback = ctx.classify_playback_type(query, lang)
        structure = ctx.classify_structure(query, lang)
        genres = ctx.classify_content_form_genres(query, lang)

        result = MediaClassification(
            media_type=media_type,
            playback_type=playback,
            structure=structure,
            domain=domain,
            genres=genres,
            confidence=conf,
        )
        return self._apply_player_status(ctx, query, lang, result, player_status)
