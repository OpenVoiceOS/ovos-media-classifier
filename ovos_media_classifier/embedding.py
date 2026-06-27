"""Optional **embedding-router** backend for ovos-media-classifier.

This is the learned guided-categorical-embeddings (GCE) router.  It is the
opt-in counterpart to the lean ``.voc`` keyword classifier: where keyword
matching is a high-precision *first pass* on explicit cues, the embedding
router handles the keyword-LESS cases and **abstains to GENERIC when unsure**
so a wrong route never prunes the provider that actually had the content.

Runtime dependencies — numpy + onnxruntime ONLY
-----------------------------------------------
The model is trained offline with ``guided-categorical-embeddings`` (PyTorch →
ONNX, ``training/train_embedding_router.py``).  At inference NO torch / GCE is
imported: every head is a plain ONNX graph run by ``onnxruntime``, the feature
row is assembled in pure numpy, and the abstain / reject decision is a numpy
threshold.  Both deps are imported lazily (inside ``from_path`` / ``__init__``)
so importing this module never pulls them in.

Two-stream feature layout (matches GCE's ``FeatureCombiner``)
-------------------------------------------------------------
The model input is ``[static | entity]`` concatenated::

    static  — the categorical ``kw_*`` / ``verb_*`` / ``mod_*`` columns from
              :class:`~ovos_media_classifier.features.CategoricalFeatureExtractor`,
              one-hot encoded in the saved ``vocab.json`` (``key=value``) order.
    entity  — one slot per train-time NER label (``artist_name`` / ``movie_title``
              / ``anime_title`` / …), in the ``entity_labels`` order recorded at
              export.  At train time these are seeded from representative entity
              samples; at runtime the user's OWN library is injected via
              :meth:`register_user_entities` **without retraining** — the entity
              block fires the same slots, so the router can route a bare title
              the keyword backend has no cue for.

Per-axis router bundle format
-----------------------------
The bundle is a directory with one GCE per-axis sub-export per routing axis,
plus a manifest::

    <bundle>/
      ├── router_meta.json        # {"axes": ["media_type","playback_type"],
      │                           #  "thresholds": {"media_type": 0.5, ...}}
      ├── media_type/             # GCE combiner export (classifier.onnx, vocab.json,
      │                           #  metadata.json{static_dim,entity_labels,labels,
      │                           #  temperature,abstain_label}, ner_entities.json)
      └── playback_type/          # idem

Each axis head emits **calibrated** probabilities (the train-time temperature is
baked into ``classifier.onnx``); the head argmaxes and, when the top
probability is below the axis threshold OR the argmax is the trained
``abstain_label``, routes to GENERIC (``predict_with_reject`` semantics).
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.features import CategoricalFeatureExtractor
from ovos_media_classifier.intents import (
    LABEL_TO_MEDIA_TYPE,
    NER_LABEL_TO_GENRES,
    NER_LABEL_TO_MEDIA_TYPE,
    MediaType,
    OCPDomain,
)

_ROUTER_META = "router_meta.json"
# axis value that means "no confident route" — the safe outcome.
GENERIC = "GENERIC"
# media-type sentinels (mediavocab values) that mean abstain on the routing eval.
_ABSTAIN_MEDIA = {"generic", "not_media", "control"}


def _softmax(arr):
    import numpy as np

    a = np.asarray(arr, dtype="float64").reshape(-1)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


class _NumpyEntityMatcher:
    """Word-boundary entity matcher → one-hot block (numpy, no ahocorasick).

    Mirrors GCE's ``EntityFeaturizer.one_hot_encode`` for the labels the model
    was trained on.  The label→samples map starts from the bundle's
    ``ner_entities.json`` (representative training entities) and is extended at
    runtime with the user's own library via :meth:`register`.  Only labels that
    were present at export (``entity_labels``) ever fire a column — unknown
    labels are accepted but ignored, matching ``RuntimeEntityInjector``.
    """

    def __init__(self, entity_labels: List[str]) -> None:
        self._labels = list(entity_labels)
        self._index = {lbl: i for i, lbl in enumerate(self._labels)}
        # label -> compiled word-boundary regex over its phrases
        self._phrases: Dict[str, List[str]] = {lbl: [] for lbl in self._labels}
        self._rx: Dict[str, "re.Pattern"] = {}

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def register(self, name: str, samples: List[str]) -> None:
        """Add entity phrases for a label (no-op for labels not in the bundle)."""
        if name not in self._index or not samples:
            return
        existing = set(self._phrases[name])
        added = [s for s in samples if s and s not in existing]
        if not added:
            return
        self._phrases[name].extend(added)
        self._compile(name)

    def _compile(self, name: str) -> None:
        phrases = self._phrases.get(name) or []
        if not phrases:
            self._rx.pop(name, None)
            return
        # longest first so a multi-word phrase wins over a fragment
        ordered = sorted(set(phrases), key=len, reverse=True)
        alt = "|".join(re.escape(p) for p in ordered)
        self._rx[name] = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)

    def one_hot(self, utterance: str):
        """Binary vector of length ``len(entity_labels)`` of fired entity slots."""
        import numpy as np

        vec = np.zeros((len(self._labels),), dtype="float32")
        if not utterance:
            return vec
        for name, rx in self._rx.items():
            if rx.search(utterance):
                vec[self._index[name]] = 1.0
        return vec

    def fired_labels(self, utterance: str) -> List[str]:
        """The entity labels whose phrases appear in *utterance*."""
        if not utterance:
            return []
        return [name for name, rx in self._rx.items() if rx.search(utterance)]


class _AxisHead:
    """One per-axis GCE export: vocab + entity layout + calibrated ONNX head."""

    def __init__(self, axis_dir: str, threshold: float) -> None:
        import onnxruntime as ort

        from guided_categorical_embeddings.vectorizer import CategoricalVectorizer

        with open(os.path.join(axis_dir, "metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.static_dim: int = int(meta["static_dim"])
        self.entity_labels: List[str] = list(meta.get("entity_labels", []))
        self.labels: List[str] = list(meta["labels"])
        self.abstain_label: Optional[str] = meta.get("abstain_label")
        self.temperature: float = float(meta.get("temperature", 1.0) or 1.0)
        self.threshold: float = float(threshold)

        self.vectorizer = CategoricalVectorizer()
        self.vectorizer.load(os.path.join(axis_dir, "vocab.json"))

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            os.path.join(axis_dir, "classifier.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Runtime entity matcher starts EMPTY: the entity stream's value is the
        # user's OWN injected library, not the bundle's representative training
        # seeds (those are noisy across millions of titles and would over-match
        # arbitrary words at inference — e.g. "death metal" hitting a seeded
        # ``book_genre``).  The trained head already abstains on featureless
        # inputs; entity slots only fire once the user injects real titles via
        # ``register_user_entities``.  ``ner_entities.json`` still ships for the
        # export contract / ``RuntimeEntityInjector`` provenance.
        self.matcher = _NumpyEntityMatcher(self.entity_labels)

    def _row(self, feat: Dict[str, str], utterance: str):
        import numpy as np

        static = self.vectorizer.transform([feat])  # (1, static_dim)
        entity = self.matcher.one_hot(utterance).reshape(1, -1)  # (1, entity_dim)
        return np.hstack([static, entity]).astype("float32")

    def predict(self, feat: Dict[str, str], utterance: str) -> Tuple[str, float]:
        """``(label, confidence)`` argmax over calibrated probabilities."""
        import numpy as np

        probs = self._session.run(None, {self._input_name: self._row(feat, utterance)})[0]
        probs = np.asarray(probs, dtype="float64").reshape(-1)
        # classifier.onnx already emits a softmax distribution; guard a logit head.
        if probs.min() < 0.0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
            probs = _softmax(probs)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def predict_with_reject(self, feat: Dict[str, str],
                            utterance: str) -> Tuple[str, float]:
        """``(label, conf)`` with reject → :data:`GENERIC` below threshold.

        Routes to GENERIC when the calibrated top probability is below the axis
        threshold OR the argmax is the trained abstain class — the harmless
        outcome (every provider still searches).
        """
        label, conf = self.predict(feat, utterance)
        if conf < self.threshold or label == self.abstain_label:
            return GENERIC, conf
        return label, conf


class EmbeddingMediaClassifier(AbstractMediaClassifier):
    """Learned per-axis embedding router (numpy + onnxruntime inference).

    Routes ``media_type`` and ``playback_type`` from independent calibrated GCE
    heads over a two-stream ``[categorical | entity]`` feature vector, abstaining
    to GENERIC when unsure.  The gate (``classify_domain`` / ``is_ocp_query``)
    and the content-policy axis (``classify_content_form_genres`` → the adult
    lexicon) are intentionally **kept on the keyword backend** (composed via the
    hybrid, or derived here) so the router never regresses the 0.0 adult-leak
    floor or introduces a false hijack.

    Use :meth:`from_path` to load a router bundle; use
    :meth:`register_user_entities` to inject the user's media library at runtime
    (no retraining).

    Args:
        media_type_head: the ``media_type`` axis head (required).
        playback_type_head: optional ``playback_type`` axis head.
        extractor: the categorical feature extractor (keyword ``.voc`` columns).
    """

    def __init__(
        self,
        media_type_head: _AxisHead,
        playback_type_head: Optional[_AxisHead],
        extractor: CategoricalFeatureExtractor,
    ) -> None:
        self._mt = media_type_head
        self._pb = playback_type_head
        self._extractor = extractor
        self._heads: Dict[str, _AxisHead] = {"media_type": media_type_head}
        if playback_type_head is not None:
            self._heads["playback_type"] = playback_type_head

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        model_dir: str,
        locale_dir: Optional[str] = None,
    ) -> "EmbeddingMediaClassifier":
        """Load an :class:`EmbeddingMediaClassifier` from a router bundle dir.

        Args:
            model_dir: bundle dir with ``router_meta.json`` and one sub-dir per
                axis (``media_type/``, optional ``playback_type/``).
            locale_dir: override the locale dir for keyword features.

        Raises:
            ImportError: when onnxruntime / numpy / GCE's vectorizer are missing.
            FileNotFoundError / ValueError: when the bundle is incomplete.
        """
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise ImportError(
                "The embedding-router backend requires onnxruntime + numpy. "
                "Install with: pip install ovos-media-classifier[onnx]"
            ) from exc

        meta_path = os.path.join(model_dir, _ROUTER_META)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"router bundle missing {meta_path}")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        axes = meta.get("axes") or []
        thresholds = meta.get("thresholds") or {}
        if "media_type" not in axes:
            raise ValueError("router_meta.json must list a 'media_type' axis")

        def _head(axis: str) -> Optional[_AxisHead]:
            axis_dir = os.path.join(model_dir, axis)
            if not os.path.isdir(axis_dir):
                return None
            return _AxisHead(axis_dir, float(thresholds.get(axis, 0.5)))

        mt_head = _head("media_type")
        if mt_head is None:
            raise FileNotFoundError(
                f"router bundle missing media_type/ axis in {model_dir}")
        pb_head = _head("playback_type") if "playback_type" in axes else None

        extractor = CategoricalFeatureExtractor.from_locale_dir(locale_dir)
        return cls(mt_head, pb_head, extractor)

    # ------------------------------------------------------------------
    # Runtime entity injection (no retraining)
    # ------------------------------------------------------------------

    def register_user_entities(self, name: str, samples: List[str]) -> None:
        """Inject the user's own library for an entity label across every head.

        ``name`` is a train-time NER label (``artist_name`` / ``movie_title`` /
        ``anime_title`` / ``audiobook_title`` / …); ``samples`` are the user's
        titles.  Unknown labels are ignored.  No retraining: the entity block
        fires the matching slot so the router can route a bare title.
        """
        for head in self._heads.values():
            head.matcher.register(name, samples)

    def register_user_library(self, library: Dict[str, List[str]]) -> None:
        """Bulk :meth:`register_user_entities` from a ``{label: [titles]}`` dict."""
        for name, samples in library.items():
            self.register_user_entities(name, samples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feat(self, query: str, lang: str) -> Dict[str, str]:
        return self._extractor.extract(query, lang)

    def _entity_media_type(self, query: str) -> Optional[MediaType]:
        """A confident MediaType implied by a fired entity slot, if any.

        When a runtime-injected entity matches, its NER label deterministically
        maps to a MediaType (``anime_title`` → EPISODIC_SERIES, ``audiobook_title``
        → AUDIOBOOK, …).  This is what lets a freshly injected library route a
        bare title the model itself never saw — and it is a *confident* entity
        signal, so it overrides a low-confidence head abstain.
        """
        fired = self._mt.matcher.fired_labels(query)
        for label in fired:
            mt = NER_LABEL_TO_MEDIA_TYPE.get(label)
            if mt is not None and mt != MediaType.GENERIC:
                return mt
        return None

    # ------------------------------------------------------------------
    # AbstractMediaClassifier implementation
    # ------------------------------------------------------------------

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Route the leaf ``MediaType``, abstaining to GENERIC when unsure.

        A fired runtime entity (injected library) gives a confident
        deterministic route; otherwise the calibrated ``media_type`` head decides
        with its reject threshold.
        """
        feat = self._feat(query, lang)
        label, conf = self._mt.predict_with_reject(feat, query)

        media_type = MediaType.GENERIC
        if label != GENERIC:
            try:
                media_type = MediaType(label)
            except ValueError:
                media_type = LABEL_TO_MEDIA_TYPE.get(label, MediaType.GENERIC)

        # entity signal: a fired injected library entity is a confident route
        # that rescues a head abstain (the entity-gap mis-routes).
        if media_type == MediaType.GENERIC:
            ent_mt = self._entity_media_type(query)
            if ent_mt is not None:
                media_type, conf = ent_mt, max(conf, self._mt.threshold)

        if media_type == MediaType.GENERIC:
            return MediaType.GENERIC, 0.0
        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0
        return media_type, conf

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Derive the domain from the routed leaf (no dedicated gate head).

        The router has no negative training data, so it does NOT decide the gate
        on its own — a non-GENERIC leaf infers OCP_PLAY, else NOT_OCP.  In the
        hybrid the keyword gate is authoritative; standalone this stays
        conservative (abstain → NOT_OCP) to avoid false hijacks.
        """
        media_type, conf = self.classify(query, lang)
        if media_type != MediaType.GENERIC:
            return OCPDomain.OCP_PLAY, conf
        return OCPDomain.NOT_OCP, 0.0

    def classify_playback_type(self, query: str, lang: str):
        """PlaybackType from the head (reject → derive from the leaf)."""
        from mediavocab import PlaybackType

        if self._pb is not None:
            feat = self._feat(query, lang)
            label, _ = self._pb.predict_with_reject(feat, query)
            if label != GENERIC:
                try:
                    return PlaybackType(label)
                except ValueError:
                    pass
        return super().classify_playback_type(query, lang)


class HybridMediaClassifier(AbstractMediaClassifier):
    """Keyword first pass + embedding-router fallback (the recommended wiring).

    The keyword backend is the **high-precision first pass**: it owns the gate
    (``classify_domain`` / ``is_ocp_query``) and the content-policy axis (the
    adult lexicon → 0.0 leak), and when it confidently routes a leaf
    ``MediaType`` from an explicit cue that route wins.  The embedding router is
    consulted ONLY for the keyword-LESS cases — when keyword abstains to GENERIC
    — and itself abstains when unsure.  So the hybrid:

    * never overrides a confident keyword route into a wrong one (keyword wins
      ties on explicit cues);
    * keeps adult-leak / false-hijack exactly at the keyword floor (those axes
      are answered by keyword);
    * recovers the keyword backend's abstains/entity-gap mis-routes via the
      router + runtime entity injection.

    Args:
        keyword: the keyword (``.voc``) backend (high-precision first pass).
        router: the :class:`EmbeddingMediaClassifier` (keyword-less fallback).
    """

    def __init__(self, keyword: AbstractMediaClassifier,
                 router: EmbeddingMediaClassifier) -> None:
        self.keyword = keyword
        self.router = router

    @classmethod
    def from_path(cls, model_dir: str,
                  locale_dir: Optional[str] = None,
                  keyword: Optional[AbstractMediaClassifier] = None
                  ) -> "HybridMediaClassifier":
        """Load the router bundle and pair it with a keyword first pass."""
        from ovos_media_classifier.keyword import KeywordMediaClassifier

        router = EmbeddingMediaClassifier.from_path(model_dir, locale_dir)
        return cls(keyword or KeywordMediaClassifier(), router)

    # ---- runtime entity injection delegates to the router ----
    def register_user_entities(self, name: str, samples: List[str]) -> None:
        self.router.register_user_entities(name, samples)

    def register_user_library(self, library: Dict[str, List[str]]) -> None:
        self.router.register_user_library(library)

    # ---- gate + content policy: keyword is authoritative ----
    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Keyword owns the gate verbatim — preserves false-hijack / control.

        The router never moves the gate: it has no negative-gate training data,
        and upgrading NOT_OCP→OCP_PLAY on a fired entity would hijack ordinary
        speech that merely contains a library phrase (a common short title in
        "the daily forecast"). The router only refines the *leaf* once keyword
        has admitted the turn into OCP, so adult-leak / false-hijack stay exactly
        at the keyword floor.
        """
        return self.keyword.classify_domain(query, lang)

    def classify_control(self, query: str, lang: str):
        return self.keyword.classify_control(query, lang)

    def classify_content_form_genres(self, query: str, lang: str) -> List[str]:
        # the adult lexicon lives in keyword → never regress 0.0 leak.
        return self.keyword.classify_content_form_genres(query, lang)

    def classify_genres(self, query: str, lang: str) -> List[str]:
        return self.keyword.classify_genres(query, lang)

    # ---- media_type: keyword confident route wins, else router ----
    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Injected-entity match wins, else keyword's confident leaf, else router.

        Precedence:

        1. A fired **user-library entity** (injected at runtime) is the highest
           -precision evidence — "Attack on Titan" being in the user's anime
           library beats the generic ``watch`` → MOVIE keyword cue — so it
           overrides even a confident keyword leaf.  This is what closes the
           keyword backend's entity-gap mis-routes.  (Empty by default, so with
           no injected library this branch is inert and keyword wins unchanged.)
        2. Keyword's confident leaf (explicit cue) — its high-precision route.
        3. The router, which routes the keyword-less cases and abstains when
           unsure (so the keyword floor's mis-routes are never made worse).
        """
        ent_mt = self.router._entity_media_type(query)
        if ent_mt is not None:
            if valid_labels is None or ent_mt in valid_labels:
                return ent_mt, max(0.5, self.router._mt.threshold)

        kw_mt, kw_conf = self.keyword.classify(query, lang, valid_labels)
        if kw_mt != MediaType.GENERIC:
            return kw_mt, kw_conf
        return self.router.classify(query, lang, valid_labels)

    def classify_playback_type(self, query: str, lang: str):
        """Playback kept consistent with the leaf :meth:`classify` returns.

        Whatever wins the leaf decides the modality: an injected-entity or
        keyword leaf derives its playback from that type (so a video anime title
        never reports AUDIO from a confident-but-wrong playback head); only when
        the leaf itself came from the router head do we trust the router's
        playback head, else derive.
        """
        from mediavocab import infer_playback_type

        # entity override / keyword confident leaf → derive from that leaf so the
        # two axes never disagree for the headline injected-library case.
        ent_mt = self.router._entity_media_type(query)
        if ent_mt is not None:
            return infer_playback_type(ent_mt)
        kw_mt, _ = self.keyword.classify(query, lang)
        if kw_mt != MediaType.GENERIC:
            return self.keyword.classify_playback_type(query, lang)
        # keyword-less: the router decides both leaf and playback.
        rt_pb = self.router.classify_playback_type(query, lang)
        if rt_pb is not None and rt_pb != infer_playback_type(MediaType.GENERIC):
            return rt_pb
        return self.keyword.classify_playback_type(query, lang)
