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
# synthetic entity label for cross-media-ambiguous gazetteer titles: they fire
# the matcher (longest-match) but map to abstain, never to a confident MediaType.
AMBIGUOUS_LABEL = "__ambiguous__"
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

    Three precision tiers are tracked per label:

    * **user** — the user's own injected library; high precision, allowed to
      override even a confident keyword route in the hybrid.
    * **gazetteer** — the Layer A default offline gazetteer of common real
      titles, restricted to titles that are UNAMBIGUOUS across media types; lower
      precision, so the hybrid consults it ONLY to fill an abstention, never to
      override a confident keyword leaf.
    * **ambiguous** — cross-media-ambiguous gazetteer titles (a title present in
      more than one type pool, e.g. "Dune" film/tv/book, "Moby Dick" book/tv).
      These are registered so the matcher still SEES them (longest-match
      semantics: "moby dick" beats the artist "moby"), but they deterministically
      map to **abstain** (GENERIC) — the gazetteer cannot disambiguate them, so a
      wrong confident leaf is never emitted.  Registered under the synthetic
      :data:`AMBIGUOUS_LABEL` so it has no MediaType.

    ``one_hot`` (the model's entity stream) fires on the user + gazetteer tiers
    (real entity slots only); the ambiguous tier never fires a model slot — it
    exists purely to win the longest-match and force an abstain.  The hybrid's
    *override* decision distinguishes tiers via ``fired_labels(..., tier=...)``.
    """

    _TIERS = ("user", "gazetteer", "ambiguous")
    #: tiers whose matches feed the model's one-hot entity stream (real labels)
    _MODEL_TIERS = ("user", "gazetteer")

    def __init__(self, entity_labels: List[str]) -> None:
        self._labels = list(entity_labels)
        self._index = {lbl: i for i, lbl in enumerate(self._labels)}
        # tier -> label -> phrases, and tier -> label -> compiled regex.
        # The ambiguous tier keeps its phrases under the synthetic AMBIGUOUS_LABEL
        # (no model slot), so it can carry titles not in ``entity_labels``.
        self._phrases: Dict[str, Dict[str, List[str]]] = {
            t: {lbl: [] for lbl in self._labels} for t in self._TIERS}
        self._phrases["ambiguous"] = {AMBIGUOUS_LABEL: []}
        self._rx: Dict[str, Dict[str, "re.Pattern"]] = {
            t: {} for t in self._TIERS}

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def register(self, name: str, samples: List[str],
                 tier: str = "user") -> None:
        """Add entity phrases for a label in *tier* (no-op for unknown labels).

        The ``ambiguous`` tier stores phrases under :data:`AMBIGUOUS_LABEL`
        regardless of *name*; every other tier requires a known entity label.
        """
        if not samples or tier not in self._TIERS:
            return
        if tier == "ambiguous":
            name = AMBIGUOUS_LABEL
        elif name not in self._index:
            return
        existing = set(self._phrases[tier][name])
        added = [s for s in samples if s and s not in existing]
        if not added:
            return
        self._phrases[tier][name].extend(added)
        self._compile(tier, name)

    def _compile(self, tier: str, name: str) -> None:
        phrases = self._phrases[tier].get(name) or []
        if not phrases:
            self._rx[tier].pop(name, None)
            return
        # longest first so a multi-word phrase wins over a fragment
        ordered = sorted(set(phrases), key=len, reverse=True)
        alt = "|".join(re.escape(p) for p in ordered)
        self._rx[tier][name] = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)

    def one_hot(self, utterance: str):
        """Binary vector of length ``len(entity_labels)`` of fired entity slots.

        Fires on the model tiers (user + gazetteer) only — the model's entity
        stream sees any known *real-label* title.  The ambiguous tier has no
        model slot (it maps to abstain) so it never sets a column.
        """
        import numpy as np

        vec = np.zeros((len(self._labels),), dtype="float32")
        if not utterance:
            return vec
        for tier in self._MODEL_TIERS:
            for name, rx in self._rx[tier].items():
                if name in self._index and rx.search(utterance):
                    vec[self._index[name]] = 1.0
        return vec

    def fired_labels(self, utterance: str,
                     tier: Optional[str] = None) -> List[str]:
        """The entity labels whose phrases appear in *utterance*.

        *tier* ``None`` → either tier; ``"user"`` / ``"gazetteer"`` → that tier
        only (so the hybrid can let only user entities override keyword).
        """
        return [lbl for lbl, _ in self.fired_labels_with_len(utterance, tier)]

    def fired_labels_with_len(self, utterance: str, tier: Optional[str] = None
                              ) -> List[Tuple[str, int]]:
        """``(label, matched_phrase_len)`` for every fired label.

        The matched length lets the caller prefer the longest match — so a
        multi-word "Moby Dick" (book) beats a single-word "Moby" (artist) that
        fires inside it.
        """
        if not utterance:
            return []
        tiers = self._TIERS if tier is None else (tier,)
        out: Dict[str, int] = {}
        for t in tiers:
            for name, rx in self._rx[t].items():
                m = rx.search(utterance)
                if m:
                    out[name] = max(out.get(name, 0), len(m.group(0)))
        return sorted(out.items(), key=lambda kv: kv[1], reverse=True)


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

        These are the high-precision ``user`` tier — they may override even a
        confident keyword route in the hybrid.
        """
        for head in self._heads.values():
            head.matcher.register(name, samples, tier="user")

    def register_user_library(self, library: Dict[str, List[str]]) -> None:
        """Bulk :meth:`register_user_entities` from a ``{label: [titles]}`` dict."""
        for name, samples in library.items():
            self.register_user_entities(name, samples)

    def _with_ner_context(self, ner_list):
        """Inject the per-query available-entity list (``{label: [entity]}``).

        Threads the caller's live entity context (skill-registered keywords + the
        user's library) into the entity stream as high-precision ``user``-tier
        entities, so the router routes a bare title the user actually has — no
        retraining.  Registration dedups, so passing the same list each turn is
        cheap.  Returns ``self`` (the matcher is the shared injection point).
        """
        if ner_list:
            self.register_user_library(ner_list)
        return self

    def register_default_gazetteer(self, top_n: Optional[int] = None,
                                   path: Optional[str] = None) -> int:
        """Inject the **Layer A offline popularity gazetteer** as a default library.

        Loads the popularity-ranked gazetteer of common real titles
        (:mod:`ovos_media_classifier.gazetteer`) and registers it across every
        head's entity matcher *in addition to* anything the user injected — so
        the router recognises common real titles ("cowboy bebop" → anime →
        EPISODIC_SERIES) offline, with NO network call and without the user
        having saved them.  Adult labels are never present in the gazetteer.

        Args:
            top_n: per-type cap (``None`` → :data:`gazetteer.DEFAULT_TOP_N`;
                ``<=0`` → no cap).
            path: explicit gazetteer JSON; default resolution otherwise.

        Returns:
            the number of titles registered (0 when no gazetteer is available).
        """
        from ovos_media_classifier.gazetteer import (
            DEFAULT_TOP_N, cross_pool_titles, load_default_gazetteer,
        )
        cap = DEFAULT_TOP_N if top_n is None else top_n
        # Ambiguity is a property of the FULL pools, so compute it on the
        # uncapped gazetteer: a title is cross-media-ambiguous if it appears in
        # >1 type pool anywhere, even when the per-type cap would later keep it in
        # only one.  (Capping first could hide an ambiguity and let a wrong leaf
        # route.)
        full = load_default_gazetteer(top_n=None, path=path,
                                      drop_ambiguous=False)
        ambiguous = set(cross_pool_titles(full))
        # The routable (capped) gazetteer with the ambiguous titles removed.
        gaz = load_default_gazetteer(top_n=cap, path=path, drop_ambiguous=True)
        registered = 0
        for name, samples in gaz.items():
            for head in self._heads.values():
                head.matcher.register(name, samples, tier="gazetteer")
            registered += len(samples)
        # Register the ambiguous titles into the abstain tier so the matcher
        # still SEES them (longest-match): "moby dick" beats the artist "moby"
        # inside it, and resolves to abstain rather than a wrong MUSIC route.
        for name, samples in full.items():
            ambig = [s for s in samples if str(s).lower() in ambiguous]
            if ambig:
                for head in self._heads.values():
                    head.matcher.register(name, ambig, tier="ambiguous")
        return registered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feat(self, query: str, lang: str) -> Dict[str, str]:
        return self._extractor.extract(query, lang)

    def _entity_media_type(self, query: str,
                           tier: Optional[str] = None) -> Optional[MediaType]:
        """A confident MediaType implied by a fired entity slot, if any.

        When a runtime-injected entity matches, its NER label deterministically
        maps to a MediaType (``anime_title`` → EPISODIC_SERIES, ``audiobook_title``
        → AUDIOBOOK, …).  This is what lets a freshly injected library route a
        bare title the model itself never saw — and it is a *confident* entity
        signal, so it overrides a low-confidence head abstain.

        *tier* restricts which precision tier may fire (``"user"`` → only the
        user's library; ``None`` → user + gazetteer).

        Disambiguation:

        * **Longest match wins** — a multi-word "Moby Dick" (book) beats a
          single-word "Moby" (artist) firing inside it.
        * **Conflict → abstain** — when the longest-matching phrase maps to
          MORE THAN ONE distinct MediaType (a title that genuinely exists across
          media, e.g. "Dune" movie/tv/book, "Watchmen" movie/comic), the
          gazetteer cannot disambiguate, so it returns ``None`` (safe abstain —
          a wrong confident leaf would prune the right provider) rather than
          guessing.  A title that fires several labels mapping to the SAME type
          still routes.
        """
        fired = self._mt.matcher.fired_labels_with_len(query, tier=tier)
        if not fired:
            return None
        best_len = fired[0][1]
        types = set()
        for label, length in fired:
            if length < best_len:
                break  # only consider the longest-matching phrases
            mt = NER_LABEL_TO_MEDIA_TYPE.get(label)
            if mt is not None and mt != MediaType.GENERIC:
                types.add(mt)
        if len(types) == 1:
            return next(iter(types))
        return None  # no fire, or ambiguous across types → abstain

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

    Layered fall-through (cheapest → most expensive)
    ------------------------------------------------
    Routing falls through three layers, and a later layer only ever **fills an
    abstention** — it never overrides a confident earlier route nor moves the
    gate:

    1. **keyword** (explicit cues) — owns the gate + adult policy; a confident
       leaf wins outright.
    2. **embedding router + user library + Layer A offline gazetteer** (offline)
       — fills keyword's abstains for common real titles, no network.
    3. **online metadatarr** (network, *opt-in*, :data:`online`) — consulted
       ONLY when the cheaper layers abstain, and itself abstains on
       failure/timeout/low-confidence.  Off unless an online backend is wired.

    Args:
        keyword: the keyword (``.voc``) backend (high-precision first pass).
        router: the :class:`EmbeddingMediaClassifier` (keyword-less fallback).
        online: optional online backend (e.g.
            :class:`~ovos_media_classifier.metadatarr_backend.MetadatarrMediaClassifier`),
            consulted last and only when the cheaper layers abstain.
    """

    def __init__(self, keyword: AbstractMediaClassifier,
                 router: EmbeddingMediaClassifier,
                 online: Optional[AbstractMediaClassifier] = None) -> None:
        self.keyword = keyword
        self.router = router
        self.online = online

    @classmethod
    def from_path(cls, model_dir: str,
                  locale_dir: Optional[str] = None,
                  keyword: Optional[AbstractMediaClassifier] = None,
                  online: Optional[AbstractMediaClassifier] = None
                  ) -> "HybridMediaClassifier":
        """Load the router bundle and pair it with a keyword first pass.

        Pass *online* (an instantiated online backend) to enable the network
        last-resort layer; omit it to keep the hybrid fully offline.
        """
        from ovos_media_classifier.keyword import KeywordMediaClassifier

        router = EmbeddingMediaClassifier.from_path(model_dir, locale_dir)
        return cls(keyword or KeywordMediaClassifier(), router, online=online)

    # ---- runtime entity injection delegates to the router ----
    def register_user_entities(self, name: str, samples: List[str]) -> None:
        self.router.register_user_entities(name, samples)

    def register_user_library(self, library: Dict[str, List[str]]) -> None:
        self.router.register_user_library(library)

    def _with_ner_context(self, ner_list):
        """Inject the per-query available-entity list into the router (see
        :meth:`EmbeddingMediaClassifier._with_ner_context`)."""
        if ner_list:
            self.register_user_library(ner_list)
        return self

    def register_default_gazetteer(self, top_n: Optional[int] = None,
                                   path: Optional[str] = None) -> int:
        """Inject the Layer A offline gazetteer into the router (see
        :meth:`EmbeddingMediaClassifier.register_default_gazetteer`)."""
        return self.router.register_default_gazetteer(top_n=top_n, path=path)

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
        # only the USER tier overrides a confident keyword leaf; the gazetteer
        # (lower precision) must not — it only fills keyword abstains via the
        # router fallback below, so "listen to attack on titan soundtrack" keeps
        # its confident keyword MUSIC route instead of being hijacked to anime.
        ent_mt = self.router._entity_media_type(query, tier="user")
        if ent_mt is not None:
            if valid_labels is None or ent_mt in valid_labels:
                return ent_mt, max(0.5, self.router._mt.threshold)

        kw_mt, kw_conf = self.keyword.classify(query, lang, valid_labels)
        if kw_mt != MediaType.GENERIC:
            return kw_mt, kw_conf
        rt_mt, rt_conf = self.router.classify(query, lang, valid_labels)
        if rt_mt != MediaType.GENERIC:
            return rt_mt, rt_conf
        # Layer 3 — online metadatarr, only when keyword + offline both abstain.
        if self.online is not None:
            on_mt, on_conf = self.online.classify(query, lang, valid_labels)
            if on_mt != MediaType.GENERIC:
                return on_mt, on_conf
        return MediaType.GENERIC, 0.0

    def classify_playback_type(self, query: str, lang: str):
        """Playback kept consistent with the leaf :meth:`classify` returns.

        Whatever wins the leaf decides the modality: an injected-entity or
        keyword leaf derives its playback from that type (so a video anime title
        never reports AUDIO from a confident-but-wrong playback head); only when
        the leaf itself came from the router head do we trust the router's
        playback head, else derive.
        """
        from mediavocab import infer_playback_type

        # user-tier entity override / keyword confident leaf → derive from that
        # leaf so the two axes never disagree for the headline injected-library
        # case (gazetteer entities are NOT an override here, only a fallback).
        ent_mt = self.router._entity_media_type(query, tier="user")
        if ent_mt is not None:
            return infer_playback_type(ent_mt)
        kw_mt, _ = self.keyword.classify(query, lang)
        if kw_mt != MediaType.GENERIC:
            return self.keyword.classify_playback_type(query, lang)
        # keyword-less: the router decides both leaf and playback.
        rt_mt, _ = self.router.classify(query, lang)
        if rt_mt != MediaType.GENERIC:
            rt_pb = self.router.classify_playback_type(query, lang)
            if rt_pb is not None and rt_pb != infer_playback_type(MediaType.GENERIC):
                return rt_pb
            return infer_playback_type(rt_mt)
        # Layer 3 — online: derive playback from whatever online resolved.
        if self.online is not None:
            on_mt, _ = self.online.classify(query, lang)
            if on_mt != MediaType.GENERIC:
                return self.online.classify_playback_type(query, lang)
        return self.keyword.classify_playback_type(query, lang)
