"""ovos-media-classifier — media-type classification for OCP.

The initial release ships a single, zero-ML-dependency strategy: **keyword
(``.voc``) matching** — the minimum required for OCP to be functional.  Richer
strategies (ONNX, NER, …) are opt-in plugins discovered through the
``opm.media.classifier`` entry-point group and land as independent additions.

Public API::

    from ovos_media_classifier import load_media_classifier

    clf = load_media_classifier()                       # bundled .voc keyword classifier
    media_type, conf = clf.classify("play some jazz", "en-us")   # -> (mediavocab.MediaType, conf)
    is_ocp, conf     = clf.is_ocp_query("play something", "en-us")
    domain, conf     = clf.classify_domain("pause the music", "en-us")
    genres           = clf.classify_genres("play hentai", "en-us")   # -> ["anime", "adult"]

    # content filtering (adult blocked by default)
    from ovos_media_classifier import ContentFilter
    blocked, reason = ContentFilter().check(clf, "play porn", "en-us")

    # use an external classifier plugin (opm.media.classifier)
    clf = load_media_classifier({"media_classifier_plugin": "ovos-media-classifier-onnx"})

The classifier models media on the real axes only: the ``OCPDomain`` (play /
control / not-ocp) × the canonical ``mediavocab.MediaType`` + ``mediavocab``
genre tags.  Raw detection labels resolve straight to ``(MediaType, genres)``.
"""
from typing import Callable, Optional

from ovos_utils.log import LOG

from ovos_media_classifier.axes import (
    Structure,
    MediaClassification,
    MEDIA_TYPE_TO_STRUCTURE,
    infer_structure,
)
from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.content_filter import ContentFilter
from ovos_media_classifier.context import (
    PlayerStatus,
    PlayerState,
)
from ovos_media_classifier.intents import (
    MediaType,
    OCPDomain,
    OCPControlIntent,
    OCPEntityLabel,
    LABEL_TO_MEDIA_TYPE,
    LABEL_TO_GENRES,
    NER_LABEL_TO_MEDIA_TYPE,
    NER_LABEL_TO_GENRES,
    genres_for_label,
)
from ovos_media_classifier.keyword import KeywordMediaClassifier
from ovos_media_classifier.embedding import (
    EmbeddingMediaClassifier,
    HybridMediaClassifier,
)
from ovos_media_classifier.gazetteer import (
    build_gazetteer,
    load_default_gazetteer,
)
from ovos_media_classifier.slots import (
    KeywordFeatureSlot,
    KEYWORD_FEATURE_SLOTS,
    slot_for_label,
    slots_for_media_type,
)
from ovos_media_classifier.plugins import (
    find_media_classifier_plugins,
    load_media_classifier_plugin,
)
from ovos_media_classifier.version import __version__

__all__ = [
    "AbstractMediaClassifier",
    "MediaType",
    "ContentFilter",
    "PlayerStatus",
    "PlayerState",
    "Structure",
    "MediaClassification",
    "MEDIA_TYPE_TO_STRUCTURE",
    "infer_structure",
    "OCPDomain",
    "OCPControlIntent",
    "OCPEntityLabel",
    "LABEL_TO_MEDIA_TYPE",
    "LABEL_TO_GENRES",
    "NER_LABEL_TO_MEDIA_TYPE",
    "NER_LABEL_TO_GENRES",
    "genres_for_label",
    "KeywordMediaClassifier",
    "EmbeddingMediaClassifier",
    "HybridMediaClassifier",
    "build_gazetteer",
    "load_default_gazetteer",
    "KeywordFeatureSlot",
    "KEYWORD_FEATURE_SLOTS",
    "slot_for_label",
    "slots_for_media_type",
    "find_media_classifier_plugins",
    "load_media_classifier_plugin",
    "load_media_classifier",
    "__version__",
]


# ---------------------------------------------------------------------------
# Backend factory
#
# Each opt-in backend is one (name, trigger keys, install hint, builder) spec.
# ``load_media_classifier`` walks the specs in priority order; a backend whose
# trigger key is present is attempted, and *any* failure (missing extra, bad
# bundle) logs a warning and falls through so the zero-ML keyword default is
# always preserved.
# ---------------------------------------------------------------------------


def _build_external_plugin(config: dict) -> AbstractMediaClassifier:
    """``media_classifier_plugin`` — an external classifier registered under
    the ``opm.media.classifier`` entry-point group."""
    plugin_name = config["media_classifier_plugin"]
    clf = load_media_classifier_plugin(plugin_name, config)
    LOG.info(f"OCP media classifier: external plugin ({plugin_name})")
    return clf


def _build_onnx(config: dict) -> AbstractMediaClassifier:
    """``media_classifier_onnx_model`` — the opt-in ONNX trained backend
    loaded from a bundle directory. Lazy import: onnxruntime/numpy must NOT
    be touched on the default import path."""
    onnx_model = config["media_classifier_onnx_model"]
    from ovos_media_classifier.onnx import OnnxMediaClassifier
    clf = OnnxMediaClassifier.from_path(onnx_model)
    LOG.info(f"OCP media classifier: ONNX trained backend ({onnx_model})")
    return clf


def _build_embedding_router(config: dict) -> AbstractMediaClassifier:
    """``media_classifier_embedding_router`` — the learned guided-categorical
    -embeddings router loaded from a bundle dir (``router_meta.json`` +
    per-axis sub-dirs).

    By default it is wired as a *hybrid*: the keyword backend stays the
    high-precision first pass (gate + adult lexicon) and the router fills the
    keyword-less cases, abstaining to GENERIC when unsure (never regressing
    adult-leak / false-hijack).  Set
    ``media_classifier_embedding_router_hybrid=False`` for the router alone.
    ``media_classifier_entity_library`` ({label: [titles]}) injects the user's
    own media library at runtime (no retraining) so bare titles route.
    Numpy + onnxruntime only (the ``onnx`` extra).
    """
    router_bundle = config["media_classifier_embedding_router"]
    hybrid = config.get("media_classifier_embedding_router_hybrid", True)
    if hybrid:
        from ovos_media_classifier.embedding import HybridMediaClassifier
        # Layer B — online metadatarr last-resort. Opt-in (latency), OFF by
        # default; lazy-imported only when enabled so the runtime stays lean.
        # Falls through to offline-only on import failure.
        online = None
        if config.get("media_classifier_online_metadatarr", False):
            try:
                from ovos_media_classifier.metadatarr_backend import (
                    MetadatarrMediaClassifier,
                )
                online = MetadatarrMediaClassifier(
                    timeout_s=config.get(
                        "media_classifier_online_timeout", 4.0),
                    min_confidence=config.get(
                        "media_classifier_online_min_confidence", 0.5),
                )
                LOG.info("OCP media classifier: online metadatarr layer enabled")
            except Exception as e:
                LOG.warning(
                    f"online metadatarr layer requested but unavailable "
                    f"({e}); install the 'online' extra. Offline-only.")
        clf = HybridMediaClassifier.from_path(router_bundle, online=online)
    else:
        from ovos_media_classifier.embedding import EmbeddingMediaClassifier
        clf = EmbeddingMediaClassifier.from_path(router_bundle)
    # Layer A — default OFFLINE gazetteer of common real titles, injected so
    # bare titles route without a network call or user setup.  On by default;
    # ``media_classifier_gazetteer=False`` disables it,
    # ``media_classifier_gazetteer_size`` caps titles per type.
    if config.get("media_classifier_gazetteer", True):
        size = config.get("media_classifier_gazetteer_size")
        n = clf.register_default_gazetteer(top_n=size)
        LOG.info(f"OCP media classifier: offline gazetteer ({n} titles)")
    library = config.get("media_classifier_entity_library")
    if library:
        clf.register_user_library(library)
    LOG.info(f"OCP media classifier: embedding router "
             f"({'hybrid' if hybrid else 'standalone'}, {router_bundle})")
    return clf


def _build_ner(config: dict) -> AbstractMediaClassifier:
    """``media_classifier_entities`` / ``media_classifier_wordlists`` /
    ``media_classifier_ner_csv`` — the entity-matching (Aho-Corasick) backend.

    ``media_classifier_entities`` is an ``EntitiesContainer.from_config``
    dict; it accepts the source-agnostic ``entity_lists`` list (file paths /
    HF dicts / inline ``{label: [values]}`` dicts / media-server dicts) as
    well as the legacy structured keys (csv/wordlists/<server>). See
    docs/entity-lists.md.
    """
    from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
    from ovos_media_classifier.entities import EntitiesContainer

    entities_cfg = config.get("media_classifier_entities")
    wordlists_cfg = config.get("media_classifier_wordlists")
    if entities_cfg is not None:
        container = EntitiesContainer.from_config(entities_cfg)
        clf = AhocorasickMediaClassifier.from_container(container)
    elif wordlists_cfg is not None:
        clf = AhocorasickMediaClassifier.from_wordlists(wordlists_cfg)
    else:
        clf = AhocorasickMediaClassifier.from_csv(
            config["media_classifier_ner_csv"])
    LOG.info("OCP media classifier: NER (Aho-Corasick entity matching)")
    return clf


def _plugin_requested(config: dict) -> bool:
    return bool(config.get("media_classifier_plugin"))


def _onnx_requested(config: dict) -> bool:
    return bool(config.get("media_classifier_onnx_model"))


def _router_requested(config: dict) -> bool:
    return bool(config.get("media_classifier_embedding_router"))


def _ner_requested(config: dict) -> bool:
    return any(config.get(k) is not None
               for k in ("media_classifier_entities",
                         "media_classifier_wordlists",
                         "media_classifier_ner_csv"))


#: Selection order: external plugin > onnx > embedding router > ner > keyword.
_BACKEND_SPECS = (
    ("external plugin", _plugin_requested, _build_external_plugin, None),
    ("ONNX trained backend", _onnx_requested, _build_onnx, "onnx"),
    ("embedding router", _router_requested, _build_embedding_router, "onnx"),
    ("NER classifier", _ner_requested, _build_ner, "ner"),
)


def load_media_classifier(
    config: Optional[dict] = None,
    voc_match_func: Optional[Callable] = None,
) -> AbstractMediaClassifier:
    """Return a media classifier for the given config.

    Selection (first configured backend wins, each falling through to the
    next on any failure — see ``_BACKEND_SPECS``):

    1. ``config["media_classifier_plugin"]`` → an external classifier
       registered under the ``opm.media.classifier`` entry-point group.
    2. ``config["media_classifier_onnx_model"]`` → the opt-in ONNX trained
       backend (requires the ``onnx`` extra).
    3. ``config["media_classifier_embedding_router"]`` → the learned
       embedding router, hybrid with the keyword backend by default
       (requires the ``onnx`` extra).
    4. ``media_classifier_entities`` / ``_wordlists`` / ``_ner_csv`` → the
       entity-matching backend (requires the ``ner`` extra).
    5. Otherwise → the built-in keyword (``.voc``) classifier. When
       *voc_match_func* is given (e.g. a pipeline's ``voc_match``) it is
       used; otherwise the bundled locale files are read directly.

    Args:
        config: OCP config block (see the ``_build_*`` builders for the full
            key reference).
        voc_match_func: optional ``(phrase, vocab_name, *, lang) -> bool`` matcher.
    """
    config = config or {}

    for name, requested, builder, extra in _BACKEND_SPECS:
        if not requested(config):
            continue
        try:
            return builder(config)
        except ImportError as e:
            hint = (f" Install it with: pip install "
                    f"ovos-media-classifier[{extra}]." if extra else "")
            LOG.warning(f"{name} requested but unavailable ({e}).{hint} "
                        "Falling back to the keyword classifier.")
        except Exception as e:
            LOG.warning(f"Failed to load {name}: {e}. "
                        "Falling back to the keyword classifier.")

    if voc_match_func is not None:
        LOG.debug("OCP media classifier: keyword (voc_match)")
        return KeywordMediaClassifier(voc_match_func)

    LOG.debug("OCP media classifier: keyword (bundled locale)")
    return KeywordMediaClassifier()
