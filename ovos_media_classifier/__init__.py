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
    "KeywordFeatureSlot",
    "KEYWORD_FEATURE_SLOTS",
    "slot_for_label",
    "slots_for_media_type",
    "find_media_classifier_plugins",
    "load_media_classifier_plugin",
    "load_media_classifier",
    "__version__",
]


def load_media_classifier(
    config: Optional[dict] = None,
    voc_match_func: Optional[Callable] = None,
) -> AbstractMediaClassifier:
    """Return a media classifier for the given config.

    Selection:

    1. ``config["media_classifier_plugin"]`` → load that external classifier
       registered under the ``opm.media.classifier`` entry-point group.
    2. ``config["media_classifier_onnx_model"]`` → load the optional, opt-in
       ONNX trained backend from that bundle directory (requires the ``onnx``
       extra: ``pip install ovos-media-classifier[onnx]``).  On any failure
       (missing extra, bad bundle) this logs a warning and falls through.
    3. Otherwise → the built-in keyword (``.voc``) classifier. When
       *voc_match_func* is given (e.g. a pipeline's ``voc_match``) it is used;
       otherwise the bundled locale files are read directly.

    Args:
        config: OCP config block (``media_classifier_plugin`` selects a plugin;
            ``media_classifier_onnx_model`` selects the ONNX bundle).
        voc_match_func: optional ``(phrase, vocab_name, *, lang) -> bool`` matcher.
    """
    config = config or {}

    plugin_name = config.get("media_classifier_plugin")
    if plugin_name:
        try:
            clf = load_media_classifier_plugin(plugin_name, config)
            LOG.info(f"OCP media classifier: external plugin ({plugin_name})")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load media classifier plugin {plugin_name!r}: {e}. "
                "Falling back to the keyword classifier."
            )

    # Optional ONNX trained backend — requires the ``[onnx]`` extra. Lazy import:
    # onnxruntime/numpy must NOT be touched on the default import path.
    onnx_model = config.get("media_classifier_onnx_model")
    if onnx_model:
        try:
            from ovos_media_classifier.onnx import OnnxMediaClassifier
            clf = OnnxMediaClassifier.from_path(onnx_model)
            LOG.info(f"OCP media classifier: ONNX trained backend ({onnx_model})")
            return clf
        except ImportError as e:
            LOG.warning(
                f"ONNX media classifier requested but onnxruntime/numpy are not "
                f"installed ({e}). Install the 'onnx' extra. "
                "Falling back to the keyword classifier."
            )
        except Exception as e:
            LOG.warning(
                f"Failed to load ONNX media classifier from {onnx_model!r}: {e}. "
                "Falling back to the keyword classifier."
            )

    # Optional embedding-router backend — the learned guided-categorical
    # -embeddings router.  ``media_classifier_embedding_router`` is a bundle dir
    # (``router_meta.json`` + per-axis sub-dirs).  By default it is wired as a
    # *hybrid*: the keyword backend stays the high-precision first pass (gate +
    # adult lexicon) and the router fills the keyword-less cases, abstaining to
    # GENERIC when unsure (never regressing adult-leak / false-hijack).  Set
    # ``media_classifier_embedding_router_hybrid=False`` for the router alone.
    # ``media_classifier_entity_library`` ({label: [titles]}) injects the user's
    # own media library at runtime (no retraining) so bare titles route. Numpy +
    # onnxruntime only (the ``[onnx]`` extra); falls through to keyword on any
    # failure so the zero-ML default is always preserved.
    router_bundle = config.get("media_classifier_embedding_router")
    if router_bundle:
        try:
            hybrid = config.get("media_classifier_embedding_router_hybrid", True)
            if hybrid:
                from ovos_media_classifier.embedding import HybridMediaClassifier
                clf = HybridMediaClassifier.from_path(router_bundle)
            else:
                from ovos_media_classifier.embedding import EmbeddingMediaClassifier
                clf = EmbeddingMediaClassifier.from_path(router_bundle)
            library = config.get("media_classifier_entity_library")
            if library:
                clf.register_user_library(library)
            LOG.info(f"OCP media classifier: embedding router "
                     f"({'hybrid' if hybrid else 'standalone'}, {router_bundle})")
            return clf
        except ImportError as e:
            LOG.warning(
                f"Embedding router requested but onnxruntime/numpy are not "
                f"installed ({e}). Install the 'onnx' extra. "
                "Falling back to the keyword classifier."
            )
        except Exception as e:
            LOG.warning(
                f"Failed to load embedding router from {router_bundle!r}: {e}. "
                "Falling back to the keyword classifier."
            )

    # Optional NER (entity-based) backend — requires the ``[ner]`` extra (and
    # ``[media_servers]`` / ``[huggingface]`` for the live loaders). On
    # ImportError / failure we fall through to the lean keyword classifier so the
    # zero-ML-dependency default is always preserved.
    #
    # ``media_classifier_entities`` is an EntitiesContainer.from_config dict; it
    # accepts the source-agnostic ``entity_lists`` list (file paths / HF dicts /
    # inline ``{label: [values]}`` dicts / media-server dicts) as well as the
    # legacy structured keys (csv/wordlists/huggingface/<server>). See
    # docs/entity-lists.md.
    entities_cfg = config.get("media_classifier_entities")
    wordlists_cfg = config.get("media_classifier_wordlists")
    ner_csv = config.get("media_classifier_ner_csv")
    if entities_cfg is not None or wordlists_cfg is not None or ner_csv is not None:
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            from ovos_media_classifier.entities import EntitiesContainer

            if entities_cfg is not None:
                container = EntitiesContainer.from_config(entities_cfg)
                clf = AhocorasickMediaClassifier.from_container(container)
            elif wordlists_cfg is not None:
                clf = AhocorasickMediaClassifier.from_wordlists(wordlists_cfg)
            else:
                clf = AhocorasickMediaClassifier.from_csv(ner_csv)
            LOG.info("OCP media classifier: NER (Aho-Corasick entity matching)")
            return clf
        except ImportError as e:
            LOG.warning(
                f"NER classifier backend unavailable: {e}. "
                "Install it with: pip install ovos-media-classifier[ner]. "
                "Falling back to the keyword classifier."
            )
        except Exception as e:
            LOG.warning(
                f"Failed to build NER media classifier: {e}. "
                "Falling back to the keyword classifier."
            )

    if voc_match_func is not None:
        LOG.debug("OCP media classifier: keyword (voc_match)")
        return KeywordMediaClassifier(voc_match_func)

    LOG.debug("OCP media classifier: keyword (bundled locale)")
    return KeywordMediaClassifier()
