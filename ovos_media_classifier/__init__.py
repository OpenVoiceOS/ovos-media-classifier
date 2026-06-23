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

The classifier returns the canonical ``mediavocab.MediaType`` taxonomy; the
fine-grained internal label space (``OCPPlayIntent``) maps to type + genre tags.
"""
from typing import Callable, Optional

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.content_filter import ContentFilter
from ovos_media_classifier.intents import (
    MediaType,
    OCPDomain,
    OCPPlayIntent,
    OCPControlIntent,
    OCPEntityLabel,
    PLAY_INTENT_TO_MEDIA_TYPE,
    PLAY_INTENT_TO_GENRES,
    MEDIA_TYPE_TO_PLAY_INTENT,
    LABEL_TO_MEDIA_TYPE,
    LABEL_TO_GENRES,
    NER_LABEL_TO_PLAY_INTENT,
    genres_for_label,
)
from ovos_media_classifier.keyword import KeywordMediaClassifier
from ovos_media_classifier.plugins import (
    find_media_classifier_plugins,
    load_media_classifier_plugin,
)
from ovos_media_classifier.version import __version__

__all__ = [
    "AbstractMediaClassifier",
    "MediaType",
    "ContentFilter",
    "OCPDomain",
    "OCPPlayIntent",
    "OCPControlIntent",
    "OCPEntityLabel",
    "PLAY_INTENT_TO_MEDIA_TYPE",
    "PLAY_INTENT_TO_GENRES",
    "MEDIA_TYPE_TO_PLAY_INTENT",
    "LABEL_TO_MEDIA_TYPE",
    "LABEL_TO_GENRES",
    "NER_LABEL_TO_PLAY_INTENT",
    "genres_for_label",
    "KeywordMediaClassifier",
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
    2. Otherwise → the built-in keyword (``.voc``) classifier. When
       *voc_match_func* is given (e.g. a pipeline's ``voc_match``) it is used;
       otherwise the bundled locale files are read directly.

    Args:
        config: OCP config block (``media_classifier_plugin`` selects a plugin).
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

    if voc_match_func is not None:
        LOG.debug("OCP media classifier: keyword (voc_match)")
        return KeywordMediaClassifier(voc_match_func)

    LOG.debug("OCP media classifier: keyword (bundled locale)")
    return KeywordMediaClassifier()
