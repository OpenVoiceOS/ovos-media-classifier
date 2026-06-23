"""ovos-media-classifier — pluggable media-type classification for OCP.

Public API::

    from ovos_media_classifier import load_media_classifier

    # Keyword-based (default, no ML deps needed):
    clf = load_media_classifier(config={}, voc_match_func=self.voc_match)

    # AhocorasickNER from media servers + HuggingFace (runtime-aware):
    clf = load_media_classifier(
        config={"media_classifier_entities": {
            "radarr":   {"url": "http://localhost:7878", "api_key": "…"},
            "sonarr":   {"url": "http://localhost:8989", "api_key": "…"},
            "lidarr":   {"url": "http://localhost:8686", "api_key": "…"},
            "jellyfin": {"url": "http://localhost:8096", "api_key": "…"},
            "huggingface": [{"dataset": "TigreGotico/ocp-entities"}],
        }},
    )
    # Or from a static word list:
    clf = load_media_classifier(
        config={"media_classifier_wordlists": {"music": ["jazz", "blues"]}},
    )

    # scikit-learn (TF-IDF + LogisticRegression):
    clf = load_media_classifier(
        config={"media_classifier_sklearn_model": "/path/to/model.joblib"},
    )

    # Padatious (pattern + ML intent matching):
    clf = load_media_classifier(
        config={"media_classifier_padatious_dir": "/path/to/locale"},
    )

    # Model2Vec (hierarchical neural model):
    clf = load_media_classifier(
        config={"media_classifier_model": "/path/to/m2v_model"},
    )

    media_type, conf = clf.classify("play some jazz", "en-us")
    is_ocp, conf    = clf.is_ocp_query("play something", "en-us")
    domain, conf    = clf.classify_domain("pause the music", "en-us")

Backends (selection priority when config keys are set)
------------------------------------------------------
1. Model2Vec          — ``media_classifier_model``            (ML, best accuracy)
2. GuidedEmbeddings   — ``media_classifier_guided_model``     (ONNX categorical, no torch at runtime)
3. scikit-learn       — ``media_classifier_sklearn_model``    (ML, fast)
4. Padatious          — ``media_classifier_padatious_dir``    (pattern + ML)
5. AhocorasickNER     — ``media_classifier_entities``         (media servers + HF datasets, runtime-aware)
                      — ``media_classifier_wordlists`` or
                         ``media_classifier_ner_csv``         (static exact match)
6. Keyword            — ``voc_match_func`` supplied           (pipeline voc files)
7. Keyword            — fallback, uses bundled locale files

All backends are optional — missing deps cause a warning and fallback to the
next backend in the priority list.

``media_classifier_entities`` config example::

    "media_classifier_entities": {
        "jellyfin":        {"url": "http://localhost:8096", "api_key": "…"},
        "radarr":          {"url": "http://localhost:7878", "api_key": "…"},
        "sonarr":          {"url": "http://localhost:8989", "api_key": "…"},
        "lidarr":          {"url": "http://localhost:8686", "api_key": "…"},
        "whisparr":        {"url": "http://localhost:6969", "api_key": "…"},
        "stash":           {"url": "http://localhost:9999", "api_key": "…"},
        "music_assistant": {"url": "http://localhost:8095"},
        "huggingface":     [{"dataset": "TigreGotico/ocp-entities"}],
        "csv":             ["/path/to/extra_entities.csv"],
        "wordlists":       {"artist_name": ["Radiohead"]}
    }
"""
from typing import Callable, Dict, List, Optional

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.content_filter import ContentFilter
from ovos_media_classifier.entities import EntitiesContainer
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
    # base / taxonomy
    "AbstractMediaClassifier",
    "MediaType",
    # content filtering
    "ContentFilter",
    # entity container
    "EntitiesContainer",
    # enums
    "OCPDomain",
    "OCPPlayIntent",
    "OCPControlIntent",
    "OCPEntityLabel",
    # mappings
    "PLAY_INTENT_TO_MEDIA_TYPE",
    "PLAY_INTENT_TO_GENRES",
    "MEDIA_TYPE_TO_PLAY_INTENT",
    "LABEL_TO_MEDIA_TYPE",
    "LABEL_TO_GENRES",
    "NER_LABEL_TO_PLAY_INTENT",
    "genres_for_label",
    # classifiers
    "KeywordMediaClassifier",
    "GuidedEmbeddingsMediaClassifier",
    # external plugin discovery
    "find_media_classifier_plugins",
    "load_media_classifier_plugin",
    # factory
    "load_media_classifier",
    "__version__",
]


def load_media_classifier(
    config: Optional[dict] = None,
    voc_match_func: Optional[Callable] = None,
) -> AbstractMediaClassifier:
    """Return the best available media classifier for the given config.

    Selection logic (first matching key wins):

    1. ``media_classifier_model``         → Model2VecMediaClassifier
    2. ``media_classifier_guided_model``  → GuidedEmbeddingsMediaClassifier
    3. ``media_classifier_sklearn_model`` → SklearnMediaClassifier
    4. ``media_classifier_padatious_dir`` → PadatiousMediaClassifier
    5. ``media_classifier_wordlists`` or
       ``media_classifier_ner_csv``       → AhocorasickMediaClassifier
    6. *voc_match_func* supplied          → KeywordMediaClassifier (pipeline mode)
    7. Fallback                           → KeywordMediaClassifier (bundled locale)

    On any import or load error the function logs a warning and tries the
    next backend in the list.

    Args:
        config: Pipeline config dict (the "OCP" block from mycroft.conf).
        voc_match_func: ``voc_match`` method bound to the pipeline plugin
            instance, used by the keyword classifier.
    """
    config = config or {}

    # 0. External (3rd-party) classifier registered under opm.media.classifier
    plugin_name = config.get("media_classifier_plugin")
    if plugin_name:
        try:
            clf = load_media_classifier_plugin(plugin_name, config)
            LOG.info(f"OCP media classifier: external plugin ({plugin_name})")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load external media classifier {plugin_name!r}: {e}. "
                "Trying built-in backends."
            )

    # 1. Model2Vec
    model_path = config.get("media_classifier_model")
    if model_path:
        try:
            from ovos_media_classifier.m2v import Model2VecMediaClassifier
            domain_thresh = config.get("media_classifier_domain_threshold", 0.5)
            intent_thresh = config.get("media_classifier_intent_threshold", 0.3)
            clf = Model2VecMediaClassifier.from_path(
                model_path,
                domain_threshold=domain_thresh,
                intent_threshold=intent_thresh,
            )
            LOG.info(f"OCP media classifier: Model2Vec ({model_path})")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load Model2Vec classifier from {model_path!r}: {e}. "
                "Trying next backend."
            )

    # 2. GuidedEmbeddings (ONNX categorical embeddings on pre-extracted features)
    guided_path = config.get("media_classifier_guided_model")
    if guided_path:
        try:
            from ovos_media_classifier.guided import GuidedEmbeddingsMediaClassifier
            container = None
            if config.get("media_classifier_entities"):
                from ovos_media_classifier.entities import EntitiesContainer
                container = EntitiesContainer.from_config(
                    config["media_classifier_entities"]
                )
            clf = GuidedEmbeddingsMediaClassifier.from_path(
                guided_path,
                entities_container=container,
                domain_threshold=config.get(
                    "media_classifier_domain_threshold", 0.5
                ),
                play_threshold=config.get(
                    "media_classifier_play_threshold", 0.3
                ),
            )
            LOG.info(f"OCP media classifier: GuidedEmbeddings ({guided_path})")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load GuidedEmbeddings classifier from "
                f"{guided_path!r}: {e}. Trying next backend."
            )

    # 3. scikit-learn
    sklearn_path = config.get("media_classifier_sklearn_model")
    if sklearn_path:
        try:
            from ovos_media_classifier.sklearn import SklearnMediaClassifier
            play_thresh = config.get("media_classifier_play_threshold", 0.3)
            domain_thresh = config.get("media_classifier_domain_threshold", 0.5)
            clf = SklearnMediaClassifier.from_path(
                sklearn_path,
                play_threshold=play_thresh,
                domain_threshold=domain_thresh,
            )
            LOG.info(f"OCP media classifier: sklearn ({sklearn_path})")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load sklearn classifier from {sklearn_path!r}: {e}. "
                "Trying next backend."
            )

    # 4. Padatious
    padatious_dir = config.get("media_classifier_padatious_dir")
    if padatious_dir:
        try:
            from ovos_media_classifier.padatious import PadatiousMediaClassifier
            lang = config.get("lang", "en-us")
            domain_dir = config.get("media_classifier_padatious_domain_dir")
            play_cache = config.get("media_classifier_padatious_cache")
            play_thresh = config.get("media_classifier_play_threshold", 0.5)
            domain_thresh = config.get("media_classifier_domain_threshold", 0.5)
            clf = PadatiousMediaClassifier.from_locale_dir(
                padatious_dir,
                lang=lang,
                domain_dir=domain_dir,
                play_cache_dir=play_cache,
                play_threshold=play_thresh,
                domain_threshold=domain_thresh,
            )
            LOG.info(f"OCP media classifier: padatious ({padatious_dir})")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load padatious classifier from {padatious_dir!r}: {e}. "
                "Trying next backend."
            )

    # 5a. AhocorasickNER backed by EntitiesContainer (media servers / HuggingFace)
    entities_cfg: Optional[dict] = config.get("media_classifier_entities")
    if entities_cfg:
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            from ovos_media_classifier.entities import EntitiesContainer
            container = EntitiesContainer.from_config(entities_cfg)
            clf = AhocorasickMediaClassifier.from_container(container)
            LOG.info("OCP media classifier: AhocorasickNER + EntitiesContainer (%d entities)", len(container))
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to build EntitiesContainer classifier: {e}. "
                "Trying next backend."
            )

    # 5b. AhocorasickNER (static wordlists / CSV)
    wordlists: Optional[Dict[str, List[str]]] = config.get("media_classifier_wordlists")
    ner_csv: Optional[str] = config.get("media_classifier_ner_csv")
    if wordlists or ner_csv:
        try:
            from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
            if wordlists:
                clf = AhocorasickMediaClassifier.from_wordlists(wordlists)
            else:
                clf = AhocorasickMediaClassifier.from_csv(ner_csv)
            LOG.info("OCP media classifier: AhocorasickNER")
            return clf
        except Exception as e:
            LOG.warning(
                f"Failed to load AhocorasickNER classifier: {e}. "
                "Trying next backend."
            )

    # 6. Pipeline keyword (external voc_match_func)
    if voc_match_func is not None:
        LOG.debug("OCP media classifier: keyword (voc_match)")
        return KeywordMediaClassifier(voc_match_func)

    # 7. Standalone keyword (bundled locale files)
    LOG.debug("OCP media classifier: keyword (bundled locale)")
    return KeywordMediaClassifier()
