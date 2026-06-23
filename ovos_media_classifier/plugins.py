"""Discovery of external (3rd-party) media classifiers via OPM.

A classifier is any subclass of
:class:`~ovos_media_classifier.base.AbstractMediaClassifier` registered under
the ``opm.media.classifier`` entry-point group::

    # in a 3rd-party package's pyproject.toml
    [project.entry-points."opm.media.classifier"]
    my-classifier = "my_pkg:MyMediaClassifier"

It is then loadable by name through the factory
(``load_media_classifier(config={"media_classifier_plugin": "my-classifier"})``)
or directly via :func:`load_media_classifier_plugin`.

This module intentionally talks to ``ovos_plugin_manager`` through the raw
entry-point group string so discovery works even before a
``PluginTypes.MEDIA_CLASSIFIER`` constant lands in a released plugin-manager.
"""
from typing import Dict, Optional, Type

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier

#: entry-point group external classifiers register under
MEDIA_CLASSIFIER_GROUP = "opm.media.classifier"
#: companion group for default per-plugin config (mirrors other OPM types)
MEDIA_CLASSIFIER_CONFIG_GROUP = "opm.media.classifier.config"


def find_media_classifier_plugins() -> Dict[str, Type[AbstractMediaClassifier]]:
    """Return a mapping of ``name -> classifier class`` for installed plugins.

    Never raises — returns ``{}`` if the plugin manager is unavailable.
    """
    try:
        from ovos_plugin_manager.utils import find_plugins
    except Exception as e:  # pragma: no cover - PM always present in practice
        LOG.debug(f"ovos-plugin-manager unavailable, no external classifiers: {e}")
        return {}
    try:
        return dict(find_plugins(MEDIA_CLASSIFIER_GROUP) or {})
    except Exception as e:
        LOG.warning(f"failed to enumerate {MEDIA_CLASSIFIER_GROUP} plugins: {e}")
        return {}


def load_media_classifier_plugin(
    name: str,
    config: Optional[dict] = None,
) -> AbstractMediaClassifier:
    """Instantiate the external classifier registered as *name*.

    Args:
        name: entry-point name under ``opm.media.classifier``.
        config: passed to the plugin constructor as ``config=...``.

    Raises:
        ValueError: when no plugin with that name is installed.
    """
    plugins = find_media_classifier_plugins()
    clazz = plugins.get(name)
    if clazz is None:
        available = ", ".join(sorted(plugins)) or "<none installed>"
        raise ValueError(
            f"no media classifier plugin named {name!r}; available: {available}"
        )
    try:
        return clazz(config=config or {})
    except TypeError:
        # tolerate classifiers whose __init__ takes no config kwarg
        return clazz()
