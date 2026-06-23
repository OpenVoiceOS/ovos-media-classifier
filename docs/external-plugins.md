# External plugins

Third-party media classifiers integrate through OPM (the OVOS Plugin Manager).
A classifier is any subclass of
[`AbstractMediaClassifier`](stable-api.md) registered under the
`opm.media.classifier` entry-point group. `AbstractMediaClassifier` is the contract
the host relies on.

## Registering a classifier

```python
# my_pkg/__init__.py
from ovos_media_classifier import AbstractMediaClassifier, MediaType

class MyMediaClassifier(AbstractMediaClassifier):
    def __init__(self, config=None):
        self.config = config or {}

    def classify(self, query, lang, valid_labels=None):
        ...
        return MediaType.MUSIC, 0.9
```

```toml
# my_pkg's pyproject.toml
[project.entry-points."opm.media.classifier"]
my-classifier = "my_pkg:MyMediaClassifier"
```

The entry-point name (`my-classifier`) is how the host selects the plugin. A
companion group `opm.media.classifier.config` mirrors other OPM types for shipping
default per-plugin config.

## Loading a classifier

Via the factory, by setting `media_classifier_plugin` in config:

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier(config={"media_classifier_plugin": "my-classifier"})
```

The whole config dict is forwarded to the plugin constructor as `config=…`. If the
plugin fails to load, the factory logs a warning and falls through to the built-in
backends — an external plugin never hard-fails the pipeline.

Or load one directly:

```python
from ovos_media_classifier import (
    load_media_classifier_plugin,
    find_media_classifier_plugins,
)

find_media_classifier_plugins()                 # {name: class} for installed plugins
clf = load_media_classifier_plugin("my-classifier", config={})
```

`load_media_classifier_plugin` raises `ValueError` if no plugin with that name is
installed (the message lists the available names). It tolerates classifiers whose
`__init__` takes no `config` kwarg. `find_media_classifier_plugins()` never raises —
it returns `{}` when the plugin manager is unavailable.

## Implementing the contract

A plugin must implement `classify()`. It may override `classify_domain()`,
`is_ocp_query()`, and `classify_genres()` when it has cheaper or richer signal —
e.g. a dedicated domain head, control-intent support, or genre detection so the
[content filter](content-filtering.md) can block on it. See
[stable-api.md](stable-api.md) for the full contract and the default
implementations.
