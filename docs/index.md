# ovos-media-classifier documentation

`ovos-media-classifier` provides media-type **command/intent** classification for
OCP (OVOS Common Playback). It maps a spoken command to an `OCPDomain` (play /
control / not-OCP), and for play requests to a `mediavocab.MediaType` plus genre
tags. A detect-to-block content filter sits on top so OVOS can refuse sensitive
requests by default.

This release ships a **single** classifier: the bundled-`.voc` keyword classifier
(zero ML dependencies — the minimum required for OCP to be functional). Richer
strategies (trained ONNX models, NER from media servers, …) are **not** in this
release; they arrive as independent plugins through the `opm.media.classifier`
entry-point group — see [external-plugins.md](external-plugins.md).

This classifies *voice commands*. It is distinct from `mediavocab.text.classify`,
which classifies *catalog content* — see [taxonomy.md](taxonomy.md#query-vs-content-classification).

## Contents

| Page | What it covers |
|---|---|
| [taxonomy.md](taxonomy.md) | mediavocab `MediaType` enforcement, the internal `OCPPlayIntent`→type/genre mapping, and the query-vs-content distinction |
| [backends.md](backends.md) | The bundled keyword classifier and how to add a classifier plugin |
| [content-filtering.md](content-filtering.md) | `ContentFilter`: detect-to-block content moderation / parental control |
| [external-plugins.md](external-plugins.md) | Registering 3rd-party / future classifiers via the `opm.media.classifier` entry-point group |
| [stable-api.md](stable-api.md) | The `AbstractMediaClassifier` contract and return types |

For the reproducible accuracy/latency harness, see
[benchmarks](../benchmarks/README.md). It evaluates whichever classifiers are
installed; only the keyword classifier ships in this release.

For model training (not shipped in the wheel; it feeds future classifier plugins,
not this package), see `training/README.md`.

## Public API at a glance

```python
from ovos_media_classifier import (
    load_media_classifier,          # factory: picks a backend from config
    load_media_classifier_plugin,   # load an external classifier by name
    AbstractMediaClassifier,        # the plugin / backend contract
    ContentFilter,                  # detect-to-block moderation
    MediaType,                      # re-exported mediavocab taxonomy
    OCPDomain, OCPPlayIntent, OCPEntityLabel,
)
```
