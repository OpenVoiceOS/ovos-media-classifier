# ovos-media-classifier documentation

`ovos-media-classifier` provides media-type **command/intent** classification for
OCP (OVOS Common Play). It maps a spoken command onto a small set of
**orthogonal axes** — domain (play / control / not-OCP), modality
(`playback_type`: audio / video / paged / interactive), structure
(single / episodic / continuous / collection), and the concrete
`mediavocab.MediaType` leaf — plus orthogonal genre tags. A detect-to-block
content filter sits on top so OVOS can refuse sensitive requests by default.

This classifies *voice commands*. It is distinct from `mediavocab.text.classify`,
which classifies *catalog content* — see
[taxonomy.md](taxonomy.md#query-vs-content-classification).

## Quickstart

```bash
pip install ovos-media-classifier
```

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()                  # bundled .voc keyword classifier
clf.classify("play some music", "en-us")       # -> (<MediaType.MUSIC: 'music'>, 0.6)
clf.classify_full("play a podcast", "en-us").as_dict()
# {'media_type': 'podcast', 'playback_type': 'audio', 'structure': 'episodic',
#  'domain': 'ocp_play', 'genres': [], 'confidence': 0.6, 'control_intent': None}
```

## Start here

| You are… | Read |
|---|---|
| **New to the project** | [glossary.md](glossary.md) (every term in one place), then this page's quickstart and [stable-api.md](stable-api.md) |
| **An operator tuning backends** | [backends.md](backends.md) (selection + config keys), [entity-lists.md](entity-lists.md), [contextual-classification.md](contextual-classification.md) |
| **Blocking content** | [content-filtering.md](content-filtering.md) |
| **A plugin author** writing an external classifier | [external-plugins.md](external-plugins.md), then [stable-api.md](stable-api.md) for the contract |
| **A contributor** training a model | [classification-model.md](classification-model.md) (the multi-head model) and [benchmarks](../benchmarks/README.md) |

New to the vocabulary (OCP, OPM, domain, axis, NER, …)? Read
[glossary.md](glossary.md) first.

## Contents

| Page | What it covers |
|---|---|
| [glossary.md](glossary.md) | Every term and acronym, the 30-second mental model, and the command-vs-content distinction |
| [classification-model.md](classification-model.md) | The multi-axis model: the four axes + tags, why orthogonal axes rather than a strict tree, and the `MediaType`→(playback_type, structure) defaults |
| [taxonomy.md](taxonomy.md) | `mediavocab.MediaType` enforcement, the internal `OCPPlayIntent`→type/genre mapping, and the query-vs-content distinction |
| [backends.md](backends.md) | The keyword, NER and ONNX backends, how `load_media_classifier` selects between them, and adding an external classifier |
| [entity-lists.md](entity-lists.md) | Entity lists (`label → list of strings`): the source-agnostic store (runtime / `.csv` / `.tsv` / `.jsonl` / HuggingFace / media-server / inline) the NER backend consumes |
| [contextual-classification.md](contextual-classification.md) | How the media you actually have biases prediction, via keyword feature slots |
| [content-filtering.md](content-filtering.md) | `ContentFilter`: detect-to-block content moderation / parental control |
| [external-plugins.md](external-plugins.md) | Registering a third-party classifier via the `opm.media.classifier` entry-point group |
| [stable-api.md](stable-api.md) | The `AbstractMediaClassifier` contract, the multi-axis methods, and return types |

For the reproducible accuracy/latency harness, see
[benchmarks](../benchmarks/README.md). For runnable, commented scripts that
exercise the public API, see [examples/](../examples/).

## Public API at a glance

```python
from ovos_media_classifier import (
    load_media_classifier,          # factory: picks a backend from config
    load_media_classifier_plugin,   # load an external classifier by name
    AbstractMediaClassifier,        # the plugin / backend contract
    ContentFilter,                  # detect-to-block moderation
    MediaType,                      # re-exported mediavocab taxonomy (the leaf axis)
    Structure,                      # the structure axis (single/episodic/continuous/collection)
    MediaClassification,            # the full multi-axis result (classify_full)
    OCPDomain, OCPPlayIntent, OCPEntityLabel,
)
```

See [stable-api.md](stable-api.md) for the full method reference.
