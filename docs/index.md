# ovos-media-classifier documentation

`ovos-media-classifier` provides media-type **command/intent** classification for
OCP (OVOS Common Play). It is **multi-task**: it maps a spoken command onto a set
of **orthogonal axes** at once — domain (play / control / not-OCP), modality
(`playback_type`: audio / video / paged / interactive), structure
(single / episodic / continuous / collection), explicitness (clean / adult), and
the concrete `mediavocab.MediaType` leaf — plus the mediavocab descriptive axes:
**`content_form`** (trailer / behind_scenes / …), **`programme_format`**
(documentary / news / …), **`accessibility`** (subtitles / audio_description),
**`variant`** (directors / remastered / …), the **`content_genres`** (⊆
`KNOWN_GENRES`) and the content-form genre tags. A detect-to-block content filter
sits on top so OVOS can refuse sensitive requests by default.

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
| **A contributor** training a model | [model.md](model.md) (the multi-task model + the ladder + limitations), [dataset.md](dataset.md) (the data + its generator), and [benchmarks](../benchmarks/README.md) |
| **An ML engineer** extending it | [extending.md](extending.md) (add a backend, retrain, add a new axis/head), then [model.md](model.md) |

New to the vocabulary (OCP, OPM, domain, axis, NER, …)? Read
[glossary.md](glossary.md) first.

## Contents

| Page | What it covers |
|---|---|
| [glossary.md](glossary.md) | Every term and acronym, the 30-second mental model, and the command-vs-content distinction |
| [classification-model.md](classification-model.md) | The multi-axis model: the core axes + the mediavocab descriptive axes, why orthogonal axes rather than a strict tree, and the `MediaType`→(playback_type, structure) defaults |
| [model.md](model.md) | The trained model for an ML engineer: the feature representation, the multi-task per-axis heads + soft-gating, the rules→context→context+NER ladder, the self-describing bundle/retrain contract, the benchmark table, and the honest limitations |
| [extending.md](extending.md) | Step-by-step: implement a new `AbstractMediaClassifier` backend + register it under `opm.media.classifier`, retrain the ONNX bundle with `train_sklearn.py`, and add a new axis/head end-to-end |
| [taxonomy.md](taxonomy.md) | `mediavocab.MediaType` enforcement, the raw label→type/genre mapping, and the query-vs-content distinction |
| [backends.md](backends.md) | The keyword, NER, ONNX and embedding-router backends, how `load_media_classifier` selects between them, and adding an external classifier |
| [embedding-router.md](embedding-router.md) | The learned guided-categorical-embeddings router: two-stream `[categorical | entity]` features, the routing-aware cost-matrix/abstain/calibration objective, the keyword+router hybrid gating, runtime entity injection (no retraining), and the routing-eval promote/hold verdict |
| [entity-lists.md](entity-lists.md) | Entity lists (`label → list of strings`): the source-agnostic store (runtime / `.csv` / `.tsv` / `.jsonl` / HuggingFace / media-server / inline) the NER backend consumes |
| [dataset.md](dataset.md) | The canonical `ocp-media-intents` dataset and its on-demand generator: every column, the rebuild command, the `.intent`/`.voc` templates, confusables, and the content-filter slice |
| [data-sources.md](data-sources.md) | Every HF dataset + local scraper feeding the entity pools, the slot label each feeds, licenses, and how the training set is assembled |
| [plots/dataset/](plots/dataset/) | Dataset characterization plots — rows per media type, the slot×media-type heatmap, axis distributions, entity-pool sizes (regenerate with `python -m training.dataset_plots`) |
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
    OCPDomain, OCPControlIntent, OCPEntityLabel,
)
```

See [stable-api.md](stable-api.md) for the full method reference.
