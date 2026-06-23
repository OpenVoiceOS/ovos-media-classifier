# ovos-media-classifier documentation

`ovos-media-classifier` provides media-type **command/intent** classification for
OCP (OVOS Common Playback). It maps a spoken command onto a small set of
**orthogonal axes** — domain (play / control / not-OCP), modality
(`playback_type`: audio / video / paged / interactive), structure
(single / episodic / continuous / collection), and the concrete
`mediavocab.MediaType` leaf — plus orthogonal genre tags. A detect-to-block
content filter sits on top so OVOS can refuse sensitive requests by default.

The design rationale for this multi-axis model — why orthogonal axes rather than a
strict tree — is in [classification-model.md](classification-model.md).

This release ships a **single** classifier: the bundled-`.voc` keyword classifier
(zero ML dependencies — the minimum required for OCP to be functional). It predicts
the leaf `MediaType` and derives the coarse axes from it; richer strategies
(trained ONNX models with a head per axis, NER from media servers, …) are **not**
in this release — they arrive as independent plugins through the
`opm.media.classifier` entry-point group, see
[external-plugins.md](external-plugins.md).

This classifies *voice commands*. It is distinct from `mediavocab.text.classify`,
which classifies *catalog content* — see [taxonomy.md](taxonomy.md#query-vs-content-classification).

## Contents

| Page | What it covers |
|---|---|
| [classification-model.md](classification-model.md) | **The multi-axis model**: the four axes + tags, why orthogonal axes beat a strict tree, and the `MediaType`→(playback_type, structure) defaults |
| [taxonomy.md](taxonomy.md) | mediavocab `MediaType` enforcement, the internal `OCPPlayIntent`→type/genre mapping, and the query-vs-content distinction |
| [backends.md](backends.md) | The bundled keyword classifier and how to add a classifier plugin |
| [content-filtering.md](content-filtering.md) | `ContentFilter`: detect-to-block content moderation / parental control |
| [external-plugins.md](external-plugins.md) | Registering 3rd-party / future classifiers via the `opm.media.classifier` entry-point group |
| [stable-api.md](stable-api.md) | The `AbstractMediaClassifier` contract, the multi-axis methods, and return types |

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
    MediaType,                      # re-exported mediavocab taxonomy (the leaf axis)
    Structure,                      # the structure axis (single/episodic/continuous/collection)
    MediaClassification,            # the full multi-axis result (classify_full)
    OCPDomain, OCPPlayIntent, OCPEntityLabel,
)

clf = load_media_classifier()
clf.classify_full("play a podcast", "en-us").as_dict()
# {'media_type': 'podcast', 'playback_type': 'audio', 'structure': 'episodic',
#  'domain': 'ocp_play', 'genres': [], 'confidence': 0.6}
```
