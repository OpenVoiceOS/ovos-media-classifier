# ovos-media-classifier

Media-type classification for **OVOS Common Playback (OCP)**. Given a spoken
request ("play some jazz", "watch the news"), it answers — fast, offline —
*what kind of media* is wanted, so the OCP pipeline can route it to the right
provider and player.

It is the single home for OCP's media NLP: it classifies along orthogonal
[axes](docs/classification-model.md) and emits a provider-ready
[`mediavocab.Signals`](https://github.com/TigreGotico/mediavocab).

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()                       # bundled .voc keyword classifier
clf.classify_full("i want to watch an anime", "en-us")
# MediaClassification(media_type=EPISODIC_SERIES, playback_type=video,
#                     structure=episodic, domain=ocp_play, genres=['anime'], ...)
clf.to_signals("play some jazz", "en-us")           # -> mediavocab.Signals (hand to a MediaProvider)
```

## Install

```bash
pip install ovos-media-classifier          # lean: ovos-utils + mediavocab only
```

The default ships the zero-dependency **keyword (`.voc`) classifier** — the
minimum needed for OCP to work. Richer strategies are opt-in extras:

```bash
pip install ovos-media-classifier[onnx]    # trained ONNX backend (experimental)
pip install ovos-media-classifier[ner]     # entity-list matching (the user's library)
```

## What it does

- **Multi-axis classification** — coarse-to-fine: `domain → modality
  (audio/video/…) → structure (single/episodic/…) → media type`, plus genre
  tags. Predicting the coarse axes first constrains the leaf and improves
  accuracy. See [the model](docs/classification-model.md).
- **Content filtering** — recognises sensitive requests so OCP can **block them
  by default** (adult is blocked unless `allow_adult_content`). Detect-to-block,
  not provision. See [content filtering](docs/content-filtering.md).
- **Pluggable** — pick a backend by config, or load a 3rd-party classifier via
  the `opm.media.classifier` entry point. `AbstractMediaClassifier` is the
  contract.

```python
from ovos_media_classifier import ContentFilter
ContentFilter().check(clf, "play porn", "en-us")    # (True, 'blocked genre: adult')
```

## Backends

| backend | install | what it is |
|---|---|---|
| **keyword** (`.voc`) | core (default) | zero-dep coarse-to-fine matching — the functional minimum |
| **ONNX** | `[onnx]` | trained model (onnxruntime + numpy); experimental |
| **NER** | `[ner]` | matches the user's real [entity lists](docs/entity-lists.md) (csv/tsv/jsonl/HF/runtime) |
| external | a plugin | any `opm.media.classifier` plugin |

See [docs/backends.md](docs/backends.md) and [docs/external-plugins.md](docs/external-plugins.md).

## Documentation

[docs/index.md](docs/index.md) — start here.
[classification model](docs/classification-model.md) ·
[stable API](docs/stable-api.md) ·
[taxonomy](docs/taxonomy.md) ·
[content filtering](docs/content-filtering.md) ·
[entity lists](docs/entity-lists.md) ·
[external plugins](docs/external-plugins.md) ·
[benchmarks](docs/benchmarks/README.md) ·
[examples/](examples/) · [training/](training/)

## Credits

The original OCP dataset was sponsored by [@NeonGeckoCom](https://github.com/NeonGeckoCom/);
more recent media-metadata datasets are by **TigreGotico** on
[Hugging Face](https://huggingface.co/collections/TigreGotico/media-metadata).
