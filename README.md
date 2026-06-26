# ovos-media-classifier

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Media-type classification for **OVOS Common Play (OCP)**. Given a spoken request
— *"play some music"*, *"watch the news"* — it answers, fast and offline, **what
kind of media is wanted**, so the OCP pipeline can route it to the right provider
and player.

It is the single home for OCP's media-command NLP: it classifies a request along
orthogonal [axes](docs/classification-model.md) (domain · modality · structure ·
media type, plus genre tags) and emits a provider-ready
[`mediavocab.Signals`](https://github.com/TigreGotico/mediavocab).

## Quickstart

```bash
pip install ovos-media-classifier
```

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()                  # bundled .voc keyword classifier
clf.classify("play some music", "en-us")       # -> (<MediaType.MUSIC: 'music'>, 0.6)
```

That is the whole minimum: install, load, classify. The default needs no model
files and no ML dependencies — it runs fully offline.

The full multi-axis result and a provider-ready `Signals`:

```python
clf.classify_full("i want to watch an anime", "en-us").as_dict()
# {'media_type': 'episodic_series', 'playback_type': 'video', 'structure': 'episodic',
#  'domain': 'ocp_play', 'genres': ['anime'], 'confidence': 0.6, 'control_intent': None}

clf.to_signals("play some music", "en-us")      # -> mediavocab.Signals (hand to a MediaProvider)
```

## Content filtering

A **detect-to-block** moderation layer recognises sensitive requests so OVOS can
refuse them. `adult` is blocked by default (lift it with `allow_adult_content`).

```python
from ovos_media_classifier import ContentFilter
ContentFilter().check(clf, "play porn", "en-us")   # (True, 'blocked genre: adult')
```

See [content filtering](docs/content-filtering.md).

## Backends

`load_media_classifier(config)` returns one classifier. They all implement the
same `AbstractMediaClassifier` contract, so callers never care which ran.

| Backend | What it is | Install |
|---|---|---|
| **keyword** (`.voc`) | zero-dependency phrase matching — the offline default | core |
| **NER** | Aho-Corasick exact match over the user's [entity lists](docs/entity-lists.md) | `[ner]` |
| **ONNX** | trained domain + play heads loaded from a model bundle | `[onnx]` |
| **external** | any classifier registered under `opm.media.classifier` | a plugin |

```bash
pip install ovos-media-classifier[ner]     # entity-list matching (the user's library)
pip install ovos-media-classifier[onnx]    # trained ONNX backend
```

See [docs/backends.md](docs/backends.md) and [docs/external-plugins.md](docs/external-plugins.md).

## Command vs content classification

This package classifies a **voice command** (*what does the user want?*). That is
a different problem from `mediavocab.text.classify`, which classifies a piece of
**catalog content** (*what kind of item is this?*). They share the
`mediavocab.MediaType` vocabulary but answer opposite questions — do not
substitute one for the other. See
[taxonomy.md](docs/taxonomy.md#query-vs-content-classification).

## Documentation

**Start at [docs/index.md](docs/index.md)** for the audience-routing table, or
read the [glossary](docs/glossary.md) first if the terms are new.

- New here → [glossary](docs/glossary.md) · [index](docs/index.md) · [examples/](examples/)
- API reference → [stable API](docs/stable-api.md)
- The model → [classification model](docs/classification-model.md) · [taxonomy](docs/taxonomy.md)
- Tuning backends → [backends](docs/backends.md) · [entity lists](docs/entity-lists.md) · [contextual classification](docs/contextual-classification.md)
- Moderation → [content filtering](docs/content-filtering.md)
- Writing a classifier → [external plugins](docs/external-plugins.md)
- Measuring → [benchmarks](benchmarks/README.md)

## Credits

Media-metadata datasets by **TigreGotico** on
[Hugging Face](https://huggingface.co/collections/TigreGotico/media-metadata).

## License

Apache-2.0.
