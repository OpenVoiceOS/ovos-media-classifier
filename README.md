# ovos-media-classifier

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

> **Work in progress: pre-release software.** The project is under active
> development. It is not yet deployed in OpenVoiceOS, and its APIs may change
> without notice. It is published in the open for transparency. Do not depend
> on it in production yet.

**A self-describing, pluggable media-intent classifier for voice assistants.**
Given a spoken request, such as *"play some music"*, *"watch an anime"*, or
*"read me a chapter of Dune"*, it answers, fast and offline, **what kind of media is
wanted**. The **OVOS Common Play (OCP)** pipeline then routes the request to
the right provider and player.

It is a **router, not a resolver.** It gates the request (*is this a media
request at all?*), routes it by `media_type` / `playback_type` (which
`MediaProvider`s to call, and which to skip), applies content policy (`adult`
drops adult providers), and hands the providers a
[`mediavocab.Signals`](https://github.com/TigreGotico/mediavocab) as **search
context**. It does not resolve a title to a stream. The providers do that,
so `Signals.title` carries the raw query by design. The real test of a router
is whether it routes real speech correctly. See the
[routing eval](docs/routing-eval.md).

It is the single home for OCP's media-command NLP. It is **multi-task**:
every request is classified along several orthogonal
[axes](docs/classification-model.md) at once, instead of into one label.

- **domain**: is this a media request at all (`ocp_play` / `ocp_control` / `not_ocp`)
- **media_type**: the concrete `mediavocab.MediaType` leaf (`music`, `movie`, `podcast`, …)
- **playback_type**: the modality (`audio` / `video` / `paged` / `interactive`)
- **structure**: the temporal shape (`single` / `episodic` / `continuous` / `collection`)
- **explicitness**: `clean` / `adult`
- **tags**: a multi-label, namespaced descriptive axis: `genre:rock` / `mood:chill` / `era:1980s`
- **qualifiers**: result-narrowing filters: `black_and_white` / `silent` / `live` / `subtitled` / …
- **content-form genres**: `adult` / `anime` / `animation` / `asmr` (drives the content filter)

The supervision comes from translatable `.intent` templates, slot-filled with
**real entity metadata** (IMDb, MusicBrainz, AniList, LibriVox, …). The
trained backend ships as a **rules to learned-context to
learned-context+NER ladder**, packaged in self-describing ONNX bundles. Each
rung is a drop-in upgrade over the last.

## Quickstart

```bash
pip install ovos-media-classifier
```

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()                  # bundled .voc keyword classifier, zero deps
clf.classify("play some music", "en-us")       # -> (<MediaType.MUSIC: 'music'>, 0.6)
```

That is the whole minimum: install, load, classify. The default needs no
model files and no ML dependencies. It runs fully offline.

The full multi-axis result, and a provider-ready `mediavocab.Signals`:

```python
clf.classify_full("i want to watch an anime", "en-us").as_dict()
# {'media_type': 'episodic_series', 'playback_type': 'video', 'structure': 'episodic',
#  'domain': 'ocp_play', 'genres': ['anime'], 'confidence': 0.6, 'control_intent': None}

clf.classify_tags("play some 80s rock", "en-us")  # -> ['genre:rock', 'era:1980s']  (trained backend)

clf.to_signals("play some music", "en-us")        # -> mediavocab.Signals (hand to a MediaProvider)
```

(`classify_full().as_dict()` carries the single-label axes plus the
content-form `genres`. The multi-label `tags` / `qualifiers` axes are read
with their own `classify_tags()` / `classify_qualifiers()` methods. On the
zero-dep keyword default, `tags` stays empty. Only the trained ONNX backend
fills it.)

`classify_full` is **context-aware**. The minimal call is
`classify_full(query, lang)`. Two optional arguments thread per-query context
with no retraining: `player_status` (now-playing state, for control /
*"play something else"* follow-ups) and `ner_list` (the user's live
entities, for NER matching and the embedding router's runtime injection). See
[contextual classification](docs/contextual-classification.md).

## Benchmark

The headline result is the **lift across the ladder**: the deterministic
keyword rules, then a model on context (keyword) features only, then the
same model once a NER store has surfaced the user's entities. Per-axis, on
the held-out **test split (34,700 synthetic utterances)**:

| axis (metric) | rules | learned-context | learned-context+NER |
|---|---|---|---|
| domain (acc) | 0.833 | 0.866 | **0.986** |
| media_type (acc) | 0.629 | 0.778 | **0.964** |
| playback_type (acc) | 0.702 | 0.895 | **0.988** |
| structure (acc) | 0.708 | 0.907 | **0.990** |
| explicitness (acc) | 0.988 | 0.989 | **0.997** |
| content_form_genres (macro-F1) | 0.706 | 0.738 | **0.975** |
| qualifiers (macro-F1) | 0.000 | 0.746 | **0.906** |
| tags (macro-F1) | 0.000 | 0.547 | **0.581** |

Content filter (driven by the `content_form_genres` axis), same ladder:

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | bundle |
|---|---|---|---|---|---|---|
| rules | 0.481 | 0.510 | 0.000 | 0.32 | 0.50 | — |
| learned-context | 0.481 | 0.510 | 0.000 | 0.21 | 0.25 | 176 KiB |
| learned-context+NER | **0.922** | **0.936** | 0.001 | 0.21 | 0.26 | 289 KiB |

Sub-millisecond, in a 289 KiB bundle. **Honesty notes:** these numbers come
from the *synthetic* eval split. They measure the model's *capability* given
populated features, not field accuracy on arbitrary speech (the keyword
floor is about 0.29 on a neutral real-text split). The **keyword backend is
the zero-dependency default**. The context+NER column needs a wired-in NER
store to surface the `ner_*` features. And `tags` stays low by design
(about 0.58). See [limitations](docs/model.md#6-limitations). Full table
and method: [docs/model.md](docs/model.md) ·
[benchmarks/](benchmarks/README.md).

## Axes vs. tags

The single-label heads above are **axes**: exactly one answer per query (a
request *is* `audio`, *is* `episodic`). The open-vocabulary descriptive
signals, **genre, mood, and era**, are not axes. A query can carry several
at once, and they all live in slot *value text* (the decade is *in* the
year, the mood is *in* the activity phrase). So they fold into one
**multi-label, namespaced `tags` head** (`genre:` / `mood:` / `era:`)
instead of three starved single-label heads. `classify_content_genres()` /
`classify_mood()` / `classify_era()` read the matching slice. This framing
keeps the axis count honest while still modeling the descriptive signal.
See [classification-model.md](docs/classification-model.md).

## Backends

`load_media_classifier(config)` returns one classifier. They all implement
the same `AbstractMediaClassifier` contract, so callers never care which one
ran, and any load failure falls back to the keyword default.

| Backend | What it is | Install |
|---|---|---|
| **keyword** (`.voc`) | zero-dependency phrase matching, high-precision, abstains when unsure. The offline default | core |
| **NER** | Aho-Corasick exact match over the user's [entity lists](docs/entity-lists.md) (their real library) | `[ner]` |
| **embedding-router** (hybrid) | learned [open-vocab router](docs/embedding-router.md): keyword stays the floor, the router fills keyword's abstains using a gazetteer plus the user's injected library | `[onnx]` |
| **ONNX** | the trained multi-task per-axis heads, loaded from a self-describing bundle | `[onnx]` |
| **external** | any classifier registered under `opm.media.classifier` | a plugin |

```bash
pip install ovos-media-classifier[ner]     # entity-list matching (the user's library)
pip install ovos-media-classifier[onnx]    # trained ONNX + embedding-router backends
```

**The keyword backend is the floor; a learned backend has to earn its
place.** The zero-dep keyword classifier is deliberately high-precision and
*abstains* to GENERIC when it has no cue. This is a safe outcome, because an
abstain still lets every provider search. The
[embedding-router hybrid](docs/embedding-router.md) keeps keyword as the
first pass and only fills those abstains. It resolves open-vocabulary titles
by injecting the **user's own library** as entities at runtime (no
retraining). That entity injection is what lowers the mis-route rate
*below* the keyword floor. See the [routing eval](docs/routing-eval.md).

> **Live routing uses a bounded entity set.** Entity / gazetteer matching
> cost scales with the number of injected titles, so **live** classification
> runs on a bounded set: the user's library plus a capped popular
> [gazetteer](docs/metadatarr-routing.md) (default about 1000 titles per
> type, p95 a few ms). A **1M-entity set (for example full MusicBrainz) is
> for OFFLINE tagging only, never live classification.** The optional
> online `metadatarr.resolve` layer (about seconds per title) is for offline
> tagging or a long-running agent. It stays **off by default** and never
> runs in the live OCP pipeline.

See [docs/backends.md](docs/backends.md). To write your own backend, or
train your own bundle (including adding a brand-new axis end-to-end), see
[docs/extending.md](docs/extending.md).

## Content filtering

A **detect-to-block** moderation layer recognizes sensitive requests so OVOS
can refuse them. It reads the `content_form_genres` axis, so `adult` can be
flagged **independently of the media-type leaf** (a single leaf mistake
never unblocks it). `adult` is **blocked by default** (lift it with
`allow_adult_content`). The adult / hentai data exists for detection only,
never for provision.

```python
from ovos_media_classifier import ContentFilter
ContentFilter().check(clf, "play some porn", "en-us")   # (True, 'blocked genre: adult')
```

See [content filtering](docs/content-filtering.md).

## Command vs content classification

This package classifies a **voice command** (*what does the user want?*).
That is a different problem from `mediavocab.text.classify`, which
classifies a piece of **catalog content** (*what kind of item is this?*).
They share the `mediavocab.MediaType` vocabulary but answer opposite
questions. Do not substitute one for the other. See
[taxonomy.md](docs/taxonomy.md#query-vs-content-classification).

## Documentation

**Start at [docs/index.md](docs/index.md)** for the audience-routing table,
or read the [glossary](docs/glossary.md) first if the terms are new.

- New here → [glossary](docs/glossary.md) · [index](docs/index.md) · [examples/](examples/)
- API reference → [stable API](docs/stable-api.md)
- The model → [classification model](docs/classification-model.md) · [the trained model](docs/model.md) · [taxonomy](docs/taxonomy.md) · [hierarchical experiment](docs/hierarchical.md)
- The data → [dataset](docs/dataset.md) · [data sources](docs/data-sources.md) · [dataset plots](docs/plots/dataset/)
- Tuning backends → [backends](docs/backends.md) · [embedding-router](docs/embedding-router.md) · [entity lists](docs/entity-lists.md) · [contextual classification](docs/contextual-classification.md) · [open-vocab routing](docs/metadatarr-routing.md)
- Moderation → [content filtering](docs/content-filtering.md)
- Writing / training a classifier → [extending](docs/extending.md) · [external plugins](docs/external-plugins.md)
- Measuring → [routing eval](docs/routing-eval.md) (the source of truth) · [benchmarks](benchmarks/README.md)

## Credits

Media-metadata datasets by **TigreGotico** on
[Hugging Face](https://huggingface.co/collections/TigreGotico/media-metadata).

## License

Apache-2.0.
</content>
