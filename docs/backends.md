# Backends

`load_media_classifier(config=None, voc_match_func=None)` returns one classifier.
Every backend implements [`AbstractMediaClassifier`](stable-api.md), so callers
treat them identically, the choice only affects accuracy, dependencies and what
signal is available.

The package ships several backends. The **keyword** backend is the default and
has no ML dependencies; **ONNX**, the learned **embedding-router**
([embedding-router.md](embedding-router.md)) and **NER** are opt-in extras. Any
`opm.media.classifier` plugin can also be loaded by name.

## Selection

`load_media_classifier(config)` picks a backend in this order; the first matching
config key wins, and any load failure falls back to the keyword classifier so the
zero-dependency default is always available:

| Order | Backend | Selected by | Extra |
|---|---|---|---|
| 1 | external plugin | `config["media_classifier_plugin"]` | a plugin package |
| 2 | ONNX | `config["media_classifier_onnx_model"]` | `[onnx]` |
| 3 | embedding-router (hybrid) | `config["media_classifier_embedding_router"]` | `[onnx]` |
| 4 | NER | `config["media_classifier_entities"]`, `media_classifier_wordlists`, or `media_classifier_ner_csv` | `[ner]` |
| 5 | keyword (default) | nothing else set | core |

```python
from ovos_media_classifier import load_media_classifier

load_media_classifier()                                                   # keyword (default)
load_media_classifier({"media_classifier_onnx_model": "/models/ocp"})     # ONNX bundle dir
load_media_classifier({"media_classifier_embedding_router": "/models/router",  # learned router (hybrid)
                       "media_classifier_entity_library": {"anime_title": ["Attack on Titan"]}})
load_media_classifier({"media_classifier_entities": {"entity_lists": [...]}})  # NER
load_media_classifier({"media_classifier_plugin": "my-classifier"})       # external plugin
```

## Keyword classifier (default, zero-dep)

```python
clf = load_media_classifier()                                # bundled locale
clf = load_media_classifier(voc_match_func=self.voc_match)   # pipeline mode
```

Matches the query against bundled per-language `.voc` files
(`ovos_media_classifier/locale/<lang>/<Vocab>.voc`), using word-boundary matching
from `ovos-spec-tools`. No ML dependencies, no model files, fully offline. In
*pipeline mode* the OCP pipeline plugin owns the `.voc` files and passes its
`voc_match` method as `voc_match_func`; in *standalone mode* the classifier reads
the bundled files directly. Matching runs in a fixed priority order (e.g.
`MusicVideoKeyword` before `MusicKeyword`) and surfaces genre tags (`anime`,
`adult`, …) via `classify_genres()`.

It predicts the coarse axes (`playback_type`, `structure`) from their own `.voc`
cues and chooses the leaf `MediaType` leaf-first, so `classify_full()` returns a
complete multi-axis result with no model file. How and why this differs from a
trained head's soft-gating is in
[classification-model.md](classification-model.md#41-predict-coarse-to-fine-the-keyword-classifier).

Because it derives the domain from the media-type result, the keyword classifier
reports `OCP_PLAY` / `NOT_OCP` and does not detect `OCP_CONTROL`.

**Use when:** you want the offline default with no setup. Use *pipeline mode*
when you are inside the OCP pipeline and already have a `voc_match` function;
otherwise use the no-arg bundled mode.

Languages bundled: `ca-es`, `da-dk`, `de-de`, `en-us`, `es-es`, `eu-es`, `fr-fr`,
`gl-es`, `it-it`, `nl-nl`, `pl-pl`, `pt-br`, `pt-pt`.

## NER classifier (`[ner]` extra)

"NER" here is the user-facing name for **exact entity-list matching**, not
statistical named-entity recognition: the backend is
`AhocorasickMediaClassifier`, which compiles the configured entity lists into
an Aho-Corasick automaton and matches the user's *actual* library (artists,
titles, stations) verbatim. No model guesses spans — an entity either is in
a registered list or it does not match.

```python
pip install ovos-media-classifier[ner]
```

```python
from ovos_media_classifier.entities import EntitiesContainer
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

container = EntitiesContainer.from_sources(["/data/library.csv"])
clf = AhocorasickMediaClassifier.from_container(container)
clf.classify("play inception", "en-us")     # -> (<MediaType.MOVIE: 'movie'>, ...)
```

The NER backend matches the user's *real* media, their artists, titles and
stations, captured as **entity lists** (`label → list of strings`), with an
Aho-Corasick automaton for fast exact substring matching. Each entity label maps
to a media type via `NER_LABEL_TO_MEDIA_TYPE` (and to genres via
`NER_LABEL_TO_GENRES`), so *"play Inception"* resolves to `MOVIE` because
*Inception* is in the library, high confidence, language-agnostic, zero
linguistic guessing.

It is selected by config (`media_classifier_entities` /
`media_classifier_wordlists` / `media_classifier_ner_csv`) or built directly via
`AhocorasickMediaClassifier.from_container` / `from_wordlists` / `from_csv`. The
entity lists, their source specs (`.csv` / `.tsv` / `.jsonl` / HuggingFace /
media-server / inline / runtime) and the perf/memory tradeoff of loading more
entities are documented in [entity-lists.md](entity-lists.md).

**Use when:** the user has a known library (Jellyfin, the \*arr stack, Music
Assistant, a static roster) and you want titles to resolve exactly. See
[contextual-classification.md](contextual-classification.md).

## ONNX classifier (`[onnx]` extra)

```python
pip install ovos-media-classifier[onnx]
```

```python
load_media_classifier({"media_classifier_onnx_model": "/models/ocp-bundle"})
```

A trained classifier with **one ONNX head per axis**, `domain`, `media_type`,
`playback_type`, `structure`, `explicitness`, `content_form`, `programme_format`,
`variant` (single-label) and `content_form_genres`, `content_genres`,
`accessibility` (multi-label), plus `numpy`. It
depends on raw `onnxruntime` + `numpy` only. The factory loads it from a
self-describing **model-bundle directory** of `<axis>.onnx` files plus a
`meta.json` manifest (which carries the ordered feature names, the per-head
index→label maps and the multi-label thresholds, so the runtime makes no
hard-coded assumptions and loads whichever heads the bundle ships, a missing head
simply derives its axis). Set `media_classifier_onnx_model` to that directory. The
full bundle layout and retrain contract are in [model.md](model.md#4-self-describing-bundle--retrain-contract).

A trained model predicts each axis with its own head and can capture
per-utterance nuance the keyword backend cannot, see
[classification-model.md](classification-model.md#42-predict-each-axis-with-its-own-head-trained-plugins).
The bundle format is documented in `ovos_media_classifier/onnx.py`.

**Use when:** you have a trained model bundle and want real-query accuracy beyond
the keyword floor.

## External plugins

Any subclass of [`AbstractMediaClassifier`](stable-api.md) registered under the
`opm.media.classifier` entry-point group is loadable by name via
`config["media_classifier_plugin"]`. `load_media_classifier()` falls back to the
keyword classifier if it fails to load, so an external plugin never hard-fails the
pipeline. The full recipe, registering, loading by name, and which methods to
override, is in [external-plugins.md](external-plugins.md).

---
[← Taxonomy](taxonomy.md) · [Home](index.md) · [Embedding router →](embedding-router.md)
