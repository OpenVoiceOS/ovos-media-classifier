# ovos-media-classifier

Media-type **command/intent** classification for [OCP (OVOS Common Playback)](https://github.com/OpenVoiceOS/ovos-core). Given a spoken command such as _"play some music"_ it resolves a small set of **orthogonal axes** in one pass: the **domain** (is this a media request at all — play / control / not-OCP), the **modality** (`playback_type`: audio / video / paged / interactive), the **structure** (single / episodic / continuous / collection), and the concrete `mediavocab.MediaType` **leaf** — plus orthogonal genre tags. Each axis is a coarse, high-signal question; together they pin down what the user wants without forcing every request into one rigid label. An always-on content filter lets OVOS block sensitive (e.g. adult) requests by default.

Why axes instead of a single label or a strict tree — and how trained backends predict each axis with its own head — is in [docs/classification-model.md](docs/classification-model.md).

The default classifier is the bundled-`.voc` **KeywordMediaClassifier** — zero ML dependencies, no model files, fully offline. It is the minimum required for OCP to be functional. An **optional** NER (entity-based) backend is available behind extras (see [NER backend](#ner-entity-based-backend-optional)); other richer strategies (trained ONNX models, …) land later as independent, separately-reviewed plugins through the `opm.media.classifier` mechanism (see [External plugins](#external-plugins-opm)). The lean keyword default is always preserved — nothing extra is imported unless you opt in.

> **Command classification, not catalog classification.** This package classifies a *voice command* ("play the news"). It is distinct from `mediavocab.text.classify`, which classifies a piece of *catalog content* (a video's title/description) into a media type. Use this package in the OCP pipeline; use `mediavocab.text` when tagging library items.

## Install

```bash
pip install ovos-media-classifier
```

The only runtime dependencies are `ovos-utils` and `mediavocab` — the keyword
classifier reads the bundled `.voc` files directly. The optional NER backend is
opt-in via extras (`[ner]`, `[media_servers]`, `[huggingface]`); see
[NER backend](#ner-entity-based-backend-optional).

## Quickstart

`load_media_classifier()` with no config returns the bundled `.voc` keyword
classifier — no ML dependencies, no model files, works offline.

`classify_full()` resolves every axis in one call and returns a
`MediaClassification`:

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()

print(clf.classify_full("play a podcast", "en-us").as_dict())
print(clf.classify_full("play the breaking bad tv series", "en-us").as_dict())
print(clf.classify_full("turn off the kitchen lights", "en-us").as_dict())
```

Verified output:

```
{'media_type': 'podcast', 'playback_type': 'audio', 'structure': 'episodic', 'domain': 'ocp_play', 'genres': [], 'confidence': 0.6}
{'media_type': 'episodic_series', 'playback_type': 'video', 'structure': 'episodic', 'domain': 'ocp_play', 'genres': [], 'confidence': 0.6}
{'media_type': 'generic', 'playback_type': 'unknown', 'structure': 'unknown', 'domain': 'not_ocp', 'genres': [], 'confidence': 0.0}
```

The non-media request (_"turn off the kitchen lights"_) collapses every content
axis to `unknown` / `not_ocp` — IoT/device control is **not media**, so it is
routed back to the rest of the OVOS pipeline.

The individual axes are also available one at a time:

```python
print(clf.classify("play some music", "en-us"))             # leaf MediaType + confidence
print(clf.classify_playback_type("play a movie", "en-us"))  # modality axis
print(clf.classify_structure("put on the radio", "en-us"))  # structure axis
print(clf.classify_genres("play some anime", "en-us"))      # orthogonal genre tags
print(clf.classify_domain("play a podcast", "en-us"))       # domain axis
print(clf.is_ocp_query("what is the weather", "en-us"))     # is this OCP at all?
```

Verified output:

```
(<MediaType.MUSIC: 'music'>, 0.6)
PlaybackType.VIDEO
Structure.CONTINUOUS
['anime']
(<OCPDomain.OCP_PLAY: 'ocp_play'>, 0.6)
(False, 0.0)
```

See [docs/stable-api.md](docs/stable-api.md) for every method and
[docs/classification-model.md](docs/classification-model.md) for the axis model.

## The keyword classifier

`load_media_classifier()` returns the bundled keyword backend. It substring-matches
the query against per-language `.voc` files
(`ovos_media_classifier/locale/<lang>/<Vocab>.voc`) — no ML dependencies, no model
files, fully offline.

```python
clf = load_media_classifier()                                # bundled locale
clf = load_media_classifier(voc_match_func=self.voc_match)   # pipeline mode
```

In *pipeline mode* the OCP pipeline plugin owns the `.voc` files and passes its
`voc_match` method as `voc_match_func`; in *standalone mode* the classifier reads the
bundled files directly. Matching runs in a fixed priority order (e.g.
`MusicVideoKeyword` before `MusicKeyword`) and surfaces genre tags (`anime`,
`adult`, …) via `classify_genres()`. Bundled languages: `ca-es`, `da-dk`, `de-de`,
`en-us`, `es-es`, `eu-es`, `fr-fr`, `gl-es`, `it-it`, `nl-nl`, `pl-pl`, `pt-br`,
`pt-pt`.

See [docs/backends.md](docs/backends.md) for details, and
[docs/external-plugins.md](docs/external-plugins.md) for writing a richer classifier.

## NER (entity-based) backend (optional)

The **NER backend** matches an utterance against the **real entities the user
has** — movie titles, artist names, streaming-service names, … — using an
[Aho-Corasick](https://pypi.org/project/ahocorasick-ner/) automaton:

```
"play Inception"   → MOVIE   (Inception is a known movie_title)
"put on Radiohead" → MUSIC    (Radiohead is a known artist_name)
```

It is fully opt-in and pulled in only via extras — the default install stays
lean (`ovos-utils` + `mediavocab` only):

```bash
pip install ovos-media-classifier[ner]              # AhocorasickMediaClassifier
pip install ovos-media-classifier[media_servers]    # Radarr/Sonarr/Lidarr/Jellyfin/Music Assistant loaders (requests)
pip install ovos-media-classifier[huggingface]      # datasets loader
```

Select it through config — the factory chooses the NER backend before the
keyword fallback, and falls back to keyword (with a warning) if the optional
dependency is missing:

```python
from ovos_media_classifier import load_media_classifier

# inline wordlists of entities the user owns
clf = load_media_classifier({
    "media_classifier_wordlists": {
        "movie_title": ["Inception", "The Matrix"],
        "artist_name": ["Radiohead"],
    },
})

# or pull entities live from media servers / HuggingFace
clf = load_media_classifier({
    "media_classifier_entities": {
        "radarr":   {"url": "http://localhost:7878", "api_key": "…"},
        "lidarr":   {"url": "http://localhost:8686", "api_key": "…"},
        "jellyfin": {"url": "http://localhost:8096", "api_key": "…"},
        "huggingface": [{"dataset": "TigreGotico/ocp-entities"}],
    },
})

# or from a CSV of (label, value) / (entity, label) rows
clf = load_media_classifier({"media_classifier_ner_csv": "/path/to/entities.csv"})
```

Entities can also be registered at runtime (e.g. by skills announcing their
content) via an `EntitiesContainer` — newly added entities are reflected in
`classify()` immediately, with no automaton rebuild. See
[`examples/ner_backend.py`](examples/ner_backend.py).

## Future strategies (plugins)

Trained ONNX models and other richer strategies arrive as independent,
separately-reviewed additions that register under the `opm.media.classifier`
entry-point group and load by name — the same mechanism any third party uses (see
[External plugins](#external-plugins-opm)). The core package stays lean: one
zero-dependency keyword classifier (plus the opt-in NER backend) and the plugin
contract.

## Optional ONNX trained backend (opt-in)

An **experimental, opt-in** trained backend ships in-tree as
`ovos_media_classifier.onnx.OnnxMediaClassifier`. It uses **raw `onnxruntime` +
`numpy` only** (no heavy ML framework), and both are imported lazily — the
default install and the keyword path never touch them.

```bash
pip install ovos-media-classifier[onnx]
```

```python
from ovos_media_classifier import load_media_classifier

# point at a self-describing model bundle directory
clf = load_media_classifier({"media_classifier_onnx_model": "/path/to/bundle"})
```

A bundle is a self-describing directory:

```
<bundle>/
  ├── domain.onnx   # domain head  (ocp_play / ocp_control / not_ocp)
  ├── play.onnx     # play head    (fine-grained media-type label)
  └── meta.json     # {feature_names, domain_labels, play_labels, ...}
```

`meta.json` records the ordered `feature_names` (the categorical feature columns
the model was trained on) plus `domain_labels` / `play_labels` (output-index →
label). At inference a sparse keyword-feature dict (from the bundled `.voc`
files) is vectorized in `feature_names` order, run through both heads
(softmax → argmax), and mapped to a `mediavocab.MediaType` + genres + coarse
axes. If the extra is missing or the bundle is invalid, the factory logs a
warning and falls back to the keyword classifier. See
[`examples/onnx_backend.py`](examples/onnx_backend.py) and the module docstring
for the full bundle contract.

## Content filtering

`ContentFilter` is a detect-to-block content-moderation / parental-control layer.
The classifier surfaces a `mediavocab` genre signal (`adult` for adult/hentai/porn
queries) and the filter decides whether the request is allowed. **`adult` is
blocked by default** (`allow_adult_content: false`). This package detects sensitive
media so OVOS can refuse it; it is not a content provider.

```python
from ovos_media_classifier import load_media_classifier, ContentFilter

clf = load_media_classifier()
cf = ContentFilter()  # default policy: adult blocked
print(cf.check(clf, "play some porn", "en-us"))
```

Verified output:

```
(True, 'blocked genre: adult')
```

Configure via `media_content_filter` (`enabled`, `blocked_genres`,
`blocked_media_types`) and the top-level `allow_adult_content` flag. See
[docs/content-filtering.md](docs/content-filtering.md).

## External plugins (OPM)

Richer classifiers — including the future trained/NER strategies — register under
the `opm.media.classifier` entry-point group and load by name.
`AbstractMediaClassifier` is the contract.

```toml
# in a 3rd-party package's pyproject.toml
[project.entry-points."opm.media.classifier"]
my-classifier = "my_pkg:MyMediaClassifier"
```

```python
clf = load_media_classifier(config={"media_classifier_plugin": "my-classifier"})
```

If the named plugin fails to load, the factory logs a warning and falls back to the
built-in keyword classifier — an external plugin never hard-fails the pipeline. See
[docs/external-plugins.md](docs/external-plugins.md).

## Training

Model training lives in the top-level `training/` directory and is **not** shipped
in the wheel. It produces models consumed by future classifier plugins, not by this
package. Run it from a checkout:

```bash
python -m training.build_dataset
```

See `training/README.md` for the full data-gathering and training workflow.

## Documentation

- [docs/index.md](docs/index.md) — table of contents
- [docs/classification-model.md](docs/classification-model.md) — the multi-axis model: the four axes + tags, why orthogonal axes beat a strict tree, and the `MediaType`→(playback_type, structure) defaults
- [docs/taxonomy.md](docs/taxonomy.md) — mediavocab enforcement, intent→type/genre mapping, query-vs-content distinction
- [docs/backends.md](docs/backends.md) — the bundled keyword classifier and how to add a classifier plugin
- [docs/content-filtering.md](docs/content-filtering.md) — detect-to-block content moderation
- [docs/external-plugins.md](docs/external-plugins.md) — registering 3rd-party classifiers via `opm.media.classifier`
- [docs/stable-api.md](docs/stable-api.md) — the `AbstractMediaClassifier` contract and return types

See [benchmarks](benchmarks/README.md) for the reproducible accuracy/latency
harness (it evaluates whichever classifiers are installed; only the keyword
classifier ships in this release).

## Credits

The original OCP training dataset was sponsored by **NeonGecko**. More recent
media-metadata datasets are published by **TigreGotico** on Hugging Face:
<https://huggingface.co/collections/TigreGotico/media-metadata>.
