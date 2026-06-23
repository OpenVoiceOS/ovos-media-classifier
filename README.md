# ovos-media-classifier

Pluggable media-type **command/intent** classification for [OCP (OVOS Common Playback)](https://github.com/OpenVoiceOS/ovos-core). Given a spoken command such as _"play some music"_ it decides three things: whether the command targets OCP at all, which OCP **domain** it belongs to (play / control / not-OCP), and — for play requests — which `mediavocab.MediaType` and genre tags the user is asking for. Multiple backends sit behind one stable interface, from a zero-dependency bundled-keyword matcher to trained ONNX models, and an always-on content filter lets OVOS block sensitive (e.g. adult) requests by default.

> **Command classification, not catalog classification.** This package classifies a *voice command* ("play the news"). It is distinct from `mediavocab.text.classify`, which classifies a piece of *catalog content* (a video's title/description) into a media type. Use this package in the OCP pipeline; use `mediavocab.text` when tagging library items.

## Install

```bash
pip install ovos-media-classifier              # core: keyword backend, zero ML deps
pip install ovos-media-classifier[guided]      # ONNX trained backend (recommended; torch-free at runtime)
pip install ovos-media-classifier[sklearn]     # TF-IDF + LogisticRegression
pip install ovos-media-classifier[padatious]   # pattern + ML intent matching
pip install ovos-media-classifier[ner]         # AhocorasickNER (live media-server / HF entities)
pip install ovos-media-classifier[m2v]         # Model2Vec hierarchical model
pip install ovos-media-classifier[all]         # every backend
```

## Quickstart

The default backend (`load_media_classifier()` with no config) uses the bundled
`.voc` keyword files — no ML dependencies, no model files, works offline.

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()

print(clf.classify("play some music", "en-us"))
print(clf.classify("play a movie", "en-us"))
print(clf.classify_genres("play some anime", "en-us"))
print(clf.classify_domain("play a podcast", "en-us"))
print(clf.is_ocp_query("what is the weather", "en-us"))
```

Verified output:

```
(<MediaType.MUSIC: 'music'>, 0.6)
(<MediaType.MOVIE: 'movie'>, 0.6)
['anime']
(<OCPDomain.OCP_PLAY: 'ocp_play'>, 0.6)
(False, 0.0)
```

`classify` returns a `mediavocab.MediaType` plus a confidence; `classify_genres`
returns the orthogonal genre tags (anime, adult, …); `classify_domain` returns an
`OCPDomain`; `is_ocp_query` returns a bool. See [docs/stable-api.md](docs/stable-api.md).

## Backends

`load_media_classifier(config)` selects a backend from the config keys present
(first match wins), falling through to the next on any import/load error. The
default with no config is the bundled keyword backend.

| Backend | Config key | Extra | Notes |
|---|---|---|---|
| Model2Vec | `media_classifier_model` | `[m2v]` | Hierarchical neural model |
| **GuidedEmbeddings (ONNX)** | `media_classifier_guided_model` | `[guided]` | **Recommended trained backend**; torch-free at runtime; loads any export matching the feature contract |
| scikit-learn | `media_classifier_sklearn_model` | `[sklearn]` | TF-IDF + LogisticRegression; fast |
| Padatious | `media_classifier_padatious_dir` | `[padatious]` | Pattern + ML; native `ocp_control` head |
| AhocorasickNER (live) | `media_classifier_entities` | `[ner]` (+ `[media_servers]`/`[huggingface]`) | Entities pulled from media servers / HF datasets at runtime |
| AhocorasickNER (static) | `media_classifier_wordlists` / `media_classifier_ner_csv` | `[ner]` | Exact-match from a fixed word list or CSV |
| Keyword | `voc_match_func=` supplied | — | Delegates `.voc` lookups to the OCP pipeline plugin |
| Keyword (default) | — | — | Bundled per-language `.voc` files |

See [docs/backends.md](docs/backends.md) for config details, thresholds, and
when to use each.

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

Third-party classifiers register under the `opm.media.classifier` entry-point
group and load by name. `AbstractMediaClassifier` is the contract.

```toml
# in a 3rd-party package's pyproject.toml
[project.entry-points."opm.media.classifier"]
my-classifier = "my_pkg:MyMediaClassifier"
```

```python
clf = load_media_classifier(config={"media_classifier_plugin": "my-classifier"})
```

See [docs/external-plugins.md](docs/external-plugins.md).

## Training

Model training lives in the top-level `training/` directory and is **not** shipped
in the wheel. It is gated by the `[train]` extra and run as a module:

```bash
pip install ovos-media-classifier[train]
python -m training.build_dataset
```

See `training/README.md` for the full data-gathering and training workflow.

## Documentation

- [docs/index.md](docs/index.md) — table of contents
- [docs/taxonomy.md](docs/taxonomy.md) — mediavocab enforcement, intent→type/genre mapping, query-vs-content distinction
- [docs/backends.md](docs/backends.md) — every backend, its config keys, extras, and trade-offs
- [docs/content-filtering.md](docs/content-filtering.md) — detect-to-block content moderation
- [docs/external-plugins.md](docs/external-plugins.md) — registering 3rd-party classifiers
- [docs/stable-api.md](docs/stable-api.md) — the `AbstractMediaClassifier` contract and return types

See [benchmarks](docs/benchmarks/README.md) for accuracy/latency across backends.

## Credits

The original OCP training dataset was sponsored by **NeonGecko**. More recent
media-metadata datasets are published by **TigreGotico** on Hugging Face:
<https://huggingface.co/collections/TigreGotico/media-metadata>.
