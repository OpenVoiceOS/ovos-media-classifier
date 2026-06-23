# ovos-media-classifier

Media-type **command/intent** classification for [OCP (OVOS Common Playback)](https://github.com/OpenVoiceOS/ovos-core). Given a spoken command such as _"play some music"_ it decides three things: whether the command targets OCP at all, which OCP **domain** it belongs to (play / control / not-OCP), and — for play requests — which `mediavocab.MediaType` and genre tags the user is asking for. An always-on content filter lets OVOS block sensitive (e.g. adult) requests by default.

This initial release ships **one** classifier: the bundled-`.voc` **KeywordMediaClassifier** — zero ML dependencies, no model files, fully offline. It is the minimum required for OCP to be functional. Richer strategies (trained ONNX models, NER from media servers, …) are **not** in this release; they land later as independent, separately-reviewed plugins through the `opm.media.classifier` mechanism (see [External plugins](#external-plugins-opm)).

> **Command classification, not catalog classification.** This package classifies a *voice command* ("play the news"). It is distinct from `mediavocab.text.classify`, which classifies a piece of *catalog content* (a video's title/description) into a media type. Use this package in the OCP pipeline; use `mediavocab.text` when tagging library items.

## Install

```bash
pip install ovos-media-classifier
```

The only runtime dependencies are `ovos-utils` and `mediavocab`. There are no ML
extras — the keyword classifier reads the bundled `.voc` files directly.

## Quickstart

`load_media_classifier()` with no config returns the bundled `.voc` keyword
classifier — no ML dependencies, no model files, works offline.

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

## Future strategies (plugins, not in this release)

Trained ONNX models, live NER against a user's media servers, and other richer
strategies are **not** part of this release. They arrive as independent,
separately-reviewed additions that register under the `opm.media.classifier`
entry-point group and load by name — the same mechanism any third party uses (see
[External plugins](#external-plugins-opm)). The core package stays lean: one
zero-dependency keyword classifier plus the plugin contract.

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
