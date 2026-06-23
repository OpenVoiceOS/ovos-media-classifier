# Backends

This release ships a **single** classifier: the bundled-`.voc`
**keyword classifier**. It implements [`AbstractMediaClassifier`](stable-api.md) and
is what `load_media_classifier()` returns with no config — the minimum required for
OCP to be functional, with zero ML dependencies.

`load_media_classifier(config)` selects between exactly two things:

| Order | Backend | Selected by | Notes |
|---|---|---|---|
| 0 | external plugin | `config["media_classifier_plugin"]` | a 3rd-party / future classifier registered under `opm.media.classifier`; falls back to keyword on any load error |
| 1 | keyword (default) | nothing else set | the bundled `.voc` classifier described below |

Richer strategies (trained ONNX models, live NER from media servers, …) are **not**
in the core package. They land as independent, separately-reviewed plugins through
the `opm.media.classifier` mechanism — see
[writing a classifier plugin](#writing-a-classifier-plugin) below and
[external-plugins.md](external-plugins.md).

> The NER backend (and the future guided-embeddings strategy) classify by
> matching the user's own media — their artists, titles and stations, captured as
> **entity lists** (`label → list of strings`). That shared machinery, its source
> specs (`.csv` / `.tsv` / `.jsonl` / HuggingFace / inline / runtime) and the
> perf/memory tradeoff of loading more entities are documented in
> [entity-lists.md](entity-lists.md).

## Keyword classifier (default, zero-dep)

```python
clf = load_media_classifier()                                # bundled locale
clf = load_media_classifier(voc_match_func=self.voc_match)   # pipeline mode
```

Substring-matches the query against bundled per-language `.voc` files
(`ovos_media_classifier/locale/<lang>/<Vocab>.voc`). No ML dependencies, no model
files, fully offline. In *pipeline mode* the OCP pipeline plugin owns the `.voc`
files and passes its `voc_match` method as `voc_match_func`; in *standalone mode*
the classifier reads the bundled files directly. Matching runs in a fixed priority
order (e.g. `MusicVideoKeyword` before `MusicKeyword`) and surfaces genre tags
(`anime`, `adult`, …) via `classify_genres()`.

It predicts only the leaf `MediaType` and **derives** the coarse axes — `domain`,
`playback_type`, `structure` — from it (cheap and exact for the type defaults), so
`classify_full()` returns a complete multi-axis result with no extra model. A
trained plugin predicts each axis with its own head instead — see
[classification-model.md](classification-model.md#4-how-the-axes-are-produced).

Because it derives the domain from the media-type result, the keyword classifier
reports `OCP_PLAY` / `NOT_OCP` but does not detect `OCP_CONTROL` — a dedicated
control head is something a future plugin can add by overriding `classify_domain()`.

**Use when:** always — it is the default and the only classifier in this release.
Use *pipeline mode* when you are inside the OCP pipeline and already have a
`voc_match` function; otherwise use the no-arg bundled mode.

Languages bundled: `ca-es`, `da-dk`, `de-de`, `en-us`, `es-es`, `eu-es`, `fr-fr`,
`gl-es`, `it-it`, `nl-nl`, `pl-pl`, `pt-br`, `pt-pt`.

## Writing a classifier plugin

To add a richer strategy — a trained model, NER over a user's library, a control
head — implement [`AbstractMediaClassifier`](stable-api.md) and register it under
the `opm.media.classifier` entry-point group. The host selects it by name via
`config["media_classifier_plugin"]`, and `load_media_classifier()` falls back to the
keyword classifier if it fails to load. The full recipe — registering, loading by
name, and which methods to override — is in
[external-plugins.md](external-plugins.md).
