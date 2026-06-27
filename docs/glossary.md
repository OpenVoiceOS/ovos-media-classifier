# Glossary & core concepts

New to `ovos-media-classifier`? Read this once and the rest of the docs click
into place. It defines every acronym and — more importantly — the handful of
concepts people most often mix up.

## The 30-second mental model

A user says *"play some music"*. This package answers three questions about that
sentence, fast and offline:

1. **Is this a media request at all?** (the *domain* — play / control / not-OCP)
2. **What kind of media is wanted?** (the `mediavocab.MediaType` leaf, plus the
   coarse axes — audio vs video, single vs episodic, …)
3. **Should it be allowed?** (the *content filter* — block `adult` by default)

The output is a `(MediaType, confidence)` pair — and, in full form, a
[`mediavocab.Signals`](https://github.com/TigreGotico/mediavocab) object — that
the OCP pipeline hands to a media provider to actually search and play. This
package does the **NLP**; it does not search catalogs or play audio.

## The one distinction to get right: command vs content

This is the #1 point of confusion. Two different problems share the same
`mediavocab.MediaType` vocabulary but answer opposite questions.

| | `ovos-media-classifier` (this package) | `mediavocab.text.classify` |
|---|---|---|
| Input | a **voice command** — *"play the news"* | a **catalog item** — a title + description |
| Question | what does the user want OVOS to do? | what kind of thing is this item? |
| Output | `(MediaType, conf)` + domain + genres | a classification for the item |
| Used by | the OCP pipeline (intent stage) | catalog tagging / ingestion |

Classify a *request* here; classify an *item* there. Do not substitute one for
the other. See [taxonomy.md](taxonomy.md#query-vs-content-classification).

## Terms

| Term | Meaning |
|------|---------|
| **OCP** | *OpenVoiceOS Common Play* — the framework for voice-driven media ("play X"). The **OCP pipeline** is the OVOS stage that classifies a "play …" utterance and asks providers to search; this package is its classification step. |
| **OPM** | *ovos-plugin-manager* — discovers plugins via Python entry points. External classifiers register under the `opm.media.classifier` group. |
| **MediaType** | A [`mediavocab.MediaType`](https://github.com/TigreGotico/mediavocab) — *what kind* of content (music, movie, podcast, radio, audiobook, game, …). The public output type; the "leaf" axis. |
| **PlaybackType** | A `mediavocab.PlaybackType` — the *modality* axis: `audio` / `video` / `paged` / `interactive` (plus `unknown`). What surface renders it. |
| **Structure** | The *temporal-shape* axis: `single` / `episodic` / `continuous` / `collection` (plus `unknown`). Orthogonal to modality — a podcast is `audio` + `episodic`. Defined in this package (`axes.py`). |
| **domain** | The top-level question, an `OCPDomain`: `ocp_play` (play media) / `ocp_control` (control the player) / `not_ocp` (unrelated). |
| **control** | A player-transport request — pause / stop / next / shuffle / seek. Enumerated by `OCPControlIntent`. Surfaced only by a backend with a dedicated control head. |
| **genre / tag** | A `mediavocab` genre tag (`anime`, `animation`, `asmr`, `adult`, …) attached to a result. Orthogonal to `MediaType` — *"play some hentai"* is an `episodic_series` tagged `["anime", "adult"]`. The `adult` tag is what the content filter blocks on. |
| **content_genres (the head)** | The real genre(s) (`rock` / `action` / `horror` / …), constrained to `mediavocab.KNOWN_GENRES`. `classify_genres()` returns them. Distinct from the content-form `genres` above (`anime`/`adult`/…) that the filter reads. A release year is not modelled as a genre — it feeds `Signals.year` directly. |
| **content_form** | The experiential kind, a `mediavocab.ContentForm` (`trailer` / `teaser` / `behind_scenes` / `excerpt` / `supplement` / …) — supplementary content that is not a media type. `classify_content_form()`; surfaced on `Signals.content_form`. |
| **programme_format** | The structural format, a `mediavocab.ProgrammeFormat` (`documentary` / `news` / `concert` / `stand_up` / `sports` / …). `classify_programme_format()`. |
| **accessibility** | Requested accessibility assets, `mediavocab.AccessibilityKind` (`subtitles` / `audio_description` / `sign_language` / `dubbed` / …). Multi-label, `classify_accessibility()`; surfaced on `Signals.accessibility`. |
| **variant** | The work-level cut, a `mediavocab.VariantKind` (`directors` / `extended` / `remastered` / `colorized` / `fanedit` / …). `classify_variant()`; surfaced on `Signals.variant_kind`. |
| **picture_format** | The presentation attribute (T6), a `mediavocab.PictureFormat` (`black_and_white` / `silent` / `3d`). Multi-label, `classify_picture_format()`; surfaced on `Signals.picture_format` (single-valued, first wins). |
| **content filter** | A **detect-to-block** moderation / parental-control layer (`ContentFilter`). It does not provide content — it recognises sensitive requests so OVOS can refuse them. `adult` is blocked by default. |
| **command vs content classification** | Classifying a spoken *request* (this package) versus classifying a catalog *item* (`mediavocab.text.classify`). See above. |
| **axis (multi-axis model)** | One coordinate of the classification — domain, modality, structure, or the leaf `MediaType`. The full result is a point in this small product space, not one label. See [classification-model.md](classification-model.md). |
| **MediaClassification** | The dataclass holding all axes at once (`media_type`, `playback_type`, `structure`, `domain`, `genres`, `confidence`), returned by `classify_full()`. |
| **Signals** | A [`mediavocab.Signals`](https://github.com/TigreGotico/mediavocab) — the parsed request a media provider receives. `to_signals()` packages the classification (plus the raw title) into one. |
| **raw detection label** | The string a backend emits before resolution (e.g. `"music"`, `"adult"`, `"anime"` from a `.voc` file or a model head). It maps **directly** to `(MediaType, genres)` via `LABEL_TO_MEDIA_TYPE` / `LABEL_TO_GENRES` — there is no per-media-type intent layer. |
| **OCPEntityLabel** | The **NER vocabulary** (`artist_name`, `movie_title`, `radio_station`, …). Each label maps directly to a `MediaType` via `NER_LABEL_TO_MEDIA_TYPE` (and to genres via `NER_LABEL_TO_GENRES`), so an entity hit resolves to a media type. |
| **entity list** | A `label → list of strings` mapping of the user's *real* media (their artists, titles, stations). Feeds the NER backend. See [entity-lists.md](entity-lists.md). |
| **keyword feature slot** | A named entity category (`artist_name`, `movie_title`, …) bound to a point on the taxonomy. Filled at runtime from the user's library so available media biases prediction. See [contextual-classification.md](contextual-classification.md). |
| **`.voc` file** | A per-language list of keyword phrases (`locale/<lang>/<Vocab>.voc`). The keyword backend matches against these. |
| **ONNX** | The Open Neural Network Exchange runtime. The trained backend loads **one ONNX head per axis** (`domain` / `media_type` / `playback_type` / `structure` / `explicitness` / `content_form_genres` / `content_genres` / `content_form` / `programme_format` / `accessibility` / `variant`) plus `numpy` from a self-describing model bundle directory. Needs the `[onnx]` extra. |
| **NER** | *Named-entity recognition*. The NER backend matches the user's entity lists with an Aho-Corasick automaton — *"play Inception"* → `MOVIE` because *Inception* is in the library. Needs the `[ner]` extra. |
| **OPM entry-point group** | `opm.media.classifier` — the group an external classifier package registers under so `load_media_classifier` can load it by name. |

## The backends at a glance

`load_media_classifier(config)` returns one classifier. All implement the same
`AbstractMediaClassifier` contract, so callers never care which one ran.

| Backend | One-liner | Install |
|---------|-----------|---------|
| **keyword** | `.voc` phrase matching; zero ML deps; the offline default | core |
| **NER** | Aho-Corasick exact match over the user's entity lists | `[ner]` |
| **ONNX** | the trained multi-task per-axis heads, loaded from a self-describing bundle | `[onnx]` |
| **external** | any classifier registered under `opm.media.classifier` | a plugin |

See [backends.md](backends.md) for selection and config keys.

## Which doc do I want?

- *I just want it running* → [index.md](index.md) quickstart, then [stable-api.md](stable-api.md)
- *I want to pick / tune a backend* → [backends.md](backends.md)
- *I want to block content* → [content-filtering.md](content-filtering.md)
- *I want my library to guide prediction* → [contextual-classification.md](contextual-classification.md) · [entity-lists.md](entity-lists.md)
- *I'm writing an external classifier* → [external-plugins.md](external-plugins.md)
- *I want to understand the model* → [classification-model.md](classification-model.md) · [taxonomy.md](taxonomy.md)
- *I'm measuring accuracy* → [benchmarks](../benchmarks/README.md)
