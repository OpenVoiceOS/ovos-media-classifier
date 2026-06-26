# Taxonomy

This page covers the **leaf axis** — `mediavocab.MediaType` — and the internal
intent space that feeds it. It is one of four orthogonal axes; the coarse axes
(domain, modality, structure) and the rationale for splitting them are in
[classification-model.md](classification-model.md).

## mediavocab is the source of truth

This package does **not** define its own media taxonomy. The public output type
is `mediavocab.MediaType`, a string-Enum owned by the `mediavocab` package, and it
is re-exported here as `MediaType` purely for convenience:

```python
from ovos_media_classifier import MediaType   # this IS mediavocab.MediaType
from mediavocab import MediaType as MV
assert MediaType is MV
```

The classifier's `classify()` return value is mapped onto a `mediavocab.MediaType`
at the boundary, so the public contract enforces the shared vocabulary.

## Internal intent space vs. public types

Internally the classifier works against a richer, fine-grained label space —
`OCPPlayIntent` — that draws distinctions the public taxonomy treats as **genre** or
**content-form** rather than as media types. (Trained backends use this same
label space.) For example `anime`, `cartoon`, `asmr`, and the adult
variants are intents, but they are *not* their own `MediaType`: they collapse onto
a base type and carry their nuance as a genre tag.

Two maps perform this projection (`ovos_media_classifier/intents.py`):

- `PLAY_INTENT_TO_MEDIA_TYPE: dict[OCPPlayIntent, MediaType]` — the base type.
- `PLAY_INTENT_TO_GENRES: dict[OCPPlayIntent, list[str]]` — genre tags preserving
  the lost nuance. Only tags present in `mediavocab`'s `KNOWN_GENRES` are emitted,
  so genres are taxonomy-enforced too.

Representative examples:

| `OCPPlayIntent` | → `MediaType` | → genres |
|---|---|---|
| `MUSIC` | `MUSIC` | — |
| `MOVIE` | `MOVIE` | — |
| `TV_SHOW` | `EPISODIC_SERIES` | — |
| `ANIME` | `EPISODIC_SERIES` | `["anime"]` |
| `CARTOON` | `EPISODIC_SERIES` | `["animation"]` |
| `ASMR` | `PROCEDURAL_AMBIENT` | `["asmr"]` |
| `ADULT` | `MOVIE` | `["adult"]` |
| `ADULT_AUDIO` | `MUSIC` | `["adult"]` |
| `HENTAI` | `EPISODIC_SERIES` | `["anime", "adult"]` |
| `NEWS` | `RADIO` | — |
| `DOCUMENTARY` | `MOVIE` | — |

So a query like _"play some hentai"_ yields a `MediaType.EPISODIC_SERIES` from
`classify()` and `["anime", "adult"]` from `classify_genres()`. The `adult` tag is
what the [content filter](content-filtering.md) blocks on — the type alone never
carries that signal.

Classifiers that emit raw label strings instead of `OCPPlayIntent` use the
string-keyed equivalents `LABEL_TO_MEDIA_TYPE` and `LABEL_TO_GENRES`, plus the
helper `genres_for_label(label) -> list[str]`.

## Domains and control intents

Above media types sits `OCPDomain`, the top-level question "does this command
target OCP at all?":

| `OCPDomain` | Meaning |
|---|---|
| `OCP_PLAY` | a playback request — refine with `OCPPlayIntent` |
| `OCP_CONTROL` | a player-control request — refine with `OCPControlIntent` |
| `NOT_OCP` | unrelated to media playback |

`OCPControlIntent` enumerates control actions (`PLAY`, `PAUSE`, `STOP`, `NEXT`,
`SHUFFLE`, `SEEK_FORWARD`, …). Only a classifier with a dedicated control head
classifies these; the bundled keyword classifier derives the domain from the
media-type result and so reports `OCP_PLAY` / `NOT_OCP` only. A backend that
models control intents detects `OCP_CONTROL`.

## Entity labels

`OCPEntityLabel` is the NER vocabulary (`artist_name`, `movie_title`,
`tv_show_title`, `radio_station`, …). Each entity label maps to an `OCPPlayIntent`
via `NER_LABEL_TO_PLAY_INTENT`, so an entity hit ("play *Dune: Part Two*") resolves
to a media type.

The [NER backend](backends.md) (`AhocorasickMediaClassifier`) uses these labels to
tag the user's [entity lists](entity-lists.md). They are also exported so that an
out-of-tree classifier — and the skills that register content under these labels at
runtime — shares the same vocabulary.

## Query vs. content classification

This package classifies a **command** — the natural-language request a user
speaks. That is a different problem from classifying a piece of **content**:

| | This package | `mediavocab.text` |
|---|---|---|
| Input | a voice command, e.g. _"play the news"_ | catalog metadata, e.g. a video title + description |
| Question | what does the user want OVOS to do? | what kind of thing is this item? |
| Output | `(MediaType, conf)` + domain + genres | `ClassificationResult` for the item |
| Used by | the OCP pipeline (intent stage) | catalog tagging / ingestion |

They share the `mediavocab.MediaType` vocabulary but answer opposite questions:
intent-of-a-request here, type-of-an-item there. Do not substitute one for the
other.
