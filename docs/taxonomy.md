# Taxonomy

This page covers the **leaf axis** — `mediavocab.MediaType` — and the genre tags
that ride alongside it. It is one of four orthogonal axes; the coarse axes
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

## Type + genres, not a per-type intent

The classifier models a play request along the **real axes** only: the
`OCPDomain` (play / control / not-ocp) × a `mediavocab.MediaType` + `mediavocab`
genre tags. There is **no** separate per-media-type intent enum. Distinctions
the public taxonomy treats as **genre** or **content-form** rather than as media
types — `anime`, `cartoon`, `asmr`, and the adult variants — are *not* their own
`MediaType`: they resolve to a base type and carry their nuance as a genre tag.

Backends emit a raw detection label (a `.voc` / model-head string such as
`"music"`, `"adult"`, `"anime"`) which resolves **directly** to `(MediaType,
genres)` via two string-keyed maps in `ovos_media_classifier/intents.py`:

- `LABEL_TO_MEDIA_TYPE: dict[str, MediaType]` — the base type.
- `LABEL_TO_GENRES: dict[str, list[str]]` — genre tags preserving the nuance the
  type drops. Only tags present in `mediavocab`'s `KNOWN_GENRES` are emitted, so
  genres are taxonomy-enforced too. The helper
  `genres_for_label(label) -> list[str]` reads this map.

Representative examples:

| raw label | → `MediaType` | → genres |
|---|---|---|
| `music` | `MUSIC` | — |
| `movie` | `MOVIE` | — |
| `tv_show` | `EPISODIC_SERIES` | — |
| `anime` | `EPISODIC_SERIES` | `["anime"]` |
| `cartoon` | `EPISODIC_SERIES` | `["animation"]` |
| `asmr` | `PROCEDURAL_AMBIENT` | `["asmr"]` |
| `ambient` | `PROCEDURAL_AMBIENT` | — |
| `audiobook` | `AUDIOBOOK` | — |
| `book` | `BOOK` | — |
| `comic` | `COMIC` | — |
| `playlist` | `PLAYLIST` | — |
| `sound_effect` | `SOUND_EFFECT` | — |
| `interactive_fiction` | `INTERACTIVE_FICTION` | — |
| `adult` | `MOVIE` | `["adult"]` |
| `adult_audio` | `MUSIC` | `["adult"]` |
| `hentai` | `EPISODIC_SERIES` | `["anime", "adult"]` |
| `news` | `RADIO` | — |
| `documentary` | `MOVIE` | — |

So a query like _"play some hentai"_ yields a `MediaType.EPISODIC_SERIES` from
`classify()` and `["anime", "adult"]` from `classify_genres()`. The `adult` tag is
what the [content filter](content-filtering.md) blocks on — the type alone never
carries that signal.

Every non-sentinel `mediavocab.MediaType` is reachable from a raw label, so the
whole taxonomy is exercised. A few distinctions are carried by **how a work is
consumed**: `audiobook` (play a narration → `AUDIOBOOK`) vs `book` (TTS-read a
text → `BOOK`); `anime` (watch → `EPISODIC_SERIES`) vs `comic`/manga (read →
`COMIC`); `asmr`/`ambient` (both `PROCEDURAL_AMBIENT`, the former genre-tagged).

## Domains and control intents

Above media types sits `OCPDomain`, the top-level question "does this command
target OCP at all?":

| `OCPDomain` | Meaning |
|---|---|
| `OCP_PLAY` | a playback request — the *what* is a `MediaType` + genres |
| `OCP_CONTROL` | a player-control request — refine with `OCPControlIntent` |
| `NOT_OCP` | unrelated to media playback |

`OCPControlIntent` enumerates control actions (`PLAY`, `PAUSE`, `STOP`, `NEXT`,
`SHUFFLE`, `SEEK_FORWARD`, …). Only a classifier with a dedicated control head
classifies these; the bundled keyword classifier derives the domain from the
media-type result and so reports `OCP_PLAY` / `NOT_OCP` only. A backend that
models control intents detects `OCP_CONTROL`.

## Entity labels

`OCPEntityLabel` is the NER vocabulary (`artist_name`, `movie_title`,
`tv_show_title`, `radio_station`, …). Each entity label maps directly to a
`mediavocab.MediaType` via `NER_LABEL_TO_MEDIA_TYPE` (and to genre tags via
`NER_LABEL_TO_GENRES`), so an entity hit ("play *Dune: Part Two*") resolves to a
media type — and an `adult_title` / `pornstar` hit still surfaces the `adult`
genre the content filter blocks on.

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
