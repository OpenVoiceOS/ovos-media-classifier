# The dataset & its generator

`TigreGotico/ocp-media-intents` is the canonical training/benchmark dataset for
OCP media-command classification. It is **rebuilt on demand from source** with a
single reproducible command, and ships every feature precomputed so it trains a
classifier — or a slot-filler / NER / entity-linker — directly, with no
extraction step.

For where the entities come from, see [data-sources.md](data-sources.md).

## Rebuild on demand

```bash
pip install -e ".[train]"

# 1. ingest the entity pools (→ data/entities/<label>.csv)
python -m training.ingest_entities

# 2. build the dataset (→ data/release/{full,train,validation,test}.{csv,parquet})
python -m training.build_dataset
```

`build_dataset` is the single entry point. Everything downstream of the entity
pools is deterministic for a fixed `--seed` (default 42).

Useful options:

| flag | meaning | default |
|---|---|---|
| `--langs` | languages to build | the 7 core langs |
| `--fills-per-template` | real-entity fills per slotted sample | 6 |
| `--target-per-type` | cap per (non-adult) `media_type` | 20000 |
| `--adult-cap` | total adult rows (the learnable minority) | 7000 |
| `--seed` | RNG seed (reproducibility) | 42 |
| `--push --repo … [--private]` | publish a `DatasetDict` + card to the Hub | off |

## Columns

Each row is one natural-language command. Columns are grouped:

### Core labels

| column | meaning |
|---|---|
| `sentence` | the realised utterance (slots filled with real entities) |
| `lang` | BCP-47 language code |
| `domain` | `ocp_play` (template rows are play requests) |
| `intent` | raw media label — a `LABEL_TO_MEDIA_TYPE` key (`music`, `movie`, `adult`, …) |
| `media_type` | canonical `mediavocab.MediaType` (the leaf axis) |
| `genres` | JSON list of `mediavocab` genre tags — carries the `adult` content-filter signal |
| `playback_type` | modality axis (`audio` / `video` / `paged` / `interactive`) |
| `structure` | structure axis (`single` / `episodic` / `continuous` / `collection`) |
| `binary_label` | `ocp` / `not_ocp` |

### Keyword feature columns (one per `CategoricalFeatureExtractor` feature)

`kw_*`, `verb_*`, `mod_*`, `fmt_*` — each a 0/1 flag computed **on the realised
sentence** by the same `CategoricalFeatureExtractor` the runtime uses (the
bundled `.voc` files). A model trains on the exact features it will see at
inference, so there is no extraction step. The full menu and column order live in
`ovos_media_classifier.features._KEYWORD_VOCABS`.

### NER features by construction (ground truth)

`ner_<entity_label>` — one 0/1 column per `OCPEntityLabel`, set to 1 when that
`{entity_label}` slot was filled in this row. These are **exact, not predicted**
(we know which slot we filled), so they are perfect supervision. `slot_values`
maps each filled slot to its concrete entity string. Together they make the set
double as NER / slot-filling / entity-linking training data.

### Provenance

| column | meaning |
|---|---|
| `template_id` | `<lang>:<intent>:<n>` — the template the row came from |
| `template` | the raw `.intent` line |
| `n_slots` | number of slots filled |
| `entity_labels` | JSON list of the slot labels filled (mirrors the `ner_*` flags) |

`genres`, `entity_labels` and `slot_values` are JSON strings.

### Sample row

```
sentence       : "put on the soundtrack to The Dark Knight"
lang           : en-us
domain         : ocp_play
intent         : music
media_type     : music
genres         : []
playback_type  : audio
structure      : single
binary_label   : ocp
kw_soundtrack  : 1
kw_music       : 1
ner_movie_title: 1          ← a movie title is present …
…              : …
media_type     : music      ← … but the correct type is MUSIC (the soundtrack)
slot_values    : {"movie_title": "The Dark Knight"}
template       : "<lead_play_audio> the soundtrack to {movie_title}"
```

## Templates: `.intent` + `.voc`, managed via ovos-localize

Templates are **translatable OVOS-INTENT-1 files**, not a hard-coded list, so the
user manages and translates them through ovos-localize:

```
training/templates/
  vocab/<lang>/<lead_*>.voc       shared lead-ins (lead_play_audio, lead_watch, …)
  <lang>/<intent>.intent          one file per media label
```

Each `.intent` line is expanded by `ovos_spec_tools.expand(template, vocabularies)`,
which resolves:

* `<lead_play_audio>` — a `.voc` reference, expands to each member phrase;
* `(a|b|c)` — inline alternation;
* `[word]` — optional word;
* `{slot}` — left **opaque** for the slot-filler (an `OCPEntityLabel` name).

The lead-in `<…>` references are the componential request openers; the inline
alternations are the slot-pattern variants. `build_dataset` then fills each
`{slot}` from the real entity pools.

### To add or translate templates

* **New phrasings** — add lines to `training/templates/<lang>/<intent>.intent`.
* **New language** — add `training/templates/<lang>/` with translated `.intent`
  files and a `vocab/<lang>/` of translated lead-in `.voc` files.
* **Regenerate the bundled English set** from the authoring source —
  `python -m training.author_templates`.

`build_dataset` picks up any new files with no code change.

### Entity-role richness & confusables

Beyond plain titles, templates exercise the **full entity-role space** — for
films: `directed by {movie_director}`, `starring {movie_actor}`,
`produced by {movie_producer}`, `written by {movie_writer}`,
`scored by {movie_composer}`; analogously per type (music: `featuring …`,
`on {record_label}`; tv: `on {tv_network}`; podcast: `hosted by {podcast_host}`;
audiobook: `narrated by {audiobook_narrator}`; anime: `by {anime_studio}`).

Templates also include **confusables**: cross-type lines where a foreign entity
of type X appears but the correct `media_type` is Y, driven by a context word —
e.g. *"the {movie_title} soundtrack"* is **music**, *"the {movie_title} trailer"*
is a **trailer**, *"a documentary about {artist_name}"* is a **documentary**,
*"the audiobook of {movie_title}"* is an **audiobook**. The foreign slot is still
filled (so `ner_movie_title=1`), but the row is labelled with the correct type —
that mismatch is the hard signal that teaches disambiguation. The disambiguating
context words are themselves keyword features (`kw_soundtrack`, `kw_trailer`,
`kw_bts`, `kw_documentary`, `kw_music_video`, `attr_topic`, `kw_audiobook`), so
the keyword/NER model can learn the override.

## Content-filter slice

The `adult` / `adult_audio` / `hentai` intents are slot-filled with **real adult
performer / title entities** so the slice is diverse, then kept a deliberate
minority (`--adult-cap`). Every adult row carries the `adult` genre via
`LABEL_TO_GENRES`; the [content filter](content-filtering.md) blocks on that
genre. The entities exist for detect-to-block only — never for provision. See
[data-sources.md](data-sources.md#content-filter-data-adult-detect-to-block).
