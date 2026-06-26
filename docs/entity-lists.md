# Entity lists

An **entity list** is a mapping of `label → list of strings`:

```
artist_name  → ["Radiohead", "Björk", "Aphex Twin", …]
movie_title  → ["Inception", "The Dark Knight", …]
radio_station → ["BBC Radio 1", "KEXP", …]
```

It is the user's *real* media — the artists, titles and stations they actually
own or stream — captured as plain strings under an
[`OCPEntityLabel`](taxonomy.md). The store that holds them is
[`EntitiesContainer`](#api).

Entity lists feed the **NER backend** —
[`AhocorasickMediaClassifier`](backends.md), which loads the lists into an
Aho-Corasick automaton for fast exact substring matching ("play *Inception*" →
`MOVIE`). They are also the natural input for any classifier that uses the user's
known entities as features (a known `movie_title` token in the utterance is a
strong signal for the `MOVIE` axis), so the same lists serve more than one
strategy — build them once.

## Where entity lists come from

There are two channels.

> The same labelled entity lists also seed **training**: the
> [dataset generator](dataset.md) slot-fills templates from large real-entity
> pools ingested from the [TigreGotico media-metadata collection](data-sources.md).
> Build them once; they serve both runtime NER and training.

### 1. Provided at runtime

The OCP pipeline registers the user's media as it discovers it — a skill
announcing its content, a background media-server sync. This is the `add` /
`add_many` path, and updates are reflected in classification **immediately**,
with no rebuild step (the automaton is shared by reference):

```python
from ovos_media_classifier.entities import EntitiesContainer
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

container = EntitiesContainer()
clf = AhocorasickMediaClassifier.from_container(container)

# the pipeline discovers a skill's content and registers it live:
container.add("artist_name", "Radiohead")
container.add_many([
    ("movie_title", "Inception"),
    ("movie_title", "The Dark Knight"),
])
# clf.classify("play radiohead", "en-us") now returns MUSIC — no rebuild
```

> `add`/`add_many` are the path the OCP pipeline uses to register entities it
> discovers at runtime. Everything below is the *config* channel.

### 2. Provided via config as source specs

A **source spec** is one of:

| Spec | Example | Loaded by |
|---|---|---|
| `.csv` path | `"/data/library.csv"` | `load_csv` |
| `.tsv` path | `"/data/library.tsv"` | `load_tsv` |
| `.jsonl` path | `"/data/aliases.jsonl"` | `load_jsonl` |
| HuggingFace dict | `{"dataset": "TigreGotico/ocp-entities"}` | `load_huggingface` |
| inline dict | `{"artist_name": ["Radiohead"]}` | added directly |
| media-server dict | `{"radarr": {"url": …, "api_key": …}}` | `load_radarr`, … |

`load_source(spec)` dispatches a single spec by shape; `from_sources([...])`
(classmethod) and `load_lists([...])` (instance method) take a list of them:

```python
from ovos_media_classifier.entities import EntitiesContainer

container = EntitiesContainer.from_sources([
    "/data/library.csv",                      # .csv  → load_csv
    "/data/extra.tsv",                        # .tsv  → load_tsv
    "/data/aliases.jsonl",                    # .jsonl → load_jsonl
    {"artist_name": ["Radiohead", "Björk"]},  # inline {label: [values]}
    {"dataset": "TigreGotico/ocp-entities"},  # HuggingFace dataset
    {"radarr": {"url": "http://localhost:7878", "api_key": "…"}},  # media server
])
```

A bad spec (missing file, dead media server) is logged and skipped — one
failure does not abort the rest of the list.

> Loading lists from **files and inline dicts needs no optional dependencies**.
> The HuggingFace spec needs the `huggingface` extra (`datasets`), and
> media-server specs need the `media_servers` extra (`requests`). The
> Aho-Corasick matcher itself needs the `ner` extra — but list *loading* is
> independent of the matcher.

#### File formats

**CSV / TSV** accept either named columns (`entity`, `label`, optional
`source`) or a plain two-column `label,value` form. TSV is handy when entity
strings themselves contain commas.

```csv
entity,label,source
Inception,movie_title,radarr
Radiohead,artist_name,manual
```

**JSONL** — one JSON object per line, two shapes (mixable in one file):

```jsonl
{"label": "movie_title", "entity": "Inception"}
{"artist_name": ["Radiohead", "Björk"]}
```

- *per-entity rows* carry `label` + `entity` (`value` is also accepted as the
  entity key);
- *list rows* are `{label: [values]}` (a single string instead of a list also
  works). A row carrying the reserved `label`/`entity` keys is always treated as
  a per-entity row.

## Configuration

The factory selects the NER backend when any of `media_classifier_entities`,
`media_classifier_wordlists`, or `media_classifier_ner_csv` is set (see
[backends.md](backends.md)). The preferred, source-agnostic form passes an
`entity_lists` list under `media_classifier_entities`:

```json
{
  "media_classifier_entities": {
    "entity_lists": [
      "/data/library.csv",
      "/data/aliases.jsonl",
      {"artist_name": ["Radiohead", "Björk"]},
      {"dataset": "TigreGotico/ocp-entities"},
      {"radarr": {"url": "http://localhost:7878", "api_key": "…"}}
    ]
  }
}
```

The structured keys (`csv`, `wordlists`, `huggingface`, and the per-server
`radarr`/`sonarr`/`lidarr`/`jellyfin`/`music_assistant` keys) are also accepted and
merged in addition to `entity_lists`.

## Performance / memory tradeoff

Entity lists are a **deliberate, bounded choice**. The Aho-Corasick automaton
holds every entity string in memory, and every entity added widens the matcher:

- **the more entities loaded, the slower** the per-utterance tagging;
- **the more entities loaded, the larger** the memory footprint.

So load the user's *actual* library (typically a few thousand titles) — not an
open-ended public catalogue. A bloated entity vocabulary dilutes the signal
rather than sharpening it. Prefer a handful of focused lists over one giant dump.

## API

| Method | Purpose |
|---|---|
| `add(label, entity)` / `add_many(pairs)` | runtime registration (the OCP pipeline path) |
| `load_csv(path)` / `load_tsv(path)` / `load_jsonl(path)` | load one entity-list file |
| `load_huggingface(dataset_name, …)` | load from a HuggingFace dataset |
| `load_source(spec)` | dispatch a single source spec by shape |
| `load_lists(specs)` | load a list of source specs (instance method) |
| `EntitiesContainer.from_sources(specs)` | build a container from a list of specs |
| `EntitiesContainer.from_config(cfg)` | build from a config dict (`entity_lists` + the structured keys) |
| `wordlists` / `stats` | `{label: [values]}` snapshot / per-label counts |

See also: [backends.md](backends.md) (the NER backend that consumes these
lists), [classification-model.md](classification-model.md) (how entity hits feed
the multi-axis result), and [taxonomy.md](taxonomy.md) (the `OCPEntityLabel`
label space).
