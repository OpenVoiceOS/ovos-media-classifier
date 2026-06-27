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

# 1. ingest the flat entity pools (→ data/entities/<label>.csv) AND the
#    coherent relational records (→ data/relational/<group>.jsonl)
python -m training.ingest_entities --relations

# 2. build the IMDb-join relations + popularity weights + bw/silent + episode
#    pools (→ data/relational/{movies,episodes,bw_silent}.jsonl, _imdb_votes.csv)
python -m training.imdb_relations

# 3. build the dataset (→ data/release/{full,train,validation,test}.{csv,parquet})
python -m training.build_dataset
```

`build_dataset` is the single entry point. Everything downstream of the entity
pools is deterministic for a fixed `--seed` (default 42).

Once a release is built, `python -m training.dataset_plots` renders the dataset
characterization plots (rows per media type, the slot×media-type heatmap, axis
distributions, entity-pool sizes) into [`docs/plots/dataset/`](plots/dataset/).

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
| `content_form_genres` | JSON list of sensitive / content-form tags (`adult`/`anime`/`animation`/`asmr`) — the content-filter axis |
| `content_genres` | JSON list of the real genre(s), constrained to `mediavocab.KNOWN_GENRES` |
| `content_form` | `mediavocab.ContentForm` (single) — `trailer` / `teaser` / `behind_scenes` / `excerpt` / `supplement` |
| `programme_format` | `mediavocab.ProgrammeFormat` (single) — `documentary` / `news` |
| `accessibility` | JSON list of `mediavocab.AccessibilityKind` (`subtitles` / `audio_description` / `sign_language` / `dubbed`) |
| `variant` | `mediavocab.VariantKind` (single) — `directors` / `extended` / `remastered` / `colorized` / `fanedit` |
| `picture_format` | JSON list of `mediavocab.PictureFormat` (`black_and_white` / `silent` / `3d`) |
| `year` | the release year (feeds `Signals.year`) |
| `explicitness` | `clean` / `adult` |

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

## Templates: `.intent` + `.voc` translatable locale resources

Templates are **hand-authored, translatable OVOS-INTENT-1 files** under the
package `locale/`, not a hard-coded Python list — so ovos-localize discovers and
translates them like any other locale resource, and the translations stick (no
generator regenerates or overwrites them):

```
ovos_media_classifier/locale/
  <lang>/<lead_*>.voc             shared lead-ins (lead_play_audio, lead_watch, …)
  <lang>/dataset/<intent>.intent  one file per media label
```

The `dataset/` subdir keeps the dataset templates separate from the runtime OCP
control intents (`play.intent`, `featured.intent`, …) at the locale root;
`ovos_spec_tools` resolves resources recursively under `<lang>/`, so a
`<lead_play_audio>` reference inside `dataset/music.intent` still expands from
`<lang>/lead_play_audio.voc` in the parent language directory. `build_dataset`
reads them via the `ovos_spec_tools` locale helpers (`find_lang_dir`), the same
way the runtime keyword classifier resolves its `.voc` files.

Each `.intent` line is expanded by `ovos_spec_tools.expand(template, vocabularies)`,
which resolves:

* `<lead_play_audio>` — a `.voc` reference, expands to each member phrase;
* `(a|b|c)` — inline alternation;
* `[word]` — optional word;
* `{slot}` — left **opaque** for the slot-filler (an `OCPEntityLabel` name).

The lead-in `<…>` references are the componential request openers; the inline
alternations are the slot-pattern variants. `build_dataset` then fills each
`{slot}` from the real entity pools.

### Coherent (relational) slot-filling

When a template fills **several slots of one domain** — `{album_name} by
{artist_name}`, `episode {n} of {tv_show}`, `{audiobook_title} by
{audiobook_author}` — filling each slot independently would yield an incoherent
sentence (a real album credited to the wrong real artist). `build_dataset`
instead draws those slots from **one real record** of a `RelationalGroup`:

| group | coherent fields | source |
|---|---|---|
| `music` | album ↔ artist ↔ year | musicbrainz-releases |
| `tv` | show ↔ network ↔ genre ↔ year | tvmaze-shows |
| `anime` | title ↔ studio ↔ genre ↔ year | anilist-anime |
| `audiobook` | title ↔ author ↔ narrator ↔ genre | librivox |
| `book` | title ↔ author ↔ genre ↔ year | openlibrary |
| `podcast` | title ↔ host ↔ genre | podcastindex |
| `movies` | title ↔ genre ↔ year (↔ director/writer/actor *) | IMDb titles |
| `episodes` | series ↔ season ↔ episode ↔ episode_title | IMDb episodes |

`*` movie **person** slots are coherent only when the `--credits` hook resolves
names (a `media-metadata-imdb-credits` or `…-imdb-names` dataset); otherwise the
person slots fill **independently** from the flat `movie_director` /
`movie_writer` / `movie_actor` pools (the hook logs which path it took and
upgrades automatically on the next run once that dataset lands). **No credits
dataset exists yet**, so the current build runs on the **fallback** path —
`movies.jsonl` carries no person fields and the movie director/writer/actor
slots fill independently. Movie person-coherence is therefore *pending the
credits dataset*; re-running `training.imdb_relations` once it lands upgrades
those slots with no other change. **Single-slot templates and confusable slots
stay independent by design** — coherence is only coordinated when a template
uses ≥ 2 slots of one group.

Two IMDb signals further shape sampling:

* **popularity weighting** — `movie_title` is sampled `∝ log1p(num_votes)` (from
  IMDb ratings) with a floor of 1.0, so popular titles dominate the realistic
  head while the long tail still appears.
* **real qualifier titles** — the `bw_movie_title` / `silent_movie_title` pools
  are real black-and-white / silent films (IMDb technical-specs joined to the
  title), so a qualifier template fills a genuine title that *is* the qualifier.

The IMDb relations + the `_imdb_votes.csv` weight table are built by
`python -m training.imdb_relations`; the other groups by
`python -m training.ingest_entities --relations`.

### To add or translate templates

These files are the **source of truth** — edit them directly (or translate them
through ovos-localize); nothing regenerates them.

* **New phrasings** — add lines to
  `ovos_media_classifier/locale/<lang>/dataset/<intent>.intent`. Optional
  decorations are themselves translatable grammar, e.g.
  `[hey|ok|please] <lead_play_audio> {artist_name} [please|for me]`.
* **New language** — add `ovos_media_classifier/locale/<lang>/dataset/` with
  translated `.intent` files and `ovos_media_classifier/locale/<lang>/<lead_*>.voc`
  translated lead-in `.voc` files.

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

## Taxonomy coverage

Every non-sentinel `mediavocab.MediaType` is exercised by at least one intent, so
the dataset covers the whole taxonomy — including the read-vs-play distinctions
(`book` vs `audiobook`, `comic`/manga vs `anime`) and the rarer leaves
`playlist`, `sound_effect`, `interactive_fiction`, and non-asmr `ambient`
(`PROCEDURAL_AMBIENT`). See [taxonomy.md](taxonomy.md).

## Content-filter slice

The `adult` / `adult_audio` / `hentai` intents are slot-filled with **real adult
performer / title / hentai entities** so the slice is diverse, then kept a
deliberate minority (`--adult-cap`). Every adult row carries the `adult` genre via
`LABEL_TO_GENRES` (hentai → `["anime", "adult"]`); the
[content filter](content-filtering.md) blocks on that genre, including descriptive
forms (*"an adult video with a brunette performer"*). The entities exist for
detect-to-block only — never for provision. See
[data-sources.md](data-sources.md#content-filter-data-adult-detect-to-block).
