# Data sources

The training/benchmark dataset is built from **real media-metadata entities**
slot-filled into translatable templates. This page lists every source, the slot
label it feeds, and how the set is assembled. For the generator itself (columns,
rebuild command, how to add templates) see [dataset.md](dataset.md).

## Entity pools

Entities are ingested with `python -m training.ingest_entities` into
`data/entities/<label>.csv` (one column, `value`, deduplicated case-insensitively,
capped at 200 000 per label). Sources are read from the local `metadatarr`
scraper cache when present (the freshest, complete dump), otherwise from
HuggingFace `TigreGotico/<id>`.

The curated genre-specific archives are ingested **before** the bulk MusicBrainz
dump so they always reach the (capped) `artist_name` pool rather than being
crowded out.

### TigreGotico media-metadata collection → slot labels

| HuggingFace dataset | slot label(s) | source / license |
|---|---|---|
| `musicbrainz-artists` | `artist_name` | MusicBrainz (CC0) |
| `musicbrainz-releases` | `album_name`, `artist_name` | MusicBrainz (CC0) |
| `audiodb-artists` | `artist_name`, `music_genre`, `record_label` | TheAudioDB |
| `media-metadata-jazz-artists` | `artist_name`, `music_genre` | Jazz Music Archives |
| `media-metadata-progarchives-artists` | `artist_name`, `music_genre` | Prog Archives |
| `media-metadata-metal-archives` | `artist_name`, `music_genre`, `record_label` | Metal Archives |
| `media-metadata-classical-composers` | `artist_name` | classical-composer catalogue |
| `media-metadata-imdb-titles` | **primary title source** — split by `titleType`: `movie`/`tvMovie`→`movie_title`, `tvSeries`/`tvMiniSeries`→`tv_show_title`, `short`/`tvShort`→`short_film_title`, `videoGame`→`game_title`; `genres`→`content_genre` (+ `movie_genre`/`tv_genre`/`game_genre`), `startYear`→`release_year`/`release_decade`; `isAdult==1` routes to `adult_title` (never the clean pools). Also the **join key** (`imdb_id`) for the relational IMDb sources below | IMDb |
| `media-metadata-imdb-episodes` | joined (`series_id`→series title, episode `imdb_id`→episode title) into the `episodes` relation + `season_number`/`episode_number`/`episode_title` pools | IMDb |
| `media-metadata-imdb-technical-specs` + `media-metadata-imdb-bw-silent` | joined to the real title → the `bw_movie_title` / `silent_movie_title` pools tagged `black_and_white` / `silent` (the `bw_silent` relation) | IMDb |
| `media-metadata-imdb-ratings` | `num_votes` → **popularity-weighted** `movie_title` sampling (`_imdb_votes.csv`) | IMDb |
| `media-metadata-imdb-crew` (+ `…-imdb-credits` / `…-imdb-names` when present) | the `--credits` hook: resolves `(movie, director, writer, actor)` coherently; absent today → person slots fill independently | IMDb |
| `media-metadata-tvmaze-shows` | `tv_show_title`, `tv_genre`, `tv_network` | TVmaze |
| `media-metadata-anilist-anime` | `anime_title`, `anime_studio` (the `is_adult` / Hentai subset → `hentai_title`/`hentai_studio`) | AniList |
| `media-metadata-jikan-manga` | `comic_title`, `comic_genre` (manga is **read** → COMIC; the Hentai subset → `hentai_title`) | Jikan / MyAnimeList |
| `media-metadata-gutenberg-books` | `book_title`, `book_author`, `book_genre` (readable text → **BOOK**) | Project Gutenberg |
| `media-metadata-librivox-audiobooks` | `audiobook_title`, `audiobook_author`, **`audiobook_narrator`**, `audiobook_genre` (narrated → **AUDIOBOOK**) | LibriVox (public domain) |
| `media-metadata-openlibrary-books` | `book_title`, `book_author`, `book_genre`, `record_label` (publisher), `release_year` (readable text → **BOOK**) | Open Library |
| `media-metadata-steam-games` | `game_title`, `game_genre` | Steam |
| `media-metadata-radiobrowser-stations` | `radio_station`, `radio_genre` | Radio Browser (CC0) |
| `media-metadata-podcastindex-podcasts` | `podcast_title`, `podcast_host`, `podcast_genre` | Podcast Index |
| `media-metadata-listennotes-podcasts` | `podcast_title`, `podcast_host`, `podcast_genre` | Listen Notes |
| `media-metadata-wikidata-entities` | split by `entity_type` → `movie_title` (films), `tv_show_title`, `cartoon_title`, `anime_title`, `tv_channel`, `youtube_channel`, `radio_station`, `podcast_title`, `record_label`, `artist_name`, … | Wikidata (CC0) |
| `movie_actors` | `movie_actor` | TigreGotico movie-role set |
| `movie_directors` | `movie_director` | TigreGotico movie-role set |
| `movie_producers` | `movie_producer` | TigreGotico movie-role set |
| `movie_writers` | `movie_writer` | TigreGotico movie-role set |
| `movie_composers` | `movie_composer` | TigreGotico movie-role set |

**IMDb** (`media-metadata-imdb-titles`) is the authoritative title source: it is
far larger than the Wikidata split and carries the `titleType` distinction, so it
fills `movie_title` / `tv_show_title` / `short_film_title` / `game_title` from the
right rows and keeps `isAdult` titles out of the clean pools. It is listed first
in `SOURCE_SPECS` so its values populate the (capped) pools ahead of the smaller
Wikidata fallback. The `titleType`→slot map is `_IMDB_TYPE_TO_SLOT` in
`training/ingest_entities.py`.

The Wikidata `entity_type` split is the fallback for `movie_title` **real film
titles** (rather than fabricated strings) — see `WIKIDATA_TYPE_TO_LABEL` in
`training/ingest_entities.py` for the full type→label map.

### book vs audiobook vs comic (read vs play)

The book sources are routed by **how they are consumed**, matching the
`mediavocab` taxonomy:

* **LibriVox** is narrated audio → `audiobook_*` (`AUDIOBOOK`); its `readers`
  column populates `audiobook_narrator`.
* **Gutenberg / Open Library** are readable texts → `book_*` (`BOOK`, TTS-read).
* **Jikan manga** is read → `comic_*` (`COMIC`); **AniList anime** is watched →
  `anime_*` (`EPISODIC_SERIES`). The adult subset of either → `hentai_*`.

### Descriptive attribute pools

Beyond primary names, descriptive columns are mined into their own pools so
templates can phrase requests by attribute:

| pool | from | example values |
|---|---|---|
| `release_year` | TVmaze `premiered`, AniList `season_year`, Steam `release_date` | `1999`, `2014` |
| `release_decade` | derived from the above | `1990s`, `2010s` |
| `music_genre` / `tv_genre` / `game_genre` / `video_genre` | per-source genre columns | `jazz`, `drama`, `rpg` |
| `record_label` / `tv_network` / `anime_studio` | label / network / studio columns | — |
| `media_country` | curated seed (clean adjectives) | `French`, `Japanese` |

### Local `metadatarr` scraper cache

When `~/.cache/metadatarr/scrapers/<name>.jsonl` exists it is used in place of
the corresponding HuggingFace download (same schema, same emitter). This covers
MusicBrainz, TVmaze, AniList, Jikan, Gutenberg, LibriVox, Open Library, Steam,
Radio Browser, Podcast Index, Listen Notes, AudioDB, and Wikidata.

### Curated seed pools

A few slots have no metadata dump (provider / platform names). Small curated
lists ship in `training/seed_entities/<label>.csv` and are merged into the pools:
`news_provider`, `news_category`, `game_platform`, `asmr_artist`,
`adult_streaming_service`, `media_country`, plus the taxonomy-completion slots
`playlist_mood`, `playlist_activity`, `sound_name`, `ambient_sound`,
`comic_genre`.

### Slot aliases

A handful of template slots reuse a closely-related real pool (no separate
metadata source): `movie_genre`→`video_genre`, `trailer_title`/`bts_title`/
`silent_movie_title`/`bw_movie_title`→`movie_title`, `music_video_title`→
`tv_show_title`, `track_name`→`album_name`. The alias map lives in
`SLOT_ALIASES` (`training/build_dataset.py`).

## Content-filter data (adult, detect-to-block)

These sets exist **solely to generate adult-DETECTION training examples** that
the [content filter](content-filtering.md) blocks on. They are never used to
provide adult content.

| HuggingFace dataset | slot label(s) | role |
|---|---|---|
| `adult-metadata-stashdb-performers` | `pornstar` + `adult_eye_color`/`adult_hair_color`/`adult_ethnicity`/`adult_body_type`/`adult_marking` | performers + physical attributes |
| `adult-metadata-iafd-performers` | `pornstar`, `adult_title` | performers + filmography titles |
| `adult-metadata-iafd-titles` | `adult_title`, `adult_studio` | real adult film titles + studios |
| `adult-metadata-iafd-distributors` | `adult_studio` | adult studios / distributors |
| `adult-metadata-freeones-performers` | `pornstar` + `adult_country`/attributes | performers + attributes |
| `adult-metadata-boobpedia-performers` | `pornstar` + `adult_ethnicity`/`adult_hair_color`/attributes | performers + attributes |
| `adult-metadata-thenude-performers` | `pornstar` + attributes | performers + attributes |
| `adult-metadata-hanime` | `hentai_title`, `hentai_studio` | hanime.tv hentai catalogue |
| `adult-metadata-mal-hentai` | `hentai_title`, `hentai_studio` | MyAnimeList hentai |
| `adult-metadata-hentaisea` | `hentai_title` | hentaisea hentai catalogue |

These are **private** datasets — ingestion uses the HuggingFace token from the
environment. Performer rosters overlap across stashdb / iafd / freeones /
boobpedia / thenude, so they are **deduplicated case-insensitively into one
`pornstar` pool** rather than summed.

The dedicated hentai sets are the real corpus for `hentai_title` (the anilist /
jikan `is_adult` subset is merged in too); these are kept **out of the clean
`anime_title` / `comic_title` pools** so a normal "watch an anime" / "read a
manga" template never fills an adult title. A hentai row is labelled `hentai` →
`EPISODIC_SERIES` + `["anime", "adult"]`, so the content filter blocks it.

The physical-attribute pools (`adult_eye_color`, `adult_hair_color`,
`adult_ethnicity`, `adult_body_type`, `adult_marking`, `adult_country`) exist so
detection fires on a **description** (*"porn with red hair"*, *"some asian porn"*,
*"a performer with tattoos"*) and not only on a named performer — otherwise the
filter would be trivially evaded. They are detect-to-block training signals only.

Every adult template is labelled `adult` / `adult_audio` / `hentai`, which map to
a real `mediavocab.MediaType` plus the `adult` **genre** via `LABEL_TO_GENRES`.
That genre is the signal the content filter blocks on. The slice is a **deliberate
minority** (`--adult-cap`, default 7 000 rows) — enough for the model to learn
detection, far below a normal class so it never dominates training.

## How the training set is assembled

```
.intent templates (translatable locale resources:
                   ovos_media_classifier/locale/<lang>/dataset/<intent>.intent
                   + lead-in vocs ovos_media_classifier/locale/<lang>/<lead_*>.voc)
        │  ovos_spec_tools.expand()  — (a|b) alternations, [optional], <voc> refs
        ▼
slot-free samples with opaque {slot} placeholders
        │  slot-fill {slot} from the entity pools (real entities, sampled)
        ▼
labelled rows  + rich columns (keyword + NER-by-construction + axes + provenance)
        │  balance per media_type (adult kept a minority)
        ▼
stratified 80/10/10 train / validation / test  →  CSV + parquet + dataset card
```

One reproducible command runs the whole pipeline:
`python -m training.build_dataset`. Per-column meaning, the rebuild recipe, and
how to add `.intent` templates (via ovos-localize) are in [dataset.md](dataset.md).

See also [entity-lists.md](entity-lists.md) (the same labelled entity lists the
NER backend consumes at runtime) and [content-filtering.md](content-filtering.md).
