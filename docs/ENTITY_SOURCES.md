
# Entity Pool Sources — ovos-media-classifier

## Overview

`scripts/ocp_entities.csv` is the primary entity pool used by
`AhocorasickMediaClassifier` and the categorical feature extractor.  It is also
consumed during synthetic utterance generation to fill template slots like
`{artist}`, `{movie_title}`, `{radio_station}`, etc.

The CSV is populated from two channels:

1. **Bundled data** — curated hand-built entries + data from Radarr/Sonarr/Lidarr
   media-server scrapes (see `scrap_homeserver.sh`)
2. **Downloaded entity pools** — free public API datasets fetched by
   `scripts/scrape_entity_sources.py`

---

## ocp_entities.csv Schema

| Column | Description |
|--------|-------------|
| `title` | Primary entity string (the value used for NER matching) |
| `ocp_label` | `OCPEntityLabel` value (e.g. `artist_name`, `movie_title`) |
| `media_type` | `MediaType` string (e.g. `MUSIC`, `MOVIE`, `AUDIOBOOK`) |
| `genre` | Optional genre string (pipe-separated if multiple) |
| `actor` | Pipe-separated cast names (movies/TV) |
| `director` | Director name(s) |
| `producer` | Producer name(s) |
| `writer` | Writer name(s) |
| `composer` | Composer name(s) |
| `artist` | Artist name (music) |
| `album` | Album name (music) |
| `author` | Author name (audiobooks) |
| `narrator` | Narrator name (audiobooks) |
| `studio` | Studio / label name |
| `source` | Data provenance tag (e.g. `gutendex`, `librivox`, `radarr`) |

All additional columns (beyond `title` and `ocp_label`) are optional metadata
that may be extracted and emitted as secondary entity rows by
`generate_categorical_features.py:load_entity_wordlists()`.

---

## Entity Scraping Script

**File**: `scripts/scrape_entity_sources.py`

Fetches entity data from free public APIs and writes:
- One intermediate CSV per source into the entity cache dir
- Optionally merges/appends all rows into `scripts/ocp_entities.csv`

### Usage

```bash
# Fetch all sources and merge into ocp_entities.csv
python scripts/scrape_entity_sources.py

# Fetch specific sources only
python scripts/scrape_entity_sources.py --sources gutendex,librivox,anilist

# Write intermediate CSVs without merging
python scripts/scrape_entity_sources.py --sources steam,radio_garden --no-merge

# Dry-run (count only, write nothing)
python scripts/scrape_entity_sources.py --dry-run

# Custom output dir + merge target
python scripts/scrape_entity_sources.py \
    --output ~/.cache/ovos-media-classifier/entities/ \
    --merge scripts/ocp_entities.csv
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sources` | all | Comma-separated source names to fetch |
| `--output` | `~/.cache/ovos-media-classifier/entities/` | Intermediate CSV output dir |
| `--merge` | `scripts/ocp_entities.csv` | Target CSV to merge results into |
| `--no-merge` | False | Skip merging, write intermediate CSVs only |
| `--dry-run` | False | Fetch and count but do not write files |

---

## Data Sources

### 1. Gutendex (Project Gutenberg)
- **Source key**: `gutendex`
- **URL**: `https://gutendex.com/books/`
- **Auth**: None required
- **Rate limit**: No official limit; script uses 300ms delays
- **Entity labels**: `audiobook_title`, `audiobook_author`
- **Size**: ~78k books, ~200 pages of paginated JSON
- **Notes**: Public domain books only; very broad coverage of classic literature

### 2. LibriVox
- **Source key**: `librivox`
- **URL**: `https://librivox.org/api/feed/audiobooks`
- **Auth**: None required
- **Rate limit**: Lenient; script uses 500ms delays
- **Entity labels**: `audiobook_title`, `audiobook_author`, `audiobook_narrator`
- **Size**: ~20k audiobooks
- **Notes**: All entries have audio versions; narrator metadata where available

### 3. Radio Garden (Search API)
- **Source key**: `radio_garden`
- **URL**: `https://radio.garden/api/search?q=<term>`
- **Auth**: None (requires `Referer: https://radio.garden/` header)
- **Rate limit**: Max 20 results per query; script uses a curated list of
  30 genre/format search terms
- **Entity labels**: `radio_station`, `radio_genre`
- **Size**: ~550 station names + 31 genre tags per run
- **Notes**: The bulk channel endpoint (`/api/ara/content/channels`) was
  deprecated; search API is the available replacement

### 4. Anime Offline Database
- **Source key**: `anime_offline_db`
- **URL**: GitHub raw JSON dump (manami-project/anime-offline-database)
- **Auth**: None
- **Rate limit**: Single download
- **Entity labels**: `anime_title` (+ synonyms)
- **Size**: ~20k+ titles when the repository is accessible
- **Notes**: Falls back gracefully if the repository URL changes; use
  `anilist` source as a reliable alternative

### 5. AniList GraphQL API
- **Source key**: `anilist`
- **URL**: `https://graphql.anilist.co`
- **Auth**: None
- **Rate limit**: ~90 req/min; script uses 1.2s delays with 429 backoff
- **Entity labels**: `anime_title`, `anime_studio`
- **Size**: Thousands of anime + hundreds of studios (sorted by popularity)
- **Notes**: Most reliable anime source; sorted by POPULARITY_DESC so the
  most recognisable titles come first

### 6. Steam / SteamSpy
- **Source key**: `steam`
- **URL**: `https://steamspy.com/api.php?request=all` (fallback from official API)
- **Auth**: None
- **Rate limit**: ~1 request per session for the `all` endpoint
- **Entity labels**: `game_title`
- **Size**: ~1,000 top games from SteamSpy (free tier)
- **Notes**: The official Steam ISteamApps endpoint is currently returning 404;
  SteamSpy provides the top ~1k games sorted by player count.  For more
  comprehensive coverage, use the IGDB API (requires free registration).

### 7. Radio Browser (radio-browser.info)
- **Source key**: `radio_browser`
- **URL**: `https://de1.api.radio-browser.info/json/stations`
- **Auth**: None required (community-maintained, mirrors at de1/nl1/at1)
- **Rate limit**: None; single request for full dataset
- **Entity labels**: `radio_station`, `radio_genre`
- **Size**: 43,856 station names + 10,063 genre tags in one request
- **Notes**: By far the largest source for radio data. Sorted by votes so the
  most popular stations come first. The `tags` field contains comma-separated
  genre/format strings reused as `radio_genre` entries.

### 8. Adult / Hentai
- **Source key**: `adult`
- **Auth**: None required
- **Entity labels**: `adult_title`, `hentai_title`, `pornstar`, `porn_genre`, `adult_streaming_service`
- **Size**: 8,736 rows per run
- **Sources (3 combined)**:
  1. **Existing `ocp_entities.csv`** — re-emits performer names from `actor` column and studio names from `studio` column of existing adult rows (Whisparr-sourced) with proper OCPEntityLabel values: `pornstar`, `adult_streaming_service`, `adult_title`
  2. **Curated lists** — 65 porn genres, 38 hentai genres, 58 adult studios/services (all publicly known)
  3. **AniList `isAdult=true`** — hentai titles and studios via the same GraphQL endpoint used by the `anilist` source, filtered to adult content (900+ titles, 620+ studios per run)
- **Notes**: The `adult` source must be run after `ocp_entities.csv` is populated with Whisparr/media-server data. Running it again is idempotent (deduplication removes re-fetched rows).

### 9. IMDB / Wikidata SPARQL
- **Source key**: `imdb`
- **URL**: `https://query.wikidata.org/sparql`
- **Auth**: None required (Wikidata is fully public)
- **Rate limit**: Polite 1s delay between queries; Wikidata allows up to 5,000 rows per query
- **Entity labels**: `movie_title`, `tv_show_title`, `cartoon_title`, `movie_actor`, `movie_director`
- **Size per run**: 5,000 movies + 4,999 TV shows + 3,371 cartoons + 5,000 actors + 5,000 directors = 23,370 rows
- **Notes**: Wikidata is structurally equivalent to IMDB — it cross-links to IMDB IDs. Queries are simple SPARQL fetching English labels. Increase `titles_per_type` / `persons_per_role` args to fetch more (max ~10,000 per query before Wikidata times out). Re-running produces different random samples (no ordering) — useful for data augmentation.

### 10. PornHub Webmasters API
- **Source key**: `pornhub`
- **URL**: `https://www.pornhub.com/webmasters/`
- **Auth**: None required (public Webmasters API)
- **Rate limit**: None encountered; two requests total per run
- **Entity labels**: `pornstar`, `porn_genre`
- **Size**: 20,413 performers + 153 genre tags (single run)
- **Endpoints**:
  - `/webmasters/stars_detailed?page=1` — full performer list (27,933 entries, filtered to those with ≥1 video)
  - `/webmasters/categories` — 164 category/genre slugs (converted from `big-tits` to `big tits`)
- **Notes**: The stars list is returned in a single page — no pagination needed. Category slugs are normalised (hyphens → spaces, numeric suffixes stripped). Low-signal tags (`hd-porn`, `4k`, `vr`, etc.) are filtered via `_PH_SKIP_CATS`.

### 11. Open Library
- **Source key**: `open_library`
- **URL**: `https://openlibrary.org/subjects/<subject>.json`
- **Auth**: None
- **Rate limit**: Moderate; script uses 500ms delays
- **Entity labels**: `audiobook_title`, `audiobook_author`
- **Size**: 1,000 entries per subject, 8 subjects by default
- **Notes**: Complements Gutendex with different coverage; especially good
  for genre-filtered queries

---

## Intermediate CSV Files

Each source writes to:
```
~/.cache/ovos-media-classifier/entities/
  gutendex_entities.csv
  librivox_entities.csv
  radio_garden_entities.csv
  anime_offline_db_entities.csv
  anilist_entities.csv
  steam_entities.csv
  open_library_entities.csv
```

These intermediate CSVs use the same `ocp_entities.csv` schema and can be:
- Inspected independently before merging
- Re-merged at any time with `--merge scripts/ocp_entities.csv`
- Used directly with `EntitiesContainer.load_csv(path)`

---

## Deduplication

The merge step deduplicates on `(title.lower(), ocp_label.lower())` — case-
insensitive exact matches.  A row in the existing CSV is preserved unless a
new row has an identical `(title, ocp_label)` key.

Rows with empty `title` or empty `ocp_label` are dropped during deduplication
(they cannot be used for NER matching).

---

## Entity Label Coverage Target

| Label | Existing | Target | Recommended Source |
|-------|----------|--------|--------------------|
| `audiobook_title` | ~700 | 50,000+ | Gutendex + LibriVox + Open Library |
| `audiobook_author` | ~600 | 30,000+ | Gutendex + LibriVox |
| `audiobook_narrator` | ~50 | 5,000+ | LibriVox |
| `radio_station` | ~500 | 5,000+ | Radio Garden (search) |
| `radio_genre` | ~10 | 50+ | Radio Garden |
| `anime_title` | ~5,000 | 25,000+ | AniList + Anime Offline DB |
| `anime_studio` | ~100 | 1,000+ | AniList |
| `game_title` | ~2,000 | 20,000+ | SteamSpy + IGDB (free tier) |
| `podcast_title` | ~1,000 | 10,000+ | Listen Notes (requires free API key) |
| `news_provider` | ~500 | 2,000+ | NewsAPI (requires free API key) |

---

## Adding New Sources

To add a new entity source:

1. Write a `fetch_<name>() -> List[Dict[str, str]]` function that returns rows
   following the `_ENTITY_COLS` schema
2. Add it to `_FETCHERS` dict in `scrape_entity_sources.py`
3. Add it to `_ALL_SOURCES` list
4. Update this document

The row builder helper is `_row(**kwargs)` — any column not specified defaults
to an empty string.

Example:
```python
def fetch_my_source() -> List[Dict[str, str]]:
    rows = []
    data = _get("https://example.com/api/data")
    for item in data.get("items", []):
        rows.append(_row(
            title=item["name"],
            ocp_label="game_title",
            media_type="GAME",
            genre=item.get("category", ""),
            source="my_source",
        ))
    return rows

_FETCHERS["my_source"] = fetch_my_source
_ALL_SOURCES.append("my_source")
```

---

## Integration with NER Pipeline

Entity rows in `ocp_entities.csv` are consumed by two paths:

### 1. Categorical Feature Extraction
`scripts/generate_categorical_features.py:load_entity_wordlists()` reads the
CSV and builds `{ocp_label: [entity_strings]}` wordlists.  These are passed to
worker processes which construct an `EntitiesContainer` + `AhocorasickNER`.
The NER then tags each utterance with substring matches — identical to how
`AhocorasickMediaClassifier` works at inference time.

### 2. Synthetic Data Generation
`scripts/generate_synthetic.py` uses entity strings as slot-filler pools:
`{artist}` is filled from `artist_name` entries, `{title}` from `movie_title`,
etc.  More entities in the pool → more diverse generated utterances.

### 3. Runtime Entity Registration
At inference time, users can call:
```python
from ovos_media_classifier import EntitiesContainer
ents = EntitiesContainer()
ents.load_csv("scripts/ocp_entities.csv", entity_col="title", label_col="ocp_label")
```
to register all curated entities into the NER automaton.

---

## See Also

- `DATA_SOURCES.md` — high-level gap analysis and Phase 1/2/3 plan
- `scripts/scrape_entity_sources.py` — the downloader script
- `scripts/ocp_entities.csv` — the merged entity pool
- `scripts/generate_categorical_features.py` — uses entity pool for feature extraction
- `docs/NER_LABELS.md` — full list of `OCPEntityLabel` values and their meanings
