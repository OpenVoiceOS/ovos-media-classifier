# Media Server Integrations

This document covers the media-server data pipeline: how to pull your real
media library into OCP classifier datasets and the `EntitiesContainer` NER.

Two complementary outputs exist:

| Script | Output | Purpose |
|---|---|---|
| `generate_dataset_from_media.py` | Wide-format CSV (one row per item, attributes as columns) | Dataset building, analytics, training |
| `EntitiesContainer` loaders | Flat `{label: Set[str]}` entity store | Runtime NER population |

---

## Supported sources

| Source | Default port | API type | Auth | Content |
|---|---|---|---|---|
| **Radarr** | 7878 | REST (JSON) | `X-Api-Key` header | Movies + cast/crew metadata |
| **Sonarr** | 8989 | REST (JSON) | `X-Api-Key` header | TV shows, anime, documentaries |
| **Lidarr** | 8686 | REST (JSON) | `X-Api-Key` header | Artists, albums, tracks, genres |
| **Readarr** | 8787 | REST (JSON) | `X-Api-Key` header | Books / audiobooks + authors |
| **Whisparr** | 6969 | REST (JSON) | `X-Api-Key` header | Adult movies + performers (Radarr fork) |
| **Jellyfin** | 8096 | REST (JSON) | `api_key` query param | Movies, TV, music (audiobooks/podcasts opt-in) |
| **Stash** | 9999 | GraphQL (`/graphql`) | `ApiKey` header | Adult scenes, performers, studios, tags |
| **Music Assistant** | 8095 | REST (JSON) | none | Artists, albums, tracks, radio stations |
| **Audiobookshelf** | 13378 | REST (JSON) | `Authorization: Bearer TOKEN` | Audiobooks + podcasts with author/narrator/genre |
| **Listenarr** | 6969* | REST (JSON) | `X-Api-Key` header | Audiobooks with authors, narrators, series |
| **Kapowarr** | 5656 | REST (JSON) | `X-Api-Key` header | Comic volumes (maps to `VISUAL_STORY`) |
| **Mylar3** | 8090 | REST (JSON) | `?apikey=` query param | Comic series with publisher |
| **Podgrab** | 8080 | REST (JSON) | HTTP basic auth (optional) | Podcasts with categories |

> *Listenarr default port may vary — check your deployment configuration.

Any source can be omitted — the script and loaders skip missing sources
gracefully.

---

## generate_dataset_from_media.py

### Output format

The script produces a **wide-format CSV** where every row is a single primary
media item and every column is a metadata attribute.

```
title,ocp_label,media_type,genre,actor,director,producer,writer,composer,artist,album,author,studio,source
The Dark Knight,movie_title,MOVIE,Action|Crime|Drama,Christian Bale|Heath Ledger,Christopher Nolan,...,Warner Bros.,radarr
Bohemian Rhapsody,track_name,MUSIC,,,,,,Freddie Mercury,A Night at the Opera,,,lidarr
Breaking Bad,tv_show_title,TV_SHOW,Crime|Drama,,,,,,,,,AMC,sonarr
```

**Column reference:**

| Column | Type | Description |
|---|---|---|
| `title` | string | Primary title of the media item |
| `ocp_label` | string | `OCPEntityLabel` string value (e.g. `movie_title`) |
| `media_type` | string | OCP `MediaType` name (e.g. `MOVIE`, `MUSIC`, `TV_SHOW`) |
| `genre` | pipe-separated | Genres from the source |
| `actor` | pipe-separated | Actor / performer names |
| `director` | pipe-separated | Director names |
| `producer` | pipe-separated | Producer names |
| `writer` | pipe-separated | Writer / screenplay names |
| `composer` | pipe-separated | Music composer names |
| `artist` | pipe-separated | Recording artist names (music rows) |
| `album` | string | Album name (track rows) |
| `author` | string | Author name (audiobook rows) |
| `narrator` | string | Narrator name (audiobook rows) |
| `studio` | string | Studio, network, or record label |
| `source` | string | Which service provided this row |

Multi-value fields use `|` as the separator (chosen to avoid conflicts with
commas in titles and semicolons in subtitles).

### Usage

```bash
python generate_dataset_from_media.py --output media_dataset.csv \
    --radarr-url     http://localhost:7878 --radarr-api-key   KEY \
    --sonarr-url     http://localhost:8989 --sonarr-api-key   KEY \
    --lidarr-url     http://localhost:8686 --lidarr-api-key   KEY \
    --readarr-url    http://localhost:8787 --readarr-api-key  KEY \
    --whisparr-url   http://localhost:6969 --whisparr-api-key KEY \
    --jellyfin-url   http://localhost:8096 --jellyfin-api-key KEY \
    --stash-url      http://localhost:9999 --stash-api-key    KEY \
    --music-assistant-url http://localhost:8095
```

Jellyfin item types default to `Movie,Series,MusicAlbum,MusicArtist,Audio`.
To include audiobooks or podcasts, pass `--jellyfin-types` explicitly:

```bash
python generate_dataset_from_media.py \
    --jellyfin-url   http://localhost:8096 --jellyfin-api-key KEY \
    --jellyfin-types Movie,Series,MusicAlbum,MusicArtist,Audio,Book,AudioBook,Podcast
```

> **Why are Book/AudioBook/Podcast excluded by default?**  Jellyfin's metadata
> scraper for these types often returns sparse or incorrect data (missing authors,
> wrong genres, episode titles used as book titles).  Use **Readarr** for
> audiobooks/books — it has purpose-built metadata from sources like Goodreads
> and OpenLibrary.

All `--*-url` arguments can be replaced by environment variables:

```bash
export RADARR_URL=http://localhost:7878
export RADARR_API_KEY=mykey
# etc.
python generate_dataset_from_media.py
```

### OCP label → MediaType mapping

The `ocp_label` and `media_type` columns are derived as follows:

| `ocp_label` | `media_type` | Source |
|---|---|---|
| `movie_title` | `MOVIE` | Radarr, Jellyfin |
| `anime_title` | `ANIME` | Radarr (anime genre), Sonarr (anime series type), Jellyfin |
| `cartoon_title` | `CARTOON` | Radarr/Sonarr (animation genre) |
| `documentary_title` | `DOCUMENTARY` | Radarr/Sonarr (documentary genre) |
| `tv_show_title` | `TV_SHOW` | Sonarr, Jellyfin |
| `track_name` | `MUSIC` | Lidarr, Jellyfin, Music Assistant |
| `album_name` | `MUSIC` | Lidarr, Jellyfin, Music Assistant |
| `artist_name` | `MUSIC` | Lidarr, Jellyfin, Music Assistant |
| `radio_station` | `RADIO` | Music Assistant |
| `audiobook_title` | `AUDIOBOOK` | Readarr, Jellyfin (opt-in) |
| `audiobook_author` | `AUDIOBOOK` | Readarr |
| `podcast_title` | `PODCAST` | Jellyfin (opt-in) |
| `adult` | `ADULT` | Whisparr, Stash |

Genre-based label refinement is applied to movies and TV shows: a movie tagged
`"anime"` becomes `anime_title` (`ANIME`) rather than `movie_title` (`MOVIE`).
Priority: anime > documentary > cartoon > default.

### What each source provides

#### Radarr

- One row per movie title (plus one row per alternate title).
- Genre-based label refinement applied.
- Cast extracted from `credits.castMembers` → `actor` column.
- Crew extracted from `credits.crewMembers`, classified by job string:
  - `"direct…"` → `director`
  - `"produc…"` → `producer`
  - `"writ…"` / `"screenplay"` → `writer`
  - `"compos…"` / `"original score"` → `composer`
- `studio` from `movie.studio`.

#### Sonarr

- One row per series title (plus one row per alternate title).
- `seriesType == "anime"` or `"anime"` genre → `anime_title`.
- `network` → `studio` column.
- No cast/crew (Sonarr does not expose per-series credits).

#### Lidarr

- One row per **artist** (`ocp_label = artist_name`).
- One row per **album** (`ocp_label = album_name`), with `artist` populated.
- One row per **track** inside each album (`ocp_label = track_name`), with
  `artist` and `album` columns populated.
- Artist genres flow down to album and track rows.

#### Readarr

- One row per **author** (`ocp_label = audiobook_author`), with genres from the
  author object.
- One row per **book** (`ocp_label = audiobook_title`), with `author` column
  populated from the nested author object.
- When a book belongs to a series (`seriesTitle` field), one additional
  `audiobook_title` row is emitted for the series name (deduplicated across all
  books in the series).
- API endpoints used: `GET /api/v1/author`, `GET /api/v1/book`.
- All `ocp_label = audiobook_*` rows map to `media_type = AUDIOBOOK`.

#### Whisparr

- One row per adult movie title (plus alternate titles).
- Cast from `credits.castMembers` → `actor`.
- `studio` from `movie.studio`.

#### Stash (GraphQL)

- Performers and studios are pre-fetched into lookup maps.
- One row per **scene title** (`ocp_label = adult`).
- `actor` = performer names embedded in the scene object.
- `genre` = scene tags.
- `studio` = studio name from the scene object.

#### Jellyfin

- Paginated fetch (500 items per page) per item type:
  `Movie`, `Series`, `MusicAlbum`, `MusicArtist`, `Audio`, `Book`/`AudioBook`, `Podcast`.
- Genre-based label refinement for movies and series.
- `People` list parsed for actor, director, producer, writer, composer, author roles.
- `Studios` list → `studio` column.
- `Artists` list → `artist` column (for `Audio` items).
- `Album` field → `album` column (for `Audio` items).

#### Audiobookshelf

- Auth: `Authorization: Bearer <api_token>` (get from Settings → Users → API Token).
- `GET /api/libraries` → list libraries; fetches only those with `mediaType = "book"` or
  `"podcast"`.
- `GET /api/libraries/{id}/items?limit=100&page=N` → paginated items.
- **Book items** → `audiobook_title` rows:
  - `author` = `media.metadata.authorName`
  - `narrator` = `media.metadata.narratorName`
  - `genre` = `media.metadata.genres[]`
  - `studio` = `media.metadata.publisherName`
  - `seriesName` emitted as a second `audiobook_title` row (if different from title)
- **Podcast items** → `podcast_title` rows:
  - `author` = `media.metadata.author`
  - `genre` = `media.metadata.genres[]`

#### Listenarr

- Auth: `X-Api-Key` header.
- `GET /api/v1/library` → list audiobooks.
- Fields: `Title`, `Authors[]`, `Narrators[]`, `Genres[]`, `Series`.
- Series names emitted as separate `audiobook_title` rows (deduplicated).
- `narrator` column populated from `Narrators[]`.

#### Kapowarr

- Auth: `X-Api-Key` header.
- `GET /api/volumes` → list comic volumes.
- Fields: `title`, `alt_title`, `publisher`, `year`, `volume_number`.
- Alternate titles emitted as separate rows with the same metadata.
- OCP mapping: `comic_title` → `VISUAL_STORY` (no dedicated comic MediaType exists in OCP yet).

#### Mylar3

- Auth: `?apikey=KEY` query parameter.
- `GET /api?apikey=KEY&cmd=getComicList` → list all monitored comic series.
- Fields: `name` (series title), `publisher`.
- OCP mapping: `comic_title` → `VISUAL_STORY`.

#### Podgrab

- Auth: HTTP basic auth (`--podgrab-username` / `--podgrab-password`), optional.
- `GET /podcasts` → list all subscribed podcasts.
- Fields: `title`, `categories[]` (→ `genre`), `author`.
- OCP mapping: `podcast_title` → `PODCAST`.

#### Music Assistant

- `/api/music/artists` → `artist_name` rows (with genres from metadata).
- `/api/music/albums` → `album_name` rows (artist names from embedded artist objects).
- `/api/music/tracks` → `track_name` rows (artist + album populated).
- `/api/music/radio` → `radio_station` rows.
- Pagination via `limit`/`offset` parameters.

---

## EntitiesContainer — runtime NER population

`EntitiesContainer` uses the same sources to populate the AhocorasickNER with
flat entity strings.  Unlike the wide-format CSV, it stores only the entity
string and its label — relationships between entities are not preserved.

```python
from ovos_media_classifier.entities import EntitiesContainer
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

container = EntitiesContainer()

container.load_radarr("http://radarr:7878",     api_key="…")
container.load_sonarr("http://sonarr:8989",     api_key="…")
container.load_lidarr("http://lidarr:8686",     api_key="…")
container.load_whisparr("http://whisparr:6969", api_key="…")
container.load_jellyfin("http://jellyfin:8096", api_key="…",
                        user_id="optional-uuid")
container.load_stash("http://stash:9999",       api_key="…")
container.load_music_assistant("http://ma:8095")

clf = AhocorasickMediaClassifier.from_container(container)
print(container.stats)
# {'artist_name': 3412, 'album_name': 7201, 'movie_title': 941, ...}
```

### Entities added per source

| Source | Labels populated |
|---|---|
| Radarr | `movie_title` / `anime_title` / `cartoon_title` / `documentary_title`, `movie_actor`, `movie_director`, `movie_producer`, `movie_writer`, `movie_composer`, `movie_streaming_service` |
| Sonarr | `tv_show_title` / `anime_title` / `cartoon_title` / `documentary_title`, `tv_streaming_service` |
| Lidarr | `artist_name`, `album_name`, `track_name`, `music_genre` |
| Readarr | `audiobook_title`, `audiobook_author` |
| Whisparr | `adult`, `movie_actor`, `porn_streaming_service` |
| Jellyfin | `movie_title`, `tv_show_title`, `album_name`, `artist_name`, `track_name` (default); optionally `audiobook_title`, `audiobook_author`, `podcast_title` |
| Stash | `adult`, `movie_actor`, `porn_streaming_service` |
| Music Assistant | `artist_name`, `album_name`, `track_name`, `radio_station`, `music_genre` |
| Audiobookshelf | `audiobook_title`, `podcast_title` |
| Listenarr | `audiobook_title` |
| Kapowarr | `comic_title` |
| Mylar3 | `comic_title` |
| Podgrab | `podcast_title` |

### Config-driven loading

`EntitiesContainer.from_config()` accepts a dict so loaders can be driven from
the OVOS config file:

```python
from ovos_media_classifier.entities import EntitiesContainer

container = EntitiesContainer.from_config({
    "radarr":   {"url": "http://radarr:7878",   "api_key": "…"},
    "sonarr":   {"url": "http://sonarr:8989",   "api_key": "…"},
    "lidarr":   {"url": "http://lidarr:8686",   "api_key": "…"},
    "jellyfin": {"url": "http://jellyfin:8096", "api_key": "…"},
    "stash":    {"url": "http://stash:9999",    "api_key": "…"},
    "music_assistant": {"url": "http://ma:8095"},
    "csv": ["/path/to/extra.csv"],
    "huggingface": [{"dataset": "TigreGotico/ocp-entities"}],
    "wordlists": {
        "artist_name": ["My Local Band"],
        "movie_title":  ["My Home Movie"],
    },
})
```

### Runtime updates

Because the `AhocorasickNER` is shared by reference with the container, any
call to `container.add()` after `from_container()` is immediately reflected in
classifier results — no rebuild step required:

```python
# Skill announces new content at runtime
container.add("artist_name", "Radiohead")
clf.classify("play radiohead", "en-us")
# → (MediaType.MUSIC, 0.6)  — works immediately
```

---

## CSV vs EntitiesContainer — when to use each

| Use case | Use |
|---|---|
| Training a new ML model with real-world metadata | `generate_dataset_from_media.py` |
| Building a dataset for analytics / entity resolution | `generate_dataset_from_media.py` |
| Populating the NER at OVOS startup | `EntitiesContainer` loaders |
| Incrementally adding new content at runtime | `container.add()` |
| Seeding from a curated HuggingFace dataset | `container.load_huggingface()` |
| Seeding from a local CSV | `container.load_csv()` |

The wide-format CSV produced by `generate_dataset_from_media.py` can be
converted back to the flat `entity,label` format expected by `load_csv()` using
pandas:

```python
import pandas as pd

df = pd.read_csv("media_dataset.csv")

# Title rows
titles = df[["title", "ocp_label"]].rename(columns={"title": "entity", "ocp_label": "label"})

# Actor rows
actors = df[df["actor"] != ""].assign(
    entity=df["actor"].str.split("|")
).explode("entity").assign(label="movie_actor")[["entity", "label"]]

flat = pd.concat([titles, actors]).dropna().drop_duplicates()
flat.to_csv("entities_flat.csv", index=False)
```

---

## Dependencies

```bash
pip install ovos-media-classifier[media_servers]
# Installs: requests
```

Stash additionally requires `requests` (already included).  No extra
dependencies are needed for Music Assistant.
