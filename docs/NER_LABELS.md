# NER Entity Label Taxonomy

This document is the canonical reference for `OCPEntityLabel` — the set of
named entity labels used by the AhocorasickNER classifier backend.

Understanding this taxonomy matters if you are:
- Writing an OCP skill that registers content with the classifier.
- Training or extending the NER database.
- Debugging why a query is or is not matched.

---

## Overview

When the `AhocorasickMediaClassifier` receives a query, it scans the text for
known entities using an Aho-Corasick automaton.  Each word or phrase in the
automaton is associated with an **entity label** — a string from the
`OCPEntityLabel` enum.

The label determines which `OCPPlayIntent` (and thus which `MediaType`) is
returned:

```
Query: "play something by Metallica"
  → NER hit: "Metallica" with label "artist_name"
  → NER_LABEL_TO_PLAY_INTENT["artist_name"] = OCPPlayIntent.MUSIC
  → MediaType.MUSIC, confidence=0.6
```

---

## The full taxonomy

### Streaming service labels

These are registered by OCP skills when they announce themselves at startup
(via `ovos.common_play.announce` bus message).

| OCPEntityLabel constant | String value | Maps to |
|---|---|---|
| `MUSIC_STREAMING_SERVICE` | `music_streaming_service` | `music` |
| `MOVIE_STREAMING_SERVICE` | `movie_streaming_service` | `movie` |
| `SHORTS_STREAMING_SERVICE` | `shorts_streaming_service` | `short_film` |
| `PODCAST_STREAMING_SERVICE` | `podcast_streaming_service` | `podcast` |
| `AUDIOBOOK_STREAMING_SERVICE` | `audiobook_streaming_service` | `audiobook` |
| `NEWS_PROVIDER` | `news_provider` | `news` |
| `TV_STREAMING_SERVICE` | `tv_streaming_service` | `tv` (live TV) |
| `RADIO_STREAMING_SERVICE` | `radio_streaming_service` | `radio` |
| `ADULT_STREAMING_SERVICE` | `adult_streaming_service` | `adult` |

Pre-seeded examples: Spotify, Netflix, Audible, BBC Radio, BBC News, etc.

### Music entity labels

Populated from HuggingFace music datasets (metal, jazz, prog, classical,
trance) and from music skills at runtime.

| OCPEntityLabel constant | String value | Maps to |
|---|---|---|
| `ARTIST_NAME` | `artist_name` | `music` |
| `TRACK_NAME` | `track_name` | `music` |
| `ALBUM_NAME` | `album_name` | `music` |
| `ALBUM_TYPE` | `album_type` | `music` |
| `MUSIC_GENRE` | `music_genre` | `music` |
| `RECORD_LABEL` | `record_label` | `music` |
| `RADIO_STATION` | `radio_station` | `radio` |

### Video entity labels

Populated from IMDB datasets (actor, director, producer, writer, composer
names) and from video skills at runtime.

| OCPEntityLabel constant | String value | Maps to |
|---|---|---|
| `MOVIE_TITLE` | `movie_title` | `movie` |
| `MOVIE_ACTOR` | `movie_actor` | `movie` |
| `MOVIE_DIRECTOR` | `movie_director` | `movie` |
| `MOVIE_PRODUCER` | `movie_producer` | `movie` |
| `MOVIE_WRITER` | `movie_writer` | `movie` |
| `MOVIE_COMPOSER` | `movie_composer` | `movie` |
| `TV_SHOW_TITLE` | `tv_show_title` | `tv_show` (episodic series) |
| `ANIME_TITLE` | `anime_title` | `anime` |
| `CARTOON_TITLE` | `cartoon_title` | `cartoon` |
| `DOCUMENTARY_TITLE` | `documentary_title` | `documentary` |
| `TRAILER_TITLE` | `trailer_title` | `trailer` |
| `BTS_TITLE` | `bts_title` | `behind_the_scenes` |
| `MUSIC_VIDEO_TITLE` | `music_video_title` | `music_video` |
| `VISUAL_STORY_TITLE` | `visual_story_title` | `visual_story` |
| `SILENT_MOVIE_TITLE` | `silent_movie_title` | `silent_movie` |
| `BW_MOVIE_TITLE` | `bw_movie_title` | `bw_movie` |
| `HENTAI_TITLE` | `hentai_title` | `hentai` |
| `RADIO_DRAMA_TITLE` | `radio_drama_title` | `radio_theatre` |
| `ADULT_TITLE` | `adult_title` | `adult` |
| `PORNSTAR` | `pornstar` | `adult` |
| `PORN_GENRE` | `porn_genre` | `adult` |

### TV / live stream entity labels

| OCPEntityLabel constant | String value | Maps to |
|---|---|---|
| `TV_CHANNEL` | `tv_channel` | `tv` (live TV) |
| `YOUTUBE_CHANNEL` | `youtube_channel` | `video_episodes` |

Examples: "CNN", "BBC One", "Eurosport"

### Other media entity labels

| OCPEntityLabel constant | String value | Maps to |
|---|---|---|
| `PODCAST_TITLE` | `podcast_title` | `podcast` |
| `PODCAST_HOST` | `podcast_host` | `podcast` |
| `PODCAST_EPISODE` | `podcast_episode` | `podcast` |
| `AUDIOBOOK_TITLE` | `audiobook_title` | `audiobook` |
| `AUDIOBOOK_AUTHOR` | `audiobook_author` | `audiobook` |
| `NEWS_CATEGORY` | `news_category` | `news` |
| `GAME_TITLE` | `game_title` | `game` |
| `ASMR_ARTIST` | `asmr_artist` | `asmr` |

> **Note**: `NEWS_TOPIC` (formerly here) was removed — it was untrainable because
> virtually any noun qualifies as a news topic.  `NEWS_CATEGORY` replaces it with
> a coarse set of values ("sports", "weather", "politics") that a model can learn.

### Media-type keyword labels

Generic vocabulary labels.  These are matched when no specific named entity is
found but a keyword strongly signals the media type.  They have lower priority
than named entity labels.

| OCPEntityLabel constant | String value | Maps to |
|---|---|---|
| `MUSIC_KEYWORD` | `music` | `music` |
| `PODCAST_KEYWORD` | `podcast` | `podcast` |
| `RADIO_KEYWORD` | `radio` | `radio` |
| `AUDIOBOOK_KEYWORD` | `audiobook` | `audiobook` |
| `NEWS_KEYWORD` | `news` | `news` |
| `MOVIE_KEYWORD` | `movie` | `movie` |
| `TV_KEYWORD` | `tv` | `tv` (live TV) |
| `TV_SHOW_KEYWORD` | `tv_show` | `tv_show` (episodic series) |
| `VIDEO_KEYWORD` | `video` | `video` |
| `VIDEO_EPISODES_KEYWORD` | `video_episodes` | `video_episodes` |
| `AUDIO_KEYWORD` | `audio` | `audio` |
| `GAME_KEYWORD` | `game` | `game` |
| `ANIME_KEYWORD` | `anime` | `anime` |
| `CARTOON_KEYWORD` | `cartoon` | `cartoon` |
| `DOCUMENTARY_KEYWORD` | `documentary` | `documentary` |
| `SHORT_FILM_KEYWORD` | `short_film` | `short_film` |
| `SILENT_MOVIE_KEYWORD` | `silent_movie` | `silent_movie` |
| `BW_MOVIE_KEYWORD` | `bw_movie` | `bw_movie` |
| `RADIO_THEATRE_KEYWORD` | `radio_theatre` | `radio_theatre` |
| `VISUAL_STORY_KEYWORD` | `visual_story` | `visual_story` |
| `ASMR_KEYWORD` | `asmr` | `asmr` |
| `AUDIO_DESCRIPTION_KEYWORD` | `audio_description` | `audio_description` |
| `MUSIC_VIDEO_KEYWORD` | `music_video` | `music_video` |
| `TRAILER_KEYWORD` | `trailer` | `trailer` |
| `BEHIND_THE_SCENES_KEYWORD` | `behind_the_scenes` | `behind_the_scenes` |
| `ADULT_KEYWORD` | `adult` | `adult` |
| `ADULT_AUDIO_KEYWORD` | `adult_audio` | `adult_audio` |
| `HENTAI_KEYWORD` | `hentai` | `hentai` |

---

## Priority ordering

When multiple entity labels match a query, the backend resolves the conflict
using `_INTENT_PRIORITY` — a list from highest priority (index 0) to lowest.
The general rule:

1. **Streaming service labels** — a named service beats everything.
2. **Named entity labels** (titles, people) — beat generic keywords.
3. **Specific media keywords** — beat generic `video`/`audio` labels.
4. **Generic keywords** (`video`, `audio`) — last resort.

This ensures:
- _"play something on Netflix"_ → `movie` (Netflix is a movie streaming service),
  not `music`.
- _"put on Blade Runner"_ → `movie` (title match), not `video` (keyword).
- _"play some music"_ → `music` (keyword match).

---

## EntitiesContainer — populating the NER from real media libraries

`EntitiesContainer` is the recommended way to pre-populate the NER with
content from a user's actual media library.  It wraps an `AhocorasickNER`
and exposes loaders for every supported media server.  Because the NER is
shared by reference, calling `container.add()` after the classifier is built
is immediately reflected in classification results.

For dataset generation and analytics (one row per media item, with actor /
director / artist / album / studio columns) use `generate_dataset_from_media.py`
instead — see [MEDIA_SERVER_INTEGRATIONS.md](MEDIA_SERVER_INTEGRATIONS.md).

```python
from ovos_media_classifier import EntitiesContainer
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

container = EntitiesContainer()
container.load_radarr("http://radarr:7878",    api_key="…")  # movie_title, cast, crew
container.load_sonarr("http://sonarr:8989",    api_key="…")  # tv_show_title, anime_title
container.load_lidarr("http://lidarr:8686",    api_key="…")  # artist_name, album_name, track_name
container.load_jellyfin("http://jellyfin:8096",api_key="…")  # all types incl. audiobooks
container.load_whisparr("http://whisparr:6969",api_key="…")  # adult, movie_actor
container.load_stash("http://stash:9999",      api_key="…")  # performers, studios, scenes
container.load_music_assistant("http://ma:8095")              # artists, albums, radio stations
container.load_huggingface("TigreGotico/ocp-entities")        # curated OCP entity dataset
container.load_csv("/path/to/extra.csv")                      # supplemental CSV

clf = AhocorasickMediaClassifier.from_container(container)

# Introspect what was loaded
print(container.stats)
# {'artist_name': 3412, 'album_name': 7201, 'movie_title': 941, ...}
```

See [BACKENDS.md](BACKENDS.md) for the full loader reference and config-driven
construction via `load_media_classifier()`.

---

## Runtime registration protocol

OCP skills register entities at startup via two bus messages:

### `ovos.common_play.register_keyword`

Registers a single word or phrase under a label.

```python
from ovos_bus_client import Message

self.bus.emit(Message("ovos.common_play.register_keyword", {
    "label": "artist_name",       # OCPEntityLabel string value
    "match": "Pink Floyd",        # word or phrase to register
    "media_type": MediaType.MUSIC, # for cross-reference (informational)
}))
```

### `ovos.common_play.announce`

Sent by OCP skills when they start up, announcing what service they represent.
The pipeline plugin extracts the service name and registers it under the
appropriate streaming service label.

```python
self.bus.emit(Message("ovos.common_play.announce", {
    "skill_id": "ovos-skill-spotify.openvoiceos",
    "service_name": "Spotify",
    "media_types": [MediaType.MUSIC],
}))
```

The pipeline maps `MediaType.MUSIC` → `"music_streaming_service"` and
registers `"Spotify"` under that label.

---

## Design rationale

### Why are keyword labels and named entity labels in the same enum?

At training time the sklearn model is trained on NER features — boolean flags
indicating which entity labels were found.  Generic keywords (`music_genre`,
`artist_name`) and streaming service labels are all features in the same
feature vector.  Keeping them in one enum makes the feature space explicit and
prevents label string drift between training and inference.

### Why does the NER depend on the user's media library?

This is intentional.  If the user has no podcast skill installed, then no
podcast titles are registered, and a query mentioning a specific podcast will
_not_ be matched as `podcast`.  This is correct — OCP can only play what it
has access to.  A generic ML model might still detect "podcast" from context,
but the NER backs that up with concrete knowledge of what the user can actually
play.  The two approaches complement each other.

### Adding a new label type

1. Add a constant to `OCPEntityLabel` in `intents.py`.
2. Add a mapping from the new label to an `OCPPlayIntent` in
   `NER_LABEL_TO_PLAY_INTENT` in `intents.py`.
3. Update `_INTENT_PRIORITY` in `ahocorasick.py` to give the new label an
   appropriate priority position.
4. If HuggingFace data exists for the new entity type, add a loader in
   `train/ner_datasets.py`.
