
# Dataset Generation Pipeline — ovos-media-classifier

## Overview

The dataset generation pipeline synthesizes training data for the media classifier by combining:

1. **Real-world entities** from Wikidata, media servers (Radarr, Sonarr, Lidarr, Jellyfin), and HuggingFace datasets
2. **Sentence templates** with placeholders (`{artist}`, `{track}`, `{genre}`, etc.)
3. **Literal fallbacks** for verbs, media type keywords, genres, and providers
4. **Deduplication** against existing datasets to avoid bias

**Goal**: Generate diverse, realistic utterances that exercise all slots in a template system without manual annotation.

---

## Architecture

### Data Flow

```
HuggingFace (OCP templates)
    ↓
+---────────────────────────────────────+
| generate_from_ocp_templates.py         |  ← Large pre-built template dataset
+---────────────────────────────────────+
    ↓
Wikidata entities (movie/artist/game names)
    ↓
+---────────────────────────────────────+
| generate_synthetic.py                  |  ← Lightweight, customizable
| (templates embedded in code)           |
+---────────────────────────────────────+
    ↓
Media server entities (Radarr/Sonarr/Lidarr/Jellyfin)
    ↓
Local CSV templates
    ↓
+---────────────────────────────────────+
| build_dataset.py                       |  ← Combine sources, train model
| (optional: llm_augment.py for UDA)    |
+---────────────────────────────────────+
    ↓
output/ocp_dataset.csv
```

### Key Scripts

| Script | Purpose | Inputs | Outputs | When to Use |
|--------|---------|--------|---------|-------------|
| `generate_from_ocp_templates.py` | Fill large HuggingFace template set with Wikidata entities | HF OCP_templates, WikidataMediaEntities | CSV with 7 columns (lang, domain, intent, ..., sentence) | Training from public sources, high diversity |
| `generate_synthetic.py` | Generate from hardcoded templates + optional media server exports | CSV templates (local), HF Wikidata, media server data | CSV | Rapid iteration, controlled entity scope |
| `generate_dataset_from_media.py` | Extract entities from Radarr, Sonarr, Lidarr, Jellyfin | Media server URLs + API keys | Wide CSV (title, actor, director, genre, ...) | Bridge between media library and template generation |
| `build_dataset.py` | Combine multiple sources, dedup, balance, train model | CSVs from above | `ocp_dataset.csv`, trained model checkpoint | Final assembly and validation |
| `llm_augment.py` | Augment dataset using LLM paraphrasing | CSV dataset, LLM API keys | Augmented CSV | Semi-supervised learning when labeled data is scarce |
| `download_datasets.py` | Pre-cache HuggingFace datasets locally | (none) | `~/.cache/huggingface/datasets/` | Initial setup, offline work |

---

## Input Formats

### 1. Sentence Templates

**Location**: `scripts/generate_synthetic.py` (hardcoded) or `templates/*.csv` (external)

**Schema**: Two columns
```
category,template
music_artist,"play {artist}"
music_artist,"play some {artist} music"
music_track,"play {track} by {artist}"
movie_generic,"play a movie"
```

**Slot Naming Convention**:
- `{artist}`, `{track}`, `{album}` — Music entities
- `{title}`, `{show}` — Video entities (TV, anime, movies)
- `{actor}`, `{director}`, `{writer}`, `{producer}` — Person names
- `{genre}` — Genre string (e.g., "action", "jazz")
- `{topic}` — General text (podcasts)
- `{provider}` — Streaming service name

### 2. Entity Pools

**Sources** (in priority order, later wins on conflict):

1. **Literal slots** (`_LITERAL_SLOTS` in `generate_from_ocp_templates.py`)
   - Verbs: `play`, `stream`, `watch`, `launch`
   - Media types: `movie`, `podcast`, `game`, `anime`
   - Genres: `action`, `comedy`, `jazz`, `horror`
   - Providers: `Netflix`, `Spotify`, `Steam`

2. **Local CSV entities** (optional, highest priority)
   - From `generate_dataset_from_media.py` output
   - Real titles, actors, directors from user's media library

3. **HuggingFace NER datasets** (curated music/film archives)
   - `Jarbas/metal-archives-bands` (artist_name, album_name, track_name)
   - `Jarbas/jazz-music-archives` (artist, genre)
   - `Jarbas/movie_actors`, `Jarbas/movie_directors`, etc.
   - See `load_ner_entity_pools()` for full list

4. **Wikidata dataset** (`Jarbas/WikidataMediaEntities`)
   - 1.6M+ entities: movie names, actor names, game titles, anime names, etc.
   - Mapped to template slots via `_WIKIDATA_SLOT_MAP`
   - Most comprehensive source, but generic (no personalization)

### 3. Label Mappings

**Template → MediaType**: `_LABEL_TO_OCP` in `generate_from_ocp_templates.py`

```python
_LABEL_TO_OCP = {
    "music":        ("ocp_play", "music"),
    "movie":        ("ocp_play", "movie"),
    "podcast":      ("ocp_play", "podcast"),
    "anime":        ("ocp_play", "anime"),
    "not_media":    ("not_ocp", "not_ocp"),
    # ... etc
}
```

**Entity type → Template slot**: `_WIKIDATA_SLOT_MAP`

```python
_WIKIDATA_SLOT_MAP = {
    "artist_name":     ["music_artist_name", "artist_name"],
    "movie_name":      ["movie_name", "trailer_name"],
    "game_name":       ["game_name"],
    # ... etc
}
```

---

## Template Design Best Practices

### 1. **Variety Over Quantity**

Write 10–15 diverse templates per intent, not 100 similar ones.

✅ **Good**:
```
play {artist}
play some {artist} music
I want to listen to {artist}
play {artist}'s greatest hits
I'm in the mood for {artist}
```

❌ **Bad**:
```
play {artist}
play {artist} music
play music from {artist}
play {artist} tracks
play {artist} songs
... (5 more minor variations)
```

**Rationale**: Small template set with diverse wording + random entity sampling creates more natural variation than many near-identical templates.

### 2. **Slot Independence**

Ensure each slot can be filled from an independent entity pool without semantic conflict.

✅ **Good** — slot independence:
```
play {track} by {artist}
```
(`track_name` pool and `artist_name` pool are independent)

❌ **Bad** — forced pairing:
```
play {album} from {artist}
```
(Real albums are by specific artists; random pairing creates nonsense like "Play The Beatles' _The Wall_")

**Fix**: Add curated album-artist pairs in entity pool, or skip the template.

### 3. **Specificity Hierarchy**

Order templates from most-specific (entity required) to least-specific (entity optional):

```python
# High specificity: requires entity
("play {track} by {artist}", ["track", "artist"]),  ← filled from pools
("play {artist} radio", ["artist"]),
("play {genre} music", ["genre"]),

# Medium specificity: optional fallback slots
("play some music", []),  ← always succeeds, generic
("play something", []),
```

### 4. **Realistic Phrasings**

Include natural speech patterns and filler words:

✅ **Good** (natural variance):
```
play {artist}
put on some {artist}
I want to listen to {artist}
find me some {artist} music
I'm in the mood for {artist}
can you play {artist}
```

❌ **Bad** (stilted):
```
play {artist}
play artist {artist}
play music by {artist} please
I wish to audition {artist}
```

### 5. **Cross-Modal Clarity**

Avoid templates that work equally well for both audio and video:

❌ **Bad** (ambiguous):
```
play {something}
```
(Could be music OR TV; classifier cannot infer intent from template alone)

✅ **Good** (clear modality):
```
listen to {artist}  ← audio
watch {show}        ← video
```

### 6. **Negation Sparsely**

Negation is hard for classifiers; use sparingly and only at sentence periphery.

❌ **Bad**:
```
don't play {artist}
play anything except {genre}
I don't want {media_type}
```

✅ **OK** (periphery):
```
I'm not in the mood for {artist}
```

### 7. **Provider-Agnostic**

Unless testing provider-specific behavior, avoid hardcoding provider names in templates. Let `{provider}` be a slot:

❌ **Bad**:
```
play {artist} on Spotify
stream {track} via Apple Music
```

✅ **Good**:
```
play {artist}
play {artist} on {provider}  (provider is a slot)
```

---

## Entity Pool Strategy

### By MediaType

| MediaType | Primary Source | Fallback | Notes |
|-----------|---|---|---|
| MUSIC | Wikidata (`artist_name`, `album_name`, `song_name`) + HF music archives (metal, jazz, prog) | Curated list (50–200 entries) | Rich: many artists available |
| MOVIE | Wikidata + HF movie person datasets | IMDB curated top 200 | Moderate: many titles but actor/director pairing important |
| TV_SHOW | Curated list only (HF dataset sparse) | Wikidata series_name | Limited: ~450 titles hardcoded; augment with user's Sonarr library |
| PODCAST | Curated list (popular shows) + HF podcast dataset | Topics list (50+ standard podcasts) | Niche: few datasets; topics generic |
| GAME | Wikidata game_name | Steam top 500 | Moderate: good coverage |
| ANIME | Wikidata anime_name + curated (MAL top 100) | Curated list | Good: popular titles well-known |
| AUDIOBOOK | Wikidata book_name + HF book authors | Curated classics (Harry Potter, Dune, etc.) | Good: many books in Wikidata |
| RADIO | Curated list (BBC, NPR, SomaFM, etc.) | HF radio dataset (if exists) | Fixed: radio stations are static |
| NEWS | Curated news providers (BBC, CNN, Reuters) | Wikidata if available | Fixed: providers are static; topics variable |
| DOCUMENTARY | Wikidata + topic slots | Topic list (nature, history, science) | Mixed: title + topic combo |
| TRAILER, BEHIND_THE_SCENES | Wikidata movie_name | Movie title list | Derived: use same as MOVIE |
| SHORT_FILM, SILENT_MOVIE, BW_MOVIE | Wikidata + curated lists | Wikipedia/IMDB | Limited: small niches |
| ASMR, GAME | Curated fallbacks | Wikidata | Very limited: few public datasets |

### Entity Pool Gaps

**Known gaps** (where templates need improvement):

1. **Podcast show names**: Only ~45 popular shows hardcoded; many podcasts missing
   - **Fix**: Add `load_from_podcastindex_dataset()` or user-generated list

2. **Radio stations**: Only BBC/NPR/SomaFM; regional stations missing
   - **Fix**: Add user's TuneIn/iHeartRadio library via API

3. **Game titles**: Wikidata has ~10K games but sparse; many indie games missing
   - **Fix**: Supplement with `load_from_steam_api()` or local Steam library

4. **Anime titles**: Good (500+) but missing new shows
   - **Fix**: Add `load_from_myanimelist()` dataset

5. **TV shows**: Only ~450 hardcoded; many regional shows missing
   - **Fix**: Integrate user's Sonarr library automatically

---

## Workflow

### Quickstart: Generate from Local Templates

```bash
# 1. Create a templates directory with MediaType-specific CSV files
mkdir templates/
# Add templates/music.csv, templates/movie.csv, etc. (see below)

# 2. (Optional) Export from your media servers
python scripts/generate_dataset_from_media.py \
  --radarr http://localhost:7878 --radarr-key YOUR_KEY \
  --sonarr http://localhost:8989 --sonarr-key YOUR_KEY \
  --output media_entities.csv

# 3. Generate synthetic data
python -m ovos_media_classifier.train.generate_synthetic \
  --templates-dir templates/ \
  --media-csv media_entities.csv \
  --max-per-intent 500 \
  --output synthetic.csv

# 4. Combine with existing data
python scripts/build_dataset.py \
  --input synthetic.csv \
  --dedup \
  --output ocp_dataset.csv
```

### Advanced: Full Pipeline with HuggingFace

```bash
# 1. Download and cache HuggingFace datasets
python scripts/download_datasets.py

# 2. Generate from OCP templates + Wikidata (3000–5000 utterances per intent)
python -m ovos_media_classifier.train.generate_from_ocp_templates \
  --n 30 \
  --output ocp_wikidata.csv

# 3. Augment with local templates + media server entities
python -m ovos_media_classifier.train.generate_synthetic \
  --templates-dir templates/ \
  --media-csv media_entities.csv \
  --dedup-against ocp_wikidata.csv \
  --output local_synthetic.csv

# 4. Combine + deduplicate
cat ocp_wikidata.csv local_synthetic.csv > combined.csv
python scripts/build_dataset.py \
  --input combined.csv \
  --dedup \
  --balance \
  --output final_dataset.csv

# 5. (Optional) Augment with LLM paraphrasing for semi-supervised learning
python scripts/llm_augment.py \
  --input final_dataset.csv \
  --model gpt-4 \
  --augment-factor 2 \
  --output augmented_dataset.csv
```

---

## Output Schema

**All generators produce CSV with this schema**:

```csv
lang,domain,intent,binary_label,playback_label,media_label,sentence
en,ocp_play,music,ocp,audio,music,play some jazz by john coltrane
en,ocp_play,movie,ocp,video,movie,watch a film directed by christopher nolan
en,not_ocp,not_ocp,not_ocp,undefined,not_ocp,set an alarm for tomorrow
```

**Columns**:
- `lang` — Always "en" (currently; could expand for multilingual)
- `domain` — "ocp_play" | "ocp_control" | "not_ocp"
- `intent` — OCPPlayIntent value (music, movie, podcast, etc.)
- `binary_label` — "ocp" | "not_ocp" (domain-level; used for fast screening)
- `playback_label` — "audio" | "video" | "undefined" (modality hint)
- `media_label` — Intent name (same as column 3 for ocp_play)
- `sentence` — The actual utterance

---

## Quality Assurance

### Checks

1. **Deduplication**: Each sentence appears ≤ 1 time across all datasets
2. **Balance**: Each intent has ≥ 100 utterances (preferably 200–500)
3. **Coverage**: All required slots are filled (no `{slot_name}` left in output)
4. **No nulls**: Every column filled; no empty cells
5. **Length**: Utterances 5–100 characters (filters out malformed templates)

### Manual Review

For production models:
1. Sample 50 random utterances per intent
2. Verify semantic correctness (e.g., "play Naruto" for anime, not music)
3. Check for absurd combinations (e.g., "play The Beatles by Steven Spielberg")
4. Flag any OOV (out-of-vocabulary) or typos in entities

---

## Extending the Pipeline

### Add a New MediaType

1. **Create template file**: `templates/your_media_type.csv`
   ```csv
   category,template
   primary,"play {title}"
   primary,"watch {title}"
   secondary,"put on {title}"
   ```

2. **Add slot mapping** (if using Wikidata):
   - Edit `_WIKIDATA_SLOT_MAP` in `generate_from_ocp_templates.py`
   - Map entity types to your slots (e.g., `"your_entity_type": ["title"]`)

3. **Add label mapping**:
   - Edit `_LABEL_TO_OCP` in `generate_from_ocp_templates.py`
   - Map your media type to OCP intent (e.g., `"your_type": ("ocp_play", "your_intent")`)

4. **Test**:
   ```bash
   python -m ovos_media_classifier.train.generate_synthetic \
     --templates-dir templates/ \
     --max-per-intent 100 \
     --output test.csv
   ```

### Add a Custom Entity Source

Example: Load from a JSON file of your entities:

```python
# In generate_from_ocp_templates.py

def load_custom_entities(json_path: str) -> dict[str, list[str]]:
    import json
    with open(json_path) as f:
        data = json.load(f)
    pools = {}
    for slot, values in data.items():
        pools[slot] = values
    return pools

# Merge into all_pools in generate_from_templates():
custom_pools = load_custom_entities("my_entities.json")
for slot, values in custom_pools.items():
    if slot in all_pools:
        all_pools[slot] = list(dict.fromkeys(all_pools[slot] + values))
    else:
        all_pools[slot] = values
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "Many templates skipped (missing slots)" | Entity pools incomplete | Add more entities to CSV or enable HF Wikidata |
| "Generated dataset has duplicates" | `--dedup-against` not used | Specify existing dataset to dedup against |
| "Unbalanced dataset (intent X has 50, intent Y has 500)" | Different # of templates per intent | Add more templates for underrepresented intents |
| "Silly combinations (e.g., 'play Mozart by Tom Cruise')" | Bad slot independence | Review template–entity pairings; use hierarchical sampling |
| "Missing domain entity for my media library" | Entity loader not enabled | Use `--media-csv` flag or `load_from_media_servers()` |
| "HuggingFace download slow/fails" | No internet or cache full | Run `download_datasets.py` offline first; check `~/.cache` size |

---

## Performance Tips

1. **Reduce template count**: 100 templates × 10 fills = 1000 utterances. Increase fills rather than templates for better diversity.

2. **Parallel entity loading**: `load_wikidata_pools()` is slow (1.6M entities). Cache the result and reuse.

3. **Limit entity pool size**: Use `max_per_type=10000` in `load_wikidata_pools()` to cap memory usage.

4. **Dedup smartly**: Store existing sentences in a set (O(1) lookup) rather than list.

5. **Seed for reproducibility**: All generators use `random.seed(42)` by default; change with `--seed`.

---

## References

- [OpenVoiceOS OCP_templates dataset](https://huggingface.co/datasets/OpenVoiceOS/OCP_templates)
- [Jarbas/WikidataMediaEntities](https://huggingface.co/datasets/Jarbas/WikidataMediaEntities)
- [Jarbas NER datasets](https://huggingface.co/Jarbas)
- `scripts/generate_synthetic.py` — Templates + entity pools
- `scripts/generate_from_ocp_templates.py` — Large-scale Wikidata pipeline
- `scripts/generate_dataset_from_media.py` — Media server extraction
- `scripts/build_dataset.py` — Final assembly and training
