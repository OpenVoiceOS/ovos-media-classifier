
# FAQ — ovos-media-classifier

## Installation

**Q: What is the minimal install for production use?**

```bash
pip install ovos-media-classifier[m2v]
```
This installs the `Model2VecMediaClassifier` (highest accuracy). Falls back to `KeywordMediaClassifier` if torch/model2vec are unavailable.

**Q: How do I install all backends?**

```bash
pip install ovos-media-classifier[all]
```

**Q: How do I install the GuidedEmbeddings ONNX backend (no torch at runtime)?**

```bash
pip install ovos-media-classifier[guided]
```

Requires a pre-trained model directory (produced by `ovos-ocp-train-guided` CLI).
Config key: `media_classifier_guided_model`.

**Q: What install do I need for training a guided-embeddings model?**

```bash
pip install ovos-media-classifier[train]
ovos-ocp-train-guided \
    --input categorical_features.parquet \
    --output /path/to/models/guided
```

---

## Usage

**Q: How do I get a classifier instance?**

Use the factory — it automatically selects the best available backend:

```python
from ovos_media_classifier import load_media_classifier
clf = load_media_classifier()
```

**Q: How do I force a specific backend?**

Pass `backend=` to `load_media_classifier()`:

```python
from ovos_media_classifier import load_media_classifier
clf = load_media_classifier(backend="keyword")   # "keyword", "sklearn", "padatious", "ner", "m2v"
```

**Q: What does the classifier return?**

A `dict` with:
- `"media_type"`: `MediaType` enum value
- `"intent"`: `OCPPlayIntent` enum value
- `"domain"`: `OCPDomain` enum value (M2V backend only)
- `"conf"`: float confidence 0.0–1.0

**Q: How do I check if a query is an OCP media request at all?**

```python
clf.is_ocp_query("play some jazz")   # True
clf.is_ocp_query("what time is it")  # False
```

**Q: How do I restrict predictions to specific media types?**

Pass `valid_labels=` to `classify()`:

```python
from ovos_media_classifier.intents import MediaType
result = clf.classify("play some jazz", valid_labels=[MediaType.MUSIC, MediaType.RADIO])
```

---

## Backends

**Q: Which backend is most accurate?**

`Model2VecMediaClassifier` (neural, two-headed architecture). Requires `[m2v]` extra and a trained model file.

**Q: Which backend works with no ML dependencies?**

`KeywordMediaClassifier` — uses only `.voc` files bundled in the package. Zero extra dependencies.

**Q: What is the fallback order?**

`m2v → sklearn → ner (ahocorasick) → padatious → keyword`

The factory tries each in order and uses the first that is available and configured.

**Q: Can I use multiple backends together?**

Not directly. Use `AhocorasickMediaClassifier` which combines NER entity detection with fallback intent classification.

---

## Entity Registration

**Q: How do I register runtime entities (e.g. my movie library)?**

```python
from ovos_media_classifier import EntitiesContainer
ents = EntitiesContainer()
ents.register_movie("Blade Runner")
ents.register_movie("Dune")
```

**Q: How do I load entities from a Jellyfin/Radarr/etc. media server?**

```python
from ovos_media_classifier import EntitiesContainer
ents = EntitiesContainer()
ents.load_from_jellyfin("http://localhost:8096", api_key="YOUR_KEY")
```

Requires `[media_servers]` extra.

**Q: How do I load entities from a HuggingFace dataset?**

```python
ents.load_from_huggingface("some-org/media-dataset")
```

Requires `[huggingface]` extra.

---

## Localization

**Q: What languages are supported?**

14 languages: `en-us`, `fr-fr`, `de-de`, `es-es`, `pt-pt`, `it-it`, `nl-nl`, `ca-es`, `da-dk`, `sv-se`, `ru-ru`, `uk-ua`, `zh-zh`, `ja-jp`.

**Q: How do I add a new language?**

See `docs/LANG_SUPPORT.md`. Add `.voc` and `.intent` files under `ovos_media_classifier/locale/<lang>/`.

---

## Training

**Q: How do I train a new sklearn or m2v model?**

See `docs/TRAINING.md`. Use the CLI entry points installed with the package (`ovos-ocp-build-dataset`, `ovos-ocp-gen-features`, `ovos-ocp-train-guided`) to generate datasets, then run training.

**Q: Where are training templates stored?**

`ovos_media_classifier/train/templates/*.csv` — 27 template files covering all media types.

---

## Troubleshooting

**Q: `load_media_classifier()` always returns a `KeywordMediaClassifier` — why?**

The factory falls back to keyword when all other backends fail. Check that optional dependencies are installed (`pip install ovos-media-classifier[m2v]`) and a trained model file exists.

**Q: `classify()` returns `MediaType.GENERIC` — why?**

The classifier could not match the utterance with sufficient confidence. Try a different backend or lower the confidence threshold via `threshold=` parameter.

**Q: I get `ImportError` for `ahocorasick_ner` — what do I do?**

```bash
pip install ovos-media-classifier[ner]
```

**Q: Where is the M2V model file loaded from?**

By default from `~/.local/share/ovos/media_classifier/`. Pass `model_path=` to `Model2VecMediaClassifier()` to override.

---

## Categorical Features Dataset

**Q: What is the categorical features dataset?**

A machine learning-ready dataset with 1.9M utterances and 28 NER entity features (binary columns). Located at `~/.cache/ovos-media-classifier/output/categorical_features.parquet`.

**Q: How large is the dataset and what format is it in?**

- **Size**: 34 MB (Apache Parquet, Snappy compression)
- **Rows**: 1,912,507 utterances
- **Columns**: 35 (7 original + 28 NER features)
- **Alternative**: CSV format available (~200 MB uncompressed)

**Q: What are the 28 NER features?**

Binary (0/1) indicators for entity presence:

*Music Entities (4)*: artist_name, track_name, album_name, music_genre
*Movie/Video (4)*: movie_title, movie_actor, movie_director, video_genre
*TV (2)*: tv_show_title, tv_genre
*Podcast (2)*: podcast_title, podcast_host
*Audiobook (3)*: audiobook_title, audiobook_author, audiobook_narrator
*Radio (2)*: radio_station, radio_genre
*Game (3)*: game_title, game_genre, game_platform
*Anime (2)*: anime_title, anime_studio
*News (2)*: news_provider, news_category
*Structural (5)*: proper_noun, entity_indicator, temporal_reference, quoted_title, + one duplicate

**Q: What is the feature coverage?**

Top features by prevalence:
- artist_name: 51.0% (most common — music-heavy dataset)
- entity_indicator: 45.3% (keywords: by, from, starring, with, featuring)
- track_name: 27.7%
- quoted_title: 19.7% (titles in quotes)
- album_name: 10.5%
- music_genre: 3.0%
- Others: <2% (rare entities)

**Q: How do I generate the categorical features dataset?**

```bash
# Full dataset (1.9M rows, ~10 minutes)
python scripts/generate_categorical_features_fast.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --output categorical_features.parquet \
    --format parquet \
    --workers 8

# Small sample for testing (10k rows, <30 seconds)
python scripts/generate_categorical_features_fast.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --output test_features.parquet \
    --format parquet \
    --sample 10000 \
    --workers 4
```

**Q: What script generates the categorical features?**

`scripts/generate_categorical_features_fast.py` — optimized multiprocessing implementation with checkpoint/resume support. See `CATEGORICAL_FEATURES_GUIDE.md` for full documentation.

**Q: Does the script support checkpoint/resume?**

Yes! Use `--resume` flag:

```bash
python scripts/generate_categorical_features_fast.py \
    --input ... --output ... --format parquet --resume
```

Progress is saved every 10 batches. If interrupted, just re-run with `--resume` to continue from the last checkpoint.

**Q: What entity detection methods are used?**

1. **Regex Pattern Matching** (20+ patterns): Hand-crafted patterns for each entity type
   - Example: `r"(play|by|from|artist|musician)\s+([A-Z][a-zA-Z\s&.,\-']+)"` for artist_name

2. **Intent-Specific Filtering**: Only extract relevant entities for each intent
   - Music intent → check for artist_name, track_name, album_name, music_genre
   - Movie intent → check for movie_title, movie_actor, movie_director, video_genre

3. **Heuristics**: 
   - Capitalization for proper nouns
   - Year detection (19XX, 20XX)
   - Quotation marks for titles
   - Entity indicator keywords

**Q: What are the limitations of the categorical features?**

1. Heuristic-based (regex, not full NLP parsing)
2. English-centric pattern vocabulary
3. Intent-dependent extraction (sparse for rare intents)
4. Low recall for rare entities (news, podcast, game <1%)
5. No entity boundaries (presence only, not spans)
6. Ambiguous patterns ("by" = author/performer/director)

**Q: How do I use the dataset for machine learning?**

For classification tasks:
```python
import pandas as pd
df = pd.read_parquet('categorical_features.parquet')

# Split by language/intent
train = df[df['lang'] == 'en-us']

# Use feature columns for input
feature_cols = df.columns[7:]
X = df[feature_cols].values
y = df['binary_label'].values  # or 'media_label', etc.

# Train any sklearn model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
```

**Q: What language distribution does the dataset have?**

13 languages:
- English (en/en-us): ~90%
- Catalan (ca): ~4%
- Portuguese (pt): ~1%
- Galician (gl), German (de), Spanish (es), Italian (it), others: <1% each

**Q: How do I prepare the dataset for HuggingFace?**

1. Generate train/val/test split:
```bash
python scripts/explore_dataset.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --split-output splits/ --format parquet
```

2. Create dataset card (README.md) with metadata

3. Upload to HuggingFace Hub:
```bash
huggingface-cli login
huggingface-cli upload OpenVoiceOS/ovos-ocp-utterances-categorical splits/ \
    --repo-type dataset
```

See `CATEGORICAL_FEATURES_GUIDE.md` → "HuggingFace Preparation" section.

**Q: What is the performance of the generation script?**

- **Speed**: ~10 minutes for 1.9M rows (with 8 workers)
- **Per-row**: ~5ms overhead
- **Bottleneck**: Regex pattern matching (not I/O)
- **Throughput**: ~3,200 rows/second

**Q: What are the feature statistics?**

- Average features per row: 1.6
- Feature density: 5.72% (sparse matrix)
- Total feature activations: 3.06M
- Densest: artist_name (976k activations)
- Sparsest: anime_studio (1 activation)

**Q: Can I extend the features?**

Yes! Add patterns to `ENTITY_PATTERNS` dict in the script:

```python
ENTITY_PATTERNS["my_custom_entity"] = [
    re.compile(r"my_pattern_1", re.IGNORECASE),
    re.compile(r"my_pattern_2", re.IGNORECASE),
]

# Add to intent mapping
INTENT_ENTITIES["my_intent"].append("my_custom_entity")
```

**Q: Is the dataset multilingual?**

Yes, but pattern matching is English-centric. Non-English utterances are processed with English patterns (low recall). Future enhancement: translate patterns to all 13 languages.

**Q: What's the next step for this dataset?**

1. ✓ Generate categorical features
2. ✓ Create comprehensive documentation
3. Generate train/val/test splits
4. Upload to HuggingFace Hub
5. Add BERT embeddings (optional)
6. Build baseline ML models

---

## Dataset Generation — Where Is the Logic?

**Q: Where does dataset generation logic live?**

All dataset generation logic lives in `ovos_media_classifier/train/`:

| Module | Role |
|--------|------|
| `train/download_datasets.py` | Download CSV + HuggingFace sources to cache |
| `train/gather_dataset.py` | Normalise CSV sources → `ocp_gathered.csv` |
| `train/generate_from_ocp_templates.py` | Fill Wikidata templates → `ocp_templates.csv` |
| `train/generate_keyword_csv.py` | Keyword-based utterances → `ocp_keyword.csv` |
| `train/generate_synthetic.py` | Multilingual synthetic utterances → `ocp_synthetic.csv` |
| `train/generate_dataset_from_media.py` | Pull from media servers → `ocp_media.csv` |
| `train/generate_categorical_features.py` | NER + keyword features → `categorical_features.parquet` |
| `train/explore_dataset.py` | EDA plots and train/val/test splits |
| `train/train_guided_embeddings.py` | Train ONNX guided-embeddings models |
| `train/build_dataset.py` | Master orchestrator — runs all steps in order |

**Q: How do I run a single pipeline step standalone?**

Each module is runnable as `python -m ovos_media_classifier.train.<module>`:

```bash
# Download only
python -m ovos_media_classifier.train.download_datasets --dry-run

# Generate keyword data only
python -m ovos_media_classifier.train.generate_keyword_csv --help

# Generate categorical features only
python -m ovos_media_classifier.train.generate_categorical_features \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --output categorical_features.parquet --workers 8
```

Or use the installed CLI entry points:

```bash
ovos-ocp-build-dataset         # full pipeline
ovos-ocp-gen-features --help   # categorical features
ovos-ocp-explore --help        # EDA + splits
ovos-ocp-train-guided --help   # train ONNX models
```

**Q: Why are constants like `_KEYWORD_VOCABS` defined in `features.py` instead of `generate_categorical_features.py`?**

`features.py` is the single source of truth because these constants are needed at **both** training time (`train/generate_categorical_features.py`) and **inference** time (`CategoricalFeatureExtractor`). Defining them in a training script would break the runtime feature extractor.

**Q: Where are confidence threshold defaults defined?**

`ovos_media_classifier/constants.py` — all classifiers import from there. Do not add magic numbers inline.

---

## Dataset Pipeline

**Q: How do I run the entire dataset generation pipeline?**

Use the unified `dataset.py` tool:

```bash
# Run full pipeline (all steps)
python dataset.py build

# Run specific step
python dataset.py synthesize --langs en-us,de-de,fr-fr --synthetic-n 1000

# Run with custom parameters
python dataset.py download --force
python dataset.py synthesize --langs all --workers 8
```

**Q: What steps does the pipeline include?**

1. `download` — Download HuggingFace datasets (entities, non-OCP utterances)
2. `gather` — Combine downloaded data
3. `templates` — Process template CSVs
4. `keyword` — Generate keyword vocabulary
5. `synthesize` — Generate synthetic utterances (multilingual)
6. `media` — Process media server data
7. `merge` — Combine all datasets
8. `metrics` — Generate quality metrics
9. `explore` — Dataset exploration + train/val/test split
10. `build` — Run all steps in sequence

**Q: Where are the dataset outputs stored?**

`~/.cache/ovos-media-classifier/output/` directory:
- `ocp_final.csv` — Final merged dataset (1.9M rows)
- `categorical_features.parquet` — Categorical features dataset
- `ocp_synthetic.csv` — Synthetic utterances only
- `ocp_keyword.csv` — Keyword-based data
- Other intermediate files

**Q: How do I skip certain pipeline steps?**

```bash
# Skip download, gather, templates (use cached versions)
python dataset.py synthesize \
    --skip-download --skip-gather --skip-templates \
    --langs en-us,de-de,fr-fr \
    --synthetic-n 1000
```

**Q: How long does the full pipeline take?**

- Full pipeline (all 10 steps): ~45-60 minutes
- Synthesis only (all 13 languages): ~15-20 minutes
- Categorical features generation: ~10 minutes
- Exploration/metrics: ~5 minutes

---

## Entity Pool Scraping

**Q: How do I expand the entity pools used for NER and synthetic data generation?**

Use the scraper script:

```bash
# Download all free public sources and merge into ocp_entities.csv
python scripts/scrape_entity_sources.py

# Specific sources only (fast)
python scripts/scrape_entity_sources.py --sources radio_garden,steam

# Dry-run to preview counts
python scripts/scrape_entity_sources.py --dry-run
```

**Q: What sources does `scrape_entity_sources.py` support?**

| Source key | Entity labels | Auth | Size |
|------------|---------------|------|------|
| `gutendex` | `audiobook_title`, `audiobook_author` | None | ~78k books |
| `librivox` | `audiobook_title`, `audiobook_author`, `audiobook_narrator` | None | ~20k books |
| `radio_garden` | `radio_station`, `radio_genre` | None | ~550 stations |
| `anime_offline_db` | `anime_title` | None | ~20k titles (when available) |
| `anilist` | `anime_title`, `anime_studio` | None | ~900 titles/run |
| `steam` | `game_title` | None | ~990 games (SteamSpy) |
| `open_library` | `audiobook_title`, `audiobook_author` | None | ~8k books |

All sources are free public APIs — no API key required.

**Q: What is the ocp_entities.csv schema?**

```
title, ocp_label, media_type, genre, actor, director, producer, writer,
composer, artist, album, author, narrator, studio, source
```

- `title` — the entity string used for NER substring matching
- `ocp_label` — the `OCPEntityLabel` value (e.g. `artist_name`, `movie_title`)
- `source` — provenance tag (e.g. `gutendex`, `radarr`)

All other columns are optional metadata and may be emitted as secondary NER entries.

**Q: How does the entity pool affect inference accuracy?**

`AhocorasickMediaClassifier` matches entity substrings at inference time:
- Larger pools → more utterances trigger NER-based classification
- Entity matches are intentionally "greedy" — false positives are expected (e.g.
  "cheese" → `movie_title` because "Cheese" is a real film)
- The downstream ML model learns to disambiguate from context; this is by design

**Q: How do I add a source without a public API (e.g. IGDB, Listen Notes)?**

Sources that require API keys can still use `scrape_entity_sources.py`:
1. Register for a free API key at the provider
2. Write a `fetch_<name>()` function using `_get()` / `_post_json()` helpers
3. Pass the key as a module-level constant or environment variable
4. Add to `_FETCHERS` and `_ALL_SOURCES`

See `docs/ENTITY_SOURCES.md` for the full guide.

**Q: How do I populate adult and hentai entity pools?**

Run the `adult` source — it combines three datasets automatically:

```bash
python scripts/scrape_entity_sources.py --sources adult
```

This produces:
- `adult_title` — 3,877+ titles (from Whisparr/media-server data)
- `pornstar` — 3,295+ performer names (extracted from actor column)
- `hentai_title` — 900+ titles (AniList `isAdult=true` filter)
- `adult_streaming_service` — 562 studios/services (curated + AniList)
- `porn_genre` — 102 genre tags (curated: amateur, BDSM, cosplay, etc. + hentai-specific)

**Q: Are there templates for adult and hentai utterances?**

Yes — `templates/en-us/adult.csv` and `templates/en-us/hentai.csv`.

Adult slots: `{title}`, `{pornstar}`, `{genre}`, `{studio}`, `{service}`
Hentai slots: `{title}`, `{genre}`, `{studio}`

Sample templates:
```
play a video with {pornstar}
I want to watch {genre} hentai
find something from {studio}
play the hentai {title}
```

**Q: Can I merge entity CSVs from media servers (Radarr, Sonarr, Jellyfin)?**

Yes. The `EntitiesContainer` loaders produce compatible two-column CSVs:

```python
from ovos_media_classifier import EntitiesContainer
ents = EntitiesContainer()
ents.load_from_radarr("http://localhost:7878", api_key="YOUR_KEY")
# Export to CSV:
import csv
with open("radarr_entities.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["entity", "label"])
    for label, words in ents._wordlists.items():
        for word in words:
            writer.writerow([word, label])
```

Then merge with `--merge scripts/ocp_entities.csv`.

---

## Known Issues & Limitations

**Q: Some entities are very sparse (<1% coverage) — why?**

The dataset is heavily weighted toward music (86.5% of utterances are music queries). Rare intents (news, podcast, game) have low feature activation rates. This is expected behavior — use intent filtering if needed.

**Q: Non-English utterances have very low entity coverage — why?**

Entity detection patterns are optimized for English. Non-English patterns are future work. For now, apply language-specific preprocessing or use multilingual embeddings.

**Q: The dataset has 2.9% unresolved slots in the original data — how do I handle them?**

The categorical features dataset does NOT include partially-filled rows. All sentences are complete. For the raw synthetic data with unfilled slots, see `DATA_SOURCES.md` for entity pool completion strategy.

