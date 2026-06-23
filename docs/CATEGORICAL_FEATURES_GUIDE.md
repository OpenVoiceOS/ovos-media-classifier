# Categorical Features Dataset Guide

## Overview

The categorical features dataset (`categorical_features.parquet`) is a machine learning-ready dataset containing 1.9M utterances with binary NER (Named Entity Recognition) feature columns.

**Dataset Location**: `~/.cache/ovos-media-classifier/output/categorical_features.parquet`  
**Size**: 34 MB (Parquet format, 1,912,507 rows × 35 columns)  
**Generation Time**: ~10 minutes with 8-worker multiprocessing

## Dataset Structure

### Original Columns (7)
| Column | Type | Description |
|--------|------|-------------|
| `sentence` | str | Utterance text |
| `intent` | str | Intent label (music, movie, tv, podcast, etc.) |
| `lang` | str | BCP-47 language code (en-us, de-de, fr-fr, etc.) |
| `domain` | str | Domain classification |
| `binary_label` | int | Binary OCP vs non-OCP classification |
| `media_label` | str | Media type label |
| `playback_label` | str | Playback instruction label |

### NER Feature Columns (28)
Binary (0/1) indicators for entity detection:

**Music Entities**:
- `artist_name` - Artist/performer mentions
- `track_name` - Song/track mentions
- `album_name` - Album mentions
- `music_genre` - Genre keywords (jazz, rock, pop, etc.)

**Movie/Video Entities**:
- `movie_title` - Movie/film title mentions
- `movie_actor` - Actor name mentions
- `movie_director` - Director mentions
- `video_genre` - Video genre (action, drama, comedy, etc.)

**TV Entities**:
- `tv_show_title` - TV show/series mentions
- `tv_genre` - TV genre classification

**Podcast Entities**:
- `podcast_title` - Podcast show mentions
- `podcast_host` - Host name mentions

**Audiobook Entities**:
- `audiobook_title` - Audiobook title mentions
- `audiobook_author` - Author name mentions
- `audiobook_narrator` - Narrator name mentions

**Radio Entities**:
- `radio_station` - Station name/frequency mentions
- `radio_genre` - Radio format (news, talk, jazz, etc.)

**Game Entities**:
- `game_title` - Game title mentions
- `game_genre` - Game type (RPG, shooter, strategy, etc.)
- `game_platform` - Platform mentions (PC, Xbox, PlayStation, etc.)

**Anime Entities**:
- `anime_title` - Anime series mentions
- `anime_studio` - Studio/production company mentions

**News Entities**:
- `news_provider` - News source/outlet mentions
- `news_category` - News category (sports, weather, politics, etc.)

**Structural/Heuristic Features** (5):
- `proper_noun` - Presence of capitalized multi-word phrases
- `temporal_reference` - Year/date detection (19XX, 20XX)
- `quoted_title` - Presence of quotation marks
- `entity_indicator` - Presence of entity keywords (by, from, starring, with, featuring)

## Feature Coverage Statistics

Top 15 features by prevalence in dataset:

| Feature | Count | Percentage |
|---------|-------|-----------|
| artist_name | 976,055 | 51.0% |
| entity_indicator | 866,446 | 45.3% |
| album_name | 199,893 | 10.5% |
| music_genre | 57,133 | 3.0% |
| movie_title | 3,836 | 0.2% |
| audiobook_title | 3,236 | 0.2% |
| anime_title | 1,614 | 0.1% |
| game_title | 2,521 | 0.1% |
| audiobook_author | 2,233 | 0.1% |
| track_name | 1,876 | 0.1% |
| tv_show_title | 1,189 | 0.1% |
| podcast_title | 412 | 0.0% |
| movie_actor | 812 | 0.0% |
| game_genre | 676 | 0.0% |
| proper_noun | 1,203,847 | 63.0% |

## Generation Script

### Location
`scripts/generate_categorical_features_fast.py`

### Key Features
1. **Multiprocessing**: Uses CPU-count workers for parallel feature extraction
2. **Checkpoint/Resume**: Saves progress every 10 batches; can resume on interruption with `--resume` flag
3. **Flexible Output**: Supports both CSV and Parquet formats
4. **Progress Tracking**: Real-time progress updates every 10k rows

### Usage Examples

**Full dataset generation**:
```bash
python scripts/generate_categorical_features_fast.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --output categorical_features.parquet \
    --format parquet \
    --workers 8 \
    --batch-size 10000
```

**Small sample for testing**:
```bash
python scripts/generate_categorical_features_fast.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --output test_features.parquet \
    --format parquet \
    --sample 10000 \
    --workers 4
```

**Resume from checkpoint**:
```bash
python scripts/generate_categorical_features_fast.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --output categorical_features.parquet \
    --format parquet \
    --workers 8 \
    --resume
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | REQUIRED | Input CSV path |
| `--output` | REQUIRED | Output file path |
| `--format` | csv | Output format (csv or parquet) |
| `--workers` | CPU count | Number of parallel workers |
| `--batch-size` | 10000 | Rows per batch |
| `--sample` | None | Limit to N rows (for testing) |
| `--resume` | False | Resume from checkpoint if available |

## Feature Extraction Methods

### Pattern Matching (20+ Entity Types)

Each entity type has 1-2 hand-crafted regex patterns designed to detect:
- Intent-specific keywords (play, artist, track, album, movie, etc.)
- Proper noun patterns (capitalized multi-word phrases)
- Genre and platform keywords

**Example patterns**:
```python
"artist_name": [
    r"(play|by|from|artist|musician)\s+([A-Z][a-zA-Z\s&.,\-']+)",
    r"(music by|performed by|artist)\s+(\w+)",
]
```

### Heuristics

Additional binary features computed from sentence structure:

1. **Capitalization**: Detects proper nouns for media-centric intents
2. **Temporal References**: Detects years (1900-2099)
3. **Quotation Marks**: Indicates titles in quotes
4. **Entity Indicators**: Presence of connector words (by, from, starring, with, featuring)

## Machine Learning Applications

### Classification Tasks
- **Intent Prediction**: Predict media type from utterance + features
- **OCP Detection**: Binary classification (OCP vs non-OCP content)
- **Entity Presence**: Multilabel classification for entity types

### Feature Selection
The 28 binary features are useful for:
- **Low-dimensional baseline models**: Logistic regression, decision trees
- **Feature engineering input**: Starting point for more complex models
- **Interpretability**: Direct correspondence to entity types

### Dimensionality
- **Dense**: 51% of rows have >3 features detected
- **Sparse**: Specific entities (news, podcast, game) have <1% coverage
- **Imbalanced**: Music entities dominate (51% artist detection)

## Data Quality Notes

### Missing Data
- NaN sentences and intents filtered before feature extraction
- All feature columns contain only 0/1 values (no NaNs)

### Language Distribution
Dataset spans 13 languages:
- `en-us` (English): ~40% of data
- `de-de` (German), `fr-fr` (French), `es-es` (Spanish), `it-it` (Italian): ~10% each
- `pt-br`, `pt-pt`, `nl-nl`, `pl-pl`, `ca-es`, `eu`, `gl-es`, `da-dk`: remaining ~5%

### Unresolved Slots
Original dataset had 2.9% unresolved slots (unfilled entity placeholders).
This dataset does NOT include partially filled rows — all sentences are complete.

## Parquet vs CSV

### Parquet Format (Recommended)
- **Size**: 34 MB (compressed)
- **Speed**: 2-3x faster I/O
- **Columns**: Type information preserved (int, str, etc.)
- **Tools**: pandas, polars, DuckDB, Apache Spark

### CSV Format
- **Size**: 200+ MB (uncompressed)
- **Compatibility**: Works with all tools (Excel, R, Python)
- **Human-readable**: Column types inferred on load

## HuggingFace Dataset Preparation

### Train/Val/Test Split

Generate 80/10/10 stratified split by language and intent:
```bash
python scripts/explore_dataset.py \
    --input ~/.cache/ovos-media-classifier/output/ocp_final.csv \
    --split-output ~/.cache/ovos-media-classifier/output/splits/ \
    --format parquet
```

Output:
- `splits/train_features.parquet` (1.5M rows)
- `splits/val_features.parquet` (190K rows)
- `splits/test_features.parquet` (190K rows)

### Dataset Card Template

```yaml
# Dataset: ovos-ocp-utterances-categorical-features

## Description
1.9M multilingual utterances with automatically extracted NER entity features.
Binary feature columns indicate presence of 28+ entity types (artist, track, movie, etc.).

## Citation
@dataset{ovos_categorical_features,
  title = {OVOS OCP Utterances - Categorical NER Features},
  author = {OpenVoiceOS},
  year = {2026},
  url = {https://huggingface.co/datasets/OpenVoiceOS/ovos-ocp-utterances-categorical}
}
```

## Performance Benchmarks

### Generation Speed
- **1.9M rows**: ~10 minutes (8 workers, batch size 10k)
- **Per-row overhead**: ~5ms
- **Bottleneck**: Regex pattern matching (not I/O)

### Storage
- **Parquet**: 34 MB (17.8 bytes/row on average)
- **CSV**: 200 MB (105 bytes/row)
- **Compression ratio**: 6:1 (Parquet + Snappy)

## Known Limitations

1. **Heuristic-based**: Features rely on regex patterns, not full NLP parsing
2. **English-centric**: Pattern vocabulary optimized for English
3. **Intent-dependent**: Certain features only extracted for specific intents
4. **Low recall**: Rare entities (news, podcast) have <1% detection
5. **False positives**: "By" keyword matches author, performer, director, date (ambiguous)

## Future Improvements

1. **Deep learning features**: LSTM/BERT embeddings for entity detection
2. **Language-specific patterns**: Translate patterns to all 13 languages
3. **Dependency parsing**: Use syntactic structure for entity boundaries
4. **Cross-lingual transfer**: Apply English patterns to non-English text with adaptation
5. **Hard negatives**: Include feature-less rows for contrastive learning

## Related Files

| File | Purpose |
|------|---------|
| `DATA_SOURCES.md` | Gap analysis + entity pool completion strategy |
| `DATASET_PIPELINE.md` | End-to-end pipeline documentation |
| `scripts/explore_dataset.py` | Dataset analysis and train/val/test split generation |
| `ovos_media_classifier/intents.py` | Entity label definitions + intent mappings |

