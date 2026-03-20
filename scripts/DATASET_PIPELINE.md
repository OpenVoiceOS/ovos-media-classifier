# Unified Dataset Pipeline

The `dataset.py` script consolidates all dataset generation, processing, and analysis into a single unified tool with subcommands.

## Quick Start

### Full Pipeline (All Steps)
```bash
# Run complete pipeline with defaults
python dataset.py build

# Run with custom parameters
python dataset.py build --synthetic-n 5000 --langs en-us,de-de,fr-fr

# Skip certain steps
python dataset.py build --skip download --skip gather --skip templates
```

### Individual Steps

```bash
# Download datasets (CSV + HuggingFace)
python dataset.py download

# Normalize downloaded CSVs
python dataset.py gather

# Fill OCP Wikidata templates
python dataset.py templates --templates-n 20

# Generate keyword-based utterances
python dataset.py keyword --keyword-n 3000

# Synthesize utterances (multilingual)
python dataset.py synthesize --langs en-us,de-de,fr-fr --synthetic-n 5000

# Extract from local media servers
python dataset.py media

# Merge and deduplicate all CSVs
python dataset.py merge

# Compute statistics and generate plots
python dataset.py metrics

# Analyze dataset composition
python dataset.py explore --input ~/.cache/ovos-media-classifier/output/ocp_final.csv --plots-dir plots/
```

## Multilingual Support

The `synthesize` subcommand supports generating utterances in multiple languages:

```bash
# Generate for all 13 supported languages
python dataset.py synthesize \
  --langs en-us,de-de,fr-fr,es-es,it-it,pt-br,pt-pt,nl-nl,pl-pl,ca-es,eu,gl-es,da-dk \
  --synthetic-n 5000

# Generate for subset of languages
python dataset.py synthesize --langs en-us,de-de,fr-fr --synthetic-n 3000

# Default is English only
python dataset.py synthesize --synthetic-n 5000  # en-us only
```

### Supported Languages

| Code   | Language                  |
|--------|---------------------------|
| en-us  | English (United States)   |
| de-de  | German                    |
| fr-fr  | French                    |
| es-es  | Spanish (Spain)           |
| it-it  | Italian                   |
| pt-br  | Portuguese (Brazil)       |
| pt-pt  | Portuguese (Europe)       |
| nl-nl  | Dutch                     |
| pl-pl  | Polish                    |
| ca-es  | Catalan                   |
| eu     | Basque                    |
| gl-es  | Galician                  |
| da-dk  | Danish                    |

## Command Reference

### `dataset.py build`
**Run complete pipeline (all steps)**

Options:
- `--only STEP` — Run only this step (download, gather, templates, keyword, synthesize, media, merge, metrics)
- `--skip-download` — Skip download step
- `--skip-gather` — Skip gather step
- `--skip-templates` — Skip templates step
- `--skip-keyword` — Skip keyword step
- `--skip-synthetic` — Skip synthesis step
- `--skip-media` — Skip media servers step
- `--skip-hf` — Skip HuggingFace datasets, use curated lists only
- `--templates-n N` — Samples per OCP template (default: 20)
- `--keyword-n N` — Utterances per keyword intent (default: 3000)
- `--synthetic-n N` — Utterances per synthetic intent (default: 5000)
- `--langs LANGS` — Comma-separated BCP-47 language codes (default: en-us)
- `--dry-run` — Dry-run mode (list files only)

Example:
```bash
python dataset.py build --skip-download --synthetic-n 3000 --langs en-us,de-de
```

### `dataset.py download`
**Download all external data sources**

Options:
- `--dry-run` — List files that would be downloaded (no actual download)

Downloads:
- CSV sources from configured URLs
- HuggingFace datasets for entity extraction
- Cached to `OVOS_MEDIA_CLASSIFIER_CACHE/csv/` and `hf_cache/`

### `dataset.py gather`
**Normalize and combine downloaded CSVs**

Outputs: `ocp_gathered.csv`

### `dataset.py templates`
**Fill OCP Wikidata templates**

Options:
- `--templates-n N` — Samples per template (default: 20)
- `--skip-hf` — Use curated lists only

Outputs: `ocp_templates.csv`

### `dataset.py keyword`
**Generate keyword-based utterances**

Options:
- `--keyword-n N` — Utterances per intent (default: 3000)

Outputs: `ocp_keyword.csv`

### `dataset.py synthesize`
**Synthesize utterances from templates + entities (multilingual)**

Options:
- `--langs LANGS` — Comma-separated language codes (default: en-us)
- `--synthetic-n N` — Utterances per intent (default: 5000)
- `--skip-hf` — Use curated lists only

Outputs: `ocp_synthetic.csv` with `lang` column

Example:
```bash
# Generate for 3 languages
python dataset.py synthesize --langs en-us,de-de,fr-fr --synthetic-n 3000

# Generate for all supported languages
python dataset.py synthesize --langs en-us,de-de,fr-fr,es-es,it-it,pt-br,pt-pt,nl-nl,pl-pl,ca-es,eu,gl-es,da-dk
```

### `dataset.py media`
**Extract from local media servers**

Extracts media information from:
- Radarr (movies)
- Sonarr (TV shows)
- Lidarr (music)
- Jellyfin (media libraries)
- Music Assistant
- AudioBookShelf
- Podgrab
- Kapowarr

Configuration via environment variables:
```bash
RADARR_URL=http://localhost:7878 RADARR_API_KEY=xxx python dataset.py media
```

Outputs: `ocp_media.csv`

### `dataset.py merge`
**Merge and deduplicate all CSVs**

Combines all individual CSVs (gathered, templates, keyword, synthetic, media) into:
- `ocp_final.csv` — deduplicated, ready for training

### `dataset.py metrics`
**Compute dataset statistics and generate plots**

Generates:
- `dataset_plots/dataset_overview.png` — overview plots
- stdout — summary statistics

### `dataset.py explore`
**Analyze dataset composition and generate visualizations**

Options:
- `--input PATH` — Input CSV path (required)
- `--plots-dir DIR` — Output plots directory (default: dataset_plots/explore/)
- `--lang LANG` — Filter to single language (optional)
- `--split-output DIR` — Create train/val/test split in this directory (optional)

Generates:
- `utterance_length_analysis.png` — length distribution
- `lang_intent_coverage.png` — language×intent heatmap (if multilingual)
- `train.csv`, `val.csv`, `test.csv` — stratified splits (if `--split-output` provided)
- stdout — detailed metrics

Example:
```bash
# Analyze final dataset with visualization
python dataset.py explore --input ~/.cache/ovos-media-classifier/output/ocp_final.csv --plots-dir plots/

# Create train/val/test split
python dataset.py explore --input ocp_final.csv --split-output splits/

# Analyze single language
python dataset.py explore --input ocp_final.csv --lang de-de
```

## Environment Variables

```bash
# Override cache directory
OVOS_MEDIA_CLASSIFIER_CACHE=/data/ocp

# Media server endpoints (for `media` step)
RADARR_URL=http://localhost:7878
RADARR_API_KEY=xxx
SONARR_URL=http://localhost:8989
SONARR_API_KEY=xxx
LIDARR_URL=http://localhost:8686
LIDARR_API_KEY=xxx
JELLYFIN_URL=http://localhost:8096
JELLYFIN_API_KEY=xxx
MUSIC_ASSISTANT_URL=http://localhost:8095
AUDIOBOOKSHELF_URL=http://localhost:8000
AUDIOBOOKSHELF_API_KEY=xxx
PODGRAB_URL=http://localhost:8080
KAPOWARR_URL=http://localhost:5656
KAPOWARR_API_KEY=xxx
```

## Output Structure

```
~/.cache/ovos-media-classifier/output/
├── ocp_gathered.csv           # Step 2: normalized CSVs
├── ocp_templates.csv          # Step 3: OCP Wikidata templates
├── ocp_keyword.csv            # Step 4: keyword utterances
├── ocp_synthetic.csv          # Step 5: synthetic utterances (multilingual)
├── ocp_media.csv              # Step 6: media server extracts
├── ocp_final.csv              # Step 7: merged & deduplicated
└── dataset_plots/
    ├── dataset_overview.png   # Step 8: overview plots
    └── explore/
        ├── utterance_length_analysis.png
        ├── lang_intent_coverage.png
        └── ...
```

## Examples

### Complete Pipeline with 3 Languages
```bash
python dataset.py build \
  --skip-download \
  --langs en-us,de-de,fr-fr \
  --synthetic-n 3000
```

### Just Synthesize Multiple Languages
```bash
python dataset.py synthesize \
  --langs en-us,de-de,fr-fr,es-es,it-it,pt-br,pt-pt,nl-nl,pl-pl,ca-es,eu,gl-es,da-dk \
  --synthetic-n 5000
```

### Merge Existing CSVs and Analyze
```bash
python dataset.py merge && \
python dataset.py metrics && \
python dataset.py explore --input ~/.cache/ovos-media-classifier/output/ocp_final.csv
```

### Create Train/Val/Test Split
```bash
python dataset.py explore \
  --input ocp_final.csv \
  --plots-dir plots/ \
  --split-output splits/
```

## Migration from Old Scripts

The old scripts are still available but deprecated:

| Old Script                  | New Command                      |
|-----------------------------|----------------------------------|
| `scripts/download_datasets.py` | `dataset.py download`            |
| `scripts/gather_dataset.py`    | `dataset.py gather`              |
| `scripts/generate_from_ocp_templates.py` | `dataset.py templates`   |
| `scripts/generate_keyword_csv.py` | `dataset.py keyword`        |
| `scripts/generate_synthetic.py` | `dataset.py synthesize`         |
| `scripts/generate_dataset_from_media.py` | `dataset.py media`    |
| `scripts/build_dataset.py`      | `dataset.py build`              |

All functionality is preserved in the unified tool.

## Troubleshooting

**Download fails**: Check network connectivity and URL configuration in `ovos_media_classifier/train/sources.py`

**Synthesis is slow**: First run downloads HuggingFace datasets; use `--skip-hf` for faster runs with curated lists only

**Memory issues**: Reduce `--synthetic-n` or process languages one at a time instead of all at once

**Missing templates**: Ensure `templates/{lang}/` directory exists with `.csv` files for each media type
