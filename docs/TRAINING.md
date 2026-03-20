# Training Guide

This document explains how to reproduce or extend the trained OCP classifier
models.  It covers dataset collection, sklearn training, Model2Vec training,
and NER dataset construction.

---

## Prerequisites

```bash
pip install ovos-media-classifier[train]
# Installs: scikit-learn, joblib, torch, model2vec, pandas,
#           matplotlib, seaborn, tqdm, ahocorasick-ner, datasets
```

---

## Full pipeline — one command

`build_dataset.py` (in the repo root) orchestrates all steps end-to-end:

```bash
# Full build — downloads everything, generates all sources, merges, plots
uv run python build_dataset.py

# Skip downloading (use cached data)
uv run python build_dataset.py --skip-download

# Only recompute metrics/plots for an existing ocp_final.csv
uv run python build_dataset.py --only metrics

# Pull from local media servers via env vars
RADARR_URL=http://localhost:7878 RADARR_API_KEY=KEY \
SONARR_URL=http://localhost:8989 SONARR_API_KEY=KEY \
uv run python build_dataset.py
```

Steps run in order: download → gather → templates → keyword → synthetic → media → merge → metrics.
Each step deduplicates against the previous output.  Use `--skip-*` to
bypass individual steps.  Final output: `<cache>/output/ocp_final.csv`.

---

## Individual steps

### Step 1 — Gather the dataset

`ovos_media_classifier.train.gather_dataset` downloads multilingual intent
datasets, normalises them, and maps every `(domain, intent)` pair to an OCP
label.

```bash
python -m ovos_media_classifier.train.gather_dataset
```

This writes to `<cache>/output/`:

```
output/
  ocp_dataset.csv              # full dataset: lang, domain, intent, sentence
  ocp_dataset_play_only.csv    # ocp_play rows only
  by_lang/
    ocp_en.csv
    ocp_de.csv
    ...
  dataset_plots/
    domain_distribution.png
    play_intent_distribution.png
    control_intent_distribution.png
    lang_distribution.png
    lang_domain_heatmap.png
```

### Data sources

| Source | Content | Label assigned |
|---|---|---|
| `OpenVoiceOS/ovos-common-query-intents` | General OVOS intents (multi-lang) | Mapped by `_intent_to_ocp_label()` |
| `OpenVoiceOS/ovos-intents-massive-subset` | MASSIVE dataset subset | Mapped |
| `OpenVoiceOS/ovos-llm-augmented-intents` | LLM-augmented intents | Mapped |
| `lang-support-tracker/intents_*.csv` | Per-language OVOS intents | Mapped |
| `Jarbas/music_queries_*` | Music query templates | Forced to `ocp_play:music` |
| `OpenVoiceOS/ovos-weather-intents` | Weather queries | `not_ocp` |
| Language-specific PT/CA/ES/NL test sets | Test intents | Mapped |

### Label mapping

`_intent_to_ocp_label(domain, intent)` performs the mapping:

1. If `domain` is in `OCP_PLAY_SKILL_DOMAINS` or starts with `"ocp"`:
   - Scan `_CONTROL_INTENT_PATTERNS` first (control takes priority over play).
   - Then scan `_PLAY_INTENT_PATTERNS`.
   - Default: `ocp_play:generic`.
2. If `domain` is a known core skill with control-style intent: `ocp_control`.
3. Otherwise: `not_ocp`.

To add support for a new OCP skill domain, add its name to
`OCP_PLAY_SKILL_DOMAINS` in `gather_dataset.py`.

### CSV schema

```
lang,domain,intent,sentence
en,ocp_play,music,play something by pink floyd
en,ocp_control,pause,pause the music
en,not_ocp,not_ocp,what is the weather tomorrow
```

---

## Step 2a — Train the sklearn classifier

```bash
# Default: TF-IDF + LogisticRegression (fast, good accuracy)
python -m ovos_media_classifier.train.train_ocp_sklearn

# Single-level guided embeddings (BoW + MLP) alongside TF-IDF
python -m ovos_media_classifier.train.train_ocp_sklearn --guided

# Two-level hierarchical guided embeddings (OCP-specific, saves guided model)
python -m ovos_media_classifier.train.train_ocp_sklearn --hierarchical
```

### Options

```
--csv PATH       Dataset CSV (default: output/ocp_dataset.csv)
--out PATH       Output joblib path (default: output/ocp_sklearn.joblib)
--guided         Also train a single-level LabelGuidedEmbeddingsTransformer
                 alongside TF-IDF; saves the TF-IDF model.
--hierarchical   Train a two-level MultiLabelGuidedEmbeddingsTransformer
                 (domain guide → play guide); saves the guided model.
                 Mutually exclusive with --guided.
```

### What is trained

**Default / `--guided` mode:**

Two TF-IDF pipelines are trained and saved together:

1. **Domain pipeline** — trained on all rows
   - `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` → `LogisticRegression(C=5.0)`

2. **Play pipeline** — trained only on `ocp_play` rows
   - Same structure, predicts fine-grained `OCPPlayIntent` labels

When `--guided` is also specified the script trains a `LabelGuidedEmbeddingsTransformer`
for each head (domain and play) and evaluates it for comparison, but still saves
the TF-IDF model.  Use this flag to compare the two approaches without
committing to the guided model.

**`--hierarchical` mode:**

Trains a `MultiLabelGuidedEmbeddingsTransformer` with two chained embedders —
the original approach from the beta OCP classifier:

```
embedder[0] guided by domain labels  →  domain embeddings (shape: [N, 3*n_domain])
embedder[1] guided by play labels    →  play embeddings on domain-enriched features
```

Both a domain head and a play head are fitted on top of the respective
embedding levels.  This is saved as the output model.

### Guided categorical embeddings

Implemented in `ovos_media_classifier.embeddings` (no external dependencies
beyond scikit-learn and numpy).

The core idea:

```python
# 1. Convert sentence to boolean BoW feature dict
sentence = "play some pink floyd"
features = {"contains_play": True, "contains_some": True,
            "contains_pink": True, "contains_floyd": True}

# 2. Train MLP to predict the label from these features
mlp.fit(X_dict, y_labels)

# 3. Extract hidden-layer activations as dense embeddings
embeddings = mlp.coefs_[0] @ X + mlp.intercepts_[0]  # ReLU → guided repr.

# 4. Optional PCA compression
pca.fit_transform(embeddings)
```

The MLP learns that `contains_pink` and `contains_floyd` co-occur with `music`,
so the hidden activations cluster music queries together even if no single word
is in the vocabulary.  TF-IDF would give those tokens equal weight regardless
of their label association.

### Evaluating

The training script prints accuracy, macro-F1, and per-class F1 to stdout,
and saves plots to `output/sklearn_plots/`:

```
output/sklearn_plots/
  domain_confusion_matrix.png
  play_intent_confusion_matrix.png
  domain_per_class_f1.png
  play_intent_per_class_f1.png
  domain_pca.png
  domain_tsne.png
  play_intent_pca.png          # (TF-IDF feature space)
  play_intent_tsne.png
  domain_guided_pca.png        # (--guided)
  domain_guided_tsne.png
  play_guided_pca.png
  play_guided_tsne.png
  domain_hierarchical_pca.png  # (--hierarchical)
  domain_hierarchical_tsne.png
  play_hierarchical_pca.png
  play_hierarchical_tsne.png
```

### Loading the trained model

```python
from ovos_media_classifier.sklearn import SklearnMediaClassifier

clf = SklearnMediaClassifier.from_path("output/ocp_sklearn.joblib")
media_type, conf = clf.classify("play some jazz", "en-us")
```

The joblib file includes a `"mode"` key (`"tfidf"`, `"guided"`, or
`"hierarchical_guided"`) for introspection.  The hierarchical model also
includes `"multi_transformer"` — the fitted `MultiLabelGuidedEmbeddingsTransformer`
— which can be inspected or used to generate embeddings for new data.

---

## Step 2b — Train the Model2Vec classifier

The hierarchical Model2Vec model uses `StaticModelForHierarchicalClassification`
from `ovos_media_classifier.models`.  A training script is planned at
`ovos_media_classifier/train/train_ocp_hierarchical.py`.

Until that script is written, training can be done programmatically:

```python
from ovos_media_classifier.models import StaticModelForHierarchicalClassification
import pandas as pd

df = pd.read_csv("output/ocp_dataset.csv")
X = df["sentence"].tolist()
y_domain = df["domain"].tolist()
y_intent = df["intent"].tolist()

model = StaticModelForHierarchicalClassification.from_pretrained(
    "minishlab/potion-base-8M"   # or any model2vec model
)
model.fit(X, y_domain, y_intent)   # requires: pip install lightning
model.save("output/m2v_ocp_model")
```

Loading:

```python
from ovos_media_classifier.m2v import Model2VecMediaClassifier
clf = Model2VecMediaClassifier.from_path("output/m2v_ocp_model")
```

---

## Step 3 — Generate a dataset from your media library (optional)

If you run media servers (Radarr, Sonarr, Lidarr, Jellyfin, etc.) you can
generate a **wide-format CSV** of your actual library.  Every row is one
media item; columns capture relational metadata (actors, directors, artists,
album, studio, …) alongside the `ocp_label` and `media_type`.

```bash
pip install ovos-media-classifier[media_servers]

python generate_dataset_from_media.py --output media_dataset.csv \
    --radarr-url     http://localhost:7878 --radarr-api-key   KEY \
    --sonarr-url     http://localhost:8989 --sonarr-api-key   KEY \
    --lidarr-url     http://localhost:8686 --lidarr-api-key   KEY \
    --readarr-url    http://localhost:8787 --readarr-api-key  KEY \
    --jellyfin-url   http://localhost:8096 --jellyfin-api-key KEY \
    --music-assistant-url http://localhost:8095
```

Output columns: `title`, `ocp_label`, `media_type`, `genre`, `actor`,
`director`, `producer`, `writer`, `composer`, `artist`, `album`, `author`,
`studio`, `source`.  Multi-value fields are pipe-separated.

Jellyfin fetches only `Movie,Series,MusicAlbum,MusicArtist,Audio` by default
(audiobooks and podcasts excluded due to poor Jellyfin metadata).  Use
`--readarr-url` for audiobooks and `--jellyfin-types` to override.

See [MEDIA_SERVER_INTEGRATIONS.md](MEDIA_SERVER_INTEGRATIONS.md) for the full
source reference, field mapping, and instructions for converting the wide CSV
to the flat `entity,label` format accepted by `EntitiesContainer.load_csv()`.

---

## Step 4 — Build the NER database (optional, for AhocorasickNER backend)

```python
from ovos_media_classifier.train.ner_datasets import OCPMediaNER

# Downloads from HuggingFace and saves to disk (~1-2 GB of entity data)
ner = OCPMediaNER(path="output/ocp_media.ahocorasick")
```

This pre-populates the NER with:
- Music: artist names, track names, album names, music genres, record labels
  (metal, jazz, progressive, classical, trance genres)
- Film: actor, director, producer, writer, composer names from IMDB datasets
- Streaming services: hard-coded well-known service names

The resulting `.ahocorasick` file can be loaded at runtime:

```python
from ahocorasick_ner import AhocorasickNER
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

ner = AhocorasickNER()
ner.load("output/ocp_media.ahocorasick")
clf = AhocorasickMediaClassifier(ner)
```

### Extending the NER

To add your own entities (e.g. from a personal music library):

```python
clf.add_word("artist_name", "My Favourite Local Band")
clf.add_word("movie_title", "My Indie Film")
```

Entity label strings must match the canonical values in `OCPEntityLabel`
(see [NER_LABELS.md](NER_LABELS.md)).

---

## Data augmentation tips

### Class imbalance

The dataset typically has far more `not_ocp` examples than `ocp_play` and
`ocp_control`.  Within `ocp_play`, `music` and `generic` dominate.  Options:

- Use `class_weight="balanced"` in `LogisticRegression`.
- Oversample rare classes using `imbalanced-learn`.
- Add synthetic examples for rare types (asmr, visual_story, bw_movie, etc.)

### Shared dataset sources

All URL lists and intent classification sets live in one place:
`ovos_media_classifier/train/sources.py`.  Both `gather_dataset.py` and
`download_datasets.py` import from there — no more duplicate lists.

Key exports:
- `CSV_SOURCES` — general HuggingFace intent CSVs
- `MUSIC_CSV_SOURCES` — music query CSVs (forced to `ocp_play:music`)
- `GITHUB_CSV_SOURCES` — per-language GitHub intent CSVs
- `ALL_CSV_SOURCES` — union of all three (used by `download_datasets.py`)
- `HF_DATASETS` — HuggingFace NER + entity datasets
- `AUDIO_INTENTS`, `VIDEO_INTENTS` — shared playback classification sets
- `SCHEMA_COLUMNS` — standard column order for all output CSVs

### Adding more training data

1. Add a new URL to `CSV_SOURCES` or `MUSIC_CSV_SOURCES` in `sources.py`.
2. Add a `forced_domain` / `forced_intent` override in `gather_dataset.py`
   if the source doesn't follow the standard column format.
3. Re-run `build_dataset.py --skip-download` and re-train.

### Language expansion

Add your language's BCP-47 code to `GITHUB_CSV_LANGS` in `sources.py` and
provide a corresponding intent CSV at the expected GitHub URL.

---

## Benchmark all backends

A comprehensive benchmark script comparing all backends on the OCP test set is
planned.  In the meantime, quick ad-hoc evaluation:

```python
from ovos_media_classifier import load_media_classifier
import pandas as pd

df = pd.read_csv("output/ocp_dataset.csv").sample(500, random_state=42)
clf = load_media_classifier(
    config={"media_classifier_sklearn_model": "output/ocp_sklearn.joblib"}
)

correct = 0
for _, row in df.iterrows():
    pred_domain, _ = clf.classify_domain(row["sentence"], row["lang"])
    if pred_domain.value == row["domain"]:
        correct += 1

print(f"Domain accuracy: {correct / len(df):.2%}")
```
