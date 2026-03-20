
# ovos-media-classifier — Documentation Index

`ovos-media-classifier` is a pluggable media-type classification library for the [OCP (OVOS Common Play)](https://github.com/OpenVoiceOS/ovos-core) subsystem. It classifies natural-language utterances into media types (music, movie, podcast, etc.) and play intents using a tiered set of ML and keyword-based backends.

## Quick Start

```bash
pip install ovos-media-classifier[m2v]
```

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()
result = clf.classify("play some jazz")
# {"media_type": MediaType.MUSIC, "intent": OCPPlayIntent.PLAY_MUSIC, "conf": 0.92}
```

## Documentation Files

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Quick-start guide, installation, basic usage examples |
| [THEORY.md](THEORY.md) | Classification concepts, ML background, design rationale |
| [BACKENDS.md](BACKENDS.md) | Deep dive on each backend (keyword, NER, sklearn, padatious, m2v) |
| [TRAINING.md](TRAINING.md) | Dataset generation, model training, benchmarks |
| [NER_LABELS.md](NER_LABELS.md) | Complete entity label taxonomy |
| [LANG_SUPPORT.md](LANG_SUPPORT.md) | Language support guide, adding new locales |
| [MAINTAINERS_GUIDE.md](MAINTAINERS_GUIDE.md) | Release workflow, repository layout, dependency policy |
| [OCP_TAXONOMY_SPEC.md](OCP_TAXONOMY_SPEC.md) | Full OCP media type taxonomy specification |
| [MEDIA_SERVER_INTEGRATIONS.md](MEDIA_SERVER_INTEGRATIONS.md) | Radarr, Sonarr, Lidarr, Jellyfin, Whisparr entity loaders |

## Architecture Overview

```
load_media_classifier()          ← factory; 6-tier fallback
        │
        ├── Model2VecMediaClassifier  [m2v]      ← neural, two-headed (domain + intent)
        ├── SklearnMediaClassifier    [sklearn]   ← TF-IDF + LogisticRegression
        ├── AhocorasickMediaClassifier [ner]      ← NER substring matching
        ├── PadatiousMediaClassifier  [padatious] ← padatious/padacioso intent matching
        └── KeywordMediaClassifier    (bundled)   ← .voc file keyword lookup (always available)

AbstractMediaClassifier             ← ABC defining the contract
EntitiesContainer                   ← runtime entity registry (media server loaders)
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `ovos_media_classifier.__init__` | Public API, `load_media_classifier()` factory |
| `ovos_media_classifier.base` | `AbstractMediaClassifier` ABC |
| `ovos_media_classifier.intents` | `MediaType`, `OCPDomain`, `OCPPlayIntent`, `OCPEntityLabel` enums + mappings |
| `ovos_media_classifier.entities` | `EntitiesContainer` runtime entity registry |
| `ovos_media_classifier.keyword` | Keyword backend (`.voc` files) |
| `ovos_media_classifier.ahocorasick` | NER backend |
| `ovos_media_classifier.sklearn` | sklearn backend |
| `ovos_media_classifier.padatious` | padatious backend |
| `ovos_media_classifier.m2v` | Model2Vec neural backend |
| `ovos_media_classifier.models` | `StaticModelForHierarchicalClassification` architecture |
| `ovos_media_classifier.features` | `CategoricalFeatureExtractor`; defines `_KEYWORD_VOCABS` (41 entries), `_ENTITY_LABEL_VALUES` |
| `ovos_media_classifier.constants` | Centralized confidence threshold defaults |
| `ovos_media_classifier.train` | Dataset generation package (all pipeline steps) |
| `ovos_media_classifier.train.build_dataset` | Master pipeline orchestrator (`ovos-ocp-build-dataset` CLI) |
| `ovos_media_classifier.train.download_datasets` | Download CSV + HuggingFace sources |
| `ovos_media_classifier.train.generate_categorical_features` | NER+keyword feature extraction (`ovos-ocp-gen-features` CLI) |
| `ovos_media_classifier.train.train_guided_embeddings` | Train ONNX guided-embeddings models (`ovos-ocp-train-guided` CLI) |
| `ovos_media_classifier.train.explore_dataset` | EDA + train/val/test splits (`ovos-ocp-explore` CLI) |

## Cross-Package References

- **OCP integration**: `ovos-core` — OCP pipeline consumes classifier results
- **Plugin discovery**: `ovos-plugin-manager` — this package is a library, not an OPM plugin
- **Entity sources**: `ahocorasick-ner` (`ahocorasick_ner`) — NER container used by `AhocorasickMediaClassifier`

## Maintenance

- `QUICK_FACTS.md` — machine-readable reference for RAG retrieval
- `FAQ.md` — common questions and troubleshooting
- `AUDIT.md` — known issues, technical debt, security notes
- `MAINTENANCE_REPORT.md` — date-stamped change log
- `SUGGESTIONS.md` — proposed improvements
