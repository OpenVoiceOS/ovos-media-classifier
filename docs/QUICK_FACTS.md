
# QUICK_FACTS — ovos-media-classifier

## Package Identity

| Field | Value |
|-------|-------|
| **Package Name** | `ovos-media-classifier` |
| **Version** | `0.0.1a1` |
| **License** | Apache-2.0 |
| **Python** | `>=3.10` |
| **Author** | OpenVoiceOS |
| **Repository** | https://github.com/OpenVoiceOS/ovos-media-classifier |

## Entry Points

No OPM plugin entry points — this is a **library package**, not a plugin.

Public API imported via:

```python
from ovos_media_classifier import load_media_classifier
```

## Core Abstractions

| Symbol | Module | Role |
|--------|--------|------|
| `AbstractMediaClassifier` | `ovos_media_classifier.base` | ABC defining the classifier contract |
| `load_media_classifier()` | `ovos_media_classifier.__init__` | Factory — 6-tier fallback chain |
| `EntitiesContainer` | `ovos_media_classifier.entities` | Runtime entity registry; loaders for media servers |
| `MediaType` | `ovos_media_classifier.intents` | Enum of all supported media categories |
| `OCPDomain` | `ovos_media_classifier.intents` | High-level media domain enum |
| `OCPPlayIntent` | `ovos_media_classifier.intents` | Play intent enum |
| `OCPControlIntent` | `ovos_media_classifier.intents` | Playback control intent enum |
| `OCPEntityLabel` | `ovos_media_classifier.intents` | NER entity label enum |

## Backends (in fallback order)

| Backend | Module | Extra | Strategy |
|---------|--------|-------|----------|
| `Model2VecMediaClassifier` | `ovos_media_classifier.m2v` | `[m2v]` | Hierarchical neural (domain + intent heads) |
| `SklearnMediaClassifier` | `ovos_media_classifier.sklearn` | `[sklearn]` | TF-IDF + LogisticRegression |
| `AhocorasickMediaClassifier` | `ovos_media_classifier.ahocorasick` | `[ner]` | NER substring matching |
| `PadatiousMediaClassifier` | `ovos_media_classifier.padatious` | `[padatious]` | padatious/padacioso intent |
| `KeywordMediaClassifier` | `ovos_media_classifier.keyword` | (none) | .voc file keyword matching |

## Optional Dependency Extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[m2v]` | `torch>=2.0.0`, `model2vec>=0.3.0` | Neural M2V backend |
| `[sklearn]` | `scikit-learn>=1.3.0`, `joblib>=1.3.0`, `numpy>=1.24.0` | TF-IDF backend |
| `[padatious]` | `ovos-padatious>=0.1.0` | Padatious backend |
| `[ner]` | `ahocorasick-ner>=0.1.1` | NER backend |
| `[media_servers]` | `requests>=2.28.0` | Radarr/Sonarr/Lidarr/Jellyfin entity loaders |
| `[huggingface]` | `datasets>=2.0.0` | HuggingFace entity loader |
| `[all]` | all of the above | Full install |
| `[train]` | all + `pandas`, `matplotlib`, `seaborn`, `tqdm` | Training utilities |

## Locale Support

14 language directories under `ovos_media_classifier/locale/`:
`en-us`, `fr-fr`, `de-de`, `es-es`, `pt-pt`, `it-it`, `nl-nl`, `ca-es`, `da-dk`, `sv-se`, `ru-ru`, `uk-ua`, `zh-zh`, `ja-jp`

Each locale contains `.voc` (keyword lists) and `.intent` (padatious training samples).

## Key Mappings (intents.py)

| Mapping | Type | Purpose |
|---------|------|---------|
| `PLAY_INTENT_TO_MEDIA_TYPE` | `Dict[OCPPlayIntent, MediaType]` | Intent → MediaType |
| `MEDIA_TYPE_TO_PLAY_INTENT` | `Dict[MediaType, OCPPlayIntent]` | MediaType → Intent |
| `LABEL_TO_MEDIA_TYPE` | `Dict[OCPEntityLabel, MediaType]` | NER label → MediaType |
| `NER_LABEL_TO_PLAY_INTENT` | `Dict[OCPEntityLabel, OCPPlayIntent]` | NER label → Intent |

## Version File

`ovos_media_classifier/version.py` — uses `START_VERSION_BLOCK`/`END_VERSION_BLOCK` markers; `__version__` derived from `VERSION_MAJOR.VERSION_MINOR.VERSION_BUILD[aN]`.

## Test Coverage

Tests: `test/test_classifier.py` — 47 unit tests covering: `KeywordMediaClassifier`, `Model2VecMediaClassifier`, `AbstractMediaClassifier` contract, `load_media_classifier` factory.

Gaps: No tests for `AhocorasickMediaClassifier`, `SklearnMediaClassifier`, `PadatiousMediaClassifier`, `EntitiesContainer` loaders.
