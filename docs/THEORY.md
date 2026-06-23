# Theory — How OCP Media Classification Works

> **Superseded.** The current, authoritative design is the orthogonal multi-axis
> model in [classification-model.md](classification-model.md); the shipped public
> API and contract are in [stable-api.md](stable-api.md) and
> [taxonomy.md](taxonomy.md). This page is retained as background design notes:
> it predates the multi-axis model (it describes a two-level hierarchy) and
> discusses several ML backends — sklearn, padatious, Model2Vec, NER — that are
> **not** part of this release; those land later as
> [external plugins](external-plugins.md). Treat it as history, not as the spec.

This document explains the conceptual foundation of `ovos-media-classifier`:
the classification problem it solves, the two-level hierarchy it uses, how
confidence scores are computed, and why each backend exists.

---

## 1. The classification problem

When a user says something to a voice assistant, the assistant must decide what
to do.  Most utterances are general — _"set a timer for ten minutes"_, _"what's
the capital of France?"_ — but a significant fraction are requests to play some
kind of media.  These are **OCP queries**.

OCP (OpenVoiceOS Common Play) is the media subsystem of OVOS.  Its pipeline
must answer two questions for every utterance:

**Q1 — Is this an OCP query?**
_"play something by Pink Floyd"_ → yes
_"what is the weather tomorrow?"_ → no

**Q2 — If yes, what kind of media?**
_"put on the news"_ → `news`
_"stream Naruto"_ → `anime`
_"I want to hear a podcast about history"_ → `podcast`

Answering Q1 wrong is expensive: a false positive routes a general query to OCP
(confusing the user) or blocks another skill from handling it; a false negative
means media the user wanted is never played.

Answering Q2 wrong means the wrong OCP skills are prioritised — a music-only
skill gets a movie request, or vice versa.

---

## 2. The two-level hierarchy

The classifier models this as a **hierarchical** decision with two levels:

```
Level 1 — Domain
  ocp_play     → the user wants to play media      (go to Level 2)
  ocp_control  → the user wants to control playback (pause/next/…)
  not_ocp      → unrelated to media                (hand off to another pipeline stage)

Level 2 — Play intent  (only when domain = ocp_play)
  music            podcast          radio
  audiobook        news             movie
  tv_show          video            video_episodes
  audio            game             anime
  cartoon          documentary      short_film
  silent_movie     bw_movie         radio_theatre
  visual_story     asmr             audio_description
  adult            adult_audio      hentai
  generic          (uncertain / mixed)
```

The enumerations are defined in `ovos_media_classifier.intents`:

```python
from ovos_media_classifier.intents import OCPDomain, OCPPlayIntent, OCPControlIntent
```

### Why two levels?

- **Performance**: most backends can cheaply answer "is this OCP at all?" before
  running the more expensive fine-grained classifier.
- **Accuracy**: `ocp_control` (pause/next/stop) looks very different from
  `ocp_play` (play X by Y) and from general queries.  Separating them reduces
  confusion between classes.
- **Graceful degradation**: backends without a domain head (e.g. keyword) derive
  the domain from the play result — if a media type is matched, the domain is
  `ocp_play`; otherwise it is `not_ocp`.

---

## 3. The AbstractMediaClassifier interface

Every backend implements the same three methods:

```python
class AbstractMediaClassifier:

    def classify(query, lang, valid_labels=None) -> (MediaType, float):
        """Fine-grained media type + confidence."""

    def classify_domain(query, lang) -> (OCPDomain, float):
        """Top-level domain + confidence.
        Default: derived from classify() result."""

    def is_ocp_query(query, lang) -> (bool, float):
        """True if domain != NOT_OCP.
        Default: derived from classify_domain()."""
```

The base class provides default implementations of `classify_domain` and
`is_ocp_query` that delegate to `classify`.  Backends with a dedicated domain
head override `classify_domain` directly.

---

## 4. Confidence scores

Every prediction is paired with a **confidence score** in `[0.0, 1.0]`.

| Score | Meaning |
|---|---|
| `0.0` | No match (GENERIC fallback) |
| `0.4–0.6` | Keyword or NER match — medium certainty |
| `0.7–0.9` | ML model prediction — good certainty |
| `1.0` | Reserved for perfect matches (rarely used) |

Confidence is not a calibrated probability in the statistical sense — it is a
heuristic signal used by the OCP pipeline to rank competing skill candidates.
A media type predicted with confidence `0.3` might still be acted on if no
higher-confidence result is available.

The OCP pipeline plugin applies thresholds (configurable per backend) to discard
low-confidence predictions before acting.

---

## 5. Backends — a conceptual overview

### 5.1 Keyword (zero-dependency)

The simplest backend.  It reads `.voc` vocabulary files (one keyword per line)
and checks whether any keyword appears in the lowercased utterance.

```
"play some jazz" → contains "jazz"? no → contains "music"? no
                 → contains "song"? no → contains "soundtrack"? no
                 → … → no match → GENERIC
```

Limitations:
- Requires the keyword to appear literally in the utterance.
- Language-dependent (needs translated `.voc` files for each locale).
- Cannot handle paraphrases or implicit intent.

Strengths:
- Zero extra dependencies.
- Fully deterministic and transparent.
- Works offline without a model file.

### 5.2 AhocorasickNER (named entity matching)

Uses the `ahocorasick-ner` library, an Aho-Corasick automaton, to detect
**named entities** (artist names, film titles, streaming service names) in the
utterance.

```
"play some pink floyd" → NER hit: "pink floyd" → label: artist_name
                       → artist_name maps to OCPPlayIntent.MUSIC → MediaType.MUSIC
```

The automaton is populated at two times:
- **Training time**: from HuggingFace datasets (millions of artist names, film
  directors, etc.) — see `MusicNER`, `ImdbNER`, `OCPMediaNER` in
  `ovos_media_classifier.train.ner_datasets`.
- **Runtime**: OCP skills register their own content via the bus message
  `ovos.common_play.register_keyword`.  A music skill registers the artists and
  albums it can play; a streaming service skill registers its service name.

This makes the NER classifier **personalised to the user's actual media
library**.  If the user has no anime skill installed, anime titles will not be
registered and anime queries will not be matched — which is correct behaviour.

Entity label taxonomy: see [NER_LABELS.md](NER_LABELS.md).

### 5.3 scikit-learn (TF-IDF + LogisticRegression)

Trains two sklearn pipelines from a labelled CSV:
- **Domain classifier**: all examples → `ocp_play` / `ocp_control` / `not_ocp`
- **Play classifier**: only `ocp_play` rows → fine-grained media type

Features are TF-IDF bigrams (sublinear term frequency, L2 norm).  The
classifier is LogisticRegression (fast, well-calibrated confidence via
`predict_proba`).

Alternative training mode uses **guided categorical embeddings**: boolean
word-presence features (`{"contains_music": True, "contains_play": True, …}`)
fed through an MLP hidden layer, giving dense semantic embeddings for words.
This approach was used for the original beta OCP classifier.

Saved as a `joblib` dict; loaded by `SklearnMediaClassifier.from_path()`.

### 5.4 Padatious / padacioso

Trains an `IntentContainer` on `.intent` files — padatious-style patterns with
optional slots (`{query}`, `{artist}`, etc.).

The bundled locale files (ported from the OCP pipeline plugin) provide
training patterns in 13 languages.  The `from_locale_dir()` factory loads
them automatically.

Two containers are trained:
- **Play container**: pattern per `OCPPlayIntent` value.
- **Domain container** (optional): patterns for `ocp_play` / `ocp_control` / `not_ocp`.

Falls back from `ovos-padatious` (fast, Cython) to `padacioso` (pure Python).

### 5.5 Model2Vec (hierarchical neural)

The highest-accuracy backend.  Uses a `StaticModelForHierarchicalClassification`
that embeds utterances with a `model2vec` static embedding model and runs two
heads:
- **Domain head**: outputs `ocp_play` / `ocp_control` / `not_ocp`
- **Intent head**: outputs fine-grained media type

The domain head is used in `classify_domain()` directly (cheap, no need to also
run the intent head).  The intent head runs only when the domain is `ocp_play`.

Static embeddings (no GPU needed at inference) make this practical on
resource-constrained devices like a Raspberry Pi.

---

## 6. The training pipeline

```
HuggingFace datasets          OVOS GitHub intents CSVs
        │                              │
        └──────────┬───────────────────┘
                   ▼
           gather_dataset.py
           (normalise, re-label to OCP schema)
                   │
        ┌──────────┼─────────────┐
        ▼          ▼             ▼
  domain CSV   play-only CSV   per-lang CSVs
        │          │
        ▼          ▼
  train_ocp_sklearn.py         train_ocp_hierarchical.py
  (TF-IDF pipelines)           (Model2Vec heads)
        │                              │
        ▼                              ▼
  ocp_sklearn.joblib            m2v_model/
```

The dataset pipeline:
1. Downloads multilingual OVOS intent datasets and music query datasets.
2. Normalises column names and text.
3. Maps each `(domain, intent)` pair to an `(OCPDomain, OCPPlayIntent)` label
   via `_intent_to_ocp_label()`.
4. Deduplicates and saves to CSV.

---

## 7. Language handling

### Keyword backend
Language is explicit: `.voc` files live in `locale/<lang-tag>/`.  The
`_VocMatcher` tries the exact BCP-47 tag first (`en-us`) then the base language
(`en`).

### ML backends
The trained models are language-agnostic at inference time — the model2vec
embeddings and TF-IDF features operate on raw text.  The `lang` parameter is
passed through but not currently used to select a sub-model.  Training data is
multilingual; adding more language data improves multilingual coverage.

### Padatious backend
Patterns are loaded per language.  Call `from_locale_dir(locale_dir, lang="de-de")`
to get a German classifier.

---

## 8. The label taxonomy

All canonical label strings live in `ovos_media_classifier.intents`:

| Enum | Purpose |
|---|---|
| `OCPDomain` | Top-level routing: `ocp_play`, `ocp_control`, `not_ocp` |
| `OCPPlayIntent` | Fine-grained media type: `music`, `movie`, `podcast`, … |
| `OCPControlIntent` | Playback action: `pause`, `next`, `stop`, … |
| `OCPEntityLabel` | NER entity label names: `artist_name`, `movie_title`, … |

Mappings between these and `ovos_utils.ocp.MediaType` are pre-built in the
same module (`PLAY_INTENT_TO_MEDIA_TYPE`, `NER_LABEL_TO_PLAY_INTENT`, …).
All backends use these shared mappings so label strings are consistent
everywhere.
