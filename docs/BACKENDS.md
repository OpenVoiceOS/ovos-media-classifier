# Backend Reference

> **Superseded.** This release ships **one** classifier — the bundled `.voc`
> keyword backend, documented in [backends.md](backends.md). The six backends
> described below (keyword, NER, sklearn, padatious, guided-embeddings,
> Model2Vec) are pre-refactor design notes; the ML backends are **not** packaged
> here and arrive later as [external plugins](external-plugins.md). For the
> current model and API see [classification-model.md](classification-model.md)
> and [stable-api.md](stable-api.md). Treat this page as history.

This document describes all six classifier backends in depth.  For the
conceptual overview see [THEORY.md](THEORY.md).

---

## Common interface

Every backend is a subclass of `AbstractMediaClassifier` and exposes:

```python
clf.classify(query: str, lang: str, valid_labels=None) -> (MediaType, float)
clf.classify_domain(query: str, lang: str)             -> (OCPDomain, float)
clf.is_ocp_query(query: str, lang: str)                -> (bool, float)
```

`lang` must be a BCP-47 tag (e.g. `"en-us"`, `"de-de"`).  ML backends ignore
it at inference time; keyword and padatious backends use it to select the right
locale file.

---

## 1. Keyword backend

**Class**: `ovos_media_classifier.keyword.KeywordMediaClassifier`
**Deps**: none
**Accuracy**: low–medium
**Speed**: very fast

### How it works

Checks whether any word from a `.voc` vocabulary file appears as a substring
of the (lowercased) utterance.  Checks are done in a fixed priority order that
resolves ambiguities (e.g. `AudioDramaKeyword` before `RadioKeyword`,
`MusicKeyword` before `MovieKeyword`).

### Two operation modes

**Pipeline mode** — pass a `voc_match_func` callable (the pipeline plugin's
`self.voc_match`).  The pipeline owns the locale files; the classifier just
calls the function.

```python
clf = KeywordMediaClassifier(voc_match_func=self.voc_match)
```

**Standalone mode** — no callable needed.  The classifier reads the bundled
`.voc` files from `ovos_media_classifier/locale/<lang>/`.

```python
clf = KeywordMediaClassifier()                          # bundled locale
clf = KeywordMediaClassifier.from_locale_dir("/my/dir") # custom locale
```

### Domain detection

The keyword backend has no separate domain head.  `classify_domain()` is
derived: if `classify()` returns something other than `GENERIC`, the domain is
`ocp_play`.  Control intents are **not** detected by this backend.

### Config keys

None — this backend is the fallback when no config key matches.

### Limitations

- Requires the keyword to appear literally in the query.
- Cannot detect paraphrases or implicit intent.
- Misses control intents entirely.

---

## 2. AhocorasickNER backend

**Class**: `ovos_media_classifier.ahocorasick.AhocorasickMediaClassifier`
**Deps**: `ahocorasick-ner` (`pip install ovos-media-classifier[ner]`)
**Accuracy**: medium–high (depends on NER coverage)
**Speed**: very fast (Aho-Corasick automaton, O(n) in utterance length)

### How it works

Uses the Aho-Corasick multi-pattern string matching algorithm.  The automaton
is pre-built from a dictionary mapping `{entity_label: [word, word, ...]}`.
When a query is tagged, every matched entity label is collected; the label with
the highest priority in `_INTENT_PRIORITY` wins.

```python
"play something by Metallica"
  → NER hit: "Metallica" → label: "artist_name"
  → NER_LABEL_TO_PLAY_INTENT["artist_name"] = OCPPlayIntent.MUSIC
  → MediaType.MUSIC, conf=0.6
```

### Entity label priority

When multiple entity types match, the backend resolves conflicts in a fixed
priority list (highest → lowest):

```
MOVIE_STREAMING_SERVICE, TV_STREAMING_SERVICE, MUSIC_STREAMING_SERVICE,
PODCAST_STREAMING_SERVICE, AUDIOBOOK_STREAMING_SERVICE,
RADIO_STREAMING_SERVICE, NEWS_PROVIDER,
MOVIE_TITLE, MOVIE_ACTOR, MOVIE_DIRECTOR, ...,
TV_SHOW_TITLE, ANIME_TITLE, CARTOON_TITLE, DOCUMENTARY_TITLE,
ARTIST_NAME, TRACK_NAME, ALBUM_NAME, ...,
PODCAST_TITLE, AUDIOBOOK_TITLE, GAME_TITLE, ASMR_ARTIST,
DOCUMENTARY_KEYWORD, MOVIE_KEYWORD, MUSIC_KEYWORD, ...
```

Named entities (like movie titles) beat generic keywords (like `"movie"`).

### EntitiesContainer — runtime-aware entity registry

The recommended way to build the AhocorasickNER backend is through an
`EntitiesContainer`.  The container and the classifier share the same
`AhocorasickNER` by reference, so every `container.add()` call propagates
immediately to `classify()` without any rebuild step.

```python
from ovos_media_classifier import EntitiesContainer, load_media_classifier

container = EntitiesContainer()

# Load from media servers (requires pip install ovos-media-classifier[media_servers])
container.load_radarr("http://localhost:7878",   api_key="…")
container.load_sonarr("http://localhost:8989",   api_key="…")
container.load_lidarr("http://localhost:8686",   api_key="…")
container.load_jellyfin("http://localhost:8096", api_key="…")
container.load_whisparr("http://localhost:6969", api_key="…")   # adult content
container.load_stash("http://localhost:9999",    api_key="…")   # GraphQL
container.load_music_assistant("http://localhost:8095")          # no key needed

# Load from a HuggingFace dataset (requires pip install ovos-media-classifier[huggingface])
container.load_huggingface("TigreGotico/ocp-entities")

# Load from a CSV produced by generate_dataset_from_media.py
container.load_csv("/path/to/entities.csv")

# Add inline word lists
container.add("music_streaming_service", "Spotify")
container.add("artist_name", "Radiohead")

# Build the classifier
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
clf = AhocorasickMediaClassifier.from_container(container)

# Runtime update — immediately effective, no rebuild:
container.add("movie_title", "Dune: Part Two")
clf.classify("play dune part two", "en-us")  # → (MediaType.MOVIE, 0.6)
```

#### Media server loaders

| Method | Source | Key entities extracted |
|---|---|---|
| `load_radarr(url, api_key)` | Radarr v3 | `movie_title`, `movie_actor`, `movie_director`, `movie_producer`, `movie_writer`, `movie_composer`, `movie_streaming_service` (studio) |
| `load_sonarr(url, api_key)` | Sonarr v3 | `tv_show_title`, `anime_title`, `documentary_title`, `cartoon_title`, `tv_streaming_service` (network) |
| `load_lidarr(url, api_key)` | Lidarr v1 | `artist_name`, `album_name`, `track_name`, `music_genre` |
| `load_jellyfin(url, api_key)` | Jellyfin | All of the above + `audiobook_title`, `audiobook_author`, `podcast_title` |
| `load_whisparr(url, api_key)` | Whisparr (Radarr fork) | `adult`, `movie_actor`, `porn_streaming_service` |
| `load_stash(url, api_key?)` | Stash (GraphQL) | `movie_actor` (performers), `porn_streaming_service` (studios), `adult` (scene titles) |
| `load_music_assistant(url)` | Music Assistant | `artist_name`, `album_name`, `track_name`, `radio_station`, `music_genre` |

Genre-aware label refinement is applied to movies and TV shows — items tagged
`anime`, `documentary`, or `animation` genres are stored under the appropriate
specific label (`anime_title`, `documentary_title`, `cartoon_title`) rather
than the generic `movie_title` / `tv_show_title`.

#### Config-driven construction

The factory function `load_media_classifier()` builds a container automatically
from the `media_classifier_entities` config key:

```python
clf = load_media_classifier(config={
    "media_classifier_entities": {
        "jellyfin":        {"url": "http://localhost:8096", "api_key": "…"},
        "radarr":          {"url": "http://localhost:7878", "api_key": "…"},
        "sonarr":          {"url": "http://localhost:8989", "api_key": "…"},
        "lidarr":          {"url": "http://localhost:8686", "api_key": "…"},
        "whisparr":        {"url": "http://localhost:6969", "api_key": "…"},
        "stash":           {"url": "http://localhost:9999", "api_key": "…"},
        "music_assistant": {"url": "http://localhost:8095"},
        "huggingface": [
            {"dataset": "TigreGotico/ocp-entities"}
        ],
        "csv": ["/path/to/extra.csv"],
        "wordlists": {"artist_name": ["Radiohead"]}
    }
})
```

### Static construction (no media servers)

```python
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

# From word lists
clf = AhocorasickMediaClassifier.from_wordlists({
    "artist_name": ["Pink Floyd", "Metallica", "The Beatles"],
    "movie_title": ["Blade Runner", "Inception"],
})

# From a CSV (entity/label columns, or legacy label/value columns)
clf = AhocorasickMediaClassifier.from_csv("my_entities.csv")

# From a pre-built AhocorasickNER object
from ahocorasick_ner import AhocorasickNER
ner = AhocorasickNER()
ner.add_word("artist_name", "Pink Floyd")
clf = AhocorasickMediaClassifier(ner)
```

### Runtime registration

OCP skills register their content at runtime via bus messages:

```python
# In an OCP skill, on startup:
self.bus.emit(Message("ovos.common_play.register_keyword", {
    "label": "artist_name",
    "match": "Pink Floyd",
    "media_type": MediaType.MUSIC,
}))
```

The pipeline plugin calls `clf.add_word(label, word)` on receipt, updating the
automaton in-place.  When the classifier is backed by an `EntitiesContainer`,
`add_word()` also updates the container's deduplication index.

### Config keys

| Key | Type | Description |
|---|---|---|
| `media_classifier_entities` | `dict` | Container config — media servers, HuggingFace, CSVs, wordlists (see above) |
| `media_classifier_wordlists` | `dict[str, list[str]]` | Static word lists keyed by entity label |
| `media_classifier_ner_csv` | `str` (path) | CSV file with entity label/word pairs |

`media_classifier_entities` takes priority over `media_classifier_wordlists` /
`media_classifier_ner_csv` in the factory selection order.

---

## 3. scikit-learn backend

**Class**: `ovos_media_classifier.sklearn.SklearnMediaClassifier`
**Deps**: `scikit-learn`, `joblib` (`pip install ovos-media-classifier[sklearn]`)
**Accuracy**: good
**Speed**: fast

### How it works

Two sklearn `Pipeline` objects are loaded from a single `joblib` file:

- **Domain pipeline**: `TfidfVectorizer → LogisticRegression`
  Input: utterance string → Output: `ocp_play` / `ocp_control` / `not_ocp`
- **Play pipeline**: `TfidfVectorizer → LogisticRegression`
  Input: utterance string → Output: fine-grained media type

Both pipelines use `ngram_range=(1, 2)` and `sublinear_tf=True`.

For classifiers that expose `predict_proba` (LogisticRegression), the
confidence is the probability of the winning class.  For classifiers that only
expose `decision_function` (LinearSVC), the confidence is derived from the raw
score via sigmoid.

### Loading

```python
from ovos_media_classifier.sklearn import SklearnMediaClassifier

clf = SklearnMediaClassifier.from_path(
    "/path/to/ocp_sklearn.joblib",
    play_threshold=0.3,
    domain_threshold=0.5,
)
```

### Training (from code)

```python
clf = SklearnMediaClassifier.from_training_data(
    X=sentences,
    y=play_labels,
    X_domain=sentences,
    y_domain=domain_labels,
)
clf.save("/path/to/ocp_sklearn.joblib")
```

See [TRAINING.md](TRAINING.md) for the full training pipeline.

### Config keys

| Key | Type | Description |
|---|---|---|
| `media_classifier_sklearn_model` | `str` (path) | Path to `.joblib` model file |
| `media_classifier_play_threshold` | `float` | Min confidence for play classifier (default `0.3`) |
| `media_classifier_domain_threshold` | `float` | Min confidence for domain classifier (default `0.5`) |

### Joblib file format

```python
{
    "play_pipeline":   sklearn.pipeline.Pipeline,  # required
    "domain_pipeline": sklearn.pipeline.Pipeline,  # optional
}
```

---

## 4. Padatious backend

**Class**: `ovos_media_classifier.padatious.PadatiousMediaClassifier`
**Deps**: `ovos-padatious` or `padacioso` (`pip install ovos-media-classifier[padatious]`)
**Accuracy**: medium–good
**Speed**: medium

### How it works

Trains an `IntentContainer` (padatious or padacioso) on `.intent` files — one
file per intent label, each containing padatious-style utterance patterns:

```
# music.intent
play {query} music
put on some {genre}
I want to listen to {artist}
stream {artist}
```

Two containers are trained:
- **Play container**: one intent per `OCPPlayIntent` value.
- **Domain container** (optional): three intents — `ocp_play`, `ocp_control`,
  `not_ocp` — for fast domain gating.

### Loading from bundled locale

```python
from ovos_media_classifier.padatious import PadatiousMediaClassifier
import os, ovos_media_classifier

LOCALE = os.path.join(os.path.dirname(ovos_media_classifier.__file__), "locale")

clf = PadatiousMediaClassifier.from_locale_dir(LOCALE, lang="en-us")
```

### Loading from samples dict

```python
clf = PadatiousMediaClassifier.from_samples(
    play_samples={
        "music":   ["play {query} music", "put on some {genre}"],
        "podcast": ["play the {name} podcast"],
        "movie":   ["watch {title}", "play the movie {name}"],
    },
    domain_samples={
        "ocp_play":    ["play {query}", "watch {thing}"],
        "ocp_control": ["pause", "resume", "next track"],
        "not_ocp":     ["set an alarm", "what is the weather"],
    },
)
```

### Padatious vs padacioso

`ovos-padatious` is a Cython implementation — fast but requires compilation.
`padacioso` is pure Python — slower but works everywhere without compilation.
The backend tries `ovos-padatious` first and falls back to `padacioso`.

### Config keys

| Key | Type | Description |
|---|---|---|
| `media_classifier_padatious_dir` | `str` (path) | Locale root directory for `.intent` files |
| `media_classifier_padatious_domain_dir` | `str` (path) | Separate locale dir for domain intents (optional) |
| `media_classifier_padatious_cache` | `str` (path) | Cache dir for compiled padatious models (optional) |
| `media_classifier_play_threshold` | `float` | Min confidence (default `0.5`) |
| `media_classifier_domain_threshold` | `float` | Min domain confidence (default `0.5`) |
| `lang` | `str` | BCP-47 tag for locale selection |

---

## 5. Model2Vec backend

**Class**: `ovos_media_classifier.m2v.Model2VecMediaClassifier`
**Deps**: `torch`, `model2vec` (`pip install ovos-media-classifier[m2v]`)
**Accuracy**: highest
**Speed**: fast at inference (static embeddings, no GPU needed)

### How it works

`StaticModelForHierarchicalClassification` (from `ovos_media_classifier.models`)
embeds each utterance using a `model2vec` static embedding model and passes
the result through two linear classification heads:

- **Domain head**: 3 outputs (`ocp_play`, `ocp_control`, `not_ocp`)
- **Intent head**: N outputs (one per `OCPPlayIntent`)

At inference:
1. Embed the utterance (very fast — model2vec uses pre-computed token embeddings).
2. Run the domain head (`classify_domain()`).
3. If domain is `ocp_play`, run the intent head (`classify()`).

### Loading

```python
from ovos_media_classifier.m2v import Model2VecMediaClassifier

clf = Model2VecMediaClassifier.from_path(
    "/path/to/m2v_model",
    domain_threshold=0.5,
    intent_threshold=0.3,
)
```

The model directory can be:
- A local path produced by `StaticModelForHierarchicalClassification.save()`
- A HuggingFace model name (passed to `StaticModel.from_pretrained()`)

### Why static embeddings?

Traditional transformer models (BERT, etc.) are too slow for real-time voice
assistant inference on edge hardware.  Static embedding models pre-compute word
embeddings once and combine them with pooling — inference is essentially just a
dot product.  `model2vec` produces static models distilled from larger
transformers, giving near-transformer quality at orders-of-magnitude less cost.

### Config keys

| Key | Type | Description |
|---|---|---|
| `media_classifier_model` | `str` (path or HF name) | Model directory or HuggingFace model name |
| `media_classifier_domain_threshold` | `float` | Min domain confidence (default `0.5`) |
| `media_classifier_intent_threshold` | `float` | Min intent confidence (default `0.3`) |

---

## 6. No-op fallback

When no config key matches and no `voc_match_func` is supplied,
`load_media_classifier()` returns `KeywordMediaClassifier()` using the bundled
locale files.  This is always functional — it will correctly identify many
common media queries in English and 12 other languages using vocabulary
keywords.

In the unlikely case you explicitly want a no-op classifier (e.g. for testing),
construct one directly:

```python
from ovos_media_classifier.keyword import KeywordMediaClassifier
clf = KeywordMediaClassifier(voc_match_func=lambda phrase, vocab, **kw: False)
```

---

## Choosing a backend

| Situation | Recommended backend |
|---|---|
| Embedded device, no ML deps | Keyword (bundled locale) |
| Fast inference, no training required | Keyword or AhocorasickNER |
| Have a Jellyfin / Radarr / Sonarr / Lidarr library | AhocorasickNER + EntitiesContainer |
| Have a music library (Lidarr / Music Assistant) | AhocorasickNER + EntitiesContainer |
| Adult content (Whisparr / Stash) | AhocorasickNER + EntitiesContainer |
| Want accuracy, have a trained model | sklearn |
| Best accuracy on edge hardware | Model2Vec |
| Multilingual with pattern-based approach | Padatious |
| Debugging / always-GENERIC sentinel | No-op fallback |
