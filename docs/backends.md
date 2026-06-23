# Backends

All backends implement [`AbstractMediaClassifier`](stable-api.md) and are selected
by `load_media_classifier(config)`. Selection is **first matching config key
wins**, in the order below; any import or load error logs a warning and falls
through to the next backend. With no config keys set, the bundled keyword backend
is used.

| Order | Backend | Config key | Extra |
|---|---|---|---|
| 0 | external plugin | `media_classifier_plugin` | (3rd-party) |
| 1 | Model2Vec | `media_classifier_model` | `[m2v]` |
| 2 | GuidedEmbeddings (ONNX) | `media_classifier_guided_model` | `[guided]` |
| 3 | scikit-learn | `media_classifier_sklearn_model` | `[sklearn]` |
| 4 | Padatious | `media_classifier_padatious_dir` | `[padatious]` |
| 5a | AhocorasickNER (live) | `media_classifier_entities` | `[ner]` (+ loaders) |
| 5b | AhocorasickNER (static) | `media_classifier_wordlists` / `media_classifier_ner_csv` | `[ner]` |
| 6 | Keyword (pipeline) | `voc_match_func=` argument | — |
| 7 | Keyword (default) | — | — |

Common thresholds accepted by the trained backends:
`media_classifier_domain_threshold` (default 0.5),
`media_classifier_play_threshold` / `media_classifier_intent_threshold` (default
0.3, padatious 0.5). Below threshold the backend returns `(MediaType.GENERIC, 0.0)`
/ `NOT_OCP`.

## Keyword (default, zero-dep)

```python
clf = load_media_classifier()                      # bundled locale
clf = load_media_classifier(voc_match_func=self.voc_match)   # pipeline mode
```

Substring-matches the query against bundled per-language `.voc` files
(`ovos_media_classifier/locale/<lang>/<Vocab>.voc`). No ML dependencies, no model
files, fully offline. In *pipeline mode* the OCP pipeline plugin owns the `.voc`
files and passes its `voc_match` method as `voc_match_func`; in *standalone mode*
the classifier reads the bundled files directly. Matching runs in a fixed priority
order (e.g. `MusicVideoKeyword` before `MusicKeyword`) and surfaces genre tags
(`anime`, `adult`, …) via `classify_genres()`.

**Use when:** you want a dependency-free default, or you are inside the OCP
pipeline and already have a `voc_match` function. Languages bundled: `ca-es`,
`da-dk`, `de-de`, `en-us`, `es-es`, `eu-es`, `fr-fr`, `gl-es`, `it-it`, `nl-nl`,
`pl-pl`, `pt-br`, `pt-pt`.

## GuidedEmbeddings (ONNX) — recommended trained backend

```python
clf = load_media_classifier(config={
    "media_classifier_guided_model": "/path/to/model_dir",
})
```

Uses `guided-categorical-embeddings` ONNX exports over sparse categorical
features. The model directory contains two heads, `domain/` and `play/`, mirroring
the M2V architecture. Inference: extract a categorical feature dict from the query
(keyword + NER features), run the domain head, and — when the domain is `ocp_play`
— run the play head and map its label to a `MediaType`.

It is **torch-free at runtime** (only `onnxruntime`), which is why it is the
recommended trained backend, and it can load **any** exported model that conforms
to the feature contract — including models you train and export yourself.
Optionally pass `media_classifier_entities` alongside the model so NER features are
drawn from a live `EntitiesContainer`.

**Use when:** you want trained accuracy without a heavyweight runtime, or you have
your own exported guided-embeddings model.

## scikit-learn

```python
clf = load_media_classifier(config={
    "media_classifier_sklearn_model": "/path/to/model.joblib",
})
```

A TF-IDF + LogisticRegression pipeline serialized with joblib. Fast inference,
small footprint, good baseline accuracy.

**Use when:** you want a quick trained backend and already have a `.joblib` model.

## Padatious

```python
clf = load_media_classifier(config={
    "media_classifier_padatious_dir": "/path/to/locale",
    # optional: media_classifier_padatious_domain_dir, media_classifier_padatious_cache
})
```

Pattern + ML intent matching over `.intent` sample files. Unlike the keyword and
NER backends it has a real `ocp_control` head, so it can recognise control
commands ("pause", "next track") in addition to play requests.

**Use when:** you need native control-intent detection or already maintain
padatious intent files.

## AhocorasickNER

Aho-Corasick substring NER over a set of named entities. Each matched entity label
(`artist_name`, `movie_title`, …) maps to a media type via
`NER_LABEL_TO_PLAY_INTENT`. The key property is that the entity set is **live**:
content registered at runtime becomes classifiable immediately.

### Live entities from media servers / Hugging Face

```python
clf = load_media_classifier(config={
    "media_classifier_entities": {
        "radarr":          {"url": "http://localhost:7878", "api_key": "…"},
        "sonarr":          {"url": "http://localhost:8989", "api_key": "…"},
        "lidarr":          {"url": "http://localhost:8686", "api_key": "…"},
        "jellyfin":        {"url": "http://localhost:8096", "api_key": "…"},
        "whisparr":        {"url": "http://localhost:6969", "api_key": "…"},
        "stash":           {"url": "http://localhost:9999", "api_key": "…"},
        "music_assistant": {"url": "http://localhost:8095"},
        "huggingface":     [{"dataset": "TigreGotico/ocp-entities"}],
        "csv":             ["/path/to/extra_entities.csv"],
        "wordlists":       {"artist_name": ["Radiohead"]},
    },
})
```

Server loaders need the `[media_servers]` extra (`requests`); the Hugging Face
loader needs `[huggingface]` (`datasets`). You can also build the container
directly and add entries at runtime:

```python
from ovos_media_classifier import EntitiesContainer
from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

container = EntitiesContainer()
container.load_jellyfin("http://localhost:8096", api_key="…")
clf = AhocorasickMediaClassifier.from_container(container)

container.add("movie_title", "Dune: Part Two")     # immediately classifiable
```

**Use when:** you want personalised matching against the user's actual library,
with content that changes at runtime as skills register entities.

### Static word list / CSV

```python
clf = load_media_classifier(config={
    "media_classifier_wordlists": {"music": ["jazz", "blues"]},
})
clf = load_media_classifier(config={
    "media_classifier_ner_csv": "/path/to/entities.csv",
})
```

Exact-match against a fixed word list or CSV — no servers, no network.

**Use when:** you have a known, fixed entity set and want deterministic matching.

## Model2Vec

```python
clf = load_media_classifier(config={
    "media_classifier_model": "/path/to/m2v_model",
})
```

A two-headed (domain + play-intent) hierarchical neural model built on
`model2vec` static embeddings. Highest-accuracy backend; takes top selection
priority when its config key is present.

**Use when:** accuracy matters most and the model/runtime size is acceptable.
