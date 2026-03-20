# ovos-media-classifier

A pluggable, language-aware media-type classifier for the [OpenVoiceOS Common Play (OCP)](https://github.com/OpenVoiceOS/ovos-ocp-pipeline-plugin) pipeline.

Given a natural-language utterance like _"play something by Pink Floyd"_, it determines:

1. **Is this an OCP query at all?** (`ocp_play` / `ocp_control` / `not_ocp`)
2. **What kind of media does the user want?** (`music` / `movie` / `podcast` / …)

The classifier is a thin abstraction layer. Six backends are supported, from a
zero-dependency keyword matcher to a neural hierarchical model — all behind the
same three-method interface.

---

## Documentation index

| Document | Audience | What it covers |
|---|---|---|
| **[THEORY.md](THEORY.md)** | Developers / Researchers | How classification works, the two-level hierarchy, ML background, confidence scores |
| **[BACKENDS.md](BACKENDS.md)** | Developers | All backends in depth: `EntitiesContainer`, media-server loaders, config keys, limitations |
| **[TRAINING.md](TRAINING.md)** | ML Engineers | Gathering data, training sklearn / M2V / NER models, benchmarks |
| **[NER_LABELS.md](NER_LABELS.md)** | Skill authors / Pipeline devs | Full `OCPEntityLabel` taxonomy, `EntitiesContainer` population, runtime registration protocol |
| **[LANG_SUPPORT.md](LANG_SUPPORT.md)** | Translators / Contributors | Supported languages, how to add a new locale |
| **[MAINTAINERS_GUIDE.md](MAINTAINERS_GUIDE.md)** | Maintainers | Release workflow, repository layout, dependency policy, CI, testing |

---

## Quick start

```python
from ovos_media_classifier import load_media_classifier

# Zero-dependency mode — reads bundled vocabulary files
clf = load_media_classifier()

media_type, conf = clf.classify("play some jazz", "en-us")
# → (MediaType.MUSIC, 0.6)

is_ocp, conf = clf.is_ocp_query("pause the music", "en-us")
# → (True, ...)

domain, conf = clf.classify_domain("what is the weather", "en-us")
# → (OCPDomain.NOT_OCP, 0.0)
```

### Using a trained ML model

```python
# sklearn (fast, good accuracy)
clf = load_media_classifier(
    config={"media_classifier_sklearn_model": "/path/to/ocp_sklearn.joblib"}
)

# Neural Model2Vec (best accuracy)
clf = load_media_classifier(
    config={"media_classifier_model": "/path/to/m2v_model"}
)
```

### Personalised matching from your media library

`EntitiesContainer` pulls entity strings (movie titles, artist names, TV shows,
…) directly from your media servers and keeps the classifier in sync at runtime:

```python
from ovos_media_classifier import EntitiesContainer, load_media_classifier

container = EntitiesContainer()
container.load_radarr("http://localhost:7878",    api_key="…")
container.load_sonarr("http://localhost:8989",    api_key="…")
container.load_lidarr("http://localhost:8686",    api_key="…")
container.load_jellyfin("http://localhost:8096",  api_key="…")
container.load_music_assistant("http://localhost:8095")

from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
clf = AhocorasickMediaClassifier.from_container(container)

# New content added at runtime is immediately classifiable:
container.add("movie_title", "Dune: Part Two")
clf.classify("play dune part two", "en-us")  # → (MediaType.MOVIE, 0.6)
```

Or let the factory build and configure everything from a config dict:

```python
clf = load_media_classifier(config={
    "media_classifier_entities": {
        "radarr":          {"url": "http://localhost:7878", "api_key": "…"},
        "sonarr":          {"url": "http://localhost:8989", "api_key": "…"},
        "lidarr":          {"url": "http://localhost:8686", "api_key": "…"},
        "jellyfin":        {"url": "http://localhost:8096", "api_key": "…"},
        "music_assistant": {"url": "http://localhost:8095"},
        "whisparr":        {"url": "http://localhost:6969", "api_key": "…"},
        "stash":           {"url": "http://localhost:9999", "api_key": "…"},
        "huggingface": [{"dataset": "TigreGotico/ocp-entities"}],
    }
})
```

### Plugging into the OCP pipeline

```python
# Inside OCPPipelineMatcher (ovos-ocp-pipeline-plugin)
clf = load_media_classifier(
    config=self.config,
    voc_match_func=self.voc_match,   # delegates .voc lookups to the skill layer
)
```

---

## Installation

```bash
# Core (keyword backend only — no ML deps)
pip install ovos-media-classifier

# With AhocorasickNER backend
pip install ovos-media-classifier[ner]

# With media-server loaders (Jellyfin, Radarr, Sonarr, Lidarr, Whisparr, Stash, Music Assistant)
pip install ovos-media-classifier[ner,media_servers]

# With HuggingFace dataset loader
pip install ovos-media-classifier[ner,huggingface]

# With sklearn backend
pip install ovos-media-classifier[sklearn]

# With neural Model2Vec backend
pip install ovos-media-classifier[m2v]

# With padatious backend
pip install ovos-media-classifier[padatious]

# Everything
pip install ovos-media-classifier[all]

# Training tools
pip install ovos-media-classifier[train]
```

---

## Backend priority

When `load_media_classifier()` is called it selects the best available backend
based on which config keys are set:

```
1. Model2Vec          media_classifier_model            (neural, best accuracy)
2. scikit-learn       media_classifier_sklearn_model    (ML, fast inference)
3. Padatious          media_classifier_padatious_dir    (pattern + ML)
4a. AhocorasickNER    media_classifier_entities         (media servers + HuggingFace,
                                                         runtime-aware)
4b. AhocorasickNER    media_classifier_wordlists /      (static exact match)
                      media_classifier_ner_csv
5. Keyword            voc_match_func supplied           (pipeline .voc files)
6. Keyword            fallback                          (bundled locale files)
```

Each backend is optional — a missing dependency or missing file causes a
warning and falls through to the next backend.

---

## Supported languages

The bundled keyword locale covers 13 languages.  ML backends are language-agnostic (the model
handles normalisation).  See [LANG_SUPPORT.md](LANG_SUPPORT.md) for details.

`ca-es` `da-dk` `de-de` `en-us` `es-es` `eu` `fr-fr` `gl-es` `it-it` `nl-nl` `pl-pl` `pt-br` `pt-pt`
