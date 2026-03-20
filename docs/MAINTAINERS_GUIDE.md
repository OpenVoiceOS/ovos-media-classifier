# Maintainers Guide

This document is for maintainers and contributors of `ovos-media-classifier`.
It covers the project architecture, how to add a new backend, the release
workflow, CI setup, and testing conventions.

---

## Repository layout

```
ovos-media-classifier/
  ovos_media_classifier/
    __init__.py            # load_media_classifier() factory + public exports
    base.py                # AbstractMediaClassifier ABC
    intents.py             # OCPDomain, OCPPlayIntent, OCPControlIntent,
                           # OCPEntityLabel, all mapping dicts
    entities.py            # EntitiesContainer — unified entity registry
                           #   loaders: Radarr, Sonarr, Lidarr, Jellyfin,
                           #            Whisparr, Stash, Music Assistant,
                           #            HuggingFace datasets, CSV
    keyword.py             # KeywordMediaClassifier + _VocMatcher
    ahocorasick.py         # AhocorasickMediaClassifier (accepts EntitiesContainer)
    sklearn.py             # SklearnMediaClassifier
    padatious.py           # PadatiousMediaClassifier
    m2v.py                 # Model2VecMediaClassifier
    models.py              # StaticModelForHierarchicalClassification
    version.py             # VERSION_MAJOR/MINOR/BUILD/ALPHA + __version__
    locale/                # bundled .voc and .intent files (13 languages)
      en-us/
        MusicKeyword.voc
        play.intent
        ...
      de-de/
        ...
    train/
      __init__.py
      gather_dataset.py    # download + normalise multilingual dataset
      ner_datasets.py      # MusicNER, ImdbNER, OCPMediaNER
      train_ocp_sklearn.py # train sklearn domain + play classifiers
  docs/
    README.md
    THEORY.md
    BACKENDS.md
    TRAINING.md
    NER_LABELS.md
    LANG_SUPPORT.md
    MAINTAINERS_GUIDE.md   # this file
  generate_dataset_from_media.py  # CLI: export entities from media servers to CSV
  requirements/
    requirements.txt       # hard deps: ovos-utils (for LOG, bus utilities)
                           # NOTE: MediaType is defined locally in intents.py,
                           # not imported from ovos-utils (see Dependency Policy)
  setup.py
  test/
    test_classifier.py
```

---

## Design principles

1. **All backends share the same three-method interface.**
   New backends must implement `AbstractMediaClassifier`.  No pipeline-specific
   logic belongs in the classifier.

2. **All label strings live in `intents.py`.**
   If you need a new intent label or entity label, add it there first — never
   as a bare string in a backend file.

2a. **`MediaType` is defined locally in `intents.py`.**
    `ovos-utils` keeps its own `MediaType` for backward compatibility with the
    rest of OVOS.  Within `ovos-media-classifier` all modules must import
    `MediaType` from `ovos_media_classifier.intents`, **not** from
    `ovos_utils.ocp`.  Integer values for shared types are identical, so the
    two enums interoperate via int comparison.  Do **not** add
    `from ovos_utils.ocp import MediaType` anywhere in this package.

3. **Every ML dependency is optional.**
   Backend files import ML libraries at the top level but the factory catches
   `ImportError` and falls through to the next backend.  Test that each backend
   can be imported without its optional dep installed.

4. **The locale directory is bundled and self-contained.**
   Keyword and padatious backends must work without requiring the OCP pipeline
   plugin to be installed.

5. **Confidence scores are heuristics, not calibrated probabilities.**
   Do not try to make them directly comparable across backends — they are
   signals for the OCP pipeline to use, not statistical probabilities.

---

## Adding a new backend

### 1. Create the backend module

```
ovos_media_classifier/my_new_backend.py
```

Implement `AbstractMediaClassifier`:

```python
from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import OCPDomain, LABEL_TO_MEDIA_TYPE

class MyNewMediaClassifier(AbstractMediaClassifier):

    def classify(self, query, lang, valid_labels=None):
        # ... your logic ...
        return MediaType.MUSIC, 0.7

    def classify_domain(self, query, lang):
        # Optional — override if you have a cheap domain head.
        # Default derives from classify().
        return super().classify_domain(query, lang)
```

### 2. Add optional dependencies to `setup.py`

```python
extras_require={
    "my_backend": ["my-library>=1.0.0"],
    "all": [
        ...,
        "my-library>=1.0.0",
    ],
}
```

### 3. Integrate into the factory

In `__init__.py`, add a new numbered step **before the keyword fallback**:

```python
# 5. My new backend
my_model_path = config.get("media_classifier_my_model")
if my_model_path:
    try:
        from ovos_media_classifier.my_new_backend import MyNewMediaClassifier
        clf = MyNewMediaClassifier.from_path(my_model_path)
        LOG.info(f"OCP media classifier: my_new_backend ({my_model_path})")
        return clf
    except Exception as e:
        LOG.warning(f"Failed to load my_new_backend: {e}. Trying next backend.")
```

Update the docstring at the top of `__init__.py` to list the new config key.

### 4. Export the class

Add it to `__all__` in `__init__.py` and import it at the top of the file.

### 5. Add tests

In `test/test_classifier.py` (or a new test file), add:

- A unit test that creates the classifier with a mock model.
- A test that verifies it falls back gracefully when the dep is missing
  (use `unittest.mock.patch` to simulate `ImportError`).
- A test of `classify()`, `classify_domain()`, and `is_ocp_query()`.

### 6. Document the backend

Add a section to [BACKENDS.md](BACKENDS.md) following the existing format.

---

## Adding a new media type

If OCP gains a new `MediaType` value:

1. Add a constant to `OCPPlayIntent` in `intents.py`.
2. Add the mapping to `PLAY_INTENT_TO_MEDIA_TYPE`.
3. `MEDIA_TYPE_TO_PLAY_INTENT` and `LABEL_TO_MEDIA_TYPE` are auto-derived —
   no changes needed there.
4. If the type has NER entities, add constants to `OCPEntityLabel` and entries
   to `NER_LABEL_TO_PLAY_INTENT`.
5. Update `keyword.py` to check the appropriate vocab file in `classify()`.
6. Add the vocab files to all 13 locale directories.
7. Update `BACKENDS.md` and `NER_LABELS.md`.

---

## Adding a new control intent

1. Add a constant to `OCPControlIntent` in `intents.py`.
2. Add a `<name>.intent` file to every locale directory.
3. Update `gather_dataset.py` `_CONTROL_INTENT_PATTERNS` to map the new intent
   name.

---

## Testing

### Running tests

```bash
pip install -e ".[all]"
pytest test/ -v
```

### Test conventions

- Tests live in `test/unittests/` (or `test/` for now).
- No real model files in tests — use mocks.
- Each backend test should:
  - Verify construction from a minimal mock.
  - Verify `classify()` returns the right `MediaType`.
  - Verify `classify_domain()` returns the right `OCPDomain`.
  - Verify fallback behaviour (GENERIC when nothing matches).
- Use `pytest.mark.skipif` to skip tests when optional deps are not installed.

### Testing without ML deps

```bash
pip install ovos-media-classifier   # no extras
pytest test/test_classifier.py -k "keyword or factory"
```

---

## Release workflow

This repository follows the standard OVOS release flow via `gh-automations`.

### Branches

- `dev` — active development.  All PRs target `dev`.
- `master` — stable releases.  Only `release-X.Y.Z` PRs merge here.

### Cutting a new release

1. Open a PR from `dev` into `dev` (or merge a feature PR).
2. The `release_workflow.yml` CI job fires on PR close:
   - Bumps the version in `version.py`.
   - Publishes an alpha to PyPI.
   - Opens a `release-X.Y.ZaN` PR targeting `master`.
3. Review and merge the release PR into `master`.
4. The `publish_stable.yml` CI job fires:
   - Removes the alpha flag from the version.
   - Tags the release on GitHub.
   - Publishes the stable release to PyPI.
   - Syncs `master` → `dev`.

### Version file format

```python
# ovos_media_classifier/version.py
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_BUILD = 0
VERSION_ALPHA = 0   # 0 = stable, N = alphaN

__version__ = (
    f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
    + (f"a{VERSION_ALPHA}" if VERSION_ALPHA else "")
)
```

---

## CI workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `unit_tests.yml` | Push/PR to `dev` | Run pytest on Python 3.10 and 3.11 |
| `build_tests.yml` | Push to `master` | Build sdist + wheel |
| `license_tests.yml` | Push/PR | Check Apache-2.0 compliance |
| `release_workflow.yml` | PR closed on `dev` | Bump version, publish alpha, open release PR |
| `publish_stable.yml` | Push to `master` | Publish stable to PyPI |
| `conventional-label.yaml` | PR opened | Add conventional-commit labels |

---

## Package data

The `locale/` directory is shipped with the package.  `setup.py` specifies:

```python
package_data={
    "ovos_media_classifier": ["locale/**/*.voc", "locale/**/*.intent"],
},
include_package_data=True,
```

When adding a new locale language, verify it will be included by running:

```bash
python setup.py egg_info
grep "locale" ovos_media_classifier.egg-info/SOURCES.txt | head -20
```

---

## Dependency policy

| Dependency | Where | Required? |
|---|---|---|
| `ovos-utils` | `requirements/requirements.txt` | Yes (provides `MediaType`, `LOG`) |
| `scikit-learn`, `joblib` | extras `[sklearn]` | No |
| `torch`, `model2vec` | extras `[m2v]` | No |
| `ovos-padatious` | extras `[padatious]` | No |
| `ahocorasick-ner` | extras `[ner]` | No — needed to materialise the NER from an `EntitiesContainer` |
| `requests` | extras `[media_servers]` | No — needed for media-server loaders in `EntitiesContainer` |
| `datasets` | extras `[huggingface]` | No — needed for `EntitiesContainer.load_huggingface()` |
| `pandas`, `matplotlib`, `seaborn` | extras `[train]` | No (training only) |

Never add an ML library or `requests` to `requirements.txt`.  The keyword
backend must always work with zero extra dependencies.

### `entities.py` import safety

`entities.py` imports only from the Python standard library and `ovos-utils` at
module level.  All optional imports (`ahocorasick_ner`, `requests`, `datasets`)
happen inside methods and raise `ImportError` with install instructions when the
dep is missing.  This ensures `from ovos_media_classifier import EntitiesContainer`
never fails regardless of which extras are installed.

---

## Common gotchas

### `OCPEntityLabel` string values vs. enum members

`OCPEntityLabel` extends `str`, so `OCPEntityLabel.ARTIST_NAME == "artist_name"`
is `True`.  But the `NER_LABEL_TO_PLAY_INTENT` dict is keyed by the **enum
member**, not the string.  When looking up from raw NER output (a string), use:

```python
intent = NER_LABEL_TO_PLAY_INTENT.get(OCPEntityLabel(raw_label))
# or
intent = NER_LABEL_TO_PLAY_INTENT.get(raw_label)   # works because str subclass
```

### `classify_domain()` default behaviour

The base class `classify_domain()` calls `classify()` and returns `OCP_PLAY` if
the result is not `GENERIC`.  This means a backend that always returns `GENERIC`
from `classify()` will always return `NOT_OCP` from `classify_domain()`.

If your backend can detect control intents, **you must override
`classify_domain()`** — the default implementation cannot return `OCP_CONTROL`.

### Padatious `.intent` file naming

The file name (without `.intent`) becomes the intent label string passed to
`LABEL_TO_MEDIA_TYPE`.  File names must match `OCPPlayIntent` or `OCPDomain`
string values exactly:

- `music.intent` → label `"music"` → `OCPPlayIntent.MUSIC`
- `ocp_play.intent` → label `"ocp_play"` → `OCPDomain.OCP_PLAY`

Misnamed files will result in an unrecognised label and a `GENERIC` fallback.
