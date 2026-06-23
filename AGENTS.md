# ovos-media-classifier — agent guide

Pluggable, language-aware media-type classifier for the OCP (Open Conversation Platform) pipeline. Given an utterance it answers "is this an OCP query?" (`ocp_play` / `ocp_control` / `not_ocp`) and "what media type?" (`music` / `movie` / `podcast` / …) behind a single three-method interface (`classify`, `classify_domain`, `is_ocp_query`) with several interchangeable backends.

## Setup

```bash
pip install -e .                 # core, keyword backend only (dep: ovos-utils)
pip install -e .[all]            # every backend + media-server loaders + HF
pip install -e .[train]          # training tools (torch, pandas, pyarrow)
```

Optional extras, each gating a backend or loader: `guided` (ONNX categorical embeddings via `guided-categorical-embeddings`), `sklearn`, `ner` (`ahocorasick-ner`), `padatious`, `m2v` (`model2vec`), `media_servers` (`requests`), `huggingface` (`datasets`).

Note: `[tool.uv.sources]` in `pyproject.toml` pins `guided-categorical-embeddings` to a local editable path (`../../Machine Learning Workspace/guided-categorical-embeddings`). That path is machine-specific; `pip` ignores it but `uv` resolution depends on it.

## Test

```bash
pytest test/
```

139 unit tests across `test/test_classifier.py` and `test/test_guided_classifier.py`. All ML/ONNX/media-server dependencies are mocked, so the suite runs with only the core install.

## Lint/Typecheck

No lint or typecheck config in `pyproject.toml`. A `.ruff_cache/` is present (ruff was run ad hoc) but no ruff settings are committed.

## Layout

- `ovos_media_classifier/base.py` — `AbstractMediaClassifier` (the interface).
- `ovos_media_classifier/intents.py` — authoritative `MediaType`, `OCPDomain`, `OCPPlayIntent`, `OCPControlIntent`, `OCPEntityLabel` taxonomy and `PLAY_INTENT_TO_MEDIA_TYPE` map. The canonical taxonomy lives here, not in ovos-utils.
- Backends: `keyword.py`, `ahocorasick.py` (NER), `sklearn.py`, `padatious.py`, `m2v.py` (neural Model2Vec), `guided.py` (ONNX categorical embeddings), `features.py` (categorical feature extraction).
- `entities.py` — `EntitiesContainer` with runtime loaders: `load_radarr/sonarr/lidarr/jellyfin/whisparr/stash/music_assistant/huggingface/csv`.
- `__init__.py` — `load_media_classifier()` factory; selects backend by config key priority (m2v → guided → sklearn → padatious → ahocorasick → keyword/voc → bundled locale).
- `ovos_media_classifier/locale/` — bundled keyword vocab, 13 locales.
- `ovos_media_classifier/train/` — dataset build / training CLIs (entry points below).
- `templates/` — per-language, per-media-type CSV templates for synthetic dataset generation.
- `docs/` — THEORY, BACKENDS, TRAINING, NER_LABELS, LANG_SUPPORT, MAINTAINERS_GUIDE, taxonomy spec.

Entry-point group: **`console_scripts` only** (not an OPM/OVOS plugin). Scripts: `ovos-ocp-build-dataset`, `ovos-ocp-train-guided`, `ovos-ocp-explore`, `ovos-ocp-gen-features`. This is a library imported by `ovos-ocp-pipeline-plugin`, not a discoverable plugin.

## Conventions (org hard rules)

- Branches: work on `dev`, stable on `master`. NEVER `main`.
- Never edit `ovos_media_classifier/version.py`. gh-automations bumps semver from conventional-commit prefixes (`feat:` / `fix:` / `feat!:`).
- New repos private by default; do not make public without asking.
- Commit identity: JarbasAi <jarbasai@mailfence.com>.
- CI is provided by OpenVoiceOS/gh-automations reusable workflows referenced at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary: describe current state only — no history, dates, or "design mistake" framing in code, docs, commits, or PRs.

## Gotchas

- No git remote yet (local-only). Not pushed to GitHub; no CI runs.
- `pyproject.toml` sets `readme = "README.md"` but there is no `README.md` at the repo root (the rich README lives at `docs/README.md`). Builds will warn or fail to attach a long description.
- Committed scratch artifacts are tracked in git: `.coverage`, `ovos_media_classifier.egg-info/`, and `__pycache__/*.pyc`. There is no `.gitignore`.
- Every non-core backend degrades gracefully: a missing dependency or model file logs a warning and falls through to the next backend, ending at the bundled keyword matcher. A "wrong" classification can therefore mean a backend silently failed to load — check logs.
- This repo owns the canonical `MediaType` taxonomy; ovos-utils keeps an int-compatible copy. Keep integer values aligned when adding types.
