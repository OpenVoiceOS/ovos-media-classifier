# ovos-media-classifier — agent guide

Pluggable, language-aware media-intent classifier for the OCP (Open Conversation
Platform) pipeline. Given an utterance it answers "is this a media request at
all?" (`ocp_play` / `ocp_control` / `not_ocp`) and, if so, classifies it along
several orthogonal axes at once (`media_type`, `playback_type`, `structure`,
`explicitness`, `tags`, `qualifiers`) behind a single `AbstractMediaClassifier`
interface (`classify`, `classify_domain`, `is_ocp_query`), with several
interchangeable backends behind it.

## Setup

```bash
pip install ovos-media-classifier                    # core: bundled .voc keyword classifier, zero ML deps
pip install ovos-media-classifier[onnx]               # ONNX trained backend + embedding router (onnxruntime, numpy)
pip install ovos-media-classifier[ner]                 # Aho-Corasick entity-matching backend (ahocorasick-ner)
pip install ovos-media-classifier[online]              # metadatarr-backed online resolution layer
pip install ovos-media-classifier[media_servers]       # Radarr/Sonarr/Jellyfin/... entity loaders (requests)
pip install ovos-media-classifier[huggingface]         # HF dataset-backed entity loaders (datasets)
pip install ovos-media-classifier[train]               # offline training toolchain (torch, gensim, scikit-learn, skl2onnx, pandas)
```

The core install has one runtime dependency chain: `ovos-utils`, `mediavocab`
(the canonical `MediaType`/`PlaybackType`/`Signals` vocabulary), and
`ovos-spec-tools` (the OVOS-INTENT-2 `.voc` loader + word-boundary matcher used
by the bundled locale files). `mediavocab` is floor-pinned to the prerelease
that first ships the `Structure`/`PictureFormat`/`infer_*` symbols this package
imports, and ceilinged below its next major to avoid an unreviewed breaking
bump landing silently.

No lockfile is committed. Resolve with `uv --prerelease=allow` (or plain `pip`)
against the floor pins in `pyproject.toml`; do not hand-write or commit a
`uv.lock`.

## Test

```bash
pip install -e .[test] && pytest test/
```

This is the *minimal* mode: it exercises every backend's structure and fallback
logic with hand-built fakes (fake ONNX sessions emitting canned logits, a
mocked `metadatarr.resolve`) and plain `numpy` arrays — no `onnxruntime`, no
trained model files, no network. It passes in full except for a handful of
explicit `skipif`-gated tests that need a real trained bundle
(`data/models/context_ner`) or the `[train]` toolchain
(`scikit-learn`/`skl2onnx`/`onnxruntime`/`torch`) — those skips are the suite
working as designed, not missing coverage.

```bash
pip install -e .[test,onnx,ner,media_servers,online,huggingface,train] && pytest test/
```

The full mode additionally installs every optional backend and the training
toolchain, which lifts the `skipif` guards above and exercises the real ONNX
export/import round trip (`test_text_pipeline_bundle.py`, `test_torch_bundle.py`)
and the trained-bundle-backed paths in `test_public_api.py`.

Run `pytest test/ -q` after any change; a regression in either mode is a bug.

## Layout

- `ovos_media_classifier/base.py` — `AbstractMediaClassifier`, the interface
  every backend implements.
- `ovos_media_classifier/intents.py` — `MediaType`, `OCPDomain`, and the
  raw-label → `(MediaType, genres)` maps. `mediavocab` owns the canonical
  vocabulary; this module maps dataset/NER labels onto it.
- Backends, in the order `load_media_classifier` tries them:
  - `keyword.py` — the default. Coarse-to-fine `.voc` keyword matching
    (domain → modality → structure → constrained leaf). Zero ML deps; reads
    the bundled `ovos_media_classifier/locale/<lang>/` files directly, or
    binds to a pipeline's own `voc_match` function. Registered as the
    reference `opm.media.classifier` plugin (see Entry points below).
  - `onnx.py` — opt-in trained backend. Loads a self-describing multi-head
    ONNX bundle (one head per axis); depends on raw `onnxruntime` + `numpy`
    only, imported lazily. Requires the `onnx` extra and
    `media_classifier_onnx_model` in config.
  - `embedding.py` — the learned embedding router (guided-categorical
    embeddings trained offline, plain ONNX + numpy at inference). Wired as a
    hybrid by default: keyword stays the high-precision first pass, the
    router handles the keyword-less cases and abstains to `GENERIC` rather
    than guess. Requires the `onnx` extra and
    `media_classifier_embedding_router` in config.
  - `ahocorasick.py` — Aho-Corasick substring matching over
    runtime-registered entity dictionaries (service/artist/genre names).
    Requires the `ner` extra and one of `media_classifier_entities` /
    `_wordlists` / `_ner_csv` in config.
  - `metadatarr_backend.py` — Layer B, online last resort. Bare real titles
    that the offline layers abstain on get resolved via `metadatarr` (network,
    opt-in via the `online` extra + a config flag, default OFF). Consulted
    only after the cheaper layers give up.
  - `gazetteer.py` — Layer A, the offline popularity gazetteer built from
    entity-pool CSVs; injected into the embedding router to resolve bare
    titles without a network round trip.
- `features.py` / `features_text.py` / `features_wordvec.py` — feature
  extraction shared by the trained backends.
- `entities.py` — `EntitiesContainer` with runtime loaders
  (`load_radarr/sonarr/lidarr/jellyfin/whisparr/stash/music_assistant/huggingface/csv`)
  that feed the Aho-Corasick and gazetteer backends real entity metadata.
- `__init__.py` — `load_media_classifier()`, the factory. Backend selection
  order: external plugin (`media_classifier_plugin`) → ONNX
  (`media_classifier_onnx_model`) → embedding router
  (`media_classifier_embedding_router`) → NER → bundled keyword. Each backend
  falls through to the next on any load failure, ending at the keyword
  matcher, so a missing optional dependency degrades gracefully instead of
  crashing — check logs if a classification looks wrong, a backend may have
  silently failed to load.
- `ovos_media_classifier/locale/` — bundled `.voc` keyword vocab and
  translatable `.intent` dataset templates, per locale.
- `ovos_media_classifier/data/gazetteer.json` — the small bundled default
  gazetteer shipped in the wheel (the larger generated `data/` at repo root
  is gitignored scratch).
- `training/` — dataset build and training CLIs
  (`build_dataset.py`, `build_corpus.py`, `train_sklearn.py`, `train_torch.py`).
  Not shipped in the wheel; run from a checkout. See `training/README.md`.
- `docs/` — theory, per-backend docs, taxonomy spec, routing eval, and the
  stable API reference.

### Entry points

`[project.entry-points."opm.media.classifier"]` registers
`ovos-media-classifier-keyword = ovos_media_classifier.keyword:KeywordMediaClassifier`.
This package is itself a discoverable OPM (Open Plugin Manager) media
classifier plugin — the keyword backend is the reference implementation for
that entry-point group, not merely a library imported by the OCP pipeline
plugin.

## Conventions (org hard rules)

- Branches: work on `dev`, stable on `master`. NEVER `main`.
- Never hand-edit `ovos_media_classifier/version.py`; gh-automations bumps
  semver from conventional-commit prefixes (`feat:` / `fix:` / `feat!:`).
- No lockfiles committed (`uv.lock` is gitignored); resolve against floor pins
  with `uv --prerelease=allow`.
- Dependency versions are declared only in `pyproject.toml`, never duplicated
  in CI config.
- Commit identity: `JarbasAi <jarbasai@mailfence.com>`.
- CI is provided by OpenVoiceOS/gh-automations reusable workflows referenced
  at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary: describe current state only — no history, dates, or
  "design mistake" framing in code, docs, commits, or PRs.

## Gotchas

- Every non-core backend degrades gracefully: a missing dependency or model
  file logs a warning and falls through to the next backend, ending at the
  bundled keyword matcher. A "wrong" classification can therefore mean a
  backend silently failed to load — check logs.
- This repo owns the canonical `MediaType` taxonomy via `mediavocab`; keep any
  downstream int-compatible copies aligned when adding types.
- The `test` extra installs `metadatarr` and `numpy` even though the minimal
  test mode never hits the network or imports real ONNX: `test_embedding.py`
  and `test_gazetteer.py` build ONNX-shaped heads by hand with plain numpy
  arrays, and `test_metadatarr_backend.py` patches `metadatarr.resolve`, which
  needs the module importable for the patch target to exist.
