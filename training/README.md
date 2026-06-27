# training/

Dataset-building and model-training tooling for `ovos-media-classifier`.

This directory is **not shipped in the wheel** — run it from a checkout with the
`[train]` extra installed:

```bash
pip install -e ".[train]"
```

It produces the canonical **`TigreGotico/ocp-media-intents`** dataset (the source
of truth classifier backends train and benchmark on) and trains/exports models.

## Pipeline — two steps

```bash
# 1. ingest real entity pools from the TigreGotico media-metadata collection
#    (local metadatarr cache when present, else HuggingFace) → data/entities/<label>.csv
python -m training.ingest_entities

# 2. build the dataset: expand templates → slot-fill → features → balance → split
python -m training.build_dataset                       # → data/release/
python -m training.build_dataset --push --repo TigreGotico/ocp-media-intents
```

The `.intent` / `.voc` templates are the hand-authored **source of truth** —
nothing regenerates them, so ovos-localize translations stick.

### Train a model bundle

```bash
# sklearn ladder (categorical features → tiny ONNX bundles, the reference path)
python -m training.train_sklearn                       # → data/models/<feature_set>/

# neural path (PyTorch → ONNX) with richer text features
python -m training.build_corpus                        # domain word vectors → data/wordvec/
python -m training.train_torch                         # → data/models_torch/<variant>/

# guided-categorical-embeddings routing-aware per-axis router (the embedding-router
# backend) — two-stream [categorical | entity] features, cost-matrix abstain,
# temperature-calibrated; supports runtime entity injection without retraining
python -m training.train_embedding_router              # → data/models/embedding_router/

# head-to-head benchmark: rules → sklearn → neural × feature set, on the test split
python -m benchmarks.ladder                            # → benchmarks/ladder_results.{md,json}
# harm-weighted OOD routing eval: keyword vs embedding-router vs hybrid(+inject)
python -m benchmarks.routing_eval                      # → benchmarks/routing_eval_results.{md,json}
```

`train_torch.py` trains a shared-trunk multi-task net on categorical ⊕ char-hash ⊕
trained-word-vector features and exports the **same** self-describing bundle the
runtime already loads (runtime stays `onnxruntime` + `numpy`; torch/gensim are
train-only). See [`docs/model.md` §7](../docs/model.md) for the comparison and
[`docs/extending.md`](../docs/extending.md) Part C for adding an architecture.
All model artifacts land under the gitignored `data/` — they stay **local**.

`build_dataset` is the **single entry point** and is fully reproducible for a
fixed `--seed`. See [`docs/dataset.md`](../docs/dataset.md) for every column, the
rebuild recipe, and how to add/translate templates; see
[`docs/data-sources.md`](../docs/data-sources.md) for every source → slot-label
mapping and licenses.

## Templates are translatable `.intent` / `.voc` locale resources

Templates are hand-authored OVOS-INTENT-1 files under the package `locale/`, so
ovos-localize picks them up and translates them like any other locale resource:

```
ovos_media_classifier/locale/
  <lang>/<lead_*>.voc            shared lead-in vocabularies (request openers)
  <lang>/dataset/<intent>.intent one file per media label
```

The `dataset/` subdir namespaces the dataset templates away from the runtime OCP
control intents (`play.intent`, `featured.intent`, …) that live at the locale
root; `ovos_spec_tools` resolves resources recursively under `<lang>/`, so the
`<lead_*>` references in `dataset/*.intent` expand from the lead-in `.voc` files
in the parent language directory.

Each `.intent` line is expanded by `ovos_spec_tools.expand` (`(a|b)`
alternations, `[optional]` openers/closers, `<voc>` references) into its sample
set; `build_dataset` then fills the `{slot}` placeholders with real entities.
These files are the **source of truth** — add phrasings, optional decorations
(`[hey|ok|please] … [please|for me]`), or a new language by editing/translating
them; no code regenerates or overwrites them.

## Entities

`ingest_entities.py` reads every source in the collection (see the source
registry `SOURCE_SPECS`) into `data/entities/<label>.csv`. Curated provider /
platform slots that have no metadata dump ship as committed seed lists in
`seed_entities/<label>.csv`.

## Notes

- **Adult metadata** is kept only as a **content-filter signal** (the `adult`
  genre, a deliberate minority) — see
  [content filtering](../docs/content-filtering.md). No adult provider/scraper
  work here.
- **agentpipe** (`generate_agentpipe.py`) adds an optional naturalistic LLM layer
  using **FREE agents only** (`opencode-free`/`kilo`) — never claude or paid.
- Homeserver loaders (`generate_dataset_from_media.py`) and the metadatarr
  refresher (`gather_metadatarr.py`) pull live entities from a user's own
  Radarr/Sonarr/Lidarr/Jellyfin/Music-Assistant stack.
- Model export → an ONNX bundle the optional [`onnx`](../docs/backends.md)
  backend loads (`train_sklearn.py` / `train_torch.py`), and the learned
  [embedding-router](../docs/embedding-router.md) bundle (`train_embedding_router.py`).
