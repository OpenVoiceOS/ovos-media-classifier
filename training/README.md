# training/

Dataset-building and model-training tooling for `ovos-media-classifier`.

This directory is **not shipped in the wheel** — run it from a checkout with the
`[train]` extra installed:

```bash
pip install -e ".[train]"
```

It produces the canonical **`TigreGotico/ocp-media-intents`** dataset (the source
of truth classifier backends train and benchmark on) and trains/exports models.

## Pipeline — three steps

```bash
# 1. ingest real entity pools from the TigreGotico media-metadata collection
#    (local metadatarr cache when present, else HuggingFace) → data/entities/<label>.csv
python -m training.ingest_entities

# 2. (re)author the .intent / .voc templates from the bundled source
#    (only needed when you edit author_templates.py; the files are committed)
python -m training.author_templates

# 3. build the dataset: expand templates → slot-fill → features → balance → split
python -m training.build_dataset                       # → data/release/
python -m training.build_dataset --push --repo TigreGotico/ocp-media-intents
```

`build_dataset` is the **single entry point** and is fully reproducible for a
fixed `--seed`. See [`docs/dataset.md`](../docs/dataset.md) for every column, the
rebuild recipe, and how to add/translate templates; see
[`docs/data-sources.md`](../docs/data-sources.md) for every source → slot-label
mapping and licenses.

## Templates are `.intent` / `.voc` files

Templates live under `templates/` as translatable OVOS-INTENT-1 files, managed
through ovos-localize:

```
templates/
  vocab/<lang>/<lead_*>.voc     shared lead-in vocabularies (request openers)
  <lang>/<intent>.intent        one file per media label
```

Each `.intent` line is expanded by `ovos_spec_tools.expand` (`(a|b)`
alternations, `[optional]`, `<voc>` references) into its sample set;
`build_dataset` then fills the `{slot}` placeholders with real entities. Add
phrasings or a new language by editing these files — no code change needed.
`author_templates.py` regenerates the bundled English set from its source lists
(natural phrasings + entity-role variants + cross-type **confusables**).

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
  backend loads (`train_guided_embeddings.py`).
