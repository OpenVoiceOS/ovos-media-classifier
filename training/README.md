# training/

Dataset-building and model-training tooling for `ovos-media-classifier`.

This directory is **not shipped in the wheel** — it is run from a checkout with the
`[train]` extra installed:

```bash
pip install -e ".[train]"
```

It produces the canonical **`TigreGotico/ocp-media-intents`** dataset (the source of
truth that classifier backends train and benchmark on) and trains/exports models.

## The dataset

Each row is a natural-language media command labelled across the
[multi-axis model](../docs/classification-model.md):

| column | meaning |
|---|---|
| `lang` | BCP-47 language code |
| `domain` | `ocp_play` / `ocp_control` / `not_ocp` |
| `intent` / `media_label` | fine-grained `OCPPlayIntent` |
| `binary_label` | `ocp` / `not_ocp` |
| `playback_label` / `playback_type` | modality (audio/video/…) |
| `structure` | single / episodic / continuous / collection |
| `mediavocab_type` | canonical `mediavocab.MediaType` (enforced taxonomy) |
| `genres` | `;`-joined mediavocab genre tags (carries the `adult` content-filter signal) |
| `sentence` | the utterance |

## Pipeline (run order)

```bash
# 1. gather real entity pools (curated HF sets + public catalogue APIs; NO wikidata — noisy)
python -m training.gather_entities --output ./entities
python -m training.gather_metadatarr --output ./entities          # OpenLibrary/TVmaze/AudioDB freshness

# 2. stage the bundled templates, then expand them with the real entities
python -m training.stage_master_templates --lang en-us --out ./templates_new
python -m training.generate_slot_filled_dataset --entities-dir ./entities \
    --templates-dir ./templates_new --langs en-us --n 400 --output ./ocp_slot_filled.csv
python -m training.generate_slot_literal_dataset --templates-dir ./templates_new --output ./ocp_slot_literal.csv
python -m training.generate_keyword_csv --n 3000                    # keyword utterances (all intents)

# 3. naturalistic LLM layer via agentpipe (FREE agents only — never claude)
python -m training.generate_agentpipe --langs en-us --n 250 --provider opencode-free \
    --entities-dir ./entities --out ./ocp_agentpipe.csv

# 4. merge + enforce taxonomy + add coarse axes + cap-per-type + split + publish
python -m training.build_and_publish --inputs ./ocp_*.csv \
    --cap-per-type 250000 --out-dir ./release \
    --push --repo TigreGotico/ocp-media-intents --private
```

`build_and_publish` merges every input CSV, dedups on `sentence`, derives
`mediavocab_type` + `playback_type` + `structure` + `genres`, balances the (music-heavy)
mix via `--cap-per-type`, makes a stratified 80/10/10 `train`/`validation`/`test` split
(`random_state=42`), and (optionally) pushes a `DatasetDict` + dataset card to the Hub.

## Notes

- **agentpipe uses FREE agents only** (`opencode-free`/`kilo`) — never claude or paid models.
- **Adult metadata** is kept only as a **content-filter signal** (the `adult` genre) — see
  [content filtering](../docs/content-filtering.md). No adult provider/scraper work here.
- Entity lists (csv/tsv/jsonl/HF) are the same machinery the NER backend consumes at runtime —
  see [entity lists](../docs/entity-lists.md).
- Model export → an ONNX bundle the optional [`onnx`](../docs/backends.md) backend loads.
