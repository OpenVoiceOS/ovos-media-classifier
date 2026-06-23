# Benchmarks

A reproducible, network-free harness that evaluates the `ovos-media-classifier`
backends against a labeled eval set and produces a metrics table plus matplotlib
plots.

The suite lives at the repo top level (it is **not** part of the installed
package). Run everything from the repo root with the project venv.

## Quick start

```bash
# metrics only -> benchmarks/results.json + benchmarks/results.md
~/.venvs/ovos/bin/python -m benchmarks.run

# metrics + plots -> also writes docs/benchmarks/*.png
~/.venvs/ovos/bin/python -m benchmarks.run --plots

# regenerate the committed eval set from the bundled .voc files
~/.venvs/ovos/bin/python -m benchmarks.dataset
# or force a rebuild as part of a run
~/.venvs/ovos/bin/python -m benchmarks.run --plots --rebuild-dataset
```

`matplotlib` and `numpy` are required for `--plots` (install into the venv with
`~/.venvs/ovos/bin/pip install matplotlib`).

## Artifacts

| File | What it is |
|---|---|
| `benchmarks/dataset.py` | Builds the deterministic eval set from the bundled `.voc` keyword files. |
| `benchmarks/eval_set.csv` | The committed eval set: `utterance, lang, expected_media_type, expected_genres`. |
| `benchmarks/run.py` | Loads the eval set, runs every available backend, computes metrics + content-filter recall. |
| `benchmarks/plots.py` | Renders the PNGs from the report dict. |
| `benchmarks/results.json` | Machine-readable per-backend metrics + confusion matrices. |
| `benchmarks/results.md` | Human-readable results table (regenerated each run). |
| `docs/benchmarks/*.png` | Plots (see below). |

## How the eval set is built

The ground truth is sourced **entirely from the bundled `.voc` keyword files** —
the same vocabulary the keyword backend ships with — so the benchmark needs no
network and no external dataset. For each `<Type>Keyword.voc`:

1. The voc file name maps to an `OCPPlayIntent` (mirroring the branch order in
   `KeywordMediaClassifier._classify_intent`).
2. That intent maps to a canonical `mediavocab.MediaType` via
   `PLAY_INTENT_TO_MEDIA_TYPE`, and to genre tags via `PLAY_INTENT_TO_GENRES`
   (this is where the `adult` signal comes from).
3. Each keyword phrase is dropped into play-style templates
   (`"play {kw}"`, `"put on some {kw}"`, `"i want to watch a {kw}"`, …) to make
   realistic utterances.

Keyword phrases that are shared across media types (genuinely ambiguous for the
keyword backend) are dropped so the labels stay honest. Sampling uses a fixed
seed (`SEED = 42`) and the CSV is sorted, so it is byte-reproducible. Languages:
`en-us` (mandatory) plus `de-de` and `pt-pt`.

## Backends

Backends needing optional dependencies or trained model files that are not
present in a bare checkout are detected and recorded as **unavailable** (with the
reason) — they never crash the run. The zero-dependency **keyword** backend is
always available. The **ahocorasick** backend is seeded from the bundled keyword
vocs so it produces a meaningful (exact-match) baseline.

To benchmark the ML backends, point the harness at trained artifacts via env
vars: `MEDIA_CLF_SKLEARN_MODEL`, `MEDIA_CLF_PADATIOUS_DIR`, `MEDIA_CLF_M2V_MODEL`,
`MEDIA_CLF_GUIDED_MODEL`.

## Metrics

Per backend: accuracy and macro-F1 over `mediavocab.MediaType`, per-type
precision/recall/F1/support, median + p95 inference latency (ms/utterance) and
rows/sec. Globally: **content-filter recall** — of the adult-genre rows, the
fraction the default `ContentFilter().check(clf, utterance, lang)` blocks — plus
the false-block rate over the non-adult slice.

## Latest results

Eval set: **936 utterances** across 3 languages (`de-de`=308, `en-us`=360,
`pt-pt`=268), **64 adult-genre rows**.

| backend | status | accuracy | macro-F1 | median ms | p95 ms | rows/s | CF recall | false-block |
|---|---|---|---|---|---|---|---|---|
| keyword | available | 0.981 | 0.989 | 0.0132 | 0.0337 | 53915 | 0.938 (60/64) | 0.000 |
| ahocorasick | available | 0.481 | 0.613 | 0.0050 | 0.0120 | 141910 | 0.000 (0/64) | 0.000 |
| sklearn | unavailable | – | – | – | – | – | – | – |
| padatious | unavailable | – | – | – | – | – | – | – |
| model2vec | unavailable | – | – | – | – | – | – | – |
| guided_onnx | unavailable | – | – | – | – | – | – | – |

Notes from this run:

- The keyword backend's only systematic errors are `episodic_series` ("tv show")
  resolving to `tv` (the `TVKeyword`/`IPTVKeyword` branch outranks `SeriesKeyword`)
  and a handful of `movie` titles caught by an `audiobook` substring.
- Keyword content-filter recall is **0.938** (60/64); the 4 misses are German
  `sexfilm` utterances that the `de-de` `AdultKeyword.voc` does not cover — a real
  localization gap surfaced by the benchmark. False-block rate on non-adult rows
  is **0.000**.
- The ahocorasick exact-match baseline blocks **0** adult rows because it does not
  surface a genre signal (`classify_genres` is the default empty), so the content
  filter has nothing to act on — useful to keep visible.

See `benchmarks/results.md` for the full per-type tables and
`benchmarks/results.json` for the raw numbers and confusion matrices.

## Plots

All saved under `docs/benchmarks/` at dpi 140.

| Plot | Description |
|---|---|
| ![confusion matrix](../docs/benchmarks/confusion_matrix_keyword.png) | `confusion_matrix_keyword.png` — true vs predicted media type for the keyword backend (one per available backend). |
| ![per-type F1](../docs/benchmarks/per_type_f1.png) | `per_type_f1.png` — per-media-type F1 bars for the keyword backend. |
| ![accuracy vs latency](../docs/benchmarks/accuracy_vs_latency.png) | `accuracy_vs_latency.png` — accuracy vs median latency across available backends. |
| ![content-filter recall](../docs/benchmarks/content_filter_recall.png) | `content_filter_recall.png` — adult-slice block recall and false-block rate per backend. |
