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

1. The voc file name maps directly to a canonical `mediavocab.MediaType` and any
   genre tags (`_VOC_TO_MEDIA_TYPE` / `_VOC_TO_GENRES`, mirroring the branch
   outcomes in `KeywordMediaClassifier._classify_leaf`) — this is where the
   `adult` signal comes from.
3. Each keyword phrase is dropped into **modality-consistent** play-style
   templates. The keyword backend is hierarchical (coarse-to-fine): it predicts
   the playback modality from the carrier verb FIRST and constrains the leaf to
   it, so the carrier must agree with the leaf's modality — a video leaf gets a
   "watch …" carrier, an audio leaf a "listen to …" carrier, etc. (no
   self-inflicted "watch a radio" mislabels). Every leaf also gets the neutral
   `"play {kw}"` carriers.

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

## Latest results — read this framing first

There are **two very different numbers**, and conflating them is misleading:

**1. Synthetic eval set (`benchmarks/eval_set.csv`)** — utterances generated *from
the bundled `.voc` files*. This measures **vocabulary coverage / wiring** (is every
keyword reachable, does adult precedence hold), **NOT generalization.** A keyword
matcher scores ~0.99 here almost by construction — the test is built from its own
vocabulary.

| backend | accuracy | macro-F1 | median ms | rows/s | CF recall | false-block |
|---|---|---|---|---|---|---|
| keyword | **0.990** | 0.984 | 0.027 | 7900 | 1.000 (64/64) | 0.000 |
| ahocorasick | 0.495 | 0.628 | 0.003 | 408000 | 0.469 (30/64) | 0.000 |

**2. Neutral HF test split (`--hf-dataset TigreGotico/ocp-media-intents`)** — real,
naturally-phrased commands the classifier has never seen. This measures **actual
generalization**, and it is the number that matters:

| backend | accuracy | macro-F1 | CF recall |
|---|---|---|---|
| keyword | **0.290** | 0.442 | (3-row adult sample, not significant) |
| ahocorasick | 0.169 | 0.244 | — |

### What this means

- **The keyword backend is a deterministic FLOOR**, not a production-accuracy
  classifier: ~0.29 on real queries. It exists so OCP works offline with zero ML
  deps; **real accuracy comes from the trained ONNX backend.** Do not cite the 0.99.
- Neither the **coarse-to-fine** restructure nor **word-boundary** matching moved the
  real-query number (both improved the *synthetic* score; on the HF split they are
  neutral-to-slightly-negative — word boundaries trade a little natural-language
  recall for correctness, e.g. German "sexfilm" no longer false-matches "film").
  They are justified by **architecture and safety**, not a keyword accuracy gain.
- **Adult content-filter recall is 1.000** on the 64-row synthetic adult slice (the
  reliable measure); the HF split has too few adult rows to be meaningful.
- The gap between 0.99 and 0.29 is the **dataset itself**: it is ~99% synthetic
  slot-fill, so models trained/measured on it overfit templated phrasing. Closing
  the gap needs more *natural* training data, not more keyword tuning.

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
