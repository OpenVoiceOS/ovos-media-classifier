# Extending the classifier

There are two ways to extend `ovos-media-classifier`, covered in turn:

* **Part A** — ship a *new backend* (a different model or strategy) behind the
  same contract, discovered as an OPM plugin.
* **Part B** — *retrain* the bundled ONNX backend, and add a *new axis/head*
  end-to-end through the training pipeline.

For the model itself — features, heads, soft-gating, limitations — read
[model.md](model.md) first.

---

## Part A — Add a backend

Every classifier, bundled or third-party, implements
[`AbstractMediaClassifier`](../ovos_media_classifier/base.py). The host relies on
nothing beyond this contract.

### Implement the contract

Only one method is abstract:

```python
from ovos_media_classifier import AbstractMediaClassifier, MediaType

class MyMediaClassifier(AbstractMediaClassifier):
    def __init__(self, config=None):
        self.config = config or {}

    def classify(self, query, lang, valid_labels=None):
        """Return (MediaType, confidence) for an ocp_play utterance.

        Return (MediaType.GENERIC, 0.0) when nothing matches, and — when
        valid_labels is given — only ever return one of those labels.
        """
        ...
        return MediaType.MUSIC, 0.9
```

Everything else has a working default (derive-from-leaf or empty). Override an
axis method only when your backend has real signal for it:

| method | default | override when… |
|---|---|---|
| `classify_domain` | derives from `classify()` | you have a cheap domain head |
| `classify_control` | `None` | you model transport-control actions |
| `is_ocp_query` | derives from `classify_domain()` | you also handle `ocp_control` |
| `classify_genres` | `[]` | you can surface genre tags (⊆ `KNOWN_GENRES`) |
| `classify_content_form_genres` | delegates to `classify_genres()` | you have a dedicated content-form head (lets the content filter block on it) |
| `classify_content_form` | `None` | you predict the `mediavocab.ContentForm` (trailer / behind_scenes / …) |
| `classify_programme_format` | `None` | you predict the `mediavocab.ProgrammeFormat` (documentary / news / …) |
| `classify_accessibility` | `[]` | you predict `mediavocab.AccessibilityKind` assets |
| `classify_variant` | `None` | you predict the `mediavocab.VariantKind` cut |
| `classify_playback_type` | derives from the leaf | you have a modality head |
| `classify_structure` | derives from the leaf | you have a structure head |
| `classify_picture_format` | `[]` | you predict `mediavocab.PictureFormat` (bw / silent / 3d) |
| `classify_explicitness` | derives from the form genres | you have an explicitness head |
| `classify_control_intent` | delegates to `classify_control()` | — |
| `classify_full` | combines the above (derive-from-leaf) | you predict the axes directly and want to soft-gate the leaf |
| `to_signals` | builds `mediavocab.Signals` from `classify_full` + genres + content_form + variant | you extract entities (artist / year / season / episode) to enrich the `Signals` |

A trained backend SHOULD override the coarse-axis methods so it predicts each axis
with its own head and soft-gates the leaf — that is the whole point of the
multi-axis model (see [model.md](model.md#2-multi-task-per-axis-heads--the-key-design)).

### Register and load it

Register the concrete class under the `opm.media.classifier` entry-point group:

```toml
# your package's pyproject.toml
[project.entry-points."opm.media.classifier"]
my-classifier = "my_pkg:MyMediaClassifier"
```

The factory loads it by the entry-point name, forwarding the whole config dict to
the constructor as `config=…`:

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier({"media_classifier_plugin": "my-classifier"})
```

If the plugin fails to load the factory logs a warning and falls through to the
built-in backends — an external plugin never hard-fails the pipeline. See
[external-plugins.md](external-plugins.md) for discovery details and
[stable-api.md](stable-api.md) for the full contract and return types.

---

## Part B — Train and export an ONNX bundle

The reference "train your own backend" pipeline is
[`training/train_sklearn.py`](../training/train_sklearn.py). It turns the canonical
`ocp-media-intents` dataset into the self-describing multi-head bundle that
[`OnnxMediaClassifier.from_path`](../ovos_media_classifier/onnx.py) loads
unchanged.

### 1. Install the training extra

```bash
pip install ovos-media-classifier[train]
```

### 2. Build the dataset

```bash
python -m training.build_dataset            # → data/release/{train,validation,test}.{csv,parquet}
```

[`build_dataset`](../training/build_dataset.py) expands the `.intent` templates,
slot-fills them with real entities, and emits every feature and axis column. The
columns are documented in [dataset.md](dataset.md). (If `--data-dir` has no split,
`train_sklearn.py` will build it for you.)

### 3. Train every head on both feature sets

```bash
python -m training.train_sklearn            # data/release → data/models/<feature_set>/
```

This trains every head in `HEAD_SPECS` on both `FEATURE_SETS` (`context` and
`context_ner`), selecting the best estimator per head by validation macro-F1, and
writes one bundle directory per feature set (`data/models/context/` and
`data/models/context_ner/`).

### 4. Point the runtime at the bundle

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier({"media_classifier_onnx_model": "data/models/context_ner"})
```

or load it directly:

```python
from ovos_media_classifier.onnx import OnnxMediaClassifier

clf = OnnxMediaClassifier.from_path("data/models/context_ner")
```

---

## Part C — Neural backend + custom architectures (PyTorch → ONNX)

[`training/train_torch.py`](../training/train_torch.py) is the **neural** trainer.
It produces the *same* self-describing bundle format as `train_sklearn.py` — one
ONNX graph per axis + `meta.json` — so `OnnxMediaClassifier.from_path` loads a
torch bundle **unchanged**. The difference is the model (a shared-trunk multi-task
net) and the *features* (it can add char-hash text + trained word vectors that the
sklearn ladder's binary flags can't express). The why/what + the full head-to-head
comparison live in [model.md §7](model.md#7-neural-backend--richer-text-features-does-seeing-the-value-text-help);
this is the how-to. `torch` + `gensim` ship in the `[train]` extra and are
**train-only** — runtime stays `onnxruntime` + `numpy`.

### 1. (optional) train the domain word vectors

```bash
python -m training.build_corpus            # data/{entities,relational,release} → data/wordvec/
```

Trains a `gensim` Word2Vec over every entity pool + relational record + utterance,
and writes `wordvec.npy` (the matrix) + `wordvec_vocab.json`. Skip this if you only
want the `cat` / `cat_text` variants (no word-vector block).

### 2. train the neural variants

```bash
python -m training.train_torch             # data/release → data/models_torch/<variant>/
python -m training.train_torch --variants cat cat_text --epochs 12
```

Each variant in `VARIANTS` trains the shared trunk on a different feature block and
exports a bundle. The benchmark compares them with
`python -m benchmarks.ladder` (which adds the `neural *` rungs automatically).

### Add a custom architecture

A variant is one dict entry in `VARIANTS` in
[`train_torch.py`](../training/train_torch.py):

```python
VARIANTS["my_arch"] = {
    "blocks": ("cat", "text", "wordvec"),  # any subset; drives the input dim
    "hidden": [768, 512, 256],             # the shared-trunk layer widths
    "residual": True,                      # skip-connections between equal widths
}
```

`blocks` selects the feature families (`cat` = categorical, `text` = char-hash,
`wordvec` = pooled word vectors); `build_features` assembles
`[cat ⊕ text ⊕ wordvec]` and records the exact column order in `feature_names`.
The trunk (`Trunk` in `_build_modules`) is a plain LayerNorm-MLP — change its body
to try a different architecture (e.g. a learned char-embedding + attention pool):
keep the `forward(x) -> z` shape so each axis head reads a fixed-width trunk
output, and keep the per-axis `HeadExport` (trunk → head → softmax|sigmoid) so the
export still emits the bundle's expected tensors. Everything else — multi-task
loss, class weighting, early stop, ONNX export, the round-trip parity check — is
architecture-agnostic and reused.

The featurizer spec is written into `meta.json` (`text_hash` / `wordvec`) by
`export_bundle`, so the runtime reproduces your exact input vector in numpy with no
code change. To add a *new feature family*, write a numpy-only featurizer like
[`features_text.py`](../ovos_media_classifier/features_text.py) /
[`features_wordvec.py`](../ovos_media_classifier/features_wordvec.py) (a `*Spec`
dataclass with `to_meta` / `from_meta` + `feature_names`), wire it into
`build_features` (train) and `OnnxMediaClassifier._vectorize` (runtime), and record
its spec in `meta.json`.

---

## Add a NEW axis/head end-to-end

Adding a new classification axis is a four-step change. Use the real function and
symbol names below; the existing axes are worked examples of each step.

**1. Emit a ground-truth column** — add the derived label to
[`_derive_axes`](../training/build_dataset.py) in `training/build_dataset.py`. It
computes each axis from the template `intent` plus the `slot_values` that filled
the row (e.g. `content_form` / `programme_format` / `variant` from the intent
alias, `content_genres` from the genre slot value, `year` from the year slot).
Return your new key from this function and add
it to the `_AXES` column-order list so it lands in the written CSV/parquet. The
label must be ground-truth-by-construction, exactly like the existing columns.

**2. Declare the head** — add an entry to `HEAD_SPECS` in
[`training/train_sklearn.py`](../training/train_sklearn.py):
`(axis_name, column, kind)` where `kind` is `"single"` (argmax) or `"multi"`
(per-label sigmoid). `train_single_head` / `train_multi_head` then train it
automatically; a degenerate column is skipped, and `export_bundle` writes
`<axis>.onnx` plus its `meta.json` manifest entry. Multi-label heads carry a
per-label `threshold` (default `DEFAULT_MULTILABEL_THRESHOLD = 0.5`); cap an
open-vocabulary head with a `top_k` like `content_genres` does.

**3. Add the contract method** — add `classify_<axis>` to
[`AbstractMediaClassifier`](../ovos_media_classifier/base.py) with a sensible
default (derive from a cheaper axis, or return empty/`None`) so every existing
backend keeps working without change.

**4. Read the head in the runtime** — override `classify_<axis>` in
[`OnnxMediaClassifier`](../ovos_media_classifier/onnx.py) to use the head when the
bundle carries it and fall back to the inherited default otherwise. Use
`self._single_head(axis, query, lang)` for a single-label head or
`self._multi_head(axis, query, lang)` for a multi-label one — both already handle
the ONNX graph shapes the bundle contract allows. If the axis belongs in the full
result, also surface it in `classify_full`.

Because the head is recorded in `meta.json` and `from_path` loads whatever heads
are present, old bundles without your new head still load and simply derive the
axis — so the change is backward-compatible by construction.

---

## See also

* [model.md](model.md) — the feature representation, the heads, soft-gating, the
  ladder, and the honest limitations.
* [dataset.md](dataset.md) — the columns your heads are supervised on.
* [external-plugins.md](external-plugins.md) — OPM discovery for a registered
  backend.
* [stable-api.md](stable-api.md) — the full `AbstractMediaClassifier` contract.
